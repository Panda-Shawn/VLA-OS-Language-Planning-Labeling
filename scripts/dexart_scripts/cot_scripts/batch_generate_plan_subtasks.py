import re
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
import concurrent.futures
from pathlib import Path

try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    pass

import os
from dotenv import load_dotenv


class Gemini:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        # self.model = genai.GenerativeModel("gemini-1.5-pro")

    def safe_call(self, f):
        while True:
            try:
                res = f()
                return res
            except ResourceExhausted:
                print("ResourceExhausted, retrying...")
                time.sleep(5)

    def generate(self, prompt):
        chat = self.safe_call(lambda: self.model.start_chat(history=[]))
        response = self.safe_call(lambda: chat.send_message(prompt).text)

        for i in range(8):
            if "FINISHED" in response:
                # print(f"n_retries: {i}")
                return response

            response = response + self.safe_call(
                lambda: chat.send_message("Truncated, please continue.").text
            )

        print(f"n_retries: {iter}")
        return None


class Deepseek:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def safe_call(self, f):
        max_retries = 5
        for retry in range(max_retries):
            try:
                res = f()
                return res
            except Exception as e:
                print(f"Error: {e}. Retrying ({retry + 1}/{max_retries})...")
                time.sleep(5)
        raise Exception("Failed after multiple retries.")

    def generate(self, prompt):
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]

        def call_api():
            return self.client.chat.completions.create(
                model="deepseek-chat", messages=messages, stream=False
            )

        response = self.safe_call(call_api)
        return response.choices[0].message.content


def process_single_prompt(prompt_data):
    """Process a single prompt with its save path"""
    prompt, save_path, load_saved, api_provider = prompt_data
    print(f"Processing: {save_path} with provider {api_provider}")


    if load_saved and os.path.exists(save_path):
        with open(save_path, "r") as f:
            return f.read()

    try:
        if api_provider == "gemini" or api_provider == "deepseek":
            response_text = model.generate(prompt)
        elif api_provider == "openai":
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.choices[0].message.content
        else:
            raise ValueError(f"Unknown API provider: {api_provider}")

        with open(save_path, "w") as f:
            f.write(response_text)
        return response_text
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return None


def batch_get_responses(
    prompts: List[str],
    save_paths: List[str],
    batch_size: int = 10,
    load_saved: bool = True,
    api_provider: str = "gemini",
):
    """Batch process multiple prompts using concurrent execution"""
    responses = []

    # Create prompt data tuples
    prompt_data = [
        (prompt, save_path, load_saved, api_provider)
        for prompt, save_path in zip(prompts, save_paths)
    ]

    # Process in batches
    for i in range(0, len(prompt_data), batch_size):
        batch = prompt_data[i : i + batch_size]

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit all tasks in the batch
            futures = {
                executor.submit(process_single_prompt, data): idx
                for idx, data in enumerate(batch)
            }

            batch_responses = [None] * len(batch)
            # Collect results as they complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    batch_responses[futures[future]] = result
                except Exception as exc:
                    print(f"Prompt generated an exception: {exc}")

        responses.extend(batch_responses)

    return responses


def find_task_occurrences(input_string, tags):
    pattern = r"(\d+):\s*\"?"
    for tag in tags:
        pattern = pattern + r"\s*<" + tag + r">([^<]*)<\/" + tag + ">"
    pattern = pattern + r"\"?"

    matches = re.findall(pattern, input_string)
    return matches


def process_response(
    response_text,
    tags=("task", "plan", "subtask", "subtask_reason", "move", "move_reason"),
):
    """Process the API response text and extract the reasoning dictionary"""
    if response_text is None:
        return dict()

    trajectory = dict()
    matches = find_task_occurrences(response_text, tags)

    for match in matches:
        trajectory[int(match[0])] = dict(zip(tags, match[1:]))

    return trajectory


# proxy = "http://127.0.0.1:6789"
# os.environ["http_proxy"] = proxy
# os.environ["https_proxy"] = proxy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--action_horizon", type=int, default=10)
    parser.add_argument("--force_regenerate", action="store_true")
    parser.add_argument("--api_provider", type=str, default="gemini")
    parser.add_argument("--libero_dataset_dir", type=str)
    parser.add_argument("--libero_scene_desc_path", type=str, default=None)
    parser.add_argument("--libero_primitives_path", type=str, default=None)
    parser.add_argument("--llms_response_save_dir", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    args = parser.parse_args()

    cot_dir = os.path.join(args.libero_dataset_dir, "cot")
    if args.llms_response_save_dir is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.llms_response_save_dir = os.path.join(
            cot_dir, f"raw_api_responses_h{args.action_horizon}"
        )

    if args.libero_primitives_path is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.libero_primitives_path = os.path.join(
            cot_dir, f"primitives_h{args.action_horizon}.json"
        )

    if args.libero_scene_desc_path is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.libero_scene_desc_path = os.path.join(cot_dir, "scene_descriptions.json")

    with open(args.libero_primitives_path, "r") as f:
        primitives_dict = json.load(f)

    with open(args.libero_scene_desc_path, "r") as f:
        scene_description_dict = json.load(f)

    os.makedirs(args.llms_response_save_dir, exist_ok=True)
    print("Using API provider:", args.api_provider)

    # Configure API clients based on provider
    load_dotenv()
    if args.api_provider == "gemini":
        model = Gemini(api_key=os.getenv("GOOGLE_API_KEY"))
    elif args.api_provider == "deepseek":
        model = Deepseek(api_key=os.getenv("DEEPSEEK_API_KEY"))
    elif args.api_provider == "openai":
        raise NotImplementedError("OpenAI API provider needs to be fixed")
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    else:
        raise ValueError(f"Unknown API provider: {args.api_provider}")

    # Read prompt template
    prompt_template_path = PROMPT_TEMPLATE_DIR / "prompt.txt"
    with open(str(prompt_template_path), "r") as f:
        template_prompt = f.read()

    # Collect prompts in batches
    prompts = []
    save_paths = []
    filtered_step_mappings = {}
    episode_ids = []

    if args.results_path is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.results_path = os.path.join(
            cot_dir, f"chain_of_thought_h{args.action_horizon}.json"
        )

    if os.path.exists(args.results_path):
        with open(args.results_path, "r") as f:
            results = json.load(f)
        keys = list(set(scene_description_dict.keys()) - set(results.keys()))
    else:
        results = {}
        keys = list(scene_description_dict.keys())

    for episode_id in keys:
        primitives = primitives_dict[episode_id]
        scene_description = scene_description_dict[episode_id]["caption"]
        task_description = scene_description_dict[episode_id]["task_description"]

        # Remove adjacent duplicate primitives
        filtered_primitives = []
        prev_primitive = None
        filtered_step_mapping = {}
        for step, primitive in enumerate(primitives):
            if primitive != prev_primitive:
                filtered_primitives.append(primitive)
                prev_primitive = primitive
            filtered_step_mapping[step] = len(filtered_primitives) - 1
        filtered_step_mappings[episode_id] = filtered_step_mapping

        primitives_prompt = "trajectory_features = {\n"
        for step, primitive in enumerate(filtered_primitives):
            primitives_prompt += f'    {step}: "{primitive}",\n'
        primitives_prompt += "}"

        prompt = template_prompt.format(
            scene_description=scene_description,
            primitives_prompt=primitives_prompt,
            task_instruction=task_description,
        )

        prompts.append(prompt)
        save_paths.append(
            os.path.join(
                args.llms_response_save_dir,
                f"{episode_id}_{args.api_provider}.txt",
            )
        )
        episode_ids.append(episode_id)

    # Process Batches with API of LLMs
    print(f"Processing {len(prompts)} prompts in batches of {args.batch_size}...")
    with tqdm(
        total=len(prompts), desc=f"Processing batches (h={args.action_horizon})"
    ) as pbar:
        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i : i + args.batch_size]
            batch_saves = save_paths[i : i + args.batch_size]
            batch_episode_ids = episode_ids[i : i + args.batch_size]

            responses = batch_get_responses(
                batch_prompts,
                batch_saves,
                args.batch_size,
                load_saved=not args.force_regenerate,
                api_provider=args.api_provider,
            )

            # Process responses
            for response_text, episode_id in zip(responses, batch_episode_ids):
                cot_dict = process_response(response_text)
                try:
                    # Map back to original steps
                    filtered_step_mapping = filtered_step_mappings[episode_id]
                    assert len(cot_dict) == len(
                        np.unique(list(filtered_step_mapping.values()))
                    )
                    original_cot_dict = {}
                    for ori_step, filtered_step in filtered_step_mapping.items():
                        original_cot_dict[ori_step] = cot_dict[filtered_step]
                    results[episode_id] = {
                        "0": {
                            "reasoning": original_cot_dict,
                            "features": {},
                        }
                    }

                except (KeyError, AssertionError) as e:
                    print(f"{type(e)} for {episode_id}")
                    with open("error_episodes.txt", "a") as f:
                        f.write(f"{episode_id}\n")

            pbar.update(len(batch_prompts))

    with open(args.results_path, "w") as f:
        json.dump(results, f)

    print("Processing complete!")