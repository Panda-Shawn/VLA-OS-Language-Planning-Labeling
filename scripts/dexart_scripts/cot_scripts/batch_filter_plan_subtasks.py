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
from utils import PROMPT_TEMPLATE_DIR


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

    print(f"[INFO] Processing: {os.path.basename(save_path)}")

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

        print(f"[SUCCESS] Finished: {os.path.basename(save_path)}")
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


def extract_task_plan(text):
    pattern = r"<task>(.*?)<\/task>.*?<plan>(.*?)<\/plan>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return {"task": match.group(1).strip(), "plan": match.group(2).strip()}
    return {}


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


def process_task_plan_response(
    response_text,
):
    """Process the API response text and extract the reasoning dictionary"""
    if response_text is None:
        return dict()

    task_plan = extract_task_plan(response_text)

    return task_plan


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
    parser.add_argument("--libero_plan_subtasks_path", type=str, default=None)
    parser.add_argument("--task_metadata_path", type=str, default=None)
    parser.add_argument("--task_plan_path", type=str, default=None)
    parser.add_argument("--llms_response_save_dir", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="bucket")
    args = parser.parse_args()

    cot_dir = os.path.join(args.libero_dataset_dir, "cot")
    if args.llms_response_save_dir is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.llms_response_save_dir = os.path.join(
            cot_dir, f"raw_api_responses_h{args.action_horizon}_filtering"
        )

    os.makedirs(cot_dir, exist_ok=True)
    task_plan_save_dir = os.path.join(
        cot_dir, f"raw_api_responses_h{args.action_horizon}_task_plan"
    )

    if args.libero_primitives_path is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.libero_primitives_path = os.path.join(
            cot_dir, f"primitives_h{args.action_horizon}.json"
        )

    if args.libero_plan_subtasks_path is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.libero_plan_subtasks_path = os.path.join(
            cot_dir, f"plan_subtasks_h{args.action_horizon}.json"
        )

    if args.libero_scene_desc_path is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.libero_scene_desc_path = os.path.join(cot_dir, "scene_descriptions.json")

    with open(args.libero_primitives_path, "r") as f:
        primitives_dict = json.load(f)

    with open(args.libero_plan_subtasks_path, "r") as f:
        plan_subtasks_dict = json.load(f)

    with open(args.libero_scene_desc_path, "r") as f:
        scene_description_dict = json.load(f)

    os.makedirs(args.llms_response_save_dir, exist_ok=True)
    os.makedirs(task_plan_save_dir, exist_ok=True)
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
    prompt_filtering_template_path = PROMPT_TEMPLATE_DIR / "prompt_filtering.txt"
    with open(prompt_filtering_template_path, "r") as f:
        template_prompt = f.read()

    prompt_task_plan_template_path = PROMPT_TEMPLATE_DIR / "prompt_task_plan.txt"
    with open(prompt_task_plan_template_path, "r") as f:
        task_plan_template_prompt = f.read()

    # # Collect prompts in batches
    # prompts = []
    # save_paths = []
    # filtered_step_mappings = {}
    # episode_ids = []

    if args.results_path is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.results_path = os.path.join(
            cot_dir, f"filtered_reasoning_h{args.action_horizon}.json"
        )

    if os.path.exists(args.results_path):
        with open(args.results_path, "r") as f:
            results = json.load(f)
        keys = list(set(plan_subtasks_dict.keys()) - set(results.keys()))
    else:
        results = {}
        keys = list(plan_subtasks_dict.keys())

    if args.task_metadata_path is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.task_metadata_path = os.path.join(
            cot_dir, f"task_metadata.json"
        )

    if os.path.exists(args.task_metadata_path):
        with open(args.task_metadata_path, "r") as f:
            task_metadata = json.load(f)
    else:
        task_metadata = {}
        for episode_id in plan_subtasks_dict.keys():
            plan_subtasks = plan_subtasks_dict[episode_id]["0"]["reasoning"]
            scene_description = scene_description_dict[episode_id]["caption"]
            task_description = scene_description_dict[episode_id]["task_description"]
            
            # task_id = episode_id.split("_Demo")[0]
            task_id = args.task_name
            if task_id not in task_metadata.keys():
                task_metadata[task_id] = []
            task_metadata[task_id].append(episode_id)

    if args.task_plan_path is None:
        os.makedirs(cot_dir, exist_ok=True)
        args.task_plan_path = os.path.join(
            cot_dir, f"task_plan.json"
        )

    if os.path.exists(args.task_plan_path):
        with open(args.task_plan_path, "r") as f:
            task_plans = json.load(f)
        task_plan_keys = list(set(plan_subtasks_dict.keys()) - set(task_plans.keys()))
    else:
        task_plans = {}
        task_plan_keys = list(plan_subtasks_dict.keys())

    for episode_id in task_plan_keys:
        plan_subtasks = plan_subtasks_dict[episode_id]["0"]["reasoning"]
        scene_description = scene_description_dict[episode_id]["caption"]
        task_description = scene_description_dict[episode_id]["task_description"]
        task_plans[episode_id] = {"task": plan_subtasks["0"]["task"], "plan": plan_subtasks["0"]["plan"]}

    task_plan_prompts = []
    save_paths = []
    task_ids = []
    for task_id in task_metadata.keys():
        scene_prompt = scene_description_dict[task_metadata[task_id][0]]["caption"]
        task_prompt = scene_description_dict[task_metadata[task_id][0]]["task_description"]
        task_plan_prompt = "task_plan = {\n"
        for episode_id in task_metadata[task_id]:
            task = task_plans[episode_id]["task"]
            plan = task_plans[episode_id]["plan"]
            task_plan_prompt += f'    {episode_id}: "<task>{task}</task><plan>{plan}</plan>",\n'
        task_plan_prompt += "}"

        prompt = task_plan_template_prompt.format(
            scene_description=scene_prompt,
            task_plan_prompt=task_plan_prompt,
            task_instruction=task_prompt,
        )
        save_paths.append(
            os.path.join(
                task_plan_save_dir,
                f"{task_id}_{args.api_provider}.txt",
            )
        )

        task_plan_prompts.append(prompt)
        task_ids.append(task_id)
    
    # print(task_ids)
    # Process Batches with API of LLMs
    print(f"Processing {len(task_plan_prompts)} task&plan prompts in batches of {args.batch_size}...")
    with tqdm(
        total=len(task_plan_prompts), desc=f"Processing batches (h={args.action_horizon})"
    ) as pbar:
        for i in range(0, len(task_plan_prompts), args.batch_size):
            batch_prompts = task_plan_prompts[i : i + args.batch_size]
            batch_saves = save_paths[i : i + args.batch_size]
            batch_task_ids = task_ids[i : i + args.batch_size]  

            responses = batch_get_responses(
                batch_prompts,
                batch_saves,
                args.batch_size,
                load_saved=True,#not args.force_regenerate,
                api_provider=args.api_provider,
            )
            for task_id, response_text in zip(batch_task_ids, responses):
                task_plan_dict_each_episode = process_task_plan_response(response_text)
                for episode_id in task_metadata[task_id]:
                    # print(task_id, episode_id, task_plan_dict_each_episode)
                    task_plans.update({episode_id: task_plan_dict_each_episode})

                print(f"[SUCCESS] Task plan generated for: {task_id}")
        
            pbar.update(len(batch_prompts))
    
    # import pdb; pdb.set_trace()

    with open(args.task_metadata_path, "w") as f:
        json.dump(task_metadata, f)
    
    with open(args.task_plan_path, "w") as f:
        json.dump(task_plans, f)

    print("Processing task&plan prompts complete!")
    prompts = []
    save_paths = []
    filtered_step_mappings = {}
    episode_ids = []
    
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

        fixed_task = task_plans[episode_id]["task"]
        fixed_plan = task_plans[episode_id]["plan"]
        prompt = template_prompt.format(
            scene_description=scene_description,
            primitives_prompt=primitives_prompt,
            task=fixed_task,
            plan=fixed_plan,
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
                    # if len(cot_dict) != len(np.unique(list(filtered_step_mapping.values()))):
                    #     # We complete the cot_dict with the last step if it is a stop task.
                    #     num_cot_steps = len(cot_dict)
                    #     num_filtered_steps = len(np.unique(list(filtered_step_mapping.values())))
                    #     end_task = cot_dict[num_cot_steps-1]["task"]
                    #     end_move = cot_dict[num_cot_steps-1]["move"]
                    #     if end_task == "Stop." or end_move == "stop":
                    #         for i in range(num_filtered_steps-num_cot_steps):
                    #             cot_dict[num_cot_steps+i] = cot_dict[num_cot_steps-1]
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

                    print(f"[SUCCESS] CoT reasoning generated for: {episode_id}")


                except (KeyError, AssertionError) as e:
                    print(f"{type(e)} for {episode_id}: generated steps: {len(cot_dict)}, filtered steps: {len(np.unique(list(filtered_step_mapping.values())))}")
                    print(cot_dict[list(cot_dict.keys())[-1]]["task"])
                    left_set = set(filtered_step_mapping.values()) - set(cot_dict.keys())
                    for i in left_set:
                        print(f"Left step: {i}, mapping step: {list(filtered_step_mapping.keys())[list(filtered_step_mapping.values()).index(i)]}")
                    with open("error_episodes_filtering.txt", "a") as f:
                        f.write(f"{episode_id}\n")

            pbar.update(len(batch_prompts))

    with open(args.results_path, "w") as f:
        json.dump(results, f)

    print("Processing complete!")
