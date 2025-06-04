import argparse
import json
import os
import warnings
from pathlib import Path

import torch

from PIL import Image
from tqdm import tqdm
from utils import NumpyFloatValuesEncoder

from prismatic import load


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
try:
    hf_token = Path("./.hf_token").read_text().strip()
except FileNotFoundError:
    hf_token = None

import tensorflow_datasets as tfds

image_dims = (256, 256)


def create_user_prompt(lang_instruction, instance_names):
    user_prompt = (
        f"The robot task is: {lang_instruction}. "
        # "Analyze the image of a scene where a robot arm is about to perform this task on a wooden floor or table. "
        # "Analyze the image of a scene on a black table. "
        "Analyze the image of a scene where a robot arm is about to perform this task. "
        # f"There are {', '.join(instance for instance in instance_names)} in the scene. "
        "Briefly describe the objects in the image and their spatial relations to each other. "
    )
    return user_prompt
    # user_prompt = "Briefly describe the things in this scene and their spatial relations to each other."
    # # user_prompt = "Briefly describe the objects in this scene."]
    # lang_instruction = lang_instruction.strip()
    # if len(lang_instruction) > 0 and lang_instruction[-1] == ".":
    #     lang_instruction = lang_instruction[:-1]
    # if len(lang_instruction) > 0 and " " in lang_instruction:
    #     user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    # return user_prompt


def label_single_task(vlm, data_path, episode):
    results_json = {}
    for step in episode["steps"]:
        lang_instruction = step["language_instruction"].numpy().decode()
        image = Image.fromarray(step["observation"]["image"].numpy())
        # image.save(f"{data_path[1:].replace('/','_')}.png")
        break

    user_prompt = create_user_prompt(lang_instruction, None)
    prompt_builder = vlm.get_prompt_builder()
    prompt_builder.add_turn(role="human", message=user_prompt)
    prompt_text = prompt_builder.get_prompt()

    torch.manual_seed(0)
    caption = vlm.generate(
        image,
        prompt_text,
        do_sample=False,
        max_new_tokens=128,
        min_length=1,
    )
    # print(caption)

    results_json[data_path] = {
        "caption": caption,
        "task_description": lang_instruction,
    }
    # break

    return results_json


if __name__ == "__main__":
    import sys
    print("Command-line arguments:", sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--dataset_name", type=str, default="example_dataset")
    parser.add_argument("--vlm_model_path", type=str, default="prism-dinosiglip+7b")
    parser.add_argument("--results_path", type=str, default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    # Load Prismatic VLM
    device = "cuda"
    print(f"Loading Prismatic VLM ({args.vlm_model_path})...")
    # vlm = load(Path(args.vlm_model_path), hf_token=hf_token)
    vlm = load(args.vlm_model_path, hf_token=hf_token) # prism-dinosiglip+7b
    vlm = vlm.to(device, dtype=torch.bfloat16)

    data_name = args.dataset_name
    ds = tfds.load(
        data_name,
        data_dir=args.dataset_dir,
        split=f"train[{0}%:{100}%]",
    )

    # import pdb; pdb.set_trace()
    results = {}
    for episode in tqdm(ds):
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()
        # print("file_path:", file_path)

        results_json = label_single_task(vlm, file_path, episode)
        results.update(results_json)

    if args.results_path is None:
        cot_dir = os.path.join(args.dataset_dir.replace("tfds_datasets", "planning_datasets"), data_name)
        os.makedirs(cot_dir, exist_ok=True)
        args.results_path = os.path.join(cot_dir, "scene_descriptions.json")

    # Write to json file
    with open(args.results_path, "w") as f:
        json.dump(results, f, cls=NumpyFloatValuesEncoder)

    print("Finished scene descriptions.")
