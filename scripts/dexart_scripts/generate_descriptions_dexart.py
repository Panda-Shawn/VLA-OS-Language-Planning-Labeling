import argparse
import json
import os
import warnings

import tensorflow_datasets as tfds
import torch
from PIL import Image
from tqdm import tqdm
from dexart.scense_description.scripts.utils import NumpyFloatValuesEncoder
from prismatic import load

from utils import DATA_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--id", default=0, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--splits", default=1, type=int)
parser.add_argument("--results-path", default=str(DATA_DIR / "planning_datasets/bucket_dex_art_dataset/results_descriptions/bucket"))
args = parser.parse_args()

args.results_path = str(args.results_path)

device = f"cuda:{args.gpu}"
hf_token = ""
vlm_model_id = "prism-dinosiglip+7b"

warnings.filterwarnings("ignore")

split_percents = 100 // args.splits
start_percent = args.id * split_percents
end_percent = (args.id + 1) * split_percents

builder = tfds.builder_from_directory(
    "/data/gck/vla_planning/dataset/dexart/bucket_dex_art_dataset/1.0.0"
)
viz="bucket_viz"
ds = builder.as_dataset(split=f"train[{start_percent}%:{end_percent}%]")

print(f"[INFO] Loaded {builder.info.splits['train'].num_examples} examples in total.")

print(f"Loading Prismatic VLM ({vlm_model_id})...")
vlm = load(vlm_model_id, hf_token=hf_token)
vlm = vlm.to(device, dtype=torch.bfloat16)

results_json_path = os.path.join(args.results_path, f"results_{args.id}.json")
print(f"[INFO] Results will be saved to {results_json_path}")
os.makedirs(args.results_path, exist_ok=True)

def create_user_prompt(lang_instruction):
    user_prompt = "Briefly describe the things in this scene and their spatial relations to each other."
    lang_instruction = lang_instruction.strip()
    if len(lang_instruction) > 0 and lang_instruction.endswith("."):
        lang_instruction = lang_instruction[:-1]
    if len(lang_instruction) > 0 and " " in lang_instruction:
        user_prompt = f"The robot task is: '{lang_instruction}.' " + user_prompt
    return user_prompt

def create_seg_prompt(caption):
    return f"The caption is: '{caption}' List only the objects found in the caption separated by dots."

results_json = {}

for i, episode in tqdm(enumerate(ds), desc="Episodes"):
    try:
        episode_id = f"episode_{i}"
        file_path = episode["episode_metadata"]["file_path"].numpy().decode()

        for step_idx, step in tqdm(enumerate(episode["steps"]), desc=f"[Episode {episode_id}]"):
            lang_instruction = step["language_instruction"].numpy().decode()

            image_tensor = step["observation"][viz].numpy()
            image = Image.fromarray(image_tensor).convert("RGB")

            user_prompt = create_user_prompt(lang_instruction)
            prompt_builder = vlm.get_prompt_builder()
            prompt_builder.add_turn(role="human", message=user_prompt)
            prompt_text = prompt_builder.get_prompt()

            torch.manual_seed(0)
            caption = vlm.generate(
                image,
                prompt_text,
                do_sample=True,
                temperature=0.4,
                max_new_tokens=64,
                min_length=1,
            )

            frame_key = f"{file_path}_{step_idx}"
            frame_json = {
                "caption": caption,
                "task_description": lang_instruction,
            }
            results_json[frame_key] = frame_json

            break

        with open(results_json_path, "w") as f:
            json.dump(results_json, f, cls=NumpyFloatValuesEncoder, indent=2)

    except Exception as e:
        print(f"[ERR] Error in episode {i}: {e}")
        continue

print(f"[OK] Results saved to {results_json_path}")
