import argparse
import json
import os
import warnings
from tqdm import tqdm
import numpy as np

from PIL import Image
from utils import (
    NumpyFloatValuesEncoder,
    mask_to_bboxes,
    show_box
)

from env_configs import INSTANCE_ID_TO_NAMES

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

warnings.filterwarnings("ignore")

image_dims = (256, 256)


def label_single_task(data_path, episode, debug=False):
    print(f"Processing {data_path} ...")
    if debug:
        os.makedirs("debug/vis_bboxes")

    bbox_results_json = {}
    task = data_path.split("/")[-1][:9]
    instance_id_to_names = INSTANCE_ID_TO_NAMES[task]
    bboxes_list = []
    for i, step in enumerate(episode["steps"]):
        image = Image.fromarray(
            step["observation"]["image"].numpy().astype(np.uint8),
            mode="RGB",
        )
        mask = step["observation"]["mask"].numpy().astype(np.uint8)

        bboxes = mask_to_bboxes(mask, instance_id_to_names)
        # import pdb; pdb.set_trace()
        if debug:
            # Plot the image using matplotlib
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(image)
            ax.axis("off")

            # Draw bounding boxes
            for text, bbox in bboxes:
                show_box(bbox, ax, text, "red")
            
            output_path = os.path.join("debug/vis_bboxes", f"{data_path.replace('/', '_')}_step_{i}.png")
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        bboxes_list.append(bboxes)
    bbox_results_json[data_path] = bboxes_list
    return bbox_results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--dataset_name", type=str, default="example_dataset")
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    data_name = args.dataset_name
    ds = tfds.load(
        data_name,
        data_dir=args.dataset_dir,
        split=f"train[{0}%:{100}%]",
    )

    results = {}
    for episode in tqdm(ds):
        data_path = episode["episode_metadata"]["file_path"].numpy().decode()
        results_json = label_single_task(data_path, episode, args.debug)
        results.update(results_json)
        if args.debug:
            break

    if args.results_path is None:
        cot_dir = os.path.join(args.dataset_dir.replace("tfds_datasets", "planning_datasets"), data_name)
        os.makedirs(cot_dir, exist_ok=True)
        args.results_path = os.path.join(cot_dir, "bboxes.json")

    # Write to json file
    with open(args.results_path, "w") as f:
        json.dump(results, f, cls=NumpyFloatValuesEncoder)
