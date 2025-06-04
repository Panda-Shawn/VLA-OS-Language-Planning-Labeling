import json
from tqdm import tqdm
import argparse
from utils import (
    calculate_2d_position,
    calculate_camera_intrinsics,
    NumpyFloatValuesEncoder
)
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from env_configs import IMAGE_SIZE, CAMERA_PARAS


def label_single_task(data_path, episode, debug=False):
    print(f"Processing {data_path} ...")
    if debug:
        os.makedirs("debug/vis_gpos")

    gripper_pos_results_json = {}
    gripper_pos = []
    task = data_path.split("/")[-1][:9]
    for i, step in enumerate(episode["steps"]):
        image = step["observation"]["image"].numpy().astype(np.uint8)
        gripper_positions_3d = step["observation"]["state"].numpy()[:3]
        camera_pos = CAMERA_PARAS[task]["front"]["pos"]
        camera_quat = CAMERA_PARAS[task]["front"][
            "quat"
        ]
        fovy = CAMERA_PARAS[task]["front"]["fovy"]
        resolution = IMAGE_SIZE
        camera_intrinsics = calculate_camera_intrinsics(fovy, resolution)

        # Project tcp position in the image
        u, v = calculate_2d_position(
            gripper_positions_3d,
            camera_pos,
            camera_quat,
            camera_intrinsics,
            scalar_first=False,
        )
        gripper_pos.append([int(u), int(v)])
        if debug:
            # 使用 Matplotlib 可视化
            plt.figure(figsize=(8, 6))
            plt.imshow(image)  # 显示图像
            plt.scatter(
                gripper_pos[-1][0],
                gripper_pos[-1][1],
                c="red",
                s=50,
                label="Gripper Position",
            )  # 标记点
            plt.legend()
            plt.axis("off")
            plt.savefig(
                os.path.join("debug/vis_gpos", f"{data_path.replace('/', '_')}_step_{i}.png"), bbox_inches="tight", dpi=300
            )  # bbox_inches 去掉多余边距，dpi 控制分辨率
            plt.close()

    gripper_pos_results_json[data_path] = gripper_pos
    # origin_data.close()
    return gripper_pos_results_json


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
        args.results_path = os.path.join(cot_dir, "gripper_positions.json")

    # Write to json file
    with open(args.results_path, "w") as f:
        json.dump(results, f, cls=NumpyFloatValuesEncoder)
