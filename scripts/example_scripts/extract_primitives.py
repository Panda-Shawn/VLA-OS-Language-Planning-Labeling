import argparse
import numpy as np
import os
import json
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import tensorflow_datasets as tfds


def describe_move(move_vec):
    names = [
        {-1: "backward", 0: None, 1: "forward"},
        {-1: "right", 0: None, 1: "left"},
        {-1: "down", 0: None, 1: "up"},
        {-1: "rotate counterclockwise", 0: None, 1: "rotate clockwise"},
        {},
        {-1: "tilt down", 0: None, 1: "tilt up"},
        {-1: "open gripper", 0: None, 1: "close gripper"},
    ]

    xyz_move = [names[i][move_vec[i]] for i in range(0, 3)]
    xyz_move = [m for m in xyz_move if m is not None]

    if len(xyz_move) != 0:
        description = "move " + " ".join(xyz_move)
    else:
        description = ""

    if move_vec[3] == 0:
        move_vec[3] = move_vec[4]  # identify rolling and pitching

    if move_vec[3] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[3][move_vec[3]]

    if move_vec[5] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[5][move_vec[5]]

    if move_vec[6] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[6][move_vec[6]]

    if len(description) == 0:
        description = "stop"

    return description


def classify_movement(move, threshold=0.03):
    diff = move[-1] - move[0]

    if np.sum(np.abs(diff[:3])) > 3 * threshold:
        diff[:3] *= 3 * threshold / np.sum(np.abs(diff[:3]))

    diff[3:6] /= 6

    move_vec = 1 * (diff > threshold) - 1 * (diff < -threshold)

    return describe_move(move_vec), move_vec


def extract_single_task(data_path, episode, action_horizon):
    results_json = {}
    print(f"Processing {data_path} ...")

    episode_ee_trans = []
    episode_ee_rots = []
    episode_action = []
    for step in episode["steps"]:
        episode_ee_trans.append(step["observation"]["state"].numpy()[:3])
        episode_ee_rots.append(step["observation"]["state"].numpy()[3:])
        episode_action.append(step["action"].numpy())
    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    episode_action = np.array(episode_action)
    gripper_action = episode_action[:, -1:]

    episode_states = np.concatenate([episode_ee_trans, episode_ee_rots, gripper_action], axis=-1)

    move_trajs = [
        episode_states[i : i + action_horizon]
        for i in range(len(episode_states) - 1)
    ]
    primitives_list = [classify_movement(move)[0] for move in move_trajs]
    primitives_list.append(primitives_list[-1])

    results_json[data_path] = primitives_list

    return results_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--dataset_name", type=str, default="example_dataset")
    parser.add_argument("--action_horizon", type=int)
    parser.add_argument("--results_path", type=str, default=None)
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
        results_json = extract_single_task(data_path, episode, args.action_horizon)
        results.update(results_json)

    if args.results_path is None:
        cot_dir = os.path.join(args.dataset_dir.replace("tfds_datasets", "planning_datasets"), data_name)
        os.makedirs(cot_dir, exist_ok=True)
        args.results_path = os.path.join(
            cot_dir, f"primitives_h{args.action_horizon}.json"
        )

    print("Saving results to", args.results_path)
    with open(args.results_path, "w") as f:
        json.dump(results, f)
