import argparse
import json
from tqdm import tqdm
import os

# try:
#     hydra.initialize(config_path="../calvin/calvin_models/conf")
# except Exception:
#     pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planning_dir", type=str)
    parser.add_argument("--action_horizon", type=int, default=10)
    args = parser.parse_args()

    bboxes_file_path = os.path.join(args.planning_dir, "bboxes.json")
    with open(bboxes_file_path, "r") as f:
        bboxes = json.load(f)

    gripper_positions_file_path = os.path.join(args.planning_dir, "gripper_positions.json")
    with open(gripper_positions_file_path, "r") as f:
        gripper_positions = json.load(f)

    primitives_file_path = os.path.join(args.planning_dir, f"primitives_h{args.action_horizon}.json")
    with open(primitives_file_path, "r") as f:
        primitives = json.load(f)

    reasonings_file_path = os.path.join(args.planning_dir, f"filtered_reasoning_h{args.action_horizon}.json")
    with open(reasonings_file_path, "r") as f:
        reasonings = json.load(f)

    for file_path in tqdm(reasonings.keys(), desc="Merging"):
        if file_path not in bboxes:
            print(f"File path {file_path} not found in bboxes")
            continue
        if file_path not in gripper_positions:
            print(f"File path {file_path} not found in gripper_positions")
            continue
        if file_path not in primitives:
            print(f"File path {file_path} not found in primitives")
            continue
        # Check if bbox and gripper_position include the last state. If so, remove it
        bbox = bboxes[file_path]#[:-1]
        gripper_position = gripper_positions[file_path]#[:-1]
        primitive = primitives[file_path]

        try:
            assert len(bbox) == len(gripper_position) == len(primitive) == len(reasonings[file_path]["0"]["reasoning"]), f"Length mismatch for {file_path}: {len(bbox)}, {len(gripper_position)}, {len(primitive)}, {len(reasonings[file_path]['0']['reasoning'])}"
        except Exception as e:
            print(e)
            continue

        reasonings[file_path]["0"]["features"].update(
            {
                "bboxes": bbox,
                "gripper_position": gripper_position,
                "move_primitive": primitive
            }
        )

    target_file_path = os.path.join(args.planning_dir, f"reasoning.json")

    with open(target_file_path, "w") as f:
        json.dump(reasonings, f)
