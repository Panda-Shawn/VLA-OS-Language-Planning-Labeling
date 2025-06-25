# VLA-OS-Language-Planning-Labeling
Language planning labeling for VLA-OS

## Installation

First, create a virtual environment with `python==3.10` and `torch`.

```bash
# Create and activate conda environment
conda create -n plan_label python=3.10 -y
conda activate plan_label

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

Then, install this repo.

```bash
pip install -e .
```

## Usage

We will use an example dataset to illustrate how to convert a numpy dataset into a tfds dataset, and then label the language planning as described in our paper. We also provide the scripts for `libero_10` dataset as an example [here](./scripts/libero_scripts/).

### Dataset Generation

First, generate a dummy example dataset.

```bash

python tfds_dataset_generation/example_dataset/create_example_data.py
```

Then the example numpy dataset will be saved to `data/raw_datasets`. Convert it into a tfds dataset with a specified saving path `data/tfds_datasets`.

```bash

tfds build tfds_dataset_generation/example_dataset --data_dir data/tfds_datasets
```

### Language Planning Labeling

We will first generate step-wise language reasoning and then add some visual features in language.

#### Language Reasoning

First, describe the scene with Prismatic-7B VLM.

```bash

python scripts/example_scripts/describe_scene.py --dataset_dir data/tfds_datasets
```

Then, extract the primitive moves with horizon of 10. Verify whether the interpretation of the `state` and `action` values is correct. The first six elements of the state represent translation and rotation in the world coordinate frame. In `gripper_action`, a value of +1 indicates closing the gripper, while -1 indicates opening it.

```bash

python scripts/example_scripts/extract_primitives.py --dataset_dir data/tfds_datasets --action_horizon 10
```

Next, query Gemini-1.5-flash for the initial reasoning. Remember to use your own gemini api key at `GOOGLE_API_KEY=xxx`.

```bash

GOOGLE_API_KEY=xxx python scripts/example_scripts/batch_generate_plan_subtasks.py --batch_size 1 --action_horizon 10 --dataset_dir data/tfds_datasets --force_regenerate
```

Finally, query Gemini-1.5-flash again to filter the initial reasoning for the final reasoning.

```bash

GOOGLE_API_KEY=xxx python scripts/example_scripts/batch_filter_plan_subtasks.py --batch_size 1 --action_horizon 10 --dataset_dir data/tfds_datasets --force_regenerate
```

#### Visual Features

Object bounding boxes and gripper position are included for visual features in language planning. And there are different processes for simulation data and real-world data.

##### Simulation Data

In simulation, we obtain segmentation mask and camera parameters easily.

###### Object Bounding Boxes

We use the segmentation mask and the `INSTANCE_ID_TO_NAMES` dict in `env_configs.py` to get the bounding boxes for each step. The option `--debug` can be used to visualize the first episode and see if the labeling is correct.

```bash

python scripts/example_scripts/generate_bboxes.py --dataset_dir data/tfds_datasets
```

###### Gripper Position

We use the `IMAGE_SIZE` and the `CAMERA_PARAS` dict in `env_configs.py` to get the gripper position for each step. The option `--debug` can be used to visualize the first episode and see if the labeling is correct.

```bash

python scripts/example_scripts/detect_gripper_position.py --dataset_dir data/tfds_datasets
```

##### Real-World Data

The code will be coming soon.

#### Reasoning Merge

Merge the `filtered_reasoning_h10.json`, `bboxes.json` and `gripper_positions.json` under `data/planning_datasets/example_dataset/cot` to get the complete language planning json file.

```bash

python scripts/example_scripts/merge_plan_subtasks.py --planning_dir data/planning_datasets/example_dataset
```

## Acknowledgement

This repository is based on [Embodied-CoT](https://github.com/MichalZawalski/embodied-CoT).
