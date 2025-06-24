#!/bin/bash

export GOOGLE_API_KEY=



python ./cot_code/batch_filter_plan_subtasks.py \
    --libero_dataset_dir ./results/bucket/filtered \
    --libero_primitives_path ./data/bucket/primitives.json \
    --libero_scene_desc_path ./data/bucket/descriptions_bucket_fixed.json \
    --libero_plan_subtasks_path ./results/bucket/cot/chain_of_thought_h10.json \
    --api_provider gemini
