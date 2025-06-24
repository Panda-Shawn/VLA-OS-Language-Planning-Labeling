#!/bin/bash

export GOOGLE_API_KEY=

python ./scripts/dexart_scripts/cot_scripts/batch_generate_plan_subtasks.py \
    --libero_dataset_dir ./results/bucket \
    --libero_primitives_path ./data/bucket/primitives.json \
    --libero_scene_desc_path ./data/bucket/descriptions_bucket_fixed.json \
    --batch_size 10 \
    --api_provider gemini
