#!/bin/bash

python llava/eval/evaluator.py \
    --results_path PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt0.json \
    --model_type gpt-4-0613 \
    --eval_type open-free-form-classification \
    --parallel \
    --num_workers 15