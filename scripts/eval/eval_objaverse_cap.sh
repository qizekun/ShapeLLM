#!/bin/bash

python llava/eval/evaluator.py \
    --results_path PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json \
    --model_type gpt-4-0613 \
    --eval_type object-captioning \
    --parallel \
    --num_workers 15

python llava/eval/traditional_evaluator.py \
    --results_path PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json