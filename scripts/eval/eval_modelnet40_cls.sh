#!/bin/bash

python llava/eval/evaluator.py \
    --results_path ModelNet_classification_prompt0.json \
    --model_type gpt-3.5-turbo-0613 \
    --eval_type modelnet-close-set-classification \
    --parallel \
    --num_workers 15