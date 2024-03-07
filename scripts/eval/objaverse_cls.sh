#!/bin/bash

MODEL_VERSION=llama-vicuna-7b
TAG=v1.0

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.eval_objaverse \
    --model-path ./checkpoints/$MODEL_VERSION-$TAG-finetune \
    --task_type classification \
    --prompt_index 0 \
    --num_beams 5