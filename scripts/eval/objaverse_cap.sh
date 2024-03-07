#!/bin/bash

MODEL_VERSION=llama-vicuna-7b
TAG=v1.0

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.eval_objaverse \
    --model-path ./checkpoints/$MODEL_VERSION-$TAG-finetune \
    --task_type captioning \
    --prompt_index 2 \
    --num_beams 5