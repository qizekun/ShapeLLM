#!/bin/bash

MODEL_VERSION=llama-vicuna-7b
TAG=v1.0

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.eval_modelnet_cls \
    --model-path ./checkpoints/$MODEL_VERSION-$TAG-finetune \
    --prompt_index 0 \
    --num_beams 5