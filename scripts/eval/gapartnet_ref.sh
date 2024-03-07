#!/bin/bash

MODEL_VERSION=shapellm-13b
TAG=gapartnet-v1.0

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_vqa \
    --model-path ./checkpoints/$MODEL_VERSION-$TAG-finetune \
    --question-file ./playground/data/eval/gapartnet/question.jsonl \
    --point-folder ./playground/data/shapellm/gapartnet_pcs \
    --answers-file ./playground/data/eval/gapartnet/answers/$MODEL_VERSION-$TAG.jsonl \
    --conv-mode vicuna_v1 \
    --num_beams 5