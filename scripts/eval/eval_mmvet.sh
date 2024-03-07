#!/bin/bash

MODEL_VERSION=shapellm-13b
TAG=general-v1.0

mkdir -p ./playground/data/eval/3d-mm-vet/results

python -m llava.eval.eval_3dmmvet \
    --answers-file ./playground/data/eval/3d-mm-vet/answers/$MODEL_VERSION-$TAG.jsonl \
    --gt-file ./playground/data/eval/3d-mm-vet/gt.jsonl \
    --output-file ./playground/data/eval/3d-mm-vet/results/$MODEL_VERSION-$TAG.jsonl \
    --model gpt-4-0125-preview \
    --max_workers 16 \
    --times 5