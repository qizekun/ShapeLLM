#!/bin/bash

LLM_VERSION=lmsys/vicuna-13b-v1.1
MODEL_VERSION=shapellm-13b
PRETRAIN_TAG=v1.0
TAG=v1.0

type=general

if [ $type = "general" ]; then
    meta_path="./playground/data/shapellm/cap3d_objaverse_sft_45k.json"
    pcs_path="./playground/data/shapellm/cap3d_pcs"
elif [ $type = "gapartnet" ]; then
    meta_path="./playground/data/shapellm/gapartnet_sft_27k_openai.json"
    pcs_path="./playground/data/shapellm/gapartnet_pcs"
else
    echo "Unknown type"
    exit 1
fi

deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $LLM_VERSION \
    --version v1 \
    --data_path $meta_path \
    --point_folder $pcs_path \
    --vision_tower ReConV2/cfgs/pretrain/large/openshape.yaml \
    --vision_tower_path ./checkpoints/recon/large.pth \
    --sample_points_num 10000 \
    --with_color True \
    --occlusion False \
    --prompt_token_num 32 \
    --with_ape True \
    --with_local True \
    --with_global True \
    --pretrain_mm_mlp_adapter ./checkpoints/$MODEL_VERSION-$PRETRAIN_TAG-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_pt_start_end False \
    --mm_use_pt_patch_token False \
    --point_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/$MODEL_VERSION-$type-$TAG-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb