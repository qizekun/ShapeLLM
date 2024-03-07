CUDA_VISIBLE_DEVICES=$1 python ReConV2/main.py \
    --config ReConV2/cfgs/large/fewshot.yaml \
    --finetune_model \
    --exp_name $2 \
    --ckpts $3 \
    --way $4 \
    --shot $5 \
    --fold $6