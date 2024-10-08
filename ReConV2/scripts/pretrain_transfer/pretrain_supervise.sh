CUDA_VISIBLE_DEVICES=$1 python ReConV2/main.py \
    --config ReConV2/cfgs/pretrain/large/hybrid_post.yaml \
    --finetune_model \
    --exp_name $2 \
    --ckpts $3 \
    --seed $RANDOM