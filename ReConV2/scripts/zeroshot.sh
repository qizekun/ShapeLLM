CUDA_VISIBLE_DEVICES=$1 python ReConV2/main.py \
    --config ReConV2/cfgs/pretrain/large/openshape.yaml \
    --zeroshot \
    --exp_name $2 \
    --ckpts $3