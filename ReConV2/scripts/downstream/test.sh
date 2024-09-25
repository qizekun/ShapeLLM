CUDA_VISIBLE_DEVICES=$1 python ReConV2/main.py \
    --test \
    --config ReConV2/cfgs/full/finetune_modelnet.yaml \
    --exp_name $2 \
    --ckpts $3 \
    --seed $RANDOM