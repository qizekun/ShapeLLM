CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 ReConV2/main.py \
    --config ReConV2/cfgs/pretrain/large/openshape_1k.yaml \
    --exp_name $1 \
    --distributed \
    --reconstruct