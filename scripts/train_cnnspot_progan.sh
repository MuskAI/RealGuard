
GPU_NUM=1
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29517


DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

PY_ARGS=${@:1}  # Any other arguments 

CUDA_VISIBLE_DEVICES=2 python -m torch.distributed.launch $DISTRIBUTED_ARGS main_finetune_all.py \
    --model PMIL \
    --batch_size 64 \
    --blr 1e-4 \
    --epochs 20 \
    --data_path /raid0/chr/AIGCD/Datasets/CNNSpot_trainingdata/progan_train \
    --eval_data_path /raid0/chr/AIGCD/Datasets/CNNSpot_val/progan \
    --output_dir /raid5/chr/AIGCD/AIDE/results/cnnspot-reprodeuce \
    --log_dir /raid5/chr/AIGCD/AIDE/results/cnnspot-reprodeuce \
    ${PY_ARGS}
