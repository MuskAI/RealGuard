
GPU_NUM=3
WORLD_SIZE=1
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=29513


DISTRIBUTED_ARGS="
    --nproc_per_node $GPU_NUM \
    --nnodes $WORLD_SIZE \
    --node_rank $RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

PY_ARGS=${@:1}  # Any other arguments 

python -m torch.distributed.launch $DISTRIBUTED_ARGS main_finetune_all.py \
    --model PMIL \
    --batch_size 64 \
    --blr 1e-4 \
    --epochs 100 \
    ${PY_ARGS}
