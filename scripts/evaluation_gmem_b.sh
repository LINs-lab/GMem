export PYTHONPATH=./

PRECISION=${PRECISION:-bf16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CONFIG_PATHS=(
    "configs/gmem_sde_b.yaml"
)

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
    echo "========================================"
    echo "Using config: $CONFIG_PATH"
    echo "Using $PRECISION precision"

    accelerate launch \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --machine_rank $NODE_RANK \
        --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
        --num_machines $NNODES \
        --mixed_precision $PRECISION \
        engine/evaluation.py \
        --config $CONFIG_PATH

    echo "Finished running with $CONFIG_PATH"
    echo "========================================"
    echo ""
done