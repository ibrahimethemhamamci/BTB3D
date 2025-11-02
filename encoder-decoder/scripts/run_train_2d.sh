RANK=$1
MASTER_ADDR=$2
NUM_MACHINES=$3
NUM_PROCESSES=$4

IP=$(echo "$MASTER_ADDR" | awk -F':' '{print $1}')
PORT=$(echo "$MASTER_ADDR" | awk -F':' '{print $2}')

echo "node: $NODE, rank: $RANK, num_machines: $NUM_MACHINES, num_processes: $NUM_PROCESSES, master addr: $MASTER_ADDR, ip: $IP, port: $PORT"

LAUNCH_ARGS="--main_process_ip $IP \
--main_process_port $PORT \
--machine_rank $RANK \
--num_machines $NUM_MACHINES \
--num_processes $NUM_PROCESSES \
--main_training_function main \
--mixed_precision no \
--rdzv_backend static \
--dynamo_backend no \
"

if [ "$NUM_PROCESSES" -gt 1 ]; then
    echo "num processes > 1, use multi-gpu"
    LAUNCH_ARGS="$LAUNCH_ARGS\
"
fi

cmd=$(cat <<EOF
NCCL_IB_TIMEOUT=12 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_FILE=/data1/nccl_debug_%h.%p \
NCCL_IB_CUDA_SUPPORT=1 \
NCCL_IBEXT_DISABLE=1 \
NCCL_IB_DISABLE=0 \
NCCL_NVLS_ENABLE=0 \
NCCL_IB_RETRY_CNT=7 \
NCCL_IB_GID_INDEX=3 \
TORCH_DISTRIBUTED_DEBUG=INFO \
accelerate launch \
$LAUNCH_ARGS \
train.py \
--model_config_path configs/magvit2_2d_model_config.yaml \
--trainer_config_path configs/magvit2_2d_train_config.yaml
EOF
)
# Print the command with variables expanded
echo "Running command: $cmd"

# Run the command
eval "$cmd"