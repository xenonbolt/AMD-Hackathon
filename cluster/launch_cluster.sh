#!/bin/bash

# Usage: ./launch_cluster.sh <script_to_run> <master_ip> <worker_ip_1> [worker_ip_2 ...]
# Example: ./launch_cluster.sh cluster/test_gloo.py 10.0.0.1 10.0.0.2 10.0.0.3 10.0.0.4

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <script_to_run> <master_ip> <worker_ip_1> [worker_ip_2 ...]"
    exit 1
fi

SCRIPT_TO_RUN=$1
MASTER_IP=$2
shift 2

WORKER_IPS=("$@")
NNODES=$(( ${#WORKER_IPS[@]} + 1 ))
RDZV_PORT=29500
SSH_USER="mpiusers"

# Ensure absolute path for the script if it's relative
SCRIPT_PATH=$(realpath "$SCRIPT_TO_RUN")

echo "======================================"
echo "Launching PyTorch Distributed Swarm"
echo "Master IP: $MASTER_IP"
echo "Worker IPs: ${WORKER_IPS[*]}"
echo "Total Nodes: $NNODES"
echo "Script: $SCRIPT_PATH"
echo "SSH User: $SSH_USER"
echo "======================================"

# 1. Setup Swap on all nodes
echo "Setting up 16GB swap space across the swarm to prevent OOM during initialization..."
SWAP_SCRIPT_PATH=$(realpath "cluster/setup_swap.sh")

for IP in "${WORKER_IPS[@]}"; do
    echo "Configuring swap on worker: $IP..."
    ssh -n ${SSH_USER}@${IP} "bash -s" < "$SWAP_SCRIPT_PATH"
done

echo "Configuring swap on master: $MASTER_IP..."
bash "$SWAP_SCRIPT_PATH"
echo "Swap configuration complete."

# 2. Launch on worker nodes via SSH
for IP in "${WORKER_IPS[@]}"; do
    echo "Starting torchrun on worker: $IP..."
    ssh -n -f ${SSH_USER}@${IP} "bash -c 'nohup torchrun \
        --nnodes=${NNODES} \
        --nproc_per_node=1 \
        --rdzv_id=cpu_swarm_1 \
        --rdzv_backend=c10d \
        --rdzv_endpoint=${MASTER_IP}:${RDZV_PORT} \
        ${SCRIPT_PATH} > /tmp/torchrun_${IP}.log 2>&1 &'"
done

# 3. Launch on master node
echo "Starting torchrun on master: $MASTER_IP..."
torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=1 \
    --rdzv_id=cpu_swarm_1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_IP}:${RDZV_PORT} \
    ${SCRIPT_PATH}

echo "Master node finished. Check worker logs at /tmp/torchrun_<ip>.log if needed."
