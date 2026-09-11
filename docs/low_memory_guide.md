# Low Memory Training & Inference Guide

This guide details the specific optimizations implemented to run 26GB+ Large Language Models on a distributed CPU swarm where each node is physically capped at 15GB of RAM.

## 🧠 Training on Low Memory (FSDP Swarm)

The pipeline employs a hybrid software/infrastructure approach to bypass the initial OOM (Out-of-Memory) crash when loading a model before FSDP can shard it.

### Software Optimizations
1. **BFloat16 Precision**: The `fine_tune.py` and `train_fix_model.py` scripts natively load the Hugging Face base models in `torch.bfloat16`. This instantly reduces the memory footprint of weights by 50% without degrading training accuracy.
2. **`low_cpu_mem_usage`**: This Hugging Face feature is enabled to load the model sequentially, preventing transient memory spikes during initialization.
3. **Aggressive Sync**: The FSDP configuration includes `"sync_module_states"`, ensuring that FSDP begins sharding memory immediately upon wrapping the model.

### Infrastructure Optimizations (Swap Allocation)
Even at BFloat16, massive models exceed 15GB during initialization. We solve this transparently at the OS layer using the `cluster/setup_swap.sh` script.

**How it works:**
1. Before `torchrun` begins, the orchestrator connects to every node and dynamically creates a **16GB Swap file** (`/swapfile`).
2. This creates a combined virtual memory pool of **31GB** per node (15GB RAM + 16GB Swap).
3. The OS safely loads the model into this pool without crashing PyTorch.
4. Once FSDP shards the model, memory usage drops drastically (to ~4.5GB per node), and the OS pages the active memory back into high-speed physical RAM for the training loop.

> [!WARNING]
> **Important Prerequisite**: The `setup_swap.sh` script requires `sudo` privileges. Because it is executed automatically over SSH via `launch_cluster.sh`, the `mpiusers` account on all nodes **MUST have passwordless sudo configured**.
> 
> To configure this on all worker nodes, run:
> `echo "mpiusers ALL=(ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/mpiusers`

## 🏃 Running Inference on Low Memory

When you run `inference_engine.py` on a single 15GB machine (not in the swarm), you cannot load the full 26GB FSDP-consolidated model without crashing.

To run inference on a low-RAM node, you must enable **Disk Offloading**:

1. Modify your inference initialization to include `device_map="cpu"` and `offload_folder`.
2. This allows `accelerate` to load parts of the model to the SSD and stream layers into RAM sequentially as the inference engine predicts tokens.

*Example Inference Snippet:*
```python
from transformers import AutoModelForCausalLM

# Load the consolidated checkpoint on a 15GB node using offloading
model = AutoModelForCausalLM.from_pretrained(
    "./adapters/final_checkpoint",
    torch_dtype=torch.bfloat16,
    device_map="cpu", 
    max_memory={"cpu": "12GiB"}, # Leave 3GB for the OS
    offload_folder="./inference_offload"
)
```
*Note: Disk offloading is strictly for inference. It should NOT be used during training as it conflicts with PyTorch FSDP.*
