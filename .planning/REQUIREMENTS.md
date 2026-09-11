# Requirements

## v1 Requirements

### Architecture & Networking
- [ ] **ARCH-01**: Implement PyTorch DistributedDataParallel or Native FSDP using the `gloo` backend for inter-node communication.
- [ ] **ARCH-02**: Establish a 4-node swarm configuration (1 master node, 3 worker nodes).
- [ ] **ARCH-03**: Configure node discovery and rank assignment for distributed CPU training.

### Pipeline Migration
- [ ] **PIPE-01**: Refactor `fine_tune.py` and `train_fix_model.py` to remove GPU/CUDA-specific dependencies (e.g., `bitsandbytes`, `accelerate` with CUDA flags).
- [ ] **PIPE-02**: Implement CPU-based parameter sharding to fit large models within the collective RAM of the 4 Ubuntu nodes.
- [ ] **PIPE-03**: Ensure data loading (`data_preparation.py`) handles distributed data sampling (using PyTorch `DistributedSampler`).

### Validation & Metrics
- [ ] **VAL-01**: Benchmark CPU training throughput (tokens/sec) and memory utilization across the swarm.
- [ ] **VAL-02**: Verify that the fine-tuned model outputs from CPU FSDP training match expected convergence quality compared to the legacy GPU pipeline.

## v2 Requirements
- [ ] Dynamic node scaling (adding/removing nodes mid-training).
- [ ] Fault tolerance (checkpointing and restoring if a worker node goes down).

## Out of Scope
- **GPU Acceleration**: Explicitly excluded. The cluster only consists of CPU and RAM resources.
- **DeepSpeed/Ray**: Excluded for v1 to minimize dependency overhead; focusing on native PyTorch FSDP first.

## Traceability
*(To be filled by Roadmap)*
