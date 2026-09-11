# CPU-Distributed Model Fine-Tuning Swarm

## What This Is

A distributed CPU-only machine learning training swarm using PyTorch Native FSDP over Gloo. It adapts an existing single-node GPU-based ML pipeline for vulnerability detection and remediation to run distributed fine-tuning across 4 Ubuntu servers (1 master, 3 nodes) using only CPU and RAM.

## Core Value

Scale model fine-tuning horizontally using CPU and RAM resources across multiple commodity Ubuntu nodes without requiring GPUs.

## Requirements

### Validated

- ✓ Data preparation pipeline (`data_preparation.py` handling JSONL loading and tokenization) — existing
- ✓ Base model fine-tuning logic for vulnerability detection/remediation (`fine_tune.py`, `train_fix_model.py`) — existing
- ✓ Inference engines and static scanner (`inference_engine.py`, `fix_engine.py`, `scanner.py`) — existing
- ✓ API Backend (FastAPI) — existing
- ✓ React SPA Frontend — existing

### Active

- [ ] Migrate `fine_tune.py` and `train_fix_model.py` from QLoRA/GPU-centric code to CPU-only PyTorch FSDP.
- [ ] Implement PyTorch Native FSDP using the `gloo` backend for inter-node communication.
- [ ] Configure the swarm topology (1 master, 3 worker nodes).
- [ ] Enable parameter sharding across the CPUs/RAM of all 4 servers to handle larger models without OOM on a single node.

### Out of Scope

- GPU-accelerated training — Explicitly targeting a CPU/RAM-only 4-node Ubuntu swarm.

## Context

- The existing codebase uses Hugging Face `transformers`, `peft` (QLoRA), `bitsandbytes`, and `accelerate` which are heavily optimized for CUDA/GPUs. Adapting this for CPU will require careful dependency management.
- Target environment: 4 Ubuntu servers forming a compute swarm over a network.
- The user chose PyTorch Native FSDP over Gloo as the distributed training architecture for CPU.

## Constraints

- **Hardware**: Must use CPU and RAM only (no GPUs) — The 4 Ubuntu servers do not have GPU acceleration available for this workload.
- **Dependencies**: Will likely need to remove or replace `bitsandbytes` and standard QLoRA implementations since they are inherently CUDA-dependent.
- **Networking**: Must rely on the `gloo` backend since `nccl` is GPU-only.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| PyTorch Native FSDP over Gloo | Most standard approach for parameter sharding on CPUs without external frameworks | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-11 after initialization*
