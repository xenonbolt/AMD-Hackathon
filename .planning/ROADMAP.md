# Proposed Roadmap

**3 phases** | **7 requirements mapped** | All v1 requirements covered ✓

| # | Phase | Goal | Requirements | Success Criteria |
|---|-------|------|--------------|------------------|
| 1 | Architecture & Networking | Establish the 4-node swarm and Gloo communication layer | ARCH-01, ARCH-02, ARCH-03 | 3 |
| 2 | Pipeline Migration | Refactor ML codebase for CPU FSDP and distributed data loading | PIPE-01, PIPE-02, PIPE-03 | 3 |
| 3 | Validation & Metrics | Benchmark and verify convergence of CPU FSDP model | VAL-01, VAL-02 | 2 |

### Phase Details

**Phase 1: Architecture & Networking**
**Goal:** Establish the 4-node swarm and Gloo communication layer
**Mode:** mvp
**Requirements:** ARCH-01, ARCH-02, ARCH-03
**Success criteria:**
1. Master node can successfully ping and initialize a `gloo` process group with 3 worker nodes.
2. Rank assignment is correctly applied (Master is rank 0, workers 1-3).
3. A simple distributed tensor all-reduce operation passes across all 4 nodes over CPU.

**Phase 2: Pipeline Migration**
**Goal:** Refactor ML codebase for CPU FSDP and distributed data loading
**Mode:** mvp
**Requirements:** PIPE-01, PIPE-02, PIPE-03
**Success criteria:**
1. All CUDA-specific code (`bitsandbytes`, `.cuda()` calls) is gracefully removed or conditionally disabled.
2. Model parameters are successfully sharded across CPU RAM of the 4 nodes using PyTorch FSDP.
3. `DistributedSampler` ensures each node processes a unique shard of the JSONL dataset during training.

**Phase 3: Validation & Metrics**
**Goal:** Benchmark and verify convergence of CPU FSDP model
**Mode:** mvp
**Requirements:** VAL-01, VAL-02
**Success criteria:**
1. Training loop completes at least 1 full epoch without Out-Of-Memory (OOM) errors on any node.
2. CPU and RAM utilization metrics are logged across all nodes.
3. The resulting fine-tuned model passes existing inference engine checks for vulnerability detection.
