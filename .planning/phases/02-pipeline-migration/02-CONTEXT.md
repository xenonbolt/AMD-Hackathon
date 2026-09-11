# Phase 2: Pipeline Migration - Context

**Gathered:** 2026-09-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Refactoring the ML codebase (`fine_tune.py` and `train_fix_model.py`) to transition from GPU/QLoRA to CPU-based PyTorch FSDP with Gloo, ensuring distributed data loading.
</domain>

<decisions>
## Implementation Decisions

### LoRA Strategy
- **D-01:** Standard LoRA in FP32. It uses more RAM but is universally supported on CPU without quantization libraries.

### FSDP Sharding Strategy
- **D-02:** FULL_SHARD. Shards parameters, gradients, and optimizer states across all nodes. Maximizes memory savings to fit large models in CPU RAM.

### Checkpointing
- **D-03:** Consolidate on Master. FSDP will gather all shards to Rank 0 and save a single, unified model checkpoint for easy inference.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project Scope
- `.planning/PROJECT.md` — Project definition and constraints
- `.planning/REQUIREMENTS.md` — Complete v1 requirements
- `.planning/ROADMAP.md` — Phase goals and success criteria

</canonical_refs>

<code_context>
## Existing Code Insights

### Established Patterns
- Existing code relies heavily on `bitsandbytes` and `peft` with CUDA configurations. We will systematically remove CUDA assumptions.
</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches
</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope
</deferred>

---

*Phase: 2-Pipeline Migration*
*Context gathered: 2026-09-11*
