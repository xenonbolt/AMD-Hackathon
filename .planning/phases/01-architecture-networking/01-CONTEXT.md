# Phase 1: Architecture & Networking - Context

**Gathered:** 2026-09-11
**Status:** Ready for planning

<domain>
## Phase Boundary

Establishing the 4-node Ubuntu swarm and configuring the Gloo communication layer for PyTorch CPU-based distributed training.
</domain>

<decisions>
## Implementation Decisions

### Network Topology & Discovery
- **D-01:** Use PyTorch's `torchrun` with the C10d rendezvous backend. An orchestration script on the master node will SSH into all worker nodes and execute `torchrun` pointing to the rendezvous endpoint on the master. PyTorch will handle autodiscovery and dynamic rank assignment.

### the agent's Discretion
- Code & Data Synchronization: Decide whether to use a shared NFS drive or `rsync` before launch.
- Launch Mechanism specifics: Shell script structure and SSH key assumptions.

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

### Reusable Assets
- None detected specifically for distributed topology setup.

### Established Patterns
- Existing backend uses FastAPI and Python scripts (`data_preparation.py`, `fine_tune.py`). The orchestration script will need to wrap `fine_tune.py` calls with `torchrun`.

### Integration Points
- The `gloo` process group initialization will replace or mock `accelerate`/GPU-specific init inside `fine_tune.py`.
</code_context>

<specifics>
## Specific Ideas

- The master node acts as the rendezvous point. The master node should also participate in training (it will spawn its own `torchrun` local worker).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 1-Architecture & Networking*
*Context gathered: 2026-09-11*
