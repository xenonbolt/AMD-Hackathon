# Phase 1: Architecture & Networking - Discussion Log

**Date:** 2026-09-11

## Network Topology & Discovery
**Presented:**
- Hardcoded IPs
- Environment Variables (MASTER_ADDR/MASTER_PORT)
- Service Registry

**Selected:**
- Environment Variables / torchrun with C10d rendezvous

**Notes:**
- User asked: "Is it possible to run from master to all the other nodes (includding master) and have autodiscovery of connected servers?"
- We agreed to use an orchestration script on the master that uses SSH to launch `torchrun` on all workers with the master's IP as the rendezvous endpoint.

---
*This log is for reference only and is not consumed by planning agents.*
