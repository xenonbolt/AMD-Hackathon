# Phase 3: Validation & Metrics - Discussion Log

**Date:** 2026-09-11

## Metrics Logging
**Presented:**
- Add a PyTorch callback using `psutil` to log metrics periodically directly into the Hugging Face Trainer logs.
- Create a separate bash daemon that runs `top`/`htop` or `sar` in the background and writes to a log file.

**Selected:**
- Add a PyTorch callback using `psutil` to log metrics periodically directly into the Hugging Face Trainer logs. (Recommended)

## Validation Strategy
**Presented:**
- We will load the single consolidated checkpoint (as configured in Phase 2) using `AutoModelForCausalLM` and verify it runs on `inference_engine.py` (CPU).
- Run `verify_scanner.py` with the FSDP-generated adapters to ensure end-to-end scanner logic works.

**Selected:**
- We will load the single consolidated checkpoint (as configured in Phase 2) using `AutoModelForCausalLM` and verify it runs on `inference_engine.py` (CPU). (Recommended)

---
*This log is for reference only and is not consumed by planning agents.*
