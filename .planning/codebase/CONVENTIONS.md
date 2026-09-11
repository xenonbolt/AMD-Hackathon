---
last_mapped_commit: none
---
# CONVENTIONS.md

**Analysis Date:** 2026-09-11

## Code Style and Patterns
- **Python:** Modular scripts for different stages of the ML lifecycle (data prep, train, infer, scan). Uses FastAPI for the API layer.
- **Frontend:** React with TypeScript (`.tsx`). Styling with TailwindCSS.
- **Data Formats:** Heavy reliance on `.jsonl` for datasets and `.json` for vulnerability reports.

## ML Patterns
- **LoRA / PEFT:** Uses standard PEFT patterns for parameter-efficient fine-tuning.
- **Quantization:** 4-bit quantization via `bitsandbytes` for memory efficiency.
- **Deterministic Inference:** Uses `temperature = 0.0` for consistent evaluation output.

<!-- refreshed: 2026-09-11 -->
