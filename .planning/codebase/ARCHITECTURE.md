---
last_mapped_commit: none
---
# ARCHITECTURE.md

**Analysis Date:** 2026-09-11

## System Design and Patterns
The system is divided into two primary environments: a Python-based ML pipeline/backend and a React frontend.

### 1. ML Pipeline and Backend
- **Data Preparation (`data_preparation.py`):** Loads JSONL, formats instructions, tokenizes.
- **Fine-Tuning (`fine_tune.py`, `train_fix_model.py`):** Uses QLoRA with `bitsandbytes` and `peft` to fine-tune base LLMs for vulnerability detection and remediation.
- **Inference Engines (`inference_engine.py`, `fix_engine.py`):** Loads quantized models with LoRA adapters and runs deterministic inference.
- **Static Scanner (`scanner.py`):** Recursively parses Java files, chunks them, and orchestrates inference to generate JSON vulnerability reports.
- **API Backend (`api.py`):** Exposes ML capabilities via an HTTP API (FastAPI).

### 2. Frontend (`frontend/`)
- A React SPA built with Vite and TailwindCSS.
- Connects to the backend via an API endpoint.

## Data Flow
1. **Training:** `Dataset/*.jsonl` -> `data_preparation.py` -> `fine_tune.py` -> QLoRA Adapters.
2. **Scanning:** `scanner.py` -> `inference_engine.py` -> `report.json`.
3. **Remediation:** `fix_engine.py` -> Generate code fixes.

<!-- refreshed: 2026-09-11 -->
