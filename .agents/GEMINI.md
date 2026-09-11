<!-- GSD:project-start source:PROJECT.md -->

## Project

**CPU-Distributed Model Fine-Tuning Swarm**

A distributed CPU-only machine learning training swarm using PyTorch Native FSDP over Gloo. It adapts an existing single-node GPU-based ML pipeline for vulnerability detection and remediation to run distributed fine-tuning across 4 Ubuntu servers (1 master, 3 nodes) using only CPU and RAM.

**Core Value:** Scale model fine-tuning horizontally using CPU and RAM resources across multiple commodity Ubuntu nodes without requiring GPUs.

### Constraints

- **Hardware**: Must use CPU and RAM only (no GPUs) — The 4 Ubuntu servers do not have GPU acceleration available for this workload.
- **Dependencies**: Will likely need to remove or replace `bitsandbytes` and standard QLoRA implementations since they are inherently CUDA-dependent.
- **Networking**: Must rely on the `gloo` backend since `nccl` is GPU-only.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Tech Stack

- **Languages:** Python (Backend), TypeScript/JavaScript (Frontend - React), Java (Target language for vulnerabilities)
- **Backend Framework:** FastAPI (implied from `api.py` and `requirements.txt`)
- **Frontend Framework:** React 19, Vite, TailwindCSS 4
- **Machine Learning / AI:** PyTorch (`torch`), Hugging Face `transformers`, `peft` (QLoRA), `bitsandbytes`, `datasets`, `accelerate`
- **Other Dependencies:** `sentencepiece`, `protobuf`, `deep-translator`, `langdetect` (Backend); `@google/genai`, `lucide-react`, `motion` (Frontend)

## Configuration Files

- `requirements.txt`: Python package dependencies.
- `frontend/package.json`: Node dependencies and scripts.
- `frontend/vite.config.ts`: Vite bundler configuration.
- `frontend/tsconfig.json`: TypeScript configuration.

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Code Style and Patterns

- **Python:** Modular scripts for different stages of the ML lifecycle (data prep, train, infer, scan). Uses FastAPI for the API layer.
- **Frontend:** React with TypeScript (`.tsx`). Styling with TailwindCSS.
- **Data Formats:** Heavy reliance on `.jsonl` for datasets and `.json` for vulnerability reports.

## ML Patterns

- **LoRA / PEFT:** Uses standard PEFT patterns for parameter-efficient fine-tuning.
- **Quantization:** 4-bit quantization via `bitsandbytes` for memory efficiency.
- **Deterministic Inference:** Uses `temperature = 0.0` for consistent evaluation output.

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Design and Patterns

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

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.agents/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
