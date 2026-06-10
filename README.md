# Java Vulnerability Detection — QLoRA Fine-Tuning Pipeline

A production-ready, end-to-end pipeline for fine-tuning a code LLM using QLoRA to detect Java security vulnerabilities and scan local codebases.

---

## Project Structure

```
AMD/
├── Dataset/
│   └── train_classifier_final.jsonl   ← Raw training dataset (705 records)
├── outputs/
│   └── vuln-lora/                     ← Saved LoRA adapter weights (post-training)
├── data_preparation.py                ← Dataset parsing, formatting & tokenisation
├── fine_tune.py                       ← QLoRA fine-tuning orchestrator
├── inference_engine.py                ← Singleton inference engine
├── scanner.py                         ← Java codebase static analysis scanner
├── requirements.txt                   ← Python dependencies
└── README.md
```

---

## Architecture Overview

```
                          ┌─────────────────────────────────┐
                          │       data_preparation.py        │
                          │  Load JSONL → Format Prompt →   │
                          │  Tokenise → Mask Prompt Labels  │
                          └────────────────┬────────────────┘
                                           │ HuggingFace DatasetDict
                          ┌────────────────▼────────────────┐
                          │          fine_tune.py            │
                          │  Base Model (4-bit NF4)         │
                          │  + LoRA Adapters (r=16)         │
                          │  + CosineAnnealing LR           │
                          │  → Save adapter weights         │
                          └────────────────┬────────────────┘
                                           │ ./outputs/vuln-lora/
                          ┌────────────────▼────────────────┐
                          │       inference_engine.py        │
                          │  Singleton: Base + LoRA Adapter │
                          │  analyze_snippet(code) → JSON   │
                          └────────────────┬────────────────┘
                                           │
                          ┌────────────────▼────────────────┐
                          │           scanner.py             │
                          │  Walk .java files → Extract     │
                          │  Methods → Deduplicate → Infer  │
                          │  → scan_report.json             │
                          └─────────────────────────────────┘
```

---

## Setup

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python3.10 -m venv .venv && source .venv/bin/activate

# Install PyTorch with CUDA (adjust cu121 for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the rest
pip install -r requirements.txt
```

### 2. Verify GPU Access

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Usage

### Step 1 — Data Preparation (smoke-test)

Validates dataset parsing and tokenisation without launching training.

```bash
python data_preparation.py \
    --dataset Dataset/train_classifier_final.jsonl \
    --tokeniser bigcode/starcoder2-3b \
    --max-seq-length 2048
```

### Step 2 — Fine-Tuning

Trains the model with QLoRA and saves the adapter to `./outputs/vuln-lora`.

```bash
python fine_tune.py \
    --base-model bigcode/starcoder2-3b \
    --dataset Dataset/train_classifier_final.jsonl \
    --output-dir ./outputs/vuln-lora \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --max-seq-length 2048
```

**Alternative base model (larger, more capable):**

```bash
python fine_tune.py --base-model codellama/CodeLlama-7b-hf ...
```

### Step 3 — Inference Engine (smoke-test)

Tests the loaded adapter with a built-in SQL injection snippet.

```bash
python inference_engine.py \
    --adapter ./outputs/vuln-lora \
    --base-model bigcode/starcoder2-3b
```

Or test with a specific Java file:

```bash
python inference_engine.py \
    --adapter ./outputs/vuln-lora \
    --snippet path/to/VulnerableClass.java
```

### Step 4 — Scan a Codebase

Recursively scans all `.java` files in a directory and outputs a structured JSON report.

```bash
python scanner.py \
    --target ./my-java-project \
    --adapter ./outputs/vuln-lora \
    --base-model bigcode/starcoder2-3b \
    --output scan_report.json \
    --log-level INFO
```

**Test with only the first 5 files:**

```bash
python scanner.py --target ./my-java-project --max-files 5 ...
```

---

## Module Reference

### `data_preparation.py`

| Function | Description |
|---|---|
| `build_dataset(path, tokeniser, ...)` | End-to-end factory → `DatasetDict` |
| `load_raw_records(path)` | Parse JSONL → list of `VulnRecord` |
| `format_prompt(record)` | Apply instruction template |
| `tokenise_and_label(text, tok, cfg)` | Tokenise + mask prompt tokens |

### `fine_tune.py`

| Function | Description |
|---|---|
| `run_training(cfg)` | Orchestrate full training pipeline |
| `load_base_model(id, bnb_cfg, ...)` | Load 4-bit quantised model |
| `apply_lora(model, params)` | Wrap with PEFT LoRA adapters |
| `build_training_arguments(cfg)` | Construct `TrainingArguments` |

### `inference_engine.py`

| API | Description |
|---|---|
| `InferenceEngine.get_instance(...)` | Singleton factory (thread-safe) |
| `engine.analyze_snippet(code: str)` | Returns parsed vulnerability dict |
| `analyze_snippet(code, adapter, model)` | Module-level convenience wrapper |

### `scanner.py`

| Function | Description |
|---|---|
| `scan_directory(target, adapter, ...)` | Full scan → `ScanReport` |
| `discover_java_files(root)` | Recursive `.java` file discovery |
| `chunk_java_file(path)` | Method extraction + sliding window fallback |
| `extract_method_chunks(source)` | Regex + brace-balanced extraction |
| `deduplicate_chunks(chunks, seen)` | Content-hash deduplication |

---

## Scan Report Format

```json
{
  "scan_metadata": {
    "target_directory": "/abs/path/to/project",
    "scanned_at": "2026-06-10T17:30:00+00:00",
    "total_files": 42,
    "total_chunks": 187,
    "total_vulnerabilities": 9,
    "engine_adapter": "./outputs/vuln-lora",
    "base_model": "bigcode/starcoder2-3b"
  },
  "findings": [
    {
      "file_path": "src/UserService.java",
      "chunk_index": 0,
      "chunk_type": "method",
      "method_name": "getUserById",
      "line_start": 24,
      "line_end": 51,
      "vulnerabilities": [
        {
          "cwe_id": "CWE-89",
          "cwe_name": "SQL Injection",
          "severity": "critical",
          "confidence": 0.97,
          "location": { "start_line": 24, "end_line": 51, "function": "getUserById" },
          "description": "Unsanitised user input concatenated directly into SQL query.",
          "impact": "Allows arbitrary SQL execution; potential data exfiltration or deletion.",
          "recommendation": "Use PreparedStatement with parameterised queries."
        }
      ]
    }
  ]
}
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU VRAM | 10 GB (starcoder2-3b) | 24 GB (CodeLlama-7b) |
| System RAM | 16 GB | 32 GB |
| Storage | 20 GB | 50 GB |
| CUDA | 11.8 | 12.1+ |

> **Tip:** Enable gradient checkpointing (`--gradient-checkpointing`, on by default) to reduce VRAM usage at the cost of ~20% slower training.

---

## Supported Vulnerability Types

The training dataset covers 30+ CWE categories including:

- **CWE-89** — SQL Injection
- **CWE-79** — Cross-Site Scripting (XSS)  
- **CWE-90** — LDAP Injection
- **CWE-643** — XPath Injection
- **CWE-259/522** — Hard-coded Credentials
- **CWE-276/378** — Insecure File Permissions
- **CWE-400** — Resource Exhaustion
- **CWE-36** — Absolute Path Traversal
- And many more…

---

## License

MIT License — See `LICENSE` for details.
