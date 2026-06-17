# Vulnerability Detection System — Fix & Upgrade

## Overview

This document covers all changes made to fix the vulnerability scanner's output quality and build a production-grade training dataset.

---

## Problem Statement

The fine-tuned DeepSeek-Coder 6.7B model for Java vulnerability detection had two critical issues:

1. **Duplicate findings** — The inference engine produced multiple entries for the same CWE at the same line number
2. **Poor training data quality** — 100% templated descriptions, 100% identical impacts/recommendations, only 12 CWE types, zero multi-vulnerability examples, and all code from Juliet Test Suite only

---

## Part A: Deduplication Fix in Inference Engine

### Changes to `inference_engine.py`

#### 1. Deduplication Logic (Lines 804–831)
Added post-validation deduplication that groups findings by `(cwe_id, line)` and keeps only the entry with the longest (most specific) description:

```python
# --- Deduplication by (cwe_id, line) ---
seen: Dict[tuple, int] = {}
deduped: List[Dict[str, Any]] = []
for vuln in validated:
    cwe = vuln.get("cwe_id", "")
    line = vuln.get("location", {}).get("line", 0)
    key = (cwe, line)
    if key in seen:
        existing_idx = seen[key]
        existing = deduped[existing_idx]
        if len(vuln.get("description", "")) > len(existing.get("description", "")):
            deduped[existing_idx] = vuln  # Replace with longer description
    else:
        seen[key] = len(deduped)
        deduped.append(vuln)
```

#### 2. Prompt Template Update (Lines 24–51)
Replaced the minimal prompt with a structured template that includes an explicit JSON example format, enforcing English-only output, and requesting specific fields (cwe_id, cwe_name, severity, confidence, location, description, impact, recommendation).

---

## Part B: Production Data Generation Pipeline

### New File: `generate_production_data.py`

A comprehensive data generation pipeline (~66KB) that produces training data from three sources:

#### Source 1: Enhanced Juliet Test Suite
- Rewrites all 352 positive examples with **code-specific descriptions** that reference actual method names, API calls, and line numbers
- Eliminates templated boilerplate entirely
- Produces unique, contextual impact statements and recommendations per vulnerability type

#### Source 2: Vul4J Real-World CVEs
- Processes 48 real CVEs from the Vul4J dataset (reproduced real-world Java vulnerabilities)
- Maps each CVE to its CWE classification
- Generates descriptions referencing actual vulnerable code patterns from real projects (Apache Commons, Spring Framework, etc.)
- Adds 21 new CWE types not covered by Juliet

#### Source 3: Synthetic Examples
- **Multi-vulnerability files** (6 examples): Code with 2-3 different vulnerabilities in the same file
- **Hard negatives** (8 examples): Safe code that looks suspicious but contains no actual vulnerabilities (important for reducing false positives)

### Output: `Dataset/train_production.jsonl`

---

## Validation Results

### Comparison Script: `compare_datasets.py`

| Metric | Old Dataset | New Dataset | Improvement |
|---|---|---|---|
| Total records | 704 | 767 | +9% |
| Positive examples | 352 | 407 | +16% |
| Negative examples | 352 | 360 | +2% |
| Total vulnerability entries | 352 | 415 | +18% |
| **Multi-vuln files** | **0** | **6** | ✅ New capability |
| **Unique CWE types** | **12** | **33** | **+175%** |
| **Unique descriptions** | **27 / 352** | **247 / 415** | **+815%** |
| Templated descriptions | 352 (100%) | 0 (0%) | ✅ **Eliminated** |
| Templated impacts | 352 (100%) | 0 (0%) | ✅ **Eliminated** |
| Templated recommendations | 352 (100%) | 0 (0%) | ✅ **Eliminated** |

### Sample Description Quality

**Before (old):**
> *"The function `bad` contains a vulnerability associated with LDAP Injection (CWE-90)."*

**After (new):**
> *"The method `bad()` contains a dangerous call `response.getWriter().println(...)` at line 75 that processes data without adequate security controls..."*

---

## Usage

### Retraining

```bash
python fine_tune.py \
  --dataset Dataset/train_production.jsonl \
  --model_id deepseek-ai/deepseek-coder-6.7b-instruct \
  --output_dir adapters
```

### Running Inference

```bash
python inference_engine.py \
  --model_id deepseek-ai/deepseek-coder-6.7b-instruct \
  --adapter_path adapters \
  --target_path sample.java
```

---

## Files

| File | Description |
|---|---|
| `inference_engine.py` | Core inference engine with dedup logic, weighted validation, prompt template |
| `generate_production_data.py` | Production data pipeline (Vul4J + Juliet + synthetic) |
| `fine_tune.py` | Model fine-tuning script |
| `compare_datasets.py` | Quality comparison between old and new datasets |
| `analyze_data_quality.py` | Data quality analysis for a single dataset |
| `api.py` | REST API for the vulnerability scanner |
| `scanner.py` | File/directory scanning orchestrator |
| `Dataset/train_production.jsonl` | Production training dataset (767 examples, 33 CWEs) |
| `Dataset/train_classifier_precise_lines.jsonl` | Original training dataset (704 examples, 12 CWEs) |