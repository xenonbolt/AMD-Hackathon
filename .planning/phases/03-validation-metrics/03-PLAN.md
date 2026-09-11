# Phase 3: Validation & Metrics - Implementation Plan

## Proposed Changes

1. **Update requirements.txt**: Add `psutil`.
2. **Update fine_tune.py**: Add `HardwareMetricsCallback` using `psutil` to log CPU/RAM on `on_log`.
3. **Update train_fix_model.py**: Add `HardwareMetricsCallback` using `psutil`.
4. **Update inference_engine.py**: Remove `BitsAndBytesConfig` and `load_in_4bit`.
