---
last_mapped_commit: none
---
# CONCERNS.md

**Analysis Date:** 2026-09-11

## Technical Debt and Issues
- **Dataset Imbalance:** The README notes a concern with dataset imbalance requiring `balance_dataset.py` to fix negative examples skew.
- **Missing Test Frameworks:** No explicit testing frameworks like pytest or vitest in the dependencies.
- **Hardcoded Paths:** Commands in README contain hardcoded paths (`./adapters`, `path/to/java/project`), which might indicate scripts are not fully robust to different execution contexts.

<!-- refreshed: 2026-09-11 -->
