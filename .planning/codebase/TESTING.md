---
last_mapped_commit: none
---
# TESTING.md

**Analysis Date:** 2026-09-11

## Test Structure and Practices
- **Backend Testing:** A `test_api.py` exists, suggesting some unit/integration testing for the API. `verify_scanner.py` also exists. No clear overarching test framework (like pytest) is explicitly defined in `requirements.txt`.
- **Frontend Testing:** No explicit test framework (Jest/Vitest) is defined in `package.json`.
- **Validation:** Uses `sample.java` and `not-vulnerable-sample-folder` for manual or automated verification of the scanner.

<!-- refreshed: 2026-09-11 -->
