import json
import logging
import os
import re
from datasets import load_dataset

# Setup clean tracking logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_cwe_number(cwe_id: str) -> str:
    """
    Extracts the numeric portion from a CWE ID string.
    e.g., 'CWE-89' -> '89', 'CWE-22' -> '22'
    Returns empty string if not parseable.
    """
    if not cwe_id:
        return ""
    match = re.search(r'\d+', str(cwe_id))
    return match.group(0) if match else ""


def generate_java_vulnerability_dataset(output_path: str = "Dataset/java_vuln_dataset.jsonl"):
    """
    Downloads the hitoshura25/cvefixes flat dataset from Hugging Face Hub,
    filters for Java records, and serializes enriched detection-only entries
    to a JSONL file.

    This is a DETECTION-ONLY dataset — no fixed_code is included.
    Each entry captures:
      - CVE_ID         : Common Vulnerabilities and Exposures identifier
      - CWE_ID         : Full CWE ID string (e.g., 'CWE-89')
      - CWE_Number     : Numeric portion only (e.g., '89')
      - Vulnerable_code: The vulnerable Java code snippet
      - cwe_name       : Human-readable CWE name / description
      - cvss_score     : CVSS severity score (if available)
      - severity       : Qualitative severity label (if available)
      - commit_message : Git commit message describing the fix context
      - repo_url       : Source repository URL
      - language       : Confirmed programming language
    """
    logger.info("Initializing download for hitoshura25/cvefixes from Hub...")

    try:
        dataset = load_dataset("hitoshura25/cvefixes", split="train")
        logger.info(f"Successfully loaded {len(dataset)} records. Inspecting available columns...")
        logger.info(f"Columns: {dataset.column_names}")
    except Exception as e:
        logger.error(f"Failed to access the dataset repo: {str(e)}")
        return

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_count = 0
    skipped_no_code = 0
    skipped_no_cve = 0
    logger.info("Filtering records for Java source code with CVE/CWE metadata...")

    with open(output_path, "w", encoding="utf-8") as f:
        for row in dataset:
            # --- Language filter ---
            language = str(row.get("language", "")).lower()
            if "java" not in language:
                continue

            # --- Vulnerable code (required) ---
            vuln_code = row.get("vulnerable_code", "") or row.get("code_before", "") or row.get("code", "")
            if not vuln_code or not str(vuln_code).strip():
                skipped_no_code += 1
                continue

            # --- CVE identifier (required for a useful detection dataset) ---
            cve_id = (
                row.get("cve_id", "")
                or row.get("CVE_ID", "")
                or row.get("cve", "")
            )
            if not cve_id or str(cve_id).strip().lower() in ("", "none", "nan"):
                skipped_no_cve += 1
                continue

            # --- CWE fields ---
            raw_cwe_id = (
                row.get("cwe_id", "")
                or row.get("CWE_ID", "")
                or row.get("cwe", "")
                or ""
            )
            cwe_id = str(raw_cwe_id).strip() if raw_cwe_id else ""
            cwe_number = extract_cwe_number(cwe_id)

            # --- Optional enrichment fields ---
            cwe_name = str(row.get("cwe_name", "") or row.get("cwe_description", "") or "").strip()
            cvss_score = row.get("cvss_score", None) or row.get("cvss_v3_score", None) or row.get("cvss_v2_score", None)
            severity = str(row.get("severity", "") or row.get("cvss_severity", "") or "").strip()
            commit_msg = str(row.get("commit_message", "") or row.get("msg", "") or "").strip()
            repo_url = str(row.get("repo_url", "") or row.get("url", "") or "").strip()

            # --- Assemble detection-only record ---
            entry = {
                "CVE_ID": str(cve_id).strip(),
                "CWE_ID": cwe_id,
                "CWE_Number": cwe_number,
                "Vulnerable_code": str(vuln_code).strip(),
                "cwe_name": cwe_name,
                "cvss_score": float(cvss_score) if cvss_score is not None else None,
                "severity": severity,
                "commit_message": commit_msg,
                "repo_url": repo_url,
                "language": "java",
            }

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            saved_count += 1

    logger.info(
        f"Dataset build complete! "
        f"Saved: {saved_count} | "
        f"Skipped (no code): {skipped_no_code} | "
        f"Skipped (no CVE): {skipped_no_cve}"
    )
    logger.info(f"Output written to: '{output_path}'")


if __name__ == "__main__":
    generate_java_vulnerability_dataset()