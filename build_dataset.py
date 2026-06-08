import json
import logging
import os
from datasets import load_dataset

# Setup clean tracking logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def generate_java_vulnerability_dataset(output_path: str = "Dataset/java_vuln_dataset.jsonl"):
    """
    Downloads a flat version of CVEfixes containing complete code blocks,
    isolates Java pairs, and normalizes them into JSONL for LLM training.
    """
    logger.info("Initializing download for hitoshura25/cvefixes from Hub...")
    
    try:
        # Load the fully public, flat dataset configuration
        dataset = load_dataset("hitoshura25/cvefixes", split="train")
        logger.info(f"Successfully loaded {len(dataset)} records from master index.")
    except Exception as e:
        logger.error(f"Failed to access the dataset repo: {str(e)}")
        return

    # Create directory structure if missing
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_count = 0
    logger.info("Filtering records for Java source code changes...")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for row in dataset:
            # Match language metadata (handles case differences like 'Java' or 'java')
            language = str(row.get("language", "")).lower()
            if "java" not in language:
                continue
                
            # Extract explicit source string mappings
            vuln_code = row.get("vulnerable_code", "")
            fixed_code = row.get("fixed_code", "")
            cwe_info = row.get("cwe_name", "") or row.get("cwe_id", "Java Vulnerability")

            # Validate code pairs are populated and contain actual changes
            if not vuln_code or not fixed_code or str(vuln_code).strip() == str(fixed_code).strip():
                continue

            # Assemble structure matching the fine-tune input profile
            dataset_entry = {
                "vuln_code": str(vuln_code).strip(),
                "description": str(cwe_info).strip(),
                "fixed_code": str(fixed_code).strip()
            }
            
            f.write(json.dumps(dataset_entry, ensure_ascii=False) + "\n")
            saved_count += 1

    logger.info(f"Data process complete! Saved {saved_count} valid Java pairs to '{output_path}'.")

if __name__ == "__main__":
    generate_java_vulnerability_dataset()