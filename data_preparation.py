import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Any, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("data_preparation")

# ── Prompt templates (detection-only) ──────────────────────────────────────────
# The model is trained to DETECT and CLASSIFY — it does NOT produce a code fix.
# Response format: vulnerability label, CWE, CVE hint, and severity.
PROMPT_INSTRUCTION = (
    "### Instruction: Analyze the following Java code snippet and determine whether "
    "it contains a security vulnerability. If a vulnerability is detected, identify "
    "its type, CWE classification, CVE reference (if known), and severity.\n"
    "Output format:\n"
    "  VERDICT: <VULNERABLE | SAFE>\n"
    "  VULNERABILITY_TYPE: <short name, e.g. SQL Injection>\n"
    "  CWE_ID: <e.g. CWE-89>\n"
    "  CVE_REFERENCE: <e.g. CVE-2021-12345 or UNKNOWN>\n"
    "  SEVERITY: <CRITICAL | HIGH | MEDIUM | LOW | UNKNOWN>\n"
    "  DESCRIPTION: <one-sentence explanation>\n\n"
)
INPUT_TEMPLATE = "### Input:\n{vuln_code}\n\n"
RESPONSE_TEMPLATE = (
    "### Response:\n"
    "VERDICT: VULNERABLE\n"
    "VULNERABILITY_TYPE: {cwe_name}\n"
    "CWE_ID: {cwe_id}\n"
    "CVE_REFERENCE: {cve_id}\n"
    "SEVERITY: {severity}\n"
    "DESCRIPTION: {description}"
)


class JavaVulnerabilityDataset(Dataset):
    """
    Detection-only PyTorch Dataset.

    Loads Java vulnerability entries from a JSONL file produced by build_dataset.py
    and formats them into instruction-following prompts for classification training.

    Expected JSONL keys per record:
        CVE_ID          – CVE identifier (required)
        CWE_ID          – CWE identifier string, e.g. 'CWE-89'
        CWE_Number      – numeric portion of CWE, e.g. '89'
        Vulnerable_code – Java source snippet that is vulnerable (required)
        cwe_name        – human-readable CWE name / short description
        cvss_score      – CVSS base score (float or null)
        severity        – CRITICAL | HIGH | MEDIUM | LOW | UNKNOWN
        commit_message  – git commit message (optional context)
        repo_url        – source repository URL
        language        – should be 'java'

    The model is NOT trained to produce a code fix — it learns to classify
    the vulnerability type, CWE, CVE reference, and severity.
    """
    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        mask_prompt: bool = True
    ) -> None:
        """
        Args:
            jsonl_path: Path to the JSONL dataset (from build_dataset.py).
            tokenizer: Pretrained tokenizer from Hugging Face.
            max_length: Maximum token sequence length for truncation.
            mask_prompt: If True, prompt tokens are masked (-100) so the loss
                         is computed only on the response (classification) tokens.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self.examples: List[Dict[str, str]] = []

        try:
            logger.info(f"Loading detection-only dataset from {jsonl_path}")
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)

                        # Support both old key (vuln_code) and new key (Vulnerable_code)
                        vuln_code = (
                            data.get("Vulnerable_code")
                            or data.get("vuln_code")
                            or ""
                        ).strip()

                        cve_id = str(data.get("CVE_ID", "UNKNOWN")).strip()

                        if not vuln_code or not cve_id or cve_id.lower() == "unknown":
                            logger.warning(
                                f"Skipping line {line_idx + 1}: "
                                f"Missing 'Vulnerable_code' or 'CVE_ID'"
                            )
                            continue

                        self.examples.append({
                            "vuln_code": vuln_code,
                            "cve_id": cve_id,
                            "cwe_id": str(data.get("CWE_ID", "UNKNOWN")).strip(),
                            "cwe_number": str(data.get("CWE_Number", "")).strip(),
                            "cwe_name": str(data.get("cwe_name", "Security Vulnerability")).strip(),
                            "severity": str(data.get("severity", "UNKNOWN")).strip().upper() or "UNKNOWN",
                            "cvss_score": data.get("cvss_score"),
                            "commit_message": str(data.get("commit_message", "")).strip(),
                        })

                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON on line {line_idx + 1}: {e}")

            logger.info(f"Loaded {len(self.examples)} detection records successfully.")
        except Exception as e:
            logger.error(f"Failed to read dataset from {jsonl_path}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Prepares a single detection-only training sample.

        The prompt instructs the model to classify the vulnerability.
        The response is a structured label block (VERDICT, CWE_ID, CVE_REFERENCE, etc.).
        No code fix is included in the response.
        """
        example = self.examples[idx]
        vuln_code = example["vuln_code"]

        # Build a short, informative description from available metadata
        cwe_name = example.get("cwe_name") or "Security Vulnerability"
        cvss_score = example.get("cvss_score")
        description_parts = [f"{cwe_name} vulnerability detected in Java code."]
        if cvss_score is not None:
            description_parts.append(f"CVSS Score: {cvss_score}.")
        commit_msg = example.get("commit_message", "")
        if commit_msg:
            # First sentence of commit message as extra context (max 120 chars)
            first_sentence = commit_msg.split(".")[0][:120]
            description_parts.append(first_sentence)
        description = " ".join(description_parts)

        # 1. Format the instruction/input prompt
        prompt_text = (
            f"{PROMPT_INSTRUCTION}"
            f"{INPUT_TEMPLATE.format(vuln_code=vuln_code)}"
            f"### Response:\n"
        )
        # 2. Format the classification response (detection label — no fix)
        response_text = RESPONSE_TEMPLATE.format(
            cwe_name=example.get("cwe_name", "Security Vulnerability"),
            cwe_id=example.get("cwe_id", "UNKNOWN"),
            cve_id=example.get("cve_id", "UNKNOWN"),
            severity=example.get("severity", "UNKNOWN"),
            description=description,
        )

        try:
            # Tokenize the prompt and response separately
            prompt_enc = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=False
            )
            response_enc = self.tokenizer(
                response_text,
                add_special_tokens=False,
                truncation=False
            )

            # Determine BOS token inclusion
            bos_token_id = self.tokenizer.bos_token_id
            bos_tokens = [bos_token_id] if bos_token_id is not None else []

            # Determine EOS token inclusion
            eos_token_id = self.tokenizer.eos_token_id
            eos_tokens = [eos_token_id] if eos_token_id is not None else []

            prompt_ids = bos_tokens + prompt_enc["input_ids"]
            response_ids = response_enc["input_ids"] + eos_tokens

            # Truncation logic if standard sequence length exceeded
            total_len = len(prompt_ids) + len(response_ids)
            if total_len > self.max_length:
                # Truncate response tokens to preserve the prompt input as much as possible
                excess = total_len - self.max_length
                if excess < len(response_ids):
                    response_ids = response_ids[:-excess]
                else:
                    # If prompt is extremely long, truncate prompt itself
                    prompt_ids = prompt_ids[:self.max_length - 1]
                    response_ids = eos_tokens  # Keep at least EOS if response is completely truncated

            # Combine input ids and establish attention mask
            input_ids = prompt_ids + response_ids
            attention_mask = [1] * len(input_ids)

            # Labels for Causal LM training:
            # -100 instructs PyTorch CrossEntropyLoss to ignore these targets.
            if self.mask_prompt:
                labels = [-100] * len(prompt_ids) + response_ids
            else:
                labels = input_ids.copy()

            # Dynamic padding to max_length will be handled by a PyTorch DataCollator.
            # But we cast to tensors here for PyTorch loader compatibility.
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
        except Exception as e:
            logger.error(f"Error during tokenization of example index {idx}: {e}")
            raise


class CausalLMDataCollator:
    """
    Custom collator to dynamically pad batch inputs to the max length present in the batch.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        # Set pad token if not defined
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Find maximum length in this batch
        lengths = [x["input_ids"].size(0) for x in batch]
        max_len = max(lengths)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must have a valid pad_token_id.")

        # Determine padding side (causal LMs usually prefer right-padding for training, 
        # but left-padding is mandatory for batch inference. HuggingFace Trainer supports standard right-pad).
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]

            diff = max_len - input_ids.size(0)
            if diff > 0:
                pad_ids = torch.full((diff,), pad_token_id, dtype=torch.long)
                pad_mask = torch.zeros((diff,), dtype=torch.long)
                pad_labels = torch.full((diff,), -100, dtype=torch.long)

                if padding_side == "right":
                    new_input_ids = torch.cat([input_ids, pad_ids])
                    new_attention_mask = torch.cat([attention_mask, pad_mask])
                    new_labels = torch.cat([labels, pad_labels])
                else:
                    new_input_ids = torch.cat([pad_ids, input_ids])
                    new_attention_mask = torch.cat([pad_mask, attention_mask])
                    new_labels = torch.cat([pad_labels, labels])
            else:
                new_input_ids = input_ids
                new_attention_mask = attention_mask
                new_labels = labels

            batch_input_ids.append(new_input_ids)
            batch_attention_mask.append(new_attention_mask)
            batch_labels.append(new_labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels)
        }


def generate_mock_jsonl(path: Path) -> None:
    """
    Generates a sample JSONL file with mock detection-only Java vulnerability
    records matching the schema produced by build_dataset.py.
    Used for verification and testing of data_preparation.py.
    """
    mock_data = [
        {
            "CVE_ID": "CVE-2021-99001",
            "CWE_ID": "CWE-89",
            "CWE_Number": "89",
            "Vulnerable_code": (
                'public void process(String input) throws Exception {\n'
                '    Connection conn = DriverManager.getConnection(DB_URL);\n'
                '    Statement stmt = conn.createStatement();\n'
                '    ResultSet rs = stmt.executeQuery("SELECT * FROM users WHERE username = \'" + input + "\'");\n'
                '}'
            ),
            "cwe_name": "SQL Injection",
            "cvss_score": 9.8,
            "severity": "CRITICAL",
            "commit_message": "Fix SQL injection by using PreparedStatement.",
            "repo_url": "https://github.com/example/mock-repo",
            "language": "java",
        },
        {
            "CVE_ID": "CVE-2022-99002",
            "CWE_ID": "CWE-22",
            "CWE_Number": "22",
            "Vulnerable_code": (
                'public void handle(HttpServletRequest req) {\n'
                '    String path = req.getParameter("path");\n'
                '    File file = new File("/var/uploads/" + path);\n'
                '    FileInputStream fis = new FileInputStream(file);\n'
                '}'
            ),
            "cwe_name": "Path Traversal",
            "cvss_score": 7.5,
            "severity": "HIGH",
            "commit_message": "Sanitize file path to prevent directory traversal.",
            "repo_url": "https://github.com/example/mock-repo",
            "language": "java",
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for entry in mock_data:
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Generated mock detection-only JSONL dataset at: {path}")


if __name__ == "__main__":
    # Sanity check validation run
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Test and Verify Data Preparation")
    parser.add_argument("--test_file", type=str, default="test_dataset.jsonl", help="Path to JSONL output/test file")
    parser.add_argument("--tokenizer_name", type=str, default="bigcode/starcoder2-3b", help="Hugging Face tokenizer ID")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    generate_mock_jsonl(test_path)

    try:
        logger.info(f"Loading local/remote tokenizer: {args.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        
        # Instantiate detection-only dataset
        dataset = JavaVulnerabilityDataset(test_path, tokenizer, max_length=512)
        collator = CausalLMDataCollator(tokenizer)

        # Test loading first element
        sample = dataset[0]
        logger.info("Successfully processed first detection dataset sample.")
        logger.info(f"Input IDs shape : {sample['input_ids'].shape}")
        logger.info(f"Labels shape    : {sample['labels'].shape}")

        # Decode to sanity-check the formatted prompt + response
        decoded = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        logger.info(f"Sample preview  :\n{decoded[:600]}")

        # Test batching via data collator
        batch = collator([dataset[0], dataset[1]])
        logger.info("Successfully batched detection dataset samples.")
        logger.info(f"Batch Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Batch Labels shape   : {batch['labels'].shape}")

    except Exception as err:
        logger.error(f"Verification execution failed: {err}")
    finally:
        if test_path.exists():
            test_path.unlink()
            logger.info("Cleaned up temporary test file.")
