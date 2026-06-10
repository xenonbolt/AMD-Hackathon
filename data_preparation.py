"""
data_preparation.py
====================
Loads and pre-processes the Java vulnerability JSONL dataset for QLoRA fine-tuning.

Dataset schema (each line):
  {"text": "<|instruction|>\n...\n\n<|input|>\n{java_code}\n\n<|response|>\n{json_output}"}

Responsibilities
----------------
1. Parse raw JSONL records and extract (input, response) pairs.
2. Re-format each record using an explicit instruction template that is
   consistent with fine_tune.py and inference_engine.py.
3. Tokenise the formatted prompts with proper truncation / padding and
   build causal-LM labels (prompt tokens are masked to -100 so the model
   only learns to generate the *response* portion).
4. Expose a `build_dataset()` factory that returns a HuggingFace
   `DatasetDict` ready for `Trainer`.

Author : Elite AI Engineering Team
Python : 3.10+
"""

from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data_preparation.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("data_preparation")

# ---------------------------------------------------------------------------
# Constants & instruction template
# ---------------------------------------------------------------------------
INSTRUCTION_TEXT: str = (
    "Analyze the following Java code for security vulnerabilities. "
    "If one or more vulnerabilities exist, identify each one and return "
    "a structured JSON report containing cwe_id, cwe_name, severity, "
    "confidence, location (start_line, end_line, function), description, "
    "impact, and recommendation. If no vulnerability is present, return "
    '{{"vulnerabilities": []}}.'
)

PROMPT_TEMPLATE: str = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{vuln_code}\n\n"
    "### Response:\n"
)

RESPONSE_TEMPLATE: str = "{fixed_code}"

# Tokens used in the raw dataset to delimit sections
_RAW_INSTRUCTION_TAG: str = "<|instruction|>"
_RAW_INPUT_TAG: str = "<|input|>"
_RAW_RESPONSE_TAG: str = "<|response|>"


# ---------------------------------------------------------------------------
# Data-class for a single parsed record
# ---------------------------------------------------------------------------
@dataclass
class VulnRecord:
    """Holds the de-structured fields from a single JSONL line."""

    raw_instruction: str
    java_code: str
    response_json: str


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _split_raw_text(text: str) -> tuple[str, str, str]:
    """
    Splits the raw `text` field from the JSONL dataset into its three
    constituent parts using the dataset's special delimiter tags.

    Returns
    -------
    (instruction_body, input_body, response_body)

    Raises
    ------
    ValueError
        If any expected delimiter is missing from the text.
    """
    # Validate all required tags exist
    for tag in (_RAW_INSTRUCTION_TAG, _RAW_INPUT_TAG, _RAW_RESPONSE_TAG):
        if tag not in text:
            raise ValueError(f"Missing delimiter tag '{tag}' in record text.")

    after_instruction: str = text.split(_RAW_INSTRUCTION_TAG, 1)[1]
    instruction_body, rest = after_instruction.split(_RAW_INPUT_TAG, 1)
    input_body, response_body = rest.split(_RAW_RESPONSE_TAG, 1)

    return instruction_body.strip(), input_body.strip(), response_body.strip()


def _validate_response_json(response_body: str) -> str:
    """
    Validates that the response portion is well-formed JSON and returns
    a canonically serialised (compact) version of it.

    Returns the original string unchanged if JSON is valid, or raises
    ``ValueError`` on malformed JSON so the caller can skip the record.
    """
    try:
        parsed: Any = json.loads(response_body)
        return json.dumps(parsed, separators=(",", ":"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Response is not valid JSON: {exc}") from exc


def parse_jsonl_record(raw_line: str, line_number: int) -> VulnRecord | None:
    """
    Parse a single raw JSONL line into a :class:`VulnRecord`.

    Parameters
    ----------
    raw_line : str
        A single line from the .jsonl file.
    line_number : int
        Line index used only for logging context.

    Returns
    -------
    VulnRecord | None
        Parsed record, or ``None`` if the line should be skipped.
    """
    raw_line = raw_line.strip()
    if not raw_line:
        return None

    try:
        obj: dict[str, Any] = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        logger.warning("Line %d – JSON decode error: %s", line_number, exc)
        return None

    text: str | None = obj.get("text")
    if not text:
        logger.warning("Line %d – missing 'text' field; skipping.", line_number)
        return None

    try:
        instr_body, input_body, response_body = _split_raw_text(text)
        canonical_response = _validate_response_json(response_body)
    except ValueError as exc:
        logger.warning("Line %d – parsing error: %s; skipping.", line_number, exc)
        return None

    return VulnRecord(
        raw_instruction=instr_body,
        java_code=input_body,
        response_json=canonical_response,
    )


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(record: VulnRecord) -> str:
    """
    Returns the *full* training text (prompt + response) for a record.

    The model sees the entire string during training; labels for the prompt
    portion are masked to -100 in :func:`tokenise_and_label`.
    """
    prompt_part = PROMPT_TEMPLATE.format(
        instruction=INSTRUCTION_TEXT,
        vuln_code=record.java_code,
    )
    response_part = RESPONSE_TEMPLATE.format(fixed_code=record.response_json)
    return prompt_part + response_part


def format_prompt_only(record: VulnRecord) -> str:
    """
    Returns *only* the prompt portion (no response) for inference use.
    Matches the prefix built in :func:`format_prompt`.
    """
    return PROMPT_TEMPLATE.format(
        instruction=INSTRUCTION_TEXT,
        vuln_code=record.java_code,
    )


# ---------------------------------------------------------------------------
# Tokenisation & label masking
# ---------------------------------------------------------------------------

@dataclass
class TokenisationConfig:
    """Hyper-parameters for tokenisation."""

    max_seq_length: int = 2048
    padding_side: str = "right"   # "right" for causal LMs
    truncation: bool = True


def tokenise_and_label(
    full_text: str,
    tokeniser: PreTrainedTokenizerBase,
    config: TokenisationConfig,
) -> dict[str, list[int]]:
    """
    Tokenises ``full_text`` and constructs causal-LM labels by masking the
    prompt tokens (set to -100).

    Strategy
    --------
    1. Tokenise the full text (prompt + response).
    2. Tokenise only the prompt portion to discover the boundary index.
    3. Set ``labels[:prompt_len] = -100`` so the loss is computed only on
       the response tokens.

    Returns
    -------
    dict with keys: ``input_ids``, ``attention_mask``, ``labels``
    """
    full_enc = tokeniser(
        full_text,
        truncation=config.truncation,
        max_length=config.max_seq_length,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids: list[int] = full_enc["input_ids"][0].tolist()
    attention_mask: list[int] = full_enc["attention_mask"][0].tolist()

    # Determine prompt boundary to mask prompt tokens from loss
    prompt_text = full_text[: full_text.rfind("### Response:\n") + len("### Response:\n")]
    prompt_enc = tokeniser(
        prompt_text,
        truncation=config.truncation,
        max_length=config.max_seq_length,
        return_tensors="pt",
        add_special_tokens=False,
    )
    # Number of prompt tokens (capped at sequence length)
    prompt_len: int = min(
        prompt_enc["input_ids"].shape[1],
        config.max_seq_length,
    )

    labels: list[int] = input_ids.copy()
    # Mask prompt tokens – model should not learn to predict them
    for i in range(prompt_len):
        labels[i] = -100
    # Mask padding tokens
    for i, mask_val in enumerate(attention_mask):
        if mask_val == 0:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def load_raw_records(dataset_path: Path) -> list[VulnRecord]:
    """
    Reads the JSONL file and returns a list of successfully parsed records.

    Parameters
    ----------
    dataset_path : Path
        Absolute or relative path to the .jsonl file.

    Returns
    -------
    list[VulnRecord]
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    records: list[VulnRecord] = []
    skipped: int = 0

    logger.info("Loading dataset from: %s", dataset_path)

    try:
        with dataset_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                record = parse_jsonl_record(line, line_no)
                if record is not None:
                    records.append(record)
                else:
                    skipped += 1
    except OSError as exc:
        logger.error("Failed to open dataset file: %s", exc)
        raise

    logger.info(
        "Parsed %d records, skipped %d malformed lines.",
        len(records),
        skipped,
    )
    return records


def build_dataset(
    dataset_path: Path,
    tokeniser: PreTrainedTokenizerBase,
    tokenisation_config: TokenisationConfig | None = None,
    val_split: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    End-to-end factory: load → format → tokenise → split into train/val.

    Parameters
    ----------
    dataset_path : Path
        Path to ``train_classifier_final.jsonl``.
    tokeniser : PreTrainedTokenizerBase
        An already-initialised tokeniser (must have ``pad_token`` set).
    tokenisation_config : TokenisationConfig | None
        Tokenisation hyper-parameters; uses defaults if ``None``.
    val_split : float
        Fraction of data reserved for validation (default 10 %).
    seed : int
        Random seed for reproducible splits.

    Returns
    -------
    DatasetDict
        ``{"train": Dataset, "validation": Dataset}``
    """
    if tokenisation_config is None:
        tokenisation_config = TokenisationConfig()

    # Ensure tokeniser has a padding token
    if tokeniser.pad_token is None:
        tokeniser.pad_token = tokeniser.eos_token
        logger.info("Tokeniser has no pad_token; using eos_token as pad_token.")

    records: list[VulnRecord] = load_raw_records(dataset_path)

    logger.info("Formatting and tokenising %d records …", len(records))
    processed: list[dict[str, list[int]]] = []

    for idx, record in enumerate(records):
        try:
            full_text = format_prompt(record)
            token_dict = tokenise_and_label(full_text, tokeniser, tokenisation_config)
            processed.append(token_dict)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Record %d – tokenisation failed: %s; skipping.", idx, exc)

    logger.info("Successfully processed %d/%d records.", len(processed), len(records))

    # Build HuggingFace Dataset
    hf_dataset = Dataset.from_list(processed)
    hf_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Train / validation split
    split = hf_dataset.train_test_split(test_size=val_split, seed=seed)
    dataset_dict = DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
        }
    )

    logger.info(
        "Dataset ready → train: %d, validation: %d",
        len(dataset_dict["train"]),
        len(dataset_dict["validation"]),
    )
    return dataset_dict


# ---------------------------------------------------------------------------
# Standalone test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Data preparation smoke-test")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Dataset/train_classifier_final.jsonl",
        help="Path to the JSONL dataset file.",
    )
    parser.add_argument(
        "--tokeniser",
        type=str,
        default="bigcode/starcoder2-3b",
        help="HuggingFace model ID or local path for the tokeniser.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum token sequence length.",
    )
    args = parser.parse_args()

    logger.info("=== Data Preparation Smoke-Test ===")

    try:
        tok = AutoTokenizer.from_pretrained(
            args.tokeniser,
            trust_remote_code=True,
        )
        tok.pad_token = tok.eos_token
        tok.padding_side = "right"

        cfg = TokenisationConfig(max_seq_length=args.max_seq_length)
        dd = build_dataset(
            dataset_path=Path(args.dataset),
            tokeniser=tok,
            tokenisation_config=cfg,
        )

        # Print a preview of one training sample
        sample = dd["train"][0]
        logger.info("Sample input_ids length : %d", len(sample["input_ids"]))
        label_tokens = [t for t in sample["labels"] if t != -100]
        logger.info("Response label tokens   : %d", len(label_tokens))
        logger.info("Smoke-test PASSED.")

    except Exception as exc:
        logger.exception("Smoke-test FAILED: %s", exc)
        sys.exit(1)
