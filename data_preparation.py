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

# ── Special token markers (must match the JSONL format exactly) ────────────────
INSTRUCTION_MARKER = "<|instruction|>"
INPUT_MARKER       = "<|input|>"
RESPONSE_MARKER    = "<|response|>"


class JavaVulnerabilityDataset(Dataset):
    """
    PyTorch Dataset for training on ``train_classifier_final.jsonl``.

    Each JSONL record contains a single ``text`` field pre-formatted as:

        <|instruction|>
        Analyze the Java code and identify ALL security vulnerabilities.
        Return structured JSON only.

        <|input|>
        <java source code>

        <|response|>
        {
          "vulnerabilities": [
            {
              "cwe_id": "CWE-89",
              "cwe_name": "...",
              "severity": "high",
              "confidence": 0.95,
              "location": {"start_line": 1, "end_line": 42, "function": "bad"},
              "description": "...",
              "impact": "...",
              "recommendation": "..."
            }
          ]
        }

    Safe examples have ``"vulnerabilities": []``.

    The dataset:
      1. Reads each ``text`` field verbatim.
      2. Splits on ``<|response|>`` to obtain the prompt prefix and response.
      3. Tokenizes each part separately so that prompt tokens can be masked
         (-100) in the labels, training the model only to generate the JSON
         response.
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        mask_prompt: bool = True,
    ) -> None:
        """
        Args:
            jsonl_path : Path to the JSONL file (e.g. train_classifier_final.jsonl).
            tokenizer  : HuggingFace pre-trained tokenizer.
            max_length : Maximum total token sequence length (prompt + response).
                         2048 is recommended for this dataset's longer entries.
            mask_prompt: If True, prompt tokens are masked with -100 so the loss
                         is computed only on the JSON response tokens.
        """
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self.examples: List[str] = []   # raw ``text`` strings

        try:
            logger.info(f"Loading dataset from {jsonl_path}")
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", "")
                        if not text or RESPONSE_MARKER not in text:
                            logger.warning(
                                f"Skipping line {line_idx + 1}: "
                                f"missing 'text' field or '{RESPONSE_MARKER}' marker."
                            )
                            continue
                        self.examples.append(text)
                    except json.JSONDecodeError as exc:
                        logger.error(f"JSON parse error on line {line_idx + 1}: {exc}")

            logger.info(
                f"Loaded {len(self.examples)} records "
                f"({sum(1 for t in self.examples if '\"vulnerabilities\": []' in t)} safe, "
                f"{sum(1 for t in self.examples if '\"vulnerabilities\": []' not in t)} vulnerable)."
            )
        except Exception as exc:
            logger.error(f"Failed to open dataset at {jsonl_path}: {exc}")
            raise

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Splits the pre-formatted text at ``<|response|>`` and tokenizes each
        part independently so labels can be masked on the prompt side.

        Returns a dict with ``input_ids``, ``attention_mask``, and ``labels``.
        """
        text = self.examples[idx]

        # Split into prompt prefix and JSON response
        split_idx = text.find(RESPONSE_MARKER)
        # Prompt includes the marker itself so the model learns to produce
        # output immediately after seeing <|response|>
        prompt_text   = text[: split_idx + len(RESPONSE_MARKER)] + "\n"
        response_text = text[split_idx + len(RESPONSE_MARKER):].strip()

        try:
            prompt_enc   = self.tokenizer(prompt_text,   add_special_tokens=False, truncation=False)
            response_enc = self.tokenizer(response_text, add_special_tokens=False, truncation=False)

            bos_id = self.tokenizer.bos_token_id
            eos_id = self.tokenizer.eos_token_id
            bos_tokens = [bos_id] if bos_id is not None else []
            eos_tokens = [eos_id] if eos_id is not None else []

            prompt_ids   = bos_tokens + prompt_enc["input_ids"]
            response_ids = response_enc["input_ids"] + eos_tokens

            # Truncation: preserve prompt, trim response tail
            total = len(prompt_ids) + len(response_ids)
            if total > self.max_length:
                excess = total - self.max_length
                if excess < len(response_ids):
                    response_ids = response_ids[:-excess]
                else:
                    # Prompt itself is extremely long — truncate prompt head
                    prompt_ids   = prompt_ids[:self.max_length - 1]
                    response_ids = eos_tokens

            input_ids      = prompt_ids + response_ids
            attention_mask = [1] * len(input_ids)

            if self.mask_prompt:
                # Mask prompt tokens: model only trains to produce the JSON response
                labels = [-100] * len(prompt_ids) + response_ids
            else:
                labels = input_ids.copy()

            return {
                "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels":         torch.tensor(labels,         dtype=torch.long),
            }
        except Exception as exc:
            logger.error(f"Tokenization error at index {idx}: {exc}")
            raise


class CausalLMDataCollator:
    """
    Dynamic padding collator.  Pads all items in a batch to the longest
    sequence in that batch, using -100 for label padding.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len      = max(x["input_ids"].size(0) for x in batch)
        pad_id       = self.tokenizer.pad_token_id
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        if pad_id is None:
            raise ValueError("Tokenizer must have a valid pad_token_id.")

        batch_input_ids, batch_masks, batch_labels = [], [], []

        for item in batch:
            ids    = item["input_ids"]
            mask   = item["attention_mask"]
            labels = item["labels"]
            diff   = max_len - ids.size(0)

            if diff > 0:
                pad_ids    = torch.full((diff,), pad_id, dtype=torch.long)
                pad_mask   = torch.zeros((diff,),         dtype=torch.long)
                pad_labels = torch.full((diff,), -100,    dtype=torch.long)
                if padding_side == "right":
                    ids    = torch.cat([ids,    pad_ids])
                    mask   = torch.cat([mask,   pad_mask])
                    labels = torch.cat([labels, pad_labels])
                else:
                    ids    = torch.cat([pad_ids,    ids])
                    mask   = torch.cat([pad_mask,   mask])
                    labels = torch.cat([pad_labels, labels])

            batch_input_ids.append(ids)
            batch_masks.append(mask)
            batch_labels.append(labels)

        return {
            "input_ids":      torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_masks),
            "labels":         torch.stack(batch_labels),
        }


def generate_mock_jsonl(path: Path) -> None:
    """
    Writes two sample records (one vulnerable, one safe) in the
    train_classifier_final.jsonl format for unit-testing purposes.
    """
    instruction = (
        "Analyze the Java code and identify ALL security vulnerabilities. "
        "Return structured JSON only."
    )
    records = [
        {
            "text": (
                f"{INSTRUCTION_MARKER}\n{instruction}\n\n"
                f"{INPUT_MARKER}\n"
                "public void process(String input) throws Exception {\n"
                "    Statement stmt = conn.createStatement();\n"
                "    ResultSet rs = stmt.executeQuery(\n"
                "        \"SELECT * FROM users WHERE name = '\" + input + \"'\");\n"
                "}\n\n"
                f"{RESPONSE_MARKER}\n"
                "{\n"
                '  "vulnerabilities": [\n'
                "    {\n"
                '      "cwe_id": "CWE-89",\n'
                '      "cwe_name": "SQL Injection (CWE-89)",\n'
                '      "severity": "critical",\n'
                '      "confidence": 0.97,\n'
                '      "location": {"start_line": 1, "end_line": 5, "function": "process"},\n'
                '      "description": "Unsanitised input concatenated into SQL query.",\n'
                '      "impact": "Data exfiltration or destruction.",\n'
                '      "recommendation": "Use PreparedStatement with parameterised queries."\n'
                "    }\n"
                "  ]\n"
                "}"
            )
        },
        {
            "text": (
                f"{INSTRUCTION_MARKER}\n{instruction}\n\n"
                f"{INPUT_MARKER}\n"
                "public String greet(String name) {\n"
                "    return \"Hello, \" + name;\n"
                "}\n\n"
                f"{RESPONSE_MARKER}\n"
                '{\n  "vulnerabilities": []\n}'
            )
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.info(f"Mock JSONL written to: {path}")


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Validate data_preparation pipeline")
    parser.add_argument(
        "--test_file",
        type=str,
        default="Dataset/train_classifier_final.jsonl",
        help="Path to JSONL file to validate (defaults to the real dataset)",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="bigcode/starcoder2-3b",
        help="HuggingFace tokenizer ID",
    )
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Generate and use a 2-record mock file instead of the real dataset",
    )
    args = parser.parse_args()

    test_path = Path(args.test_file)
    if args.use_mock:
        mock_path = Path("Dataset/mock_test.jsonl")
        generate_mock_jsonl(mock_path)
        test_path = mock_path

    try:
        logger.info(f"Loading tokenizer: {args.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

        dataset  = JavaVulnerabilityDataset(test_path, tokenizer, max_length=2048)
        collator = CausalLMDataCollator(tokenizer)

        sample = dataset[0]
        logger.info(f"Sample 0 — input_ids: {sample['input_ids'].shape}, labels: {sample['labels'].shape}")

        # Show decoded prompt + response split
        prompt_len    = (sample["labels"] == -100).sum().item()
        response_ids  = sample["input_ids"][prompt_len:]
        logger.info(f"  Prompt tokens : {prompt_len}")
        logger.info(f"  Response tokens: {len(response_ids)}")
        logger.info(f"  Response preview: {tokenizer.decode(response_ids[:60], skip_special_tokens=True)[:200]}")

        if len(dataset) >= 2:
            batch = collator([dataset[0], dataset[1]])
            logger.info(f"Batch shapes — input_ids: {batch['input_ids'].shape}, labels: {batch['labels'].shape}")

    except Exception as err:
        logger.error(f"Validation failed: {err}", exc_info=True)
    finally:
        if args.use_mock and Path("Dataset/mock_test.jsonl").exists():
            Path("Dataset/mock_test.jsonl").unlink()
            logger.info("Cleaned up mock file.")
