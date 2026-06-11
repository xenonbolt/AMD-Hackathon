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

# Standard prompt templates
PROMPT_INSTRUCTION = (
    "### Instruction: Analyze the following Java code for vulnerabilities. "
    "If a vulnerability exists, identify it and provide the remediated code.\n\n"
)
INPUT_TEMPLATE = "### Input:\n{vuln_code}\n\n"
RESPONSE_TEMPLATE = "### Response:\n{fixed_code}"


class JavaVulnerabilityDataset(Dataset):
    """
    A custom PyTorch Dataset that loads Java vulnerability data from a JSONL file,
    formats them into instruction-following prompts, and tokenizes them for
    causal language model training (with prompt label masking).
    """
    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
        mask_prompt: bool = True
    ) -> None:
        """
        Initializes the dataset.
        
        Args:
            jsonl_path: Path to the JSONL dataset.
            tokenizer: Pretrained tokenizer from Hugging Face.
            max_length: Maximum token sequence length for truncation.
            mask_prompt: If True, prompt tokens will be set to -100 in labels so the loss is computed 
                         only on the response tokens.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt
        self.examples: List[Dict[str, str]] = []

        try:
            logger.info(f"Loading dataset from {jsonl_path}")
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if "vuln_code" in data and "fixed_code" in data:
                            self.examples.append({
                                "vuln_code": data["vuln_code"],
                                "fixed_code": data["fixed_code"],
                                "description": data.get("description", ""),
                                "instruction": data.get("instruction", "")
                            })
                        elif "instruction" in data and "input" in data and ("output" in data or "response" in data):
                            self.examples.append({
                                "vuln_code": data["input"],
                                "fixed_code": data.get("output") or data.get("response", ""),
                                "description": data.get("description", ""),
                                "instruction": data["instruction"]
                            })
                        else:
                            logger.warning(
                                f"Skipping line {line_idx + 1}: Missing expected dataset fields ('vuln_code'/'fixed_code' or 'instruction'/'input'/'output'/'response')"
                            )
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON on line {line_idx + 1}: {e}")
            logger.info(f"Loaded {len(self.examples)} records successfully.")
        except Exception as e:
            logger.error(f"Failed to read dataset from {jsonl_path}: {e}")
            raise

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Prepares a single training sample by formatting the prompt,
        tokenizing, padding/truncating, and establishing labels.
        """
        example = self.examples[idx]
        vuln_code = example["vuln_code"]
        fixed_code = example["fixed_code"]

        # 1. Format the instruction/input prompt (excluding response)
        custom_instruction = example.get("instruction", "")
        if custom_instruction:
            if custom_instruction.startswith("### Instruction:"):
                prompt_text = f"{custom_instruction}\n\n{INPUT_TEMPLATE.format(vuln_code=vuln_code)}### Response:\n"
            else:
                prompt_text = f"### Instruction: {custom_instruction}\n\n{INPUT_TEMPLATE.format(vuln_code=vuln_code)}### Response:\n"
        else:
            prompt_text = f"{PROMPT_INSTRUCTION}{INPUT_TEMPLATE.format(vuln_code=vuln_code)}### Response:\n"
        # 2. Format the target response
        response_text = fixed_code

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
    Generates a sample JSONL file containing mock Java vulnerability examples
    for verification and testing purposes.
    """
    mock_data = [
        {
            "vuln_code": "public void process(String input) throws Exception {\n    Connection conn = DriverManager.getConnection(DB_URL);\n    Statement stmt = conn.createStatement();\n    ResultSet rs = stmt.executeQuery(\"SELECT * FROM users WHERE username = '\" + input + \"'\");\n}",
            "description": "SQL Injection vulnerability due to dynamic query building.",
            "fixed_code": "public void process(String input) throws Exception {\n    Connection conn = DriverManager.getConnection(DB_URL);\n    String sql = \"SELECT * FROM users WHERE username = ?\";\n    PreparedStatement stmt = conn.prepareStatement(sql);\n    stmt.setString(1, input);\n    ResultSet rs = stmt.executeQuery();\n}"
        },
        {
            "vuln_code": "public void handle(HttpServletRequest req) {\n    String path = req.getParameter(\"path\");\n    File file = new File(\"/var/uploads/\" + path);\n    FileInputStream fis = new FileInputStream(file);\n}",
            "description": "Path Traversal vulnerability via uncontrolled parameters.",
            "fixed_code": "public void handle(HttpServletRequest req) {\n    String path = req.getParameter(\"path\");\n    File file = new File(\"/var/uploads/\" + path);\n    String canonicalPath = file.getCanonicalPath();\n    if (!canonicalPath.startsWith(\"/var/uploads/\")) {\n        throw new SecurityException(\"Unauthorized path access\");\n    }\n    FileInputStream fis = new FileInputStream(file);\n}"
        }
    ]
    with open(path, "w", encoding="utf-8") as f:
        for entry in mock_data:
            f.write(json.dumps(entry) + "\n")
    logger.info(f"Generated mock JSONL dataset at: {path}")


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
        
        # Instantiate dataset
        dataset = JavaVulnerabilityDataset(test_path, tokenizer, max_length=512)
        collator = CausalLMDataCollator(tokenizer)
        
        # Test loading first element
        sample = dataset[0]
        logger.info("Successfully processed first dataset sample.")
        logger.info(f"Input IDs shape: {sample['input_ids'].shape}")
        logger.info(f"Labels shape: {sample['labels'].shape}")

        # Test batching via data collator
        batch = collator([dataset[0], dataset[1]])
        logger.info("Successfully batched dataset samples.")
        logger.info(f"Batch Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Batch Labels shape: {batch['labels'].shape}")

    except Exception as err:
        logger.error(f"Verification execution failed: {err}")
    finally:
        if test_path.exists():
            test_path.unlink()
            logger.info("Cleaned up temporary test file.")
