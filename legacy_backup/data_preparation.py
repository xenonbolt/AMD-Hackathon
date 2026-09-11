import platform
# Monkeypatch platform.win32_ver to bypass Windows WMI query errors/hangs
platform.win32_ver = lambda release='', version='', csd='', ptype='': ('10', '10.0.22621', '', 'Multiprocessor Free')

import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("data_preparation")


class JavaVulnerabilityDataset(Dataset):
    """
    A custom PyTorch Dataset designed to process a local JSONL file containing
    a single 'text' block in the format:
    "<|instruction|>\n{instruction}\n\n<|input|>\n{java_code}\n\n<|response|>\n{json_output}"
    Configured for Causal LM training where target labels match input IDs.
    """
    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
    ) -> None:
        """
        Initializes the dataset and loads data records.

        Args:
            jsonl_path: Path to the JSONL dataset file.
            tokenizer: Pretrained tokenizer from Hugging Face.
            max_length: Maximum token sequence length for truncation.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[str] = []

        # Ensure pad_token_id is configured correctly
        if self.tokenizer.pad_token_id is None:
            logger.info("Tokenizer pad_token_id is missing. Falling back to eos_token_id.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Read JSONL file and extract the 'text' field
        jsonl_path = Path(jsonl_path)
        try:
            logger.info(f"Attempting to load dataset from: {jsonl_path}")
            if not jsonl_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {jsonl_path}")

            with open(jsonl_path, "r", encoding="utf-8") as file:
                for line_idx, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    try:
                        data: Dict[str, Any] = json.loads(line)
                        if "text" in data:
                            self.examples.append(data["text"])
                        elif "instruction" in data and "input" in data and "output" in data:
                            text = f"<|instruction|>\n{data['instruction']}\n\n<|input|>\n{data['input']}\n\n<|response|>\n{data['output']}"
                            self.examples.append(text)
                        else:
                            logger.warning(
                                f"Skipping line {line_idx} in {jsonl_path.name}: 'text' key not found."
                            )
                    except json.JSONDecodeError as decode_err:
                        logger.error(
                            f"JSON parse failure on line {line_idx} in {jsonl_path.name}: {decode_err}"
                        )
            
            logger.info(f"Successfully loaded {len(self.examples)} examples from {jsonl_path.name}")
        except Exception as err:
            logger.error(f"Failed to read dataset from {jsonl_path}: {err}", exc_info=True)
            raise

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single sample, tokenizes it, manages sequence lengths,
        and sets labels for Causal LM training with prompt masking.

        Only the response portion (after <|response|>) contributes to the
        training loss. The instruction and input tokens have their labels
        set to -100 so CrossEntropyLoss ignores them.
        """
        text = self.examples[idx]
        try:
            # Tokenize complete block with strict length limit
            encodings = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_tensors=None
            )

            input_ids: List[int] = encodings["input_ids"]
            attention_mask: List[int] = encodings["attention_mask"]

            # Create labels aligned with input_ids
            labels: List[int] = list(input_ids)

            # --- Prompt Masking ---
            # Find where the response section starts in the raw text.
            # Everything before (and including) the <|response|>\n marker is prompt;
            # the model should NOT be trained to predict those tokens.
            response_marker = "<|response|>\n"
            response_start_char = text.find(response_marker)

            if response_start_char != -1:
                # The prompt is everything up to and including the marker
                prompt_text = text[:response_start_char + len(response_marker)]

                # Tokenize just the prompt portion to find its token boundary
                prompt_encodings = self.tokenizer(
                    prompt_text,
                    add_special_tokens=True,
                    return_tensors=None
                )
                prompt_token_len = len(prompt_encodings["input_ids"])

                # Mask all prompt tokens in labels (set to -100 so loss ignores them)
                mask_len = min(prompt_token_len, len(labels))
                for i in range(mask_len):
                    labels[i] = -100

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
        except Exception as err:
            logger.error(f"Tokenization failed for dataset item at index {idx}: {err}", exc_info=True)
            raise


class CausalLMDataCollator:
    """
    Data Collator to dynamically pad batches to the maximum sequence length
    present in the current batch. Configures padding and masks label padding positions to -100.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Find maximum length in the current batch
        max_len = max(item["input_ids"].size(0) for item in batch)
        
        batch_input_ids: List[torch.Tensor] = []
        batch_attention_mask: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must possess a valid, configured pad_token_id.")
            
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
    in the expected format for verification purposes.
    """
    mock_data = [
        {
            "text": (
                "<|instruction|>\nAnalyze the Java code and identify ALL security vulnerabilities. Return structured JSON only.\n\n"
                "<|input|>\npublic void process(String input) throws Exception {\n"
                "    Connection conn = DriverManager.getConnection(DB_URL);\n"
                "    Statement stmt = conn.createStatement();\n"
                "    ResultSet rs = stmt.executeQuery(\"SELECT * FROM users WHERE username = '\" + input + \"'\");\n"
                "}\n\n"
                "<|response|>\n{\n  \"vulnerabilities\": [\n    {\n      \"cwe_id\": \"CWE-89\",\n"
                "      \"cwe_name\": \"SQL Injection\",\n      \"severity\": \"high\",\n"
                "      \"description\": \"SQL Injection vulnerability due to dynamic query building.\"\n"
                "    }\n  ]\n}"
            )
        },
        {
            "text": (
                "<|instruction|>\nAnalyze the Java code and identify ALL security vulnerabilities. Return structured JSON only.\n\n"
                "<|input|>\npublic void handle(HttpServletRequest req) {\n"
                "    String path = req.getParameter(\"path\");\n"
                "    File file = new File(\"/var/uploads/\" + path);\n"
                "    FileInputStream fis = new FileInputStream(file);\n"
                "}\n\n"
                "<|response|>\n{\n  \"vulnerabilities\": [\n    {\n      \"cwe_id\": \"CWE-22\",\n"
                "      \"cwe_name\": \"Path Traversal\",\n      \"severity\": \"high\",\n"
                "      \"description\": \"Path Traversal vulnerability via uncontrolled parameters.\"\n"
                "    }\n  ]\n}"
            )
        }
    ]
    try:
        with open(path, "w", encoding="utf-8") as file:
            for entry in mock_data:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Generated mock JSONL dataset at: {path}")
    except Exception as err:
        logger.error(f"Failed to generate mock dataset at {path}: {err}", exc_info=True)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Test and Verify Data Preparation")
    parser.add_argument("--test_file", type=str, default="test_dataset_prepared.jsonl", help="Path to JSONL output/test file")
    parser.add_argument("--tokenizer_name", type=str, default="bigcode/starcoder2-3b", help="Hugging Face tokenizer ID")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    generate_mock_jsonl(test_path)

    try:
        logger.info(f"Loading tokenizer: {args.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        
        # Instantiate dataset
        dataset = JavaVulnerabilityDataset(test_path, tokenizer, max_length=1024)
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

    except Exception as e:
        logger.error(f"Verification execution failed: {e}", exc_info=True)
    finally:
        if test_path.exists():
            test_path.unlink()
            logger.info("Cleaned up temporary test file.")
