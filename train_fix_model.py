import platform
# Monkeypatch platform.win32_ver to bypass Windows WMI query errors/hangs
platform.win32_ver = lambda release='', version='', csd='', ptype='': ('10', '10.0.22621', '', 'Multiprocessor Free')

import argparse
import logging
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_fix_model")

class FixVulnerabilityDataset(Dataset):
    """
    A custom PyTorch Dataset designed to process fixed.jsonl for training the remediation model.
    It expects 'instruction', 'input' (JSON object), and 'output' (JSON object).
    """
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[str] = []

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        jsonl_path_obj = Path(jsonl_path)
        try:
            logger.info(f"Attempting to load dataset from: {jsonl_path_obj}")
            if not jsonl_path_obj.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {jsonl_path_obj}")

            with open(jsonl_path_obj, "r", encoding="utf-8") as file:
                for line_idx, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    try:
                        data: Dict[str, Any] = json.loads(line)
                        if "instruction" in data and "input" in data and "output" in data:
                            input_str = json.dumps(data["input"], ensure_ascii=False) if isinstance(data["input"], dict) else str(data["input"])
                            output_str = json.dumps(data["output"], ensure_ascii=False) if isinstance(data["output"], dict) else str(data["output"])
                            
                            text = f"<|instruction|>\n{data['instruction']}\n\n<|input|>\n{input_str}\n\n<|response|>\n{output_str}"
                            self.examples.append(text)
                        else:
                            logger.warning(f"Skipping line {line_idx} in {jsonl_path_obj.name}: Required keys not found.")
                    except json.JSONDecodeError as decode_err:
                        logger.error(f"JSON parse failure on line {line_idx} in {jsonl_path_obj.name}: {decode_err}")
            
            logger.info(f"Successfully loaded {len(self.examples)} examples from {jsonl_path_obj.name}")
        except Exception as err:
            logger.error(f"Failed to read dataset from {jsonl_path_obj}: {err}", exc_info=True)
            raise

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]
        try:
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
            labels: List[int] = list(input_ids)

            response_marker = "<|response|>\n"
            response_start_char = text.find(response_marker)

            if response_start_char != -1:
                prompt_text = text[:response_start_char + len(response_marker)]
                prompt_encodings = self.tokenizer(
                    prompt_text,
                    add_special_tokens=True,
                    return_tensors=None
                )
                prompt_token_len = len(prompt_encodings["input_ids"])
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

class FixCausalLMDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)
        batch_input_ids: List[torch.Tensor] = []
        batch_attention_mask: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        
        pad_token_id = self.tokenizer.pad_token_id
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

def find_all_linear_names(model: torch.nn.Module) -> List[str]:
    import bitsandbytes as bnb
    cls_4bit = bnb.nn.Linear4bit
    cls_8bit = bnb.nn.Linear8bitLt
    cls_linear = torch.nn.Linear
    
    linear_layers = set()
    for name, module in model.named_modules():
        if isinstance(module, (cls_4bit, cls_8bit, cls_linear)):
            names = name.split(".")
            linear_layers.add(names[-1])
            
    for exclude_name in ["lm_head", "embed_tokens", "classification_head", "output_layer", "norm", "wte", "wpe"]:
        if exclude_name in linear_layers:
            linear_layers.remove(exclude_name)
            
    if not linear_layers:
        fallback_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        logger.warning(f"No linear layers detected dynamically. Defaulting: {fallback_targets}")
        return fallback_targets
        
    return list(linear_layers)

def run_training(
    model_id: str,
    dataset_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    lr: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    max_length: int = 2048
) -> None:
    try:
        compute_dtype = torch.float32
        bnb_config = None
        device_map = None

        if torch.cuda.is_available():
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype
            )
            device_map = "auto"

        torch_dtype = compute_dtype if torch.cuda.is_available() else torch.float32
        logger.info(f"Loading base model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        logger.info(f"Loading tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        if torch.cuda.is_available():
            model = prepare_model_for_kbit_training(model)
        else:
            model.enable_input_require_grads()

        target_modules = find_all_linear_names(model)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        train_dataset = FixVulnerabilityDataset(
            jsonl_path=dataset_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
        data_collator = FixCausalLMDataCollator(tokenizer=tokenizer)

        is_cuda = torch.cuda.is_available()
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            bf16=(is_cuda and compute_dtype == torch.bfloat16),
            fp16=(is_cuda and compute_dtype == torch.float16),
            optim="paged_adamw_8bit" if is_cuda else "adamw_torch",
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        logger.info("Executing remediation model training loop...")
        trainer.train()
        
        logger.info(f"Saving PEFT adapter configurations to output directory: {output_dir}")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    except Exception as err:
        logger.error(f"Fine-tuning failed: {err}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Script for Fix Engine")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-Coder-Next", help="Base model identifier")
    parser.add_argument("--dataset_path", type=str, default="Dataset/fixed.jsonl", help="Path to training JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./adapters_fix", help="Output directory for saved adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Strict sequence length constraint")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_training(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length
    )
