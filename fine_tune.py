import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from data_preparation import JavaVulnerabilityDataset, CausalLMDataCollator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fine_tune")


def find_all_linear_names(model: torch.nn.Module) -> List[str]:
    """
    Dynamically identifies all linear layer module names in the target model.
    This ensures target compatibility across different model architectures (e.g., Llama, StarCoder).
    """
    import bitsandbytes as bnb
    cls_4bit = bnb.nn.Linear4bit
    cls_8bit = bnb.nn.Linear8bitLt
    cls_linear = torch.nn.Linear
    
    linear_layers = set()
    for name, module in model.named_modules():
        if isinstance(module, (cls_4bit, cls_8bit, cls_linear)):
            names = name.split(".")
            # Target the leaf module name (e.g., 'q_proj', 'v_proj')
            linear_layers.add(names[-1])
            
    # Exclude output layers like lm_head or classifier
    for exclude_name in ["lm_head", "embed_tokens", "classification_head", "output_layer"]:
        if exclude_name in linear_layers:
            linear_layers.remove(exclude_name)
            
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
    eval_dataset_path: Optional[str] = None
) -> None:
    """
    Sets up QLoRA config, quantization parameters, data collators, 
    and drives the HuggingFace Trainer to completion.
    """
    try:
        logger.info(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        
        # 1. Setup Quantization Configuration (NF4)
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Setting computation dtype to: {compute_dtype}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype
        )

        # 2. Load Model & Tokenizer
        logger.info(f"Loading quantized base model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        logger.info(f"Loading tokenizer for: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 3. Prepare Model for k-bit training
        logger.info("Preparing model for k-bit training...")
        model = prepare_model_for_kbit_training(model)

        # 4. Target Linear Modules Setup
        target_modules = find_all_linear_names(model)
        logger.info(f"Detected target modules for LoRA: {target_modules}")

        # 5. Configure LoRA adapter settings
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        logger.info("LoRA configuration wrapper complete. Trainable parameters:")
        model.print_trainable_parameters()

        # 6. Load Datasets
        train_dataset = JavaVulnerabilityDataset(
            jsonl_path=dataset_path,
            tokenizer=tokenizer,
            max_length=1024,
            mask_prompt=True
        )
        
        eval_dataset = None
        if eval_dataset_path:
            eval_dataset = JavaVulnerabilityDataset(
                jsonl_path=eval_dataset_path,
                tokenizer=tokenizer,
                max_length=1024,
                mask_prompt=True
            )

        data_collator = CausalLMDataCollator(tokenizer=tokenizer)

        # 7. Configure Training Arguments
        # Includes Cosine Annealing, Gradient Accumulation, and FP16/BF16 flag parameters
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            bf16=(compute_dtype == torch.bfloat16),
            fp16=(compute_dtype == torch.float16),
            optim="paged_adamw_8bit",
            ddp_find_unused_parameters=False,
            report_to="none" # Prevents logging to WandB/Tensorboard automatically
        )

        # 8. Instantiate Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        # 9. Execute Training
        logger.info("Starting training loop...")
        trainer.train()
        logger.info("Training complete.")

        # 10. Save Output Adapters and Tokenizer
        logger.info(f"Saving fine-tuned adapter weights to {output_dir}")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saving operation complete.")

    except Exception as e:
        logger.error(f"Failed to execute training pipeline: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Script")
    parser.add_argument("--model_id", type=str, required=True, help="Base model identifier on Hugging Face Hub")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training JSONL dataset")
    parser.add_argument("--eval_dataset_path", type=str, default=None, help="Path to validation JSONL dataset (optional)")
    parser.add_argument("--output_dir", type=str, default="./adapters", help="Output directory for saved adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")

    args = parser.parse_args()

    # Create target adapter output directory if necessary
    os.makedirs(args.output_dir, exist_ok=True)

    run_training(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        eval_dataset_path=args.eval_dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
