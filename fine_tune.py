import platform
# Monkeypatch platform.win32_ver to bypass Windows WMI query errors/hangs
platform.win32_ver = lambda release='', version='', csd='', ptype='': ('10', '10.0.22621', '', 'Multiprocessor Free')

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model
)

from data_preparation import JavaVulnerabilityDataset, CausalLMDataCollator

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fine_tune")


def find_all_linear_names(model: torch.nn.Module) -> List[str]:
    """
    Dynamically identifies all linear layer module names in the target model.
    Ensures target module compatibility across different model architectures (e.g., Llama, StarCoder).
    """
    cls_linear = torch.nn.Linear
    
    linear_layers = set()
    for name, module in model.named_modules():
        if isinstance(module, cls_linear):
            names = name.split(".")
            # Target the leaf module name (e.g., 'q_proj', 'v_proj')
            linear_layers.add(names[-1])
            
    # Exclude output/embedding layers
    for exclude_name in ["lm_head", "embed_tokens", "classification_head", "output_layer", "norm", "wte", "wpe"]:
        if exclude_name in linear_layers:
            linear_layers.remove(exclude_name)
            
    # Default fallback list if no layers are identified dynamically
    if not linear_layers:
        fallback_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        logger.warning(f"No linear layers detected dynamically. Defaulting to standard projection modules: {fallback_targets}")
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
    eval_dataset_path: Optional[str] = None,
    max_length: int = 2048
) -> None:
    """
    Sets up QLoRA config, quantization parameters, data collators,
    and runs the HuggingFace Trainer to fine-tune the model.
    """
    try:
        if "LOCAL_RANK" in os.environ:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="gloo")
            local_rank = int(os.environ["LOCAL_RANK"])
            logger.info(f"Distributed training initialized via torchrun. Local rank: {local_rank}")
        else:
            logger.info("Running in single-node mode without distributed process group.")

        # CPU-only FSDP setup
        compute_dtype = torch.float32

        # 2. Load Model & Tokenizer
        logger.info(f"Loading base model: {model_id} on CPU (FP32)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=compute_dtype,
            trust_remote_code=True
        )

        logger.info(f"Loading tokenizer: {model_id}")
        if "deepseek" in model_id.lower():
            logger.info("DeepSeek model detected. Loading tokenizer via PreTrainedTokenizerFast to avoid LlamaTokenizer space bug.")
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            logger.info("Setting missing pad_token to eos_token in tokenizer.")
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        # 3. Prepare Model
        logger.info("Enabling input gradients for CPU gradient checkpointing...")
        model.enable_input_require_grads()

        # 4. Detect target modules for LoRA
        target_modules = find_all_linear_names(model)
        logger.info(f"Targeting modules for LoRA parameter tuning: {target_modules}")

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
        logger.info("LoRA configuration overlay successful. Trainable parameters summary:")
        model.print_trainable_parameters()

        # 6. Load Datasets
        train_dataset = JavaVulnerabilityDataset(
            jsonl_path=dataset_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        eval_dataset = None
        if eval_dataset_path:
            eval_dataset = JavaVulnerabilityDataset(
                jsonl_path=eval_dataset_path,
                tokenizer=tokenizer,
                max_length=max_length
            )

        data_collator = CausalLMDataCollator(tokenizer=tokenizer)

        # 7. Configure Training Arguments
        # Use FSDP configuration parameters if distributed
        fsdp_config = ["full_shard", "auto_wrap"] if "LOCAL_RANK" in os.environ else []
        
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
            bf16=False,
            fp16=False,
            use_cpu=True,
            optim="adamw_torch",
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            fsdp=fsdp_config,
            report_to="none"
        )

        # 8. Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )

        # 9. Execute Training
        logger.info("Executing model training loop...")
        trainer.train()
        logger.info("Fine-tuning pipeline execution successfully completed.")

        # 10. Save Output Adapters and Tokenizer
        logger.info(f"Saving PEFT adapter configurations to output directory: {output_dir}")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("PEFT adapter save operation complete.")

    except Exception as err:
        logger.error(f"Fine-tuning training process failed: {err}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Script")
    parser.add_argument("--model_id", type=str, required=True, help="Base model identifier on Hugging Face Hub")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training JSONL dataset")
    parser.add_argument("--eval_dataset_path", type=str, default=None, help="Path to validation JSONL dataset (optional)")
    parser.add_argument("--output_dir", type=str, default="./adapters", help="Output directory for saved adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Strict sequence length constraint")

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
        lora_dropout=args.lora_dropout,
        max_length=args.max_length
    )
