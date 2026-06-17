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
        logger.info(f"Targeting device setup. CUDA status: {torch.cuda.is_available()}")
        
        # 1. Setup Quantization Configuration (NF4)
        bnb_config = None
        device_map = None
        compute_dtype = torch.float32

        if torch.cuda.is_available():
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logger.info(f"CUDA detected. Using computation dtype: {compute_dtype}")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype
            )
            device_map = "auto"
        else:
            logger.warning("CUDA is not available. Model loading will fallback to CPU full precision (FP32).")

        # 2. Load Model & Tokenizer
        logger.info(f"Loading base model: {model_id}")
        torch_dtype = compute_dtype if torch.cuda.is_available() else torch.float32
        logger.info(f"Selected model loading precision dtype: {torch_dtype}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
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
        logger.info("[STEP 2/9 DONE] Tokenizer loaded successfully.")

        # 3. Prepare Model for k-bit training
        if torch.cuda.is_available():
            logger.info("[STEP 3/9] Preparing model for k-bit training...")
            model = prepare_model_for_kbit_training(model)
        else:
            logger.info("Enabling input gradients for CPU gradient checkpointing...")
            model.enable_input_require_grads()

        logger.info("[STEP 3/9 DONE] Model prepared for k-bit training.")

        # 4. Detect target modules for LoRA
        logger.info("[STEP 4/9] Detecting target LoRA modules...")
        target_modules = find_all_linear_names(model)
        logger.info(f"Targeting modules for LoRA parameter tuning: {target_modules}")

        # 5. Configure LoRA adapter settings
        logger.info("[STEP 5/9] Configuring LoRA adapter...")
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
        logger.info("[STEP 5/9 DONE] LoRA adapter configured.")

        # 6. Load Datasets
        logger.info("[STEP 6/9] Loading training dataset...")
        train_dataset = JavaVulnerabilityDataset(
            jsonl_path=dataset_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
        logger.info(f"[STEP 6/9 DONE] Loaded {len(train_dataset)} training examples.")
        
        eval_dataset = None
        if eval_dataset_path:
            eval_dataset = JavaVulnerabilityDataset(
                jsonl_path=eval_dataset_path,
                tokenizer=tokenizer,
                max_length=max_length
            )

        data_collator = CausalLMDataCollator(tokenizer=tokenizer)

        # 7. Configure Training Arguments
        logger.info("[STEP 7/9] Configuring training arguments...")
        is_cuda = torch.cuda.is_available()
        
        # Log GPU memory status before training
        if is_cuda:
            gpu_mem_total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            gpu_mem_alloc = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            logger.info(f"GPU Memory: {gpu_mem_alloc:.1f}GB allocated, {gpu_mem_reserved:.1f}GB reserved, {gpu_mem_total:.1f}GB total")
        
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
            bf16=(is_cuda and compute_dtype == torch.bfloat16),
            fp16=(is_cuda and compute_dtype == torch.float16),
            optim="paged_adamw_8bit" if is_cuda else "adamw_torch",
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            report_to="none"
        )
        logger.info(f"[STEP 7/9 DONE] Training args: batch_size={batch_size}, grad_accum={grad_accum}, epochs={epochs}, lr={lr}")

        # 8. Initialize Trainer
        logger.info("[STEP 8/9] Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        logger.info("[STEP 8/9 DONE] Trainer initialized.")

        # 9. Execute Training
        logger.info("[STEP 9/9] Starting training loop...")
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
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
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
