"""
fine_tune.py
=============
QLoRA fine-tuning pipeline for Java security vulnerability detection.

Responsibilities
----------------
1. Load the base model (default: bigcode/starcoder2-3b) in 4-bit NF4
   quantisation via ``bitsandbytes``.
2. Wrap it with PEFT LoRA targeting all major linear projection layers.
3. Build a HuggingFace ``Trainer`` with a ``CosineAnnealingLR`` scheduler,
   gradient accumulation, mixed-precision training, and per-epoch evaluation.
4. Save the LoRA adapter weights + tokeniser on completion.

Usage (CLI)
-----------
python fine_tune.py \\
    --base-model bigcode/starcoder2-3b \\
    --dataset Dataset/train_classifier_final.jsonl \\
    --output-dir ./outputs/vuln-lora \\
    --max-seq-length 2048 \\
    --epochs 3

Author : Elite AI Engineering Team
Python : 3.10+
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from data_preparation import TokenisationConfig, build_dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fine_tune.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("fine_tune")


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QuantisationConfig:
    """4-bit NF4 quantisation settings for bitsandbytes."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: torch.dtype = field(default=torch.bfloat16)


@dataclass
class LoRAHyperParams:
    """Low-Rank Adaptation hyper-parameters."""

    r: int = 16                          # LoRA rank
    lora_alpha: int = 32                 # Scaling factor (alpha / r = 2)
    lora_dropout: float = 0.05
    bias: str = "none"
    # Target all attention + MLP linear projections
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class FinetuneConfig:
    """Top-level training configuration."""

    base_model: str = "bigcode/starcoder2-3b"
    dataset_path: str = "Dataset/train_classifier_final.jsonl"
    output_dir: str = "./outputs/vuln-lora"
    max_seq_length: int = 2048
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8   # effective batch = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    val_split: float = 0.10
    seed: int = 42
    fp16: bool = False
    bf16: bool = True          # Use BF16 on Ampere+ GPUs
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    gradient_checkpointing: bool = True
    group_by_length: bool = True          # Reduces padding waste
    report_to: str = "none"              # Set to "wandb" / "tensorboard" as needed
    trust_remote_code: bool = True


# ---------------------------------------------------------------------------
# Cosine Annealing LR scheduler (custom callback)
# ---------------------------------------------------------------------------

class CosineAnnealingCallback(TrainerCallback):
    """
    Injects a ``CosineAnnealingLR`` scheduler into the Trainer.

    The Trainer creates its own scheduler by default; we override it in
    ``on_train_begin`` to replace it with our cosine variant.
    """

    def __init__(self, t_max: int, eta_min: float = 1e-6) -> None:
        self.t_max = t_max
        self.eta_min = eta_min
        self._scheduler: torch.optim.lr_scheduler.CosineAnnealingLR | None = None

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            logger.warning("CosineAnnealingCallback: no optimizer found; skipping.")
            return
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.t_max,
            eta_min=self.eta_min,
        )
        logger.info(
            "CosineAnnealingLR scheduler initialised (T_max=%d, eta_min=%.2e).",
            self.t_max,
            self.eta_min,
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._scheduler is not None:
            self._scheduler.step()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_bnb_config(q_cfg: QuantisationConfig) -> BitsAndBytesConfig:
    """Creates a ``BitsAndBytesConfig`` for 4-bit NF4 quantisation."""
    return BitsAndBytesConfig(
        load_in_4bit=q_cfg.load_in_4bit,
        bnb_4bit_quant_type=q_cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=q_cfg.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=q_cfg.bnb_4bit_compute_dtype,
    )


def load_tokeniser(model_id: str, trust_remote_code: bool = True) -> PreTrainedTokenizerBase:
    """
    Loads and configures the tokeniser.

    Sets ``pad_token = eos_token`` when no pad token is defined (common for
    decoder-only models) and uses right-side padding for causal LMs.
    """
    logger.info("Loading tokeniser from: %s", model_id)
    try:
        tokeniser = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if tokeniser.pad_token is None:
            tokeniser.pad_token = tokeniser.eos_token
            logger.info("Set pad_token = eos_token ('%s').", tokeniser.eos_token)
        tokeniser.padding_side = "right"
        return tokeniser
    except Exception as exc:
        logger.exception("Tokeniser loading failed: %s", exc)
        raise


def load_base_model(
    model_id: str,
    bnb_config: BitsAndBytesConfig,
    trust_remote_code: bool = True,
) -> PreTrainedModel:
    """
    Loads the causal LM in 4-bit quantisation onto the available device.

    Disables the ``cache`` mechanism (incompatible with gradient checkpointing)
    and sets ``use_reentrant=False`` for memory-efficient backward passes.
    """
    logger.info("Loading base model from: %s  (4-bit NF4)", model_id)
    device_map: str | dict = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            use_cache=False,   # Required for gradient checkpointing
        )
        model.config.use_cache = False
        model.config.pretraining_tp = 1  # Disable tensor parallelism in forward
        logger.info("Base model loaded successfully. Parameters: %s", model.num_parameters())
        return model
    except Exception as exc:
        logger.exception("Base model loading failed: %s", exc)
        raise


def apply_lora(
    model: PreTrainedModel,
    lora_params: LoRAHyperParams,
) -> PreTrainedModel:
    """
    Prepares the quantised model for k-bit training and wraps it with LoRA.

    Steps
    -----
    1. ``prepare_model_for_kbit_training`` – upcasts LayerNorm, enables
       gradient checkpointing, and freezes non-LoRA weights.
    2. ``get_peft_model`` – injects trainable LoRA adapters.
    """
    logger.info("Preparing model for k-bit training …")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_params.r,
        lora_alpha=lora_params.lora_alpha,
        lora_dropout=lora_params.lora_dropout,
        bias=lora_params.bias,
        target_modules=lora_params.target_modules,
        inference_mode=False,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_training_arguments(cfg: FinetuneConfig) -> TrainingArguments:
    """Constructs a fully specified ``TrainingArguments`` object."""
    return TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",      # HF-native cosine; CosineAnnealingCallback overrides it
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_dir=os.path.join(cfg.output_dir, "logs"),
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        eval_strategy=cfg.eval_strategy,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=cfg.greater_is_better,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=cfg.group_by_length,
        report_to=cfg.report_to,
        seed=cfg.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,     # We manage columns ourselves
        optim="paged_adamw_8bit",        # Memory-efficient paged AdamW
        ddp_find_unused_parameters=False,
    )


def run_training(cfg: FinetuneConfig) -> None:
    """
    Orchestrates the full QLoRA fine-tuning pipeline.

    Parameters
    ----------
    cfg : FinetuneConfig
        All configuration parameters for this training run.
    """
    logger.info("=== Starting QLoRA Fine-Tuning Pipeline ===")
    logger.info("Base model    : %s", cfg.base_model)
    logger.info("Output dir    : %s", cfg.output_dir)
    logger.info("Epochs        : %d", cfg.num_epochs)
    logger.info("Max seq len   : %d", cfg.max_seq_length)

    # --- Tokeniser ---
    tokeniser = load_tokeniser(cfg.base_model, cfg.trust_remote_code)

    # --- Dataset ---
    tok_config = TokenisationConfig(max_seq_length=cfg.max_seq_length)
    dataset_dict = build_dataset(
        dataset_path=Path(cfg.dataset_path),
        tokeniser=tokeniser,
        tokenisation_config=tok_config,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    # --- Quantisation config ---
    q_cfg = QuantisationConfig()
    bnb_config = build_bnb_config(q_cfg)

    # --- Load & prepare model ---
    base_model = load_base_model(cfg.base_model, bnb_config, cfg.trust_remote_code)
    lora_model = apply_lora(base_model, LoRAHyperParams())

    # --- Training arguments ---
    training_args = build_training_arguments(cfg)

    # Cosine annealing: T_max = total optimiser steps
    total_steps: int = math.ceil(
        len(dataset_dict["train"])
        / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
        * cfg.num_epochs
    )
    cosine_callback = CosineAnnealingCallback(t_max=total_steps, eta_min=1e-7)

    # Data collator – pads inputs/labels to the batch maximum
    data_collator = DataCollatorForSeq2Seq(
        tokeniser=tokeniser,
        model=lora_model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    # --- Trainer ---
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        callbacks=[cosine_callback],
    )

    logger.info("Training configuration ready. Starting training …")
    try:
        train_result = trainer.train()
        logger.info("Training complete. Metrics: %s", train_result.metrics)
    except Exception as exc:
        logger.exception("Training failed with exception: %s", exc)
        raise

    # --- Save adapter weights and tokeniser ---
    logger.info("Saving LoRA adapter weights to: %s", cfg.output_dir)
    try:
        lora_model.save_pretrained(cfg.output_dir)
        tokeniser.save_pretrained(cfg.output_dir)
        logger.info("Adapter weights and tokeniser saved successfully.")
    except Exception as exc:
        logger.exception("Failed to save adapter weights: %s", exc)
        raise

    # Save training metrics
    metrics_path = Path(cfg.output_dir) / "train_metrics.json"
    try:
        import json
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(train_result.metrics, fh, indent=2)
        logger.info("Training metrics saved to: %s", metrics_path)
    except OSError as exc:
        logger.warning("Could not save metrics file: %s", exc)

    logger.info("=== Fine-Tuning Pipeline Complete ===")


# ---------------------------------------------------------------------------
# Argument parsing & entry point
# ---------------------------------------------------------------------------

def _parse_args() -> FinetuneConfig:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Java vulnerability detection."
    )
    parser.add_argument("--base-model", default="bigcode/starcoder2-3b")
    parser.add_argument("--dataset", default="Dataset/train_classifier_final.jsonl")
    parser.add_argument("--output-dir", default="./outputs/vuln-lora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable BF16; fall back to FP16 on non-Ampere GPUs.")
    args = parser.parse_args()

    return FinetuneConfig(
        base_model=args.base_model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        val_split=args.val_split,
        seed=args.seed,
        bf16=not args.no_bf16,
        fp16=args.no_bf16,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    try:
        run_training(cfg)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(0)
    except Exception:
        sys.exit(1)
