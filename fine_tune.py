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
import json
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

    r: int = 16
    lora_alpha: int = 32
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
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    val_split: float = 0.10
    seed: int = 42
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    # save_strategy/eval_strategy are "steps" to avoid PEFT checkpoint-reload
    # conflicts with load_best_model_at_end=True.
    # save_steps/eval_steps are computed dynamically in run_training() so they
    # are always reachable regardless of dataset size. The sentinel value of -1
    # means "compute automatically" (see run_training).
    save_strategy: str = "steps"
    save_steps: int = -1          # -1 → auto-computed to ~25% of total steps
    eval_strategy: str = "steps"
    eval_steps: int = -1          # -1 → auto-computed to ~25% of total steps
    save_total_limit: int = 3
    # Disabled: avoids Trainer trying to reload a PEFT checkpoint
    load_best_model_at_end: bool = False
    gradient_checkpointing: bool = True
    group_by_length: bool = True
    report_to: str = "none"
    trust_remote_code: bool = True


# ---------------------------------------------------------------------------
# Cosine Annealing LR callback
# ---------------------------------------------------------------------------

class CosineAnnealingCallback(TrainerCallback):
    """
    Attaches a ``CosineAnnealingLR`` scheduler to the Trainer's optimizer.

    The HF Trainer already sets ``lr_scheduler_type="cosine"`` in
    TrainingArguments, but this callback provides an explicit
    PyTorch-native ``CosineAnnealingLR`` handle for full control
    over T_max and eta_min.
    """

    def __init__(self, t_max: int, eta_min: float = 1e-7) -> None:
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
            logger.warning("CosineAnnealingCallback: optimizer not available yet; skipping.")
            return
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.t_max,
            eta_min=self.eta_min,
        )
        logger.info(
            "CosineAnnealingLR initialised (T_max=%d, eta_min=%.2e).",
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
# Model loading helpers
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
    """Loads and configures the tokeniser with right-padding for causal LM training."""
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
    Loads the causal LM in 4-bit NF4 quantisation.

    ``use_cache=False`` is required when gradient checkpointing is enabled.
    """
    logger.info("Loading base model: %s  (4-bit NF4)", model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        model.config.use_cache = False

        # pretraining_tp only exists on LLaMA-family models; guard safely
        if hasattr(model.config, "pretraining_tp"):
            model.config.pretraining_tp = 1

        trainable = sum(p.numel() for p in model.parameters())
        logger.info("Base model loaded. Total parameters: %s", f"{trainable:,}")
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

    ``prepare_model_for_kbit_training`` must be called *before*
    ``get_peft_model`` to correctly upcast LayerNorm weights.
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
# Training arguments
# ---------------------------------------------------------------------------

def build_training_arguments(cfg: FinetuneConfig, save_steps: int) -> TrainingArguments:
    """
    Constructs ``TrainingArguments``.

    Key decisions
    -------------
    - ``save_strategy="steps"`` + ``load_best_model_at_end=False`` avoids
      the Trainer trying to load a full HF checkpoint on top of a PEFT model.
    - ``save_steps`` is passed in from ``run_training()`` after being computed
      dynamically so it is always ≤ total training steps.
    - ``optim="paged_adamw_8bit"`` uses bitsandbytes memory-efficient AdamW.
    - ``remove_unused_columns=False`` keeps our pre-built token columns.
    """
    # Create the output directory and its logs sub-dir eagerly so that
    # TrainingArguments never raises FileNotFoundError on logging_dir.
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    logger.info("Output directory ensured: %s", out.resolve())

    return TrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_dir=str(out / "logs"),
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        save_steps=save_steps,
        eval_strategy=cfg.eval_strategy,
        eval_steps=save_steps,          # keep eval cadence in sync with saves
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=cfg.group_by_length,
        report_to=cfg.report_to,
        seed=cfg.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
    )


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def run_training(cfg: FinetuneConfig) -> None:
    """
    Runs the full QLoRA fine-tuning pipeline end-to-end.

    Flow
    ----
    tokeniser → dataset → base model (4-bit) → LoRA wrap →
    Trainer.train() → save_pretrained() (adapter only) → save tokeniser
    """
    logger.info("=== Starting QLoRA Fine-Tuning Pipeline ===")
    logger.info("Base model : %s", cfg.base_model)
    logger.info("Output dir : %s", cfg.output_dir)
    logger.info("Epochs     : %d", cfg.num_epochs)
    logger.info("Max seq len: %d", cfg.max_seq_length)

    # --- Tokeniser -------------------------------------------------------
    tokeniser = load_tokeniser(cfg.base_model, cfg.trust_remote_code)

    # --- Dataset ---------------------------------------------------------
    tok_config = TokenisationConfig(max_seq_length=cfg.max_seq_length)
    dataset_dict = build_dataset(
        dataset_path=Path(cfg.dataset_path),
        tokeniser=tokeniser,
        tokenisation_config=tok_config,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    # --- Model -----------------------------------------------------------
    q_cfg = QuantisationConfig()
    bnb_config = build_bnb_config(q_cfg)
    base_model = load_base_model(cfg.base_model, bnb_config, cfg.trust_remote_code)
    lora_model = apply_lora(base_model, LoRAHyperParams())

    # --- Compute save_steps dynamically ----------------------------------
    # With a small dataset (e.g. 633 samples, batch 2, grad_accum 8) total
    # optimizer steps ≈ 120 over 3 epochs. A hard-coded save_steps=200 would
    # be unreachable → zero checkpoints written. We target ~4 checkpoints per
    # training run, floored at 1 so the value is always valid.
    total_steps: int = max(
        1,
        math.ceil(
            len(dataset_dict["train"])
            / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
        ) * cfg.num_epochs,
    )
    if cfg.save_steps == -1:
        save_steps = max(1, total_steps // 4)
    else:
        save_steps = min(cfg.save_steps, total_steps)  # clamp so it's reachable

    logger.info(
        "Total optimizer steps ≈ %d  |  checkpoint every %d steps",
        total_steps,
        save_steps,
    )

    # --- Training arguments ----------------------------------------------
    training_args = build_training_arguments(cfg, save_steps=save_steps)

    cosine_cb = CosineAnnealingCallback(t_max=total_steps)

    # --- Data collator ---------------------------------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokeniser,
        model=lora_model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
        padding=True,
    )

    # --- Trainer ---------------------------------------------------------
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        callbacks=[cosine_cb],
    )

    logger.info("Launching training …")
    try:
        train_result = trainer.train()
        logger.info("Training complete. Metrics: %s", train_result.metrics)
    except Exception as exc:
        logger.exception("Training loop failed: %s", exc)
        raise

    # --- Save adapter weights + tokeniser --------------------------------
    # PRIMARY: trainer.save_model() is the PEFT-aware path used by the Trainer
    # internally. It calls model.save_pretrained() through the proper interface
    # and is guaranteed to write adapter_config.json + adapter_model weights.
    adapter_save_path = Path(cfg.output_dir)
    logger.info("Saving adapter via trainer.save_model() → %s", adapter_save_path)
    try:
        trainer.save_model(str(adapter_save_path))
        logger.info("trainer.save_model() succeeded.")
    except Exception as exc:
        logger.warning("trainer.save_model() raised %s — falling back to save_pretrained().", exc)
        # FALLBACK: call save_pretrained() directly on the PEFT model
        try:
            lora_model.save_pretrained(str(adapter_save_path))
            logger.info("Fallback lora_model.save_pretrained() succeeded.")
        except Exception as exc2:
            logger.exception("Both save paths failed. Last error: %s", exc2)
            raise exc2

    # Tokeniser is always saved separately (Trainer.save_model skips it)
    try:
        tokeniser.save_pretrained(str(adapter_save_path))
        logger.info("Tokeniser saved to: %s", adapter_save_path)
    except Exception as exc:
        logger.exception("Tokeniser save failed: %s", exc)
        raise

    # --- Verify critical files exist on disk -----------------------------
    required_files = ["adapter_config.json"]
    optional_files = ["adapter_model.safetensors", "adapter_model.bin"]
    missing: list[str] = []

    for fname in required_files:
        fpath = adapter_save_path / fname
        if fpath.exists():
            logger.info("  ✓ %s  (%d bytes)", fname, fpath.stat().st_size)
        else:
            missing.append(fname)
            logger.error("  ✗ MISSING: %s", fname)

    adapter_weights_found = any(
        (adapter_save_path / f).exists() for f in optional_files
    )
    if adapter_weights_found:
        for f in optional_files:
            fp = adapter_save_path / f
            if fp.exists():
                logger.info("  ✓ %s  (%d bytes)", f, fp.stat().st_size)
    else:
        missing.append("adapter_model.safetensors / adapter_model.bin")
        logger.error("  ✗ MISSING: adapter weight file (no .safetensors or .bin found)")

    if missing:
        raise RuntimeError(
            f"Save appeared to succeed but these files are absent in "
            f"{adapter_save_path}: {missing}. "
            f"Check disk space and directory permissions."
        )

    # Log full directory listing for visibility
    saved_files = sorted(adapter_save_path.iterdir())
    logger.info("Contents of %s:", adapter_save_path)
    for p in saved_files:
        logger.info("  %s  (%d bytes)", p.name, p.stat().st_size if p.is_file() else 0)

    # --- Save training metrics -------------------------------------------
    metrics_path = adapter_save_path / "train_metrics.json"
    try:
        with metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(train_result.metrics, fh, indent=2)
        logger.info("Training metrics → %s", metrics_path)
    except OSError as exc:
        logger.warning("Could not write metrics file: %s", exc)

    logger.info("=== Fine-Tuning Pipeline Complete ===")
    logger.info("Adapter saved to: %s", adapter_save_path.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> FinetuneConfig:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Java vulnerability detection."
    )
    parser.add_argument("--base-model", default="bigcode/starcoder2-3b",
                        help="HuggingFace model ID (default: bigcode/starcoder2-3b)")
    parser.add_argument("--dataset", default="Dataset/train_classifier_final.jsonl",
                        help="Path to the JSONL training dataset")
    parser.add_argument("--output-dir", default="./outputs/vuln-lora",
                        help="Directory to save LoRA adapter weights")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (effective batch = batch-size × grad-accum)")
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=-1,
                        help="Save a Trainer checkpoint every N steps. "
                             "Default -1 = auto (25%% of total steps, always reachable).")
    parser.add_argument("--eval-steps", type=int, default=-1,
                        help="Run evaluation every N steps (synced with save-steps by default).")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Disable BF16; use FP16 instead (for non-Ampere GPUs)")
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
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
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
