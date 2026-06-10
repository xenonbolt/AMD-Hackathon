"""
fine_tune.py
=============
QLoRA fine-tuning pipeline for Java security vulnerability detection.

Design goals
------------
- Robust: the LoRA adapter is saved even if training crashes (try/finally).
- Safe model reference: uses ``trainer.model`` (not the stale pre-training
  handle) to guarantee the post-training weights are what is persisted.
- Auto-detects LoRA target modules from the actual model architecture so
  the training is never silently applied to zero layers.
- Uses a simple DefaultDataCollator because the dataset is pre-tokenised
  and pre-padded inside data_preparation.py; no re-padding is needed.

Usage
-----
python fine_tune.py \\
    --base-model bigcode/starcoder2-3b \\
    --dataset Dataset/train_classifier_final.jsonl \\
    --output-dir ./outputs/vuln-lora \\
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
    DefaultDataCollator,
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
# Known linear layer name patterns across common code LLMs.
# We intersect this list with the actual model's named modules at runtime
# so LoRA is never silently applied to zero layers.
# ---------------------------------------------------------------------------
_CANDIDATE_LINEAR_NAMES: list[str] = [
    # LLaMA / Mistral / StarCoder2 attention
    "q_proj", "k_proj", "v_proj", "o_proj",
    # LLaMA / Mistral / StarCoder2 MLP
    "gate_proj", "up_proj", "down_proj",
    # Falcon / older GPT attention aliases
    "query_key_value", "dense",
    # BERT-style
    "query", "key", "value",
    # Generic catch-all for models that name layers differently
    "c_attn", "c_proj",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    """All training hyper-parameters in one place."""

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
    # -1 → auto-computed at runtime so save_steps ≤ total optimiser steps
    save_steps: int = -1
    save_total_limit: int = 3
    gradient_checkpointing: bool = True
    report_to: str = "none"
    trust_remote_code: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


# ---------------------------------------------------------------------------
# Helpers: target module auto-detection
# ---------------------------------------------------------------------------

def detect_target_modules(model: PreTrainedModel) -> list[str]:
    """
    Returns the intersection of ``_CANDIDATE_LINEAR_NAMES`` with the names of
    ``torch.nn.Linear`` (or quantized equivalents) layers in ``model``.

    This guarantees LoRA is applied to real layers and avoids the silent
    "no parameters to train" trap that occurs when all names miss.

    Raises
    ------
    RuntimeError
        If zero matching modules are found — better to fail loudly than
        produce an adapter with no trainable parameters.
    """
    found: list[str] = []
    for name, module in model.named_modules():
        # Get the leaf module name (last segment after the final '.')
        leaf = name.split(".")[-1]
        if leaf in _CANDIDATE_LINEAR_NAMES and leaf not in found:
            if hasattr(module, "weight"):
                found.append(leaf)

    if not found:
        raise RuntimeError(
            "Could not auto-detect any LoRA target modules. "
            f"Checked candidates: {_CANDIDATE_LINEAR_NAMES}. "
            "Override with --target-modules."
        )

    logger.info("Auto-detected LoRA target modules: %s", found)
    return found


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_bnb_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantisation configuration for bitsandbytes."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokeniser(
    model_id: str,
    trust_remote_code: bool = True,
) -> PreTrainedTokenizerBase:
    """Loads the tokeniser and ensures it has a pad token."""
    logger.info("Loading tokeniser: %s", model_id)
    try:
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
            logger.info("pad_token set to eos_token ('%s').", tok.eos_token)
        tok.padding_side = "right"
        return tok
    except Exception as exc:
        logger.exception("Tokeniser loading failed: %s", exc)
        raise


def load_base_model(
    model_id: str,
    bnb_config: BitsAndBytesConfig,
    trust_remote_code: bool = True,
) -> PreTrainedModel:
    """Loads the causal LM in 4-bit NF4 quantisation with gradient-checkpointing support."""
    logger.info("Loading base model: %s  (4-bit NF4)", model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16,
            use_cache=False,   # must be False for gradient checkpointing
        )
        model.config.use_cache = False
        # pretraining_tp is LLaMA-specific; guard before setting
        if hasattr(model.config, "pretraining_tp"):
            model.config.pretraining_tp = 1
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Base model loaded. Total parameters: %s", f"{total_params:,}")
        return model
    except Exception as exc:
        logger.exception("Base model loading failed: %s", exc)
        raise


def apply_lora(
    model: PreTrainedModel,
    cfg: FinetuneConfig,
    target_modules: list[str],
) -> PreTrainedModel:
    """
    Prepares the quantised model for k-bit training and wraps with LoRA.

    Order matters: ``prepare_model_for_kbit_training`` must run before
    ``get_peft_model`` to correctly upcast LayerNorm weights.
    """
    logger.info("Preparing model for k-bit training …")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=cfg.gradient_checkpointing,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=target_modules,
        inference_mode=False,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError(
            "LoRA produced 0 trainable parameters. "
            "The target_modules list likely does not match any layer in the model. "
            f"target_modules={target_modules}"
        )

    logger.info("Trainable parameters: %s", f"{trainable:,}")
    return model


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _unwrap_model(model: Any) -> Any:
    """
    Unwraps DataParallel / DistributedDataParallel / other wrappers so we
    always call ``save_pretrained`` on the actual PeftModel instance.
    """
    if hasattr(model, "module"):
        return _unwrap_model(model.module)
    return model


def save_adapter(model: Any, tokeniser: PreTrainedTokenizerBase, output_dir: Path) -> None:
    """
    Saves the LoRA adapter weights and tokeniser to ``output_dir``.

    Uses the unwrapped model to avoid issues with DataParallel wrappers.
    After saving, verifies the critical files are present on disk and raises
    ``RuntimeError`` if anything is missing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = _unwrap_model(model)

    logger.info("Saving LoRA adapter to: %s", output_dir.resolve())
    try:
        unwrapped.save_pretrained(str(output_dir))
        logger.info("save_pretrained() completed.")
    except Exception as exc:
        logger.exception("save_pretrained() failed: %s", exc)
        raise

    logger.info("Saving tokeniser to: %s", output_dir.resolve())
    try:
        tokeniser.save_pretrained(str(output_dir))
        logger.info("Tokeniser saved.")
    except Exception as exc:
        logger.exception("Tokeniser save failed: %s", exc)
        raise

    # ----- Verify critical files exist -----------------------------------
    required = ["adapter_config.json"]
    weight_options = ["adapter_model.safetensors", "adapter_model.bin"]

    missing: list[str] = []
    for fname in required:
        fp = output_dir / fname
        if fp.exists() and fp.stat().st_size > 0:
            logger.info("  ✓ %-40s  %d bytes", fname, fp.stat().st_size)
        else:
            missing.append(fname)
            logger.error("  ✗ MISSING or EMPTY: %s", fname)

    weight_found = any((output_dir / w).exists() for w in weight_options)
    if weight_found:
        for w in weight_options:
            fp = output_dir / w
            if fp.exists():
                logger.info("  ✓ %-40s  %d bytes", w, fp.stat().st_size)
    else:
        missing.append("adapter_model.safetensors or adapter_model.bin")
        logger.error("  ✗ MISSING: no adapter weight file found")

    if missing:
        raise RuntimeError(
            f"Adapter save verification failed — missing: {missing}. "
            "Check disk space and write permissions on: "
            + str(output_dir.resolve())
        )

    # ----- Full directory listing ----------------------------------------
    logger.info("Directory listing for %s:", output_dir)
    for p in sorted(output_dir.iterdir()):
        size = p.stat().st_size if p.is_file() else 0
        logger.info("  %-45s  %d bytes", p.name, size)


# ---------------------------------------------------------------------------
# Cosine Annealing LR callback
# ---------------------------------------------------------------------------

class CosineAnnealingCallback(TrainerCallback):
    """Attaches a PyTorch CosineAnnealingLR scheduler to the Trainer's optimizer."""

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
            logger.warning("CosineAnnealingCallback: optimizer not available yet.")
            return
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=self.eta_min
        )
        logger.info("CosineAnnealingLR attached (T_max=%d, eta_min=%.2e).", self.t_max, self.eta_min)

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
# Training
# ---------------------------------------------------------------------------

def run_training(cfg: FinetuneConfig) -> None:
    """
    Full QLoRA fine-tuning pipeline.

    Adapter save is wrapped in ``try/finally`` so weights are persisted even
    if training is interrupted (e.g. by OOM or KeyboardInterrupt).
    """
    logger.info("=== Starting QLoRA Fine-Tuning Pipeline ===")
    logger.info("Base model : %s", cfg.base_model)
    logger.info("Output dir : %s", cfg.output_dir)
    logger.info("Epochs     : %d", cfg.num_epochs)
    logger.info("Max seq len: %d", cfg.max_seq_length)

    # ------------------------------------------------------------------
    # 1. Create output directory early — fail immediately if not writable
    # ------------------------------------------------------------------
    out_dir = Path(cfg.output_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "logs").mkdir(parents=True, exist_ok=True)
        # Write-permission smoke test
        test_file = out_dir / ".write_test"
        test_file.write_text("ok")
        test_file.unlink()
        logger.info("Output directory OK: %s", out_dir.resolve())
    except OSError as exc:
        logger.error("Cannot write to output directory %s: %s", out_dir, exc)
        raise

    # ------------------------------------------------------------------
    # 2. Tokeniser
    # ------------------------------------------------------------------
    tokeniser = load_tokeniser(cfg.base_model, cfg.trust_remote_code)

    # ------------------------------------------------------------------
    # 3. Dataset
    # ------------------------------------------------------------------
    tok_cfg = TokenisationConfig(max_seq_length=cfg.max_seq_length)
    dataset_dict = build_dataset(
        dataset_path=Path(cfg.dataset_path),
        tokeniser=tokeniser,
        tokenisation_config=tok_cfg,
        val_split=cfg.val_split,
        seed=cfg.seed,
    )

    # ------------------------------------------------------------------
    # 4. Base model + LoRA
    # ------------------------------------------------------------------
    bnb_config = build_bnb_config()
    base_model = load_base_model(cfg.base_model, bnb_config, cfg.trust_remote_code)

    # Auto-detect target modules from the actual model graph
    target_modules = detect_target_modules(base_model)

    lora_model = apply_lora(base_model, cfg, target_modules)

    # ------------------------------------------------------------------
    # 5. Compute save_steps dynamically — must be ≤ total optimiser steps
    # ------------------------------------------------------------------
    steps_per_epoch = math.ceil(
        len(dataset_dict["train"])
        / (cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps)
    )
    total_steps = max(1, steps_per_epoch * cfg.num_epochs)

    if cfg.save_steps <= 0:
        # Target ~4 checkpoints per run, at minimum every epoch
        save_steps = max(1, min(steps_per_epoch, total_steps // 4))
    else:
        save_steps = min(cfg.save_steps, total_steps)

    logger.info(
        "Steps: %d/epoch × %d epochs = %d total  |  checkpoint every %d steps",
        steps_per_epoch, cfg.num_epochs, total_steps, save_steps,
    )

    # ------------------------------------------------------------------
    # 6. TrainingArguments
    # ------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(out_dir),
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
        logging_dir=str(out_dir / "logs"),
        logging_steps=cfg.logging_steps,
        # Step-based saving avoids the load_best_model_at_end+PEFT conflict
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=False,   # NEVER True with PEFT adapters
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        report_to=cfg.report_to,
        seed=cfg.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,    # keep our pre-built token columns
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
    )

    cosine_cb = CosineAnnealingCallback(t_max=total_steps)

    # ------------------------------------------------------------------
    # 7. Data collator
    # Data is already padded to max_seq_length inside data_preparation.py.
    # DefaultDataCollator simply stacks tensors — no re-padding or shape
    # changes. This avoids DataCollatorForSeq2Seq trying to re-pad
    # already-padded sequences and potentially causing shape mismatches.
    # ------------------------------------------------------------------
    data_collator = DefaultDataCollator(return_tensors="pt")

    # ------------------------------------------------------------------
    # 8. Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        data_collator=data_collator,
        callbacks=[cosine_cb],
    )

    # ------------------------------------------------------------------
    # 9. Train — always attempt to save adapter in finally block so
    #    weights are persisted even on OOM or keyboard interrupt.
    # ------------------------------------------------------------------
    train_result: Any = None
    logger.info("Launching training …")
    try:
        train_result = trainer.train()
        logger.info("Training complete. Metrics: %s", train_result.metrics)
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user — saving partial adapter …")
    except Exception as exc:
        logger.exception("Training loop raised an exception: %s", exc)
        logger.warning("Attempting to save partial adapter despite error …")
    finally:
        # Always save — use trainer.model (the live reference post-training),
        # not the original lora_model variable (may be stale after wrapping).
        logger.info("--- SAVE PHASE ---")
        try:
            save_adapter(trainer.model, tokeniser, out_dir)
        except Exception as save_exc:
            logger.exception("Adapter save failed in finally block: %s", save_exc)
            raise

    # ------------------------------------------------------------------
    # 10. Save training metrics
    # ------------------------------------------------------------------
    if train_result is not None:
        metrics_path = out_dir / "train_metrics.json"
        try:
            with metrics_path.open("w", encoding="utf-8") as fh:
                json.dump(train_result.metrics, fh, indent=2)
            logger.info("Training metrics → %s", metrics_path)
        except OSError as exc:
            logger.warning("Could not write metrics file: %s", exc)

    logger.info("=== Fine-Tuning Pipeline Complete ===")
    logger.info("Adapter location: %s", out_dir.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> FinetuneConfig:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Java vulnerability detection."
    )
    parser.add_argument("--base-model", default="bigcode/starcoder2-3b",
                        help="HuggingFace model ID")
    parser.add_argument("--dataset", default="Dataset/train_classifier_final.jsonl")
    parser.add_argument("--output-dir", default="./outputs/vuln-lora",
                        help="Directory to save LoRA adapter weights (created if absent)")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device train batch size")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--val-split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=-1,
                        help="Checkpoint every N steps. -1 = auto (25%% of total steps)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--no-bf16", action="store_true",
                        help="Use FP16 instead of BF16 (for non-Ampere GPUs)")
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        bf16=not args.no_bf16,
        fp16=args.no_bf16,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    try:
        run_training(cfg)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        sys.exit(0)
    except Exception:
        sys.exit(1)
