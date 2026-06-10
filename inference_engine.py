"""
inference_engine.py
====================
Dedicated inference pipeline for Java security vulnerability detection.

Loads the base causal LM + fine-tuned LoRA adapter, wraps model calls in a
clean ``analyze_snippet`` API, and handles all resource management internally.

Key design choices
------------------
* **Singleton pattern** – the model is loaded once per process via
  ``InferenceEngine.get_instance()`` to avoid redundant GPU memory usage.
* **Deterministic decoding** – temperature=0.1 (near-greedy) for consistent,
  reproducible vulnerability reports.
* **Response extraction** – reliably strips the prompt prefix from the
  generated text so the caller always receives only the JSON response.
* **JSON validation** – normalises and validates the model output before
  returning it to the caller.

Usage (standalone)
------------------
python inference_engine.py --adapter ./outputs/vuln-lora --snippet path/to/File.java

Author : Elite AI Engineering Team
Python : 3.10+
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("inference_engine.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("inference_engine")

# ---------------------------------------------------------------------------
# Prompt template (MUST match data_preparation.py)
# ---------------------------------------------------------------------------
INSTRUCTION_TEXT: str = (
    "Analyze the following Java code for security vulnerabilities. "
    "If one or more vulnerabilities exist, identify each one and return "
    "a structured JSON report containing cwe_id, cwe_name, severity, "
    "confidence, location (start_line, end_line, function), description, "
    "impact, and recommendation. If no vulnerability is present, return "
    '{{"vulnerabilities": []}}.'
)

PROMPT_TEMPLATE: str = (
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{vuln_code}\n\n"
    "### Response:\n"
)

# Fallback response for when the model produces un-parseable output
_EMPTY_VULN_RESPONSE: dict[str, Any] = {"vulnerabilities": []}


# ---------------------------------------------------------------------------
# Inference engine (singleton)
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Thread-safe singleton that encapsulates model + tokeniser lifecycle.

    Parameters
    ----------
    adapter_path : str | Path
        Directory that contains the LoRA adapter weights saved by fine_tune.py.
    base_model_id : str
        HuggingFace model ID of the *original* base model (required to load
        the quantised base before merging the adapter).
    max_new_tokens : int
        Upper bound on generated response tokens (default 1024).
    temperature : float
        Sampling temperature; use ≤0.1 for near-deterministic output.
    """

    _instance: "InferenceEngine | None" = None
    _lock: Lock = Lock()

    def __init__(
        self,
        adapter_path: str | Path,
        base_model_id: str = "bigcode/starcoder2-3b",
        max_new_tokens: int = 1024,
        temperature: float = 0.05,
        trust_remote_code: bool = True,
    ) -> None:
        self.adapter_path = Path(adapter_path)
        self.base_model_id = base_model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.trust_remote_code = trust_remote_code

        self._model: PreTrainedModel | None = None
        self._tokeniser: PreTrainedTokenizerBase | None = None
        self._device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Singleton factory
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        adapter_path: str | Path,
        base_model_id: str = "bigcode/starcoder2-3b",
        **kwargs: Any,
    ) -> "InferenceEngine":
        """
        Returns the singleton ``InferenceEngine``, creating it on first call.

        Thread-safe via double-checked locking.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    engine = cls(adapter_path, base_model_id, **kwargs)
                    engine._load()
                    cls._instance = engine
        return cls._instance

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _build_bnb_config(self) -> BitsAndBytesConfig:
        """4-bit NF4 quantisation config (mirrors fine_tune.py)."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    @staticmethod
    def _resolve_adapter_path(adapter_path: Path) -> Path:
        """
        Returns the directory that actually contains ``adapter_config.json``.

        Resolution order
        ----------------
        1. ``adapter_path`` itself — the normal case after successful training.
        2. Latest ``checkpoint-N`` sub-directory — created by the Trainer's
           step-based saving; used when training was interrupted before the
           explicit ``save_adapter()`` call at the end of ``fine_tune.py``.

        Raises
        ------
        FileNotFoundError
            With a clear, actionable message when no valid adapter can be found.
        """
        # Case 1: root directory has adapter_config.json
        if (adapter_path / "adapter_config.json").exists():
            logger.info("Adapter found at: %s", adapter_path)
            return adapter_path

        # Case 2: look for checkpoint sub-directories
        if adapter_path.is_dir():
            checkpoints = sorted(
                [
                    d for d in adapter_path.iterdir()
                    if d.is_dir()
                    and d.name.startswith("checkpoint-")
                    and (d / "adapter_config.json").exists()
                ],
                key=lambda d: int(d.name.split("-")[-1]),
            )
            if checkpoints:
                latest = checkpoints[-1]
                logger.warning(
                    "adapter_config.json not found in root '%s'. "
                    "Using latest checkpoint: %s",
                    adapter_path,
                    latest,
                )
                return latest

        # Nothing found — provide a clear, actionable error
        contents = (
            [p.name for p in adapter_path.iterdir()]
            if adapter_path.is_dir()
            else ["<directory does not exist>"]
        )
        raise FileNotFoundError(
            f"\n\n{'='*60}\n"
            f"  adapter_config.json NOT FOUND\n"
            f"{'='*60}\n"
            f"  Looked in  : {adapter_path.resolve()}\n"
            f"  Contents   : {contents}\n\n"
            f"  This means fine-tuning has not completed successfully yet,\n"
            f"  or the --adapter path is wrong.\n\n"
            f"  To train the model, run:\n"
            f"    python3 fine_tune.py --output-dir {adapter_path}\n"
            f"{'='*60}\n"
        )

    def _load(self) -> None:
        """
        Loads base model + LoRA adapter and tokeniser into memory.
        Called exactly once by :meth:`get_instance`.
        """
        logger.info("Initialising InferenceEngine …")
        logger.info("  Base model  : %s", self.base_model_id)
        logger.info("  Adapter dir : %s", self.adapter_path)
        logger.info("  Device      : %s", self._device)

        # Resolve (and validate) the adapter path before loading anything
        self.adapter_path = self._resolve_adapter_path(self.adapter_path)

        # --- Tokeniser --------------------------------------------------
        # Prefer loading from the adapter directory (saved there by fine_tune.py);
        # fall back to the base model hub ID.
        tokeniser_source = (
            str(self.adapter_path)
            if (self.adapter_path / "tokenizer_config.json").exists()
            else self.base_model_id
        )
        logger.info("Loading tokeniser from: %s", tokeniser_source)
        try:
            self._tokeniser = AutoTokenizer.from_pretrained(
                tokeniser_source,
                trust_remote_code=self.trust_remote_code,
                use_fast=True,
            )
            if self._tokeniser.pad_token is None:
                self._tokeniser.pad_token = self._tokeniser.eos_token
            self._tokeniser.padding_side = "left"  # Left-pad for generation
        except Exception as exc:
            logger.exception("Tokeniser load failed: %s", exc)
            raise

        # --- Base model (quantised) -------------------------------------
        logger.info("Loading quantised base model …")
        try:
            bnb_cfg = self._build_bnb_config()
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_id,
                quantization_config=bnb_cfg,
                device_map="auto",
                trust_remote_code=self.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_cache=True,
            )
        except Exception as exc:
            logger.exception("Base model load failed: %s", exc)
            raise

        # --- LoRA adapter -----------------------------------------------
        logger.info("Loading LoRA adapter from: %s", self.adapter_path)
        try:
            self._model = PeftModel.from_pretrained(
                base_model,
                str(self.adapter_path),
                is_trainable=False,
            )
            self._model.eval()
        except Exception as exc:
            logger.exception("PEFT adapter load failed: %s", exc)
            raise

        self._is_loaded = True
        logger.info("InferenceEngine ready.")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def build_prompt(java_code: str) -> str:
        """
        Formats the input Java snippet into the training prompt template.

        Returns
        -------
        str
            The full prompt string (without the response).
        """
        return PROMPT_TEMPLATE.format(
            instruction=INSTRUCTION_TEXT,
            vuln_code=java_code.strip(),
        )

    # ------------------------------------------------------------------
    # Response extraction & validation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_response(generated_text: str, prompt: str) -> str:
        """
        Strips the prompt prefix from the raw model output and returns
        only the generated response text.

        Falls back to a regex search for ``{`` if prefix stripping fails.
        """
        # Primary strategy: strip known prompt prefix
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()

        # Secondary strategy: locate "### Response:\n" marker
        marker = "### Response:\n"
        idx = generated_text.rfind(marker)
        if idx != -1:
            return generated_text[idx + len(marker):].strip()

        # Last resort: return everything after the first '{'
        brace_idx = generated_text.find("{")
        if brace_idx != -1:
            return generated_text[brace_idx:].strip()

        return generated_text.strip()

    @staticmethod
    def _parse_and_validate_json(raw: str) -> dict[str, Any]:
        """
        Attempts to parse the model's raw output as JSON.

        Applies light-touch fixes (trailing commas, partial outputs) before
        falling back to the empty vulnerabilities structure.
        """
        # Strip markdown code fences if the model wrapped its JSON
        fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
        if fence_match:
            raw = fence_match.group(1)

        # Remove trailing commas before closing braces/brackets (common LLM artifact)
        raw = re.sub(r",\s*([}\]])", r"\1", raw)

        # Truncate to the first complete JSON object
        depth = 0
        end_idx = len(raw)
        for i, ch in enumerate(raw):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        raw = raw[:end_idx]

        try:
            parsed = json.loads(raw)
            # Ensure the expected top-level key exists
            if "vulnerabilities" not in parsed:
                logger.warning("Response JSON missing 'vulnerabilities' key; wrapping.")
                return {"vulnerabilities": [parsed] if isinstance(parsed, dict) else []}
            return parsed
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse failed (%s); returning empty result.", exc)
            return dict(_EMPTY_VULN_RESPONSE)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_snippet(self, code: str) -> dict[str, Any]:
        """
        Analyses a Java code snippet for security vulnerabilities.

        Parameters
        ----------
        code : str
            Raw Java source code (single method, class, or file).

        Returns
        -------
        dict[str, Any]
            Parsed JSON structure with a ``"vulnerabilities"`` list.
            Each entry contains: ``cwe_id``, ``cwe_name``, ``severity``,
            ``confidence``, ``location``, ``description``, ``impact``,
            ``recommendation``.

        Raises
        ------
        RuntimeError
            If the engine has not been properly initialised.
        """
        if not self._is_loaded or self._model is None or self._tokeniser is None:
            raise RuntimeError("InferenceEngine is not initialised; call get_instance() first.")

        prompt: str = self.build_prompt(code)

        try:
            inputs = self._tokeniser(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=False,
            ).to(self._device)
        except Exception as exc:
            logger.exception("Tokenisation failed: %s", exc)
            return dict(_EMPTY_VULN_RESPONSE)

        generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0.0,
            top_p=0.9 if self.temperature > 0.0 else 1.0,
            repetition_penalty=1.1,
            eos_token_id=self._tokeniser.eos_token_id,
            pad_token_id=self._tokeniser.pad_token_id,
        )

        try:
            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    generation_config=generation_config,
                )
        except Exception as exc:
            logger.exception("Model generation failed: %s", exc)
            return dict(_EMPTY_VULN_RESPONSE)

        # Decode only the newly generated tokens (exclude prompt tokens)
        prompt_len: int = inputs["input_ids"].shape[1]
        new_token_ids = output_ids[0][prompt_len:]
        raw_response: str = self._tokeniser.decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        logger.debug("Raw model response: %s", raw_response[:500])

        result = self._parse_and_validate_json(raw_response)
        vuln_count = len(result.get("vulnerabilities", []))
        logger.info("Snippet analysis complete. Found %d vulnerability/-ies.", vuln_count)
        return result


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def analyze_snippet(
    code: str,
    adapter_path: str | Path = "./outputs/vuln-lora",
    base_model_id: str = "bigcode/starcoder2-3b",
) -> str:
    """
    Module-level wrapper around ``InferenceEngine.analyze_snippet``.

    Returns the result serialised as a pretty-printed JSON string so it
    can be consumed directly by ``scanner.py`` or printed to stdout.

    Parameters
    ----------
    code : str
        Java source code to analyse.
    adapter_path : str | Path
        Path to the LoRA adapter directory.
    base_model_id : str
        HuggingFace model ID of the base model.

    Returns
    -------
    str
        Pretty-printed JSON string.
    """
    engine = InferenceEngine.get_instance(
        adapter_path=adapter_path,
        base_model_id=base_model_id,
    )
    result = engine.analyze_snippet(code)
    return json.dumps(result, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry point for isolated testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference engine smoke-test")
    parser.add_argument(
        "--adapter",
        type=str,
        default="./outputs/vuln-lora",
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="bigcode/starcoder2-3b",
        help="HuggingFace model ID of the base model.",
    )
    parser.add_argument(
        "--snippet",
        type=str,
        default=None,
        help="Path to a .java file to analyse (optional).",
    )
    args = parser.parse_args()

    # Build a minimal test snippet if no file provided
    if args.snippet:
        try:
            code = Path(args.snippet).read_text(encoding="utf-8")
            logger.info("Loaded snippet from: %s", args.snippet)
        except OSError as exc:
            logger.error("Could not read snippet file: %s", exc)
            sys.exit(1)
    else:
        code = (
            'String query = "SELECT * FROM users WHERE id = " + request.getParameter("id");\n'
            "Statement stmt = conn.createStatement();\n"
            "ResultSet rs = stmt.executeQuery(query);"
        )
        logger.info("Using built-in SQL injection test snippet.")

    logger.info("=== Inference Engine Smoke-Test ===")
    try:
        output = analyze_snippet(code, adapter_path=args.adapter, base_model_id=args.base_model)
        print("\n--- Vulnerability Report ---")
        print(output)
        logger.info("Smoke-test PASSED.")
    except Exception as exc:
        logger.exception("Smoke-test FAILED: %s", exc)
        sys.exit(1)
