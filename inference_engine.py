import argparse
import logging
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("inference_engine")

# ── Prompt templates — must match train_classifier_final.jsonl format ─────────
PROMPT_TEMPLATE = (
    "<|instruction|>\n"
    "Analyze the Java code and identify ALL security vulnerabilities. "
    "Return structured JSON only.\n\n"
    "{rag_context}"
    "<|input|>\n{vuln_code}\n\n"
    "<|response|>\n"
)

# Inserted after the instruction line when RAG results are available
RAG_CONTEXT_PREFIX = (
    "Relevant CVE/CWE context (use to inform your classification):\n"
    "{context}\n\n"
)

RESPONSE_MARKER = "<|response|>"


class VulnerabilityInferenceEngine:
    """
    Manages loading a base model along with trained LoRA adapter parameters, 
    and handles deterministic analysis of Java code snippets.
    """
    def __init__(
        self,
        model_id: str,
        adapter_path: Optional[Union[str, Path]] = None,
        load_in_4bit: bool = True
    ) -> None:
        """
        Initializes the inference engine.

        Args:
            model_id: HuggingFace hub id or path to local base model.
            adapter_path: Path to directory containing LoRA adapter weights (optional).
            load_in_4bit: If True, loads the base model in 4-bit precision to reduce VRAM footprint.
        """
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            logger.info(f"Loading tokenizer for model: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load quantized or float16 model based on configuration
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            
            if load_in_4bit and torch.cuda.is_available():
                logger.info("Configuring 4-bit quantization config for inference...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype
                )
                logger.info(f"Loading base model in 4-bit quantization: {model_id}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                logger.info(f"Loading base model in FP16/BF16: {model_id}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=compute_dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    trust_remote_code=True
                )
                if not torch.cuda.is_available():
                    base_model = base_model.to("cpu")

            # Load LoRA adapter if specified
            if self.adapter_path:
                logger.info(f"Applying LoRA adapter checkpoints from: {adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, adapter_path)
            else:
                logger.info("No LoRA adapter specified. Running inference with base model.")
                self.model = base_model
                
            self.model.eval()
            logger.info("Model loading and state initialization complete.")
        except Exception as e:
            logger.error(f"Failed to initialize Inference Engine: {e}", exc_info=True)
            raise

    def analyze_snippet(
        self,
        code: str,
        max_new_tokens: int = 1024,
        rag_context: str = "",
    ) -> dict:
        """
        Runs detection inference on a Java code snippet.

        The model is trained to return a JSON object:
            {
              "vulnerabilities": [
                {
                  "cwe_id": "CWE-89",
                  "cwe_name": "...",
                  "severity": "critical" | "high" | "medium" | "low",
                  "confidence": 0.0–1.0,
                  "location": {"start_line": N, "end_line": N, "function": "..."},
                  "description": "...",
                  "impact": "...",
                  "recommendation": "..."
                }
              ]
            }

        An empty ``vulnerabilities`` list means the code is considered safe.

        Args:
            code          : Java source code snippet.
            max_new_tokens: Cap on generated tokens (384 covers most JSON responses).
            rag_context   : Optional RAG context string injected into the prompt
                            before the code, grounding CWE/severity predictions.

        Returns:
            Dict with keys:
                ``raw``             – raw decoded response string
                ``vulnerabilities`` – parsed list of vulnerability dicts (may be [])
                ``parse_error``     – True if JSON decoding failed
        """
        try:
            # Build RAG context block
            rag_block = ""
            if rag_context and rag_context.strip():
                rag_block = RAG_CONTEXT_PREFIX.format(context=rag_context.strip())

            prompt = PROMPT_TEMPLATE.format(vuln_code=code, rag_context=rag_block)
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids      = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # Deterministic (greedy) decoding
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # Decode full sequence and isolate the response
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Slice off everything up to and including <|response|>
            marker_idx = decoded.find(RESPONSE_MARKER)
            if marker_idx != -1:
                raw_response = decoded[marker_idx + len(RESPONSE_MARKER):].strip()
            else:
                # Fallback: strip the prompt text
                clean_prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                raw_response = decoded[len(clean_prompt):].strip()

            # Attempt to parse JSON
            import json as _json
            import re
            parse_error = False
            vulnerabilities = []
            
            # Clean up potential markdown code blocks
            clean_response = re.sub(r"^```(?:json)?", "", raw_response, flags=re.IGNORECASE).strip()
            clean_response = re.sub(r"```$", "", clean_response).strip()
            
            try:
                # The model may emit extra text after the closing brace;
                # find the outermost JSON object.
                brace_start = clean_response.find("{")
                brace_end   = clean_response.rfind("}")
                if brace_start != -1 and brace_end != -1:
                    json_str = clean_response[brace_start: brace_end + 1]
                    parsed   = _json.loads(json_str)
                    vulnerabilities = parsed.get("vulnerabilities", [])
                else:
                    parse_error = True
            except _json.JSONDecodeError:
                parse_error = True

            return {
                "raw": raw_response,
                "vulnerabilities": vulnerabilities,
                "parse_error": parse_error,
            }
            
        except Exception as e:
            logger.error(f"Error during analysis of code snippet: {e}", exc_info=True)
            return f"Error: Analysis execution failed. Details: {str(e)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and Run Inference Engine")
    parser.add_argument("--model_id", type=str, required=True, help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter weights (optional)")
    parser.add_argument("--snippet_path", type=str, required=True, help="Path to Java code snippet file to analyze")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization loading for base model")
    args = parser.parse_args()

    snippet_file = Path(args.snippet_path)
    if not snippet_file.exists():
        logger.error(f"Specified code snippet path does not exist: {snippet_file}")
        exit(1)

    try:
        logger.info(f"Reading snippet file: {snippet_file}")
        with open(snippet_file, "r", encoding="utf-8") as f:
            code_content = f.read()

        engine = VulnerabilityInferenceEngine(
            model_id=args.model_id,
            adapter_path=args.adapter_path,
            load_in_4bit=not args.no_quant
        )

        logger.info("Running deterministic snippet analysis...")
        result = engine.analyze_snippet(code_content)

        print("\n" + "=" * 60)
        print("VULNERABILITY DETECTION REPORT:")
        print("=" * 60)
        print(result)
        print("=" * 60 + "\n")

    except Exception as err:
        logger.error(f"Inference run failed: {err}")
