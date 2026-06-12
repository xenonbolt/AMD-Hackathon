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

PROMPT_TEMPLATE = (
    "### Instruction: Analyze the following Java code for security vulnerabilities. "
    "If a vulnerability exists, respond with a JSON object in this exact format:\n"
    '{{"vulnerabilities": [{{"cwe_id": "CWE-XXX", "cwe_name": "...", '
    '"severity": "critical|high|medium|low", "confidence": 0.0-1.0, '
    '"description": "...", "location": {{"start_line": 1, "end_line": 1}}}}]}}\n'
    "If no vulnerability exists, respond with: "
    '{{"vulnerabilities": []}}\n'
    "Respond with JSON only. No explanation outside the JSON.\n\n"
    "### Input:\n{vuln_code}\n\n"
    "### Response:\n"
)

VERIFIER_PROMPT_TEMPLATE = (
    "### Instruction: You are a senior security engineer performing a second-pass verification review. "
    "A vulnerability scanner has flagged the following Java code as potentially vulnerable. "
    "Your job is to carefully re-examine the code and determine whether this is a TRUE POSITIVE "
    "(real vulnerability) or a FALSE POSITIVE (safe code). "
    "Respond ONLY with a valid JSON object — no markdown, no explanation outside JSON.\n\n"
    "### Flagged Finding:\n"
    "- Initial Description: {initial_description}\n\n"
    "### Code Under Review:\n{code}\n\n"
    "### Response (JSON only):\n"
    '{{\n'
    '  "is_vulnerable": true or false,\n'
    '  "cwe_id": "CWE-XXX" or null,\n'
    '  "cwe_name": "Full CWE name" or null,\n'
    '  "severity": "critical" or "high" or "medium" or "low" or null,\n'
    '  "confidence": 0.0 to 1.0,\n'
    '  "reason": "brief explanation of decision"\n'
    '}}\n'
)


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

    def analyze_snippet(self, code: str, max_new_tokens: int = 512) -> str:
        """
        Executes analysis on a Java code snippet using deterministic (greedy) decoding.

        Args:
            code: Java source code snippet to analyze.
            max_new_tokens: Maximum number of response tokens to generate.

        Returns:
            Extracted remediation text / output from the model.
        """
        try:
            # Construct exact prompt template matching training
            prompt = PROMPT_TEMPLATE.format(vuln_code=code)
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
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

            # Decode complete sequence
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Isolate the response segment from prompt text
            # In cases where model generated EOS or standard formatting, slice after response header
            response_marker = "### Response:\n"
            marker_idx = decoded_output.find(response_marker)
            if marker_idx != -1:
                response = decoded_output[marker_idx + len(response_marker):].strip()
            else:
                # Fallback: slice using length of constructed prompt if marker not found cleanly
                # (Skipping special characters or potential formatting variations)
                clean_prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                response = decoded_output[len(clean_prompt):].strip()
                
            return response
            
        except Exception as e:
            logger.error(f"Error during analysis of code snippet: {e}", exc_info=True)
            return f"Error: Analysis execution failed. Details: {str(e)}"

    def verify_finding(
        self,
        code: str,
        initial_description: str = "",
        max_new_tokens: int = 384
    ) -> dict:
        """
        Second-pass verifier: re-analyzes a flagged code snippet to confirm whether
        it is a true positive or false positive, and returns structured CWE metadata.

        Args:
            code: The Java code snippet that was flagged by the first pass.
            initial_description: The description generated by the first-pass detector.
            max_new_tokens: Maximum response tokens (smaller than detect pass — JSON only).

        Returns:
            A dict with keys:
                - is_vulnerable (bool)
                - cwe_id (str | None)
                - cwe_name (str | None)
                - severity (str | None)
                - confidence (float)
                - reason (str)
            On parse failure, returns a safe default that preserves the finding
            with reduced confidence.
        """
        _SAFE_DEFAULT = {
            "is_vulnerable": True,   # conservative: keep finding if verifier fails
            "cwe_id": None,
            "cwe_name": None,
            "severity": None,
            "confidence": 0.4,
            "reason": "Verifier output could not be parsed; finding preserved conservatively."
        }
        try:
            prompt = VERIFIER_PROMPT_TEMPLATE.format(
                initial_description=initial_description or "No description provided.",
                code=code
            )

            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
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

            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Isolate response after prompt marker
            response_marker = "### Response (JSON only):\n"
            marker_idx = decoded_output.find(response_marker)
            if marker_idx != -1:
                raw_response = decoded_output[marker_idx + len(response_marker):].strip()
            else:
                clean_prompt = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                raw_response = decoded_output[len(clean_prompt):].strip()

            # Attempt to parse JSON from the response
            import json
            import re
            parsed = None
            for pattern in [
                r"```json\s*(\{.*?\})\s*```",
                r"```\s*(\{.*?\})\s*```",
                r"(\{[\s\S]*\})"
            ]:
                match = re.search(pattern, raw_response, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(1).strip())
                        break
                    except json.JSONDecodeError:
                        pass

            if parsed is None:
                # Try parsing the raw response directly
                try:
                    parsed = json.loads(raw_response)
                except json.JSONDecodeError:
                    logger.warning(
                        "Verifier: could not parse JSON response; "
                        "falling back to natural-language signal detection."
                    )
                    # Natural-language fallback: scan the raw text for explicit
                    # safe/vulnerable signals so we don't blindly keep everything.
                    lower_raw = raw_response.lower()
                    NL_SAFE_SIGNALS = [
                        "not vulnerable", "no vulnerability", "no vulnerabilities",
                        "false positive", "not a vulnerability", "safe code",
                        "no security issue", "benign", "no exploit", "no issue",
                        "no security concern", "not exploitable"
                    ]
                    NL_VULN_SIGNALS = [
                        "vulnerable", "vulnerability", "vulnerabilities",
                        "cwe-", "injection", "traversal", "forgery",
                        "overflow", "xss", "ssrf", "hardcoded", "insecure",
                        "exploit", "attacker", "arbitrary code", "true positive"
                    ]
                    has_safe = any(s in lower_raw for s in NL_SAFE_SIGNALS)
                    has_vuln = any(v in lower_raw for v in NL_VULN_SIGNALS)

                    if has_safe and not has_vuln:
                        # Model clearly says it's safe
                        return {
                            "is_vulnerable": False,
                            "cwe_id": None,
                            "cwe_name": None,
                            "severity": None,
                            "confidence": 0.7,
                            "reason": "Verifier NL fallback: model response indicated safe/not-vulnerable."
                        }
                    elif has_vuln and not has_safe:
                        # Model clearly says it's vulnerable — extract CWE if present
                        cwe_match = re.search(r'(CWE-\d+)', raw_response, re.IGNORECASE)
                        nl_cwe_id = cwe_match.group(1).upper() if cwe_match else None
                        return {
                            "is_vulnerable": True,
                            "cwe_id": nl_cwe_id,
                            "cwe_name": None,
                            "severity": None,
                            "confidence": 0.6,
                            "reason": "Verifier NL fallback: model response indicated vulnerability."
                        }
                    else:
                        # Ambiguous — use conservative safe default
                        return _SAFE_DEFAULT

            # Normalise and validate fields
            is_vulnerable = bool(parsed.get("is_vulnerable", True))
            cwe_id = parsed.get("cwe_id") or None
            cwe_name = parsed.get("cwe_name") or None
            severity = parsed.get("severity") or None
            if severity is not None:
                severity = severity.lower()
                if severity not in ("critical", "high", "medium", "low"):
                    severity = None
            try:
                confidence = float(parsed.get("confidence", 0.8))
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.8
            reason = str(parsed.get("reason", ""))

            return {
                "is_vulnerable": is_vulnerable,
                "cwe_id": cwe_id,
                "cwe_name": cwe_name,
                "severity": severity,
                "confidence": confidence,
                "reason": reason
            }

        except Exception as e:
            logger.error(f"Error during verifier pass: {e}", exc_info=True)
            return _SAFE_DEFAULT


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

        print("\n" + "=" * 50)
        print("INFERENCE RUN REPORT:")
        print("=" * 50)
        print(result)
        print("=" * 50 + "\n")

    except Exception as err:
        logger.error(f"Inference run failed: {err}")
