import platform
# Monkeypatch platform.win32_ver to bypass Windows WMI query errors/hangs
platform.win32_ver = lambda release='', version='', csd='', ptype='': ('10', '10.0.22621', '', 'Multiprocessor Free')

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("fix_engine")

PROMPT_TEMPLATE = (
    "<|instruction|>\nFix the vulnerability in the code\n\n"
    "<|input|>\n{input_json}\n\n"
    "<|response|>\n"
)

class FixInferenceEngine:
    """
    Manages loading a base model in 4-bit precision, overlays a PEFT adapter,
    and handles generating fixes for vulnerabilities in Java files.
    """
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        adapter_path: Optional[Union[str, Path]] = "./adapters_fix",
        load_in_4bit: bool = True
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        is_local = Path(model_id).is_dir()

        try:
            logger.info(f"Loading tokenizer for remediation model: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                local_files_only=is_local
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            bnb_config = None
            device_map = "auto" if torch.cuda.is_available() else None
            torch_dtype = torch.float32

            if torch.cuda.is_available():
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            if load_in_4bit and torch.cuda.is_available():
                logger.info("Configuring 4-bit quantization for FixEngine.")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype
                )

            logger.info(f"Loading base model for FixEngine: {model_id}")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
                trust_remote_code=True,
                local_files_only=is_local
            )

            if self.adapter_path and Path(self.adapter_path).exists():
                logger.info(f"Overlaying PEFT adapter from path: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
                self.using_adapter = True
            else:
                logger.warning("No PEFT adapter specified or found. Running raw base model.")
                self.model = base_model
                self.using_adapter = False

            self.model.config.use_cache = True
            self.model.eval()
            logger.info("FixEngine setup completed successfully.")

        except Exception as err:
            logger.error(f"Failed to initialize FixInferenceEngine: {err}", exc_info=True)
            raise

    def remediate_file_content(
        self,
        raw_code: str,
        vulnerability: Dict[str, Any],
        max_new_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Generates a fix for the given vulnerability in the source code.
        """
        try:
            # Construct input JSON
            input_dict = {
                "cwe_name": vulnerability.get("cwe_name", ""),
                "cwe_id": vulnerability.get("cwe_id", ""),
                "vulnerable_code": raw_code
            }
            if "cve_id" in vulnerability:
                input_dict["cve_id"] = vulnerability["cve_id"]
                
            input_json = json.dumps(input_dict, ensure_ascii=False)
            
            if self.using_adapter:
                prompt = PROMPT_TEMPLATE.format(input_json=input_json)
            else:
                messages = [
                    {"role": "system", "content": "You are an expert security engineer. Fix the vulnerability in the provided code. Output ONLY valid JSON containing an 'explanation' string and a 'fixed_code' string. No markdown formatting, just pure JSON."},
                    {"role": "user", "content": f"Fix the vulnerability in the code:\n{input_json}"}
                ]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    do_sample=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            generated_tokens = outputs[0][len(input_ids[0]):]
            generated_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            eos_token_str = self.tokenizer.eos_token
            if eos_token_str and eos_token_str in generated_response:
                generated_response = generated_response.split(eos_token_str)[0].strip()

            logger.info(f"Fix generation raw output length: {len(generated_response)}")

            # Parse JSON safely
            result = self._parse_json_safely(generated_response)

            # Construct return format
            return {
                "remediationExplanation": result.get("explanation", "No explanation provided."),
                "fullRemediatedContent": result.get("fixed_code", raw_code),
                "remediatedSnippet": result.get("fixed_code", raw_code) # UI can use this as full string for display
            }

        except Exception as err:
            logger.error(f"Remediation failed: {err}", exc_info=True)
            return {
                "error": str(err),
                "remediationExplanation": "An error occurred during generation.",
                "fullRemediatedContent": raw_code,
                "remediatedSnippet": raw_code
            }

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        cleaned_text = text.strip()
        if cleaned_text.startswith("```"):
            first_newline = cleaned_text.find("\n")
            if first_newline != -1:
                cleaned_text = cleaned_text[first_newline:].strip()
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3].strip()

        start_brace = cleaned_text.find("{")
        if start_brace != -1:
            json_candidate = None
            brace_count = 0
            in_string = False
            escape = False
            for idx in range(start_brace, len(cleaned_text)):
                char = cleaned_text[idx]
                if escape:
                    escape = False
                    continue
                if char == '\\':
                    escape = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_candidate = cleaned_text[start_brace:idx + 1]
                            break
            
            if json_candidate is None:
                json_candidate = cleaned_text[start_brace:]
        else:
            json_candidate = cleaned_text

        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode output JSON: {e}. Raw: {json_candidate}")
            return {}


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Run the Remediation Engine locally.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model ID.")
    parser.add_argument("--adapter_path", type=str, default="./adapters_fix", help="Path to PEFT adapters. If not specified, runs base model only.")
    parser.add_argument("--target_path", type=str, required=True, help="Path to the Java file to fix.")
    parser.add_argument("--cwe_id", type=str, default="CWE-89", help="The CWE ID of the vulnerability to fix (e.g., CWE-89).")
    parser.add_argument("--cwe_name", type=str, default="SQL Injection", help="The CWE Name.")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization.")
    
    args = parser.parse_args()

    if not Path(args.target_path).exists():
        print(f"Error: Target file {args.target_path} does not exist.")
        sys.exit(1)

    print(f"Loading FixEngine with model={args.model_id}, adapter={args.adapter_path}, quant={not args.no_quant}")
    try:
        engine = FixInferenceEngine(
            model_id=args.model_id,
            adapter_path=args.adapter_path,
            load_in_4bit=not args.no_quant
        )
    except Exception as e:
        print(f"Failed to load engine: {e}")
        sys.exit(1)

    with open(args.target_path, "r", encoding="utf-8") as f:
        raw_code = f.read()

    vulnerability_details = {
        "cwe_id": args.cwe_id,
        "cwe_name": args.cwe_name
    }

    print(f"\nAnalyzing {args.target_path} for {args.cwe_id}...")
    result = engine.remediate_file_content(raw_code, vulnerability_details)
    
    print("\n" + "="*50)
    print("REMEDIATION EXPLANATION:")
    print("="*50)
    print(result.get("remediationExplanation", "No explanation."))
    
    print("\n" + "="*50)
    print("FIXED CODE:")
    print("="*50)
    print(result.get("fullRemediatedContent", "No fixed code."))
    print("="*50 + "\n")
