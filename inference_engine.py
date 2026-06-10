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
    "### Instruction: Analyze the following Java code snippet and determine whether "
    "it contains a security vulnerability. If a vulnerability is detected, identify "
    "its type, CWE classification, CVE reference (if known), and severity.\n"
    "Output format:\n"
    "  VERDICT: <VULNERABLE | SAFE>\n"
    "  VULNERABILITY_TYPE: <short name, e.g. SQL Injection>\n"
    "  CWE_ID: <e.g. CWE-89>\n"
    "  CVE_REFERENCE: <e.g. CVE-2021-12345 or UNKNOWN>\n"
    "  SEVERITY: <CRITICAL | HIGH | MEDIUM | LOW | UNKNOWN>\n"
    "  DESCRIPTION: <one-sentence explanation>\n\n"
    "{rag_context}"
    "### Input:\n{vuln_code}\n\n"
    "### Response:\n"
)

# Inserted between Instruction and Input when RAG results are available
RAG_CONTEXT_PREFIX = (
    "### Relevant CVE/CWE Context (use to inform your classification):\n"
    "{context}\n\n"
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

    def analyze_snippet(self, code: str, max_new_tokens: int = 256, rag_context: str = "") -> str:
        """
        Executes detection analysis on a Java code snippet using deterministic
        (greedy) decoding.

        The model produces a structured classification response:
            VERDICT: VULNERABLE | SAFE
            VULNERABILITY_TYPE: ...
            CWE_ID: ...
            CVE_REFERENCE: ...
            SEVERITY: ...
            DESCRIPTION: ...

        Args:
            code: Java source code snippet to analyze.
            max_new_tokens: Maximum number of response tokens to generate.
                            256 is sufficient for the structured label output.
            rag_context: Optional pre-formatted RAG context string (CVE/CWE
                         intelligence) to inject into the prompt. When provided,
                         the model uses this grounded knowledge to fill in
                         CVE_REFERENCE, CWE_ID, and SEVERITY rather than
                         relying solely on its trained weights.

        Returns:
            Raw structured detection text from the model.
        """
        try:
            # Inject RAG context block if provided
            rag_block = ""
            if rag_context and rag_context.strip():
                rag_block = RAG_CONTEXT_PREFIX.format(context=rag_context.strip())

            # Construct prompt with optional RAG context
            prompt = PROMPT_TEMPLATE.format(vuln_code=code, rag_context=rag_block)
            
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
