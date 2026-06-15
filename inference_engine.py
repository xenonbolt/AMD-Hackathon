import platform
# Monkeypatch platform.win32_ver to bypass Windows WMI query errors/hangs
platform.win32_ver = lambda release='', version='', csd='', ptype='': ('10', '10.0.22621', '', 'Multiprocessor Free')

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("inference_engine")

PROMPT_TEMPLATE = (
    "<|instruction|>\nAnalyze the Java code and identify ALL security vulnerabilities. Return structured JSON only.\n\n"
    "<|input|>\n{raw_code}\n\n"
    "<|response|>\n"
)


class VulnerabilityInferenceEngine:
    """
    Manages loading a base model in 4-bit precision, overlays a PEFT adapter,
    and handles deterministic vulnerability analysis of Java files.
    """
    def __init__(
        self,
        model_id: str,
        adapter_path: Optional[Union[str, Path]] = None,
        load_in_4bit: bool = True
    ) -> None:
        """
        Initializes the inference engine by loading the model and tokenizer.

        Args:
            model_id: HuggingFace model hub ID or local path to base model.
            adapter_path: Local path to trained PEFT adapter checkpoints (optional).
            load_in_4bit: Whether to load the base model in 4-bit precision (requires CUDA).
        """
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Check if model_id is a local directory containing a merged model
        is_local = Path(model_id).is_dir()

        try:
            logger.info(f"Loading tokenizer for model: {model_id} (local: {is_local})")
            if "deepseek" in model_id.lower():
                logger.info("DeepSeek model detected. Loading tokenizer via PreTrainedTokenizerFast to avoid LlamaTokenizer space bug.")
                from transformers import PreTrainedTokenizerFast
                self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=is_local
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=is_local
                )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                if self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Determine quantization and device map
            bnb_config = None
            device_map = "auto" if torch.cuda.is_available() else None
            torch_dtype = torch.float32

            if torch.cuda.is_available():
                torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

            if load_in_4bit and torch.cuda.is_available():
                logger.info(f"Configuring 4-bit BitsAndBytes quantization. Compute dtype: {torch_dtype}")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch_dtype
                )
            else:
                logger.warning(f"BitsAndBytes 4-bit is disabled or CUDA is unavailable. Loading base model in precision: {torch_dtype} (device_map={device_map})")

            logger.info(f"Loading base/merged model: {model_id} (local: {is_local})")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
                trust_remote_code=True,
                local_files_only=is_local
            )

            # Apply PEFT adapter if provided
            if self.adapter_path:
                logger.info(f"Overlaying PEFT adapter from path: {self.adapter_path}")
                self.model = PeftModel.from_pretrained(base_model, str(self.adapter_path))
            else:
                if is_local:
                    logger.info("No adapter path specified. Using local model directly (assumed merged/standalone).")
                else:
                    logger.warning("No PEFT adapter specified. Running raw base model.")
                self.model = base_model

            # Explicitly force KV caching to be enabled for fast autoregressive generation
            self.model.config.use_cache = True
            self.model.eval()
            logger.info("Inference engine model configuration and setup completed successfully.")

        except Exception as err:
            logger.error(f"Failed to initialize VulnerabilityInferenceEngine: {err}", exc_info=True)
            raise

    def analyze_file_content(
        self,
        raw_code: str,
        max_new_tokens: int = 1024,
        file_line_count: Optional[int] = None,
        file_path: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Wraps raw Java code into the specialized token prompt layout, generates
        a vulnerability report, parses JSON output, and validates results.

        Args:
        raw_code: The raw Java source code content string.
            max_new_tokens: Maximum tokens allowed to be generated by the model.
            file_line_count: Actual number of lines in the source file (for validation).
            file_path: File path string (for logging context).
            min_confidence: Minimum confidence threshold (0.0 to 1.0).

        Returns:
            A parsed JSON dictionary of vulnerabilities or an error dictionary.
        """
        log_ctx = file_path or "<code block>"

        # Try greedy decoding first, then sampling as fallback
        generation_configs = [
            {  # Attempt 1: Greedy with repetition penalty
                "do_sample": False,
                "repetition_penalty": 1.2,
            },
            {  # Attempt 2: Sampling fallback (different token distribution)
                "do_sample": True,
                "temperature": 0.3,
                "top_p": 0.9,
                "repetition_penalty": 1.15,
            },
        ]

        last_error = None
        for attempt_idx, gen_config in enumerate(generation_configs, 1):
            try:
                # 1. Wrap raw code in Prompt structure
                prompt = PROMPT_TEMPLATE.format(raw_code=raw_code)

                # Tokenize input prompt
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to(self.model.device)

                # 2. Run generation
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        use_cache=True,
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        **gen_config
                    )

                # 3. Extract and decode ONLY the generated tokens
                generated_tokens = outputs[0][len(input_ids[0]):]
                generated_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                # Ensure we stop if the tokenizer EOS token is in string representation
                eos_token_str = self.tokenizer.eos_token
                if eos_token_str and eos_token_str in generated_response:
                    generated_response = generated_response.split(eos_token_str)[0].strip()

                # 3b. Sanitize any residual repetition patterns in the output
                generated_response = self._sanitize_repetition(generated_response)

                # 3c. Debug: log the raw model output before JSON parsing
                logger.info(f"[{log_ctx}] Attempt {attempt_idx} raw output ({len(generated_response)} chars): {generated_response[:500]}")
                result = self._parse_json_safely(generated_response)

                # 5. If parsing failed and we have more attempts, retry
                if "error" in result and attempt_idx < len(generation_configs):
                    last_error = result
                    logger.warning(f"Attempt {attempt_idx} failed for {log_ctx}. Retrying with sampling...")
                    continue

                # 6. Validate and clean vulnerability entries
                if "vulnerabilities" in result:
                    result["vulnerabilities"] = self._validate_vulnerabilities(
                        result["vulnerabilities"], file_line_count, raw_code, min_confidence
                    )

                return result

            except Exception as err:
                logger.error(f"Inference attempt {attempt_idx} failed on {log_ctx}: {err}", exc_info=True)
                last_error = {
                    "vulnerabilities": [],
                    "error": f"Inference execution failed: {str(err)}"
                }

        return last_error or {"vulnerabilities": [], "error": "All inference attempts failed"}

    def _sanitize_repetition(self, text: str) -> str:
        """
        Detects and truncates degenerate repetition patterns in model output.
        For example: 'Exploitation of CWE-287 (Exploitation of CWE-287 (Exploitation of ...'
        will be truncated to just 'Exploitation of CWE-287'.

        IMPORTANT: This is conservative — it only activates on 5+ consecutive
        repeats and skips text that looks like valid JSON to avoid false positives
        on structured vulnerability reports with similar-looking entries.
        """
        if not text or len(text) < 100:
            return text

        # Skip sanitization entirely if the text looks like valid structured JSON
        # (structured JSON with multiple vulnerability entries can trigger false positives)
        stripped = text.strip()
        if stripped.startswith('{') and ('"vulnerabilities"' in stripped or '"cwe_id"' in stripped):
            return text

        # Strategy: find any substring of length 20-80 that repeats 5+ times consecutively
        # (raised from 3x to 5x to avoid false positives on structured output)
        min_repeats = 5
        for pattern_len in range(20, min(81, len(text) // min_repeats + 1)):
            for start in range(len(text) - pattern_len * min_repeats + 1):
                candidate = text[start:start + pattern_len]
                # Skip candidates that are mostly whitespace/structural JSON chars
                if candidate.strip() in ('', '{', '}', '[', ']', ','):
                    continue
                repeat_count = 1
                pos = start + pattern_len
                while pos + pattern_len <= len(text) and text[pos:pos + pattern_len] == candidate:
                    repeat_count += 1
                    pos += pattern_len
                if repeat_count >= min_repeats:
                    logger.warning(
                        f"Detected degenerate repetition ({repeat_count}x): '{candidate[:50]}...'. Truncating."
                    )
                    # Keep everything before the first repetition, plus one instance
                    return text[:start + pattern_len].strip()

        return text

    def _attempt_json_repair(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Last-resort attempt to repair truncated JSON by closing open delimiters
        in the correct nesting order and handling trailing commas.
        Returns parsed dict on success, None on failure.
        """
        if not text or '{' not in text:
            return None

        # Find the start of JSON
        start_brace = text.find('{')
        repaired = text[start_brace:]

        # Use a stack to track delimiter nesting order
        delimiter_stack = []  # stores '{' or '['
        in_string = False
        escape = False

        for char in repaired:
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
                if char in ('{', '['):
                    delimiter_stack.append(char)
                elif char == '}' and delimiter_stack and delimiter_stack[-1] == '{':
                    delimiter_stack.pop()
                elif char == ']' and delimiter_stack and delimiter_stack[-1] == '[':
                    delimiter_stack.pop()

        # Close any open string
        if in_string:
            repaired += '"'

        # Strip trailing commas and whitespace before closing
        repaired = repaired.rstrip()
        if repaired.endswith(','):
            repaired = repaired[:-1]

        # Close open delimiters in reverse nesting order (stack-based)
        while delimiter_stack:
            opener = delimiter_stack.pop()
            # Strip any trailing comma before each closer
            repaired = repaired.rstrip()
            if repaired.endswith(','):
                repaired = repaired[:-1]
            if opener == '{':
                repaired += '}'
            else:
                repaired += ']'

        try:
            parsed = json.loads(repaired)
            logger.info("Successfully repaired truncated JSON output.")
            return parsed
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _normalize_json_keys(obj: Any) -> Any:
        """
        Recursively strips leading/trailing whitespace from JSON keys.
        Handles the model's tendency to produce ' cwe_name' instead of 'cwe_name'.
        """
        if isinstance(obj, dict):
            return {k.strip(): VulnerabilityInferenceEngine._normalize_json_keys(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [VulnerabilityInferenceEngine._normalize_json_keys(item) for item in obj]
        return obj

    def _parse_json_safely(self, text: str) -> Dict[str, Any]:
        """
        Cleans markdown wrappers and parses JSON blocks from raw text responses.
        Falls back to JSON repair if initial parsing fails.
        """
        cleaned_text = text.strip()

        # Remove markdown code blocks if the model wrapped the JSON output
        if cleaned_text.startswith("```"):
            first_newline = cleaned_text.find("\n")
            if first_newline != -1:
                cleaned_text = cleaned_text[first_newline:].strip()
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3].strip()

        # Extract the balanced outer JSON block
        json_candidate = None
        start_brace = cleaned_text.find("{")
        if start_brace != -1:
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
            json_candidate = cleaned_text

        # Try parsing the extracted JSON (or repaired JSON as fallback)
        for attempt_label, candidate in [("direct", json_candidate), ("repair", None)]:
            if attempt_label == "repair":
                # Fallback: attempt JSON repair on truncated output
                logger.warning(f"Initial JSON parse failed. Attempting repair...")
                repaired_text = self._attempt_json_repair(cleaned_text)
                if repaired_text is None:
                    break
                candidate = repaired_text  # _attempt_json_repair already returns a dict
                # If repair returned a dict directly, use it
                if isinstance(candidate, dict):
                    candidate = self._normalize_json_keys(candidate)
                    if "vulnerabilities" not in candidate:
                        candidate = {"vulnerabilities": [], "raw_output": candidate}
                    return candidate
                continue

            try:
                parsed_data: Dict[str, Any] = json.loads(candidate)
                parsed_data = self._normalize_json_keys(parsed_data)
                # Standardize returned output layout
                if "vulnerabilities" not in parsed_data:
                    # Check alternative keys the model might have used
                    for alt_key in ("findings", "issues", "results", "vulnerability_list"):
                        if alt_key in parsed_data and isinstance(parsed_data[alt_key], list):
                            parsed_data["vulnerabilities"] = parsed_data.pop(alt_key)
                            break
                    else:
                        # If the parsed data is itself a list, treat it as vulnerability entries
                        if isinstance(parsed_data, list):
                            parsed_data = {"vulnerabilities": parsed_data}
                        # If the parsed data looks like a single vulnerability entry
                        elif isinstance(parsed_data, dict) and ("cwe_id" in parsed_data or "severity" in parsed_data or "type" in parsed_data):
                            parsed_data = {"vulnerabilities": [parsed_data]}
                        else:
                            parsed_data = {"vulnerabilities": [], "raw_output": parsed_data}
                return parsed_data
            except json.JSONDecodeError as decode_err:
                if attempt_label == "direct":
                    continue  # Will try repair next

        # Both attempts failed
        logger.error(f"JSON parsing and repair both failed. Raw output: {text}")
        return {
            "vulnerabilities": [],
            "error": f"Invalid JSON format",
            "raw_response": text
        }

    @staticmethod
    def _clean_cwe_name(name: str) -> str:
        """
        Cleans up repetitive/nested CWE name patterns the model tends to produce.
        E.g. 'Exploitation of CWE-287 (Exploitation of CWE-287 (Exploitation of CWE-287))'
        becomes 'Exploitation of CWE-287'.
        """
        if not name:
            return name
        # Detect nested parenthetical repetition:  "X (X (X))" -> "X"
        # Look for pattern where the content before '(' repeats inside
        paren_start = name.find('(')
        if paren_start > 0:
            prefix = name[:paren_start].strip()
            inner = name[paren_start + 1:].rstrip(')')
            if prefix and inner.startswith(prefix):
                return prefix
        # Also detect "Exploitation of CWE-XXX" prefix duplication
        match = re.match(r'^(Exploitation of )(CWE-\d+)\b', name)
        if match:
            return match.group(2)  # Return just the CWE ID
        return name

    @staticmethod
    def _infer_cwe(vuln: Dict[str, Any], code_content: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """
        Dynamically infers CWE ID and standardized name based on keywords in the finding
        or by scanning the code content against known sink heuristics.
        """
        keyword_map = {
            "path": ("CWE-22", "Relative Path Traversal"),
            "file-forbidden": ("CWE-22", "Relative Path Traversal"),
            "directory": ("CWE-22", "Relative Path Traversal"),
            "sql": ("CWE-89", "SQL Injection"),
            "xss": ("CWE-79", "Cross-Site Scripting"),
            "cross-site": ("CWE-79", "Cross-Site Scripting"),
            "command": ("CWE-78", "OS Command Injection"),
            "ldap": ("CWE-90", "LDAP Injection"),
            "xpath": ("CWE-643", "XPath Injection"),
            "hardcoded": ("CWE-321", "Use of Hard-coded Cryptographic Key"),
            "secret": ("CWE-321", "Use of Hard-coded Cryptographic Key"),
            "cleartext": ("CWE-319", "Cleartext Transmission of Sensitive Information"),
            "credential": ("CWE-522", "Insufficiently Protected Credentials"),
            "auth": ("CWE-522", "Insufficiently Protected Credentials"),
            "redirect": ("CWE-601", "URL Redirection to Untrusted Site ('Open Redirect')"),
            "resource": ("CWE-400", "Uncontrolled Resource Consumption"),
            "permission": ("CWE-276", "Incorrect Default Permissions")
        }

        search_text = " ".join([
            str(vuln.get("cwe_name", "")),
            str(vuln.get("type", "")),
            str(vuln.get("name", "")),
            str(vuln.get("message", "")),
            str(vuln.get("description", ""))
        ]).lower()

        # 1. Keyword mapping
        for keyword, (cwe_id, std_name) in keyword_map.items():
            if keyword in search_text:
                return cwe_id, std_name

        # 2. Code heuristics fallback
        if code_content:
            sink_heuristics = {
                "CWE-78": (r"(Runtime\.getRuntime\(\)\.exec\(|new ProcessBuilder\(|ProcessBuilder\()", "OS Command Injection"),
                "CWE-20": (r"(Runtime\.getRuntime\(\)\.exec\(|new ProcessBuilder\(|ProcessBuilder\()", "Improper Input Validation"),
                "CWE-22": (r"(new File\(|new FileReader\(|new FileInputStream\(|new FileOutputStream\(|Paths\.get\()", "Relative Path Traversal"),
                "CWE-276": (r"(new File\(|new FileReader\(|createTempFile\(|setExecutable\(|setReadable\(|FileOutputStream\()", "Incorrect Default Permissions"),
                "CWE-89": (r"(executeQuery\(|prepareStatement\(|executeUpdate\(|Statement |createStatement\()", "SQL Injection"),
                "CWE-79": (r"(getWriter\(\)\.print|out\.println)", "Cross-Site Scripting"),
                "CWE-90": (r"(InitialDirContext|search\(|lookup\()", "LDAP Injection"),
                "CWE-643": (r"(XPath |evaluate\(|compile\()", "XPath Injection"),
                "CWE-321": (r"(SecretKeySpec|AES|DES)", "Use of Hard-coded Cryptographic Key"),
                "CWE-319": (r"(HttpURLConnection|Socket |http://|ftp://|SocketChannel)", "Cleartext Transmission of Sensitive Information"),
                "CWE-522": (r"(getConnection\(|DriverManager|password|login)", "Insufficiently Protected Credentials"),
                "CWE-601": (r"(sendRedirect\(|setHeader\(\"Location\")", "URL Redirection to Untrusted Site ('Open Redirect')"),
                "CWE-400": (r"(Thread\.sleep\(|readLine\(\)|while \(|for \()", "Uncontrolled Resource Consumption"),
            }
            for cwe_id, (pattern, std_name) in sink_heuristics.items():
                if re.search(pattern, code_content):
                    return cwe_id, std_name

        return None, None

    @staticmethod
    def _validate_vulnerabilities(
        vulns: List[Dict[str, Any]],
        file_line_count: Optional[int] = None,
        code_content: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Validates and cleans each vulnerability entry:
        - Clamps hallucinated line numbers to actual file size
        - Cleans repetitive CWE names
        - Ensures required fields have sensible defaults
        - Keeps entries even if cwe_id/cwe_name is missing (fills defaults)
        - Drops entries that do not meet min_confidence
        - Applies code-aware post-validation: drops findings if known required sinks are missing in code.
        """
        if not isinstance(vulns, list):
            return []

        validated = []
        dropped = 0
        for vuln in vulns:
            if not isinstance(vuln, dict):
                dropped += 1
                continue

            # If the entry is completely empty (no keys at all), skip it
            if not vuln:
                dropped += 1
                continue

            # Fill in defaults for missing cwe fields instead of dropping the entry.
            # The model may produce valid findings with slightly different key names.
            if not vuln.get("cwe_id") and not vuln.get("cwe_name"):
                # Try to infer from other keys the model might have used
                for alt_key in ("type", "vulnerability", "name", "category", "vuln_type"):
                    if alt_key in vuln:
                        vuln["cwe_name"] = str(vuln[alt_key])
                        break
                else:
                    # Still keep it — just label it unknown
                    vuln.setdefault("cwe_name", "Unknown Vulnerability")
                logger.debug(f"Vulnerability entry missing cwe_id/cwe_name, filled default: {vuln}")

            # Properly infer CWE using our generalized heuristics and keyword mapping
            if not vuln.get("cwe_id"):
                inferred_id, inferred_name = VulnerabilityInferenceEngine._infer_cwe(vuln, code_content)
                if inferred_id:
                    vuln["cwe_id"] = inferred_id
                    vuln["cwe_name"] = inferred_name

            # Clean CWE name of repetitive patterns
            if "cwe_name" in vuln:
                vuln["cwe_name"] = VulnerabilityInferenceEngine._clean_cwe_name(vuln["cwe_name"])

            # Validate and clamp line numbers against actual file size
            location = vuln.get("location", {})
            if isinstance(location, dict) and file_line_count is not None:
                for key in ("start_line", "end_line", "line"):
                    if key in location:
                        try:
                            val = int(location[key])
                            location[key] = max(1, min(val, file_line_count))
                        except (ValueError, TypeError):
                            del location[key]
                # Ensure start_line <= end_line
                if "start_line" in location and "end_line" in location:
                    if location["start_line"] > location["end_line"]:
                        location["start_line"], location["end_line"] = location["end_line"], location["start_line"]

            # Ensure severity is a valid value
            severity = str(vuln.get("severity", "medium")).lower()
            if severity not in ("critical", "high", "medium", "low", "info"):
                severity = "medium"
            vuln["severity"] = severity

            # Ensure confidence is a valid float in [0, 1]
            confidence = vuln.get("confidence")
            if confidence is not None:
                try:
                    confidence = float(confidence)
                    vuln["confidence"] = round(max(0.0, min(1.0, confidence)), 3)
                except (ValueError, TypeError):
                    vuln.pop("confidence", None)

            if "confidence" in vuln and vuln["confidence"] < min_confidence:
                dropped += 1
                logger.info(f"Dropped finding due to low confidence ({vuln['confidence']} < {min_confidence}): {vuln.get('cwe_name')}")
                continue

            # Code-Aware Post Validation (SINK_HEURISTICS)
            if code_content and vuln.get("cwe_id"):
                cwe = vuln["cwe_id"].upper()
                sink_heuristics = {
                    "CWE-78": r"(Runtime\.getRuntime\(\)\.exec\(|new ProcessBuilder\(|ProcessBuilder\()",
                    "CWE-20": r"(Runtime\.getRuntime\(\)\.exec\(|new ProcessBuilder\(|ProcessBuilder\()",
                    "CWE-22": r"(new File\(|new FileReader\(|new FileInputStream\(|new FileOutputStream\(|Paths\.get\()",
                    "CWE-276": r"(new File\(|new FileReader\(|createTempFile\(|setExecutable\(|setReadable\(|FileOutputStream\()",
                    "CWE-89": r"(executeQuery\(|prepareStatement\(|executeUpdate\(|Statement |createStatement\()",
                    "CWE-79": r"(getWriter\(\)\.print|out\.println)",
                    "CWE-90": r"(InitialDirContext|search\(|lookup\()",
                    "CWE-643": r"(XPath |evaluate\(|compile\()",
                    "CWE-321": r"(SecretKeySpec|AES|DES)",
                    "CWE-319": r"(HttpURLConnection|Socket |http://|ftp://|SocketChannel)",
                    "CWE-522": r"(getConnection\(|DriverManager|password|login)",
                    "CWE-601": r"(sendRedirect\(|setHeader\(\"Location\")",
                    "CWE-400": r"(Thread\.sleep\(|readLine\(\)|while \(|for \()",
                }
                
                if cwe in sink_heuristics:
                    if not re.search(sink_heuristics[cwe], code_content):
                        dropped += 1
                        logger.warning(f"Code-Aware Validation: Dropped {cwe} hallucination (sink pattern not found in code)")
                        continue

            validated.append(vuln)

        if dropped > 0:
            logger.warning(f"Validation dropped {dropped} entries, kept {len(validated)}")
        logger.info(f"Validation: {len(vulns)} entries in → {len(validated)} entries out")

        return validated


def format_report_as_markdown(report: Dict[str, Any]) -> str:
    """
    Converts a structured vulnerability report dictionary into a user-readable Markdown string.
    """
    if "error" in report:
        return (
            f"# Vulnerability Scan Report\n\n"
            f"❌ **Inference Error Encountered**\n\n"
            f"> {report['error']}\n\n"
            f"### Raw Response\n"
            f"```text\n{report.get('raw_response', 'No raw response available.')}\n```\n"
        )
        
    vulns = report.get("vulnerabilities", [])
    if not vulns:
        return (
            f"# Vulnerability Scan Report\n\n"
            f"✅ **No vulnerabilities detected!**\n\n"
            f"The scan completed successfully and identified no security issues in the provided code.\n"
        )
        
    md = []
    md.append("# Vulnerability Scan Report\n")
    md.append(f"### 📊 Summary: **{len(vulns)}** vulnerability/vulnerabilities detected.\n")
    md.append("---")
    
    for idx, vuln in enumerate(vulns, 1):
        cwe_id = vuln.get("cwe_id", "N/A")
        cwe_name = vuln.get("cwe_name", "Unknown CWE")
        severity = vuln.get("severity", "medium").upper()
        confidence = vuln.get("confidence", "N/A")
        description = vuln.get("description", vuln.get("cwe_description", "No description provided."))
        impact = vuln.get("impact", "No impact details provided.")
        import re
        def clean_recommendation_text(text: str) -> str:
            if not text: return text
            text = text.replace("applysecurecodingpracticestoremediate", "apply secure coding practices to remediate ")
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
            text = re.sub(r'([a-zA-Z0-9])\(', r'\1 (', text)
            return text
            
        recommendation = clean_recommendation_text(vuln.get("recommendation", "No recommendation provided."))
        
        # Determine a color/emoji for severity
        sev_emoji = "⚪"
        if severity == "CRITICAL":
            sev_emoji = "🔴"
        elif severity == "HIGH":
            sev_emoji = "fiber_manual_record"
            sev_emoji = "🔴"
        elif severity == "MEDIUM":
            sev_emoji = "🟡"
        elif severity == "LOW":
            sev_emoji = "🔵"
            
        md.append(f"\n## {idx}. {sev_emoji} {cwe_name} ({cwe_id})")
        md.append(f"- **Severity**: `{severity}`")
        if confidence != "N/A":
            md.append(f"- **Confidence**: `{confidence}`")
        
        location = vuln.get("location", {})
        if location:
            loc_str = []
            if "class" in location:
                loc_str.append(f"Class: `{location['class']}`")
            if "function" in location:
                loc_str.append(f"Function: `{location['function']}`")
            elif "method" in location:
                loc_str.append(f"Method: `{location['method']}`")
            if "start_line" in location and "end_line" in location:
                loc_str.append(f"Lines: `{location['start_line']}`–`{location['end_line']}`")
            elif "start_line" in location:
                loc_str.append(f"Line: `{location['start_line']}`")
            elif "line" in location:
                loc_str.append(f"Line: `{location['line']}`")
            if loc_str:
                md.append(f"- **Location**: {', '.join(loc_str)}")
                
        md.append(f"\n### 📝 Description\n{description}")
        if impact and impact != "No impact details provided.":
            md.append(f"\n### ⚠️ Impact\n{impact}")
        md.append(f"\n### 💡 Recommendation\n{recommendation}")
        
        # If there is a fixed_code or remediation block, show it
        if "fixed_code" in vuln:
            md.append(f"\n#### Suggested Remediation Code:\n```java\n{vuln['fixed_code']}\n```")
        elif "fixed_code" in report:  # fallback if model returned it at top level
            md.append(f"\n#### Suggested Remediation Code:\n```java\n{report['fixed_code']}\n```")
            
        md.append("\n---")
        
    return "\n".join(md)


def format_project_summary(all_reports: List[Dict[str, Any]]) -> str:
    """
    Generates a project-level summary across all scanned files.
    """
    total_files = len(all_reports)
    files_with_vulns = 0
    total_vulns = 0
    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    error_count = 0

    for report in all_reports:
        if "error" in report:
            error_count += 1
            continue
        vulns = report.get("vulnerabilities", [])
        if vulns:
            files_with_vulns += 1
            total_vulns += len(vulns)
            for v in vulns:
                sev = v.get("severity", "medium").upper()
                if sev in severity_counts:
                    severity_counts[sev] += 1

    md = []
    md.append("\n" + "=" * 60)
    md.append("# 📋 Project Scan Summary")
    md.append("=" * 60)
    md.append(f"- **Files scanned**: {total_files}")
    md.append(f"- **Files with vulnerabilities**: {files_with_vulns}")
    md.append(f"- **Total vulnerabilities found**: {total_vulns}")
    if error_count > 0:
        md.append(f"- **Scan errors**: {error_count}")
    md.append(f"\n### Severity Breakdown")
    md.append(f"- 🔴 Critical: **{severity_counts['CRITICAL']}**")
    md.append(f"- 🔴 High: **{severity_counts['HIGH']}**")
    md.append(f"- 🟡 Medium: **{severity_counts['MEDIUM']}**")
    md.append(f"- 🔵 Low: **{severity_counts['LOW']}**")
    md.append("=" * 60 + "\n")
    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Engine Verification")
    parser.add_argument("--model_id", type=str, required=True, help="Base model identifier or local merged model directory")
    parser.add_argument("--adapter_path", type=str, default=None, help="PEFT adapter weights directory")
    parser.add_argument("--target_path", type=str, required=True, help="Path to Java file or directory to scan")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum new tokens to generate")
    parser.add_argument("--min_confidence", type=float, default=0.0, help="Minimum confidence score to include finding (e.g. 0.8)")
    parser.add_argument("--format", type=str, choices=["json", "markdown"], default="markdown", help="Output format (default: markdown)")
    args = parser.parse_args()

    target_path = Path(args.target_path)
    if not target_path.exists():
        logger.error(f"Target path not found: {target_path}")
        exit(1)

    if target_path.is_dir():
        java_files = sorted(target_path.rglob("*.java"))
    else:
        java_files = [target_path] if target_path.suffix == ".java" else []

    if not java_files:
        logger.error(f"No Java files found in: {target_path}")
        exit(1)

    logger.info(f"Discovered {len(java_files)} Java file(s) to scan.")

    try:
        engine = VulnerabilityInferenceEngine(
            model_id=args.model_id,
            adapter_path=args.adapter_path,
            load_in_4bit=not args.no_quant
        )

        all_reports = []
        scan_start = time.time()

        for file_idx, j_file in enumerate(java_files, 1):
            logger.info(f"[{file_idx}/{len(java_files)}] Scanning: {j_file}")
            with open(j_file, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read()

            file_line_count = len(code_content.splitlines())

            report = engine.analyze_file_content(
                code_content,
                max_new_tokens=args.max_tokens,
                file_line_count=file_line_count,
                file_path=str(j_file),
                min_confidence=args.min_confidence
            )
            report["file_path"] = str(j_file)
            report["file_line_count"] = file_line_count
            all_reports.append(report)

        scan_duration = time.time() - scan_start
        logger.info(f"Scan completed in {scan_duration:.1f}s across {len(java_files)} file(s).")

        if args.format == "markdown":
            for report in all_reports:
                print(f"\n# File: {report['file_path']} ({report['file_line_count']} lines)\n")
                formatted_report = format_report_as_markdown(report)
                print(formatted_report)
            # Print project-level summary
            print(format_project_summary(all_reports))
        else:
            print("\n" + "=" * 50)
            print("INFERENCE VULNERABILITY REPORT (JSON):")
            print("=" * 50)
            print(json.dumps(all_reports, indent=4))
            print("=" * 50 + "\n")

    except Exception as e:
        logger.error(f"Verification run failed: {e}", exc_info=True)
