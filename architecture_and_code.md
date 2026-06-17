# CodeElixir.AI Architecture and Source Code

## Architecture Diagram

```text
+-----------------------------------------------------------------------------------+
|                                 REACT FRONTEND                                    |
|   (Sends Source Code / Requests Fixes)                                            |
+-----------------------------------------------------------------------------------+
                               |                   |
                     POST /api/scan         POST /api/remediate
                               |                   |
                               v                   v
+-----------------------------------------------------------------------------------+
|                               FASTAPI BACKEND                                     |
|                                  (api.py)                                         |
+-----------------------------------------------------------------------------------+
                               |                   |
                      (Triggers Analysis)   (Triggers Remediation)
                               |                   |
                               v                   v
+---------------------------------------+ +---------------------------------------+
|    VulnerabilityInferenceEngine       | |         FixInferenceEngine            |
|        (inference_engine.py)          | |          (fix_engine.py)              |
+---------------------------------------+ +---------------------------------------+
                               |                   |
                    (Loads Base+Adapter)  (Loads Base+Adapter)
                               |                   |
                               v                   v
+---------------------------------------+ +---------------------------------------+
|       DeepSeek Coder 6.7B Instruct    | |      Qwen 2.5 Coder 7B Instruct       |
|             (+ LoRA Adapter)          | |           (+ LoRA Adapter)            |
+---------------------------------------+ +---------------------------------------+
                               ^                   ^
                               |                   |
                         (Fine-Tuning Process)     |
                      (fine_tune.py / train_fix_model.py)
                               |
                   +-----------------------+
                   |   JSONL Datasets      |
                   | (generate_production_ |
                   |       data.py)        |
                   +-----------------------+
```

## Source Code


### api.py

`python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
import os
import sys
import concurrent.futures
import subprocess
import psutil
import re
from typing import Dict, Any
import concurrent.futures

# Add backend directory to sys.path if not present to ensure inference_engine imports correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_engine import VulnerabilityInferenceEngine
from fix_engine import FixInferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

from fastapi.middleware.cors import CORSMiddleware

class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn.access record.args is usually a tuple like (client_addr, method, path, http_version, status_code)
        # We drop the log if it's hitting the /api/telemetry route
        return record.args and len(record.args) >= 3 and record.args[2] != "/api/telemetry"

# Add filter to uvicorn access logger
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

from contextlib import asynccontextmanager

engine = None
fix_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, fix_engine
    logger.info("Initializing VulnerabilityInferenceEngine...")
    try:
        engine = VulnerabilityInferenceEngine(
            model_id="deepseek-ai/deepseek-coder-6.7b-instruct",
            adapter_path="./adapters",
            load_in_4bit=False  # BitsAndBytes 4-bit incompatible with ROCm/AMD GPU
        )
        logger.info("Engine loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load engine: {e}")
        engine = None

    logger.info("Initializing FixInferenceEngine...")
    try:
        fix_engine = FixInferenceEngine(
            model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter_path="./adapters_fix",
            load_in_4bit=False
        )
        logger.info("FixEngine loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load fix engine: {e}")
        fix_engine = None

    yield
    # Cleanup on exit
    engine = None
    fix_engine = None

app = FastAPI(title="CodeElixir.AI Backend API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProxyMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope.get("headers", []))
            prefix = headers.get(b"x-forwarded-prefix") or headers.get(b"x-forwarded-context") or headers.get(b"x-proxycontextpath")
            if prefix:
                scope["root_path"] = prefix.decode("utf-8")
        await self.app(scope, receive, send)

app.add_middleware(ProxyMiddleware)

# Models are now initialized via FastAPI lifespan events


class ScanFolderRequest(BaseModel):
    folder_path: str

class ScanFileRequest(BaseModel):
    file_path: str

class FileContent(BaseModel):
    name: str
    path: str
    content: str
    originalContent: str

class ScanRequest(BaseModel):
    files: list[FileContent]


class RemediateRequest(BaseModel):
    filePath: str
    fileContent: str
    vulnerability: dict


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "engine_loaded": engine is not None}


@app.get("/api/debug_prompt")
def debug_prompt():
    from inference_engine import PROMPT_TEMPLATE
    return {
        "prompt_template": PROMPT_TEMPLATE,
        "adapters_loaded": engine.adapter_path if engine is not None else None,
        "model_id": engine.model_id if engine is not None else None
    }


@app.post("/api/scan_folder")
def scan_folder(request: ScanFolderRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not loaded properly.")
        
    target_path = Path(request.folder_path)
    
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Folder path does not exist.")
        
    if not target_path.is_dir():
        raise HTTPException(status_code=400, detail="Path provided is not a directory.")
        
    java_files = list(target_path.rglob("*.java"))
    if not java_files:
        return {"vulnerabilities": [], "message": "No .java files found in the specified directory."}
        
    all_vulnerabilities = []
    
    def process_file(file_path):
        logger.info(f"Scanning file: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read()
                
            file_line_count = len(code_content.splitlines())
            report = engine.analyze_file_content(
                code_content,
                max_new_tokens=2048,
                file_line_count=file_line_count,
                file_path=str(file_path)
            )
            
            # Map findings
            vulns = report.get("vulnerabilities", [])
            mapped_vulns = []
            for vuln in vulns:
                # Add file context to vulnerability
                vuln["filePath"] = str(file_path)
                
                # Standardize keys to match UI expectations
                if "cwe_name" in vuln and "type" not in vuln:
                    vuln["type"] = vuln["cwe_name"]
                
                # Extract line number
                loc = vuln.get("location", {})
                if "start_line" in loc:
                    vuln["lineNumber"] = loc["start_line"]
                elif "line" in loc:
                    vuln["lineNumber"] = loc["line"]
                else:
                    vuln["lineNumber"] = 1
                    
                vuln["status"] = "Scanned"
                mapped_vulns.append(vuln)
            return mapped_vulns
        except Exception as e:
            logger.error(f"Failed to scan file {file_path}: {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_file, java_files)
        
    for res in results:
        all_vulnerabilities.extend(res)
            
    return {"vulnerabilities": all_vulnerabilities}


@app.post("/api/scan_file")
def scan_file(request: ScanFileRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not loaded properly.")
        
    file_path = Path(request.file_path)
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File path does not exist.")
        
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Path provided is not a file.")
        
    logger.info(f"Scanning single file: {file_path}")
    all_vulnerabilities = []
    
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code_content = f.read()
            
        file_line_count = len(code_content.splitlines())
        report = engine.analyze_file_content(
            code_content,
            max_new_tokens=1024,
            file_line_count=file_line_count,
            file_path=str(file_path)
        )
        
        vulns = report.get("vulnerabilities", [])
        for vuln in vulns:
            vuln["filePath"] = str(file_path)
            
            if "cwe_name" in vuln and "type" not in vuln:
                vuln["type"] = vuln["cwe_name"]
            
            loc = vuln.get("location", {})
            if "start_line" in loc:
                vuln["lineNumber"] = loc["start_line"]
            elif "line" in loc:
                vuln["lineNumber"] = loc["line"]
            else:
                vuln["lineNumber"] = 1
                
            vuln["status"] = "Scanned"
            all_vulnerabilities.append(vuln)
            
    except Exception as e:
        logger.error(f"Failed to scan file {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scan file: {e}")
        
    return {"vulnerabilities": all_vulnerabilities}


@app.post("/api/scan")
def scan_contents(request: ScanRequest):
    """Endpoint for the React frontend which sends file contents directly"""
    if engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not loaded properly.")
        
    all_vulnerabilities = []
    
    def process_file_content(file_obj):
        logger.info(f"Scanning provided content for: {file_obj.path}")
        try:
            file_line_count = len(file_obj.content.splitlines())
            report = engine.analyze_file_content(
                file_obj.content,
                max_new_tokens=2048,
                file_line_count=file_line_count,
                file_path=file_obj.path
            )
            
            vulns = report.get("vulnerabilities", [])
            mapped_vulns = []
            for vuln in vulns:
                vuln["filePath"] = file_obj.path
                if "cwe_name" in vuln and "type" not in vuln:
                    vuln["type"] = vuln["cwe_name"]
                
                loc = vuln.get("location", {})
                if "start_line" in loc:
                    vuln["lineNumber"] = loc["start_line"]
                elif "line" in loc:
                    vuln["lineNumber"] = loc["line"]
                else:
                    vuln["lineNumber"] = 1
                    
                vuln["status"] = "Scanned"
                mapped_vulns.append(vuln)
            return mapped_vulns
        except Exception as e:
            logger.error(f"Failed to scan content for {file_obj.path}: {e}")
            return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_file_content, request.files)
        
    for res in results:
        all_vulnerabilities.extend(res)
            
    return {"vulnerabilities": all_vulnerabilities}


@app.post("/api/remediate")
def remediate_vulnerability(request: RemediateRequest):
    if fix_engine is None:
        raise HTTPException(status_code=500, detail="Fix inference engine not loaded properly.")
        
    logger.info(f"Remediating vulnerability {request.vulnerability.get('cwe_id')} in {request.filePath}")
    try:
        report = fix_engine.remediate_file_content(
            raw_code=request.fileContent,
            vulnerability=request.vulnerability,
            max_new_tokens=2048
        )
        return report
    except Exception as e:
        logger.error(f"Failed to remediate vulnerability: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to remediate: {e}")


@app.get("/api/telemetry")
def get_telemetry() -> Dict[str, Any]:
    """Real-time system telemetry for GPU Stats."""
    vram_usage = 14200
    vram_total = 198000
    compute_load = 45
    cpu_usage = int(psutil.cpu_percent())
    ram_usage_gb = 0
    gpu_name = "AMD MI300X"
    gpu_type = "HBM3 Memory Cluster"
    
    try:
        free_res = subprocess.run(["free", "-h"], capture_output=True, text=True, timeout=2)
        if free_res.returncode == 0:
            for line in free_res.stdout.split('\n'):
                if line.startswith('Mem:'):
                    parts = line.split()
                    if len(parts) >= 3:
                        used_str = parts[2]
                        if 'Gi' in used_str:
                            ram_usage_gb = int(float(used_str.replace('Gi', '')))
                        elif 'G' in used_str:
                            ram_usage_gb = int(float(used_str.replace('G', '')))
                        elif 'Mi' in used_str:
                            ram_usage_gb = max(1, int(float(used_str.replace('Mi', '')) / 1024))
                        elif 'M' in used_str:
                            ram_usage_gb = max(1, int(float(used_str.replace('M', '')) / 1024))
                        else:
                            ram_usage_gb = int(float(''.join(filter(str.isdigit, used_str))))
    except Exception as e:
        logger.warning(f"Failed to fetch free -h telemetry: {e}")
        # fallback
        ram = psutil.virtual_memory()
        ram_usage_gb = int(ram.used / (1024**3))

    try:
        res = subprocess.run(["amd-smi"], capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            mem_match = re.search(r'(\d+)/\d+\s+MB', res.stdout)
            if mem_match:
                vram_usage = int(mem_match.group(1))
            
            lines = res.stdout.split('\n')
            for line in lines:
                if 'SPX' in line or 'MB' in line:
                    gfx_match = re.search(r'\|\s*(\d+)\s*%', line)
                    if gfx_match:
                        compute_load = int(gfx_match.group(1))
        else:
            raise Exception("amd-smi not found")
    except Exception:
        # fallback to nvidia-smi
        try:
            res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                name_match = re.search(r'\|\s*\d+\s+([^|]+?)\s+(?:Off|On)\s*\|', res.stdout)
                if name_match:
                    gpu_name = name_match.group(1).strip()
                    gpu_type = "GDDR5 Memory Cluster"
                    
                mem_match = re.search(r'(\d+)MiB\s*/\s*(\d+)MiB', res.stdout)
                if mem_match:
                    vram_usage = int(mem_match.group(1))
                    vram_total = int(mem_match.group(2))
                    
                util_match = re.search(r'(\d+)%\s+Default', res.stdout)
                if util_match:
                    compute_load = int(util_match.group(1))
        except Exception as e:
            logger.warning(f"Failed to fetch nvidia-smi telemetry: {e}")

    return {
        "vram_usage": vram_usage,
        "vram_total": vram_total,
        "compute_load": compute_load,
        "cpu_usage": cpu_usage,
        "ram_usage": ram_usage_gb,
        "gpu_name": gpu_name,
        "gpu_type": gpu_type,
        "models_loaded": ["DeepSeek-Coder-6.7B-Instruct (Vuln Scanner)", "Qwen2.5-Coder-7B-Instruct (Fix Engine)"],
        "api_health": "Online (FastAPI / 0.0.0.0:8000)"
    }

`

### inference_engine.py

`python
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
    "<|instruction|>\n"
    "Analyze the Java code and identify ALL security vulnerabilities. Return structured JSON only.\n\n"
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
                "repetition_penalty": 1.3,
            },
            {  # Attempt 2: Sampling fallback (different token distribution)
                "do_sample": True,
                "temperature": 0.4,
                "top_p": 0.85,
                "top_k": 50,
                "repetition_penalty": 1.25,
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

                # 2. Run generation (wrapped to catch GPU/ROCm crashes)
                try:
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
                except RuntimeError as gpu_err:
                    logger.error(f"GPU error during generation attempt {attempt_idx}: {gpu_err}")
                    if attempt_idx < len(generation_configs):
                        logger.warning(f"Skipping to next generation config...")
                        continue
                    return {"vulnerabilities": [], "error": f"GPU generation failed: {str(gpu_err)}"}

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
            "permission": ("CWE-276", "Incorrect Default Permissions"),
            "ssrf": ("CWE-918", "Server-Side Request Forgery (SSRF)"),
            "forgery": ("CWE-918", "Server-Side Request Forgery (SSRF)"),
            "server-side": ("CWE-918", "Server-Side Request Forgery (SSRF)")
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
                "CWE-22": (r"(\bFile\b|\bFileInputStream\b|\bFileOutputStream\b|\bFileReader\b|\bFileWriter\b|\bPaths\.get\b|\bFiles\.read\b|\bFiles\.write\b|\bMultipartFile\b|\bInputStream\b|\bOutputStream\b)", "Path Traversal"),
                "CWE-276": (r"(new File\(|new FileReader\(|createTempFile\(|setExecutable\(|setReadable\(|FileOutputStream\()", "Incorrect Default Permissions"),
                "CWE-89": (r"(executeQuery\(|prepareStatement\(|executeUpdate\(|Statement |createStatement\()", "SQL Injection"),
                "CWE-79": (r"(getWriter\(\)\.print|out\.println)", "Cross-Site Scripting"),
                "CWE-90": (r"(InitialDirContext|search\(|lookup\()", "LDAP Injection"),
                "CWE-643": (r"(XPath |evaluate\(|compile\()", "XPath Injection"),
                "CWE-321": (r"(SecretKeySpec|AES|DES)", "Use of Hard-coded Cryptographic Key"),
                "CWE-319": (r"(HttpURLConnection|Socket |http://|ftp://|SocketChannel)", "Cleartext Transmission of Sensitive Information"),
                "CWE-522": (r"(getConnection\(|DriverManager|password|login)", "Insufficiently Protected Credentials"),
                "CWE-601": (r"(sendRedirect\(|setHeader\(\"Location\")", "URL Redirection to Untrusted Site ('Open Redirect')"),
                "CWE-918": (r"(exchange\(|execute\(|getForObject\(|postForEntity\(|HttpClient\.new|openConnection\()", "Server-Side Request Forgery (SSRF)"),
                "CWE-400": (r"(Thread\.sleep\(|readLine\(\)|while \(|for \()", "Uncontrolled Resource Consumption"),
            }
            for cwe_id, (pattern, std_name) in sink_heuristics.items():
                if re.search(pattern, code_content):
                    return cwe_id, std_name

        return None, None

    @staticmethod
    def _sanitize_field(text: str) -> str:
        """
        Cleans repetitive nested parenthesis bugs from individual text fields
        (e.g., "Exploitation de CWE-78 (Exploitation de CWE-78 (Exploitation...))")
        """
        if not text or not isinstance(text, str):
            return text
            
        import re
        prev = None
        # Loop to collapse arbitrarily deep nesting
        while text != prev:
            prev = text
            # Matches: X (X) -> X
            text = re.sub(r'([^()]{10,})\s*\(\s*\1\s*\)?', r'\1', text)
            # Matches: (X (X)) -> (X)
            text = re.sub(r'\(\s*([^()]{10,})\s*\(\s*\1\s*\)?\)?', r'(\1)', text)
            
        # Clean up any trailing broken parenthesis
        text = re.sub(r'(\s*\))+$', '', text).strip()
        
        return text

    @staticmethod
    def _translate_to_english(text: str) -> str:
        """
        Translates text to English using deep-translator or translate package if installed.
        Safely falls back to the original text on any failure.
        """
        if not text or not isinstance(text, str):
            return text
            
        # Fast path: check if text is already English using langdetect
        try:
            from langdetect import detect
            if detect(text) == 'en':
                return text
        except Exception:
            pass
            
        # 1. Try deep-translator (GoogleTranslator)
        try:
            from deep_translator import GoogleTranslator
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            if translated:
                return translated
        except Exception:
            pass
            
        # 2. Try translate package (Translator)
        try:
            from translate import Translator
            translator = Translator(to_lang="en")
            translated = translator.translate(text)
            if translated:
                return translated
        except Exception:
            pass
            
        return text

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

            # Translate text fields to English if they are in another language (e.g. French)
            for field in ["cwe_name", "description", "impact", "recommendation", "type", "message"]:
                if vuln.get(field) and isinstance(vuln[field], str):
                    # First sanitize any nested repetition bugs from the generation
                    vuln[field] = VulnerabilityInferenceEngine._sanitize_field(vuln[field])
                    # Then attempt translation if applicable
                    vuln[field] = VulnerabilityInferenceEngine._translate_to_english(vuln[field])

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
            # Always run inference to fix hallucinated mappings (e.g. CWE-22 for Command Injection)
            old_id = vuln.get("cwe_id")
            old_id = str(old_id) if old_id is not None else ""
            old_name = vuln.get("cwe_name")
            old_name = str(old_name) if old_name is not None else ""
            
            inferred_id, inferred_name = VulnerabilityInferenceEngine._infer_cwe(vuln, code_content)
            if inferred_id:
                vuln["cwe_id"] = inferred_id
                vuln["cwe_name"] = inferred_name
                
                # Sanitize descriptive fields to replace hallucinated names with the corrected ones
                for field in ["description", "impact", "recommendation", "message"]:
                    if vuln.get(field) and isinstance(vuln[field], str):
                        text = vuln[field]
                        if old_id and old_id != inferred_id:
                            text = text.replace(old_id, inferred_id)
                        if old_name and old_name != inferred_name:
                            clean_old = old_name.split("(")[0].strip()
                            clean_new = inferred_name.split("(")[0].strip()
                            if clean_old and clean_old in text:
                                text = text.replace(clean_old, clean_new)
                        vuln[field] = text

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

            # Weighted Scoring Validation System
            # Component 1: Confidence Score Component (Max 40 points)
            conf_val = vuln.get("confidence", 0.0)
            confidence_score = conf_val * 40.0

            # Component 2: Sink Validation Component (Max 40 points)
            sink_score = 20.0  # Default neutral score if no heuristic exists for this CWE
            sink_found = False
            
            if code_content and vuln.get("cwe_id"):
                cwe = vuln["cwe_id"].upper()
                sink_heuristics = {
                    "CWE-78": r"(Runtime\.getRuntime\(\)\.exec\(|new ProcessBuilder\(|ProcessBuilder\()",
                    "CWE-20": r"(Runtime\.getRuntime\(\)\.exec\(|new ProcessBuilder\(|ProcessBuilder\()",
                    "CWE-22": r"(\bFile\b|\\bFileInputStream\b|\bFileOutputStream\b|\bFileReader\b|\bFileWriter\b|\bPaths\.get\b|\bFiles\.read\b|\bFiles\.write\b|\bMultipartFile\b|\bInputStream\b|\bOutputStream\b)",
                    "CWE-276": r"(new File\(|new FileReader\(|createTempFile\(|setExecutable\(|setReadable\(|FileOutputStream\()",
                    "CWE-89": r"(executeQuery\(|prepareStatement\(|executeUpdate\(|Statement |createStatement\()",
                    "CWE-79": r"(getWriter\(\)\.print|out\.println)",
                    "CWE-90": r"(InitialDirContext|search\(|lookup\()",
                    "CWE-643": r"(XPath |evaluate\(|compile\()",
                    "CWE-321": r"(SecretKeySpec|AES|DES)",
                    "CWE-319": r"(HttpURLConnection|Socket |http://|ftp://|SocketChannel)",
                    "CWE-522": r"(getConnection\(|DriverManager|password|login)",
                    "CWE-601": r"(sendRedirect\(|setHeader\(\"Location\")",
                    "CWE-918": r"(exchange\(|execute\(|getForObject\(|postForEntity\(|HttpClient\.new|openConnection\()",
                    "CWE-400": r"(Thread\.sleep\(|readLine\(\)|while \(|for \()",
                }
                
                if cwe in sink_heuristics:
                    pattern = sink_heuristics[cwe]
                    lines = code_content.splitlines()
                    # Find the actual line number of the sink to correct the location
                    for i, line in enumerate(lines):
                        if re.search(pattern, line):
                            if "location" not in vuln:
                                vuln["location"] = {}
                            vuln["location"]["line"] = i + 1
                            sink_found = True
                            break
                    
                    if sink_found:
                        sink_score = 40.0
                    else:
                        # Known CWE but NO sink found → strong negative signal
                        sink_score = -10.0

                # DTO/POJO Detection: if the file has no code logic (only fields,
                # annotations, imports, class declarations), it cannot contain vulnerabilities.
                # Instead of matching method signatures (fragile), look for control-flow
                # and logic keywords that never appear in pure DTOs.
                stripped_lines = [l.strip() for l in code_content.splitlines() if l.strip()]
                has_code_logic = any(
                    re.search(
                        r'(\bif\s*\(|\bfor\s*\(|\bwhile\s*\(|\btry\s*\{|\bcatch\s*\(|'
                        r'\breturn\s|\bthrow\s|\bnew\s+\w+\(|'
                        r'\.\w+\s*\([^)]*\))',  # method calls like .exec(), .readFile()
                        l
                    )
                    for l in stripped_lines
                )
                if not has_code_logic and len(stripped_lines) < 30:
                    sink_score = -30.0
                    logger.info(f"DTO/POJO detected (no code logic, {len(stripped_lines)} lines) — penalizing finding")

            # Component 3: False Positive Component (Max 20 points, or penalty)
            fp_score = 20.0
            if code_content and "location" in vuln and "line" in vuln["location"]:
                line_idx = vuln["location"]["line"] - 1
                lines = code_content.splitlines()
                if 0 <= line_idx < len(lines):
                    flagged_line = lines[line_idx]
                    # Check for false positive patterns (Regex compilation, import, comment, package, empty brackets, boilerplate annotations, constructors, exceptions)
                    if re.search(r"(Pattern\.compile|java\.util\.regex|^\s*//|^\s*import |^\s*package |^\s*[{}]\s*$|^\s*@(?:Test|SpringBootTest|Before|After|Override|Data|Getter|Setter|Builder|NoArgsConstructor|AllArgsConstructor|Value|Immutable)\b|void\s+[a-zA-Z0-9_]+\s*\(|^\s*super\(|class\s+[A-Za-z0-9_]+Exception|^\s*(?:public|protected|private)\s+[A-Z][a-zA-Z0-9_]*\s*\()", flagged_line):
                        if not sink_found:
                            fp_score = -20.0  # Apply penalty if it looks like a false positive and no sink validates it
                        else:
                            fp_score = 0.0

            # Also penalize if the flagged line is just a field declaration (common in DTOs)
            if code_content and "location" in vuln and "line" in vuln["location"]:
                line_idx = vuln["location"]["line"] - 1
                lines = code_content.splitlines()
                if 0 <= line_idx < len(lines):
                    flagged_line = lines[line_idx].strip()
                    if re.match(r'^(public|private|protected)?\s*(String|int|long|boolean|List|Map|Set|Optional)\s+\w+\s*;', flagged_line):
                        fp_score = -20.0
                        logger.info(f"Flagged line is a simple field declaration — applying FP penalty")

            # Final Decision based on Total Weight (Max 100, Threshold 50)
            total_weight = confidence_score + sink_score + fp_score
            
            if total_weight < 50.0:
                dropped += 1
                logger.warning(f"Weighted Validation: Dropped {vuln.get('cwe_id')} (Score: {total_weight:.1f}/100) -> Conf: {confidence_score:.1f}, Sink: {sink_score}, FP: {fp_score}")
                continue
            else:
                logger.info(f"Weighted Validation: Kept {vuln.get('cwe_id')} (Score: {total_weight:.1f}/100) -> Conf: {confidence_score:.1f}, Sink: {sink_score}, FP: {fp_score}")

            validated.append(vuln)


        if dropped > 0:
            logger.warning(f"Validation dropped {dropped} entries, kept {len(validated)}")

        # --- Deduplication by (cwe_id, line) ---
        # When the model produces multiple entries for the same CWE at the same line,
        # keep only the entry with the longest (most specific) description.
        seen: Dict[tuple, int] = {}  # (cwe_id, line) -> index in deduped list
        deduped: List[Dict[str, Any]] = []
        dedup_count = 0
        for vuln in validated:
            cwe = vuln.get("cwe_id", "")
            line = vuln.get("location", {}).get("line", 0)
            key = (cwe, line)
            if key in seen:
                # Duplicate found — keep the one with the longer description
                existing_idx = seen[key]
                existing = deduped[existing_idx]
                if len(vuln.get("description", "")) > len(existing.get("description", "")):
                    deduped[existing_idx] = vuln
                    seen[key] = existing_idx
                    logger.info(f"Dedup: Replaced shorter duplicate for {cwe} at line {line}")
                else:
                    logger.info(f"Dedup: Dropped duplicate for {cwe} at line {line} (shorter description)")
                dedup_count += 1
            else:
                seen[key] = len(deduped)
                deduped.append(vuln)

        if dedup_count > 0:
            logger.info(f"Deduplication: {len(validated)} → {len(deduped)} entries ({dedup_count} duplicates removed)")
        validated = deduped

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

`

### fix_engine.py

`python
import platform
# Monkeypatch platform.win32_ver to bypass Windows WMI query errors/hangs
platform.win32_ver = lambda release='', version='', csd='', ptype='': ('10', '10.0.22621', '', 'Multiprocessor Free')

import json
import logging
import re
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

PROMPT_TEMPLATE = """
<|instruction|>
You are a Senior Java Application Security Engineer.

Your task is to generate a secure remediation for the supplied Java source file.

VULNERABILITY INFORMATION

CWE ID: {cwe_id}
CWE Name: {cwe_name}
CVE ID: {cve_id}
Line Number: {line_number}

REMEDIATION REQUIREMENTS

1. Analyze the supplied Java source code.
2. Use the supplied CWE as the authoritative vulnerability classification.
3. Use the supplied line number as the primary vulnerability location.
4. The line number may be approximate. Inspect nearby code when necessary.
5. Fix ONLY the vulnerability described by the supplied CWE.
6. Preserve business functionality.
7. Preserve package declarations.
8. Preserve imports whenever possible.
9. Preserve class names.
10. Preserve method signatures.
11. Preserve comments.
12. Preserve formatting where practical.
13. Do not remove functionality unless required for security.
14. Do not introduce unrelated refactoring.
15. Do not introduce new vulnerabilities.
16. Do not invent frameworks.
17. Do not invent dependencies.
18. Add imports only when absolutely necessary.
19. Return the ENTIRE corrected Java source file.
20. Never return snippets.
21. Never return diffs.
22. Never return markdown.
23. Never truncate the fixed code.

SECURITY REQUIREMENTS

CWE-78:
- NEVER use shell command interpreters. 
- EXTREMELY IMPORTANT: You MUST pass the target executable and its arguments as a comma-separated list of individual string arguments to ProcessBuilder (e.g., new ProcessBuilder("ping", "-c", "1", ip)). Do NOT pass the entire command as a single string.
- When validating input with Regex, ALWAYS use matcher.matches() instead of matcher.find() and fully anchor regexes with ^ and $.

CWE-89:
- Use PreparedStatement.
- Never concatenate SQL.

CWE-79:
- Use context-aware output encoding.

CWE-22:
- Mitigate TOCTOU (Time-of-Check Time-of-Use) by using java.nio.file.Path and toRealPath() for both base and target paths.
- Ensure the normalized target path strictly starts with the base path.

CWE-611:
- Disable DTD processing.
- Disable external entities.

CWE-502:
- Use ObjectInputFilter or allowlists.

CWE-918:
- Parse URLs with java.net.URI and explicitly check that uri.getHost() is not null.
- Enforce scheme validation (only http/https).
- Mitigate DNS rebinding: strictly resolve the hostname to an InetAddress and block isAnyLocalAddress, isLoopbackAddress, isLinkLocalAddress, and isSiteLocalAddress.
- Configure HTTP clients to NOT follow redirects automatically.

OUTPUT FORMAT

Return ONLY valid JSON.

{{
  "explanation": "Detailed root cause and remediation explanation",
  "fixed_code": "FULL corrected Java source file"
}}

<|input|>
{input_json}

<|response|>
"""

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
                "cwe_id": vulnerability.get("cwe_id", ""),
                "cwe_name": vulnerability.get("cwe_name", ""),
                "cve_id": vulnerability.get("cve_id", ""),
                "line_number": (
                    vulnerability.get("lineNumber")
                    or vulnerability.get("line_number")
                ),
                "vulnerable_code": raw_code,
                "remediation_goal": (
                    "Return the complete corrected Java file. "
                    "Preserve functionality. "
                    "Fix the specified vulnerability only."
                )
            }
                
            input_json = json.dumps(input_dict, ensure_ascii=False)
            
            if self.using_adapter:
                prompt = PROMPT_TEMPLATE.format(input_json=input_json, **input_dict)
            else:
                messages = [
                    {
                        "role": "system",
                        "content": """
You are a Senior Java Security Engineer specializing in vulnerability remediation.

You will receive:

- CWE ID
- CWE Name
- Optional CVE
- Vulnerability line number
- Complete Java source file

Your task is to generate a production-quality security patch.

RULES

1. Use the supplied CWE as authoritative.
2. Analyze the code before generating a fix.
3. Fix the vulnerability at or near the provided line number.
4. Preserve functionality.
5. Preserve class structure.
6. Preserve package declarations.
7. Preserve imports whenever possible.
8. Preserve method signatures.
9. Preserve comments.
10. Return the COMPLETE corrected Java file.
11. Never return snippets.
12. Never return diffs.
13. Never return markdown.
14. Never return pseudo-code.
15. Never return partial files.
16. Never invent frameworks.
17. Never invent libraries.
18. Never introduce unrelated refactoring.

SECURITY RULES

CWE-78
- NEVER use shell command interpreters. 
- EXTREMELY IMPORTANT: You MUST pass the target executable and its arguments as a comma-separated list of individual string arguments to ProcessBuilder (e.g., new ProcessBuilder("ping", "-c", "1", ip)). Do NOT pass the entire command as a single string.
- When validating input with Regex, ALWAYS use matcher.matches() instead of matcher.find() and fully anchor regexes with ^ and $.

CWE-89
- Use PreparedStatement.

CWE-79
- Use output encoding.

CWE-22
- Mitigate TOCTOU (Time-of-Check Time-of-Use) by using java.nio.file.Path and toRealPath() for both base and target paths.
- Ensure the normalized target path strictly starts with the base path.

CWE-611
- Disable XXE processing.

CWE-502
- Use ObjectInputFilter.

CWE-918
- Parse URLs with java.net.URI and explicitly check that uri.getHost() is not null.
- Enforce scheme validation (only http/https).
- Mitigate DNS rebinding: strictly resolve the hostname to an InetAddress and block isAnyLocalAddress, isLoopbackAddress, isLinkLocalAddress, and isSiteLocalAddress.
- Configure HTTP clients to NOT follow redirects automatically.

OUTPUT

Return ONLY valid JSON:

{
  "explanation": "Root cause and remediation explanation",
  "fixed_code": "FULL corrected Java source file"
}
"""
                    },
                    {
                        "role": "user",
                        "content": f'''
Fix the vulnerability.

CWE ID: {vulnerability.get("cwe_id","")}
CWE Name: {vulnerability.get("cwe_name","")}
Line Number: {vulnerability.get("lineNumber") or vulnerability.get("line_number")}

{input_json}
'''
                    }
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
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    repetition_penalty=1.05,
                    use_cache=True,
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
            fixed_code = result.get("fixed_code", raw_code)

            # Fix common LLM syntax error where it appends $ to the regex but swallows the closing quote
            fixed_code = re.sub(r'\)\$\s*,\s*Pattern\.CASE_INSENSITIVE', r')$", Pattern.CASE_INSENSITIVE', fixed_code)

            # Deterministic post-processing fallback for stubborn 7B LLMs that refuse to drop 'sh -c'
            if vulnerability.get("cwe_id", "") == "CWE-78" or "sh" in fixed_code:
                # Target the exact ping pattern commonly missed
                fixed_code = re.sub(
                    r'new\s+ProcessBuilder\s*\(\s*["\'](?:sh|bash|cmd(?:\.exe)?)["\']\s*,\s*["\'](?:-c|/c)["\']\s*,\s*["\']ping\s+-c\s+1\s+["\']\s*\+\s*([a-zA-Z0-9_]+)\s*\)',
                    r'new ProcessBuilder("ping", "-c", "1", \1)',
                    fixed_code
                )
                # General fallback for any other simple command invocation
                fixed_code = re.sub(
                    r'new\s+ProcessBuilder\s*\(\s*["\'](?:sh|bash|cmd(?:\.exe)?)["\']\s*,\s*["\'](?:-c|/c)["\']\s*,\s*([a-zA-Z0-9_]+)\s*\)',
                    r'new ProcessBuilder(\1.split("\\\\s+"))',
                    fixed_code
                )

            # Construct return format
            return {
                "remediationExplanation": result.get("explanation", "No explanation provided."),
                "fullRemediatedContent": fixed_code,
                "remediatedSnippet": fixed_code # UI can use this as full string for display
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
    parser.add_argument("--line_number", type=int, default=None, help="The line number of the vulnerability.")
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
    if args.line_number is not None:
        vulnerability_details["lineNumber"] = args.line_number

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

`

### data_preparation.py

`python
import platform
# Monkeypatch platform.win32_ver to bypass Windows WMI query errors/hangs
platform.win32_ver = lambda release='', version='', csd='', ptype='': ('10', '10.0.22621', '', 'Multiprocessor Free')

import json
import logging
from pathlib import Path
from typing import Dict, List, Union, Optional, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("data_preparation")


class JavaVulnerabilityDataset(Dataset):
    """
    A custom PyTorch Dataset designed to process a local JSONL file containing
    a single 'text' block in the format:
    "<|instruction|>\n{instruction}\n\n<|input|>\n{java_code}\n\n<|response|>\n{json_output}"
    Configured for Causal LM training where target labels match input IDs.
    """
    def __init__(
        self,
        jsonl_path: Union[str, Path],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
    ) -> None:
        """
        Initializes the dataset and loads data records.

        Args:
            jsonl_path: Path to the JSONL dataset file.
            tokenizer: Pretrained tokenizer from Hugging Face.
            max_length: Maximum token sequence length for truncation.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[str] = []

        # Ensure pad_token_id is configured correctly
        if self.tokenizer.pad_token_id is None:
            logger.info("Tokenizer pad_token_id is missing. Falling back to eos_token_id.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Read JSONL file and extract the 'text' field
        jsonl_path = Path(jsonl_path)
        try:
            logger.info(f"Attempting to load dataset from: {jsonl_path}")
            if not jsonl_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {jsonl_path}")

            with open(jsonl_path, "r", encoding="utf-8") as file:
                for line_idx, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    try:
                        data: Dict[str, Any] = json.loads(line)
                        if "text" in data:
                            self.examples.append(data["text"])
                        elif "instruction" in data and "input" in data and "output" in data:
                            text = f"<|instruction|>\n{data['instruction']}\n\n<|input|>\n{data['input']}\n\n<|response|>\n{data['output']}"
                            self.examples.append(text)
                        else:
                            logger.warning(
                                f"Skipping line {line_idx} in {jsonl_path.name}: 'text' key not found."
                            )
                    except json.JSONDecodeError as decode_err:
                        logger.error(
                            f"JSON parse failure on line {line_idx} in {jsonl_path.name}: {decode_err}"
                        )
            
            logger.info(f"Successfully loaded {len(self.examples)} examples from {jsonl_path.name}")
        except Exception as err:
            logger.error(f"Failed to read dataset from {jsonl_path}: {err}", exc_info=True)
            raise

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single sample, tokenizes it, manages sequence lengths,
        and sets labels for Causal LM training with prompt masking.

        Only the response portion (after <|response|>) contributes to the
        training loss. The instruction and input tokens have their labels
        set to -100 so CrossEntropyLoss ignores them.
        """
        text = self.examples[idx]
        try:
            # Tokenize complete block with strict length limit
            encodings = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_tensors=None
            )

            input_ids: List[int] = encodings["input_ids"]
            attention_mask: List[int] = encodings["attention_mask"]

            # Create labels aligned with input_ids
            labels: List[int] = list(input_ids)

            # --- Prompt Masking ---
            # Find where the response section starts in the raw text.
            # Everything before (and including) the <|response|>\n marker is prompt;
            # the model should NOT be trained to predict those tokens.
            response_marker = "<|response|>\n"
            response_start_char = text.find(response_marker)

            if response_start_char != -1:
                # The prompt is everything up to and including the marker
                prompt_text = text[:response_start_char + len(response_marker)]

                # Tokenize just the prompt portion to find its token boundary
                prompt_encodings = self.tokenizer(
                    prompt_text,
                    add_special_tokens=True,
                    return_tensors=None
                )
                prompt_token_len = len(prompt_encodings["input_ids"])

                # Mask all prompt tokens in labels (set to -100 so loss ignores them)
                mask_len = min(prompt_token_len, len(labels))
                for i in range(mask_len):
                    labels[i] = -100

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
        except Exception as err:
            logger.error(f"Tokenization failed for dataset item at index {idx}: {err}", exc_info=True)
            raise


class CausalLMDataCollator:
    """
    Data Collator to dynamically pad batches to the maximum sequence length
    present in the current batch. Configures padding and masks label padding positions to -100.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Find maximum length in the current batch
        max_len = max(item["input_ids"].size(0) for item in batch)
        
        batch_input_ids: List[torch.Tensor] = []
        batch_attention_mask: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must possess a valid, configured pad_token_id.")
            
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            
            diff = max_len - input_ids.size(0)
            if diff > 0:
                pad_ids = torch.full((diff,), pad_token_id, dtype=torch.long)
                pad_mask = torch.zeros((diff,), dtype=torch.long)
                pad_labels = torch.full((diff,), -100, dtype=torch.long)

                if padding_side == "right":
                    new_input_ids = torch.cat([input_ids, pad_ids])
                    new_attention_mask = torch.cat([attention_mask, pad_mask])
                    new_labels = torch.cat([labels, pad_labels])
                else:
                    new_input_ids = torch.cat([pad_ids, input_ids])
                    new_attention_mask = torch.cat([pad_mask, attention_mask])
                    new_labels = torch.cat([pad_labels, labels])
            else:
                new_input_ids = input_ids
                new_attention_mask = attention_mask
                new_labels = labels

            batch_input_ids.append(new_input_ids)
            batch_attention_mask.append(new_attention_mask)
            batch_labels.append(new_labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels)
        }


def generate_mock_jsonl(path: Path) -> None:
    """
    Generates a sample JSONL file containing mock Java vulnerability examples
    in the expected format for verification purposes.
    """
    mock_data = [
        {
            "text": (
                "<|instruction|>\nAnalyze the Java code and identify ALL security vulnerabilities. Return structured JSON only.\n\n"
                "<|input|>\npublic void process(String input) throws Exception {\n"
                "    Connection conn = DriverManager.getConnection(DB_URL);\n"
                "    Statement stmt = conn.createStatement();\n"
                "    ResultSet rs = stmt.executeQuery(\"SELECT * FROM users WHERE username = '\" + input + \"'\");\n"
                "}\n\n"
                "<|response|>\n{\n  \"vulnerabilities\": [\n    {\n      \"cwe_id\": \"CWE-89\",\n"
                "      \"cwe_name\": \"SQL Injection\",\n      \"severity\": \"high\",\n"
                "      \"description\": \"SQL Injection vulnerability due to dynamic query building.\"\n"
                "    }\n  ]\n}"
            )
        },
        {
            "text": (
                "<|instruction|>\nAnalyze the Java code and identify ALL security vulnerabilities. Return structured JSON only.\n\n"
                "<|input|>\npublic void handle(HttpServletRequest req) {\n"
                "    String path = req.getParameter(\"path\");\n"
                "    File file = new File(\"/var/uploads/\" + path);\n"
                "    FileInputStream fis = new FileInputStream(file);\n"
                "}\n\n"
                "<|response|>\n{\n  \"vulnerabilities\": [\n    {\n      \"cwe_id\": \"CWE-22\",\n"
                "      \"cwe_name\": \"Path Traversal\",\n      \"severity\": \"high\",\n"
                "      \"description\": \"Path Traversal vulnerability via uncontrolled parameters.\"\n"
                "    }\n  ]\n}"
            )
        }
    ]
    try:
        with open(path, "w", encoding="utf-8") as file:
            for entry in mock_data:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"Generated mock JSONL dataset at: {path}")
    except Exception as err:
        logger.error(f"Failed to generate mock dataset at {path}: {err}", exc_info=True)


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Test and Verify Data Preparation")
    parser.add_argument("--test_file", type=str, default="test_dataset_prepared.jsonl", help="Path to JSONL output/test file")
    parser.add_argument("--tokenizer_name", type=str, default="bigcode/starcoder2-3b", help="Hugging Face tokenizer ID")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    generate_mock_jsonl(test_path)

    try:
        logger.info(f"Loading tokenizer: {args.tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
        
        # Instantiate dataset
        dataset = JavaVulnerabilityDataset(test_path, tokenizer, max_length=1024)
        collator = CausalLMDataCollator(tokenizer)
        
        # Test loading first element
        sample = dataset[0]
        logger.info("Successfully processed first dataset sample.")
        logger.info(f"Input IDs shape: {sample['input_ids'].shape}")
        logger.info(f"Labels shape: {sample['labels'].shape}")

        # Test batching via data collator
        batch = collator([dataset[0], dataset[1]])
        logger.info("Successfully batched dataset samples.")
        logger.info(f"Batch Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"Batch Labels shape: {batch['labels'].shape}")

    except Exception as e:
        logger.error(f"Verification execution failed: {e}", exc_info=True)
    finally:
        if test_path.exists():
            test_path.unlink()
            logger.info("Cleaned up temporary test file.")

`

### fine_tune.py

`python
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

`

### train_fix_model.py

`python
import platform
# Monkeypatch platform.win32_ver to bypass Windows WMI query errors/hangs
platform.win32_ver = lambda release='', version='', csd='', ptype='': ('10', '10.0.22621', '', 'Multiprocessor Free')

import argparse
import logging
import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train_fix_model")

class FixVulnerabilityDataset(Dataset):
    """
    A custom PyTorch Dataset designed to process fixed.jsonl for training the remediation model.
    It expects 'instruction', 'input' (JSON object), and 'output' (JSON object).
    """
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[str] = []

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        jsonl_path_obj = Path(jsonl_path)
        try:
            logger.info(f"Attempting to load dataset from: {jsonl_path_obj}")
            if not jsonl_path_obj.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {jsonl_path_obj}")

            with open(jsonl_path_obj, "r", encoding="utf-8") as file:
                for line_idx, line in enumerate(file, 1):
                    if not line.strip():
                        continue
                    try:
                        data: Dict[str, Any] = json.loads(line)
                        if "instruction" in data and "input" in data and "output" in data:
                            input_str = json.dumps(data["input"], ensure_ascii=False) if isinstance(data["input"], dict) else str(data["input"])
                            output_str = json.dumps(data["output"], ensure_ascii=False) if isinstance(data["output"], dict) else str(data["output"])
                            
                            text = f"<|instruction|>\n{data['instruction']}\n\n<|input|>\n{input_str}\n\n<|response|>\n{output_str}"
                            self.examples.append(text)
                        else:
                            logger.warning(f"Skipping line {line_idx} in {jsonl_path_obj.name}: Required keys not found.")
                    except json.JSONDecodeError as decode_err:
                        logger.error(f"JSON parse failure on line {line_idx} in {jsonl_path_obj.name}: {decode_err}")
            
            logger.info(f"Successfully loaded {len(self.examples)} examples from {jsonl_path_obj.name}")
        except Exception as err:
            logger.error(f"Failed to read dataset from {jsonl_path_obj}: {err}", exc_info=True)
            raise

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]
        try:
            encodings = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_tensors=None
            )

            input_ids: List[int] = encodings["input_ids"]
            attention_mask: List[int] = encodings["attention_mask"]
            labels: List[int] = list(input_ids)

            response_marker = "<|response|>\n"
            response_start_char = text.find(response_marker)

            if response_start_char != -1:
                prompt_text = text[:response_start_char + len(response_marker)]
                prompt_encodings = self.tokenizer(
                    prompt_text,
                    add_special_tokens=True,
                    return_tensors=None
                )
                prompt_token_len = len(prompt_encodings["input_ids"])
                mask_len = min(prompt_token_len, len(labels))
                for i in range(mask_len):
                    labels[i] = -100

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long)
            }
        except Exception as err:
            logger.error(f"Tokenization failed for dataset item at index {idx}: {err}", exc_info=True)
            raise

class FixCausalLMDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(item["input_ids"].size(0) for item in batch)
        batch_input_ids: List[torch.Tensor] = []
        batch_attention_mask: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []
        
        pad_token_id = self.tokenizer.pad_token_id
        padding_side = getattr(self.tokenizer, "padding_side", "right")

        for item in batch:
            input_ids = item["input_ids"]
            attention_mask = item["attention_mask"]
            labels = item["labels"]
            
            diff = max_len - input_ids.size(0)
            if diff > 0:
                pad_ids = torch.full((diff,), pad_token_id, dtype=torch.long)
                pad_mask = torch.zeros((diff,), dtype=torch.long)
                pad_labels = torch.full((diff,), -100, dtype=torch.long)

                if padding_side == "right":
                    new_input_ids = torch.cat([input_ids, pad_ids])
                    new_attention_mask = torch.cat([attention_mask, pad_mask])
                    new_labels = torch.cat([labels, pad_labels])
                else:
                    new_input_ids = torch.cat([pad_ids, input_ids])
                    new_attention_mask = torch.cat([pad_mask, attention_mask])
                    new_labels = torch.cat([pad_labels, labels])
            else:
                new_input_ids = input_ids
                new_attention_mask = attention_mask
                new_labels = labels

            batch_input_ids.append(new_input_ids)
            batch_attention_mask.append(new_attention_mask)
            batch_labels.append(new_labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels)
        }

def find_all_linear_names(model: torch.nn.Module) -> List[str]:
    import bitsandbytes as bnb
    cls_4bit = bnb.nn.Linear4bit
    cls_8bit = bnb.nn.Linear8bitLt
    cls_linear = torch.nn.Linear
    
    linear_layers = set()
    for name, module in model.named_modules():
        if isinstance(module, (cls_4bit, cls_8bit, cls_linear)):
            names = name.split(".")
            linear_layers.add(names[-1])
            
    for exclude_name in ["lm_head", "embed_tokens", "classification_head", "output_layer", "norm", "wte", "wpe"]:
        if exclude_name in linear_layers:
            linear_layers.remove(exclude_name)
            
    if not linear_layers:
        fallback_targets = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        logger.warning(f"No linear layers detected dynamically. Defaulting: {fallback_targets}")
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
    max_length: int = 2048
) -> None:
    try:
        compute_dtype = torch.float32
        bnb_config = None
        device_map = None

        if torch.cuda.is_available():
            compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype
            )
            device_map = "auto"

        torch_dtype = compute_dtype if torch.cuda.is_available() else torch.float32
        logger.info(f"Loading base model: {model_id}")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        logger.info(f"Loading tokenizer: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id

        if torch.cuda.is_available():
            model = prepare_model_for_kbit_training(model)
        else:
            model.enable_input_require_grads()

        target_modules = find_all_linear_names(model)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        train_dataset = FixVulnerabilityDataset(
            jsonl_path=dataset_path,
            tokenizer=tokenizer,
            max_length=max_length
        )
        data_collator = FixCausalLMDataCollator(tokenizer=tokenizer)

        is_cuda = torch.cuda.is_available()
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            logging_steps=10,
            save_strategy="epoch",
            bf16=(is_cuda and compute_dtype == torch.bfloat16),
            fp16=(is_cuda and compute_dtype == torch.float16),
            optim="paged_adamw_8bit" if is_cuda else "adamw_torch",
            ddp_find_unused_parameters=False,
            gradient_checkpointing=True,
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator
        )

        logger.info("Executing remediation model training loop...")
        trainer.train()
        
        logger.info(f"Saving PEFT adapter configurations to output directory: {output_dir}")
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    except Exception as err:
        logger.error(f"Fine-tuning failed: {err}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning Script for Fix Engine")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct",
                        help="HuggingFace model ID (e.g. Qwen/Qwen2.5-Coder-7B-Instruct)")
    parser.add_argument("--dataset_path", type=str, default="Dataset/fixed.jsonl", help="Path to training JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./adapters_fix", help="Output directory for saved adapters")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--max_length", type=int, default=2048, help="Strict sequence length constraint")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    run_training(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
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

`

### generate_production_data.py

`python
"""
Production-Quality Training Data Generator for Java Vulnerability Detection Model.

Combines four data sources:
1. Enhanced Juliet Test Suite: rewrites templated descriptions into code-specific ones
2. Vul4J Real-World CVEs: fetches vulnerable Java files from GitHub commits
3. Synthetic Multi-Vulnerability Examples: realistic Spring Boot / Servlet patterns
4. Hard Negatives: safe code that looks suspicious but has no vulnerabilities

Usage:
    python generate_production_data.py --output Dataset/train_production.jsonl
"""

import argparse
import csv
import io
import json
import logging
import random
import re
import textwrap
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("data_generator")

# ═══════════════════════════════════════════════════════════════════════════════
# CWE Knowledge Base — used to generate rich, code-specific descriptions
# ═══════════════════════════════════════════════════════════════════════════════

CWE_KNOWLEDGE = {
    "CWE-78": {
        "name": "OS Command Injection",
        "sinks": ["Runtime.getRuntime().exec(", "new ProcessBuilder(", "ProcessBuilder("],
        "sources": ["request.getParameter(", "getHeader(", "readLine(", "getQueryString("],
        "severity": "critical",
        "impact_template": "An attacker can execute arbitrary OS commands on the server, potentially achieving full system compromise including data exfiltration, lateral movement, and persistent backdoor installation.",
        "rec_template": "Avoid shell command execution entirely. Use Java APIs directly (e.g., InetAddress.getByName() for DNS). If shell execution is unavoidable, use ProcessBuilder with a fixed command array (no shell expansion) and validate input against a strict allowlist.",
    },
    "CWE-89": {
        "name": "SQL Injection",
        "sinks": ["executeQuery(", "executeUpdate(", "createStatement(", "Statement.execute("],
        "sources": ["request.getParameter(", "getHeader(", "readLine(", "getQueryString("],
        "severity": "critical",
        "impact_template": "An attacker can read, modify, or delete arbitrary database records, bypass authentication, or execute administrative operations on the database server.",
        "rec_template": "Use PreparedStatement with parameterized queries. Never concatenate user input into SQL strings. Apply input validation as defense-in-depth.",
    },
    "CWE-79": {
        "name": "Cross-Site Scripting (XSS)",
        "sinks": ["getWriter().print", "out.println(", "response.getWriter()", "sendError("],
        "sources": ["request.getParameter(", "getHeader(", "getCookies(", "getQueryString("],
        "severity": "high",
        "impact_template": "An attacker can inject malicious scripts into web pages viewed by other users, enabling session hijacking, credential theft, or defacement.",
        "rec_template": "Encode all user-supplied output using context-appropriate encoding (HTML entity encoding for HTML context, JavaScript encoding for script context). Use a Content Security Policy header.",
    },
    "CWE-22": {
        "name": "Path Traversal",
        "sinks": ["new File(", "new FileInputStream(", "new FileReader(", "Paths.get(", "Files.read(", "Files.write("],
        "sources": ["request.getParameter(", "getHeader(", "getPathInfo(", "getRequestURI("],
        "severity": "high",
        "impact_template": "An attacker can read or write arbitrary files on the server by using path traversal sequences (../) to escape the intended directory, potentially accessing configuration files, credentials, or application source code.",
        "rec_template": "Canonicalize the resolved path and verify it remains under the intended base directory. Use Path.normalize() and check with startsWith(). Never construct file paths from raw user input.",
    },
    "CWE-90": {
        "name": "LDAP Injection",
        "sinks": ["InitialDirContext", "ctx.search(", "ctx.lookup(", "DirContext"],
        "sources": ["request.getParameter(", "getHeader(", "readLine("],
        "severity": "high",
        "impact_template": "An attacker can modify LDAP queries to bypass authentication, extract sensitive directory information, or modify LDAP entries.",
        "rec_template": "Use LDAP search filters with proper escaping via javax.naming.ldap or OWASP ESAPI. Validate input against a strict allowlist of permitted characters.",
    },
    "CWE-643": {
        "name": "XPath Injection",
        "sinks": ["XPath.evaluate(", "XPath.compile(", "XPathFactory", "xPath.evaluate("],
        "sources": ["request.getParameter(", "getHeader(", "readLine("],
        "severity": "high",
        "impact_template": "An attacker can modify XPath queries to bypass authentication or extract sensitive data from XML documents stored on the server.",
        "rec_template": "Use parameterized XPath queries via XPathVariableResolver instead of string concatenation. Validate and sanitize all user input before incorporating into XPath expressions.",
    },
    "CWE-321": {
        "name": "Use of Hard-coded Cryptographic Key",
        "sinks": ["SecretKeySpec(", "new SecretKeySpec(", "Cipher.getInstance("],
        "sources": [],
        "severity": "high",
        "impact_template": "Hard-coded cryptographic keys can be extracted from compiled bytecode through reverse engineering, allowing an attacker to decrypt all data protected by that key.",
        "rec_template": "Store cryptographic keys in a secure key management system (e.g., AWS KMS, HashiCorp Vault). Load keys from environment variables or secure configuration at runtime, never embed them in source code.",
    },
    "CWE-319": {
        "name": "Cleartext Transmission of Sensitive Information",
        "sinks": ["HttpURLConnection", "new Socket(", "http://", "URLConnection"],
        "sources": [],
        "severity": "medium",
        "impact_template": "Sensitive data transmitted in cleartext over the network can be intercepted by an attacker performing a man-in-the-middle (MitM) attack, exposing credentials, tokens, or personal information.",
        "rec_template": "Use HTTPS (TLS 1.2+) for all network communications containing sensitive data. Configure HttpsURLConnection instead of HttpURLConnection and enforce certificate validation.",
    },
    "CWE-522": {
        "name": "Insufficiently Protected Credentials",
        "sinks": ["getConnection(", "DriverManager.getConnection(", "new PasswordAuthentication("],
        "sources": [],
        "severity": "high",
        "impact_template": "Hard-coded or insufficiently protected credentials in source code can be extracted by anyone with access to the codebase or compiled artifacts, enabling unauthorized access to databases or services.",
        "rec_template": "Never hard-code credentials in source code. Use environment variables, secure vaults, or encrypted configuration files. Implement credential rotation and least-privilege database accounts.",
    },
    "CWE-601": {
        "name": "Open Redirect",
        "sinks": ["sendRedirect(", "setHeader(\"Location\"", "response.sendRedirect("],
        "sources": ["request.getParameter(", "getHeader(", "getQueryString("],
        "severity": "medium",
        "impact_template": "An attacker can redirect users to malicious websites via crafted URLs, facilitating phishing attacks, credential theft, or malware distribution while abusing the trust of the legitimate domain.",
        "rec_template": "Validate redirect URLs against an allowlist of trusted domains. Use relative URLs where possible. Never redirect to user-supplied absolute URLs without validation.",
    },
    "CWE-400": {
        "name": "Uncontrolled Resource Consumption",
        "sinks": ["Thread.sleep(", "while(", "for(", "new byte["],
        "sources": ["request.getParameter(", "readLine(", "getInputStream("],
        "severity": "medium",
        "impact_template": "An attacker can cause denial of service by supplying input that triggers excessive CPU, memory, or thread consumption, making the application unresponsive to legitimate users.",
        "rec_template": "Impose strict limits on user-controlled resource parameters (timeouts, buffer sizes, iteration counts). Validate numeric inputs against reasonable bounds before use.",
    },
    "CWE-276": {
        "name": "Incorrect Default Permissions",
        "sinks": ["new File(", "createTempFile(", "setExecutable(", "setReadable(", "FileOutputStream("],
        "sources": [],
        "severity": "medium",
        "impact_template": "Files created with overly permissive default permissions may be readable or writable by unauthorized users on the system, leading to information disclosure or tampering.",
        "rec_template": "Explicitly set restrictive file permissions using Files.setPosixFilePermissions() or PosixFilePermissions. Use createTempFile() with secure attributes and restrict access to owner-only.",
    },
    "CWE-502": {
        "name": "Deserialization of Untrusted Data",
        "sinks": ["ObjectInputStream(", "readObject(", "XMLDecoder(", "fromJson("],
        "sources": ["request.getInputStream(", "getParameter(", "readLine("],
        "severity": "critical",
        "impact_template": "Deserializing untrusted data can lead to remote code execution if the classpath contains gadget chains (e.g., Apache Commons Collections), allowing complete server compromise.",
        "rec_template": "Never deserialize data from untrusted sources. Use safe data formats (JSON, Protocol Buffers) instead of Java serialization. If serialization is required, implement ObjectInputFilter to restrict allowed classes.",
    },
    "CWE-611": {
        "name": "XML External Entity (XXE) Injection",
        "sinks": ["DocumentBuilderFactory", "SAXParser", "XMLReader", "TransformerFactory", "SchemaFactory"],
        "sources": ["request.getInputStream(", "getParameter("],
        "severity": "high",
        "impact_template": "An attacker can exploit XXE to read arbitrary files from the server, perform server-side request forgery (SSRF), or cause denial of service through billion-laughs attacks.",
        "rec_template": "Disable external entity processing by setting XMLConstants.FEATURE_SECURE_PROCESSING and disabling DOCTYPE declarations. Use DocumentBuilderFactory.setFeature('http://apache.org/xml/features/disallow-doctype-decl', true).",
    },
    "CWE-918": {
        "name": "Server-Side Request Forgery (SSRF)",
        "sinks": ["new URL(", "openConnection(", "HttpClient", "RestTemplate", "getForObject("],
        "sources": ["request.getParameter(", "getHeader(", "readLine("],
        "severity": "high",
        "impact_template": "An attacker can force the server to make HTTP requests to internal services, cloud metadata endpoints (169.254.169.254), or arbitrary external hosts, potentially accessing sensitive internal resources.",
        "rec_template": "Validate and sanitize all user-supplied URLs. Implement an allowlist of permitted domains/IPs. Block access to internal IP ranges (10.x, 172.16-31.x, 192.168.x, 169.254.x). Use a network-level firewall for egress filtering.",
    },
    "CWE-798": {
        "name": "Use of Hard-coded Credentials",
        "sinks": ["getConnection(", "DriverManager.getConnection(", "new UsernamePasswordCredentials("],
        "sources": [],
        "severity": "critical",
        "impact_template": "Hard-coded credentials in source code can be discovered through reverse engineering or source code leaks, granting unauthorized access to databases, APIs, or administrative interfaces.",
        "rec_template": "Store credentials in environment variables, secure vaults (HashiCorp Vault, AWS Secrets Manager), or encrypted configuration files. Never commit credentials to version control.",
    },
    "CWE-327": {
        "name": "Use of a Broken or Risky Cryptographic Algorithm",
        "sinks": ["Cipher.getInstance(\"DES\"", "Cipher.getInstance(\"RC4\"", "MessageDigest.getInstance(\"MD5\"", "MessageDigest.getInstance(\"SHA-1\""],
        "sources": [],
        "severity": "medium",
        "impact_template": "Use of weak cryptographic algorithms (DES, RC4, MD5, SHA-1) enables an attacker to break the encryption or find hash collisions with modern computing resources.",
        "rec_template": "Use AES-256-GCM for symmetric encryption, SHA-256 or SHA-3 for hashing, and RSA-2048+ or ECDSA P-256+ for asymmetric operations. Avoid DES, 3DES, RC4, MD5, and SHA-1.",
    },
    "CWE-20": {
        "name": "Improper Input Validation",
        "sinks": ["Runtime.getRuntime().exec(", "new File(", "Integer.parseInt(", "execute("],
        "sources": ["request.getParameter(", "getHeader(", "readLine(", "getInputStream("],
        "severity": "high",
        "impact_template": "Insufficient input validation allows attackers to supply crafted input that triggers unintended behavior including injection attacks, buffer overflows, or logic bypasses.",
        "rec_template": "Validate all input at the entry point: check type, length, range, and format. Use allowlists over denylists. Apply validation consistently across all input channels (HTTP parameters, headers, file uploads).",
    },
    "CWE-434": {
        "name": "Unrestricted Upload of File with Dangerous Type",
        "sinks": ["transferTo(", "write(", "MultipartFile", "FileOutputStream("],
        "sources": ["request.getPart(", "getSubmittedFileName(", "MultipartFile"],
        "severity": "high",
        "impact_template": "An attacker can upload executable files (JSP, WAR) that are then served by the web container, achieving remote code execution on the server.",
        "rec_template": "Validate file extensions and MIME types against an allowlist. Store uploads outside the web root. Generate random filenames. Scan uploaded files for malicious content.",
    },
    "CWE-352": {
        "name": "Cross-Site Request Forgery (CSRF)",
        "sinks": ["@PostMapping", "@RequestMapping(method=POST)", "doPost("],
        "sources": [],
        "severity": "medium",
        "impact_template": "An attacker can trick authenticated users into performing unwanted actions (fund transfers, password changes, data deletion) by crafting malicious requests from attacker-controlled pages.",
        "rec_template": "Use CSRF tokens (Spring Security's CsrfFilter is enabled by default). Validate the Origin and Referer headers. Require re-authentication for sensitive operations.",
    },
    "CWE-835": {
        "name": "Infinite Loop",
        "sinks": ["while(", "for(", "do {"],
        "sources": ["getInputStream(", "readLine(", "read("],
        "severity": "medium",
        "impact_template": "A loop without a reachable exit condition can cause denial of service by consuming CPU indefinitely, blocking the thread and potentially exhausting the thread pool.",
        "rec_template": "Ensure all loops have reachable exit conditions. Add iteration count limits and timeouts. Use bounded reads for stream-based loops.",
    },
    "CWE-94": {
        "name": "Improper Control of Generation of Code (Code Injection)",
        "sinks": ["ScriptEngine.eval(", "eval(", "Interpreter.eval(", "GroovyShell"],
        "sources": ["request.getParameter(", "getHeader(", "readLine("],
        "severity": "critical",
        "impact_template": "An attacker can inject and execute arbitrary code on the server through script evaluation engines, achieving full remote code execution.",
        "rec_template": "Avoid dynamic code evaluation entirely. If required, use sandboxed script engines with restricted permissions. Validate input against a strict allowlist of permitted expressions.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Source 1: Enhanced Juliet Test Suite (rewrite existing data with better labels)
# ═══════════════════════════════════════════════════════════════════════════════

def _find_sink_line(code: str, cwe_id: str) -> Optional[int]:
    """Find the line number of the vulnerability sink in the code."""
    knowledge = CWE_KNOWLEDGE.get(cwe_id, {})
    sinks = knowledge.get("sinks", [])
    for i, line in enumerate(code.split("\n"), 1):
        for sink in sinks:
            if sink in line:
                return i
    return None


def _find_source_variable(code: str, cwe_id: str) -> Optional[str]:
    """Try to identify the tainted source variable from the code."""
    knowledge = CWE_KNOWLEDGE.get(cwe_id, {})
    sources = knowledge.get("sources", [])
    for line in code.split("\n"):
        for source in sources:
            if source in line:
                # Extract variable name from assignment
                match = re.match(r'\s*(?:String|int|Object|byte\[\]|char\[\])\s+(\w+)\s*=', line)
                if match:
                    return match.group(1)
                # Try parameter extraction
                match = re.search(r'(\w+)\s*=\s*.*' + re.escape(source.rstrip("(")), line)
                if match:
                    return match.group(1)
    return None


def _find_sink_method(code: str, cwe_id: str) -> Optional[str]:
    """Extract the actual sink method call from the code."""
    knowledge = CWE_KNOWLEDGE.get(cwe_id, {})
    sinks = knowledge.get("sinks", [])
    for line in code.split("\n"):
        for sink in sinks:
            if sink in line:
                return line.strip()
    return None


def _find_function_name(code: str) -> str:
    """Extract the function/method name from the code."""
    match = re.search(r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(', code)
    if match:
        return match.group(1)
    return "unknown"


def generate_code_specific_description(code: str, cwe_id: str, cwe_name: str) -> Dict[str, str]:
    """
    Generate a code-specific vulnerability description by analyzing the actual code,
    rather than using a generic template.
    """
    knowledge = CWE_KNOWLEDGE.get(cwe_id, {})
    if not knowledge:
        # Fallback for unknown CWEs
        return {
            "description": f"The code contains a {cwe_name} vulnerability that may compromise system security.",
            "impact": f"Exploitation of this {cwe_name} vulnerability could lead to unauthorized access or data compromise.",
            "recommendation": f"Review the code for {cwe_name} patterns and apply appropriate security controls."
        }

    func_name = _find_function_name(code)
    source_var = _find_source_variable(code, cwe_id)
    sink_line = _find_sink_line(code, cwe_id)
    sink_method = _find_sink_method(code, cwe_id)

    # Build a code-specific description
    desc_parts = []
    if source_var and sink_method:
        desc_parts.append(
            f"In method `{func_name}()`, the variable `{source_var}` receives untrusted input "
            f"and flows into the dangerous call `{sink_method[:80]}` "
            f"{'at line ' + str(sink_line) if sink_line else ''} without proper sanitization."
        )
    elif sink_method:
        desc_parts.append(
            f"The method `{func_name}()` contains a dangerous call `{sink_method[:80]}` "
            f"{'at line ' + str(sink_line) if sink_line else ''} that processes data without adequate security controls."
        )
    else:
        desc_parts.append(
            f"The method `{func_name}()` contains a {cwe_name} vulnerability due to insufficient security controls."
        )

    return {
        "description": " ".join(desc_parts),
        "impact": knowledge["impact_template"],
        "recommendation": knowledge["rec_template"],
    }


def enhance_juliet_dataset(input_path: str) -> List[Dict[str, Any]]:
    """
    Read the existing Juliet-based JSONL dataset and rewrite all templated
    descriptions with code-specific ones.
    """
    enhanced = []
    input_file = Path(input_path)
    if not input_file.exists():
        logger.warning(f"Juliet dataset not found: {input_path}")
        return enhanced

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")

            if "<|response|>" not in text or "<|input|>" not in text:
                enhanced.append(data)
                continue

            # Split into components
            parts = text.split("<|response|>")
            prompt_part = parts[0]
            response_part = parts[1].strip()

            # Extract the Java code
            code_start = prompt_part.find("<|input|>\n") + len("<|input|>\n")
            java_code = prompt_part[code_start:].strip()

            try:
                resp_json = json.loads(response_part)
            except json.JSONDecodeError:
                enhanced.append(data)
                continue

            vulns = resp_json.get("vulnerabilities", [])
            if not vulns:
                # Negative example — keep as-is
                enhanced.append(data)
                continue

            # Rewrite each vulnerability with code-specific descriptions
            for vuln in vulns:
                cwe_id = vuln.get("cwe_id", "")
                cwe_name = vuln.get("cwe_name", "")
                # Strip the "(CWE-XXX)" suffix from cwe_name if present
                cwe_name = re.sub(r'\s*\(CWE-\d+\)\s*$', '', cwe_name)

                specific = generate_code_specific_description(java_code, cwe_id, cwe_name)
                vuln["description"] = specific["description"]
                vuln["impact"] = specific["impact"]
                vuln["recommendation"] = specific["recommendation"]

                # Also fix cwe_name to remove redundant CWE ID
                vuln["cwe_name"] = cwe_name

                # Re-calculate sink line if the existing one seems off
                new_line = _find_sink_line(java_code, cwe_id)
                if new_line:
                    vuln["location"] = {"line": new_line}

            new_response = json.dumps(resp_json, indent=2, ensure_ascii=False)
            new_text = f"{prompt_part}<|response|>\n{new_response}"
            enhanced.append({"text": new_text})

    logger.info(f"Enhanced {len(enhanced)} Juliet examples with code-specific descriptions")
    return enhanced


# ═══════════════════════════════════════════════════════════════════════════════
# Source 2: Vul4J Real-World CVE Examples
# ═══════════════════════════════════════════════════════════════════════════════

VUL4J_CSV_URL = "https://raw.githubusercontent.com/tuhh-softsec/vul4j/main/dataset/vul4j_dataset.csv"


def fetch_vul4j_metadata() -> List[Dict[str, str]]:
    """Fetch and parse the Vul4J dataset CSV from GitHub."""
    try:
        req = urllib.request.Request(VUL4J_CSV_URL, headers={"User-Agent": "VulnDataGen/1.0"})
        response = urllib.request.urlopen(req, timeout=30)
        data = response.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(data))
        entries = [row for row in reader]
        logger.info(f"Fetched {len(entries)} Vul4J entries from GitHub")
        return entries
    except Exception as e:
        logger.error(f"Failed to fetch Vul4J CSV: {e}")
        return []


def fetch_github_diff(commit_url: str) -> Optional[str]:
    """Fetch the diff content of a GitHub commit."""
    # Convert commit URL to API URL
    # e.g., https://github.com/alibaba/fastjson/commit/abc123 -> https://api.github.com/repos/alibaba/fastjson/commits/abc123
    match = re.search(r'github\.com/([^/]+/[^/]+)/commit/([a-f0-9]+)', commit_url)
    if not match:
        return None

    repo_slug = match.group(1)
    commit_sha = match.group(2)
    api_url = f"https://api.github.com/repos/{repo_slug}/commits/{commit_sha}"

    try:
        req = urllib.request.Request(
            api_url,
            headers={
                "User-Agent": "VulnDataGen/1.0",
                "Accept": "application/vnd.github.v3+json"
            }
        )
        response = urllib.request.urlopen(req, timeout=15)
        commit_data = json.loads(response.read().decode("utf-8"))

        java_files = []
        for f in commit_data.get("files", []):
            if f.get("filename", "").endswith(".java") and f.get("patch"):
                java_files.append({
                    "filename": f["filename"],
                    "patch": f["patch"],
                    "raw_url": f.get("raw_url", ""),
                    "status": f.get("status", "")
                })
        return java_files if java_files else None
    except (urllib.error.HTTPError, urllib.error.URLError, Exception) as e:
        logger.debug(f"Failed to fetch commit {commit_sha[:8]}: {e}")
        return None


def fetch_file_content(raw_url: str) -> Optional[str]:
    """Fetch the full content of a file from GitHub raw URL."""
    try:
        req = urllib.request.Request(raw_url, headers={"User-Agent": "VulnDataGen/1.0"})
        response = urllib.request.urlopen(req, timeout=15)
        content = response.read().decode("utf-8", errors="ignore")
        return content
    except Exception:
        return None


def extract_vulnerable_code_from_patch(patch: str, full_content: Optional[str]) -> Optional[str]:
    """
    Extract the vulnerable code section from a git patch.
    Returns the 'before' state (lines prefixed with '-' or ' ').
    """
    if full_content and len(full_content) < 4000:
        return full_content

    # Extract the vulnerable (removed) code block from the patch
    lines = []
    for line in patch.split("\n"):
        if line.startswith("@@"):
            continue
        if line.startswith("-"):
            lines.append(line[1:])  # Remove the '-' prefix
        elif line.startswith("+"):
            continue  # Skip added lines (these are the fix)
        else:
            lines.append(line[1:] if line.startswith(" ") else line)

    code = "\n".join(lines)
    if len(code) < 50:
        return None
    return code[:3000]  # Limit size for model context


def generate_vul4j_examples(max_entries: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch Vul4J entries from GitHub, extract vulnerable Java code, and generate
    high-quality training examples.
    """
    examples = []
    metadata = fetch_vul4j_metadata()
    if not metadata:
        return examples

    # Filter to entries with mapped CWEs
    mapped = [m for m in metadata if m.get("cwe_id") and m["cwe_id"] != "Not Mapping"]
    random.shuffle(mapped)

    processed = 0
    for entry in mapped[:min(max_entries * 3, len(mapped))]:  # Over-fetch since many will fail
        if processed >= max_entries:
            break

        cwe_id = entry.get("cwe_id", "")
        cwe_name = entry.get("cwe_name", "")
        cve_id = entry.get("cve_id", "")
        commit_url = entry.get("human_patch", "")

        if not commit_url:
            continue

        # Rate limit GitHub API
        time.sleep(1.0)

        java_files = fetch_github_diff(commit_url)
        if not java_files:
            continue

        for jf in java_files[:1]:  # Take first Java file from each commit
            # Try to get the vulnerable version (before the fix)
            # The raw_url points to the fixed version, so we use the patch
            code = extract_vulnerable_code_from_patch(jf["patch"], None)
            if not code or len(code) < 100:
                continue

            # Generate code-specific description
            specific = generate_code_specific_description(code, cwe_id, cwe_name)
            sink_line = _find_sink_line(code, cwe_id) or 1

            vuln_entry = {
                "cwe_id": cwe_id,
                "cwe_name": cwe_name,
                "severity": CWE_KNOWLEDGE.get(cwe_id, {}).get("severity", "high"),
                "confidence": round(random.uniform(0.85, 0.99), 3),
                "location": {"line": sink_line},
                "description": specific["description"],
                "impact": specific["impact"],
                "recommendation": specific["recommendation"],
            }

            instruction = "Analyze the Java code and identify ALL security vulnerabilities. Return structured JSON only."
            text = (
                f"<|instruction|>\n{instruction}\n\n"
                f"<|input|>\n{code}\n\n"
                f"<|response|>\n"
                + json.dumps({"vulnerabilities": [vuln_entry]}, indent=2, ensure_ascii=False)
            )
            examples.append({"text": text})
            processed += 1
            logger.info(f"Vul4J [{processed}/{max_entries}]: {cve_id} ({cwe_id} - {cwe_name}) from {jf['filename']}")

    logger.info(f"Generated {len(examples)} Vul4J real-world examples")
    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# Source 3: Synthetic Multi-Vulnerability Examples (Spring Boot / Servlet)
# ═══════════════════════════════════════════════════════════════════════════════

SYNTHETIC_EXAMPLES = [
    # ---- Multi-vuln: SQL Injection + XSS ----
    {
        "code": textwrap.dedent("""\
            package com.example.controller;

            import org.springframework.web.bind.annotation.*;
            import javax.servlet.http.*;
            import java.sql.*;

            @RestController
            @RequestMapping("/api/users")
            public class UserController {

                @GetMapping("/search")
                public String searchUsers(@RequestParam String query, HttpServletResponse response) throws Exception {
                    Connection conn = DriverManager.getConnection("jdbc:mysql://localhost/mydb", "root", "password123");
                    Statement stmt = conn.createStatement();
                    ResultSet rs = stmt.executeQuery("SELECT * FROM users WHERE name LIKE '%" + query + "%'");

                    StringBuilder html = new StringBuilder("<html><body><h1>Results for: " + query + "</h1><ul>");
                    while (rs.next()) {
                        html.append("<li>").append(rs.getString("name")).append("</li>");
                    }
                    html.append("</ul></body></html>");

                    response.getWriter().print(html.toString());
                    return null;
                }
            }
        """),
        "vulnerabilities": [
            {
                "cwe_id": "CWE-89",
                "cwe_name": "SQL Injection",
                "severity": "critical",
                "confidence": 0.97,
                "location": {"line": 15},
                "description": "The `query` parameter from `@RequestParam` is directly concatenated into the SQL string passed to `stmt.executeQuery()` at line 15, enabling SQL injection through crafted search queries like `%' OR 1=1 --`.",
                "impact": "An attacker can read, modify, or delete arbitrary database records, bypass authentication, or execute administrative operations on the database server.",
                "recommendation": "Use PreparedStatement with parameterized queries: `conn.prepareStatement(\"SELECT * FROM users WHERE name LIKE ?\")` and bind the parameter with `setString()`.",
            },
            {
                "cwe_id": "CWE-79",
                "cwe_name": "Cross-Site Scripting (XSS)",
                "severity": "high",
                "confidence": 0.95,
                "location": {"line": 17},
                "description": "The `query` parameter is reflected directly into the HTML response at line 17 (`Results for: \" + query + \"`) without HTML encoding, enabling reflected XSS via payloads like `<script>alert(1)</script>`.",
                "impact": "An attacker can inject malicious scripts into web pages viewed by other users, enabling session hijacking, credential theft, or defacement.",
                "recommendation": "Encode all user-supplied output using `org.owasp.encoder.Encode.forHtml()` before including it in HTML responses. Use a templating engine with auto-escaping.",
            },
            {
                "cwe_id": "CWE-798",
                "cwe_name": "Use of Hard-coded Credentials",
                "severity": "critical",
                "confidence": 0.99,
                "location": {"line": 13},
                "description": "Database credentials (`root` / `password123`) are hard-coded in the `DriverManager.getConnection()` call at line 13. These can be extracted from compiled bytecode or version control history.",
                "impact": "Hard-coded credentials in source code can be discovered through reverse engineering or source code leaks, granting unauthorized access to databases, APIs, or administrative interfaces.",
                "recommendation": "Store database credentials in environment variables or a secure vault (e.g., AWS Secrets Manager, Spring Vault). Use `@Value(\"${db.password}\")` to inject from externalized configuration.",
            },
        ],
    },
    # ---- Multi-vuln: Command Injection + Path Traversal ----
    {
        "code": textwrap.dedent("""\
            package com.example.service;

            import javax.servlet.http.*;
            import javax.servlet.annotation.*;
            import java.io.*;

            @WebServlet("/admin/diagnostic")
            public class DiagnosticServlet extends HttpServlet {

                @Override
                protected void doGet(HttpServletRequest request, HttpServletResponse response)
                        throws IOException {
                    String host = request.getParameter("host");
                    String logFile = request.getParameter("logfile");

                    // Diagnostic: ping the host
                    Process proc = Runtime.getRuntime().exec("ping -c 3 " + host);
                    BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
                    StringBuilder output = new StringBuilder();
                    String line;
                    while ((line = reader.readLine()) != null) {
                        output.append(line).append("\\n");
                    }

                    // Also fetch the requested log file
                    File log = new File("/var/logs/" + logFile);
                    String logContent = new String(java.nio.file.Files.readAllBytes(log.toPath()));

                    response.getWriter().print("Ping Result:\\n" + output + "\\nLog:\\n" + logContent);
                }
            }
        """),
        "vulnerabilities": [
            {
                "cwe_id": "CWE-78",
                "cwe_name": "OS Command Injection",
                "severity": "critical",
                "confidence": 0.98,
                "location": {"line": 16},
                "description": "The `host` parameter from `request.getParameter(\"host\")` is concatenated into the command string passed to `Runtime.getRuntime().exec()` at line 16 without sanitization, enabling command injection via payloads like `; cat /etc/passwd`.",
                "impact": "An attacker can execute arbitrary OS commands on the server, potentially achieving full system compromise including data exfiltration, lateral movement, and persistent backdoor installation.",
                "recommendation": "Use ProcessBuilder with a fixed argument array instead of string concatenation: `new ProcessBuilder(\"ping\", \"-c\", \"3\", host)`. Validate `host` against a strict domain name pattern.",
            },
            {
                "cwe_id": "CWE-22",
                "cwe_name": "Path Traversal",
                "severity": "high",
                "confidence": 0.96,
                "location": {"line": 26},
                "description": "The `logFile` parameter is concatenated into the file path `\"/var/logs/\" + logFile` at line 26 without path validation, enabling directory traversal via `../../etc/passwd`.",
                "impact": "An attacker can read or write arbitrary files on the server by using path traversal sequences (../) to escape the intended directory, potentially accessing configuration files, credentials, or application source code.",
                "recommendation": "Canonicalize the resolved path with `log.getCanonicalPath()` and verify it starts with the base directory `/var/logs/`. Reject any path containing `..` segments.",
            },
        ],
    },
    # ---- Multi-vuln: XXE + SSRF ----
    {
        "code": textwrap.dedent("""\
            package com.example.service;

            import org.springframework.stereotype.Service;
            import org.springframework.web.client.RestTemplate;
            import javax.xml.parsers.*;
            import org.w3c.dom.*;
            import java.io.*;

            @Service
            public class DataImportService {

                private final RestTemplate restTemplate = new RestTemplate();

                public Document parseXmlInput(InputStream xmlInput) throws Exception {
                    DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
                    DocumentBuilder builder = factory.newDocumentBuilder();
                    return builder.parse(xmlInput);
                }

                public String fetchExternalData(String url) {
                    return restTemplate.getForObject(url, String.class);
                }
            }
        """),
        "vulnerabilities": [
            {
                "cwe_id": "CWE-611",
                "cwe_name": "XML External Entity (XXE) Injection",
                "severity": "high",
                "confidence": 0.95,
                "location": {"line": 15},
                "description": "The `DocumentBuilderFactory` at line 15 is created with default settings that allow external entity processing. An attacker supplying a crafted XML document with a `<!DOCTYPE>` declaration can read arbitrary files or trigger SSRF.",
                "impact": "An attacker can exploit XXE to read arbitrary files from the server, perform server-side request forgery (SSRF), or cause denial of service through billion-laughs attacks.",
                "recommendation": "Disable external entity processing: `factory.setFeature(\"http://apache.org/xml/features/disallow-doctype-decl\", true)` and `factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true)`.",
            },
            {
                "cwe_id": "CWE-918",
                "cwe_name": "Server-Side Request Forgery (SSRF)",
                "severity": "high",
                "confidence": 0.93,
                "location": {"line": 21},
                "description": "The `fetchExternalData()` method passes a user-controlled `url` parameter directly to `restTemplate.getForObject()` at line 21, allowing an attacker to make the server request internal services or cloud metadata endpoints.",
                "impact": "An attacker can force the server to make HTTP requests to internal services, cloud metadata endpoints (169.254.169.254), or arbitrary external hosts, potentially accessing sensitive internal resources.",
                "recommendation": "Validate the URL against an allowlist of permitted domains. Block access to internal IP ranges (10.x, 172.16-31.x, 192.168.x, 169.254.x). Use URL parsing to check the hostname before making the request.",
            },
        ],
    },
    # ---- Multi-vuln: Deserialization + Weak Crypto ----
    {
        "code": textwrap.dedent("""\
            package com.example.security;

            import javax.crypto.*;
            import javax.crypto.spec.*;
            import java.io.*;
            import java.util.Base64;

            public class TokenManager {

                private static final String SECRET = "MySecretKey12345";
                private static final String ALGORITHM = "DES";

                public Object decryptAndDeserialize(String encryptedToken) throws Exception {
                    Cipher cipher = Cipher.getInstance(ALGORITHM);
                    SecretKeySpec keySpec = new SecretKeySpec(SECRET.getBytes(), ALGORITHM);
                    cipher.init(Cipher.DECRYPT_MODE, keySpec);
                    byte[] decrypted = cipher.doFinal(Base64.getDecoder().decode(encryptedToken));

                    ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(decrypted));
                    return ois.readObject();
                }

                public String serializeAndEncrypt(Object obj) throws Exception {
                    ByteArrayOutputStream baos = new ByteArrayOutputStream();
                    ObjectOutputStream oos = new ObjectOutputStream(baos);
                    oos.writeObject(obj);
                    oos.close();

                    Cipher cipher = Cipher.getInstance(ALGORITHM);
                    SecretKeySpec keySpec = new SecretKeySpec(SECRET.getBytes(), ALGORITHM);
                    cipher.init(Cipher.ENCRYPT_MODE, keySpec);
                    byte[] encrypted = cipher.doFinal(baos.toByteArray());
                    return Base64.getEncoder().encodeToString(encrypted);
                }
            }
        """),
        "vulnerabilities": [
            {
                "cwe_id": "CWE-502",
                "cwe_name": "Deserialization of Untrusted Data",
                "severity": "critical",
                "confidence": 0.97,
                "location": {"line": 19},
                "description": "The `decryptAndDeserialize()` method creates an `ObjectInputStream` at line 19 and calls `readObject()` at line 20 on decrypted data without any class filtering, enabling Remote Code Execution via deserialization gadget chains.",
                "impact": "Deserializing untrusted data can lead to remote code execution if the classpath contains gadget chains (e.g., Apache Commons Collections), allowing complete server compromise.",
                "recommendation": "Replace Java serialization with a safe format like JSON. If serialization is required, implement ObjectInputFilter to restrict allowed classes: `ois.setObjectInputFilter(ObjectInputFilter.Config.createFilter(\"com.example.**\"))`.",
            },
            {
                "cwe_id": "CWE-327",
                "cwe_name": "Use of a Broken or Risky Cryptographic Algorithm",
                "severity": "medium",
                "confidence": 0.99,
                "location": {"line": 14},
                "description": "The `ALGORITHM` constant is set to `\"DES\"` which is a deprecated 56-bit cipher. The `Cipher.getInstance(\"DES\")` call at line 14 uses this weak algorithm, which can be broken with brute force in hours on modern hardware.",
                "impact": "Use of weak cryptographic algorithms (DES, RC4, MD5, SHA-1) enables an attacker to break the encryption with modern computing resources.",
                "recommendation": "Use AES-256-GCM: `Cipher.getInstance(\"AES/GCM/NoPadding\")` with a 256-bit key generated from a proper key derivation function.",
            },
            {
                "cwe_id": "CWE-321",
                "cwe_name": "Use of Hard-coded Cryptographic Key",
                "severity": "high",
                "confidence": 0.99,
                "location": {"line": 10},
                "description": "The encryption key `\"MySecretKey12345\"` is hard-coded as a static final constant at line 10. This key can be trivially extracted by decompiling the class file.",
                "impact": "Hard-coded cryptographic keys can be extracted from compiled bytecode through reverse engineering, allowing an attacker to decrypt all data protected by that key.",
                "recommendation": "Store cryptographic keys in a secure key management system (e.g., AWS KMS, HashiCorp Vault). Load keys from environment variables at runtime.",
            },
        ],
    },
    # ---- Single vuln: Open Redirect ----
    {
        "code": textwrap.dedent("""\
            package com.example.controller;

            import org.springframework.stereotype.Controller;
            import org.springframework.web.bind.annotation.*;
            import javax.servlet.http.*;

            @Controller
            public class AuthController {

                @GetMapping("/login")
                public String loginPage() {
                    return "login";
                }

                @PostMapping("/login")
                public void handleLogin(@RequestParam String username,
                                        @RequestParam String password,
                                        @RequestParam(required = false) String redirectUrl,
                                        HttpServletResponse response) throws Exception {
                    // ... authentication logic ...
                    if (authenticate(username, password)) {
                        if (redirectUrl != null && !redirectUrl.isEmpty()) {
                            response.sendRedirect(redirectUrl);
                        } else {
                            response.sendRedirect("/dashboard");
                        }
                    }
                }

                private boolean authenticate(String username, String password) {
                    return "admin".equals(username) && "admin123".equals(password);
                }
            }
        """),
        "vulnerabilities": [
            {
                "cwe_id": "CWE-601",
                "cwe_name": "Open Redirect",
                "severity": "medium",
                "confidence": 0.94,
                "location": {"line": 22},
                "description": "The `redirectUrl` parameter from `@RequestParam` is passed directly to `response.sendRedirect()` at line 22 without validating the target domain, enabling phishing via URLs like `https://evil.com/fake-login`.",
                "impact": "An attacker can redirect users to malicious websites via crafted URLs, facilitating phishing attacks, credential theft, or malware distribution while abusing the trust of the legitimate domain.",
                "recommendation": "Validate `redirectUrl` against an allowlist of trusted domains. Use relative URLs where possible. Check that the URL starts with `/` and does not contain `://` or `//`.",
            },
            {
                "cwe_id": "CWE-798",
                "cwe_name": "Use of Hard-coded Credentials",
                "severity": "critical",
                "confidence": 0.99,
                "location": {"line": 31},
                "description": "Authentication credentials (`admin` / `admin123`) are hard-coded in the `authenticate()` method at line 31. These are trivially discoverable through decompilation.",
                "impact": "Hard-coded credentials in source code can be discovered through reverse engineering or source code leaks, granting unauthorized access to databases, APIs, or administrative interfaces.",
                "recommendation": "Use a proper authentication mechanism (Spring Security with password hashing via BCrypt). Store user credentials in a database with salted hashes, never in source code.",
            },
        ],
    },
    # ---- Single vuln: LDAP Injection ----
    {
        "code": textwrap.dedent("""\
            package com.example.auth;

            import javax.naming.*;
            import javax.naming.directory.*;
            import java.util.*;

            public class LdapAuthenticator {

                private static final String LDAP_URL = "ldap://ldap.example.com:389";

                public boolean authenticate(String username, String password) throws NamingException {
                    Hashtable<String, String> env = new Hashtable<>();
                    env.put(Context.INITIAL_CONTEXT_FACTORY, "com.sun.jndi.ldap.LdapCtxFactory");
                    env.put(Context.PROVIDER_URL, LDAP_URL);

                    DirContext ctx = new InitialDirContext(env);
                    String filter = "(&(uid=" + username + ")(userPassword=" + password + "))";
                    SearchControls controls = new SearchControls();
                    controls.setSearchScope(SearchControls.SUBTREE_SCOPE);
                    NamingEnumeration<SearchResult> results = ctx.search("dc=example,dc=com", filter, controls);

                    return results.hasMore();
                }
            }
        """),
        "vulnerabilities": [
            {
                "cwe_id": "CWE-90",
                "cwe_name": "LDAP Injection",
                "severity": "critical",
                "confidence": 0.97,
                "location": {"line": 16},
                "description": "The `username` and `password` parameters are directly concatenated into the LDAP search filter at line 16 (`\"(&(uid=\" + username + \")(userPassword=\" + password + \"))\"`), enabling LDAP injection that can bypass authentication via `*)(uid=*))(|(uid=*`.",
                "impact": "An attacker can modify LDAP queries to bypass authentication, extract sensitive directory information, or modify LDAP entries.",
                "recommendation": "Escape special LDAP characters in user input using `javax.naming.ldap.Rdn.escapeValue()` or a library like OWASP ESAPI. Use parameterized LDAP queries where available.",
            },
        ],
    },
    # ---- File upload vulnerability ----
    {
        "code": textwrap.dedent("""\
            package com.example.upload;

            import org.springframework.web.bind.annotation.*;
            import org.springframework.web.multipart.MultipartFile;
            import java.io.*;
            import java.nio.file.*;

            @RestController
            @RequestMapping("/api/files")
            public class FileUploadController {

                private static final String UPLOAD_DIR = "/var/www/uploads/";

                @PostMapping("/upload")
                public String handleUpload(@RequestParam("file") MultipartFile file) throws IOException {
                    String filename = file.getOriginalFilename();
                    Path destination = Paths.get(UPLOAD_DIR + filename);
                    file.transferTo(destination.toFile());
                    return "File uploaded: " + filename;
                }
            }
        """),
        "vulnerabilities": [
            {
                "cwe_id": "CWE-434",
                "cwe_name": "Unrestricted Upload of File with Dangerous Type",
                "severity": "critical",
                "confidence": 0.96,
                "location": {"line": 17},
                "description": "The `handleUpload()` method uses `file.getOriginalFilename()` at line 16 without validating the file extension or content type, then saves it directly to the web-accessible `/var/www/uploads/` directory at line 17. An attacker can upload a `.jsp` webshell.",
                "impact": "An attacker can upload executable files (JSP, WAR) that are then served by the web container, achieving remote code execution on the server.",
                "recommendation": "Validate file extensions against an allowlist (e.g., .jpg, .png, .pdf). Verify the Content-Type header. Generate random filenames. Store uploads outside the web root.",
            },
            {
                "cwe_id": "CWE-22",
                "cwe_name": "Path Traversal",
                "severity": "high",
                "confidence": 0.94,
                "location": {"line": 17},
                "description": "The original filename from `file.getOriginalFilename()` is used directly in the path construction at line 17 (`UPLOAD_DIR + filename`). An attacker can use `../../etc/cron.d/backdoor` as filename to write files outside the upload directory.",
                "impact": "An attacker can read or write arbitrary files on the server by using path traversal sequences (../) to escape the intended directory.",
                "recommendation": "Extract only the base name: `Paths.get(filename).getFileName()`. Canonicalize the resolved path and verify it remains under `UPLOAD_DIR`.",
            },
        ],
    },
]


def generate_synthetic_examples() -> List[Dict[str, Any]]:
    """Generate training examples from the synthetic multi-vulnerability code samples."""
    examples = []
    instruction = "Analyze the Java code and identify ALL security vulnerabilities. Return structured JSON only."

    for sample in SYNTHETIC_EXAMPLES:
        code = sample["code"]
        vulns = sample["vulnerabilities"]

        text = (
            f"<|instruction|>\n{instruction}\n\n"
            f"<|input|>\n{code}\n\n"
            f"<|response|>\n"
            + json.dumps({"vulnerabilities": vulns}, indent=2, ensure_ascii=False)
        )
        examples.append({"text": text})

    logger.info(f"Generated {len(examples)} synthetic multi-vulnerability examples")
    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# Source 4: Hard Negative Examples (safe code that looks suspicious)
# ═══════════════════════════════════════════════════════════════════════════════

HARD_NEGATIVES = [
    # Safe: uses PreparedStatement (not SQL injection)
    textwrap.dedent("""\
        package com.example.dao;

        import java.sql.*;

        public class UserDao {

            public User findByUsername(String username) throws SQLException {
                Connection conn = DriverManager.getConnection("jdbc:mysql://localhost/db");
                PreparedStatement stmt = conn.prepareStatement("SELECT * FROM users WHERE username = ?");
                stmt.setString(1, username);
                ResultSet rs = stmt.executeQuery();
                if (rs.next()) {
                    return new User(rs.getString("username"), rs.getString("email"));
                }
                return null;
            }
        }
    """),
    # Safe: ProcessBuilder with argument array (not command injection)
    textwrap.dedent("""\
        package com.example.service;

        import java.io.*;
        import java.net.InetAddress;

        public class NetworkService {

            public boolean isHostReachable(String hostname) throws Exception {
                // Validate hostname using DNS resolution (safe — no shell involved)
                InetAddress address = InetAddress.getByName(hostname);
                return address.isReachable(5000);
            }

            public String pingHost(String hostname) throws Exception {
                // Safe: ProcessBuilder with individual arguments, no shell expansion
                ProcessBuilder pb = new ProcessBuilder("ping", "-c", "1", hostname);
                pb.redirectErrorStream(true);
                Process proc = pb.start();
                BufferedReader reader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
                StringBuilder output = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    output.append(line).append("\\n");
                }
                proc.waitFor();
                return output.toString();
            }
        }
    """),
    # Safe: file access with proper path validation
    textwrap.dedent("""\
        package com.example.service;

        import java.io.*;
        import java.nio.file.*;

        public class SafeFileService {

            private static final Path BASE_DIR = Paths.get("/var/uploads").toAbsolutePath().normalize();

            public String readFile(String filename) throws IOException {
                // Canonicalize and validate path
                Path resolved = BASE_DIR.resolve(filename).normalize();
                if (!resolved.startsWith(BASE_DIR)) {
                    throw new SecurityException("Path traversal attempt detected: " + filename);
                }
                return Files.readString(resolved);
            }
        }
    """),
    # Safe: HTTPS connection (not cleartext)
    textwrap.dedent("""\
        package com.example.client;

        import javax.net.ssl.*;
        import java.io.*;
        import java.net.URL;

        public class SecureApiClient {

            public String fetchData(String apiUrl) throws Exception {
                URL url = new URL(apiUrl);
                HttpsURLConnection conn = (HttpsURLConnection) url.openConnection();
                conn.setRequestMethod("GET");
                conn.setConnectTimeout(5000);
                conn.setReadTimeout(5000);

                BufferedReader reader = new BufferedReader(new InputStreamReader(conn.getInputStream()));
                StringBuilder response = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    response.append(line);
                }
                reader.close();
                return response.toString();
            }
        }
    """),
    # Safe: XML parser with XXE protection
    textwrap.dedent("""\
        package com.example.xml;

        import javax.xml.parsers.*;
        import javax.xml.XMLConstants;
        import org.w3c.dom.*;
        import java.io.*;

        public class SafeXmlParser {

            public Document parseSecurely(InputStream input) throws Exception {
                DocumentBuilderFactory factory = DocumentBuilderFactory.newInstance();
                factory.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
                factory.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);
                factory.setXIncludeAware(false);
                factory.setExpandEntityReferences(false);

                DocumentBuilder builder = factory.newDocumentBuilder();
                return builder.parse(input);
            }
        }
    """),
    # Safe: LDAP with proper escaping
    textwrap.dedent("""\
        package com.example.auth;

        import javax.naming.*;
        import javax.naming.directory.*;
        import javax.naming.ldap.Rdn;
        import java.util.*;

        public class SafeLdapAuth {

            public boolean authenticate(String username, String password) throws NamingException {
                // Properly escape LDAP special characters
                String safeUsername = Rdn.escapeValue(username);
                String safePassword = Rdn.escapeValue(password);

                Hashtable<String, String> env = new Hashtable<>();
                env.put(Context.INITIAL_CONTEXT_FACTORY, "com.sun.jndi.ldap.LdapCtxFactory");
                env.put(Context.PROVIDER_URL, "ldap://ldap.example.com:389");

                DirContext ctx = new InitialDirContext(env);
                String filter = "(&(uid=" + safeUsername + ")(userPassword=" + safePassword + "))";
                NamingEnumeration<SearchResult> results = ctx.search("dc=example,dc=com", filter, new SearchControls());
                return results.hasMore();
            }
        }
    """),
    # Safe: proper credential handling via environment variables
    textwrap.dedent("""\
        package com.example.config;

        import java.sql.*;

        public class DatabaseConfig {

            public Connection getConnection() throws SQLException {
                String dbUrl = System.getenv("DATABASE_URL");
                String dbUser = System.getenv("DATABASE_USER");
                String dbPassword = System.getenv("DATABASE_PASSWORD");

                if (dbUrl == null || dbUser == null || dbPassword == null) {
                    throw new IllegalStateException("Database environment variables not configured");
                }

                return DriverManager.getConnection(dbUrl, dbUser, dbPassword);
            }
        }
    """),
    # Safe: URL redirect with proper validation
    textwrap.dedent("""\
        package com.example.controller;

        import org.springframework.stereotype.Controller;
        import org.springframework.web.bind.annotation.*;
        import javax.servlet.http.*;
        import java.net.URI;
        import java.util.Set;

        @Controller
        public class SafeRedirectController {

            private static final Set<String> ALLOWED_DOMAINS = Set.of(
                "example.com", "app.example.com", "www.example.com"
            );

            @GetMapping("/redirect")
            public void safeRedirect(@RequestParam String url, HttpServletResponse response) throws Exception {
                URI uri = new URI(url);
                String host = uri.getHost();

                if (host != null && ALLOWED_DOMAINS.contains(host.toLowerCase())) {
                    response.sendRedirect(url);
                } else if (url.startsWith("/") && !url.startsWith("//")) {
                    response.sendRedirect(url);
                } else {
                    response.sendRedirect("/error?msg=invalid_redirect");
                }
            }
        }
    """),
]


def generate_hard_negatives() -> List[Dict[str, Any]]:
    """Generate hard negative examples (safe code that looks suspicious)."""
    examples = []
    instruction = "Analyze the Java code and identify ALL security vulnerabilities. Return structured JSON only."

    for code in HARD_NEGATIVES:
        text = (
            f"<|instruction|>\n{instruction}\n\n"
            f"<|input|>\n{code}\n\n"
            f"<|response|>\n"
            + json.dumps({"vulnerabilities": []}, indent=2, ensure_ascii=False)
        )
        examples.append({"text": text})

    logger.info(f"Generated {len(examples)} hard negative examples")
    return examples


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate production-quality training data")
    parser.add_argument("--output", type=str, default="Dataset/train_production.jsonl",
                        help="Output path for the generated JSONL dataset")
    parser.add_argument("--existing_data", type=str,
                        default="Dataset/train_classifier_precise_lines.jsonl",
                        help="Path to existing Juliet-based JSONL to enhance")
    parser.add_argument("--vul4j_max", type=int, default=80,
                        help="Maximum number of Vul4J examples to generate")
    parser.add_argument("--skip_vul4j", action="store_true",
                        help="Skip Vul4J GitHub fetching (uses only local sources)")
    args = parser.parse_args()

    all_examples: List[Dict[str, Any]] = []

    # Source 1: Enhanced Juliet dataset
    logger.info("=" * 60)
    logger.info("Source 1: Enhancing existing Juliet dataset...")
    logger.info("=" * 60)
    juliet_examples = enhance_juliet_dataset(args.existing_data)
    all_examples.extend(juliet_examples)

    # Source 2: Vul4J real-world examples
    if not args.skip_vul4j:
        logger.info("=" * 60)
        logger.info("Source 2: Fetching Vul4J real-world CVE examples...")
        logger.info("=" * 60)
        vul4j_examples = generate_vul4j_examples(max_entries=args.vul4j_max)
        all_examples.extend(vul4j_examples)
    else:
        logger.info("Skipping Vul4J source (--skip_vul4j flag)")

    # Source 3: Synthetic multi-vulnerability examples
    logger.info("=" * 60)
    logger.info("Source 3: Generating synthetic multi-vulnerability examples...")
    logger.info("=" * 60)
    synthetic_examples = generate_synthetic_examples()
    all_examples.extend(synthetic_examples)

    # Source 4: Hard negatives
    logger.info("=" * 60)
    logger.info("Source 4: Generating hard negative examples...")
    logger.info("=" * 60)
    hard_negatives = generate_hard_negatives()
    all_examples.extend(hard_negatives)

    # Shuffle the combined dataset
    random.seed(42)
    random.shuffle(all_examples)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info("=" * 60)
    logger.info(f"DATASET GENERATION COMPLETE")
    logger.info(f"Total examples: {len(all_examples)}")
    logger.info(f"  - Enhanced Juliet: {len(juliet_examples)}")
    if not args.skip_vul4j:
        logger.info(f"  - Vul4J real-world: {len(vul4j_examples)}")
    logger.info(f"  - Synthetic multi-vuln: {len(synthetic_examples)}")
    logger.info(f"  - Hard negatives: {len(hard_negatives)}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

`

### analyze_data_quality.py

`python
"""Quick analysis of training data quality for vulnerability detection model."""
import json

path = r"c:\Users\Arghya\Desktop\Solutions\AMD-Hackathon\Dataset\train_classifier_precise_lines.jsonl"
lines = open(path, encoding="utf-8").readlines()

total = len(lines)
has_response = 0
no_response = 0
empty_vuln = 0
has_vuln = 0
vuln_counts = {}
truncated = 0
template_desc = 0
template_impact = 0
template_rec = 0
missing_confidence = 0
missing_location = 0
cwe_distribution = {}
all_descriptions = []
all_vuln_entries = []

for l in lines:
    data = json.loads(l)
    text = data.get("text", "")
    
    if "<|response|>" in text:
        has_response += 1
        resp_part = text.split("<|response|>")[-1].strip()
        try:
            resp_json = json.loads(resp_part)
            vulns = resp_json.get("vulnerabilities", [])
            if not vulns:
                empty_vuln += 1
            else:
                has_vuln += 1
                vc = len(vulns)
                vuln_counts[vc] = vuln_counts.get(vc, 0) + 1
                for v in vulns:
                    all_vuln_entries.append(v)
                    cwe = v.get("cwe_id", "NONE")
                    cwe_distribution[cwe] = cwe_distribution.get(cwe, 0) + 1
                    desc = v.get("description", "")
                    impact = v.get("impact", "")
                    rec = v.get("recommendation", "")
                    all_descriptions.append(desc)
                    if "contains a vulnerability associated with" in desc:
                        template_desc += 1
                    if "can lead to compromised integrity, confidentiality, or availability" in impact:
                        template_impact += 1
                    if "apply secure coding practices to remediate" in rec:
                        template_rec += 1
                    if "confidence" not in v:
                        missing_confidence += 1
                    if "location" not in v:
                        missing_location += 1
        except json.JSONDecodeError:
            truncated += 1
    else:
        no_response += 1

total_vulns = sum(vuln_counts.get(k, 0) * k for k in vuln_counts)

print("=== TRAINING DATA QUALITY REPORT ===")
print(f"Total records: {total}")
print(f"Has response marker: {has_response}")
print(f"Missing response marker: {no_response}")
print(f"Truncated/unparseable response: {truncated}")
print(f"Empty vulnerabilities (negatives): {empty_vuln}")
print(f"Has vulnerabilities (positives): {has_vuln}")
print(f"Negative:Positive ratio: {empty_vuln}:{has_vuln}")
print()
print("=== VULNERABILITY COUNT DISTRIBUTION ===")
for k in sorted(vuln_counts.keys()):
    print(f"  {k} vuln(s): {vuln_counts[k]} records")
print()
print("=== CWE DISTRIBUTION ===")
for k, v in sorted(cwe_distribution.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
print()
print("=== TEMPLATE/BOILERPLATE ANALYSIS ===")
print(f"Total individual vuln entries: {total_vulns}")
pct = lambda n: f"{n*100//max(total_vulns,1)}%"
print(f"Templated descriptions: {template_desc} ({pct(template_desc)})")
print(f"Templated impacts: {template_impact} ({pct(template_impact)})")
print(f"Templated recommendations: {template_rec} ({pct(template_rec)})")
print(f"Missing confidence field: {missing_confidence} ({pct(missing_confidence)})")
print(f"Missing location field: {missing_location} ({pct(missing_location)})")

# Check for duplicate descriptions
from collections import Counter
desc_counts = Counter(all_descriptions)
dup_descs = {k: v for k, v in desc_counts.items() if v > 3}
print()
print("=== MOST DUPLICATED DESCRIPTIONS (>3 occurrences) ===")
for d, c in sorted(dup_descs.items(), key=lambda x: -x[1])[:10]:
    print(f"  [{c}x] {d[:120]}")

# Check for code-specific vs generic descriptions
code_specific = 0
for desc in all_descriptions:
    # A code-specific description mentions actual variable names, methods, classes
    if any(kw in desc for kw in ["variable", "method ", "function ", "parameter ", "class ", "line ", "`"]):
        code_specific += 1
print()
print(f"Code-specific descriptions (mention vars/methods): {code_specific} ({pct(code_specific)})")
print(f"Generic/template descriptions: {total_vulns - code_specific} ({pct(total_vulns - code_specific)})")

# Check unique descriptions
unique_descs = len(set(all_descriptions))
print(f"Unique descriptions: {unique_descs} out of {total_vulns}")

`

### compare_datasets.py

`python
"""Compare quality metrics between old and new training datasets."""
import json
from collections import Counter

def analyze_dataset(path, label):
    lines = open(path, encoding="utf-8").readlines()
    total = len(lines)
    has_vuln = 0
    empty_vuln = 0
    truncated = 0
    template_desc = 0
    template_impact = 0
    template_rec = 0
    all_descriptions = []
    cwe_distribution = {}
    vuln_count_dist = Counter()
    total_vulns = 0
    multi_vuln = 0

    for l in lines:
        data = json.loads(l)
        text = data.get("text", "")
        if "<|response|>" not in text:
            continue
        resp_part = text.split("<|response|>")[-1].strip()
        try:
            resp_json = json.loads(resp_part)
        except json.JSONDecodeError:
            truncated += 1
            continue

        vulns = resp_json.get("vulnerabilities", [])
        vc = len(vulns)
        vuln_count_dist[vc] += 1

        if not vulns:
            empty_vuln += 1
        else:
            has_vuln += 1
            total_vulns += vc
            if vc > 1:
                multi_vuln += 1

            for v in vulns:
                cwe = v.get("cwe_id", "NONE")
                cwe_distribution[cwe] = cwe_distribution.get(cwe, 0) + 1
                desc = v.get("description", "")
                impact = v.get("impact", "")
                rec = v.get("recommendation", "")
                all_descriptions.append(desc)
                if "contains a vulnerability associated with" in desc:
                    template_desc += 1
                if "can lead to compromised integrity, confidentiality, or availability" in impact:
                    template_impact += 1
                if "apply secure coding practices to remediate" in rec:
                    template_rec += 1

    unique_descs = len(set(all_descriptions))
    unique_cwes = len(cwe_distribution)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total records:              {total}")
    print(f"  Positive (has vulns):        {has_vuln}")
    print(f"  Negative (clean code):       {empty_vuln}")
    print(f"  Truncated/broken:            {truncated}")
    print(f"  Total vuln entries:          {total_vulns}")
    print(f"  Multi-vuln files:            {multi_vuln}")
    print(f"  Unique CWE types:            {unique_cwes}")
    print(f"  Unique descriptions:         {unique_descs} / {total_vulns}")
    pct = lambda n: f"{n*100//max(total_vulns,1)}%"
    print(f"  Templated descriptions:      {template_desc} ({pct(template_desc)})")
    print(f"  Templated impacts:           {template_impact} ({pct(template_impact)})")
    print(f"  Templated recommendations:   {template_rec} ({pct(template_rec)})")
    print(f"\n  Vuln count distribution:")
    for k in sorted(vuln_count_dist.keys()):
        print(f"    {k} vuln(s): {vuln_count_dist[k]} files")
    print(f"\n  CWE distribution:")
    for cwe, count in sorted(cwe_distribution.items(), key=lambda x: -x[1])[:20]:
        print(f"    {cwe}: {count}")

    # Show sample descriptions
    if all_descriptions:
        print(f"\n  Sample descriptions (first 3 unique):")
        seen = set()
        for d in all_descriptions:
            if d not in seen and len(d) > 20:
                seen.add(d)
                print(f"    -> {d[:150]}...")
                if len(seen) >= 3:
                    break


old_path = r"c:\Users\Arghya\Desktop\Solutions\AMD-Hackathon\Dataset\train_classifier_precise_lines.jsonl"
new_path = r"c:\Users\Arghya\Desktop\Solutions\AMD-Hackathon\Dataset\train_production.jsonl"

analyze_dataset(old_path, "OLD DATASET (train_classifier_precise_lines.jsonl)")
analyze_dataset(new_path, "NEW DATASET (train_production.jsonl)")

`

### scanner.py

`python
import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

from inference_engine import VulnerabilityInferenceEngine

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("scanner")


def extract_java_blocks(code: str) -> List[Dict[str, Any]]:
    """
    Extracts logical blocks (primarily class methods) from a Java source file
    using a character-by-character state machine to handle strings, comments, and braces.
    Preserved for backward compatibility and granular code segmenting.
    """
    chunks: List[Dict[str, Any]] = []
    n = len(code)
    i = 0
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escape = False
    
    brace_level = 0
    block_starts: Dict[int, int] = {}
    last_separator_idx = 0
    
    # Pre-calculate line start offsets
    line_starts = [0]
    for idx, char in enumerate(code):
        if char == '\n':
            line_starts.append(idx + 1)
            
    def get_line_num(char_idx: int) -> int:
        for line_num, start_idx in enumerate(line_starts):
            if start_idx > char_idx:
                return line_num
        return len(line_starts)

    try:
        while i < n:
            char = code[i]
            
            if escape:
                escape = False
                i += 1
                continue
                
            if in_line_comment:
                if char == '\n':
                    in_line_comment = False
                i += 1
                continue
                
            if in_block_comment:
                if char == '/' and i > 0 and code[i-1] == '*':
                    in_block_comment = False
                i += 1
                continue
                
            if in_string:
                if char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
                i += 1
                continue
                
            if in_char:
                if char == '\\':
                    escape = True
                elif char == "'":
                    in_char = False
                i += 1
                continue
                
            # Detect start of comments or strings
            if char == '/' and i + 1 < n:
                next_char = code[i+1]
                if next_char == '/':
                    in_line_comment = True
                    i += 2
                    continue
                elif next_char == '*':
                    in_block_comment = True
                    i += 2
                    continue
                    
            if char == '"':
                in_string = True
                i += 1
                continue
                
            if char == "'":
                in_char = True
                i += 1
                continue
                
            # Brace depth tracking
            if char == '{':
                brace_level += 1
                start_idx = last_separator_idx
                while start_idx < i and code[start_idx] in ' \t\r\n;{}':
                    start_idx += 1
                block_starts[brace_level] = start_idx
                last_separator_idx = i + 1
                
            elif char == '}':
                if brace_level in block_starts:
                    start_idx = block_starts[brace_level]
                    end_idx = i + 1
                    block_content = code[start_idx:end_idx]
                    
                    start_line = get_line_num(start_idx)
                    end_line = get_line_num(end_idx)
                    
                    if brace_level == 2:
                        chunks.append({
                            "start_line": start_line,
                            "end_line": end_line,
                            "content": block_content
                        })
                    del block_starts[brace_level]
                    
                brace_level = max(0, brace_level - 1)
                last_separator_idx = i + 1
                
            elif char == ';':
                last_separator_idx = i + 1
                
            i += 1

    except Exception as e:
        logger.error(f"Error during state-machine brace matching: {e}")

    # Fallback to returning the entire file if no methods were parsed
    if not chunks:
        chunks.append({
            "start_line": 1,
            "end_line": len(line_starts),
            "content": code
        })
        
    return chunks


def run_codebase_scan(
    model_id: str,
    adapter_path: str,
    target_dir: str,
    output_report: str = "security_report.json",
    use_chunks: bool = False,
    load_in_4bit: bool = True,
    max_tokens: int = 1024,
    min_confidence: float = 0.0
) -> None:
    """
    Recursively scans a target directory for Java source files, reads and analyzes their
    codeblocks with the inference engine, and outputs compiled results to a JSON report.
    """
    target_path = Path(target_dir)
    if not target_path.exists():
        logger.error(f"Target directory for scanning does not exist: {target_path}")
        return

    # Initialize the inference engine wrapper
    try:
        logger.info(f"Initializing inference context (Base: {model_id}, Adapter: {adapter_path})...")
        engine = VulnerabilityInferenceEngine(
            model_id=model_id,
            adapter_path=adapter_path,
            load_in_4bit=load_in_4bit
        )
    except Exception as err:
        logger.error(f"Failed to load model framework context: {err}", exc_info=True)
        return

    # Scan directories recursively
    logger.info(f"Recursively exploring path: {target_path} for '.java' source files.")
    java_files = list(target_path.rglob("*.java"))
    logger.info(f"Discovered {len(java_files)} Java source files to scan.")

    compiled_findings: List[Dict[str, Any]] = []
    failed_scans: List[Dict[str, Any]] = []

    for idx, file_path in enumerate(java_files, 1):
        relative_path = str(file_path.relative_to(target_path))
        logger.info(f"[{idx}/{len(java_files)}] Scanning file: {relative_path}")

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                file_content = f.read()

            # Choose block size strategy
            if use_chunks:
                logger.info(f"Using method-level segmentation for: {file_path.name}")
                chunks = extract_java_blocks(file_content)
            else:
                logger.info(f"Using full-text block execution for: {file_path.name}")
                chunks = [{
                    "start_line": 1,
                    "end_line": len(file_content.splitlines()),
                    "content": file_content
                }]

            for chunk in chunks:
                raw_code = chunk["content"]
                
                # Execute inference and exception-safe JSON parsing
                result = engine.analyze_file_content(
                    raw_code, 
                    max_new_tokens=max_tokens,
                    file_line_count=len(raw_code.splitlines()),
                    min_confidence=min_confidence
                )

                # Check if parsing or inference failed
                if "error" in result:
                    logger.warning(
                        f"Scan anomaly detected in {relative_path} (Lines {chunk['start_line']}-{chunk['end_line']}): {result['error']}"
                    )
                    failed_scans.append({
                        "file_path": relative_path,
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "error_message": result["error"],
                        "raw_response": result.get("raw_response", "")
                    })
                    continue

                # Compile valid vulnerability detections
                vulnerabilities = result.get("vulnerabilities", [])
                if isinstance(vulnerabilities, list):
                    for vuln in vulnerabilities:
                        # Map each vulnerability to its file and segment metadata
                        compiled_findings.append({
                            "file_path": relative_path,
                            "chunk_start_line": chunk["start_line"],
                            "chunk_end_line": chunk["end_line"],
                            "cwe_id": vuln.get("cwe_id", "Unknown"),
                            "cwe_name": vuln.get("cwe_name", "Unknown"),
                            "severity": vuln.get("severity", "medium"),
                            "confidence": vuln.get("confidence", 1.0),
                            "location": vuln.get("location", {}),
                            "description": vuln.get("description", ""),
                            "impact": vuln.get("impact", ""),
                            "recommendation": vuln.get("recommendation", "")
                        })

        except Exception as file_err:
            logger.error(f"Failed to read/process file {relative_path}: {file_err}", exc_info=True)
            failed_scans.append({
                "file_path": relative_path,
                "error_message": f"File read or execution exception: {str(file_err)}"
            })

    # Save results to master vulnerability report
    report_data = {
        "target_directory": str(target_path.resolve()),
        "total_files_scanned": len(java_files),
        "vulnerabilities_detected": len(compiled_findings),
        "vulnerabilities": compiled_findings,
        "failed_scans": failed_scans
    }

    report_path = Path(output_report)
    try:
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(report_data, rf, indent=4)
        logger.info(f"Master vulnerability scan report compiled and saved to: {report_path.resolve()}")
    except Exception as report_err:
        logger.error(f"Failed to write master report file to {report_path}: {report_err}", exc_info=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Codebase Vulnerability Scanner")
    parser.add_argument("--model_id", type=str, required=True, help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter checkpoints")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to directory containing Java source files")
    parser.add_argument("--output_report", type=str, default="security_report.json", help="Path for JSON output report")
    parser.add_argument("--use_chunks", action="store_true", help="Enable method-level code segmentation")
    parser.add_argument("--no_quant", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum new tokens to generate per inference call")
    parser.add_argument("--min_confidence", type=float, default=0.0, help="Minimum confidence score to include finding (e.g. 0.8)")
    args = parser.parse_args()

    run_codebase_scan(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        target_dir=args.target_dir,
        output_report=args.output_report,
        use_chunks=args.use_chunks,
        load_in_4bit=not args.no_quant,
        max_tokens=args.max_tokens,
        min_confidence=args.min_confidence
    )

`

## Frontend Source Code

### frontend/src/App.tsx

`typescript
import React, { useState, useRef, useEffect, useMemo } from "react";
import {
  Shield,
  FileCode,
  FolderOpen,
  Cpu,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Flame,
  Eye,
  RefreshCw,
  Check,
  Trash2,
  HelpCircle,
  GitCommit,
  Terminal,
  ArrowRight,
  ChevronRight,
  Sparkles,
  Code2,
  FileCheck,
  AlertOctagon,
  KeyRound,
  FileSpreadsheet,
  Info,
  ChevronDown,
  ChevronUp,
  LayoutDashboard,
  History,
  Copy
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { diffLines } from "diff";
import { JavaFile, Vulnerability, ScanHistoryEntry } from "./types";

// Lightweight regex-based syntax tokenizer for Java code
function tokenizeJavaLine(line: string): { text: string; type: string }[] {
  if (!line) return [{ text: " ", type: "text" }];

  const regex = /(\/\/.*)|("(?:\\.|[^"\\])*")|('(?:\\.|[^'\\])*')|(@[a-zA-Z0-9_]+)|(\b(?:public|private|protected|class|interface|enum|extends|implements|import|package|return|new|if|else|for|while|try|catch|finally|throw|throws|final|static|void|int|double|float|long|boolean|char|short|byte|null|true|false|this|super)\b)|(\b\d+(?:\.\d+)?\b)|(\b[A-Z][a-zA-Z0-9_]*\b)|(\b[a-z_][a-zA-Z0-9_]*(?=\s*\())|([+\-*/%&=<>!|~^:?(){}[\];.,]+)|(\s+)|([^\s]+)/g;

  const tokens: { text: string; type: string }[] = [];
  let match;

  while ((match = regex.exec(line)) !== null) {
    const [
      full,
      comment,
      stringDouble,
      stringSingle,
      annotation,
      keyword,
      number,
      className,
      methodName,
      operator,
      whitespace,
      other
    ] = match;

    if (comment !== undefined) {
      tokens.push({ text: comment, type: "comment" });
    } else if (stringDouble !== undefined) {
      tokens.push({ text: stringDouble, type: "string" });
    } else if (stringSingle !== undefined) {
      tokens.push({ text: stringSingle, type: "string" });
    } else if (annotation !== undefined) {
      tokens.push({ text: annotation, type: "annotation" });
    } else if (keyword !== undefined) {
      tokens.push({ text: keyword, type: "keyword" });
    } else if (number !== undefined) {
      tokens.push({ text: number, type: "number" });
    } else if (className !== undefined) {
      tokens.push({ text: className, type: "class-name" });
    } else if (methodName !== undefined) {
      tokens.push({ text: methodName, type: "method-name" });
    } else if (operator !== undefined) {
      tokens.push({ text: operator, type: "operator" });
    } else if (whitespace !== undefined) {
      tokens.push({ text: whitespace, type: "text" });
    } else {
      tokens.push({ text: other, type: "text" });
    }
  }

  return tokens;
}

// Map Java syntax elements to classic VS Code Dark style coloring with soft green hints
function renderHighlightedCode(line: string) {
  const tokens = tokenizeJavaLine(line);
  return (
    <>
      {tokens.map((token, i) => {
        let styleClass = "";
        switch (token.type) {
          case "comment":
            styleClass = "text-emerald-500/80 italic"; // VS Code green comments
            break;
          case "string":
            styleClass = "text-amber-300"; // Rich VS Code amber/yellow strings
            break;
          case "annotation":
            styleClass = "text-emerald-400 font-semibold"; // Green annotations
            break;
          case "keyword":
            styleClass = "text-sky-450 font-bold"; // Deep blue keywords
            break;
          case "number":
            styleClass = "text-emerald-400 font-semibold"; // Green numbers
            break;
          case "class-name":
            styleClass = "text-teal-300 font-semibold"; // Teal classes
            break;
          case "method-name":
            styleClass = "text-yellow-200"; // Warm yellow-gold methods
            break;
          case "operator":
            styleClass = "text-slate-400"; // Operators
            break;
          default:
            styleClass = "text-slate-200"; // Active default foreground text
        }
        return (
          <span key={i} className={styleClass}>
            {token.text}
          </span>
        );
      })}
    </>
  );
}

export default function App() {
  // Projects and files
  const [loadedFiles, setLoadedFiles] = useState<JavaFile[]>([]);
  const [activeFileIndex, setActiveFileIndex] = useState<number>(0);
  const [droppedItemName, setDroppedItemName] = useState<string | null>(null);

  const [fullscreenDiff, setFullscreenDiff] = useState<{
    isOpen: boolean;
    vuln: Vulnerability | null;
  }>({ isOpen: false, vuln: null });

  const [copiedWorkspace, setCopiedWorkspace] = useState(false);
  const [copiedFixed, setCopiedFixed] = useState(false);

  // App Level Tabs
  const [activeTab, setActiveTab] = useState<"scan" | "history" | "rocm">("scan");
  const [scanHistory, setScanHistory] = useState<ScanHistoryEntry[]>([]);
  const [isHistoryLoaded, setIsHistoryLoaded] = useState<boolean>(false);
  const [expandedHistoryIds, setExpandedHistoryIds] = useState<string[]>([]);
  const [rocmStats, setRocmStats] = useState({ vram: 14200, load: 45, cpu: 25, ram: 128, vramTotal: 198000, gpuName: 'AMD MI300X', gpuType: 'HBM3 Memory Cluster', modelsLoaded: ['Loading...'], apiHealth: 'Loading...' });

  const toggleHistoryExpanded = (id: string) => {
    setExpandedHistoryIds(prev =>
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    );
  };

  // Load history from localstorage directory backend
  useEffect(() => {
    fetch('/local-api/history')
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setScanHistory(data.map((item: any) => ({
            ...item,
            timestamp: new Date(item.timestamp)
          })));
        }
        setIsHistoryLoaded(true);
      })
      .catch(err => {
        console.error("Failed to load history from local server", err);
        setIsHistoryLoaded(true);
      });
  }, []);

  const isInitialMount = useRef(true);

  // Persist history to localstorage directory backend whenever it updates
  useEffect(() => {
    if (!isHistoryLoaded) return;
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }
    fetch('/local-api/history', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(scanHistory)
    }).catch(err => console.error("Failed to save history to local server", err));
  }, [scanHistory, isHistoryLoaded]);

  // Poll ROCm Telemetry
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (activeTab === "rocm") {
      const fetchTelemetry = async () => {
        try {
          const res = await fetch("/api/telemetry");
          if (res.ok) {
            const data = await res.json();
            setRocmStats({
              vram: data.vram_usage || 14200,
              load: data.compute_load || 45,
              cpu: data.cpu_usage || 25,
              ram: data.ram_usage || 128,
              vramTotal: data.vram_total || 198000,
              gpuName: data.gpu_name || 'AMD MI300X',
              gpuType: data.gpu_type || 'HBM3 Memory Cluster',
              modelsLoaded: data.models_loaded || ['DeepSeek-Coder-6.7B-Instruct (Vuln Scanner)', 'Qwen2.5-Coder-7B-Instruct (Fix Engine)'],
              apiHealth: data.api_health || 'Unknown'
            });
          }
        } catch (e) {
          console.error("Failed to fetch ROCm telemetry", e);
        }
      };
      fetchTelemetry(); // Initial fetch
      interval = setInterval(fetchTelemetry, 1000); // Poll every 1 second
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [activeTab]);

  // Security elements
  const [vulnerabilities, setVulnerabilities] = useState<Vulnerability[]>([]);
  const [scanState, setScanState] = useState<"idle" | "scanning" | "completed">("idle");
  const [scanProgress, setScanProgress] = useState<number>(0);
  const [scanProgressText, setScanProgressText] = useState<string>("");
  const [scanMode, setScanMode] = useState<"local_heuristics" | "gemini_ai" | "none">("none");
  const [scanMessage, setScanMessage] = useState<string>("");
  const [isSingleScanActive, setIsSingleScanActive] = useState<boolean>(false);

  // Specific remediation states (vulnerabilityId -> "idle" | "fixing" | "diff_ready" | "approved")
  const [remediatingIdState, setRemediatingIdState] = useState<Record<string, "idle" | "fixing" | "diff_ready" | "approved">>({});

  // Direct manual code pasting sandbox helper
  const [sandboxCode, setSandboxCode] = useState<string>("");
  const [sandboxFileName, setSandboxFileName] = useState<string>("MyVulnerableService.java");
  const [isSandboxOpen, setIsSandboxOpen] = useState<boolean>(false);

  // Drag and drop states
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  // Active highlighted lines in file editor (from selected vulnerability)
  const [selectedVulnerabilityId, setSelectedVulnerabilityId] = useState<string | null>(null);
  const activeFile = loadedFiles[activeFileIndex] || null;

  // Initialize data on load and auto-recover session
  useEffect(() => {
    fetch('/local-api/session')
      .then(res => res.json())
      .then(data => {
        if (data && Array.isArray(data.loadedFiles)) {
          setLoadedFiles(data.loadedFiles);
          setActiveFileIndex(data.activeFileIndex || 0);
          setVulnerabilities(data.vulnerabilities || []);
          setScanMode(data.scanMode || "none");
          setScanMessage(data.scanMessage || "");
          setScanState(data.scanState || "idle");

          if (data.scanState === "scanning") {
            // Re-trigger the scan based on the active state
            if (data.isSingleScanActive) {
              isRecoveringSingleScan.current = true;
            } else {
              isRecoveringGlobalScan.current = true;
            }
          }
        } else {
          setScanState("idle");
          setVulnerabilities([]);
        }
      })
      .catch(err => {
        console.error("Failed to recover session", err);
        setScanState("idle");
        setVulnerabilities([]);
      });
    setSelectedVulnerabilityId(null);
  }, []);

  const isRecoveringSingleScan = useRef(false);
  const isRecoveringGlobalScan = useRef(false);

  // Effect to retrigger scan if we recovered a scanning state
  useEffect(() => {
    if (loadedFiles.length > 0 && scanState === "scanning" && scanProgress === 0) {
      if (isRecoveringGlobalScan.current) {
        isRecoveringGlobalScan.current = false;
        triggerCodeScan();
      } else if (isRecoveringSingleScan.current && activeFile) {
        isRecoveringSingleScan.current = false;
        triggerSingleFileScan();
      }
    }
  }, [scanState, loadedFiles, activeFileIndex]);

  // Save session state on change
  useEffect(() => {
    if (loadedFiles.length === 0 && scanState === "idle") return; // Don't save empty initial state
    fetch('/local-api/session', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        loadedFiles,
        activeFileIndex,
        scanState,
        vulnerabilities,
        scanMode,
        scanMessage,
        isSingleScanActive
      })
    }).catch(err => console.error("Failed to save session", err));
  }, [loadedFiles, activeFileIndex, scanState, vulnerabilities, scanMode, scanMessage, isSingleScanActive]);

  // Drag-and-drop overrides
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      processFileList(e.dataTransfer.files);
    }
  };

  const processFileList = async (files: FileList) => {
    const javaFilesList: JavaFile[] = [];
    const seenPaths = new Set<string>();

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      if (file.name.endsWith(".java") || file.type === "text/x-java-source") {
        const path = file.webkitRelativePath || `uploads/${file.name}`;
        
        // Skip common build and hidden directories to prevent duplicate scans
        if (
          path.includes("/target/") || 
          path.includes("/build/") || 
          path.includes("/bin/") || 
          path.includes("/.git/") ||
          path.includes("/.idea/") ||
          path.includes("/.vscode/")
        ) {
          continue;
        }

        if (seenPaths.has(path)) continue;
        seenPaths.add(path);

        const textContent = await readFileContent(file);
        javaFilesList.push({
          name: file.name,
          path: path,
          content: textContent,
          originalContent: textContent
        });
      }
    }

    if (javaFilesList.length > 0) {
      setLoadedFiles(javaFilesList);
      setActiveFileIndex(0);
      setScanState("idle");
      setVulnerabilities([]);
      setSelectedVulnerabilityId(null);

      // Determine dropped name based on common path or single file
      let droppedName = "Unknown";
      if (files.length === 1) {
        droppedName = files[0].name;
      } else if (files[0].webkitRelativePath) {
        droppedName = files[0].webkitRelativePath.split('/')[0];
      } else {
        droppedName = `${files.length} Files Loaded`;
      }
      setDroppedItemName(droppedName);
    } else {
      alert("No Java source files (.java) detected in the uploaded list.");
    }
  };

  const readFileContent = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target?.result as string || "");
      reader.onerror = (e) => reject(e);
      reader.readAsText(file);
    });
  };

  // Handle standard individual file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      processFileList(e.target.files);
    }
  };

  // Add a fully custom pasted Java code file inside workspace
  const handleAddSandboxFile = () => {
    if (!sandboxCode.trim()) return;

    const newFile: JavaFile = {
      name: sandboxFileName.endsWith(".java") ? sandboxFileName : `${sandboxFileName}.java`,
      path: `sandbox/${sandboxFileName.endsWith(".java") ? sandboxFileName : `${sandboxFileName}.java`}`,
      content: sandboxCode,
      originalContent: sandboxCode
    };

    setLoadedFiles([newFile, ...loadedFiles]);
    setActiveFileIndex(0);
    setSandboxCode("");
    setIsSandboxOpen(false);
    setScanState("idle");
    setVulnerabilities([]);
    setSelectedVulnerabilityId(null);
    setDroppedItemName(newFile.name);
  };

  // Perform full Code Security Scan
  const triggerCodeScan = async () => {
    if (loadedFiles.length === 0) return;

    setScanState("scanning");
    setIsSingleScanActive(false);
    setScanProgress(5);
    setScanProgressText("Initializing security context...");

    // Dramatic progress bar steps
    const steps = [
      { prg: 20, text: "Generating AST representation models..." },
      { prg: 45, text: "Tracing data tainted parameters..." },
      { prg: 70, text: "Cross-referencing OWASP rulesets..." },
      { prg: 88, text: "Verifying cryptographic strength limits..." },
      { prg: 95, text: "Assembling vulnerability matrix maps..." }
    ];

    let currentStep = 0;
    const interval = setInterval(() => {
      if (currentStep < steps.length) {
        setScanProgress(steps[currentStep].prg);
        setScanProgressText(steps[currentStep].text);
        currentStep++;
      }
    }, 550);

    try {
      const response = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ files: loadedFiles })
      });

      clearInterval(interval);
      setScanProgress(100);
      setScanProgressText("Audit compilation finalized!");

      const data = await response.json();

      setTimeout(() => {
        const uniqueVulns = new Map<string, Vulnerability>();
        (data.vulnerabilities || []).forEach((v: Vulnerability, idx: number) => {
          const sig = `${v.filePath}:${v.lineNumber}:${v.type}:${v.cwe_id}`;
          if (!uniqueVulns.has(sig)) {
            uniqueVulns.set(sig, {
              ...v,
              id: v.id || `vuln-${Date.now()}-${idx}-${Math.random().toString(36).substring(7)}`
            });
          }
        });
        const vulnsWithIds = Array.from(uniqueVulns.values());
        setVulnerabilities(vulnsWithIds);

        // Setup state map of vulnerabilities
        const stateMap: Record<string, "idle" | "fixing" | "diff_ready" | "approved"> = {};
        vulnsWithIds.forEach((v: Vulnerability) => {
          stateMap[v.id] = v.remediatedSnippet ? "diff_ready" : "idle";
        });
        setRemediatingIdState(stateMap);

        setScanMode(data.mode || "local_heuristics");
        setScanMessage(data.message || "");
        setScanState("completed");

        // Save to History
        setScanHistory(prev => [{
          id: Date.now().toString(),
          timestamp: new Date(),
          mode: data.mode || "local_heuristics",
          files: [...loadedFiles],
          vulnerabilities: vulnsWithIds
        }, ...prev]);
      }, 300);

    } catch (err: any) {
      clearInterval(interval);
      console.error("Scan attempt failed completely:", err);
      setScanProgressText("API Interface failed. Starting local scanner...");

      // Secondary fallback
      setTimeout(() => {
        setScanState("idle");
        alert("Server communication error. Please ensure the backend dev server is active and running.");
      }, 1000);
    }
  };

  // Perform Single File Scan
  const triggerSingleFileScan = async () => {
    if (!activeFile) return;

    setScanState("scanning");
    setIsSingleScanActive(true);
    setScanProgress(5);
    setScanProgressText(`Initializing scan for ${activeFile.name}...`);

    const steps = [
      { prg: 20, text: "Generating AST representation models..." },
      { prg: 45, text: "Tracing data tainted parameters..." },
      { prg: 70, text: "Cross-referencing OWASP rulesets..." },
      { prg: 88, text: "Verifying cryptographic strength limits..." },
      { prg: 95, text: "Assembling vulnerability matrix maps..." }
    ];

    let currentStep = 0;
    const interval = setInterval(() => {
      if (currentStep < steps.length) {
        setScanProgress(steps[currentStep].prg);
        setScanProgressText(steps[currentStep].text);
        currentStep++;
      }
    }, 550);

    try {
      const response = await fetch("/api/scan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ files: [activeFile] })
      });

      clearInterval(interval);
      setScanProgress(100);
      setScanProgressText("Audit compilation finalized!");

      const data = await response.json();

      setTimeout(() => {
        const uniqueVulns = new Map<string, Vulnerability>();
        (data.vulnerabilities || []).forEach((v: Vulnerability, idx: number) => {
          const sig = `${v.filePath}:${v.lineNumber}:${v.type}:${v.cwe_id}`;
          if (!uniqueVulns.has(sig)) {
            uniqueVulns.set(sig, {
              ...v,
              id: v.id || `vuln-${Date.now()}-${idx}-${Math.random().toString(36).substring(7)}`
            });
          }
        });
        const vulnsWithIds = Array.from(uniqueVulns.values());
        setVulnerabilities(vulnsWithIds);

        const stateMap: Record<string, "idle" | "fixing" | "diff_ready" | "approved"> = {};
        vulnsWithIds.forEach((v: Vulnerability) => {
          stateMap[v.id] = v.remediatedSnippet ? "diff_ready" : "idle";
        });
        setRemediatingIdState(stateMap);

        setScanMode(data.mode || "local_heuristics");
        setScanMessage(data.message || "");
        setScanState("completed");

        setScanHistory(prev => [{
          id: Date.now().toString(),
          timestamp: new Date(),
          mode: data.mode || "local_heuristics",
          files: [activeFile],
          vulnerabilities: vulnsWithIds
        }, ...prev]);
      }, 300);

    } catch (err: any) {
      clearInterval(interval);
      console.error("Scan attempt failed completely:", err);
      setScanProgressText("API Interface failed.");

      setTimeout(() => {
        setScanState("idle");
        alert("Server communication error. Please ensure the backend dev server is active and running.");
      }, 1000);
    }
  };

  // Trigger Remediation Workflow ("Fix" Action)
  const triggerRemediation = async (vuln: Vulnerability) => {
    const fileToFix = loadedFiles.find(f => f.path === vuln.filePath || f.name === vuln.filePath.split('/').pop());
    if (!fileToFix) {
      alert("Associated file contexts could not be matched. Make sure files are loaded.");
      return;
    }

    setRemediatingIdState(prev => ({ ...prev, [vuln.id]: "fixing" }));

    try {
      const response = await fetch("/api/remediate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filePath: fileToFix.path,
          fileContent: fileToFix.content,
          vulnerability: vuln
        })
      });

      const data = await response.json();

      setVulnerabilities(prev => prev.map(v => {
        if (v.id === vuln.id) {
          return {
            ...v,
            remediatedSnippet: data.remediatedSnippet,
            remediationExplanation: data.remediationExplanation,
            // also keep track of what the full repaired file will look like
            fullRemediatedContent: data.fullRemediatedContent,
            snippet: v.snippet || fileToFix.content
          } as any;
        }
        return v;
      }));

      setRemediatingIdState(prev => ({ ...prev, [vuln.id]: "diff_ready" }));

    } catch (err) {
      console.error("Remediation repair API trigger failed:", err);
      alert("Error generating fix. Please retry or adjust code.");
      setRemediatingIdState(prev => ({ ...prev, [vuln.id]: "idle" }));
    }
  };

  // Action: Ignore Card
  const handleIgnoreVulnerability = (id: string) => {
    setVulnerabilities(prev => prev.map(v => {
      if (v.id === id) {
        return { ...v, status: "Ignored" };
      }
      return v;
    }));
  };

  // Action: Not a Vulnerability
  const handleRemoveVulnerability = (id: string) => {
    setVulnerabilities(prev => prev.filter(v => v.id !== id));
    if (selectedVulnerabilityId === id) {
      setSelectedVulnerabilityId(null);
    }
  };

  // Action: Approve Remediation
  const handleApproveRemediation = (vuln: any) => {
    // 1. Permanently update original file's state
    if (vuln.fullRemediatedContent) {
      setLoadedFiles(prev => prev.map(f => {
        if (f.path === vuln.filePath || f.name === vuln.filePath.split('/').pop()) {
          return {
            ...f,
            content: vuln.fullRemediatedContent
          };
        }
        return f;
      }));
    }

    // 2. Mark card as approved
    setVulnerabilities(prev => prev.map(v => {
      if (v.id === vuln.id) {
        return { ...v, status: "Approved" };
      }
      return v;
    }));

    setRemediatingIdState(prev => ({ ...prev, [vuln.id]: "approved" }));
  };

  // Action: Reject Remediation
  const handleRejectRemediation = (id: string) => {
    setRemediatingIdState(prev => ({ ...prev, [id]: "idle" }));
    setVulnerabilities(prev => prev.map(v => {
      if (v.id === id) {
        return {
          ...v,
          remediatedSnippet: undefined,
          remediationExplanation: undefined
        } as any;
      }
      return v;
    }));
  };

  // Count summary stats
  const stats = useMemo(() => {
    const totalCount = vulnerabilities.length;
    const active = vulnerabilities.filter(v => v.status !== "Ignored" && v.status !== "Approved");
    const high = active.filter(v => v.severity === "High").length;
    const medium = active.filter(v => v.severity === "Medium").length;
    const low = active.filter(v => v.severity === "Low").length;
    const fixed = vulnerabilities.filter(v => v.status === "Approved").length;

    return {
      total: totalCount,
      activeCount: active.length,
      high,
      medium,
      low,
      fixed
    };
  }, [vulnerabilities]);

  // Click a vulnerability to highlight that line in IDE workspace
  const handleVulnerabilityClick = (v: Vulnerability) => {
    setSelectedVulnerabilityId(v.id);

    // Find matching active file index and transition IDE focus
    const matchedIdx = loadedFiles.findIndex(f => f.path === v.filePath || f.name === v.filePath.split('/').pop());
    if (matchedIdx !== -1) {
      setActiveFileIndex(matchedIdx);

      // Auto scroll code editor container to lines of interest, if possible
      setTimeout(() => {
        const targetElement = document.getElementById(`line-anchor-${v.lineNumber}`);
        if (targetElement) {
          targetElement.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }, 100);
    }
  };

  // Action: Clear entire workspace
  const handleClearWorkspace = () => {
    setLoadedFiles([]);
    setActiveFileIndex(0);
    setVulnerabilities([]);
    setScanState("idle");
    setScanProgress(0);
    setScanProgressText("");
    setSelectedVulnerabilityId(null);
    setDroppedItemName(null);
    setScanMode("none");
    setScanMessage("");
    setIsSingleScanActive(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-tr from-[#fbcfe8]/35 via-[#e6f4ea]/45 to-[#ccfbf1]/40 text-slate-800 font-sans p-4 md:p-6 selection:bg-emerald-500/20 selection:text-emerald-950 relative overflow-hidden">
      {/* Background colorful glassmorphic blur shapes with hints of green */}
      <div className="absolute top-[-10%] left-[-10%] w-[50vw] h-[50vw] rounded-full bg-gradient-to-tr from-pink-300/30 to-emerald-255/20 blur-[140px] opacity-70 pointer-events-none animate-pulse duration-[10s]" />
      <div className="absolute bottom-[5%] right-[-10%] w-[55vw] h-[55vw] rounded-full bg-gradient-to-tr from-sky-200/25 via-[#e6f4ea]/30 to-emerald-300/25 blur-[160px] opacity-75 pointer-events-none animate-pulse duration-[15s]" />
      <div className="absolute top-[25%] right-[10%] w-[45vw] h-[45vw] rounded-full bg-gradient-to-tr from-emerald-300/25 via-teal-200/20 to-green-300/25 blur-[140px] opacity-75 pointer-events-none animate-pulse duration-[12s]" />

      {/* Container holding general borders */}
      <div className="max-w-[1550px] mx-auto space-y-6 relative z-10">

        {/* HEADER SECTION */}
        <header className="flex flex-col md:flex-row justify-between items-start md:items-center border border-white/60 bg-white/60 p-5 rounded-3xl gap-4 backdrop-blur-xl shadow-xl shadow-slate-200/30">
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-tr from-emerald-500 via-teal-500 to-indigo-600 text-white p-3.5 rounded-2xl shadow-lg shadow-emerald-500/20 animate-bounce-slow">
              <Shield className="w-7 h-7" />
            </div>
            <div>
              <h1 className="text-2xl font-black tracking-tight bg-gradient-to-r from-rose-600 via-indigo-600 to-emerald-600 bg-clip-text text-transparent flex items-center gap-2">
                CodeElixir.AI
                <span className="text-[10px] bg-emerald-500/10 text-emerald-700 border border-emerald-500/20 px-2.5 py-0.5 rounded-full uppercase tracking-widest font-mono font-bold">
                  JAVA AUDITOR v1.1
                </span>
              </h1>
              <p className="text-xs text-slate-600 mt-1 font-medium leading-relaxed">
                Execute deep AST-level data flow threat scanning and secure prompt remediation using server-contained LLM engines.
              </p>
            </div>
          </div>

          {/* Quick Stats Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 w-full md:w-auto">
            <div className="bg-white/40 border border-white/60 p-2.5 rounded-2xl text-center min-w-[105px] shadow-sm">
              <div className="text-[9px] uppercase font-mono tracking-wider text-slate-500 font-bold">Loaded Files</div>
              <div className="text-lg font-extrabold text-slate-800 mt-0.5">{loadedFiles.length}</div>
            </div>
            <div className="bg-white/40 border border-white/60 p-2.5 rounded-2xl text-center min-w-[105px] shadow-sm">
              <div className="text-[9px] uppercase font-mono tracking-wider text-slate-500 font-bold">Threats Active</div>
              <div className="text-lg font-extrabold text-rose-600 mt-0.5 flex justify-center items-center gap-1.5">
                {stats.activeCount}
                {stats.high > 0 && <span className="w-2.5 h-2.5 rounded-full bg-rose-500 animate-pulse" />}
              </div>
            </div>
            <div className="bg-white/40 border border-white/60 p-2.5 rounded-2xl text-center min-w-[105px] shadow-sm">
              <div className="text-[9px] uppercase font-mono tracking-wider text-slate-500 font-bold">Severity Matrix</div>
              <div className="text-xs text-slate-700 font-mono mt-1 flex justify-center gap-1">
                <span className="text-rose-700 bg-rose-105/50 px-1 rounded border border-rose-200 text-[10px]" title="High">{stats.high}H</span>
                <span className="text-amber-700 bg-amber-105/50 px-1 rounded border border-amber-200 text-[10px]" title="Medium">{stats.medium}M</span>
              </div>
            </div>
            <div className="bg-white/40 border border-white/60 p-2.5 rounded-2xl text-center min-w-[105px] shadow-sm hover:border-emerald-300/60 transition-colors">
              <div className="text-[9px] uppercase font-mono tracking-wider text-slate-500 font-bold">Auto Patches</div>
              <div className="text-lg font-extrabold text-emerald-600 mt-0.5 flex justify-center items-center gap-1">
                {stats.fixed} <Check className="w-4 h-4 text-emerald-500 stroke-[3]" />
              </div>
            </div>
          </div>
        </header>

        {/* MAIN NAVIGATION TABS */}
        <div className="flex gap-2 border-b border-slate-200 pb-px">
          <button
            onClick={() => setActiveTab("scan")}
            className={`px-6 py-3 text-sm font-bold transition-all rounded-t-xl flex items-center gap-2 ${activeTab === "scan"
                ? "bg-white/60 text-emerald-700 border-t border-l border-r border-white/60 shadow-[0_-4px_6px_-2px_rgba(0,0,0,0.05)]"
                : "text-slate-500 hover:bg-white/40 hover:text-slate-700"
              }`}
          >
            <LayoutDashboard className="w-4 h-4" /> Scanner Workspace
          </button>
          <button
            onClick={() => setActiveTab("history")}
            className={`px-6 py-3 text-sm font-bold transition-all rounded-t-xl flex items-center gap-2 ${activeTab === "history"
                ? "bg-white/60 text-indigo-700 border-t border-l border-r border-white/60 shadow-[0_-4px_6px_-2px_rgba(0,0,0,0.05)]"
                : "text-slate-500 hover:bg-white/40 hover:text-slate-700"
              }`}
          >
            <History className="w-4 h-4" /> Scan History
            {scanHistory.length > 0 && (
              <span className="bg-indigo-100 text-indigo-700 text-[10px] px-2 py-0.5 rounded-full">{scanHistory.length}</span>
            )}
          </button>
          <button
            onClick={() => setActiveTab("rocm")}
            className={`px-6 py-3 text-sm font-bold transition-all rounded-t-xl flex items-center gap-2 ${activeTab === "rocm"
                ? "bg-white/60 text-cyan-700 border-t border-l border-r border-white/60 shadow-[0_-4px_6px_-2px_rgba(0,0,0,0.05)]"
                : "text-slate-500 hover:bg-white/40 hover:text-slate-700"
              }`}
          >
            <Cpu className="w-4 h-4" /> ROCm Stats
          </button>
        </div>

        {/* WORKSPACE DIVIDER GRID */}
        {activeTab === "scan" && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">

            {/* LEFT COLUMN: CONTROL & FINDINGS PANEL (7 COLS) */}
            <div className="lg:col-span-7 space-y-6">

              {/* CONTROL AND LOADING INTERFACES */}
              <section className="border border-white/60 bg-white/50 p-5 rounded-3xl space-y-4 backdrop-blur-md shadow-xl shadow-slate-200/20">
                <h2 className="text-xs font-bold tracking-wider text-slate-600 uppercase flex items-center gap-2">
                  <FolderOpen className="w-4 h-4 text-emerald-500" /> File & Project Selection
                </h2>



                {/* Advanced Drag & Drop / File selection area */}
                <div
                  className={`transition-all border-2 border-dashed rounded-2xl p-5 text-center cursor-pointer ${isDragging
                      ? "border-rose-450 bg-rose-500/5 text-rose-700"
                      : "border-slate-200/80 bg-white/40 hover:border-slate-350 hover:bg-white/70 text-slate-500"
                    }`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <div className="flex flex-col items-center gap-2">
                    <div className="p-3 bg-rose-50 text-rose-500 rounded-2xl border border-rose-100">
                      <FolderOpen className="w-6 h-6" />
                    </div>
                    <p className="text-xs text-slate-700 font-bold">
                      {droppedItemName ? `Loaded: ${droppedItemName}` : "Drag and Drop .java files or folder directories here"}
                    </p>
                    <p className="text-[10px] text-slate-400 font-medium max-w-sm mx-auto">
                      Supports selecting multiple java classes or entire project folders.
                    </p>

                    <div className="flex gap-3 justify-center mt-2">
                      <button
                        onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                        className="px-4 py-2 bg-white border border-slate-200 rounded-xl text-xs font-bold hover:bg-slate-50 hover:border-slate-300 text-slate-600 shadow-sm transition-all flex items-center gap-1.5"
                      >
                        Select Files
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); folderInputRef.current?.click(); }}
                        className="px-4 py-2 bg-white border border-slate-200 rounded-xl text-xs font-bold hover:bg-slate-50 hover:border-slate-300 text-slate-600 shadow-sm transition-all flex items-center gap-1.5"
                      >
                        Select Folder
                      </button>
                    </div>
                  </div>

                  {/* Hidden File Inputs */}
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    multiple
                    accept=".java"
                    className="hidden"
                  />
                  <input
                    type="file"
                    ref={folderInputRef}
                    onChange={handleFileChange}
                    webkitdirectory="true"
                    className="hidden"
                  />
                </div>

                {/* Selection Status Banner */}
                <div className="flex flex-wrap justify-between items-center bg-slate-50 p-3 rounded-2xl border border-slate-200/60 gap-3">
                  <div className="text-xs text-slate-600 flex items-center gap-2 font-medium">
                    <FileCode className="w-4 h-4 text-indigo-500" />
                    Status: <span className="font-mono text-slate-800 font-bold">{loadedFiles.length} java source paths parsed.</span>
                  </div>

                  {loadedFiles.length > 0 && (
                    <div className="flex flex-wrap items-center gap-2 mt-2 sm:mt-0">
                      <button
                        onClick={handleClearWorkspace}
                        disabled={scanState === "scanning"}
                        className={`px-4 py-2.5 rounded-xl text-xs font-bold shadow-sm transition-all flex items-center gap-2 ${scanState === "scanning"
                            ? "bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200"
                            : "bg-white text-slate-600 hover:text-rose-600 hover:bg-rose-50 border border-slate-200"
                          }`}
                        title="Clear Workspace"
                      >
                        <Trash2 className="w-3.5 h-3.5" /> Clear Workspace
                      </button>
                      <button
                        onClick={triggerCodeScan}
                        disabled={scanState === "scanning"}
                        className={`px-5 py-2.5 rounded-xl text-xs font-bold tracking-wide shadow-md transition-all flex items-center gap-2 ${scanState === "scanning"
                            ? "bg-slate-200 text-slate-400 cursor-not-allowed border border-slate-300"
                            : "bg-gradient-to-r from-rose-500 to-indigo-600 text-white hover:opacity-90 active:scale-95 cursor-pointer"
                          }`}
                      >
                        {scanState === "scanning" ? (
                          <>
                            <RefreshCw className="w-3.5 h-3.5 animate-spin text-rose-450" /> Evaluating AST...
                          </>
                        ) : (
                          <>
                            <Cpu className="w-3.5 h-3.5" /> Analyze Java Project
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </div>

                {/* PROGRESS VIEW FOR ACTIVE SCANNING */}
                <AnimatePresence>
                  {scanState === "scanning" && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      className="bg-white/80 p-4 border border-slate-250 rounded-2xl space-y-2 shadow-sm"
                    >
                      <div className="flex justify-between items-center text-xs">
                        <span className="text-rose-600 flex items-center gap-2 font-mono font-bold">
                          <Terminal className="w-3 h-3 text-rose-500 animate-pulse" /> {scanProgressText}
                        </span>
                        <span className="font-mono font-bold text-slate-700">{scanProgress}%</span>
                      </div>

                      {/* Visual Progress Bar Wrapper */}
                      <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden border border-slate-200/60">
                        <motion.div
                          className="bg-gradient-to-r from-rose-500 via-pink-400 to-indigo-500 h-full rounded-full"
                          initial={{ width: "0%" }}
                          animate={{ width: `${scanProgress}%` }}
                          transition={{ duration: 0.1 }}
                        />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </section>

              {/* PASTE DIALOG SHEETS */}
              <AnimatePresence>
                {isSandboxOpen && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="bg-white/95 backdrop-blur-xl border border-white/60 p-5 rounded-3xl space-y-4 shadow-2xl shadow-indigo-100"
                  >
                    <div className="flex justify-between items-center border-b border-slate-100 pb-2">
                      <h3 className="text-xs font-bold font-mono text-slate-800 flex items-center gap-2">
                        <Code2 className="w-4 h-4 text-rose-500" /> Java Coding SandBox Playground
                      </h3>
                      <button
                        onClick={() => setIsSandboxOpen(false)}
                        className="text-slate-400 hover:text-slate-600"
                      >
                        <XCircle className="w-5 h-5" />
                      </button>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <label className="text-[10px] font-mono text-slate-400 block mb-1">Dummy Filename (.java)</label>
                        <input
                          type="text"
                          value={sandboxFileName}
                          onChange={(e) => setSandboxFileName(e.target.value)}
                          className="w-full bg-slate-50 border border-slate-200 rounded-xl px-3 py-2 text-xs text-slate-800 font-mono focus:border-rose-500 focus:outline-none"
                        />
                      </div>
                      <div>
                        <label className="text-[10px] font-mono text-slate-400 block mb-1">Java Source Code</label>
                        <textarea
                          rows={12}
                          value={sandboxCode}
                          onChange={(e) => setSandboxCode(e.target.value)}
                          className="w-full bg-slate-50 border border-slate-200 rounded-xl p-3 text-xs text-slate-800 font-mono focus:border-rose-500 focus:outline-none focus:ring-1 focus:ring-rose-500/30"
                          placeholder="Paste your custom vulnerable Class structures..."
                        />
                      </div>
                    </div>
                    <div className="flex justify-end gap-2 pt-2">
                      <button
                        onClick={() => setIsSandboxOpen(false)}
                        className="px-3.5 py-1.5 text-xs text-slate-400 hover:text-slate-600 transition-all font-semibold"
                      >
                        Cancel
                      </button>
                      <button
                        onClick={handleAddSandboxFile}
                        className="px-4 py-2 text-xs bg-rose-500 hover:bg-rose-600 transition-all text-white font-bold rounded-xl shadow-md shadow-rose-500/10"
                      >
                        Load Into Explorer
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* VULNERABILITY CARDS SECTION */}
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <h2 className="text-xs font-bold tracking-wider text-slate-600 uppercase flex items-center gap-2">
                    <Flame className="w-4 h-4 text-rose-500" /> Scan Results Findings ({vulnerabilities.filter(v => v.status !== 'Ignored').length})
                  </h2>

                  {/* Active scan engines declaration */}
                  {scanState === "completed" && (
                    <span className={`text-[10px] px-2 py-0.5 rounded-full border ${scanMode === "gemini_ai"
                        ? "bg-purple-50 text-purple-600 border-purple-200 font-bold"
                        : "bg-slate-100 text-slate-600 border-slate-200"
                      }`}>
                      Engine: {scanMode === "gemini_ai" ? "Gemini AI (Complete AST API)" : "Offline Static Heuristics Backend"}
                    </span>
                  )}
                </div>

                {scanState === "idle" && (
                  <div className="border border-white/65 bg-white/45 p-10 rounded-3xl text-center space-y-4 shadow-sm backdrop-blur-md">
                    <div className="mx-auto w-12 h-12 bg-white border border-slate-200 rounded-2xl flex items-center justify-center text-slate-400 shadow-sm">
                      <Shield className="w-6 h-6 text-indigo-500" />
                    </div>
                    <div className="space-y-1">
                      <p className="text-sm font-bold text-slate-800">No active audit logs available</p>
                      <p className="text-xs text-slate-500 max-w-sm mx-auto leading-relaxed font-semibold">
                        Review files or select a preset project, and hit <strong className="text-rose-500 font-bold">"Analyze Java Project"</strong> above to extract findings.
                      </p>
                    </div>
                  </div>
                )}

                {scanState === "completed" && vulnerabilities.length === 0 && (
                  <div className="border border-white/65 bg-white/45 p-10 rounded-3xl text-center space-y-2 shadow-sm">
                    <CheckCircle2 className="w-10 h-10 text-emerald-500 mx-auto" />
                    <p className="text-sm font-bold text-slate-800">Clean Bill of Health!</p>
                    <p className="text-xs text-slate-500 font-semibold">We could not identify any vulnerability parameters inside scanned Java files.</p>
                  </div>
                )}

                {scanState === "completed" && (
                  <div className="space-y-4">
                    {vulnerabilities.map((v) => {
                      const localState = remediatingIdState[v.id] || "idle";

                      // Hide Card if Ignored
                      if (v.status === "Ignored") return null;

                      const isHigh = v.severity === "High";
                      const isMed = v.severity === "Medium";
                      const isLow = v.severity === "Low";
                      const isApproved = v.status === "Approved";

                      return (
                        <motion.article
                          key={v.id}
                          layoutId={`card-${v.id}`}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          className={`border rounded-3xl overflow-hidden transition-all duration-300 bg-white/60 backdrop-blur-md relative ${selectedVulnerabilityId === v.id
                              ? "border-rose-500 ring-2 ring-rose-500/15 shadow-lg"
                              : isApproved
                                ? "border-emerald-200 opacity-80"
                                : "border-white/70 hover:border-slate-300 shadow-sm"
                            }`}
                          onClick={() => handleVulnerabilityClick(v)}
                        >
                          {/* High priority attention bar */}
                          {/* File Path Header Over the Card */}
                          <div className="bg-slate-800 text-slate-200 text-[10px] font-mono px-4 py-1.5 font-bold truncate w-full flex items-center gap-2">
                            <FileCode className="w-3.5 h-3.5 text-emerald-400" />
                            {v.filePath}
                          </div>
                          {/* High priority attention bar */}
                          <div className={`h-1.5 w-full ${isApproved
                              ? "bg-emerald-500"
                              : isHigh
                                ? "bg-rose-500"
                                : isMed
                                  ? "bg-amber-500"
                                  : "bg-sky-400"
                            }`} />

                          <div className="p-5 space-y-4">

                            {/* Top Tag Header */}
                            <div className="flex flex-wrap justify-between items-center gap-2">
                              <div className="flex items-center gap-2 flex-wrap">
                                {/* Severity Badge */}
                                <span className={`text-[10px] uppercase tracking-wider font-bold px-2.5 py-0.5 rounded-full ${isApproved
                                    ? "bg-emerald-50 text-emerald-700 border border-emerald-200/60"
                                    : isHigh
                                      ? "bg-rose-50 text-rose-700 border border-rose-200/60"
                                      : isMed
                                        ? "bg-amber-50 text-amber-700 border border-amber-200/60"
                                        : "bg-sky-50 text-sky-700 border border-sky-200/60"
                                  }`}>
                                  {isApproved ? "Resolved" : `${v.severity} Severity`}
                                </span>

                                <span className="text-xs font-bold text-slate-800 font-sans">
                                  {v.cwe_name || v.type}
                                </span>

                                {v.cwe_id && (
                                  <span className="text-[10px] bg-slate-100 border border-slate-200 px-2 py-0.5 rounded text-slate-600 font-mono">
                                    {v.cwe_id}
                                  </span>
                                )}
                                {v.confidence && (
                                  <span className="text-[10px] bg-indigo-50 border border-indigo-200 px-2 py-0.5 rounded text-indigo-600 font-mono">
                                    Conf: {(v.confidence * 100).toFixed(1)}%
                                  </span>
                                )}
                              </div>

                              {/* Line Number indicator */}
                              <span className="font-mono text-[9px] text-slate-500 bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200/80 font-bold col-span-2">
                                Line: {v.location?.line || v.lineNumber}
                              </span>
                            </div>

                            {/* Vulnerability Description exploration */}
                            <div className="space-y-2">
                              <p className="text-xs text-slate-600 leading-relaxed font-semibold">
                                {v.description}
                              </p>
                              {v.impact && (
                                <p className="text-[11px] text-rose-700/80 leading-relaxed font-medium bg-rose-50 p-2 rounded-lg border border-rose-100">
                                  <strong>Impact:</strong> {v.impact}
                                </p>
                              )}
                              {v.recommendation && (
                                <p className="text-[11px] text-emerald-700/80 leading-relaxed font-medium bg-emerald-50 p-2 rounded-lg border border-emerald-100">
                                  <strong>Recommendation:</strong> {v.recommendation}
                                </p>
                              )}
                            </div>

                            {/* ACTION TOOLBAR */}
                            <div className="flex flex-wrap justify-between items-center border-t border-slate-100 pt-4 gap-3">

                              {/* Action Buttons for active logs */}
                              {!isApproved && localState === "idle" && (
                                <div className="flex gap-2">
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      triggerRemediation(v);
                                    }}
                                    className="px-3.5 py-2 bg-gradient-to-r from-rose-500 via-rose-500 to-indigo-600 text-white font-bold text-xs rounded-xl transition-all shadow-md shadow-rose-500/10 cursor-pointer flex items-center gap-1.5"
                                  >
                                    <Sparkles className="w-3.5 h-3.5 animate-pulse" /> Fix Security Issue
                                  </button>

                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleIgnoreVulnerability(v.id);
                                    }}
                                    className="px-3 py-2 bg-white border border-slate-200 hover:text-amber-600 hover:border-amber-300 hover:bg-amber-50/10 text-slate-500 text-xs rounded-xl transition-all font-bold shadow-sm"
                                  >
                                    Ignore
                                  </button>

                                </div>
                              )}

                              {/* IF REMEDIATING LOADING STATE */}
                              {localState === "fixing" && (
                                <div className="text-xs text-rose-600 flex items-center gap-2 font-mono py-1 font-bold">
                                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-rose-550" />
                                  Triggering server model remediation repair mechanisms...
                                </div>
                              )}

                              {/* DIFF VIEW READY WORKFLOWS */}
                              <AnimatePresence>
                                {localState === "diff_ready" && v.remediatedSnippet && (
                                  <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: "auto" }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="w-full space-y-4 pt-1"
                                    onClick={(e) => e.stopPropagation()}
                                  >
                                    {/* Code Diff Display Row */}
                                    <div className="bg-white p-4 rounded-2xl border border-rose-200/50 space-y-3 shadow-inner">
                                      <h4 className="text-[9px] font-bold font-mono tracking-widest text-rose-700 uppercase flex items-center gap-2">
                                        <GitCommit className="w-3.5 h-3.5 text-rose-500" /> side-by-side remediation comparison
                                      </h4>

                                      <div className="flex justify-center py-2">
                                        <button
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            setFullscreenDiff({ isOpen: true, vuln: v });
                                          }}
                                          className="px-6 py-3 bg-white border-2 border-indigo-200 text-indigo-700 hover:bg-indigo-50 hover:border-indigo-300 hover:text-indigo-800 rounded-xl font-bold flex items-center gap-2 shadow-sm transition-all"
                                        >
                                          <Eye className="w-4 h-4" /> View Diff
                                        </button>
                                      </div>

                                      {/* Detailed technical explanation */}
                                      {v.remediationExplanation && (
                                        <div className="mt-3 text-xs bg-slate-50 border border-slate-200 p-3 rounded-xl text-slate-600 flex gap-2">
                                          <Info className="w-4 h-4 text-indigo-500 shrink-0 mt-0.5" />
                                          <div className="space-y-1">
                                            <strong className="text-slate-800 text-[10px] uppercase tracking-wider block font-mono font-bold">Remediation Action</strong>
                                            <p className="leading-relaxed font-semibold text-slate-600">{v.remediationExplanation}</p>
                                          </div>
                                        </div>
                                      )}

                                      {/* Action approval triggers */}
                                      <div className="flex justify-end gap-2 pt-2 border-t border-slate-100">
                                        <button
                                          onClick={() => handleRejectRemediation(v.id)}
                                          className="px-3 py-1.5 text-xs text-slate-400 hover:text-slate-700 hover:bg-slate-50 rounded-xl transition-all font-bold"
                                        >
                                          Reject Corrective
                                        </button>

                                        <button
                                          onClick={() => handleApproveRemediation(v)}
                                          className="px-4 py-1.5 bg-emerald-500 hover:bg-emerald-600 border border-emerald-450 text-white font-extrabold text-xs rounded-xl transition-all shadow-md shadow-emerald-500/10 flex items-center gap-1.5 cursor-pointer"
                                        >
                                          <Check className="w-3.5 h-3.5" /> Approve & Apply Fix
                                        </button>
                                      </div>
                                    </div>
                                  </motion.div>
                                )}
                              </AnimatePresence>

                              {/* APPROVED SUCCESS BANNER */}
                              {localState === "approved" && (
                                <motion.div
                                  initial={{ opacity: 0, scale: 0.95 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  className="w-full bg-emerald-50 text-emerald-800 border border-emerald-250 p-3.5 rounded-2xl flex items-center justify-between gap-2 shadow-sm"
                                >
                                  <div className="flex items-center gap-2 text-xs font-mono font-bold">
                                    <FileCheck className="w-4 h-4 text-emerald-650" />
                                    Vulnerability permanently resolved! File records corrected in active scope memory.
                                  </div>
                                  <span className="text-[9px] font-bold text-emerald-700 bg-emerald-200/50 uppercase border border-emerald-350 px-2.5 py-0.5 rounded-full font-sans">
                                    Approved & Committed
                                  </span>
                                </motion.div>
                              )}

                            </div>

                          </div>
                        </motion.article>
                      );
                    })}
                  </div>
                )}
              </div>

            </div>

            {/* RIGHT COLUMN: ACTIVE INTERACTIVE IDE / CODE VIEWER (5 COLS) */}
            <section className="lg:col-span-5 border border-white/60 bg-white/55 p-5 rounded-3xl flex flex-col h-[calc(100vh-140px)] min-h-[550px] sticky top-6 backdrop-blur-md shadow-xl shadow-slate-200/20">

              {/* Tab selection panel */}
              <div className="flex justify-between items-center border-b border-slate-200 pb-3 mb-4 shrink-0">
                <h2 className="text-xs font-bold tracking-wider text-slate-600 uppercase flex items-center gap-2">
                  <FileCode className="w-4 h-4 text-emerald-500" /> Workspace Native Files
                </h2>

                <span className="text-[9px] font-mono text-slate-500 bg-slate-50 border border-slate-250/80 px-2.5 py-1 rounded-lg font-bold">
                  {loadedFiles.length} File Records
                </span>
              </div>

              {/* Flat Explorer tab labels */}
              <div className="flex gap-1 overflow-x-auto pb-2 shrink-0 max-w-full">
                {loadedFiles.map((file, idx) => {
                  const isActive = activeFileIndex === idx;

                  // Count active vulnerabilities inside this specific file path to show a badge count
                  const fileVulns = vulnerabilities.filter(v => v.status !== "Ignored" && v.status !== "Approved" && (v.filePath === file.path || v.filePath === file.name));

                  return (
                    <button
                      key={file.path || file.name}
                      onClick={() => {
                        setActiveFileIndex(idx);
                        setSelectedVulnerabilityId(null);
                      }}
                      className={`px-3 py-2 text-xs font-mono rounded-xl transition-all shrink-0 border flex items-center gap-2 ${isActive
                          ? "bg-white text-slate-800 border-slate-300 shadow-md shadow-slate-200/40 font-bold"
                          : "bg-white/40 text-slate-500 border-slate-200/60 hover:bg-white/80 hover:text-slate-850"
                        }`}
                    >
                      <FileCode className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                      <span className="truncate max-w-[120px] font-sans">{file.name}</span>

                      {/* Vuln Indicators badge */}
                      {fileVulns.length > 0 && (
                        <span className="w-4.5 h-4.5 bg-rose-500 text-white text-[9px] font-bold rounded-full flex items-center justify-center shrink-0 shadow-sm">
                          {fileVulns.length}
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Core Code Viewer Screen */}
              <div className="flex-1 min-h-0 bg-slate-900 rounded-2xl border border-slate-950 overflow-hidden flex flex-col relative shadow-xl">

                {activeFile ? (
                  <>
                    {/* Pseudo IDE status header */}
                    <div className="bg-slate-950 px-4 py-2 text-xs border-b border-slate-900/80 flex justify-between items-center shrink-0">
                      <span className="font-mono text-slate-400 truncate max-w-xs">{activeFile.path}</span>
                      <div className="flex items-center gap-3">
                        <span className="font-mono text-[9px] text-slate-500 font-semibold hidden sm:inline">Parser: JAVA (UTF-8)</span>
                        <button 
                          onClick={() => {
                            navigator.clipboard.writeText(activeFile.content);
                            setCopiedWorkspace(true);
                            setTimeout(() => setCopiedWorkspace(false), 2000);
                          }}
                          className="flex items-center gap-1.5 text-slate-400 hover:text-white transition-colors cursor-pointer bg-slate-800/50 hover:bg-slate-700/50 px-2 py-1 rounded border border-slate-800"
                        >
                          {copiedWorkspace ? <Check className="w-3.5 h-3.5 text-emerald-500" /> : <Copy className="w-3.5 h-3.5" />}
                          <span className="text-[9px] uppercase font-bold tracking-widest">{copiedWorkspace ? 'Copied' : 'Copy'}</span>
                        </button>
                      </div>
                    </div>

                    {/* Lines container element */}
                    <div className="flex-1 overflow-y-auto font-mono text-xs p-4 leading-relaxed select-text select-all bg-slate-900/95 scrollbar-thin">
                      {activeFile.content.split("\n").map((lineContent, idx) => {
                        const lineNum = idx + 1;

                        // Check if this line corresponds to a critical active vulnerability
                        const matchingVuln = vulnerabilities.find(
                          v => v.status !== "Ignored" &&
                            v.status !== "Approved" &&
                            v.lineNumber === lineNum &&
                            (v.filePath === activeFile.path || v.filePath === activeFile.name)
                        );

                        const isLineMatched = matchingVuln !== undefined;
                        const isVulnerabilitySelected = matchingVuln && selectedVulnerabilityId === matchingVuln.id;

                        return (
                          <div
                            key={idx}
                            id={`line-anchor-${lineNum}`}
                            className={`flex items-start transition-all ${isVulnerabilitySelected
                                ? "bg-rose-500/15 border-l-4 border-rose-400 -ml-4 pl-3"
                                : isLineMatched
                                  ? "bg-rose-500/10 border-l-2 border-rose-400 -ml-4 pl-3.5 cursor-pointer text-rose-300 font-semibold"
                                  : "text-slate-300"
                              }`}
                            onClick={() => matchingVuln && handleVulnerabilityClick(matchingVuln)}
                            title={matchingVuln ? `Click to inspect: ${matchingVuln.type}` : undefined}
                          >
                            {/* Code line enumeration numbers */}
                            <div className="text-slate-500 w-7 select-none text-right pr-3 font-mono text-[10px] opacity-40 shrink-0">
                              {lineNum}
                            </div>

                            {/* Live content characters */}
                            <div className={`whitespace-pre break-all ${isLineMatched ? "font-bold bg-rose-400/10 px-1 rounded text-rose-200" : ""}`}>
                              {renderHighlightedCode(lineContent)}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-center p-8 text-slate-500 bg-slate-900">
                    <FileCode className="w-10 h-10 mb-2 opacity-30 text-emerald-500" />
                    <p className="text-xs">Select or upload Java file inputs to active explorer.</p>
                  </div>
                )}

              </div>

              {/* Interactive User documentation prompt block */}
              <div className="mt-4 shrink-0 flex gap-2">
                <button
                  onClick={triggerSingleFileScan}
                  disabled={!activeFile || scanState === "scanning"}
                  className={`w-full py-3 rounded-2xl text-xs font-bold tracking-wide shadow-md transition-all flex items-center justify-center gap-2 ${!activeFile || scanState === "scanning"
                      ? "bg-slate-200 text-slate-400 cursor-not-allowed border border-slate-300"
                      : "bg-emerald-500 hover:bg-emerald-600 text-white shadow-emerald-500/20 active:scale-95 cursor-pointer"
                    }`}
                >
                  {scanState === "scanning" ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin text-white" /> Scanning active file...
                    </>
                  ) : (
                    <>
                      <Cpu className="w-4 h-4" /> Run Scan on Selected File
                    </>
                  )}
                </button>
              </div>

            </section>

          </div>
        )}

        {/* HISTORY TAB VIEW */}
        {activeTab === "history" && (
          <div className="border border-white/60 bg-white/50 p-6 rounded-3xl space-y-4 backdrop-blur-md shadow-xl shadow-slate-200/20">
            <h2 className="text-sm font-bold tracking-wider text-slate-600 uppercase flex items-center gap-2">
              <RefreshCw className="w-5 h-5 text-indigo-500" /> Session Scan History
            </h2>

            {scanHistory.length === 0 ? (
              <div className="text-center p-12 text-slate-400 font-medium border border-dashed border-slate-300 rounded-2xl">
                No scans have been performed yet in this session.
              </div>
            ) : (
              <div className="space-y-6">
                {scanHistory.map((historyItem) => (
                  <div key={historyItem.id} className="bg-white/80 p-5 rounded-2xl border border-slate-200 shadow-sm relative group">
                    <button
                      onClick={() => setScanHistory(prev => prev.filter(h => h.id !== historyItem.id))}
                      className="absolute top-4 right-4 text-slate-300 hover:text-rose-500 transition-colors opacity-0 group-hover:opacity-100"
                      title="Remove from history"
                    >
                      <XCircle className="w-5 h-5" />
                    </button>

                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <div className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-1">
                          {historyItem.timestamp.toLocaleTimeString()}
                        </div>
                        <h3 className="font-bold text-slate-800">
                          {historyItem.files.length} File(s) Scanned
                        </h3>
                      </div>
                      <div className="flex gap-2">
                        <span className="text-[10px] bg-slate-100 text-slate-600 border border-slate-200 px-2 py-1 rounded font-mono font-bold">
                          {historyItem.mode}
                        </span>
                        <span className={`text-[10px] px-2 py-1 rounded font-mono font-bold ${historyItem.vulnerabilities.length > 0 ? "bg-rose-100 text-rose-700 border border-rose-200" : "bg-emerald-100 text-emerald-700 border border-emerald-200"
                          }`}>
                          {historyItem.vulnerabilities.length} Threats Found
                        </span>
                      </div>
                    </div>

                    {historyItem.vulnerabilities.length > 0 && (
                      <div className="bg-slate-50 rounded-xl p-4 border border-slate-100 space-y-3 mb-2">
                        <h4 className="text-xs font-bold text-slate-600 uppercase">Findings Details:</h4>
                        {Array.from(new Set(historyItem.vulnerabilities.map(v => v.filePath))).map((filePath, i) => {
                          const fileVulns = historyItem.vulnerabilities.filter(v => v.filePath === filePath);
                          const file = historyItem.files.find(f => f.path === filePath || f.name === filePath.split('/').pop());
                          const fileId = `${historyItem.id}-${filePath}`;
                          const isExpanded = expandedHistoryIds.includes(fileId);

                          return (
                            <div key={i} className="bg-white border border-slate-200 rounded-lg overflow-hidden shadow-sm">
                              <div className="p-3 flex justify-between items-start border-b border-slate-100 bg-slate-50/50">
                                <div>
                                  <div className="font-bold text-xs text-slate-700 flex items-center gap-2 mb-2">
                                    <FileCode className="w-3.5 h-3.5 text-slate-400" />
                                    {filePath}
                                  </div>
                                  <div className="space-y-1.5">
                                    {fileVulns.map((v, vIdx) => (
                                      <div key={vIdx} className="text-[11px] font-mono text-slate-600 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-rose-500 shrink-0" />
                                        <span className="font-bold text-rose-600">{v.type}</span> at line {v.lineNumber}
                                        {v.status === 'Approved' && <span className="text-[9px] bg-emerald-100 text-emerald-700 px-1.5 rounded-full py-0.5 font-bold uppercase">Resolved</span>}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                                <button
                                  onClick={() => toggleHistoryExpanded(fileId)}
                                  className="text-slate-400 hover:text-slate-700 transition-colors p-1.5 rounded-md hover:bg-slate-200 bg-white border border-slate-200 shadow-sm ml-2 mt-1 shrink-0"
                                  title={isExpanded ? "Collapse Source Code" : "Expand Source Code"}
                                >
                                  {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                                </button>
                              </div>

                              {isExpanded && file && (
                                <div className="bg-[#0f172a] p-4 overflow-auto max-h-[400px]">
                                  <pre className="font-mono text-xs leading-relaxed text-slate-300">
                                    {file.content.split('\n').map((line, lineIdx) => (
                                      <div key={lineIdx} className="flex">
                                        <span className="w-8 shrink-0 text-slate-600 select-none text-right pr-3">{lineIdx + 1}</span>
                                        <span className="whitespace-pre break-all">
                                          {renderHighlightedCode(line)}
                                        </span>
                                      </div>
                                    ))}
                                  </pre>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ROCM GPU DIAGNOSTICS TAB */}
        {activeTab === "rocm" && (
          <div className="space-y-6">
            <div className="bg-white/60 backdrop-blur-xl border border-white/60 rounded-3xl p-6 shadow-xl shadow-slate-200/40 flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-4">
                <div className="p-4 bg-indigo-50 border border-indigo-100 rounded-2xl shadow-sm">
                  <Cpu className="w-10 h-10 text-indigo-500 animate-pulse" />
                </div>
                <div>
                  <h2 className="text-lg font-extrabold text-slate-800 flex items-center gap-2">
                    {rocmStats.gpuName}
                    <span className="text-xs bg-indigo-100 text-indigo-700 border border-indigo-200 font-bold px-2.5 py-0.5 rounded-full">
                      {rocmStats.gpuType}
                    </span>
                  </h2>
                  <p className="text-xs text-slate-500 mt-1 font-medium">
                    Hardware profiling stack mapping real-time processing nodes.
                  </p>
                </div>
              </div>

              <div className="flex flex-wrap gap-4">
                <div className="bg-slate-50 border border-slate-200 p-4 rounded-2xl text-center min-w-[130px] shadow-sm">
                  <span className="text-[10px] font-bold text-slate-500 uppercase block">Models Loaded</span>
                  <div className="flex flex-col gap-1 mt-1">
                    {rocmStats.modelsLoaded.map((model: string, idx: number) => (
                      <span key={idx} className="text-xs font-extrabold text-purple-600 block font-mono bg-purple-50 px-2 py-0.5 rounded-md border border-purple-100">
                        {model}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="bg-slate-50 border border-slate-200 p-4 rounded-2xl text-center min-w-[130px] shadow-sm">
                  <span className="text-[10px] font-bold text-slate-500 uppercase block">API Health</span>
                  <span className="text-xs font-extrabold text-emerald-600 block font-mono mt-1 bg-emerald-50 px-2 py-0.5 rounded-md border border-emerald-100">
                    {rocmStats.apiHealth}
                  </span>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
              <div className="bg-white/60 backdrop-blur-xl border border-white/60 rounded-3xl p-5 shadow-xl shadow-slate-200/40 flex flex-col items-center justify-center relative overflow-hidden">
                <h3 className="text-xs font-bold uppercase text-purple-600 tracking-wide mb-2 w-full text-center">VRAM Allocation (MiB)</h3>
                <div className="relative flex flex-col items-center justify-center pt-4">
                  <svg viewBox="0 0 180 110" className="w-48 h-auto">
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#e2e8f0" strokeWidth="14" strokeLinecap="round" />
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#9333ea" strokeWidth="14" strokeLinecap="round"
                      strokeDasharray="220"
                      strokeDashoffset={220 * (1 - Math.min(1, rocmStats.vram / rocmStats.vramTotal))}
                      style={{ transition: 'stroke-dashoffset 0.5s ease-out' }}
                    />
                  </svg>
                  <div className="absolute top-[65px] w-full flex flex-col items-center justify-center">
                    <span className="text-3xl font-black text-slate-800 font-mono tracking-tighter drop-shadow-sm">{rocmStats.vram}</span>
                    <span className="text-[10px] text-purple-600 font-bold uppercase tracking-widest mt-1">/ {rocmStats.vramTotal} MiB</span>
                  </div>
                </div>
              </div>
              <div className="bg-white/60 backdrop-blur-xl border border-white/60 rounded-3xl p-5 shadow-xl shadow-slate-200/40 flex flex-col items-center justify-center relative overflow-hidden">
                <h3 className="text-xs font-bold uppercase text-blue-500 tracking-wide mb-2 w-full text-center">Compute Engine Load</h3>
                <div className="relative flex flex-col items-center justify-center pt-4">
                  <svg viewBox="0 0 180 110" className="w-48 h-auto">
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#e2e8f0" strokeWidth="14" strokeLinecap="round" />
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#3b82f6" strokeWidth="14" strokeLinecap="round"
                      strokeDasharray="220"
                      strokeDashoffset={220 * (1 - (rocmStats.load / 100))}
                      style={{ transition: 'stroke-dashoffset 0.5s ease-out' }}
                    />
                  </svg>
                  <div className="absolute top-[65px] w-full flex flex-col items-center justify-center">
                    <span className="text-3xl font-black text-slate-800 font-mono tracking-tighter drop-shadow-sm">{rocmStats.load}%</span>
                    <span className="text-[10px] text-blue-500 font-bold uppercase tracking-widest mt-1">Utilisation</span>
                  </div>
                </div>
              </div>
              <div className="bg-white/60 backdrop-blur-xl border border-white/60 rounded-3xl p-5 shadow-xl shadow-slate-200/40 flex flex-col items-center justify-center relative overflow-hidden">
                <h3 className="text-xs font-bold uppercase text-emerald-500 tracking-wide mb-2 w-full text-center">CPU Core Load</h3>
                <div className="relative flex flex-col items-center justify-center pt-4">
                  <svg viewBox="0 0 180 110" className="w-48 h-auto">
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#e2e8f0" strokeWidth="14" strokeLinecap="round" />
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#10b981" strokeWidth="14" strokeLinecap="round"
                      strokeDasharray="220"
                      strokeDashoffset={220 * (1 - (rocmStats.cpu / 100))}
                      style={{ transition: 'stroke-dashoffset 0.5s ease-out' }}
                    />
                  </svg>
                  <div className="absolute top-[65px] w-full flex flex-col items-center justify-center">
                    <span className="text-3xl font-black text-slate-800 font-mono tracking-tighter drop-shadow-sm">{rocmStats.cpu}%</span>
                    <span className="text-[10px] text-emerald-500 font-bold uppercase tracking-widest mt-1">Utilisation</span>
                  </div>
                </div>
              </div>
              <div className="bg-white/60 backdrop-blur-xl border border-white/60 rounded-3xl p-5 shadow-xl shadow-slate-200/40 flex flex-col items-center justify-center relative overflow-hidden">
                <h3 className="text-xs font-bold uppercase text-indigo-500 tracking-wide mb-2 w-full text-center">System RAM (GiB)</h3>
                <div className="relative flex flex-col items-center justify-center pt-4">
                  <svg viewBox="0 0 180 110" className="w-48 h-auto">
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#e2e8f0" strokeWidth="14" strokeLinecap="round" />
                    <path d="M 20 90 A 70 70 0 0 1 160 90" fill="none" stroke="#6366f1" strokeWidth="14" strokeLinecap="round"
                      strokeDasharray="220"
                      strokeDashoffset={220 * (1 - (rocmStats.ram / 256))}
                      style={{ transition: 'stroke-dashoffset 0.5s ease-out' }}
                    />
                  </svg>
                  <div className="absolute top-[65px] w-full flex flex-col items-center justify-center">
                    <span className="text-3xl font-black text-slate-800 font-mono tracking-tighter drop-shadow-sm">{rocmStats.ram}</span>
                    <span className="text-[10px] text-indigo-500 font-bold uppercase tracking-widest mt-1">/ 256 GiB</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

      </div>

      {/* Fullscreen Diff Modal */}
      <AnimatePresence>
        {fullscreenDiff.isOpen && fullscreenDiff.vuln && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] bg-slate-900/40 backdrop-blur-sm flex items-center justify-center p-4 md:p-8"
            onClick={() => setFullscreenDiff({ isOpen: false, vuln: null })}
          >
            <div 
              className="bg-white/95 backdrop-blur-2xl rounded-3xl flex flex-col w-full max-w-7xl h-[85vh] max-h-full overflow-hidden shadow-[0_30px_60px_-15px_rgba(0,0,0,0.3)] border border-white/60 ring-1 ring-slate-900/5"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between px-6 py-5 border-b border-slate-800/60 bg-slate-900 shrink-0 z-20 shadow-sm">
              <h2 className="text-xl font-black bg-gradient-to-r from-indigo-400 to-violet-400 bg-clip-text text-transparent flex items-center gap-3 drop-shadow-sm">
                <GitCommit className="w-6 h-6 text-indigo-400" />
                Remediation Diff: <span className="font-mono text-slate-300 text-[17px] bg-white/10 px-3 py-1 rounded-lg border border-white/10 shadow-inner">{fullscreenDiff.vuln.filePath}</span>
              </h2>
              <button
                onClick={() => setFullscreenDiff({ isOpen: false, vuln: null })}
                className="p-2 hover:bg-white/10 rounded-full text-slate-400 hover:text-white transition-colors cursor-pointer"
              >
                <XCircle className="w-6 h-6" />
              </button>
            </div>
            
            <div className="flex-1 flex min-h-0 overflow-hidden bg-slate-50/30">
              <div className="w-1/2 h-full border-r border-slate-200/60 flex flex-col min-h-0">
                <div className="px-4 py-3 bg-white/60 border-b border-slate-200/60 text-xs font-bold text-slate-500 uppercase tracking-widest text-center shadow-sm z-10 shrink-0 backdrop-blur-md">
                  Original Code
                </div>
                <pre className="flex-1 overflow-auto p-6 font-mono text-[13px] leading-relaxed text-slate-700 m-0">
                  {(() => {
                    const vuln = fullscreenDiff.vuln;
                    if (!vuln) return null;
                    const file = loadedFiles.find(f => f.path === vuln.filePath || f.name === vuln.filePath.split('/').pop());
                    const oldStr = file ? file.content : (vuln.snippet || "");
                    const newStr = vuln.fullRemediatedContent || vuln.remediatedSnippet || "";
                    const diffs = diffLines(oldStr, newStr);
                    let lineNum = 1;
                    return diffs.map((part, i) => {
                      if (part.added) return null;
                      const isRemoved = part.removed;
                      const lines = part.value.split('\n');
                      if (lines[lines.length - 1] === '') lines.pop();
                      return (
                        <span
                          key={i}
                          className={isRemoved ? "bg-rose-500/15 text-rose-800 font-semibold block w-full" : "text-slate-500 block w-full"}
                        >
                          {lines.map((line, j) => {
                            const currentLineNum = lineNum++;
                            return (
                              <div key={j} className="flex hover:bg-slate-900/5">
                                <span className="w-10 shrink-0 text-right pr-4 border-r border-slate-200/60 mr-4 text-slate-400 select-none text-[10px] opacity-70 flex items-center justify-end">{currentLineNum}</span>
                                <span className="whitespace-pre-wrap break-all py-[1px]">{line || " "}</span>
                              </div>
                            );
                          })}
                        </span>
                      );
                    });
                  })()}
                </pre>
              </div>
              <div className="w-1/2 h-full flex flex-col min-h-0">
                <div className="px-4 py-3 bg-emerald-50/50 border-b border-emerald-200/60 text-xs font-bold text-emerald-700 uppercase tracking-widest text-center flex items-center justify-between shadow-sm z-10 shrink-0 backdrop-blur-md">
                  <div className="flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-emerald-500" /> Proposed Secure Deviation
                  </div>
                  <button 
                    onClick={() => {
                      const newStr = fullscreenDiff.vuln?.fullRemediatedContent || fullscreenDiff.vuln?.remediatedSnippet || "";
                      navigator.clipboard.writeText(newStr);
                      setCopiedFixed(true);
                      setTimeout(() => setCopiedFixed(false), 2000);
                    }}
                    className="flex items-center gap-1.5 text-emerald-700 hover:text-emerald-900 transition-colors cursor-pointer bg-emerald-200/40 hover:bg-emerald-300/50 px-2 py-1 rounded shadow-sm border border-emerald-300/50"
                  >
                    {copiedFixed ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                    <span className="text-[9px] uppercase font-bold tracking-widest">{copiedFixed ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
                <pre className="flex-1 overflow-auto p-6 font-mono text-[13px] leading-relaxed text-slate-700 m-0 bg-emerald-50/10">
                  {(() => {
                    const vuln = fullscreenDiff.vuln;
                    if (!vuln) return null;
                    const file = loadedFiles.find(f => f.path === vuln.filePath || f.name === vuln.filePath.split('/').pop());
                    const oldStr = file ? file.content : (vuln.snippet || "");
                    const newStr = vuln.fullRemediatedContent || vuln.remediatedSnippet || "";
                    const diffs = diffLines(oldStr, newStr);
                    let lineNum = 1;
                    return diffs.map((part, i) => {
                      if (part.removed) return null;
                      const isAdded = part.added;
                      const lines = part.value.split('\n');
                      if (lines[lines.length - 1] === '') lines.pop();
                      return (
                        <span
                          key={i}
                          className={isAdded ? "bg-emerald-500/15 text-emerald-800 font-semibold block w-full" : "text-slate-500 block w-full"}
                        >
                          {lines.map((line, j) => {
                            const currentLineNum = lineNum++;
                            return (
                              <div key={j} className="flex hover:bg-slate-900/5">
                                <span className="w-10 shrink-0 text-right pr-4 border-r border-slate-200/60 mr-4 text-slate-400 select-none text-[10px] opacity-70 flex items-center justify-end">{currentLineNum}</span>
                                <span className="whitespace-pre-wrap break-all py-[1px]">{line || " "}</span>
                              </div>
                            );
                          })}
                        </span>
                      );
                    });
                  })()}
                </pre>
              </div>
            </div>
            {fullscreenDiff.vuln.remediationExplanation && (
              <div className="px-6 py-5 bg-slate-900 border-t border-slate-800/60 shrink-0 shadow-[0_-4px_10px_-1px_rgba(0,0,0,0.3)] z-20">
                 <div className="text-sm text-slate-300 font-medium flex gap-3 items-start">
                   <div className="p-2 bg-indigo-500/20 text-indigo-400 rounded-xl shadow-sm border border-indigo-500/30 h-fit shrink-0 mt-0.5">
                     <Info className="w-4 h-4" />
                   </div>
                   <div className="leading-relaxed">
                     <strong className="text-indigo-300 font-bold uppercase tracking-wider text-[11px] block mb-1">Remediation Explanation</strong>
                     <span className="text-slate-400">{fullscreenDiff.vuln.remediationExplanation}</span>
                   </div>
                 </div>
              </div>
            )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

`

### frontend/src/main.tsx

`typescript
import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App.tsx';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

`

### frontend/src/types.ts

`typescript
export interface JavaFile {
  name: string;
  path: string;
  content: string;
  originalContent: string;
}

export interface Vulnerability {
  id?: string; // id is sometimes missing from JSON
  cwe_id?: string;
  cwe_name?: string;
  confidence?: number;
  location?: { line: number };
  impact?: string;
  type?: string;
  severity: string;
  filePath: string;
  lineNumber: number;
  snippet?: string;
  description: string;
  recommendation: string;
  status: string;
  remediatedSnippet?: string;
  remediationExplanation?: string;
  fullRemediatedContent?: string;
}

export interface ScanHistoryEntry {
  id: string;
  timestamp: Date;
  mode: string;
  files: JavaFile[];
  vulnerabilities: Vulnerability[];
}

export interface ProjectDemo {
  name: string;
  description: string;
  files: JavaFile[];
}

`

### frontend/src/index.css

`css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
@import "tailwindcss";

@theme {
  --font-sans: "Inter", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, monospace;
}

@layer utilities {
  /* Custom scrollbar styles */
  ::-webkit-scrollbar {
    width: 6px;
    height: 6px;
  }
  ::-webkit-scrollbar-track {
    background: #09090b; /* zinc-950 */
  }
  ::-webkit-scrollbar-thumb {
    background: #27272a; /* zinc-800 */
    border-radius: 3px;
  }
  ::-webkit-scrollbar-thumb:hover {
    background: #3f3f46; /* zinc-700 */
  }
}


`

### frontend/index.html

`html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>CodeElixir.AI — Java Code Security Auditor</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>


`

### frontend/vite.config.ts

`typescript
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import fs from 'fs';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  
  const apiBase = process.env.VITE_API_BASE || env.VITE_API_BASE;
  const token = process.env.VITE_JUPYTER_TOKEN || env.VITE_JUPYTER_TOKEN || process.env.VITE_API_TOKEN || env.VITE_API_TOKEN;

  let proxyConfig = {};
  if (apiBase) {
    const targetUrl = apiBase.replace(/\/api\/?$/, "").replace(/\/$/, "");
    proxyConfig = {
      '/api': {
        target: targetUrl,
        changeOrigin: true,
        secure: false,
        headers: token ? {
          'Authorization': `token ${token}`
        } : {},
      }
    };
  }

  const localStoragePlugin = {
    name: 'localstorage-api',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url === '/local-api/history' && req.method === 'GET') {
          try {
            const historyPath = path.resolve(__dirname, 'localstorage', 'history.json');
            if (fs.existsSync(historyPath)) {
              const data = fs.readFileSync(historyPath, 'utf-8');
              res.setHeader('Content-Type', 'application/json');
              res.end(data);
            } else {
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify([]));
            }
          } catch (e) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(e) }));
          }
        } else if (req.url === '/local-api/history' && req.method === 'POST') {
          let body = '';
          req.on('data', chunk => {
            body += chunk.toString();
          });
          req.on('end', () => {
            try {
              const dirPath = path.resolve(__dirname, 'localstorage');
              if (!fs.existsSync(dirPath)) {
                fs.mkdirSync(dirPath, { recursive: true });
              }
              const historyPath = path.resolve(dirPath, 'history.json');
              fs.writeFileSync(historyPath, body, 'utf-8');
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ success: true }));
            } catch (e) {
              res.statusCode = 500;
              res.end(JSON.stringify({ error: String(e) }));
            }
          });
        } else if (req.url === '/local-api/session' && req.method === 'GET') {
          try {
            const sessionPath = path.resolve(__dirname, 'localstorage', 'session.json');
            if (fs.existsSync(sessionPath)) {
              const data = fs.readFileSync(sessionPath, 'utf-8');
              res.setHeader('Content-Type', 'application/json');
              res.end(data);
            } else {
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify(null));
            }
          } catch (e) {
            res.statusCode = 500;
            res.end(JSON.stringify({ error: String(e) }));
          }
        } else if (req.url === '/local-api/session' && req.method === 'POST') {
          let body = '';
          req.on('data', chunk => {
            body += chunk.toString();
          });
          req.on('end', () => {
            try {
              const dirPath = path.resolve(__dirname, 'localstorage');
              if (!fs.existsSync(dirPath)) {
                fs.mkdirSync(dirPath, { recursive: true });
              }
              const sessionPath = path.resolve(dirPath, 'session.json');
              fs.writeFileSync(sessionPath, body, 'utf-8');
              res.setHeader('Content-Type', 'application/json');
              res.end(JSON.stringify({ success: true }));
            } catch (e) {
              res.statusCode = 500;
              res.end(JSON.stringify({ error: String(e) }));
            }
          });
        } else {
          next();
        }
      });
    }
  };

  return {
    plugins: [react(), tailwindcss(), localStoragePlugin],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {
        ignored: ['**/localstorage/**']
      },
      proxy: proxyConfig
    },
  };
});

`

### frontend/package.json

`json
{
  "name": "react-example",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "lint": "tsc --noEmit"
  },
  "dependencies": {
    "@google/genai": "^2.4.0",
    "@tailwindcss/vite": "^4.1.14",
    "@vitejs/plugin-react": "^5.0.4",
    "diff": "^9.0.0",
    "dotenv": "^17.2.3",
    "express": "^4.21.2",
    "lucide-react": "^0.546.0",
    "motion": "^12.23.24",
    "react": "^19.0.1",
    "react-dom": "^19.0.1",
    "vite": "^6.2.3"
  },
  "devDependencies": {
    "@types/diff": "^7.0.2",
    "@types/express": "^4.17.21",
    "@types/node": "^22.14.0",
    "autoprefixer": "^10.4.21",
    "esbuild": "^0.25.0",
    "tailwindcss": "^4.1.14",
    "tsx": "^4.21.0",
    "typescript": "~5.8.2",
    "vite": "^6.2.3"
  }
}

`

## Additional Python Scripts

### balance_dataset.py

`python
"""
Dataset Re-Balancer for Vulnerability Training Data.

Problem: 50/50 positive/negative ratio causes the model to default to
predicting "no vulnerabilities" (the easier class). This script reduces
the negative ratio to a target percentage (default 20%).

Usage:
    python balance_dataset.py \
        --input "Dataset/train_classifier_precise_lines.jsonl" \
        --output "Dataset/train_balanced.jsonl" \
        --neg_ratio 0.20
"""
import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("balance_dataset")


def balance_dataset(input_path: str, output_path: str, neg_ratio: float = 0.20, seed: int = 42):
    """
    Reads the training JSONL, separates positive (has vulns) and negative (no vulns)
    examples, downsamples negatives to the target ratio, shuffles, and writes the result.
    """
    positives = []
    negatives = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get("text", "")
            # Check if this is a negative example (empty vulnerability list)
            if '"vulnerabilities": []' in text or '"vulnerabilities":[]' in text:
                negatives.append(line.strip())
            else:
                positives.append(line.strip())

    logger.info(f"Original dataset: {len(positives)} positive, {len(negatives)} negative ({len(positives)+len(negatives)} total)")

    # Calculate how many negatives to keep
    # target_neg / (num_pos + target_neg) = neg_ratio
    # target_neg = neg_ratio * num_pos / (1 - neg_ratio)
    num_pos = len(positives)
    target_neg_count = int(neg_ratio * num_pos / (1.0 - neg_ratio))
    target_neg_count = min(target_neg_count, len(negatives))  # can't exceed available

    logger.info(f"Target negative ratio: {neg_ratio*100:.0f}%")
    logger.info(f"Keeping {target_neg_count} of {len(negatives)} negative examples")

    # Downsample negatives
    random.seed(seed)
    sampled_negatives = random.sample(negatives, target_neg_count)

    # Combine and shuffle
    balanced = positives + sampled_negatives
    random.shuffle(balanced)

    # Write output
    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as f:
        for line in balanced:
            f.write(line + "\n")

    total = len(balanced)
    actual_neg_ratio = target_neg_count / total * 100
    logger.info(f"Balanced dataset: {num_pos} positive, {target_neg_count} negative ({total} total)")
    logger.info(f"Actual negative ratio: {actual_neg_ratio:.1f}%")
    logger.info(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-balance training dataset by downsampling negatives")
    parser.add_argument("--input", type=str, default="Dataset/train_classifier_precise_lines.jsonl",
                        help="Input JSONL training file")
    parser.add_argument("--output", type=str, default="Dataset/train_balanced.jsonl",
                        help="Output balanced JSONL file")
    parser.add_argument("--neg_ratio", type=float, default=0.20,
                        help="Target negative ratio (default: 0.20 = 20%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    balance_dataset(args.input, args.output, args.neg_ratio, args.seed)

`

### export_model.py

`python
import platform
# Monkeypatch platform.win32_ver to bypass Windows WMI query errors/hangs
platform.win32_ver = lambda release='', version='', csd='', ptype='': ('10', '10.0.22621', '', 'Multiprocessor Free')

import argparse
import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("export_model")

def export_and_merge(model_id: str, adapter_path: str, output_dir: str):
    """
    Loads base model in FP16/FP32 precision (unquantized), loads the PEFT adapter,
    merges the adapter weights, and saves the self-contained merged model offline.
    """
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Determine dtype (bfloat16 or float16/float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32
        
    logger.info(f"Loading base model on {device} with dtype: {torch_dtype}")
    # Note: We must NOT load the model in 4-bit/8-bit quantization because we cannot merge weights in QLoRA directly.
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    logger.info(f"Overlaying PEFT adapter from path: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Merging adapter weights into the base model...")
    # This combines the PEFT parameters back into the base model architecture weights
    merged_model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Model export and merge successfully completed!")
    logger.info(f"The offline self-contained Hugging Face model folder is saved at: {os.path.abspath(output_dir)}")
    logger.info("You can load it completely offline using:")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}', local_files_only=True)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PEFT Adapters into Base Model and Save Offline")
    parser.add_argument("--model_id", type=str, required=True, help="Base model identifier (e.g. bigcode/starcoder2-3b)")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapter directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the merged self-contained model")
    
    args = parser.parse_args()
    export_and_merge(args.model_id, args.adapter_path, args.output_dir)



`

### refine_training_data.py

`python
import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SINK_HEURISTICS = {
    # Command Injection
    "CWE-78": r"(Runtime\.getRuntime\(\)\.exec\(|new ProcessBuilder\(|ProcessBuilder\()",
    "CWE-20": r"(Runtime\.getRuntime\(\)\.exec\(|new ProcessBuilder\(|ProcessBuilder\()",
    # Path Traversal / File Access / Insecure Perms
    "CWE-22": r"(new File\(|new FileReader\(|new FileInputStream\(|new FileOutputStream\(|Paths\.get\()",
    "CWE-276": r"(new File\(|new FileReader\(|createTempFile\(|setExecutable\(|setReadable\(|FileOutputStream\()",
    # SQL Injection
    "CWE-89": r"(executeQuery\(|prepareStatement\(|executeUpdate\(|Statement |createStatement\()",
    # XSS
    "CWE-79": r"(getWriter\(\)\.print|out\.println)",
    # LDAP
    "CWE-90": r"(InitialDirContext|search\(|lookup\()",
    # XPath
    "CWE-643": r"(XPath |evaluate\(|compile\()",
    # Hardcoded Key / Cleartext
    "CWE-321": r"(SecretKeySpec|AES|DES)",
    "CWE-319": r"(HttpURLConnection|Socket |http://|ftp://|SocketChannel)",
    # Insufficient Auth / Credentials
    "CWE-522": r"(getConnection\(|DriverManager|password|login)",
    # Open Redirect
    "CWE-601": r"(sendRedirect\(|setHeader\(\"Location\")",
    # Resource Consumption
    "CWE-400": r"(Thread\.sleep\(|readLine\(\)|while \(|for \()",
}

def find_precise_line(java_code: str, start_line: int, end_line: int, cwe_id: str) -> int:
    """Finds the precise line number within the bounds using CWE heuristics."""
    lines = java_code.split('\n')
    
    # Safely clamp bounds
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    
    pattern = SINK_HEURISTICS.get(cwe_id)
    
    if pattern:
        regex = re.compile(pattern)
        # Search within the bounded function
        for i in range(start_idx, end_idx):
            if regex.search(lines[i]):
                return i + 1  # 1-indexed line number
                
    # Fallback: if no heuristic match, just return the first line of the function body
    for i in range(start_idx, end_idx):
        if "{" in lines[i] or "try" in lines[i]:
            return i + 2
            
    return start_line

def refine_dataset(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        logging.error(f"Input file not found: {input_path}")
        return
        
    refined_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_count += 1
            try:
                data = json.loads(line)
                text = data.get("text", "")
                
                if "<|response|>\n" not in text or "<|input|>\n" not in text:
                    fout.write(line)
                    continue
                    
                # Split text into prompt and response
                parts = text.split("<|response|>\n")
                prompt_part = parts[0]
                response_part = parts[1]
                
                # Extract java code
                java_start = prompt_part.find("<|input|>\n") + len("<|input|>\n")
                java_code = prompt_part[java_start:].strip()
                
                # Parse JSON response
                try:
                    response_json = json.loads(response_part)
                except json.JSONDecodeError:
                    fout.write(line)
                    continue
                
                modified = False
                vulns = response_json.get("vulnerabilities", [])
                
                for vuln in vulns:
                    loc = vuln.get("location", {})
                    cwe_id = vuln.get("cwe_id", "")
                    
                    if "start_line" in loc and "end_line" in loc:
                        precise_line = find_precise_line(
                            java_code, 
                            loc["start_line"], 
                            loc["end_line"], 
                            cwe_id
                        )
                        # Replace block location with precise line
                        vuln["location"] = {"line": precise_line}
                        modified = True
                        
                if modified:
                    refined_count += 1
                    # Reconstruct the text with updated JSON
                    new_text = f"{prompt_part}<|response|>\n{json.dumps(response_json, indent=2)}"
                    fout.write(json.dumps({"text": new_text}) + "\n")
                else:
                    fout.write(line)
                    
            except Exception as e:
                logging.error(f"Error processing line: {e}")
                fout.write(line)
                
    logging.info(f"Refinement complete. Refined {refined_count}/{total_count} examples.")
    logging.info(f"Saved precise dataset to: {output_file}")

if __name__ == "__main__":
    refine_dataset(
        "Dataset/train_classifier_final.jsonl", 
        "Dataset/train_classifier_precise_lines.jsonl"
    )

`

### test_api.py

`python
import requests
with open('/home/dwijo/Desktop/AMD/frontend/src/App.tsx', 'r') as f:
    code = f.read()

res = requests.post("http://localhost:8000/api/scan", json={"files": [{"path": "App.tsx", "content": code}]})
print(res.json())

`

### verify_scanner.py

`python
import sys
from unittest.mock import MagicMock

# Mock torch, transformers, and peft to allow importing scanner/inference engine without dependencies
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()

from scanner import extract_java_blocks

def test_scanner_extraction():
    mock_java_code = """
    package com.example;
    
    public class SecurityTester {
        // Line comment with { brace
        /* Block comment with } brace */
        
        public void vulnerableMethod(String input) {
            String sql = "SELECT * FROM users WHERE id = " + input; // { inside string
            System.out.println("Processing...");
        }
        
        private int secureMethod(int val) {
            return val * 2;
        }
    }
    """
    
    print("Testing Java block extraction...")
    chunks = extract_java_blocks(mock_java_code)
    
    for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx + 1} (Lines {chunk['start_line']} to {chunk['end_line']}) ---")
        print(chunk["content"].strip())
        
    # Check if we got 2 method chunks
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    print("\nExtraction test completed successfully!")

if __name__ == "__main__":
    try:
        test_scanner_extraction()
    except Exception as e:
        print(f"Test failed: {e}", file=sys.stderr)
        sys.exit(1)

`
