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
            model_id="deepseek-ai/deepseek-coder-6.7b-base",
            adapter_path="./adapters",
            load_in_4bit=False  # No quant as requested
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
        "models_loaded": ["Qwen/Qwen2.5-Coder-7B-Instruct", "adapters_fix"],
        "api_health": "Online (FastAPI / 0.0.0.0:8000)"
    }
