from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
import os
import sys
import concurrent.futures

# Add backend directory to sys.path if not present to ensure inference_engine imports correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_engine import VulnerabilityInferenceEngine
from fix_engine import FixInferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CodeElixir.AI Backend API")

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

# Initialize the engine at startup (lazy load can save boot time, but we'll do global initialization per standard)
logger.info("Initializing VulnerabilityInferenceEngine...")
try:
    engine = VulnerabilityInferenceEngine(
        model_id="deepseek-ai/deepseek-coder-6.7b-base",
        adapter_path=None,  # Disabled adapters to match raw base model console output
        load_in_4bit=False  # No quant as requested
    )
    logger.info("Engine loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load engine: {e}")
    # We allow the app to start, but subsequent calls will fail if engine is not loaded
    engine = None

logger.info("Initializing FixInferenceEngine...")
try:
    fix_engine = FixInferenceEngine(
        model_id="Qwen/Qwen3-Coder-Next",
        adapter_path=None,  # Disabled adapters for fixed inference
        load_in_4bit=False
    )
    logger.info("FixEngine loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load fix engine: {e}")
    fix_engine = None


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
