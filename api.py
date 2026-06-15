from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import logging
import os
import sys

# Add backend directory to sys.path if not present to ensure inference_engine imports correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference_engine import VulnerabilityInferenceEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(title="CodeElixir.AI Backend API")

# Initialize the engine at startup (lazy load can save boot time, but we'll do global initialization per standard)
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
    # We allow the app to start, but subsequent calls will fail if engine is not loaded
    engine = None


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


@app.get("/api/health")
def health_check():
    return {"status": "healthy", "engine_loaded": engine is not None}


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
    
    for file_path in java_files:
        logger.info(f"Scanning file: {file_path}")
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
            
            # Map findings
            vulns = report.get("vulnerabilities", [])
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
                
                all_vulnerabilities.append(vuln)
                
        except Exception as e:
            logger.error(f"Failed to scan file {file_path}: {e}")
            
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
    
    for file_obj in request.files:
        logger.info(f"Scanning provided content for: {file_obj.path}")
        try:
            file_line_count = len(file_obj.content.splitlines())
            report = engine.analyze_file_content(
                file_obj.content,
                max_new_tokens=1024,
                file_line_count=file_line_count,
                file_path=file_obj.path
            )
            
            vulns = report.get("vulnerabilities", [])
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
                all_vulnerabilities.append(vuln)
                
        except Exception as e:
            logger.error(f"Failed to scan content for {file_obj.path}: {e}")
            
    return {"vulnerabilities": all_vulnerabilities}

