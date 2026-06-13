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
    max_tokens: int = 1024
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
                result = engine.analyze_file_content(raw_code, max_new_tokens=max_tokens)

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
    args = parser.parse_args()

    run_codebase_scan(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        target_dir=args.target_dir,
        output_report=args.output_report,
        use_chunks=args.use_chunks,
        load_in_4bit=not args.no_quant,
        max_tokens=args.max_tokens
    )
