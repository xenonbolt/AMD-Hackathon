import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from inference_engine import VulnerabilityInferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("scanner")


def extract_java_blocks(code: str) -> List[Dict[str, Any]]:
    """
    Extracts logical blocks (primarily class methods) from a Java source file
    using a character-by-character state machine to handle strings, comments, and braces.
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
    
    # Pre-calculate line start offsets to map character indices to line numbers quickly
    line_starts = [0]
    for idx, char in enumerate(code):
        if char == '\n':
            line_starts.append(idx + 1)
            
    def get_line_num(char_idx: int) -> int:
        # Linear scan since the number of lines is usually small
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
                # Trim leading whitespaces and delimiters
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
                    
                    # Store block if it is method level (level 2)
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
        # Fallback will trigger if chunks is empty

    # Fallback to returning the entire file if no methods were parsed
    if not chunks:
        chunks.append({
            "start_line": 1,
            "end_line": len(line_starts),
            "content": code
        })
        
    return chunks


def chunk_sliding_window(text: str, start_line: int, window_size: int = 60, overlap: int = 15) -> List[Dict[str, Any]]:
    """
    Chunks a block of code into smaller overlapping windows. 
    Used as a fallback for large method blocks to prevent out-of-context errors.
    """
    lines = text.splitlines()
    chunks = []
    n = len(lines)
    i = 0
    while i < n:
        end = min(i + window_size, n)
        chunk_content = "\n".join(lines[i:end])
        chunks.append({
            "start_line": start_line + i,
            "end_line": start_line + end - 1,
            "content": chunk_content
        })
        if end == n:
            break
        i += (window_size - overlap)
    return chunks


def run_codebase_scan(
    model_id: str,
    adapter_path: str,
    target_dir: str,
    output_report: str,
    max_chunk_lines: int = 100
) -> None:
    """
    Walks target codebase, extracts chunks, runs them through the inference engine, 
    and saves a structured JSON vulnerability report.
    """
    target_path = Path(target_dir)
    if not target_path.exists():
        logger.error(f"Target scan directory does not exist: {target_path}")
        return

    try:
        # Initialize inference engine once
        logger.info(f"Loading scanner model context (Model: {model_id}, Adapter: {adapter_path})...")
        engine = VulnerabilityInferenceEngine(
            model_id=model_id,
            adapter_path=adapter_path,
            load_in_4bit=True
        )
    except Exception as e:
        logger.error(f"Failed to load model context: {e}", exc_info=True)
        return

    findings: List[Dict[str, Any]] = []

    # Recurse and locate all Java source files
    logger.info(f"Scanning directory: {target_path} for .java files")
    java_files = list(target_path.rglob("*.java"))
    logger.info(f"Found {len(java_files)} Java files to scan.")

    for file_idx, file_path in enumerate(java_files):
        logger.info(f"[{file_idx + 1}/{len(java_files)}] Scanning: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                code_content = f.read()

            # Extract methods/blocks
            raw_blocks = extract_java_blocks(code_content)
            
            # Post-process blocks: split very large methods
            processed_chunks: List[Dict[str, Any]] = []
            for block in raw_blocks:
                num_lines = block["end_line"] - block["start_line"] + 1
                if num_lines > max_chunk_lines:
                    # Apply sliding window chunking to avoid context blowout
                    sub_chunks = chunk_sliding_window(
                        block["content"], 
                        start_line=block["start_line"], 
                        window_size=max_chunk_lines, 
                        overlap=20
                    )
                    processed_chunks.extend(sub_chunks)
                else:
                    processed_chunks.append(block)

            logger.info(f"Extracted {len(processed_chunks)} chunks from {file_path.name}")

            # Analyze each chunk
            for chunk in processed_chunks:
                original_code = chunk["content"]
                result = engine.analyze_snippet(original_code)
                
                # Attempt to parse structured JSON vulnerabilities from result
                vulnerabilities = []
                result_clean = result.strip()
                try:
                    data = json.loads(result_clean)
                    if isinstance(data, dict) and "vulnerabilities" in data:
                        vulnerabilities = data["vulnerabilities"]
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown or raw regex match
                    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```", r"(\{.*\})"]:
                        match = re.search(pattern, result_clean, re.DOTALL)
                        if match:
                            try:
                                data = json.loads(match.group(1).strip())
                                if isinstance(data, dict) and "vulnerabilities" in data:
                                    vulnerabilities = data["vulnerabilities"]
                                    break
                            except json.JSONDecodeError:
                                pass

                # If vulnerabilities are parsed, extract them
                if vulnerabilities:
                    logger.warning(
                        f"SUSPECTED VULNERABILITIES flagged in {file_path.name} "
                        f"(Lines {chunk['start_line']}-{chunk['end_line']})"
                    )
                    for vuln in vulnerabilities:
                        # Extract location info relative to chunk start
                        loc = vuln.get("location", {})
                        # Location lines are 1-based relative to block. Absolute lines:
                        # Handle missing or invalid location details gracefully
                        try:
                            start_offset = int(loc.get("start_line", 1)) - 1
                            end_offset = int(loc.get("end_line", 1)) - 1
                        except (ValueError, TypeError):
                            start_offset = 0
                            end_offset = chunk["end_line"] - chunk["start_line"]
                            
                        vuln_start = chunk["start_line"] + start_offset
                        vuln_end = chunk["start_line"] + end_offset

                        findings.append({
                            "file_path": str(file_path.relative_to(target_path.parent)),
                            "start_line": vuln_start,
                            "end_line": vuln_end,
                            "cwe_id": vuln.get("cwe_id", ""),
                            "cwe_name": vuln.get("cwe_name", ""),
                            "severity": vuln.get("severity", ""),
                            "confidence": vuln.get("confidence", 1.0),
                            "description": vuln.get("description", ""),
                            "impact": vuln.get("impact", ""),
                            "recommendation": vuln.get("recommendation", ""),
                            "original_code": original_code
                        })
                else:
                    # Fallback heuristic: check if output suggests any vulnerability keywords
                    # or is not just code matching original or indicating "no vulnerability"
                    clean_orig = "".join(original_code.split())
                    clean_res = "".join(result.split())
                    vulnerability_found = False
                    
                    if clean_orig != clean_res and result:
                        lower_res = result.lower()
                        if not any(phrase in lower_res for phrase in ["no vulnerability", "secure", "safe code", "no issues"]):
                            vulnerability_found = True
                    
                    if vulnerability_found:
                        findings.append({
                            "file_path": str(file_path.relative_to(target_path.parent)),
                            "start_line": chunk["start_line"],
                            "end_line": chunk["end_line"],
                            "suspected_vulnerability": "Flagged by fallback heuristic",
                            "original_code": original_code,
                            "suggested_remediation": result
                        })

        except Exception as file_err:
            logger.error(f"Failed to scan file {file_path}: {file_err}", exc_info=True)

    # Save structured output report
    report_path = Path(output_report)
    try:
        report_data = {
            "target_directory": str(target_path.resolve()),
            "total_files_scanned": len(java_files),
            "vulnerabilities_count": len(findings),
            "findings": findings
        }
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(report_data, rf, indent=4)
        logger.info(f"Structured JSON report successfully saved to: {report_path.resolve()}")
    except Exception as report_err:
        logger.error(f"Failed to write report to {report_path}: {report_err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Codebase Vulnerability Scanner")
    parser.add_argument("--model_id", type=str, required=True, help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter weights")
    parser.add_argument("--target_dir", type=str, required=True, help="Path to directory containing Java source files")
    parser.add_argument("--output_report", type=str, default="vulnerability_report.json", help="Path for JSON output report")
    parser.add_argument("--max_chunk_lines", type=int, default=100, help="Maximum lines of code per chunk")
    args = parser.parse_args()

    run_codebase_scan(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        target_dir=args.target_dir,
        output_report=args.output_report,
        max_chunk_lines=args.max_chunk_lines
    )
