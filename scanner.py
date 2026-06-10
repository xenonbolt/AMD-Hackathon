import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from inference_engine import VulnerabilityInferenceEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("scanner")

# Attempt to import RAG pipeline; gracefully degrade if unavailable
try:
    from rag_pipeline import JavaVulnRAGPipeline
    _RAG_AVAILABLE = True
except ImportError:
    _RAG_AVAILABLE = False
    logger.warning(
        "rag_pipeline module not found — CVE/CWE enrichment will be skipped. "
        "Run: pip install requests"
    )


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
    max_chunk_lines: int = 100,
    use_rag: bool = True,
    nvd_api_key: Optional[str] = None,
    force_rag_refresh: bool = False,
) -> None:
    """
    Walks target codebase, extracts chunks, runs them through the inference engine,
    and saves a structured JSON vulnerability report enriched with CVE/CWE
    metadata and severity from the RAG pipeline.
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

    # Initialize RAG pipeline for CVE/CWE enrichment
    rag_pipeline = None
    if use_rag and _RAG_AVAILABLE:
        try:
            logger.info("Initializing RAG pipeline for CVE/CWE enrichment …")
            rag_pipeline = JavaVulnRAGPipeline(
                nvd_api_key=nvd_api_key,
                force_refresh=force_rag_refresh,
            )
            logger.info("RAG pipeline ready.")
        except Exception as rag_err:
            logger.warning(f"RAG pipeline failed to initialize: {rag_err}. Continuing without enrichment.")
    elif use_rag and not _RAG_AVAILABLE:
        logger.warning("RAG requested but rag_pipeline module is unavailable.")

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

            # Since the model (StarCoder2) supports 16k+ context length and was trained on 
            # full file contents (including class/package headers), we should pass the 
            # entire file rather than destructively chunking it and breaking Java syntax.
            num_lines = len(code_content.splitlines())
            
            processed_chunks = []
            if num_lines > 2000:
                # Only use sliding window for outrageously huge files to avoid memory blowout
                processed_chunks = chunk_sliding_window(
                    code_content, 
                    start_line=1, 
                    window_size=2000, 
                    overlap=100
                )
                logger.info(f"File {file_path.name} is very large ({num_lines} lines). Split into {len(processed_chunks)} chunks.")
            else:
                processed_chunks.append({
                    "start_line": 1,
                    "end_line": num_lines,
                    "content": code_content
                })
                logger.info(f"Analyzing entire file: {file_path.name} ({num_lines} lines)")

            # Analyze each chunk
            for chunk in processed_chunks:
                original_code = chunk["content"]

                # ── Step 1: Query RAG BEFORE inference ──────────────────────────
                # Build context from the raw code snippet so the model gets
                # grounded CVE/CWE knowledge injected directly into its prompt.
                rag_context_str = ""
                rag_enrichment = {"cve_details": [], "cwe_details": [], "severity": "UNKNOWN"}
                if rag_pipeline is not None:
                    try:
                        # Use a short code excerpt as the retrieval query
                        rag_docs = rag_pipeline.query(original_code[:300], top_k=3)

                        # Format a compact context block for the prompt
                        context_lines = []
                        for doc in rag_docs:
                            if doc.get("type") == "CVE":
                                cwe_str = ", ".join(doc.get("cwe_ids", [])) or "N/A"
                                context_lines.append(
                                    f"- {doc['cve_id']} | Severity: {doc['severity']} "
                                    f"(CVSS {doc.get('cvss_score', '?')}) | CWE: {cwe_str} | "
                                    f"{doc['description'][:150]}"
                                )
                            elif doc.get("type") == "CWE":
                                context_lines.append(
                                    f"- {doc['cwe_id']}: {doc.get('name', '')} — "
                                    f"{doc.get('description', '')[:150]}"
                                )
                        rag_context_str = "\n".join(context_lines)

                        # Also pre-build the report enrichment from the same docs
                        # (avoids a second RAG query after inference)
                        cve_details = []
                        cwe_details = []
                        top_severity = "UNKNOWN"
                        severity_rank = {"CRITICAL": 5, "HIGH": 4, "MEDIUM": 3, "LOW": 2, "UNKNOWN": 1}
                        for doc in rag_docs:
                            if doc.get("type") == "CVE":
                                cve_details.append({
                                    "cve_id": doc.get("cve_id", ""),
                                    "description": doc.get("description", "")[:300],
                                    "cvss_score": doc.get("cvss_score"),
                                    "severity": doc.get("severity", "UNKNOWN"),
                                    "cwe_ids": doc.get("cwe_ids", []),
                                    "published": doc.get("published", "")[:10],
                                })
                                sev = doc.get("severity", "UNKNOWN").upper()
                                if severity_rank.get(sev, 0) > severity_rank.get(top_severity, 0):
                                    top_severity = sev
                            elif doc.get("type") == "CWE":
                                cwe_details.append({
                                    "cwe_id": doc.get("cwe_id", ""),
                                    "name": doc.get("name", ""),
                                    "description": doc.get("description", "")[:300],
                                    "url": doc.get("url", ""),
                                })
                        rag_enrichment = {
                            "cve_details": cve_details,
                            "cwe_details": cwe_details,
                            "severity": top_severity,
                        }
                    except Exception as rag_err:
                        logger.warning(f"RAG pre-inference query failed: {rag_err}")

                # ── Step 2: Run inference WITH RAG context in prompt ─────────────
                result_dict = engine.analyze_snippet(original_code, rag_context=rag_context_str)

                # ── Step 3: Handle structured JSON output ────────────────────────
                if result_dict.get("parse_error"):
                    logger.warning(
                        f"Failed to parse JSON response for {file_path.name} "
                        f"(Lines {chunk['start_line']}-{chunk['end_line']}). Raw output:\n"
                        f"{result_dict.get('raw', '')}"
                    )
                    continue

                vulnerabilities = result_dict.get("vulnerabilities", [])
                
                # If the vulnerabilities list is not empty, vulnerabilities were found
                if vulnerabilities:
                    logger.warning(
                        f"SUSPECTED VULNERABILITY flagged in {file_path.name} "
                        f"(Lines {chunk['start_line']}-{chunk['end_line']})"
                    )

                    findings.append({
                        "file_path": str(file_path.relative_to(target_path.parent)),
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "suspected_vulnerabilities": vulnerabilities,
                        "rag_severity": rag_enrichment["severity"],
                        "original_code": original_code,
                        "cve_details": rag_enrichment["cve_details"],
                        "cwe_details": rag_enrichment["cwe_details"],
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
    parser.add_argument("--no_rag", action="store_true", help="Disable RAG-based CVE/CWE enrichment")
    parser.add_argument("--nvd_api_key", type=str, default=None, help="Optional NVD API key for higher rate limits")
    parser.add_argument("--rag_refresh", action="store_true", help="Force refresh of RAG CVE/CWE caches")
    args = parser.parse_args()

    run_codebase_scan(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        target_dir=args.target_dir,
        output_report=args.output_report,
        max_chunk_lines=args.max_chunk_lines,
        use_rag=not args.no_rag,
        nvd_api_key=args.nvd_api_key,
        force_rag_refresh=args.rag_refresh,
    )
