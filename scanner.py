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

CWE_METADATA = {
    "CWE-22": {"name": "Path Traversal (CWE-22)", "severity": "high", "keywords": ["path traversal", "directory traversal"]},
    "CWE-78": {"name": "OS Command Injection (CWE-78)", "severity": "critical", "keywords": ["command injection", "os command", "runtime.exec"]},
    "CWE-79": {"name": "Cross-site Scripting (CWE-79)", "severity": "medium", "keywords": ["xss", "cross-site scripting", "cross site scripting"]},
    "CWE-89": {"name": "SQL Injection (CWE-89)", "severity": "critical", "keywords": ["sql injection", "sqli"]},
    "CWE-90": {"name": "LDAP Injection (CWE-90)", "severity": "critical", "keywords": ["ldap injection", "ldap"]},
    "CWE-276": {"name": "Temporary File Creation With Insecure Perms (CWE-276)", "severity": "high", "keywords": ["insecure perms", "temporary file", "file creation"]},
    "CWE-319": {"name": "Cleartext Transmission of Sensitive Information (CWE-319)", "severity": "high", "keywords": ["cleartext transmission", "unencrypted"]},
    "CWE-321": {"name": "Use of Hard-coded Cryptographic Key (CWE-321)", "severity": "critical", "keywords": ["hard-coded cryptographic key", "hardcoded key"]},
    "CWE-400": {"name": "Resource Exhaustion (CWE-400)", "severity": "high", "keywords": ["resource exhaustion", "denial of service"]},
    "CWE-522": {"name": "Hard Coded Password (CWE-522)", "severity": "high", "keywords": ["hard coded password", "hardcoded password"]},
    "CWE-601": {"name": "URL Redirection to Untrusted Site (CWE-601)", "severity": "medium", "keywords": ["url redirection", "open redirect", "untrusted site"]},
    "CWE-643": {"name": "XPath Injection (CWE-643)", "severity": "high", "keywords": ["xpath injection"]},
    "CWE-918": {"name": "Server-Side Request Forgery (CWE-918)", "severity": "critical", "keywords": ["ssrf", "server-side request forgery"]},
}


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
                json_parsed = False
                result_clean = result.strip()
                try:
                    data = json.loads(result_clean)
                    if isinstance(data, dict) and "vulnerabilities" in data:
                        vulnerabilities = data["vulnerabilities"]
                        json_parsed = True
                except json.JSONDecodeError:
                    # Try to extract JSON only from explicit markdown code fences.
                    # NOTE: do NOT use a greedy r"({.*})" pattern here — Java code
                    # is full of braces and would be mistakenly parsed as JSON,
                    # setting json_parsed=True with empty vulnerabilities.
                    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
                        match = re.search(pattern, result_clean, re.DOTALL)
                        if match:
                            try:
                                data = json.loads(match.group(1).strip())
                                if isinstance(data, dict) and "vulnerabilities" in data:
                                    vulnerabilities = data["vulnerabilities"]
                                    json_parsed = True
                                    break
                            except json.JSONDecodeError:
                                pass

                # If JSON was parsed, process vulnerabilities (if any)
                if json_parsed:
                    if vulnerabilities:
                        logger.warning(
                            f"SUSPECTED VULNERABILITIES flagged in {file_path.name} "
                            f"(Lines {chunk['start_line']}-{chunk['end_line']})"
                        )
                        for vuln in vulnerabilities:
                            # Extract location info relative to chunk start
                            loc = vuln.get("location", {})
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
                                "original_code": original_code
                            })
                else:
                    # -------------------------------------------------------
                    # Fallback heuristic — the model was trained on vuln→fix
                    # pairs and outputs REMEDIATED CODE as its response.
                    # A vulnerability is flagged when the model meaningfully
                    # rewrites the code (Jaccard token distance > 15%).
                    # Simple boilerplate (constructors, DTOs, stubs) is echoed
                    # back unchanged (ratio ≈00) and is correctly skipped.
                    # Remediation signals are used only for CWE identification.
                    # -------------------------------------------------------
                    REMEDIATION_SIGNALS = [
                        ("preparedstatement",    "CWE-89"),
                        ("setstring(",            "CWE-89"),
                        ("canonicalpath",         "CWE-22"),
                        ("getcanonicalpath",      "CWE-22"),
                        ("normalize(",            "CWE-22"),
                        ("processbuilder",        "CWE-78"),
                        ("allowedurls",           "CWE-918"),
                        ("whitelist",             "CWE-918"),
                        ("allowlist",             "CWE-918"),
                        ("isvalidurl",            "CWE-918"),
                        ("escapehtml",            "CWE-79"),
                        ("stringescapeutils",     "CWE-79"),
                        ("system.getenv",         "CWE-522"),
                        ("system.getproperty",    "CWE-522"),
                        ("isvalid",               "CWE-601"),
                        ("isallowed",             "CWE-601"),
                        ("https",                 "CWE-319"),
                        ("ssl",                   "CWE-319"),
                    ]

                    vulnerability_found = False
                    detected_cwe_id = None

                    if result and result.strip():
                        orig_tokens = set(original_code.lower().split())
                        result_tokens = set(result.lower().split())

                        union_size = len(orig_tokens | result_tokens)
                        intersection_size = len(orig_tokens & result_tokens)
                        change_ratio = (
                            1.0 - (intersection_size / union_size)
                            if union_size > 0 else 0.0
                        )

                        logger.debug(
                            f"Fallback token change_ratio={change_ratio:.3f} for "
                            f"{file_path.name} lines "
                            f"{chunk['start_line']}-{chunk['end_line']}"
                        )

                        # Primary gate: model must have substantially rewritten the code
                        if change_ratio > 0.15:
                            vulnerability_found = True
                            # Secondary: try to identify CWE from what the model added
                            result_lower = result.lower()
                            orig_lower = original_code.lower()
                            for pattern, cwe in REMEDIATION_SIGNALS:
                                if pattern in result_lower and pattern not in orig_lower:
                                    detected_cwe_id = cwe
                                    break

                    if vulnerability_found:
                        cwe_id = detected_cwe_id or "Unknown"
                        cwe_info = CWE_METADATA.get(
                            cwe_id, {"name": "Unknown", "severity": "Unknown"}
                        )
                        findings.append({
                            "file_path": str(file_path.relative_to(target_path.parent)),
                            "start_line": chunk["start_line"],
                            "end_line": chunk["end_line"],
                            "cwe_id": cwe_id,
                            "cwe_name": cwe_info["name"],
                            "severity": cwe_info["severity"],
                            "confidence": 0.55,
                            "description": (
                                f"Fallback heuristic: model rewrote "
                                f"{change_ratio:.0%} of tokens "
                                f"(fix pattern detected: {cwe_id})."
                            ),
                            "original_code": original_code
                        })
                    else:
                        logger.debug(
                            f"Fallback heuristic: change ratio too low for "
                            f"{file_path.name} lines "
                            f"{chunk['start_line']}-{chunk['end_line']}. Skipping."
                        )

        except Exception as file_err:
            logger.error(f"Failed to scan file {file_path}: {file_err}", exc_info=True)

    # -------------------------------------------------------------------------
    # Pass 2: Verify each suspected finding to eliminate false positives
    # and enrich confirmed findings with CWE metadata.
    # -------------------------------------------------------------------------
    total_before_verification = len(findings)
    logger.info(
        f"Starting second-pass verification on {total_before_verification} suspected finding(s)..."
    )

    verified_findings: List[Dict[str, Any]] = []
    false_positives: List[Dict[str, Any]] = []

    for finding_idx, finding in enumerate(findings):
        logger.info(
            f"[Verifier {finding_idx + 1}/{total_before_verification}] "
            f"Verifying: {finding['file_path']} "
            f"Lines {finding['start_line']}-{finding['end_line']}"
        )
        try:
            verification = engine.verify_finding(
                code=finding["original_code"],
                initial_description=finding.get("description", "")
            )
        except Exception as verify_err:
            logger.error(f"Verifier call failed for finding {finding_idx + 1}: {verify_err}")
            # Conservative: keep the finding on verifier error
            verification = {
                "is_vulnerable": True,
                "cwe_id": None,
                "cwe_name": None,
                "severity": None,
                "confidence": 0.4,
                "reason": "Verifier call failed; finding preserved conservatively."
            }

        if verification["is_vulnerable"]:
            # --- True positive: enrich with verified CWE metadata ---
            # Prefer verifier-supplied values; fall back to first-pass values if absent.
            verified_cwe_id = verification["cwe_id"] or finding.get("cwe_id") or "Unknown"
            verified_cwe_name = verification["cwe_name"] or finding.get("cwe_name") or ""
            verified_severity = verification["severity"] or finding.get("severity") or ""

            # If cwe_id still unknown, attempt CWE_METADATA lookup by keyword in reason/description
            if verified_cwe_id == "Unknown" or not verified_cwe_name:
                combined_text = (
                    (verification.get("reason") or "") + " " +
                    (verified_cwe_name or "") + " " +
                    (finding.get("description") or "")
                ).lower()
                for mapped_cwe, mapped_info in CWE_METADATA.items():
                    if any(kw in combined_text for kw in mapped_info.get("keywords", [])):
                        if verified_cwe_id == "Unknown":
                            verified_cwe_id = mapped_cwe
                        if not verified_cwe_name:
                            verified_cwe_name = mapped_info["name"]
                        if not verified_severity:
                            verified_severity = mapped_info["severity"]
                        break

            enriched = dict(finding)
            enriched.update({
                "cwe_id": verified_cwe_id,
                "cwe_name": verified_cwe_name,
                "severity": verified_severity,
                "confidence": verification["confidence"],
                "verifier_reason": verification["reason"],
            })
            verified_findings.append(enriched)
            logger.warning(
                f"  \u2714 CONFIRMED TRUE POSITIVE \u2014 "
                f"{verified_cwe_id} ({verified_severity}) in "
                f"{finding['file_path']}:{finding['start_line']}"
            )
        else:
            # --- False positive: move to separate bucket for auditability ---
            fp_entry = dict(finding)
            fp_entry["verifier_reason"] = verification["reason"]
            false_positives.append(fp_entry)
            logger.info(
                f"  \u2718 FALSE POSITIVE discarded \u2014 "
                f"{finding['file_path']}:{finding['start_line']} "
                f"Reason: {verification['reason']}"
            )

    logger.info(
        f"Verification complete: {len(verified_findings)} true positive(s), "
        f"{len(false_positives)} false positive(s) discarded."
    )

    # Save structured output report
    report_path = Path(output_report)
    try:
        # Strip internal-only fields before writing — original_code is used by the
        # verifier but should not appear in the final report output.
        def _strip_internal(findings_list):
            return [
                {k: v for k, v in f.items() if k != "original_code"}
                for f in findings_list
            ]

        report_data = {
            "target_directory": str(target_path.resolve()),
            "total_files_scanned": len(java_files),
            "total_suspected_before_verification": total_before_verification,
            "vulnerabilities_count": len(verified_findings),
            "false_positives_discarded": len(false_positives),
            "findings": _strip_internal(verified_findings),
            "false_positives": _strip_internal(false_positives)
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
