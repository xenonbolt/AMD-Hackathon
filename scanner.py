"""
scanner.py
===========
Static analysis utility that walks a local Java codebase, extracts logical
code blocks, passes them through the inference engine, and emits a
structured JSON vulnerability report.

Responsibilities
----------------
1. Recursively find all ``.java`` files under a target directory.
2. Chunk each file into logical code blocks using a two-tier extraction
   strategy:
     a) **AST-inspired regex extraction** – finds individual method
        declarations and their full bodies using brace-balanced parsing.
     b) **Sliding-window fallback** – used for files where method extraction
        yields nothing (e.g., interface-only files or enums).
3. Deduplicate chunks using content hashing to avoid redundant LLM calls.
4. Submit each unique chunk to the inference engine.
5. Aggregate findings into a structured ``ScanReport`` and serialise to JSON.

Output report schema
--------------------
{
  "scan_metadata": {
    "target_directory": "<path>",
    "scanned_at": "<ISO-8601 timestamp>",
    "total_files": N,
    "total_chunks": N,
    "total_vulnerabilities": N,
    "engine_adapter": "<path>",
    "base_model": "<model-id>"
  },
  "findings": [
    {
      "file_path": "<rel/path/to/File.java>",
      "chunk_index": 0,
      "chunk_type": "method|sliding_window",
      "method_name": "<name or null>",
      "line_start": N,
      "line_end": N,
      "vulnerabilities": [
        {
          "cwe_id": "CWE-89",
          "cwe_name": "SQL Injection",
          "severity": "critical",
          "confidence": 0.95,
          "location": {"start_line": N, "end_line": N, "function": "<name>"},
          "description": "…",
          "impact": "…",
          "recommendation": "…"
        }
      ]
    }
  ]
}

Usage
-----
python scanner.py \\
    --target ./my-java-project \\
    --adapter ./outputs/vuln-lora \\
    --output scan_report.json \\
    --workers 1

Author : Elite AI Engineering Team
Python : 3.10+
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from inference_engine import InferenceEngine, analyze_snippet

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scanner.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("scanner")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_JAVA_METHOD_PATTERN: re.Pattern = re.compile(
    r"""
    (?:(?:public|protected|private|static|final|synchronized|abstract|native|strictfp)\s+)*
    (?:[\w<>\[\],\s]+?)\s+                     # return type
    (\w+)\s*\(                                  # method name + opening paren
    [^)]*\)                                     # parameters
    (?:\s+throws\s+[\w,\s]+)?                   # optional throws clause
    \s*\{                                       # opening brace
    """,
    re.VERBOSE | re.MULTILINE,
)

# Sliding window parameters
_WINDOW_LINES: int = 60
_WINDOW_STRIDE: int = 30
_MIN_CHUNK_LINES: int = 5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A single logical code block extracted from a Java file."""

    content: str
    chunk_type: str          # "method" | "sliding_window"
    method_name: str | None  # populated for method chunks
    line_start: int          # 1-indexed, within the source file
    line_end: int
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        self.content_hash = hashlib.sha256(
            self.content.encode("utf-8")
        ).hexdigest()


@dataclass
class ChunkFinding:
    """Vulnerability findings for a single code chunk."""

    file_path: str
    chunk_index: int
    chunk: CodeChunk
    vulnerabilities: list[dict[str, Any]]


@dataclass
class ScanReport:
    """Aggregated result of scanning an entire directory."""

    target_directory: str
    scanned_at: str
    total_files: int
    total_chunks: int
    total_vulnerabilities: int
    engine_adapter: str
    base_model: str
    findings: list[ChunkFinding]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_metadata": {
                "target_directory": self.target_directory,
                "scanned_at": self.scanned_at,
                "total_files": self.total_files,
                "total_chunks": self.total_chunks,
                "total_vulnerabilities": self.total_vulnerabilities,
                "engine_adapter": self.engine_adapter,
                "base_model": self.base_model,
            },
            "findings": [
                {
                    "file_path": f.file_path,
                    "chunk_index": f.chunk_index,
                    "chunk_type": f.chunk.chunk_type,
                    "method_name": f.chunk.method_name,
                    "line_start": f.chunk.line_start,
                    "line_end": f.chunk.line_end,
                    "vulnerabilities": f.vulnerabilities,
                }
                for f in self.findings
                if f.vulnerabilities  # Only include findings with actual vulns
            ],
        }


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_java_files(root_dir: Path) -> list[Path]:
    """
    Recursively collects all ``.java`` files under ``root_dir``.

    Parameters
    ----------
    root_dir : Path
        Root of the Java project/codebase to scan.

    Returns
    -------
    list[Path]
        Sorted list of discovered ``.java`` file paths.
    """
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Target is not a directory: {root_dir}")

    java_files = sorted(root_dir.rglob("*.java"))
    logger.info("Discovered %d .java files under %s.", len(java_files), root_dir)
    return java_files


# ---------------------------------------------------------------------------
# Code chunking
# ---------------------------------------------------------------------------

def _extract_brace_balanced_body(source: str, open_brace_pos: int) -> str:
    """
    Extracts the complete brace-balanced block starting at ``open_brace_pos``.

    Returns the extracted substring (including the surrounding braces) or an
    empty string if the braces are unbalanced.
    """
    depth = 0
    in_string: bool = False
    in_char: bool = False
    in_line_comment: bool = False
    in_block_comment: bool = False

    i = open_brace_pos
    start = i

    while i < len(source):
        ch = source[i]

        # Track single-line comments
        if not in_string and not in_char and not in_block_comment:
            if source[i : i + 2] == "//":
                in_line_comment = True
                i += 2
                continue
            if source[i : i + 2] == "/*":
                in_block_comment = True
                i += 2
                continue

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if source[i : i + 2] == "*/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        # Track string literals
        if ch == '"' and not in_char:
            in_string = not in_string
        elif ch == "'" and not in_string:
            in_char = not in_char
        elif not in_string and not in_char:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return source[start : i + 1]
        i += 1

    return ""  # Unbalanced braces


def extract_method_chunks(source: str) -> list[CodeChunk]:
    """
    Extracts individual Java method bodies from source using regex + brace
    balancing.

    Returns
    -------
    list[CodeChunk]
        One ``CodeChunk`` per discovered method.
    """
    chunks: list[CodeChunk] = []
    lines = source.splitlines(keepends=True)

    for match in _JAVA_METHOD_PATTERN.finditer(source):
        method_name: str = match.group(1)
        brace_pos: int = match.end() - 1   # Position of the opening '{'

        body: str = _extract_brace_balanced_body(source, brace_pos)
        if not body:
            continue

        # Include the method signature (everything from match.start() to body end)
        full_method_text = source[match.start() : match.start() + (brace_pos - match.start()) + len(body)]

        # Determine line numbers
        pre_source = source[: match.start()]
        line_start = pre_source.count("\n") + 1
        line_end = line_start + full_method_text.count("\n")

        if len(full_method_text.splitlines()) < _MIN_CHUNK_LINES:
            continue  # Skip trivial methods (constructors, getters, etc.)

        chunks.append(
            CodeChunk(
                content=full_method_text,
                chunk_type="method",
                method_name=method_name,
                line_start=line_start,
                line_end=line_end,
            )
        )

    return chunks


def extract_sliding_window_chunks(source: str) -> list[CodeChunk]:
    """
    Fallback chunker: splits source into overlapping fixed-size windows.

    Used when :func:`extract_method_chunks` returns nothing (e.g., interface
    files, annotation-heavy classes).

    Returns
    -------
    list[CodeChunk]
    """
    lines = source.splitlines()
    chunks: list[CodeChunk] = []

    start = 0
    while start < len(lines):
        end = min(start + _WINDOW_LINES, len(lines))
        window_lines = lines[start:end]

        if len(window_lines) < _MIN_CHUNK_LINES:
            break

        chunks.append(
            CodeChunk(
                content="\n".join(window_lines),
                chunk_type="sliding_window",
                method_name=None,
                line_start=start + 1,
                line_end=end,
            )
        )

        if end == len(lines):
            break
        start += _WINDOW_STRIDE

    return chunks


def chunk_java_file(file_path: Path) -> tuple[str, list[CodeChunk]]:
    """
    Reads a ``.java`` file and returns its chunks.

    Strategy: attempt method-level extraction first; fall back to sliding
    windows if no methods are found.

    Parameters
    ----------
    file_path : Path
        Absolute path to the ``.java`` file.

    Returns
    -------
    (source_code, list[CodeChunk])
    """
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.error("Cannot read file %s: %s", file_path, exc)
        return "", []

    method_chunks = extract_method_chunks(source)
    if method_chunks:
        logger.debug(
            "%s → %d method chunk(s) extracted.", file_path.name, len(method_chunks)
        )
        return source, method_chunks

    # Fallback
    window_chunks = extract_sliding_window_chunks(source)
    logger.debug(
        "%s → no methods found; using %d sliding window(s).",
        file_path.name,
        len(window_chunks),
    )
    return source, window_chunks


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate_chunks(
    chunks: list[CodeChunk],
    seen_hashes: set[str],
) -> list[CodeChunk]:
    """
    Filters out chunks whose content hash has already been seen globally,
    updating ``seen_hashes`` in place.
    """
    unique: list[CodeChunk] = []
    for chunk in chunks:
        if chunk.content_hash not in seen_hashes:
            seen_hashes.add(chunk.content_hash)
            unique.append(chunk)
    return unique


# ---------------------------------------------------------------------------
# Scanning orchestration
# ---------------------------------------------------------------------------

def scan_directory(
    target_dir: Path,
    adapter_path: str | Path,
    base_model_id: str = "bigcode/starcoder2-3b",
    output_path: Path | None = None,
    max_files: int | None = None,
) -> ScanReport:
    """
    Full scan pipeline: discover files → chunk → infer → aggregate → save.

    Parameters
    ----------
    target_dir : Path
        Root of the Java codebase to scan.
    adapter_path : str | Path
        Path to the LoRA adapter directory.
    base_model_id : str
        HuggingFace model ID of the base model.
    output_path : Path | None
        If provided, the report is also written to this JSON file.
    max_files : int | None
        Optional cap on the number of files to scan (useful for testing).

    Returns
    -------
    ScanReport
    """
    logger.info("=== Starting Java Security Scan ===")
    logger.info("Target directory : %s", target_dir)
    logger.info("Adapter path     : %s", adapter_path)

    # --- Initialise inference engine (loaded once) -----------------------
    engine = InferenceEngine.get_instance(
        adapter_path=adapter_path,
        base_model_id=base_model_id,
    )

    # --- Discover files --------------------------------------------------
    java_files = discover_java_files(target_dir)
    if max_files is not None:
        java_files = java_files[:max_files]
        logger.info("Capped scan to %d file(s).", max_files)

    findings: list[ChunkFinding] = []
    seen_hashes: set[str] = set()
    total_chunks_processed: int = 0
    total_vulns: int = 0

    for file_idx, java_file in enumerate(java_files, start=1):
        rel_path = str(java_file.relative_to(target_dir))
        logger.info("[%d/%d] Scanning: %s", file_idx, len(java_files), rel_path)

        _, chunks = chunk_java_file(java_file)
        unique_chunks = deduplicate_chunks(chunks, seen_hashes)

        if not unique_chunks:
            logger.debug("  No unique chunks in %s; skipping.", rel_path)
            continue

        for chunk_idx, chunk in enumerate(unique_chunks):
            total_chunks_processed += 1
            logger.debug(
                "  Chunk %d/%d (%s, lines %d-%d) → calling inference engine …",
                chunk_idx + 1,
                len(unique_chunks),
                chunk.chunk_type,
                chunk.line_start,
                chunk.line_end,
            )

            try:
                result: dict[str, Any] = engine.analyze_snippet(chunk.content)
                vulns: list[dict[str, Any]] = result.get("vulnerabilities", [])
            except Exception as exc:
                logger.warning(
                    "  Inference failed for chunk %d in %s: %s",
                    chunk_idx,
                    rel_path,
                    exc,
                )
                vulns = []

            if vulns:
                total_vulns += len(vulns)
                logger.info(
                    "  ⚠  %d vulnerability/-ies found in chunk %d (lines %d-%d).",
                    len(vulns),
                    chunk_idx,
                    chunk.line_start,
                    chunk.line_end,
                )

            findings.append(
                ChunkFinding(
                    file_path=rel_path,
                    chunk_index=chunk_idx,
                    chunk=chunk,
                    vulnerabilities=vulns,
                )
            )

    scanned_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report = ScanReport(
        target_directory=str(target_dir.resolve()),
        scanned_at=scanned_at,
        total_files=len(java_files),
        total_chunks=total_chunks_processed,
        total_vulnerabilities=total_vulns,
        engine_adapter=str(adapter_path),
        base_model=base_model_id,
        findings=findings,
    )

    logger.info(
        "Scan complete. Files=%d, Chunks=%d, Vulnerabilities=%d.",
        report.total_files,
        report.total_chunks,
        report.total_vulnerabilities,
    )

    # --- Persist report --------------------------------------------------
    if output_path is not None:
        _save_report(report, output_path)

    return report


def _save_report(report: ScanReport, output_path: Path) -> None:
    """Serialises the scan report to a JSON file."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Scan report saved to: %s", output_path)
    except OSError as exc:
        logger.error("Failed to save scan report: %s", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Java vulnerability scanner powered by a QLoRA fine-tuned LLM."
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Root directory of the Java codebase to scan.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./outputs/vuln-lora",
        help="Path to the saved LoRA adapter directory.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="bigcode/starcoder2-3b",
        help="HuggingFace model ID of the base model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="scan_report.json",
        help="Output path for the JSON scan report.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of .java files to scan (useful for testing).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Apply user-specified log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        report = scan_directory(
            target_dir=Path(args.target),
            adapter_path=args.adapter,
            base_model_id=args.base_model,
            output_path=Path(args.output),
            max_files=args.max_files,
        )

        # Print summary table to stdout
        print("\n" + "=" * 70)
        print(f"{'SCAN SUMMARY':^70}")
        print("=" * 70)
        print(f"  Target     : {args.target}")
        print(f"  Files      : {report.total_files}")
        print(f"  Chunks     : {report.total_chunks}")
        print(f"  Vulnerabilities : {report.total_vulnerabilities}")
        print(f"  Report     : {args.output}")
        print("=" * 70)

        if report.total_vulnerabilities > 0:
            vuln_findings = [f for f in report.findings if f.vulnerabilities]
            print(f"\n  Top findings ({min(5, len(vuln_findings))} of {len(vuln_findings)}):")
            for finding in vuln_findings[:5]:
                for vuln in finding.vulnerabilities:
                    cwe = vuln.get("cwe_id", "N/A")
                    sev = vuln.get("severity", "N/A")
                    func = vuln.get("location", {}).get("function", "unknown")
                    print(
                        f"    [{sev.upper():8s}] {cwe} in {finding.file_path}::{func}"
                    )
        print()

    except KeyboardInterrupt:
        logger.info("Scan interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Scanner failed: %s", exc)
        sys.exit(1)
