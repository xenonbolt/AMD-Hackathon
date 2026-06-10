"""
rag_pipeline.py
===============
Retrieval-Augmented Generation (RAG) pipeline for Java vulnerability intelligence.

This module provides two complementary layers:
  1. **Retriever** – fetches the latest Java CVE/CWE entries from public APIs
     (NVD / NIST and MITRE CWE) and caches them locally.
  2. **RAG context builder** – given a piece of Java code (or a raw query), it
     retrieves the most relevant CVE/CWE records and formats them as an
     augmented context string that can be prepended to an LLM prompt.

Public APIs used (no auth required for basic use):
  - NVD CVE 2.0  : https://services.nvd.nist.gov/rest/json/cves/2.0
  - MITRE CWE    : https://cwe.mitre.org/data/xml/cwec_latest.xml.zip

Usage (standalone):
    python rag_pipeline.py --query "SQL injection in Java JDBC" --top_k 5
    python rag_pipeline.py --refresh   # force refresh the cache
"""

import argparse
import json
import logging
import math
import os
import re
import time
import zipfile
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree

import requests

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_pipeline")

# ── Constants ─────────────────────────────────────────────────────────────────
NVD_CVE_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
MITRE_CWE_XML_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"

CACHE_DIR = Path("Dataset/rag_cache")
CVE_CACHE_FILE = CACHE_DIR / "java_cves.json"
CWE_CACHE_FILE = CACHE_DIR / "cwe_catalog.json"

# How old the cache must be before auto-refresh (default 24 h)
CACHE_TTL_HOURS = 24

# NVD rate-limit: 5 req / 30 s without API key
NVD_REQUEST_DELAY_S = 7

# Java-related keyword and CPE fragments used to filter NVD results
JAVA_KEYWORDS = [
    "java",
    "spring",
    "struts",
    "log4j",
    "jackson",
    "hibernate",
    "tomcat",
    "jetty",
]

# Maximum CVEs to fetch per keyword from NVD (each page = up to 2000 records)
NVD_RESULTS_PER_PAGE = 2000
NVD_MAX_PAGES_PER_KW = 1  # one page of 2000 per keyword is usually sufficient


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_is_fresh(cache_path: Path, ttl_hours: int = CACHE_TTL_HOURS) -> bool:
    """Returns True if the cache file exists and was modified within the TTL."""
    if not cache_path.exists():
        return False
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(hours=ttl_hours)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved cache → {path}")


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── NVD CVE Fetcher ───────────────────────────────────────────────────────────

def fetch_nvd_java_cves(
    api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetches Java-related CVE records from the NVD CVE 2.0 REST API.

    Returns a deduplicated list of dicts with the keys:
        cve_id, description, cvss_score, severity, cwe_ids, published, last_modified
    """
    if not force_refresh and _cache_is_fresh(CVE_CACHE_FILE):
        logger.info(f"CVE cache is fresh. Loading from {CVE_CACHE_FILE}")
        return _load_json(CVE_CACHE_FILE)

    logger.info("Fetching latest Java CVEs from NVD …")
    headers = {}
    if api_key:
        headers["apiKey"] = api_key

    seen_ids: set = set()
    all_cves: List[Dict[str, Any]] = []

    for keyword in JAVA_KEYWORDS:
        for page in range(NVD_MAX_PAGES_PER_KW):
            start_index = page * NVD_RESULTS_PER_PAGE
            params = {
                "keywordSearch": keyword,
                "resultsPerPage": NVD_RESULTS_PER_PAGE,
                "startIndex": start_index,
            }
            try:
                logger.info(f"  NVD query: keyword='{keyword}', startIndex={start_index}")
                resp = requests.get(
                    NVD_CVE_BASE_URL,
                    params=params,
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as exc:
                logger.error(f"  NVD request failed for keyword='{keyword}': {exc}")
                break

            vulnerabilities = data.get("vulnerabilities", [])
            if not vulnerabilities:
                break

            for item in vulnerabilities:
                cve_obj = item.get("cve", {})
                cve_id = cve_obj.get("id", "")
                if not cve_id or cve_id in seen_ids:
                    continue
                seen_ids.add(cve_id)

                # Description (prefer English)
                descriptions = cve_obj.get("descriptions", [])
                description = next(
                    (d["value"] for d in descriptions if d.get("lang") == "en"),
                    descriptions[0]["value"] if descriptions else "",
                )

                # CVSS score & severity (prefer v3.1 > v3.0 > v2.0)
                metrics = cve_obj.get("metrics", {})
                cvss_score, severity = _extract_cvss(metrics)

                # CWE IDs
                weaknesses = cve_obj.get("weaknesses", [])
                cwe_ids = []
                for w in weaknesses:
                    for desc in w.get("description", []):
                        val = desc.get("value", "")
                        if val.startswith("CWE-"):
                            cwe_ids.append(val)

                all_cves.append({
                    "cve_id": cve_id,
                    "description": description,
                    "cvss_score": cvss_score,
                    "severity": severity,
                    "cwe_ids": list(set(cwe_ids)),
                    "published": cve_obj.get("published", ""),
                    "last_modified": cve_obj.get("lastModified", ""),
                    "source_keyword": keyword,
                })

            # Respect NVD rate limit
            time.sleep(NVD_REQUEST_DELAY_S)

            total_results = data.get("totalResults", 0)
            fetched_so_far = start_index + len(vulnerabilities)
            if fetched_so_far >= total_results:
                break

    logger.info(f"Fetched {len(all_cves)} unique Java CVEs from NVD.")
    _save_json(CVE_CACHE_FILE, all_cves)
    return all_cves


def _extract_cvss(metrics: Dict) -> tuple:
    """Returns (score, severity) preferring CVSSv3.1 > CVSSv3.0 > CVSSv2."""
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, [])
        if entries:
            data = entries[0].get("cvssData", {})
            score = data.get("baseScore")
            severity = data.get("baseSeverity") or entries[0].get("baseSeverity", "")
            if score is not None:
                return float(score), str(severity).upper()
    return None, "UNKNOWN"


# ── MITRE CWE Fetcher ─────────────────────────────────────────────────────────

def fetch_mitre_cwe_catalog(force_refresh: bool = False) -> Dict[str, Dict[str, str]]:
    """
    Downloads and parses the MITRE CWE XML catalog.

    Returns a dict keyed by CWE-ID string (e.g., 'CWE-89') with:
        name, description, extended_description, url
    """
    if not force_refresh and _cache_is_fresh(CWE_CACHE_FILE):
        logger.info(f"CWE catalog cache is fresh. Loading from {CWE_CACHE_FILE}")
        return _load_json(CWE_CACHE_FILE)

    logger.info("Downloading MITRE CWE XML catalog …")
    try:
        resp = requests.get(MITRE_CWE_XML_URL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.error(f"Failed to download CWE catalog: {exc}")
        return {}

    catalog: Dict[str, Dict[str, str]] = {}
    try:
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            xml_filename = next(n for n in zf.namelist() if n.endswith(".xml"))
            with zf.open(xml_filename) as xml_file:
                tree = ElementTree.parse(xml_file)

        root = tree.getroot()
        ns = {"cwe": "http://cwe.mitre.org/cwe-7"}

        for weakness in root.iter("{http://cwe.mitre.org/cwe-7}Weakness"):
            cwe_num = weakness.get("ID", "")
            cwe_id = f"CWE-{cwe_num}"
            name = weakness.get("Name", "")

            desc_el = weakness.find("{http://cwe.mitre.org/cwe-7}Description")
            description = (desc_el.text or "").strip() if desc_el is not None else ""

            ext_el = weakness.find("{http://cwe.mitre.org/cwe-7}Extended_Description")
            extended = (ext_el.text or "").strip() if ext_el is not None else ""

            catalog[cwe_id] = {
                "cwe_id": cwe_id,
                "name": name,
                "description": description,
                "extended_description": extended,
                "url": f"https://cwe.mitre.org/data/definitions/{cwe_num}.html",
            }

        logger.info(f"Parsed {len(catalog)} CWE entries from MITRE catalog.")
    except Exception as exc:
        logger.error(f"Failed to parse CWE XML: {exc}")
        return {}

    _save_json(CWE_CACHE_FILE, catalog)
    return catalog


# ── TF-IDF Retriever (no external vector DB required) ─────────────────────────

class TFIDFRetriever:
    """
    Lightweight TF-IDF based retriever to find relevant CVE/CWE records
    given a natural-language query or a code snippet.
    No GPU or external vector store required.
    """

    def __init__(self, documents: List[Dict[str, Any]], text_field: str = "text") -> None:
        self.documents = documents
        self.text_field = text_field
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        # Split on non-alphanumeric, keep numbers
        tokens = re.findall(r"[a-z0-9]+", text)
        return tokens

    def _build_index(self) -> None:
        logger.info("Building TF-IDF index …")
        N = len(self.documents)
        self._tf: List[Dict[str, float]] = []
        df: Dict[str, int] = {}

        for doc in self.documents:
            tokens = self._tokenize(doc.get(self.text_field, ""))
            tf: Dict[str, float] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            # Normalize
            total = max(sum(tf.values()), 1)
            tf = {k: v / total for k, v in tf.items()}
            self._tf.append(tf)
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1

        # IDF = log(N / (1 + df))
        self._idf: Dict[str, float] = {
            token: math.log((N + 1) / (count + 1)) + 1
            for token, count in df.items()
        }
        logger.info("TF-IDF index ready.")

    def _score(self, query: str, doc_idx: int) -> float:
        tokens = self._tokenize(query)
        tf = self._tf[doc_idx]
        score = sum(tf.get(t, 0) * self._idf.get(t, 0) for t in tokens)
        return score

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        scores = [(i, self._score(query, i)) for i in range(len(self.documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            doc = dict(self.documents[idx])
            doc["_relevance_score"] = round(score, 4)
            results.append(doc)
        return results


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class JavaVulnRAGPipeline:
    """
    End-to-end RAG pipeline for Java vulnerability intelligence.

    Steps:
      1. On init: load (or fetch) CVE + CWE data.
      2. Build a combined document index.
      3. On query: retrieve top-k docs and format as LLM-ready context.
    """

    def __init__(
        self,
        nvd_api_key: Optional[str] = None,
        force_refresh: bool = False,
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        self.nvd_api_key = nvd_api_key

        # Load data sources
        self.cve_records = fetch_nvd_java_cves(
            api_key=nvd_api_key, force_refresh=force_refresh
        )
        self.cwe_catalog = fetch_mitre_cwe_catalog(force_refresh=force_refresh)

        # Merge CWE details into CVE records for richer retrieval docs
        self._documents = self._build_documents()

        # Build retriever
        self.retriever = TFIDFRetriever(self._documents, text_field="text")
        logger.info(
            f"RAG pipeline ready. "
            f"Documents: {len(self._documents)} | "
            f"CVEs: {len(self.cve_records)} | "
            f"CWEs in catalog: {len(self.cwe_catalog)}"
        )

    def _build_documents(self) -> List[Dict[str, Any]]:
        """Converts CVE records (+ joined CWE info) into flat retrieval documents."""
        docs = []
        for cve in self.cve_records:
            cwe_details = []
            for cwe_id in cve.get("cwe_ids", []):
                info = self.cwe_catalog.get(cwe_id)
                if info:
                    cwe_details.append(
                        f"{cwe_id} – {info['name']}: {info['description']}"
                    )

            cwe_text = "; ".join(cwe_details) if cwe_details else "No CWE details available"

            # Concatenated searchable text
            full_text = (
                f"{cve['cve_id']} {cve['description']} "
                f"severity:{cve['severity']} score:{cve['cvss_score']} "
                f"cwe:{' '.join(cve.get('cwe_ids', []))} {cwe_text}"
            )

            docs.append({
                "type": "CVE",
                "cve_id": cve["cve_id"],
                "description": cve["description"],
                "cvss_score": cve["cvss_score"],
                "severity": cve["severity"],
                "cwe_ids": cve.get("cwe_ids", []),
                "cwe_details": cwe_details,
                "published": cve.get("published", ""),
                "last_modified": cve.get("last_modified", ""),
                "text": full_text,
            })

        # Also add standalone CWE entries so the retriever can match on weakness names
        for cwe_id, info in self.cwe_catalog.items():
            docs.append({
                "type": "CWE",
                "cwe_id": cwe_id,
                "name": info["name"],
                "description": info["description"],
                "url": info["url"],
                "text": f"{cwe_id} {info['name']} {info['description']} {info.get('extended_description', '')}",
            })

        return docs

    def query(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves the top-k most relevant CVE/CWE records for a given query.

        Args:
            query: Natural language question, code snippet keywords, or CVE/CWE ID.
            top_k: Override the instance-level top_k for this call.

        Returns:
            List of ranked result dicts.
        """
        k = top_k if top_k is not None else self.top_k
        return self.retriever.retrieve(query, top_k=k)

    def build_rag_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Retrieves relevant records and formats them as a ready-to-use
        LLM context block.

        Returns a multi-line string that can be injected before the main prompt.
        """
        results = self.query(query, top_k=top_k)

        lines = [
            "### Relevant CVE/CWE Intelligence (Retrieved Context):",
            f"Query: {query[:200]}",
            "",
        ]

        for rank, doc in enumerate(results, start=1):
            if doc["type"] == "CVE":
                cwe_str = ", ".join(doc["cwe_ids"]) if doc["cwe_ids"] else "N/A"
                cwe_details_str = (
                    "\n  ".join(doc["cwe_details"]) if doc["cwe_details"] else "N/A"
                )
                lines.append(
                    f"[{rank}] CVE: {doc['cve_id']}"
                    f"\n  Severity : {doc['severity']} (CVSS: {doc['cvss_score']})"
                    f"\n  CWE IDs  : {cwe_str}"
                    f"\n  CWE Info : {cwe_details_str}"
                    f"\n  Published: {doc['published'][:10] if doc['published'] else 'N/A'}"
                    f"\n  Summary  : {doc['description'][:300]}"
                )
            else:  # CWE standalone
                lines.append(
                    f"[{rank}] CWE: {doc['cwe_id']} – {doc['name']}"
                    f"\n  Details  : {doc['description'][:300]}"
                    f"\n  Reference: {doc.get('url', 'N/A')}"
                )
            lines.append("")

        return "\n".join(lines)

    def lookup_cve(self, cve_id: str) -> Optional[Dict[str, Any]]:
        """Direct CVE lookup by ID from the cached index."""
        cve_id = cve_id.upper().strip()
        for doc in self._documents:
            if doc.get("cve_id") == cve_id:
                return doc
        return None

    def lookup_cwe(self, cwe_id: str) -> Optional[Dict[str, str]]:
        """Direct CWE lookup by ID (e.g., 'CWE-89') from the catalog."""
        return self.cwe_catalog.get(cwe_id.upper().strip())

    def get_stats(self) -> Dict[str, int]:
        """Returns summary statistics about the loaded data."""
        cve_docs = [d for d in self._documents if d["type"] == "CVE"]
        cwe_docs = [d for d in self._documents if d["type"] == "CWE"]
        return {
            "total_documents": len(self._documents),
            "cve_records": len(cve_docs),
            "cwe_entries": len(cwe_docs),
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Java Vulnerability RAG Pipeline — fetch & retrieve CVE/CWE intelligence"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="SQL injection java JDBC prepared statement",
        help="Query string (code keywords, vulnerability name, etc.)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh CVE and CWE caches from the internet",
    )
    parser.add_argument(
        "--nvd_api_key",
        type=str,
        default=None,
        help="Optional NVD API key for higher rate limits (get one at nvd.nist.gov/developers)",
    )
    parser.add_argument(
        "--lookup_cve",
        type=str,
        default=None,
        help="Directly look up a specific CVE (e.g., CVE-2021-44228)",
    )
    parser.add_argument(
        "--lookup_cwe",
        type=str,
        default=None,
        help="Directly look up a specific CWE (e.g., CWE-89)",
    )
    args = parser.parse_args()

    pipeline = JavaVulnRAGPipeline(
        nvd_api_key=args.nvd_api_key,
        force_refresh=args.refresh,
        top_k=args.top_k,
    )

    stats = pipeline.get_stats()
    print(f"\n{'='*60}")
    print(f"RAG Pipeline Stats: {stats}")
    print(f"{'='*60}\n")

    if args.lookup_cve:
        result = pipeline.lookup_cve(args.lookup_cve)
        print(f"CVE Lookup: {args.lookup_cve}")
        print(json.dumps(result, indent=2) if result else "Not found in cache.")
    elif args.lookup_cwe:
        result = pipeline.lookup_cwe(args.lookup_cwe)
        print(f"CWE Lookup: {args.lookup_cwe}")
        print(json.dumps(result, indent=2) if result else "Not found in catalog.")
    else:
        context = pipeline.build_rag_context(args.query, top_k=args.top_k)
        print(context)
