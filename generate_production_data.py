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
