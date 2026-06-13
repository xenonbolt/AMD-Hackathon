import json
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SINK_HEURISTICS = {
    # Command Injection
    "CWE-78": r"(Runtime\.getRuntime\(\)\.exec\(|new ProcessBuilder\(|ProcessBuilder\()",
    "CWE-20": r"(Runtime\.getRuntime\(\)\.exec\(|new ProcessBuilder\(|ProcessBuilder\()",
    # Path Traversal / File Access / Insecure Perms
    "CWE-22": r"(new File\(|new FileReader\(|new FileInputStream\(|new FileOutputStream\(|Paths\.get\()",
    "CWE-276": r"(new File\(|new FileReader\(|createTempFile\(|setExecutable\(|setReadable\(|FileOutputStream\()",
    # SQL Injection
    "CWE-89": r"(executeQuery\(|prepareStatement\(|executeUpdate\(|Statement |createStatement\()",
    # XSS
    "CWE-79": r"(getWriter\(\)\.print|out\.println)",
    # LDAP
    "CWE-90": r"(InitialDirContext|search\(|lookup\()",
    # XPath
    "CWE-643": r"(XPath |evaluate\(|compile\()",
    # Hardcoded Key / Cleartext
    "CWE-321": r"(SecretKeySpec|AES|DES)",
    "CWE-319": r"(HttpURLConnection|Socket |http://|ftp://|SocketChannel)",
    # Insufficient Auth / Credentials
    "CWE-522": r"(getConnection\(|DriverManager|password|login)",
    # Open Redirect
    "CWE-601": r"(sendRedirect\(|setHeader\(\"Location\")",
    # Resource Consumption
    "CWE-400": r"(Thread\.sleep\(|readLine\(\)|while \(|for \()",
}

def find_precise_line(java_code: str, start_line: int, end_line: int, cwe_id: str) -> int:
    """Finds the precise line number within the bounds using CWE heuristics."""
    lines = java_code.split('\n')
    
    # Safely clamp bounds
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    
    pattern = SINK_HEURISTICS.get(cwe_id)
    
    if pattern:
        regex = re.compile(pattern)
        # Search within the bounded function
        for i in range(start_idx, end_idx):
            if regex.search(lines[i]):
                return i + 1  # 1-indexed line number
                
    # Fallback: if no heuristic match, just return the first line of the function body
    for i in range(start_idx, end_idx):
        if "{" in lines[i] or "try" in lines[i]:
            return i + 2
            
    return start_line

def refine_dataset(input_path: str, output_path: str):
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        logging.error(f"Input file not found: {input_path}")
        return
        
    refined_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_count += 1
            try:
                data = json.loads(line)
                text = data.get("text", "")
                
                if "<|response|>\n" not in text or "<|input|>\n" not in text:
                    fout.write(line)
                    continue
                    
                # Split text into prompt and response
                parts = text.split("<|response|>\n")
                prompt_part = parts[0]
                response_part = parts[1]
                
                # Extract java code
                java_start = prompt_part.find("<|input|>\n") + len("<|input|>\n")
                java_code = prompt_part[java_start:].strip()
                
                # Parse JSON response
                try:
                    response_json = json.loads(response_part)
                except json.JSONDecodeError:
                    fout.write(line)
                    continue
                
                modified = False
                vulns = response_json.get("vulnerabilities", [])
                
                for vuln in vulns:
                    loc = vuln.get("location", {})
                    cwe_id = vuln.get("cwe_id", "")
                    
                    if "start_line" in loc and "end_line" in loc:
                        precise_line = find_precise_line(
                            java_code, 
                            loc["start_line"], 
                            loc["end_line"], 
                            cwe_id
                        )
                        # Replace block location with precise line
                        vuln["location"] = {"line": precise_line}
                        modified = True
                        
                if modified:
                    refined_count += 1
                    # Reconstruct the text with updated JSON
                    new_text = f"{prompt_part}<|response|>\n{json.dumps(response_json, indent=2)}"
                    fout.write(json.dumps({"text": new_text}) + "\n")
                else:
                    fout.write(line)
                    
            except Exception as e:
                logging.error(f"Error processing line: {e}")
                fout.write(line)
                
    logging.info(f"Refinement complete. Refined {refined_count}/{total_count} examples.")
    logging.info(f"Saved precise dataset to: {output_file}")

if __name__ == "__main__":
    refine_dataset(
        "Dataset/train_classifier_final.jsonl", 
        "Dataset/train_classifier_precise_lines.jsonl"
    )
