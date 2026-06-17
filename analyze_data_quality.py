"""Quick analysis of training data quality for vulnerability detection model."""
import json

path = r"c:\Users\Arghya\Desktop\Solutions\AMD-Hackathon\Dataset\train_classifier_precise_lines.jsonl"
lines = open(path, encoding="utf-8").readlines()

total = len(lines)
has_response = 0
no_response = 0
empty_vuln = 0
has_vuln = 0
vuln_counts = {}
truncated = 0
template_desc = 0
template_impact = 0
template_rec = 0
missing_confidence = 0
missing_location = 0
cwe_distribution = {}
all_descriptions = []
all_vuln_entries = []

for l in lines:
    data = json.loads(l)
    text = data.get("text", "")
    
    if "<|response|>" in text:
        has_response += 1
        resp_part = text.split("<|response|>")[-1].strip()
        try:
            resp_json = json.loads(resp_part)
            vulns = resp_json.get("vulnerabilities", [])
            if not vulns:
                empty_vuln += 1
            else:
                has_vuln += 1
                vc = len(vulns)
                vuln_counts[vc] = vuln_counts.get(vc, 0) + 1
                for v in vulns:
                    all_vuln_entries.append(v)
                    cwe = v.get("cwe_id", "NONE")
                    cwe_distribution[cwe] = cwe_distribution.get(cwe, 0) + 1
                    desc = v.get("description", "")
                    impact = v.get("impact", "")
                    rec = v.get("recommendation", "")
                    all_descriptions.append(desc)
                    if "contains a vulnerability associated with" in desc:
                        template_desc += 1
                    if "can lead to compromised integrity, confidentiality, or availability" in impact:
                        template_impact += 1
                    if "apply secure coding practices to remediate" in rec:
                        template_rec += 1
                    if "confidence" not in v:
                        missing_confidence += 1
                    if "location" not in v:
                        missing_location += 1
        except json.JSONDecodeError:
            truncated += 1
    else:
        no_response += 1

total_vulns = sum(vuln_counts.get(k, 0) * k for k in vuln_counts)

print("=== TRAINING DATA QUALITY REPORT ===")
print(f"Total records: {total}")
print(f"Has response marker: {has_response}")
print(f"Missing response marker: {no_response}")
print(f"Truncated/unparseable response: {truncated}")
print(f"Empty vulnerabilities (negatives): {empty_vuln}")
print(f"Has vulnerabilities (positives): {has_vuln}")
print(f"Negative:Positive ratio: {empty_vuln}:{has_vuln}")
print()
print("=== VULNERABILITY COUNT DISTRIBUTION ===")
for k in sorted(vuln_counts.keys()):
    print(f"  {k} vuln(s): {vuln_counts[k]} records")
print()
print("=== CWE DISTRIBUTION ===")
for k, v in sorted(cwe_distribution.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
print()
print("=== TEMPLATE/BOILERPLATE ANALYSIS ===")
print(f"Total individual vuln entries: {total_vulns}")
pct = lambda n: f"{n*100//max(total_vulns,1)}%"
print(f"Templated descriptions: {template_desc} ({pct(template_desc)})")
print(f"Templated impacts: {template_impact} ({pct(template_impact)})")
print(f"Templated recommendations: {template_rec} ({pct(template_rec)})")
print(f"Missing confidence field: {missing_confidence} ({pct(missing_confidence)})")
print(f"Missing location field: {missing_location} ({pct(missing_location)})")

# Check for duplicate descriptions
from collections import Counter
desc_counts = Counter(all_descriptions)
dup_descs = {k: v for k, v in desc_counts.items() if v > 3}
print()
print("=== MOST DUPLICATED DESCRIPTIONS (>3 occurrences) ===")
for d, c in sorted(dup_descs.items(), key=lambda x: -x[1])[:10]:
    print(f"  [{c}x] {d[:120]}")

# Check for code-specific vs generic descriptions
code_specific = 0
for desc in all_descriptions:
    # A code-specific description mentions actual variable names, methods, classes
    if any(kw in desc for kw in ["variable", "method ", "function ", "parameter ", "class ", "line ", "`"]):
        code_specific += 1
print()
print(f"Code-specific descriptions (mention vars/methods): {code_specific} ({pct(code_specific)})")
print(f"Generic/template descriptions: {total_vulns - code_specific} ({pct(total_vulns - code_specific)})")

# Check unique descriptions
unique_descs = len(set(all_descriptions))
print(f"Unique descriptions: {unique_descs} out of {total_vulns}")
