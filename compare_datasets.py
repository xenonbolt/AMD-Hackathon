"""Compare quality metrics between old and new training datasets."""
import json
from collections import Counter

def analyze_dataset(path, label):
    lines = open(path, encoding="utf-8").readlines()
    total = len(lines)
    has_vuln = 0
    empty_vuln = 0
    truncated = 0
    template_desc = 0
    template_impact = 0
    template_rec = 0
    all_descriptions = []
    cwe_distribution = {}
    vuln_count_dist = Counter()
    total_vulns = 0
    multi_vuln = 0

    for l in lines:
        data = json.loads(l)
        text = data.get("text", "")
        if "<|response|>" not in text:
            continue
        resp_part = text.split("<|response|>")[-1].strip()
        try:
            resp_json = json.loads(resp_part)
        except json.JSONDecodeError:
            truncated += 1
            continue

        vulns = resp_json.get("vulnerabilities", [])
        vc = len(vulns)
        vuln_count_dist[vc] += 1

        if not vulns:
            empty_vuln += 1
        else:
            has_vuln += 1
            total_vulns += vc
            if vc > 1:
                multi_vuln += 1

            for v in vulns:
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

    unique_descs = len(set(all_descriptions))
    unique_cwes = len(cwe_distribution)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total records:              {total}")
    print(f"  Positive (has vulns):        {has_vuln}")
    print(f"  Negative (clean code):       {empty_vuln}")
    print(f"  Truncated/broken:            {truncated}")
    print(f"  Total vuln entries:          {total_vulns}")
    print(f"  Multi-vuln files:            {multi_vuln}")
    print(f"  Unique CWE types:            {unique_cwes}")
    print(f"  Unique descriptions:         {unique_descs} / {total_vulns}")
    pct = lambda n: f"{n*100//max(total_vulns,1)}%"
    print(f"  Templated descriptions:      {template_desc} ({pct(template_desc)})")
    print(f"  Templated impacts:           {template_impact} ({pct(template_impact)})")
    print(f"  Templated recommendations:   {template_rec} ({pct(template_rec)})")
    print(f"\n  Vuln count distribution:")
    for k in sorted(vuln_count_dist.keys()):
        print(f"    {k} vuln(s): {vuln_count_dist[k]} files")
    print(f"\n  CWE distribution:")
    for cwe, count in sorted(cwe_distribution.items(), key=lambda x: -x[1])[:20]:
        print(f"    {cwe}: {count}")

    # Show sample descriptions
    if all_descriptions:
        print(f"\n  Sample descriptions (first 3 unique):")
        seen = set()
        for d in all_descriptions:
            if d not in seen and len(d) > 20:
                seen.add(d)
                print(f"    -> {d[:150]}...")
                if len(seen) >= 3:
                    break


old_path = r"c:\Users\Arghya\Desktop\Solutions\AMD-Hackathon\Dataset\train_classifier_precise_lines.jsonl"
new_path = r"c:\Users\Arghya\Desktop\Solutions\AMD-Hackathon\Dataset\train_production.jsonl"

analyze_dataset(old_path, "OLD DATASET (train_classifier_precise_lines.jsonl)")
analyze_dataset(new_path, "NEW DATASET (train_production.jsonl)")
