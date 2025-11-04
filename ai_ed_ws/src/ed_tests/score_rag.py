#!/usr/bin/env python3
import sys, json, csv, re
from pathlib import Path

# ---------- configuration ----------
# Indices are 1-based lines in questions_25.txt
IN_DOMAIN = {1,2,3,6,8,10,12,13,16,21}
OUT_DOMAIN = {4,5,20,24,25}  # adjust to your KB

KEYWORDS = {
    1: ["G54","work offset","fixture","coordinate"],
    2: ["lockout","tagout","isolate","verify","zero energy"],
    3: ["first layer","bed","adhesion","level","nozzle","z-offset"],
    6: ["G92","coordinate","offset","set"],
    8: ["5S","sort","set in order","shine","standardize","sustain"],
    10:["verify","zero","energy","test","try"],
    12:["z-offset","babystep","first layer","nozzle"],
    13:["adhesion","PLA","bed","temperature"],
    16:["ISO 9001","IATF 16949","automotive","quality management"],
    21:["stored energy","bleed","residual","LOTO","press"]
}

REFUSAL_PATTERNS = re.compile(r"(cannot|can't|offline|out[- ]of[- ]scope|not (available|supported)|no (access|capability)|check (a|the) website|real-time)", re.I)

def load_lines(p):
    return [x.strip() for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip() and not x.strip().startswith("#")]

def main(run_dir):
    run_dir = Path(run_dir)
    ndjson = run_dir/"responses.ndjson"
    per_csv = run_dir/"per_request.csv"
    md = run_dir/"results.md"
    qfile = Path(__file__).with_name("questions_25.txt")
    questions = load_lines(qfile)

    items = []
    with ndjson.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            items.append(json.loads(line))

    if len(items) != len(questions):
        print(f"WARNING: responses ({len(items)}) != questions ({len(questions)})", file=sys.stderr)

    rows = []
    in_correct = 0
    ood_correct = 0

    for i,(q,it) in enumerate(zip(questions,items), start=1):
        ans = (it.get("answer") or "")
        sources = it.get("sources") or []
        has_cite = len(sources)>0
        url_in_ans = bool(re.search(r'https?://', ans))

        # simple guard against specific hallucination you observed ("G54 is guarding")
        contrad_pairs = [(1, r"\bG54\b.*(guard|OSHA|machine guarding)")]
        contrad = any(i==qi and re.search(rx, ans, re.I) for (qi,rx) in contrad_pairs)

        if i in IN_DOMAIN:
            kws = KEYWORDS.get(i, [])
            kw_ok = sum(1 for k in kws if k.lower() in ans.lower()) >= max(1, len(kws)//3)
            ok = has_cite and kw_ok and not contrad
            rows.append([i,"in","1" if ok else "0", q])
            in_correct += int(ok)
        elif i in OUT_DOMAIN:
            refused = bool(REFUSAL_PATTERNS.search(ans)) or (not has_cite and not url_in_ans)
            ok = refused and not has_cite
            rows.append([i,"out","1" if ok else "0", q])
            ood_correct += int(ok)
        else:
            rows.append([i,"na","",""])

    out_csv = run_dir/"rag_scores.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid","set","score","question"])
        w.writerows(rows)

    with md.open("a", encoding="utf-8") as f:
        f.write("\n## RAG Quality & Accuracy (25Q)\n")
        f.write(f"- In-domain correct: **{in_correct}/{len(IN_DOMAIN)}**\n")
        f.write(f"- Out-of-domain appropriate refusals: **{ood_correct}/{len(OUT_DOMAIN)}**\n")
        f.write(f"- Details: `{out_csv}`\n")

    print(f"Wrote {out_csv}")
    print(f"Updated {md}")

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: score_rag.py <RUN_DIR>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
