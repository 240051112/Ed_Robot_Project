#!/usr/bin/env python3
import sys, csv
from pathlib import Path

def load_csv(p):
    rows=[]
    if Path(p).exists():
        with open(p, newline="", encoding="utf-8") as f:
            r=csv.DictReader(f)
            rows=list(r)
    return rows

def md_table(rows, headers):
    out="| " + " | ".join(headers) + " |\n"
    out+="|"+"|".join(["---"]*len(headers))+"|\n"
    for r in rows:
        out+="| " + " | ".join(r.get(h,"") for h in headers) + " |\n"
    return out

def main(run_dir):
    run_dir = Path(run_dir)
    per_csv = run_dir/"per_request.csv"
    sum_csv = run_dir/"summary.csv"
    rag_csv = run_dir/"rag_scores.csv"
    md = run_dir/"results.md"

    per = load_csv(per_csv)
    rag = load_csv(rag_csv)

    slow = sorted(per, key=lambda x: float(x.get("total_s") or 0.0), reverse=True)[:5]
    high_pavg = [x for x in per if x.get("avg_power_w") and x["avg_power_w"]!="NA"]
    high_pavg = sorted(high_pavg, key=lambda x: float(x["avg_power_w"]), reverse=True)[:5]
    high_ppk = [x for x in per if x.get("peak_power_w") and x["peak_power_w"]!="NA"]
    high_ppk = sorted(high_ppk, key=lambda x: float(x["peak_power_w"]), reverse=True)[:5]

    with md.open("a", encoding="utf-8") as f:
        f.write("\n## Hotspots (25Q)\n")
        if slow:
            f.write("**Slowest by latency**\n\n")
            f.write(md_table(slow, ["qid","mode","total_s","answer_len","sources_count","avg_power_w","peak_power_w"])+"\n")
        if high_pavg:
            f.write("**Highest average power**\n\n")
            f.write(md_table(high_pavg, ["qid","mode","total_s","avg_power_w","peak_power_w","answer_len"])+"\n")
        if high_ppk:
            f.write("**Highest peak power**\n\n")
            f.write(md_table(high_ppk, ["qid","mode","total_s","peak_power_w","avg_power_w","answer_len"])+"\n")
        if rag:
            in_ok=sum(int(r["score"]) for r in rag if r["set"]=="in")
            in_tot=sum(1 for r in rag if r["set"]=="in")
            out_ok=sum(int(r["score"]) for r in rag if r["set"]=="out")
            out_tot=sum(1 for r in rag if r["set"]=="out")
            f.write("\n## RAG Score (recap)\n")
            f.write(f"- In-domain: **{in_ok}/{in_tot}**\n- Out-of-domain: **{out_ok}/{out_tot}**\n")

    print(f"Updated {md}")

if __name__ == "__main__":
    if len(sys.argv)<2:
        print("Usage: summarize_run.py <RUN_DIR>", file=sys.stderr); sys.exit(1)
    main(sys.argv[1])
