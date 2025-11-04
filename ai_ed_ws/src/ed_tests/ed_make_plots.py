#!/usr/bin/env python3
"""
ed_make_plots.py — plots for the 25-question ED evaluation

Reads a summary CSV from your automated test run and produces:
  - latency_bar.png
  - latency_hist.png
  - latency_cdf.png
  - power_vs_latency.png       (if power columns present)
  - util_vs_latency.png        (if util columns present)
  - tegrastats_power.png       (if --tegrastats provided)
Also writes metrics_summary.json with aggregate stats.

No internet required. Matplotlib only (no seaborn).
"""

import argparse, json, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Helpers
# ---------------------------
ALIASES = {
    "qid": ["qid", "id", "index"],
    "question": ["question", "prompt", "query"],
    "latency_s": ["total_s", "latency_s", "latency_sec", "total_time_s"],
    "latency_ms": ["latency_ms", "total_ms"],
    "gen_s": ["generate_s", "gen_s", "llm_s", "inference_s"],
    "retr_s": ["retrieve_s", "retrieval_s", "retr_s"],
    "pwr_avg": ["power_avg_w", "avg_power_w", "power_w", "pwr_w"],
    "pwr_peak": ["power_peak_w", "peak_power_w", "pwr_peak_w", "pwr_max_w"],
    "gpu": ["gpu_util", "gpu%", "gpu_utilization", "gpu_utilization_%"],
    "cpu": ["cpu_util", "cpu%", "cpu_utilization", "cpu_utilization_%"],
}

def find_col(df: pd.DataFrame, keys: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in cols: return cols[k]
    # try fuzzy match
    for k in keys:
        for lc, orig in cols.items():
            if k in lc:
                return orig
    return None

def map_columns(df: pd.DataFrame) -> dict:
    got = {}
    for k, al in ALIASES.items():
        c = find_col(df, al)
        if c: got[k] = c
    # derive latency_s
    if "latency_s" not in got:
        if "latency_ms" in got:
            df["__latency_s"] = df[got["latency_ms"]].astype(float) / 1000.0
            got["latency_s"] = "__latency_s"
    return got

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))

# ---------------------------
# Plotters (single-plot figs)
# ---------------------------
def plot_latency_bar(df, col_lat_s, outdir: Path):
    vals = df[col_lat_s].astype(float).values
    ids = df.index if "qid" not in df.columns else df["qid"]
    labels = [f"Q{i+1}" for i in range(len(vals))]
    plt.figure()
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), labels, rotation=0)
    plt.ylabel("Latency (s)")
    plt.title("End-to-End Latency by Question")
    plt.tight_layout()
    plt.savefig(outdir / "latency_bar.png", dpi=200)
    plt.close()

def plot_latency_hist(df, col_lat_s, outdir: Path):
    vals = df[col_lat_s].astype(float).values
    plt.figure()
    plt.hist(vals, bins=10)
    plt.xlabel("Latency (s)")
    plt.ylabel("Count")
    plt.title("Latency Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "latency_hist.png", dpi=200)
    plt.close()

def plot_latency_cdf(df, col_lat_s, outdir: Path):
    vals = np.sort(df[col_lat_s].astype(float).values)
    y = np.arange(1, len(vals)+1) / len(vals)
    plt.figure()
    plt.plot(vals, y)
    plt.xlabel("Latency (s)")
    plt.ylabel("Cumulative Fraction")
    plt.title("Latency CDF")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / "latency_cdf.png", dpi=200)
    plt.close()

def plot_power_vs_latency(df, col_lat_s, pwr_avg, pwr_peak, outdir: Path):
    if not (pwr_avg or pwr_peak): return
    plt.figure()
    if pwr_avg:
        plt.scatter(df[col_lat_s], df[pwr_avg], label="Avg Power (W)")
    if pwr_peak:
        plt.scatter(df[col_lat_s], df[pwr_peak], marker="x", label="Peak Power (W)")
    plt.xlabel("Latency (s)")
    plt.ylabel("Power (W)")
    plt.title("Power vs Latency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "power_vs_latency.png", dpi=200)
    plt.close()

def plot_util_vs_latency(df, col_lat_s, gpu, cpu, outdir: Path):
    if not (gpu or cpu): return
    plt.figure()
    if gpu:
        plt.scatter(df[col_lat_s], df[gpu], label="GPU Util (%)")
    if cpu:
        plt.scatter(df[col_lat_s], df[cpu], marker="x", label="CPU Util (%)")
    plt.xlabel("Latency (s)")
    plt.ylabel("Utilization (%)")
    plt.title("Utilization vs Latency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "util_vs_latency.png", dpi=200)
    plt.close()

# ---------------------------
# Tegrastats power parser (optional)
# ---------------------------
TEG_RE = re.compile(r"(\d+.\d+)@(\d+.\d+)W")  # fallback generic pattern

def parse_tegrastats(path: Path):
    """
    Very loose parser; tries to extract time index and power (W).
    Accepts lines like 'RAM 4000/15928MB ... POM_5V_IN 12.34W ...'
    If no time present, uses sample index as time step (1 unit per line).
    """
    times, pwrs = [], []
    with path.open() as f:
        t = 0.0
        for line in f:
            # common: look for 'POM_5V_IN <num>W'
            m = re.search(r"POM_5V_IN\s+([0-9.]+)W", line)
            if not m:
                m = re.search(r"([0-9.]+)\s*W", line)
            if m:
                pw = float(m.group(1))
                times.append(t)
                pwrs.append(pw)
            t += 1.0
    return np.array(times), np.array(pwrs)

def plot_tegrastats_power(ts_path: Path, outdir: Path):
    t, p = parse_tegrastats(ts_path)
    if len(p) == 0:
        return None
    plt.figure()
    plt.plot(t, p)
    plt.xlabel("Sample Index")
    plt.ylabel("Power (W)")
    plt.title("Tegrastats Power Trace")
    plt.tight_layout()
    out = outdir / "tegrastats_power.png"
    plt.savefig(out, dpi=200)
    plt.close()
    return dict(samples=len(p), avg=float(np.mean(p)), peak=float(np.max(p)))

# ---------------------------
# Aggregates
# ---------------------------
def summarize(df, col_lat_s, col_gen=None, col_retr=None, pwr_avg=None, pwr_peak=None, gpu=None, cpu=None):
    def stats(x):
        x = pd.Series(x).astype(float).dropna()
        return dict(
            mean=float(x.mean()),
            p50=float(x.quantile(0.5)),
            p90=float(x.quantile(0.9)),
            min=float(x.min()),
            max=float(x.max()),
            count=int(x.size),
        )

    out = {"latency_s": stats(df[col_lat_s])}

    if col_gen and col_gen in df.columns:
        out["generate_s"] = stats(df[col_gen])
    if col_retr and col_retr in df.columns:
        out["retrieve_s"] = stats(df[col_retr])
    if pwr_avg and pwr_avg in df.columns:
        out["power_avg_w"] = stats(df[pwr_avg])
    if pwr_peak and pwr_peak in df.columns:
        out["power_peak_w"] = stats(df[pwr_peak])
    if gpu and gpu in df.columns:
        out["gpu_util_%"] = stats(df[gpu])
    if cpu and cpu in df.columns:
        out["cpu_util_%"] = stats(df[cpu])

    return out

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to summary.csv")
    ap.add_argument("--out", required=True, help="Output directory for figures")
    ap.add_argument("--tegrastats", help="Optional tegrastats log for power trace")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    outdir = Path(args.out).expanduser().resolve()
    ensure_outdir(outdir)

    df = pd.read_csv(csv_path)
    colmap = map_columns(df)
    if "latency_s" not in colmap:
        raise SystemExit("Could not find latency column (looked for total_s / latency_ms).")

    col_lat = colmap["latency_s"]
    col_gen  = colmap.get("gen_s")
    col_retr = colmap.get("retr_s")
    pwr_avg  = colmap.get("pwr_avg")
    pwr_peak = colmap.get("pwr_peak")
    gpu_col  = colmap.get("gpu")
    cpu_col  = colmap.get("cpu")

    # Plots
    plot_latency_bar(df, col_lat, outdir)
    plot_latency_hist(df, col_lat, outdir)
    plot_latency_cdf(df, col_lat, outdir)
    plot_power_vs_latency(df, col_lat, pwr_avg, pwr_peak, outdir)
    plot_util_vs_latency(df, col_lat, gpu_col, cpu_col, outdir)

    summary = summarize(df, col_lat, col_gen, col_retr, pwr_avg, pwr_peak, gpu_col, cpu_col)

    # Optional tegrastats
    if args.tegrastats:
        ts_info = plot_tegrastats_power(Path(args.tegrastats).expanduser().resolve(), outdir)
        if ts_info:
            summary["tegrastats"] = ts_info

    save_json(summary, outdir / "metrics_summary.json")

    # Console hint
    print("Wrote figures to:", outdir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
