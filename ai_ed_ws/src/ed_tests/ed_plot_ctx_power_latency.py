#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _trendline(ax, x, y, label="trend"):
    # simple least-squares line if enough points
    if len(x) >= 2 and np.isfinite(x).all() and np.isfinite(y).all():
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, m*xs + b, linestyle="--", linewidth=1.0, label=label)

def bubble(ax, x, y, size, labels=None, title="", xlabel="", ylabel="", size_label=""):
    # scale sizes to reasonable marker areas
    s = np.array(size, dtype=float)
    s = np.clip(s, np.nanpercentile(s, 5), np.nanpercentile(s, 95))
    s_scaled = 50.0 * (s / np.nanmax(s))**1.0 + 10.0  # avoid zeros
    sc = ax.scatter(x, y, s=s_scaled, alpha=0.7, edgecolor="k", linewidth=0.3)
    _trendline(ax, x, y, label="trend")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # annotate top-3 latency points for quick eyeballing, if labels provided
    if labels is not None and len(labels) == len(x):
        top_idx = np.argsort(-y)[:3]
        for i in top_idx:
            ax.annotate(str(labels[i]), (x[i], y[i]),
                        textcoords="offset points", xytext=(5,5), fontsize=8)
    # legend note for bubble meaning
    if size_label:
        ax.text(0.99, 0.02, f"Bubble ~ {size_label}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, alpha=0.9, bbox=dict(fc="white", ec="0.7", lw=0.5))

def dual_axis_latency_power(ax, qnum, latency_s, power_w):
    ax.plot(qnum, latency_s, marker="o")
    ax.set_xlabel("Question #")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Latency & Average Power by Question")
    ax2 = ax.twinx()
    ax2.plot(qnum, power_w, marker="s")
    ax2.set_ylabel("Avg VDD_IN Power (W)")
    ax.grid(True, alpha=0.3)

def corr_heatmap(ax, df, cols, title="Correlation"):
    sub = df[cols].copy()
    cmat = sub.corr().values
    im = ax.imshow(cmat, vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)
    # annotate cells
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cmat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def main():
    ap = argparse.ArgumentParser(description="Plot latency vs. context vs. power (ED run).")
    ap.add_argument("--csv", required=True,
                    help="Path to results_enriched.csv (has total_s, ctx_count, sources_count, avg/peak VDD_IN, etc.)")
    ap.add_argument("--out", required=True,
                    help="Output directory for figures and summaries")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir  = Path(args.out).expanduser().resolve()
    ensure_outdir(out_dir)

    df = pd.read_csv(csv_path)
    # Column aliases for robustness
    # latency
    if "total_s" in df.columns:
        latency = df["total_s"].astype(float)
    elif "latency_ms" in df.columns:
        latency = df["latency_ms"].astype(float) / 1000.0
    else:
        raise SystemExit("Could not find latency column (need total_s or latency_ms).")

    # context/citation proxies
    ctx = df["ctx_count"] if "ctx_count" in df.columns else pd.Series([np.nan]*len(df))
    src = df["sources_count"] if "sources_count" in df.columns else pd.Series([np.nan]*len(df))

    # power metrics (VDD_IN is the board input rail)
    p_avg = df["avg_vdd_in_w"] if "avg_vdd_in_w" in df.columns else pd.Series([np.nan]*len(df))
    p_peak= df["peak_vdd_in_w"] if "peak_vdd_in_w" in df.columns else pd.Series([np.nan]*len(df))

    # labels (question number)
    if "qnum" in df.columns:
        labels = df["qnum"].astype(int).tolist()
        qnum   = df["qnum"].astype(int).values
    else:
        labels = list(range(1, len(df)+1))
        qnum   = np.arange(1, len(df)+1)

    # ===== Plots =====
    # 1) ctx_count vs latency, bubble = peak power
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.0))
    bubble(ax1,
           x=ctx.values.astype(float),
           y=latency.values.astype(float),
           size=p_peak.values.astype(float),
           labels=labels,
           title="Latency vs Context Count (bubble = Peak Power)",
           xlabel="Context count (chunks fed to LLM)",
           ylabel="Latency (s)",
           size_label="peak VDD_IN (W)")
    fig1.tight_layout()
    fig1.savefig(out_dir / "latency_vs_ctx_bubble_peakpower.png", dpi=180)
    plt.close(fig1)

    # 2) sources_count vs latency, bubble = avg power
    fig2, ax2 = plt.subplots(figsize=(7.5, 5.0))
    bubble(ax2,
           x=src.values.astype(float),
           y=latency.values.astype(float),
           size=p_avg.values.astype(float),
           labels=labels,
           title="Latency vs Sources Count (bubble = Avg Power)",
           xlabel="Sources cited in answer",
           ylabel="Latency (s)",
           size_label="avg VDD_IN (W)")
    fig2.tight_layout()
    fig2.savefig(out_dir / "latency_vs_sources_bubble_avgpower.png", dpi=180)
    plt.close(fig2)

    # 3) dual-axis line: question # vs latency & avg power
    fig3, ax3 = plt.subplots(figsize=(8.0, 4.5))
    dual_axis_latency_power(ax3, qnum, latency.values.astype(float), p_avg.values.astype(float))
    fig3.tight_layout()
    fig3.savefig(out_dir / "latency_and_avgpower_by_question.png", dpi=180)
    plt.close(fig3)

    # 4) correlation heatmap
    corr_cols = []
    for c in ["total_s", "ctx_count", "sources_count", "avg_vdd_in_w", "peak_vdd_in_w"]:
        if c in df.columns:
            corr_cols.append(c)
    if "latency_ms" in df.columns and "total_s" not in corr_cols:
        corr_cols.append("latency_ms")
    if len(corr_cols) >= 2:
        fig4, ax4 = plt.subplots(figsize=(6.0, 5.0))
        corr_heatmap(ax4, df, corr_cols, title="Correlations: latency, context, power")
        fig4.tight_layout()
        fig4.savefig(out_dir / "corr_latency_context_power.png", dpi=180)
        plt.close(fig4)

    # ===== Summaries =====
    metrics = {
        "latency_s": {
            "mean": float(np.nanmean(latency)),
            "p50":  float(np.nanpercentile(latency, 50)),
            "p90":  float(np.nanpercentile(latency, 90)),
            "min":  float(np.nanmin(latency)),
            "max":  float(np.nanmax(latency)),
            "count": int(np.sum(np.isfinite(latency))),
        },
        "ctx_count": {
            "mean": float(np.nanmean(ctx)),
            "min":  float(np.nanmin(ctx)),
            "max":  float(np.nanmax(ctx)),
        },
        "sources_count": {
            "mean": float(np.nanmean(src)),
            "min":  float(np.nanmin(src)),
            "max":  float(np.nanmax(src)),
        },
        "power_w": {
            "avg_vdd_in_mean": float(np.nanmean(p_avg)),
            "peak_vdd_in_mean": float(np.nanmean(p_peak)),
        }
    }
    with open(out_dir / "metrics_ctx_power_latency.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # correlations table
    corr_pairs = []
    def add_corr(a, b, name_a, name_b):
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() >= 3:
            r = float(np.corrcoef(a[m], b[m])[0,1])
            corr_pairs.append({"var_x": name_a, "var_y": name_b, "pearson_r": r, "n": int(m.sum())})
    add_corr(latency, ctx, "latency_s", "ctx_count")
    add_corr(latency, src, "latency_s", "sources_count")
    add_corr(latency, p_avg, "latency_s", "avg_vdd_in_w")
    add_corr(latency, p_peak, "latency_s", "peak_vdd_in_w")

    pd.DataFrame(corr_pairs).to_csv(out_dir / "correlations_latency_ctx_power.csv", index=False)

    print(f"Wrote figures to: {out_dir}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
