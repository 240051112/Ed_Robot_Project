#!/usr/bin/env python3
# Plot ED 25Q run: latency, power, util, temps + correlations & top-5 tables.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RUN_DIR = Path("runs/single")
CSV = RUN_DIR / "results.csv"
OUT = RUN_DIR / "plots"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# numeric coercion
num_cols = [
    "total_s","answer_len","sources_count","ctx_count",
    "avg_vdd_in_w","peak_vdd_in_w",
    "avg_cpu_gpu_cv_w","peak_cpu_gpu_cv_w",
    "avg_vdd_soc_w","peak_vdd_soc_w",
    "avg_gpu_pct","peak_gpu_pct",
    "avg_cpu_pct","peak_cpu_pct",
    "avg_temp_cpu_c","peak_temp_cpu_c",
    "avg_temp_gpu_c","peak_temp_gpu_c",
    "avg_temp_tj_c","peak_temp_tj_c",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# sort by question index (Q01..Q25)
def qnum(q):
    try:
        return int(str(q).strip().lstrip("Qq"))
    except Exception:
        return 0
df["qnum"] = df["qid"].apply(qnum)
df = df.sort_values("qnum").reset_index(drop=True)

# ---- Charts (1 plot per figure, no explicit colors) ----

# 1) Latency per question
plt.figure(figsize=(10, 4))
plt.bar(df["qnum"], df["total_s"])
plt.title("Latency per Question (s)")
plt.xlabel("Question #")
plt.ylabel("Seconds")
plt.tight_layout()
plt.savefig(OUT / "latency_per_question.png")
plt.close()

# 2) Board power (VDD_IN) average vs peak
plt.figure(figsize=(10, 4))
plt.plot(df["qnum"], df["avg_vdd_in_w"], marker="o", label="Avg")
plt.plot(df["qnum"], df["peak_vdd_in_w"], marker="o", label="Peak")
plt.title("Board Power (VDD_IN): Average vs Peak (W)")
plt.xlabel("Question #")
plt.ylabel("Watts")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "power_vdd_in.png")
plt.close()

# 3) Average utilization: GPU vs CPU
plt.figure(figsize=(10, 4))
plt.plot(df["qnum"], df["avg_gpu_pct"], marker="o", label="GPU avg %")
plt.plot(df["qnum"], df["avg_cpu_pct"], marker="o", label="CPU avg %")
plt.title("Average Utilization: GPU vs CPU")
plt.xlabel("Question #")
plt.ylabel("Percent")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "util_gpu_cpu.png")
plt.close()

# 4) Temperatures (avg)
plt.figure(figsize=(10, 4))
plt.plot(df["qnum"], df["avg_temp_cpu_c"], marker="o", label="CPU °C")
plt.plot(df["qnum"], df["avg_temp_gpu_c"], marker="o", label="GPU °C")
plt.plot(df["qnum"], df["avg_temp_tj_c"], marker="o", label="TJ °C")
plt.title("Average Temperatures")
plt.xlabel("Question #")
plt.ylabel("°C")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "temps_avg.png")
plt.close()

# 5) Latency vs Peak power (scatter)
plt.figure(figsize=(5, 5))
plt.scatter(df["peak_vdd_in_w"], df["total_s"])
plt.title("Latency vs Peak Power")
plt.xlabel("Peak VDD_IN (W)")
plt.ylabel("Latency (s)")
plt.tight_layout()
plt.savefig(OUT / "scatter_latency_vs_peak_power.png")
plt.close()

# 6) Correlations
corr_cols = [c for c in [
    "total_s","answer_len","ctx_count","sources_count",
    "avg_vdd_in_w","peak_vdd_in_w",
    "avg_gpu_pct","avg_cpu_pct",
    "avg_temp_cpu_c","avg_temp_gpu_c","avg_temp_tj_c",
] if c in df.columns]
corr = df[corr_cols].corr(numeric_only=True)
corr.to_csv(OUT / "correlations.csv", index=True)

# 7) Top-5 tables
top_latency = df.nlargest(5, "total_s")[["qid","mode","topic","total_s","avg_gpu_pct","avg_cpu_pct","peak_vdd_in_w"]]
top_peak    = df.nlargest(5, "peak_vdd_in_w")[["qid","mode","topic","peak_vdd_in_w","total_s","avg_gpu_pct","avg_cpu_pct"]]
top_tcpu    = df.nlargest(5, "avg_temp_cpu_c")[["qid","mode","topic","avg_temp_cpu_c","avg_temp_gpu_c","avg_temp_tj_c","total_s"]]

top_latency.to_csv(OUT / "top5_latency.csv", index=False)
top_peak.to_csv(OUT / "top5_peakpower.csv", index=False)
top_tcpu.to_csv(OUT / "top5_temps.csv", index=False)

# Also save an enriched CSV
df.to_csv(OUT / "results_enriched.csv", index=False)

print("\nSaved charts & tables in:", OUT)
for f in sorted(OUT.iterdir()):
    print(" -", f.name)
