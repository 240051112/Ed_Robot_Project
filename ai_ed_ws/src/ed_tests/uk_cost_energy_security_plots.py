#!/usr/bin/env python3
"""
UK-focused Cost/Energy/Security plots for ED runs.

Inputs:
  runs/single/results.csv  (created by test_ed_25.sh)

Outputs (in runs/single/plots_uk/):
  - 01_energy_per_question_wh.png
  - 02_cost_per_question_gbp.png
  - 03_security_events_per_question.png
  - 04_security_risk_score.png
  - 05_energy_vs_latency.png
  - energy_summary.csv
  - cost_summary.csv
  - security_summary.csv
  - results_with_energy_cost_gbp.csv

Tunables via env:
  GBP_PER_KWH   (default 0.30)  # UK electricity price in £/kWh
  PSU_EFF       (default 0.92)  # PSU efficiency (0..1)
  WATER_DT_C    (default 0)     # Cooling ΔT in °C for water estimate (0 disables)
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

RUN_DIR = Path("runs/single")
CSV = RUN_DIR / "results.csv"
OUT = RUN_DIR / "plots_uk"
OUT.mkdir(parents=True, exist_ok=True)

# --- UK defaults (override with env) ---
GBP_PER_KWH = float(os.getenv("GBP_PER_KWH", "0.30"))  # £/kWh
PSU_EFF = float(os.getenv("PSU_EFF", "0.92"))          # 0..1
WATER_DT_C = float(os.getenv("WATER_DT_C", "0"))       # e.g., 10 for 10°C rise

# --- Load ---
df = pd.read_csv(CSV)

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
    "url_count","pii_count",
]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

def qnum(q):
    try:
        return int(str(q).strip().lstrip("Qq"))
    except Exception:
        return 0
df["qnum"] = df["qid"].apply(qnum)
df = df.sort_values("qnum").reset_index(drop=True)

# --- Energy + Cost (GBP) ---
# Device energy (J) ~ avg_board_power (W) * latency (s)
df["energy_device_j"] = df["avg_vdd_in_w"] * df["total_s"]
# Wall energy accounts for PSU efficiency
df["energy_wall_j"] = df["energy_device_j"] / max(PSU_EFF, 1e-6)
df["energy_wall_wh"] = df["energy_wall_j"] / 3600.0
# Cost in £
df["cost_gbp"] = df["energy_wall_wh"] * (GBP_PER_KWH / 1000.0)

# Optional cooling-water estimate (litres) for ΔT
if WATER_DT_C > 0:
    # water grams = Joules / (4.186 J/g°C * ΔT)
    df["water_g"] = df["energy_wall_j"] / (4.186 * WATER_DT_C)
    df["water_l"] = df["water_g"] / 1000.0
else:
    df["water_g"] = 0.0
    df["water_l"] = 0.0

# --- Security risk (weighted) ---
df["security_risk"] = 3*df.get("pii_count", 0).fillna(0) + 1*df.get("url_count", 0).fillna(0)

# --- Helpers: formatting, savefig ---
gbp_fmt = FuncFormatter(lambda x, pos: f"£{x:,.4f}")
wh_fmt  = FuncFormatter(lambda x, pos: f"{x:,.3f} Wh")
s_fmt   = FuncFormatter(lambda x, pos: f"{x:,.0f}s")

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

# --- 01: Energy per question (Wh, wall) ---
plt.figure(figsize=(11,4))
plt.bar(df["qnum"], df["energy_wall_wh"])
plt.title(f"Energy per Question (Wall) — Mean {df['energy_wall_wh'].mean():.3f} Wh, "
          f"Total {df['energy_wall_wh'].sum():.3f} Wh (η={PSU_EFF:.0%})")
plt.xlabel("Question #")
plt.ylabel("Energy (Wh)")
ax = plt.gca()
ax.yaxis.set_major_formatter(wh_fmt)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
savefig(OUT / "01_energy_per_question_wh.png")

# --- 02: Cost per question (GBP) ---
plt.figure(figsize=(11,4))
plt.bar(df["qnum"], df["cost_gbp"])
plt.title(f"Cost per Question — Mean £{df['cost_gbp'].mean():.5f} | "
          f"Total £{df['cost_gbp'].sum():.5f}  (price={GBP_PER_KWH:.2f} £/kWh)")
plt.xlabel("Question #")
plt.ylabel("Cost (£)")
ax = plt.gca()
ax.yaxis.set_major_formatter(gbp_fmt)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
savefig(OUT / "02_cost_per_question_gbp.png")

# --- 03: Security events (URLs & PII) ---
plt.figure(figsize=(11,4))
plt.plot(df["qnum"], df["url_count"], marker="o", label="URL count")
plt.plot(df["qnum"], df["pii_count"], marker="o", label="PII count")
plt.title("Security Events per Question (URLs & PII)")
plt.xlabel("Question #")
plt.ylabel("Count")
plt.legend()
savefig(OUT / "03_security_events_per_question.png")

# --- 04: Security risk score ---
plt.figure(figsize=(11,4))
plt.bar(df["qnum"], df["security_risk"])
plt.title("Security Risk Score (3×PII + 1×URL)")
plt.xlabel("Question #")
plt.ylabel("Risk score")
savefig(OUT / "04_security_risk_score.png")

# --- 05: Energy vs Latency ---
plt.figure(figsize=(6,6))
plt.scatter(df["total_s"], df["energy_wall_wh"])
plt.title("Latency vs Energy (Wall)")
plt.xlabel("Latency (s)")
plt.ylabel("Energy (Wh)")
ax = plt.gca()
ax.xaxis.set_major_formatter(s_fmt)
ax.yaxis.set_major_formatter(wh_fmt)
savefig(OUT / "05_energy_vs_latency.png")

# --- Summaries ---
energy_summary = pd.DataFrame({
    "total_prompts":[len(df)],
    "sum_energy_wall_Wh":[df["energy_wall_wh"].sum()],
    "avg_energy_wall_Wh":[df["energy_wall_wh"].mean()],
    "median_energy_wall_Wh":[df["energy_wall_wh"].median()],
    "psu_efficiency":[PSU_EFF],
})
cost_summary = pd.DataFrame({
    "total_cost_gbp":[df["cost_gbp"].sum()],
    "avg_cost_gbp":[df["cost_gbp"].mean()],
    "median_cost_gbp":[df["cost_gbp"].median()],
    "gbp_per_kwh":[GBP_PER_KWH],
})
security_summary = pd.DataFrame({
    "sum_urls":[df["url_count"].sum()],
    "sum_pii":[df["pii_count"].sum()],
    "sum_security_risk":[df["security_risk"].sum()],
    "avg_security_risk":[df["security_risk"].mean()],
    "max_security_risk":[df["security_risk"].max()],
})

energy_summary.to_csv(OUT / "energy_summary.csv", index=False)
cost_summary.to_csv(OUT / "cost_summary.csv", index=False)
security_summary.to_csv(OUT / "security_summary.csv", index=False)

# Enriched table
df.to_csv(OUT / "results_with_energy_cost_gbp.csv", index=False)

print("\nSaved to:", OUT)
for f in sorted(OUT.iterdir()):
    print(" -", f.name)
