"""
05_ndsi_threshold_validation.py
===============================
Validate NDSI-based snow detection against SNOTEL ground observations.

Two analyses in one script:
  Part A — NDSI distribution comparison (snow vs no-snow) with quantile
            lines and approximate ROC threshold.
  Part B — Per-year ROC-based threshold optimisation (Youden's J) with
            accuracy metrics (OA, UA, PA) and trend plot.

Ground truth binarisation:
  SWE > 0.4 inch  →  snow = 1   (≈ 1 cm water equivalent)
  SWE ≤ 0.4 inch  →  snow = 0

Inputs (CSV files):
  - Station observations  (columns: Date, Station Id, SWE …)
  - Remote-sensing values (columns: Date, Station Id / StationID,
                           NDSI, RED, above_snowline)

Author:      Gefei Wu
Affiliation: Zhejiang University
Date:        2026-02-21
License:     MIT
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# =====================================================================
# CONFIGURATION — update paths before running
# =====================================================================
CONFIG = {
    # TODO: Set paths to your local data directories
    "station_csv_pattern": "./data/station_obs/*.csv",
    "remote_csv_pattern":  "./data/remote_sensing/*.csv",

    # Ground-truth SWE binarisation threshold (inches)
    "swe_threshold": 0.4,

    # Satellite snow classification thresholds
    # (must match the Theia two-pass settings from scripts 02 / 03)
    "ndsi_pass1": 0.25,   # conservative, applied everywhere
    "ndsi_pass2": 0.09,   # relaxed, applied above snowline

    # Output figure paths
    "fig_distribution": "./figures/NDSI_Distribution_Validation.png",
    "fig_accuracy":     "./figures/Snow_Accuracy_Yearly.png",

    # Train/test split ratio (fraction used for training)
    "train_frac": 0.3,
}

# =====================================================================
# 1. Read & merge data
# =====================================================================
print("=== Reading input CSVs ===")

station_files = glob.glob(CONFIG["station_csv_pattern"])
rs_files      = glob.glob(CONFIG["remote_csv_pattern"])

if not station_files:
    raise FileNotFoundError(
        f"No station CSVs found at {CONFIG['station_csv_pattern']}")
if not rs_files:
    raise FileNotFoundError(
        f"No remote-sensing CSVs found at {CONFIG['remote_csv_pattern']}")

print(f"  Station files: {len(station_files)}")
print(f"  RS files:      {len(rs_files)}")

station_df = pd.concat([pd.read_csv(f) for f in station_files], ignore_index=True)
rs_df      = pd.concat([pd.read_csv(f) for f in rs_files],      ignore_index=True)

# Normalise column names (strip whitespace)
station_df.columns = station_df.columns.str.strip()
rs_df.columns      = rs_df.columns.str.strip()

# Harmonise date column
station_df["Date"] = pd.to_datetime(station_df["Date"])

if "date" in rs_df.columns:
    rs_df["Date"] = pd.to_datetime(rs_df["date"])
    rs_df.drop(columns=["date"], inplace=True)
elif "Date" in rs_df.columns:
    rs_df["Date"] = pd.to_datetime(rs_df["Date"])

# Harmonise station-ID column
if "StationID" in rs_df.columns and "Station Id" not in rs_df.columns:
    rs_df.rename(columns={"StationID": "Station Id"}, inplace=True)

# Identify the SWE column (SNOTEL naming convention)
swe_cols = [c for c in station_df.columns
            if "Snow Water Equivalent" in c and "Start of Day" in c]
if not swe_cols:
    # Fallback: any column containing "SWE"
    swe_cols = [c for c in station_df.columns if "SWE" in c.upper()]
if not swe_cols:
    raise ValueError(
        "Cannot find SWE column in station data.  "
        "Expected a column containing 'Snow Water Equivalent'.")
swe_col = swe_cols[0]
print(f"  SWE column: '{swe_col}'")

# Binarise ground truth
station_df["Snow_obs"] = (
    station_df[swe_col] > CONFIG["swe_threshold"]
).astype(int)

# Satellite-based snow flag (Theia two-pass logic)
# Pass 1: NDSI > 0.25  (conservative, applied everywhere)
# Pass 2: NDSI > 0.09  (relaxed, requires above_snowline > 0)
has_snowline_col = "above_snowline" in rs_df.columns
if has_snowline_col:
    rs_df["Snow_sat"] = (
        (rs_df["NDSI"] > CONFIG["ndsi_pass1"])
        | ((rs_df["above_snowline"] > 0) & (rs_df["NDSI"] > CONFIG["ndsi_pass2"]))
    ).astype(int)
else:
    # If above_snowline column is missing, use Pass 1 only
    print("  WARNING: 'above_snowline' column not found — using Pass 1 only")
    rs_df["Snow_sat"] = (rs_df["NDSI"] > CONFIG["ndsi_pass1"]).astype(int)

# Merge on Date + Station Id
merged = pd.merge(
    station_df[["Date", "Station Id", "Snow_obs"]],
    rs_df[["Date", "Station Id", "NDSI", "RED", "Snow_sat"]],
    on=["Date", "Station Id"],
    how="inner",
).dropna(subset=["NDSI"])

merged["Year"] = merged["Date"].dt.year
print(f"  Merged sample count: {len(merged):,}")
print(f"  Years: {sorted(merged['Year'].unique())}")

# Ensure output directories exist
for key in ("fig_distribution", "fig_accuracy"):
    os.makedirs(os.path.dirname(CONFIG[key]) or ".", exist_ok=True)

# =====================================================================
# Part A — NDSI distribution plot
# =====================================================================
print("\n=== Part A: NDSI distribution ===")

nosnow = merged.loc[merged["Snow_obs"] == 0, "NDSI"].dropna()
snow   = merged.loc[merged["Snow_obs"] == 1, "NDSI"].dropna()

q90_ns = np.percentile(nosnow, 90)
q95_ns = np.percentile(nosnow, 95)
q05_sn = np.percentile(snow, 5)
q10_sn = np.percentile(snow, 10)

# Approximate optimal threshold (midpoint of class means)
best_roc_ndsi = (nosnow.mean() + snow.mean()) / 2

print(f"  No-snow samples: {len(nosnow):,}   mean NDSI = {nosnow.mean():.3f}")
print(f"  Snow samples:    {len(snow):,}   mean NDSI = {snow.mean():.3f}")
print(f"  Approximate ROC threshold (midpoint): {best_roc_ndsi:.3f}")

# Plotting
sns.set_style("white")
sns.set_context("paper", font_scale=1.2)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

COLOR_NS = "#70A5D9"
COLOR_SN = "#F0868C"
LINE_NS  = "#2E69A1"
LINE_SN  = "#B64557"

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(nosnow, bins=50, stat="density", alpha=0.45,
             label="No Snow (SWE ≤ 0.4 in)", color=COLOR_NS, ax=ax)
sns.histplot(snow,   bins=50, stat="density", alpha=0.45,
             label="Snow (SWE > 0.4 in)",    color=COLOR_SN, ax=ax)

ax.axvline(q90_ns, color=LINE_NS, ls="--", lw=2,
           label=f"No-Snow 90th = {q90_ns:.3f}")
ax.axvline(q95_ns, color=LINE_NS, ls="-",  lw=2,
           label=f"No-Snow 95th = {q95_ns:.3f}")
ax.axvline(q05_sn, color=LINE_SN, ls="-",  lw=2,
           label=f"Snow 5th = {q05_sn:.3f}")
ax.axvline(q10_sn, color=LINE_SN, ls="--", lw=2,
           label=f"Snow 10th = {q10_sn:.3f}")
ax.axvline(best_roc_ndsi, color="#6E6E6E", ls="-", lw=2.4,
           label=f"Approx. ROC = {best_roc_ndsi:.3f}")

ax.set_xlabel("NDSI", fontsize=14, fontweight="bold")
ax.set_ylabel("Density", fontsize=14, fontweight="bold")
ax.set_title("NDSI Distribution — Snow vs No Snow "
             f"(SWE threshold = {CONFIG['swe_threshold']} in)",
             fontsize=16, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", frameon=True, facecolor="white",
          framealpha=0.7, edgecolor="black", fontsize=11)
ax.text(0.02, 0.06, f"Approx. ROC NDSI = {best_roc_ndsi:.3f}",
        transform=ax.transAxes, ha="left", va="center", fontsize=13,
        fontweight="bold", color="black",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black",
                  boxstyle="round,pad=0.3"))

plt.tight_layout()
plt.savefig(CONFIG["fig_distribution"], dpi=300)
print(f"  Saved: {CONFIG['fig_distribution']}")

# =====================================================================
# Part B — Per-year ROC validation
# =====================================================================
print("\n=== Part B: yearly ROC validation ===")

results_list = []

for year, grp in merged.groupby("Year"):
    train = grp.sample(frac=CONFIG["train_frac"], random_state=42)
    test  = grp.drop(train.index)

    y_train    = train["Snow_obs"].values
    ndsi_train = train["NDSI"].values

    # Sweep NDSI thresholds on training set, maximise Youden's J
    thresholds = np.linspace(
        max(ndsi_train.min(), -1), min(ndsi_train.max(), 1), 200
    )
    best_J, best_ndsi = -1, 0.25  # default fallback

    for t in thresholds:
        y_pred = (ndsi_train >= t).astype(int)
        cm_t = confusion_matrix(y_train, y_pred, labels=[0, 1])
        if cm_t.shape == (2, 2):
            tn, fp, fn, tp = cm_t.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            J = tpr - fpr
            if J > best_J:
                best_J, best_ndsi = J, t

    # Evaluate on test set
    y_test    = test["Snow_obs"].values
    y_pred_ts = (test["NDSI"].values >= best_ndsi).astype(int)
    cm = confusion_matrix(y_test, y_pred_ts, labels=[0, 1])
    if cm.shape != (2, 2):
        tmp = np.zeros((2, 2), dtype=int)
        tmp[: cm.shape[0], : cm.shape[1]] = cm
        cm = tmp
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    total = cm.sum()
    results_list.append({
        "Year":       year,
        "Train_size": len(train),
        "Test_size":  len(test),
        "Best_NDSI":  round(best_ndsi, 4),
        "Youden_J":   round(best_J, 4),
        "OA":         (TP + TN) / total if total else np.nan,
        "UA_snow":    TP / (TP + FP) if (TP + FP) else np.nan,
        "UA_nonsnow": TN / (TN + FN) if (TN + FN) else np.nan,
        "PA_snow":    TP / (TP + FN) if (TP + FN) else np.nan,
        "PA_nonsnow": TN / (TN + FP) if (TN + FP) else np.nan,
    })

results_df = pd.DataFrame(results_list).sort_values("Year")
print(results_df.to_string(index=False))

# --- Accuracy trend plot ---
years   = results_df["Year"].values
OA      = results_df["OA"].values
UA_snow = results_df["UA_snow"].values
PA_snow = results_df["PA_snow"].values
mean_OA = np.nanmean(OA)

palette = ["#299D8F", "#E9C46A", "#D87659"]

sns.set_style("white")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("#F7F7F7")

ax.plot(years, OA,      marker="o", color=palette[0], lw=2.5, ms=7, alpha=0.85,
        label="Overall Accuracy (OA)")
ax.plot(years, UA_snow, marker="s", color=palette[1], lw=2.5, ms=7, alpha=0.85,
        label="User Accuracy — Snow (UA)")
ax.plot(years, PA_snow, marker="^", color=palette[2], lw=2.5, ms=7, alpha=0.85,
        label="Producer Accuracy — Snow (PA)")

ax.set_xlabel("Year",     fontsize=14, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
ax.set_title("Snow Classification Accuracy (Test Set, "
             f"SWE > {CONFIG['swe_threshold']} in)",
             fontsize=16, fontweight="bold")
ax.set_ylim(0, 1.05)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.grid(True, ls="-", color="white", lw=1)
ax.legend(fontsize=12, loc="lower right", frameon=False)
ax.text(0.02, 0.05, f"Mean OA = {mean_OA:.3f}",
        transform=ax.transAxes, fontsize=14, fontweight="bold",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none",
                  boxstyle="round,pad=0.3"))

plt.tight_layout()
plt.savefig(CONFIG["fig_accuracy"], dpi=300)
print(f"\nSaved: {CONFIG['fig_accuracy']}")
plt.show()
