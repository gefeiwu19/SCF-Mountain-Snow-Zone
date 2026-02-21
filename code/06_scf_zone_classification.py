"""
06_scf_zone_classification.py
=============================
Determine optimal Snow Cover Frequency (SCF) thresholds for
classifying mountain pixels into three snow zones:

  Non-snow zone     :  SCF < threshold_1
  Seasonal snow zone:  threshold_1 <= SCF < threshold_2
  Permanent snow zone:  SCF >= threshold_2

Three independent methods are compared:
  1. Modified Otsu dual-threshold (adapted for discontinuous data)
  2. Region-wise K-means clustering
  3. Cumulative-area adaptive threshold

A sensitivity analysis across threshold combinations is also produced.

Input:  CSV files with frequency-bin columns (e.g. freq_0_5, freq_5_10 …)
        exported from GEE SCF analysis.

Output:
  - 4-panel figure: full distribution, low/high detail, pie chart
  - Threshold comparison CSV
  - Sensitivity analysis CSV

Author:      Gefei Wu
Affiliation: Zhejiang University
Date:        2026-02-21
License:     MIT
"""

import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURATION
# =====================================================================
CONFIG = {
    # TODO: Update to your local data path
    "data_path": "./data/scf_frequency_bins/*.csv",

    # Gap range where no SCF data exists (sparse transition zone)
    "gap_start": 30,   # %
    "gap_end":   70,   # %

    # Output files
    "fig_output":       "./figures/SCF_Threshold_Analysis.png",
    "csv_thresholds":   "./results/Threshold_Analysis_Results.csv",
    "csv_sensitivity":  "./results/Sensitivity_Analysis.csv",

    # Sensitivity test ranges
    "test_t1": [5, 10, 15, 20, 25, 30],
    "test_t2": [70, 75, 80, 85, 90, 95],
}

# Font settings (consistent with other scripts)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

GAP_START = CONFIG["gap_start"]
GAP_END   = CONFIG["gap_end"]

# =====================================================================
# 1. Read and combine CSV data
# =====================================================================
csv_files = glob.glob(CONFIG["data_path"])
print(f"Found {len(csv_files)} CSV files")
if not csv_files:
    raise FileNotFoundError(f"No CSVs at {CONFIG['data_path']}")

combined_df = pd.concat(
    [pd.read_csv(f) for f in csv_files], ignore_index=True
)
print(f"Total records: {len(combined_df):,}")

# Ensure output directories exist
for key in ("fig_output", "csv_thresholds", "csv_sensitivity"):
    os.makedirs(os.path.dirname(CONFIG[key]) or ".", exist_ok=True)

# =====================================================================
# 2. Parse frequency-bin columns
# =====================================================================
freq_columns = [c for c in combined_df.columns if c.startswith("freq_")]
print(f"Frequency columns ({len(freq_columns)}): {freq_columns[:5]} …")


def parse_freq_range(col_name):
    """Extract (low, high, centre) from column name like 'freq_0_5'."""
    parts = col_name.replace("freq_", "").split("_")
    low, high = int(parts[0]), int(parts[1])
    return low, high, (low + high) / 2


freq_info = {}
for col in freq_columns:
    lo, hi, ctr = parse_freq_range(col)
    freq_info[col] = {"low": lo, "high": hi, "center": ctr}

sorted_freq_cols = sorted(freq_columns, key=lambda x: freq_info[x]["center"])

low_freq_cols  = [c for c in sorted_freq_cols if freq_info[c]["high"] <= GAP_START]
high_freq_cols = [c for c in sorted_freq_cols if freq_info[c]["low"]  >= GAP_END]

# Area totals per bin
freq_values = np.array([freq_info[c]["center"] for c in sorted_freq_cols])
area_values = np.array([combined_df[c].sum() for c in sorted_freq_cols])
total_area  = area_values.sum()

low_total  = sum(combined_df[c].sum() for c in low_freq_cols)
high_total = sum(combined_df[c].sum() for c in high_freq_cols)

print(f"\nLow-freq  (0–{GAP_START}%) total area : {low_total:,.0f}"
      f"  ({low_total / total_area * 100:.1f}%)")
print(f"High-freq ({GAP_END}–100%) total area: {high_total:,.0f}"
      f"  ({high_total / total_area * 100:.1f}%)")

# =====================================================================
# 3. Method 1 — Modified Otsu dual-threshold
# =====================================================================

def otsu_dual_threshold(freq_cols, freq_info_, df, gap_start, gap_end):
    """
    Find dual thresholds maximising between-class variance.

    Adapted for SCF data with a discontinuous gap region:
    threshold_1 is searched in [0, gap_start] boundaries,
    threshold_2 is searched in [gap_end, 100] boundaries.
    """
    low_cols_  = [c for c in freq_cols if freq_info_[c]["high"] <= gap_start]
    high_cols_ = [c for c in freq_cols if freq_info_[c]["low"]  >= gap_end]
    areas = {c: df[c].sum() for c in freq_cols}
    total = sum(areas.values())

    t1_cands = sorted({gap_start} | {freq_info_[c]["high"] for c in low_cols_})
    t2_cands = sorted({gap_end}   | {freq_info_[c]["low"]  for c in high_cols_})

    best_t1, best_t2, max_var = gap_start, gap_end, 0

    for t1 in t1_cands:
        for t2 in t2_cands:
            if t1 >= t2:
                continue
            w0 = sum(areas[c] for c in freq_cols if freq_info_[c]["high"] <= t1)
            w2 = sum(areas[c] for c in freq_cols if freq_info_[c]["low"]  >= t2)
            w1 = total - w0 - w2
            if 0 in (w0, w1, w2):
                continue

            mu0 = sum(freq_info_[c]["center"] * areas[c]
                      for c in freq_cols if freq_info_[c]["high"] <= t1) / w0
            mu2 = sum(freq_info_[c]["center"] * areas[c]
                      for c in freq_cols if freq_info_[c]["low"]  >= t2) / w2
            mu1 = (t1 + t2) / 2   # midpoint for gap region
            mu_t = (w0 * mu0 + w1 * mu1 + w2 * mu2) / total
            var  = (w0 / total * (mu0 - mu_t) ** 2
                  + w1 / total * (mu1 - mu_t) ** 2
                  + w2 / total * (mu2 - mu_t) ** 2)
            if var > max_var:
                max_var, best_t1, best_t2 = var, t1, t2

    return (best_t1, best_t2), max_var


otsu_thresh, _ = otsu_dual_threshold(
    sorted_freq_cols, freq_info, combined_df, GAP_START, GAP_END
)
print(f"\nMethod 1 — Modified Otsu:")
print(f"  Threshold 1 (non-snow / seasonal): {otsu_thresh[0]}%")
print(f"  Threshold 2 (seasonal / permanent): {otsu_thresh[1]}%")


# =====================================================================
# 4. Method 2 — Region-wise K-means
# =====================================================================

def kmeans_by_region(freq_cols, freq_info_, df, gap_start, gap_end):
    """
    K-means (k=2) within low-freq and high-freq regions separately.
    The midpoint between two cluster centres becomes the threshold.
    """
    results = {}
    for tag, cols, default in [
        ("thresh1",
         [c for c in freq_cols if freq_info_[c]["high"] <= gap_start],
         gap_start),
        ("thresh2",
         [c for c in freq_cols if freq_info_[c]["low"]  >= gap_end],
         gap_end),
    ]:
        if len(cols) >= 2:
            data = []
            for c in cols:
                area = df[c].sum()
                # Replicate centre values proportionally to area
                data.extend([freq_info_[c]["center"]] * max(1, int(area / 1e6)))
            if len(set(data)) >= 2:
                km = KMeans(n_clusters=2, random_state=42, n_init=10)
                km.fit(np.array(data).reshape(-1, 1))
                centres = sorted(km.cluster_centers_.flatten())
                results[tag] = (centres[0] + centres[1]) / 2
                results[f"{tag}_centres"] = centres
        results.setdefault(tag, default)
    return results


km_res = kmeans_by_region(sorted_freq_cols, freq_info, combined_df,
                          GAP_START, GAP_END)
print(f"\nMethod 2 — Region-wise K-means:")
print(f"  Threshold 1: {km_res['thresh1']:.1f}%")
print(f"  Threshold 2: {km_res['thresh2']:.1f}%")


# =====================================================================
# 5. Method 3 — Cumulative-area adaptive threshold
# =====================================================================

def adaptive_threshold(freq_cols, freq_info_, df, gap_start, gap_end):
    """
    Find the 50th-percentile-area boundary within each region.
    The boundary that splits each region into two equal-area halves.
    """
    def _find_50pct(cols, ascending=True):
        items = [(freq_info_[c]["high" if ascending else "low"], df[c].sum())
                 for c in cols]
        items.sort(key=lambda x: x[0], reverse=(not ascending))
        total = sum(a for _, a in items)
        cum, thresh = 0, items[-1][0]
        for boundary, area in items:
            cum += area
            if cum >= total * 0.5:
                thresh = boundary
                break
        return thresh

    low_cols_  = [c for c in freq_cols if freq_info_[c]["high"] <= gap_start]
    high_cols_ = [c for c in freq_cols if freq_info_[c]["low"]  >= gap_end]
    return _find_50pct(low_cols_, True), _find_50pct(high_cols_, False)


adapt_t1, adapt_t2 = adaptive_threshold(
    sorted_freq_cols, freq_info, combined_df, GAP_START, GAP_END
)
print(f"\nMethod 3 — Cumulative-area adaptive:")
print(f"  Threshold 1: {adapt_t1}%")
print(f"  Threshold 2: {adapt_t2}%")


# =====================================================================
# 6. Summary & recommended thresholds
# =====================================================================
methods = {
    "Modified Otsu":     otsu_thresh,
    "Region K-means":    (km_res["thresh1"], km_res["thresh2"]),
    "Area adaptive":     (adapt_t1, adapt_t2),
}

all_t1 = [t[0] for t in methods.values()]
all_t2 = [t[1] for t in methods.values()]
rec_t1 = float(np.median(all_t1))
rec_t2 = float(np.median(all_t2))

print("\n" + "=" * 55)
print("Threshold comparison")
print("=" * 55)
for name, (t1, t2) in methods.items():
    print(f"  {name:20s}  T1={t1:5.1f}%   T2={t2:5.1f}%")
print(f"  {'Recommended (median)':20s}  T1={rec_t1:5.1f}%   T2={rec_t2:5.1f}%")


# =====================================================================
# 7. Classification function
# =====================================================================

def classify(freq_cols, freq_info_, df, t1, t2):
    """Classify total area into three snow zones using given thresholds."""
    non_snow = seasonal = permanent = 0
    for c in freq_cols:
        area = df[c].sum()
        if freq_info_[c]["high"] <= t1:
            non_snow += area
        elif freq_info_[c]["low"] >= t2:
            permanent += area
        else:
            seasonal += area
    total = non_snow + seasonal + permanent
    return {
        "Non-snow":  (non_snow,  non_snow  / total * 100 if total else 0),
        "Seasonal":  (seasonal,  seasonal  / total * 100 if total else 0),
        "Permanent": (permanent, permanent / total * 100 if total else 0),
    }


stats = classify(sorted_freq_cols, freq_info, combined_df, rec_t1, rec_t2)
print(f"\nClassification with recommended thresholds "
      f"({rec_t1:.0f}%, {rec_t2:.0f}%):")
for cat, (area, pct) in stats.items():
    print(f"  {cat:12s}: area = {area:,.0f}  ({pct:.1f}%)")


# =====================================================================
# 8. Sensitivity analysis
# =====================================================================
print("\n" + "=" * 55)
print("Sensitivity analysis")
print("=" * 55)

test_t1 = CONFIG["test_t1"]
test_t2 = CONFIG["test_t2"]

sens = []
print(f"{'T1':>5} {'T2':>5} {'Non-snow%':>10} {'Seasonal%':>10} {'Permanent%':>10}")
for t1 in test_t1:
    for t2 in test_t2:
        r = classify(sorted_freq_cols, freq_info, combined_df, t1, t2)
        print(f"{t1:5.0f} {t2:5.0f} {r['Non-snow'][1]:10.2f} "
              f"{r['Seasonal'][1]:10.2f} {r['Permanent'][1]:10.2f}")
        sens.append({"thresh1": t1, "thresh2": t2,
                     "non_snow_pct": r["Non-snow"][1],
                     "seasonal_pct": r["Seasonal"][1],
                     "permanent_pct": r["Permanent"][1]})


# =====================================================================
# 9. Visualisation — 4-panel figure
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)

# (a) Full frequency distribution
ax = axes[0, 0]
ax.bar(freq_values, area_values, width=4, alpha=0.7,
       color="steelblue", edgecolor="black", linewidth=0.5)
ax.axvline(rec_t1, color="red",   ls="--", lw=2, label=f"T1: {rec_t1:.1f}%")
ax.axvline(rec_t2, color="green", ls="--", lw=2, label=f"T2: {rec_t2:.1f}%")
ax.axvspan(GAP_START, GAP_END, alpha=0.15, color="gray",
           label=f"Data gap ({GAP_START}–{GAP_END}%)")
ax.set_xlabel("SCF (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Area", fontsize=12, fontweight="bold")
ax.set_title("(a) SCF Frequency Distribution", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) Low-frequency detail
ax = axes[0, 1]
lf_vals  = [freq_info[c]["center"] for c in low_freq_cols]
lf_areas = [combined_df[c].sum()   for c in low_freq_cols]
ax.bar(lf_vals, lf_areas, width=4, alpha=0.7,
       color="coral", edgecolor="black", linewidth=0.5)
ax.axvline(rec_t1, color="red", ls="--", lw=2, label=f"T1: {rec_t1:.1f}%")
ax.set_xlabel("SCF (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Area", fontsize=12, fontweight="bold")
ax.set_title(f"(b) Low-frequency Range (0–{GAP_START}%)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (c) High-frequency detail
ax = axes[1, 0]
hf_vals  = [freq_info[c]["center"] for c in high_freq_cols]
hf_areas = [combined_df[c].sum()   for c in high_freq_cols]
ax.bar(hf_vals, hf_areas, width=4, alpha=0.7,
       color="lightgreen", edgecolor="black", linewidth=0.5)
ax.axvline(rec_t2, color="green", ls="--", lw=2, label=f"T2: {rec_t2:.1f}%")
ax.set_xlabel("SCF (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Area", fontsize=12, fontweight="bold")
ax.set_title(f"(c) High-frequency Range ({GAP_END}–100%)",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (d) Pie chart
ax = axes[1, 1]
sizes  = [stats[c][1] for c in ["Non-snow", "Seasonal", "Permanent"]]
colors = ["#ff9999", "#66b3ff", "#99ff99"]
ax.pie(sizes, explode=(0.02, 0.02, 0.02),
       labels=["Non-snow", "Seasonal", "Permanent"],
       colors=colors, autopct="%1.1f%%", shadow=True, startangle=90,
       textprops={"fontsize": 12, "fontweight": "bold"})
ax.set_title(f"(d) Classification (T1={rec_t1:.0f}%, T2={rec_t2:.0f}%)",
             fontsize=14, fontweight="bold")

plt.tight_layout()
plt.savefig(CONFIG["fig_output"], dpi=300, bbox_inches="tight",
            facecolor="white")
print(f"\nSaved: {CONFIG['fig_output']}")


# =====================================================================
# 10. Save results
# =====================================================================
results_df = pd.DataFrame({
    "Method":      list(methods.keys()) + ["Recommended (median)"],
    "Threshold_1": list(all_t1) + [rec_t1],
    "Threshold_2": list(all_t2) + [rec_t2],
})
results_df.to_csv(CONFIG["csv_thresholds"], index=False)
pd.DataFrame(sens).to_csv(CONFIG["csv_sensitivity"], index=False)

print(f"Saved: {CONFIG['csv_thresholds']}")
print(f"Saved: {CONFIG['csv_sensitivity']}")
