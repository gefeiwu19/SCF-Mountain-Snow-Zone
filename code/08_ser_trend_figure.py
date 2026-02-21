"""
08_ser_trend_figure.py
======================
Snowline Elevation Range (SER) trend analysis and visualisation.

Computes Sen's Slope + Mann-Kendall significance for three metrics
per GMBA mountain region:
  SER      = SLA_max - SLA_min  (snowline altitude range)
  SLA_max  = maximum snowline altitude
  SLA_min  = minimum snowline altitude

Generates a 5-panel composite figure:
  (a) Map — SER trend per mountain (hatching = significant)
  (b) Violin — SER trend by elevation band
  (c) Violin — SER trend by latitude band
  (d) Scatter — SLA_max vs SLA_min trend (quadrant analysis)
  (e) Box — SLA_max & SLA_min trend comparison

Dependencies: ee, geemap, cartopy, matplotlib, scipy, shapely,
              pymannkendall, pandas

Author:  Gefei Wu
Affiliation: Zhejiang University
Date:    2026-02-21
License: MIT
"""

import glob
import os
import warnings

import ee
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pymannkendall as mk
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import rcParams
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from shapely.geometry import shape

warnings.filterwarnings("ignore")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import matplotlib.pyplot as plt
from geemap import cartoee

# =====================================================================
# CONFIGURATION
# =====================================================================
# TODO: Replace with your GEE project ID
try:
    ee.Initialize(project="YOUR_PROJECT")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="YOUR_PROJECT")

# TODO: Set path to your snowline CSV directory
DATA_PATH = "./data/snowline_results/*.csv"

# TODO: Replace with your GEE asset path
GMBA_ASSET_ID = "projects/YOUR_PROJECT/assets/GMBA_USA_clipped"

# Column name mapping (adjust if your CSVs differ)
COL_NAME = "MountainName"    # mountain identifier
COL_YEAR = "Year"
COL_MAX  = "Max_Snowline"    # annual max snowline altitude (m)
COL_MIN  = "Min_Snowline"    # annual min snowline altitude (m)

# Map bounds
USA_REGION = [-66, 24, -125, 49]

# Output
OUTPUT_PNG = "./figures/SER_Trend_Analysis.png"
OUTPUT_PDF = "./figures/SER_Trend_Analysis.pdf"
MK_CSV     = "./results/SER_MK_Results.csv"
MK_GEO_CSV = "./results/SER_MK_Results_With_Geo.csv"

# Ensure output directories exist
for _p in [OUTPUT_PNG, OUTPUT_PDF, MK_CSV, MK_GEO_CSV]:
    os.makedirs(os.path.dirname(_p) or ".", exist_ok=True)

# =====================================================================
# 1. Read snowline data
# =====================================================================
csv_files = glob.glob(DATA_PATH)
print(f"Found {len(csv_files)} CSV files")
if not csv_files:
    raise FileNotFoundError(f"No CSVs at {DATA_PATH}")

snowline_df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

# Auto-detect column names
for candidate in ["MountainName", "Mountain_Name", "Name", "name"]:
    if candidate in snowline_df.columns:
        COL_NAME = candidate
        break

snowline_df["Snowline_Range"] = snowline_df[COL_MAX] - snowline_df[COL_MIN]
snowline_df = snowline_df.dropna(
    subset=[COL_NAME, COL_YEAR, "Snowline_Range", COL_MAX, COL_MIN]
)
snowline_df = snowline_df[snowline_df["Snowline_Range"] > 0]

print(f"Valid records: {len(snowline_df):,}   Mountains: {snowline_df[COL_NAME].nunique()}")

# =====================================================================
# 2. Mann-Kendall + Sen's Slope for each mountain
# =====================================================================
def _mk_sen(values):
    """Run MK test on a time-series; return dict of results."""
    if len(values) >= 4:
        try:
            r = mk.original_test(values)
            return dict(slope=r.slope, p=r.p, trend=r.trend, z=r.z,
                        tau=r.Tau,
                        sig010=int(r.p < 0.10), sig005=int(r.p < 0.05))
        except Exception:
            pass
    return dict(slope=np.nan, p=np.nan, trend="insufficient",
                z=np.nan, tau=np.nan, sig010=0, sig005=0)


print("Computing Sen's Slope + MK test (SER / SLA_max / SLA_min) …")
rows = []
for name, grp in snowline_df.groupby(COL_NAME):
    grp = grp.sort_values(COL_YEAR)
    row = {"Name": name, "n_years": len(grp),
           "mean_range": grp["Snowline_Range"].mean(),
           "mean_sla_max": grp[COL_MAX].mean(),
           "mean_sla_min": grp[COL_MIN].mean()}

    for prefix, vals in [("ser",     grp["Snowline_Range"].values),
                         ("sla_max", grp[COL_MAX].values),
                         ("sla_min", grp[COL_MIN].values)]:
        res = _mk_sen(vals)
        for k, v in res.items():
            row[f"{prefix}_{k}"] = v

    # Backward-compatible aliases
    row["slope"]   = row["ser_slope"]
    row["sig_010"] = row["ser_sig010"]
    rows.append(row)

mk_results = pd.DataFrame(rows)
mk_results.to_csv(MK_CSV, index=False)

print(f"  SER     significant (p<0.10): {mk_results['ser_sig010'].sum()}")
print(f"  SLA_max significant (p<0.10): {mk_results['sla_max_sig010'].sum()}")
print(f"  SLA_min significant (p<0.10): {mk_results['sla_min_sig010'].sum()}")

# =====================================================================
# 3. Fetch geographic info from GEE
# =====================================================================
GMBA_FC = ee.FeatureCollection(GMBA_ASSET_ID)
dem = ee.Image("USGS/SRTMGL1_003")

names = mk_results["Name"].tolist()
print(f"Fetching geo-info for {len(names)} mountains from GEE …")


def _add_geo(feat):
    g = feat.geometry()
    elev = dem.reduceRegion(ee.Reducer.mean(), g, 1000, maxPixels=1e9).get("elevation")
    c = g.centroid().coordinates()
    return feat.set({"mean_elevation": elev,
                     "centroid_lat": c.get(1), "centroid_lon": c.get(0)})


fc = GMBA_FC.filter(ee.Filter.inList("Name", names)).map(_add_geo)

try:
    props = fc.reduceColumns(
        ee.Reducer.toList(4), ["Name", "centroid_lat", "centroid_lon", "mean_elevation"]
    ).get("list").getInfo()
    geo_df = pd.DataFrame(props, columns=["Name", "centroid_lat", "centroid_lon",
                                           "mean_elevation"])
except Exception as e:
    print(f"  Warning: {e}")
    geo_df = pd.DataFrame(columns=["Name", "centroid_lat", "centroid_lon",
                                    "mean_elevation"])

merged_df = pd.merge(mk_results, geo_df, on="Name", how="inner")
merged_df = merged_df.dropna(subset=["slope", "mean_elevation", "centroid_lat"])
merged_df.to_csv(MK_GEO_CSV, index=False)
print(f"Merged mountains: {len(merged_df)}")

# =====================================================================
# 4. Significant-mountain geometries
# =====================================================================
sig_names = merged_df.loc[merged_df["sig_010"] == 1, "Name"].tolist()


def _geojson_to_shapely(geojson, tol=0.02):
    if geojson is None:
        return []
    out = []
    def _p(g):
        try:
            return shape(g).simplify(tol)
        except Exception:
            return None
    gt = geojson.get("type")
    if gt == "GeometryCollection":
        for g in geojson.get("geometries", []):
            s = _p(g)
            if s:
                out.append(s)
    elif gt in ("Polygon", "MultiPolygon"):
        s = _p(geojson)
        if s:
            out.append(s)
    return out


sig_geoms = []
if sig_names:
    try:
        sig_geoms = _geojson_to_shapely(
            GMBA_FC.filter(ee.Filter.inList("Name", sig_names)).geometry().getInfo()
        )
    except Exception as e:
        print(f"  Geometry warning: {e}")

# =====================================================================
# 5. GEE visualisation image
# =====================================================================
mk_feat = [ee.Feature(None, {"Name": r["Name"], "mean": r["slope"] or 0})
            for _, r in merged_df.iterrows()]
mk_fc = ee.FeatureCollection(mk_feat)
join = ee.Join.saveFirst("mk_data", outer=True)
joined = join.apply(GMBA_FC, mk_fc,
                    ee.Filter.equals(leftField="Name", rightField="Name"))


def _ext(f):
    d = ee.Feature(f.get("mk_data"))
    return f.set("mean", ee.Algorithms.If(f.get("mk_data"), d.get("mean"), 0))


result_fc = joined.map(_ext)
ser_trend_img = result_fc.reduceToImage(["mean"], ee.Reducer.first())

sv = merged_df["slope"].dropna()
abs_max = max(abs(np.percentile(sv, 5)), abs(np.percentile(sv, 95)))
slope_vis = {"min": -abs_max, "max": abs_max,
             "palette": ["#d73027", "#EFEFEF", "#4575b4"]}
final_viz = ser_trend_img.visualize(**slope_vis).clip(
    ee.Geometry.Rectangle([-66, 24, -125, 49])
)

# =====================================================================
# 6. Data prep
# =====================================================================
elev_bins   = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 9000]
elev_labels = ["<500", "500–1k", "1–1.5k", "1.5–2k",
               "2–2.5k", "2.5–3k", "3–3.5k", ">3.5k"]
merged_df["elev_group"] = pd.cut(merged_df["mean_elevation"],
                                  bins=elev_bins, labels=elev_labels)
lat_bins   = [24, 30, 35, 40, 45, 50]
lat_labels = ["24–30°N", "30–35°N", "35–40°N", "40–45°N", "45–50°N"]
merged_df["lat_group"] = pd.cut(merged_df["centroid_lat"],
                                 bins=lat_bins, labels=lat_labels)

y_lo = -abs_max * 1.2
y_hi =  abs_max * 1.2

COLORS = {"accent": "#e74c3c", "bg_light": "#f8f9fa", "text": "#2c3e50"}
rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                 "font.weight": "bold", "axes.labelweight": "bold",
                 "axes.titleweight": "bold"})

# =====================================================================
# 7. Composite figure (3 × 3 grid)
# =====================================================================
print("Drawing figure …")
fig = plt.figure(figsize=(16, 18.5), dpi=300, facecolor="white")
gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1, 1],
                      width_ratios=[1, 1, 0.15],
                      hspace=0.30, wspace=0.24,
                      left=0.08, right=0.92, top=0.96, bottom=0.04)

proj = ccrs.PlateCarree()

# ---------- (a) Map ----------
ax_map = fig.add_subplot(gs[0, :], projection=proj)
cartoee.add_layer(ax_map, final_viz, region=USA_REGION, scale=10000)

if sig_geoms:
    for g in sig_geoms:
        ax_map.add_geometries([g], proj, facecolor="none", edgecolor="#1a1a1a",
                              hatch="///", linewidth=0.6, alpha=0.85, zorder=10)

ax_map.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor="#B0B0B0", lw=0.6, ls="--", alpha=0.7)
ax_map.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#CECECE", lw=0.6)
shp = shapereader.natural_earth("50m", "cultural", "admin_0_countries")
usa = [r.geometry for r in shapereader.Reader(shp).records()
       if r.attributes.get("NAME") == "United States of America"]
ax_map.add_geometries(usa, proj, facecolor="none", edgecolor="#A0A0A0",
                      lw=1.2, zorder=12)
gl = ax_map.gridlines(draw_labels=True, lw=0.2, color="gray", alpha=0.4, ls=":")
gl.top_labels = gl.right_labels = False
gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
gl.xlabel_style = gl.ylabel_style = {
    "size": 14, "color": "#333", "weight": "bold", "family": "Times New Roman"}

# Colour-bar
cmap = mcolors.LinearSegmentedColormap.from_list("s", slope_vis["palette"], 256)
norm = mcolors.TwoSlopeNorm(slope_vis["min"], 0, slope_vis["max"])
rx, ry, rw, rh = 0.58, 0.05, 0.40, 0.22
ax_map.add_patch(plt.Rectangle((rx, ry), rw, rh, transform=ax_map.transAxes,
                 facecolor="white", alpha=0.9, edgecolor="#999", lw=1, zorder=15))
ax_cb = inset_axes(ax_map, "85%", "18%", loc="lower center",
                   bbox_to_anchor=(rx + 0.025, ry + 0.12, rw - 0.05, rh),
                   bbox_transform=ax_map.transAxes, borderpad=0)
cb = mcolorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation="horizontal")
cb.set_label("SER Trend (m/yr)", family="Times New Roman", weight="bold", size=16)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=14, width=1.2, length=5)

lp = patches.Patch(facecolor="white", edgecolor="#333", hatch="///",
                    label="Significant (p < 0.10)")
ax_map.legend(handles=[lp], loc="lower left", bbox_to_anchor=(0.01, 0.02),
              fontsize=15, frameon=True, facecolor="white", edgecolor="#999",
              prop={"family": "Times New Roman", "weight": "bold"})
ax_map.set_title("(a) Spatial Distribution of SER Trends",
                 fontsize=20, fontweight="bold", family="Times New Roman",
                 loc="left", pad=12, color=COLORS["text"])

# ---------- Helper: violin panel ----------
def _violin(ax, col, labels, title, xlabel):
    grps = merged_df.groupby(col, observed=True)["slope"].apply(list).to_dict()
    data, vlabels = [], []
    for lb in labels:
        if lb in grps and grps[lb]:
            data.append(grps[lb])
            vlabels.append(lb)
    if not data:
        return
    pos = list(range(len(vlabels)))
    vp = ax.violinplot([np.clip(d, y_lo, y_hi) for d in data],
                       pos, widths=0.7, showmeans=False,
                       showmedians=False, showextrema=False)
    cm = plt.cm.YlOrBr if "Elev" in xlabel else plt.cm.Blues
    cs = cm(np.linspace(0.25, 0.75, len(vlabels)))
    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(cs[i]); pc.set_edgecolor("#666"); pc.set_alpha(0.35)
    np.random.seed(42)
    for i, d in enumerate(data):
        a = np.array(d)
        q1, med, q3 = np.percentile(a, [25, 50, 75])
        ax.vlines(i, max(q1, y_lo), min(q3, y_hi), color="#333", lw=4, zorder=3)
        if y_lo <= med <= y_hi:
            ax.scatter(i, med, color="white", s=40, zorder=4, edgecolor="#333", lw=1.5)
        jit = np.random.uniform(-0.18, 0.18, len(a))
        ok = (a >= y_lo) & (a <= y_hi)
        ax.scatter(i + jit[ok], a[ok], color=cs[i], alpha=0.7, s=25,
                   edgecolor="white", lw=0.4, zorder=2)
        for mask, marker, pos_y in [(a < y_lo, "v", y_lo * 0.93),
                                     (a > y_hi, "^", y_hi * 0.93)]:
            if mask.sum():
                ax.scatter(i + jit[mask], np.full(mask.sum(), pos_y),
                           marker=marker, color=COLORS["accent"], s=45, zorder=5,
                           edgecolor="white", lw=0.5)
        n_out = int((a < y_lo).sum() + (a > y_hi).sum())
        txt = f"n={len(d)}\n({n_out} out)" if n_out else f"n={len(d)}"
        ax.text(i, y_hi * 1.05, txt, ha="center", fontsize=12,
                fontfamily="Times New Roman", fontweight="bold", color="#555", va="bottom")
    ax.axhline(0, color="#999", ls="--", lw=1.2, alpha=0.7)
    ax.set_xticks(pos); ax.set_xticklabels(vlabels, rotation=25, ha="right")
    ax.set_ylim(y_lo * 1.18, y_hi * 1.18)
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold", family="Times New Roman")
    ax.set_ylabel("SER Trend (m/yr)", fontsize=14, fontweight="bold", family="Times New Roman")
    ax.set_title(title, fontsize=18, fontweight="bold", family="Times New Roman",
                 loc="left", pad=12, color=COLORS["text"])
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    if len(data) > 1:
        h, p = stats.kruskal(*[g for g in data if g])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(0.97, 0.03, f"Kruskal-Wallis\nH={h:.2f}\np={p:.3f} {sig}",
                transform=ax.transAxes, fontsize=12, va="bottom", ha="right",
                fontfamily="Times New Roman", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["bg_light"],
                          alpha=0.95, edgecolor="#ccc"))


# ---------- (b) Elevation violin ----------
_violin(fig.add_subplot(gs[1, 0]), "elev_group", elev_labels,
        "(b) SER Trend by Elevation", "Elevation Range (m)")

# ---------- (c) Latitude violin ----------
_violin(fig.add_subplot(gs[1, 1]), "lat_group", lat_labels,
        "(c) SER Trend by Latitude", "Latitude Range")

# Violin legend
ax_vleg = fig.add_subplot(gs[1, 2]); ax_vleg.axis("off")
ax_vleg.legend(handles=[
    Line2D([], [], marker="v", color="w", markerfacecolor=COLORS["accent"],
           ms=12, label="Outlier (below)", markeredgecolor="white"),
    Line2D([], [], marker="^", color="w", markerfacecolor=COLORS["accent"],
           ms=12, label="Outlier (above)", markeredgecolor="white"),
    Line2D([], [], marker="o", color="w", markerfacecolor="white", ms=12,
           label="Median", markeredgecolor="#333", markeredgewidth=1.5),
], loc="center left", fontsize=13, frameon=True, facecolor="white",
   edgecolor="#ccc", labelspacing=1.2,
   prop={"family": "Times New Roman", "weight": "bold"},
   bbox_to_anchor=(-0.5, 0.5))

# ---------- (d) SLA_max vs SLA_min scatter ----------
ax_sc = fig.add_subplot(gs[2, 0])
sdf = merged_df.dropna(subset=["sla_max_slope", "sla_min_slope"]).copy()
all_sl = np.concatenate([sdf["sla_min_slope"].values, sdf["sla_max_slope"].values])
q1a, q3a = np.percentile(all_sl, [25, 75])
iqr = q3a - q1a
scat_lim = np.ceil(max(abs(q1a - 1.8 * iqr), abs(q3a + 1.8 * iqr), 50) / 10) * 10

# Quadrant background
qa = 0.045
ax_sc.fill_between([0, scat_lim], 0, scat_lim,   color="#fee08b", alpha=qa)
ax_sc.fill_between([-scat_lim, 0], 0, scat_lim,  color="#4575b4", alpha=qa)
ax_sc.fill_between([-scat_lim, 0], -scat_lim, 0, color="#fee08b", alpha=qa)
ax_sc.fill_between([0, scat_lim], -scat_lim, 0,  color="#d73027", alpha=qa)
ax_sc.axhline(0, color="#aaa", lw=0.8, alpha=0.5)
ax_sc.axvline(0, color="#aaa", lw=0.8, alpha=0.5)

np.random.seed(2024)
jsc = scat_lim * 0.018
x = sdf["sla_min_slope"].values.astype(float) + np.random.uniform(-jsc, jsc, len(sdf))
y = sdf["sla_max_slope"].values.astype(float) + np.random.uniform(-jsc, jsc, len(sdf))
sig_m = sdf["sig_010"].values == 1
margin = scat_lim * 0.92
xc, yc = np.clip(x, -margin, margin), np.clip(y, -margin, margin)

ax_sc.scatter(xc[~sig_m], yc[~sig_m], facecolors="white", edgecolors="#888",
              s=40, lw=0.9, alpha=0.65, zorder=3)
ax_sc.scatter(xc[sig_m], yc[sig_m], facecolors="#2166ac", edgecolors="white",
              s=55, lw=0.7, alpha=0.85, zorder=4)

# Quadrant labels
for tx, ty, txt, col in [
    ( 0.68,  0.68, "Both rising",     "#997700"),
    (-0.68,  0.68, "SER\nexpansion",  "#2c5aa0"),
    (-0.68, -0.68, "Both falling",    "#997700"),
    ( 0.68, -0.68, "SER\ncontraction","#b22222"),
]:
    ax_sc.text(tx * scat_lim, ty * scat_lim, txt, ha="center", va="center",
               fontsize=10, fontstyle="italic", alpha=0.45, color=col,
               fontfamily="Times New Roman")

r_corr, p_corr = stats.pearsonr(sdf["sla_min_slope"], sdf["sla_max_slope"])
ax_sc.text(0.97, 0.05,
           f"r = {r_corr:.3f},  p = {'< 0.001' if p_corr < 0.001 else f'{p_corr:.3f}'},"
           f"  n = {len(sdf)}",
           transform=ax_sc.transAxes, fontsize=12, ha="right", va="bottom",
           fontfamily="Times New Roman", fontweight="bold",
           bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92,
                     edgecolor="#ccc"))

ax_sc.set_xlim(-scat_lim, scat_lim); ax_sc.set_ylim(-scat_lim, scat_lim)
ax_sc.set_xlabel(r"SLA$_{\mathrm{min}}$ Trend (m/yr)", fontsize=14, fontweight="bold")
ax_sc.set_ylabel(r"SLA$_{\mathrm{max}}$ Trend (m/yr)", fontsize=14, fontweight="bold")
ax_sc.set_title(r"(d) SLA$_{\mathrm{max}}$ vs SLA$_{\mathrm{min}}$ Trend",
                fontsize=18, fontweight="bold", family="Times New Roman",
                loc="left", pad=12, color=COLORS["text"])
ax_sc.spines["top"].set_visible(False); ax_sc.spines["right"].set_visible(False)

# ---------- (e) Box plot ----------
ax_bx = fig.add_subplot(gs[2, 1])

sla_max_all = sdf["sla_max_slope"].values
sla_min_all = sdf["sla_min_slope"].values

# Independent y-range for box plot (may differ from scatter range)
e_all = np.concatenate([sla_min_all, sla_max_all])
e_q1, e_q3 = np.percentile(e_all, [25, 75])
e_iqr = e_q3 - e_q1
e_lim = np.ceil(max(abs(e_q1 - 2.0 * e_iqr), abs(e_q3 + 2.0 * e_iqr), 40) / 10) * 10

bp = ax_bx.boxplot([sla_min_all, sla_max_all],
                    positions=[0, 1], vert=True, widths=0.5, patch_artist=True,
                    showfliers=False,
                    medianprops=dict(color="#d73027", lw=2.2),
                    whiskerprops=dict(color="#555", lw=1.2),
                    capprops=dict(color="#555", lw=1.2),
                    boxprops=dict(lw=1.2))
for p, c in zip(bp["boxes"], ["#74add1", "#fdae61"]):
    p.set_facecolor(c); p.set_alpha(0.5); p.set_edgecolor("#555")

# Overlay jitter + outlier arrows
np.random.seed(99)
arrow_margin = e_lim * 0.93
box_colors = ["#74add1", "#fdae61"]

for i, (vals, c) in enumerate(zip([sla_min_all, sla_max_all], box_colors)):
    jit = np.random.uniform(-0.15, 0.15, len(vals))
    normal = (vals >= -e_lim) & (vals <= e_lim)
    out_lo = vals < -e_lim
    out_hi = vals > e_lim

    ax_bx.scatter(i + jit[normal], vals[normal],
                  color=c, alpha=0.45, s=18, edgecolor="white", lw=0.3, zorder=2)
    if out_lo.sum() > 0:
        ax_bx.scatter(i + jit[out_lo], np.full(out_lo.sum(), -arrow_margin),
                      marker="v", color=COLORS["accent"], s=45, zorder=5,
                      edgecolor="white", lw=0.5)
    if out_hi.sum() > 0:
        ax_bx.scatter(i + jit[out_hi], np.full(out_hi.sum(), arrow_margin),
                      marker="^", color=COLORS["accent"], s=45, zorder=5,
                      edgecolor="white", lw=0.5)

    # Mean diamond
    ax_bx.scatter(i, np.mean(vals), marker="D", color="#333", s=45, zorder=5,
                  edgecolor="white", lw=0.8)

# n and outlier count labels
for i, (label, vals) in enumerate(
    zip(["min", "max"], [sla_min_all, sla_max_all])
):
    n = len(vals)
    n_out = int(np.sum((vals < -e_lim) | (vals > e_lim)))
    txt = f"n={n} ({n_out} out)" if n_out > 0 else f"n={n}"
    ax_bx.text(i, e_lim * 1.01, txt, ha="center", va="bottom", fontsize=10.5,
               fontfamily="Times New Roman", fontweight="bold", color="#555")

ax_bx.axhline(0, color="#aaa", ls="--", lw=0.8, alpha=0.5)
ax_bx.set_ylim(-e_lim * 1.08, e_lim * 1.08)
ax_bx.set_xticks([0, 1])
ax_bx.set_xticklabels([r"SLA$_{\mathrm{min}}$", r"SLA$_{\mathrm{max}}$"],
                        fontsize=13, fontweight="bold")
ax_bx.set_ylabel("Trend (m/yr)", fontsize=14, fontweight="bold",
                 family="Times New Roman")
ax_bx.set_title(r"(e) SLA$_{\mathrm{max}}$ & SLA$_{\mathrm{min}}$ Trend",
                fontsize=18, fontweight="bold", family="Times New Roman",
                loc="left", pad=12, color=COLORS["text"])
ax_bx.spines["top"].set_visible(False); ax_bx.spines["right"].set_visible(False)
ax_bx.tick_params(axis="both", labelsize=12)
for lb in ax_bx.get_yticklabels():
    lb.set_fontfamily("Times New Roman"); lb.set_fontweight("bold")

# Bottom-row legend
ax_bl = fig.add_subplot(gs[2, 2]); ax_bl.axis("off")
leg_d = ax_bl.legend(handles=[
    Line2D([], [], marker="o", color="w", markerfacecolor="#2166ac",
           markeredgecolor="white", ms=9, lw=0, label="Significant\n(SER p<0.10)"),
    Line2D([], [], marker="o", color="w", markerfacecolor="white",
           markeredgecolor="#888", ms=9, markeredgewidth=1.2, lw=0,
           label="Not significant"),
], loc="upper left", fontsize=11, frameon=True, facecolor="white",
   edgecolor="#ccc", bbox_to_anchor=(-0.5, 1.0),
   prop={"family": "Times New Roman", "weight": "bold"},
   title="(d) Legend",
   title_fontproperties={"family": "Times New Roman", "weight": "bold", "size": 11})
ax_bl.add_artist(leg_d)
ax_bl.legend(handles=[
    patches.Patch(facecolor="#74add1", alpha=0.5, edgecolor="#555",
                  label=r"SLA$_{\mathrm{min}}$"),
    patches.Patch(facecolor="#fdae61", alpha=0.5, edgecolor="#555",
                  label=r"SLA$_{\mathrm{max}}$"),
    Line2D([], [], marker="D", color="w", markerfacecolor="#333",
           markeredgecolor="white", ms=7, lw=0, label="Mean"),
    Line2D([], [], color="#d73027", lw=2.2, label="Median"),
    Line2D([], [], marker="v", color="w", markerfacecolor=COLORS["accent"],
           markeredgecolor="white", ms=7, lw=0, label="Outlier"),
], loc="lower left", fontsize=11, frameon=True, facecolor="white",
   edgecolor="#ccc", bbox_to_anchor=(-0.5, 0.0),
   prop={"family": "Times New Roman", "weight": "bold"},
   title="(e) Legend",
   title_fontproperties={"family": "Times New Roman", "weight": "bold", "size": 11})
ax_bl.add_artist(leg_d)

# =====================================================================
# 8. Save & summary
# =====================================================================
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(OUTPUT_PDF, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUTPUT_PNG}, {OUTPUT_PDF}")

# Quadrant summary
n_tot = len(sdf)
for label, cond in [
    ("Q1 both rising",     (sdf["sla_min_slope"] > 0) & (sdf["sla_max_slope"] > 0)),
    ("Q2 SER expansion",   (sdf["sla_min_slope"] < 0) & (sdf["sla_max_slope"] > 0)),
    ("Q3 both falling",    (sdf["sla_min_slope"] < 0) & (sdf["sla_max_slope"] < 0)),
    ("Q4 SER contraction", (sdf["sla_min_slope"] > 0) & (sdf["sla_max_slope"] < 0)),
]:
    n = cond.sum()
    print(f"  {label:22s}  {n:3d}  ({n / n_tot * 100:5.1f}%)")

plt.show()
