"""
07_snow_area_trend_figure.py
============================
Generate a composite figure showing spatial trends in snow zone area
across U.S. mountain regions, with violin plots grouped by elevation
and latitude.

Layout:
  (a) Map of Sen's slope per GMBA mountain (with significance hatching)
  (b) Violin + jitter plot by elevation band
  (c) Violin + jitter plot by latitude band

Supports both Permanent Snow Area (PSA) and Seasonal Snow Area (SSA)
by switching ANALYSIS_TYPE.

Dependencies: ee, geemap, cartopy, matplotlib, scipy, shapely, pandas

Author:  Gefei Wu
Affiliation: Zhejiang University
Date:    2026-02-21
License: MIT
"""

import os
import warnings

import ee
import matplotlib.colorbar as mcolorbar
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import rcParams
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from shapely.geometry import shape

warnings.filterwarnings("ignore")

# Lazy imports that need a display or specific install
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

# Analysis type — "PSA" or "SSA"
ANALYSIS_TYPE = "PSA"

# Input data paths (Mann-Kendall results CSV)
# File should have columns: Name, slope, sig_010, …
MK_CSV_MERGED = f"./results/GMBA_{ANALYSIS_TYPE}_MK_Results_With_Elev_Lat.csv"
MK_CSV_BASIC  = f"./results/GMBA_{ANALYSIS_TYPE}_MK_Results.csv"

# TODO: Replace with your GEE asset path
GMBA_ASSET_ID = "projects/YOUR_PROJECT/assets/GMBA_USA_clipped"

# Map bounds [east, south, west, north]
USA_REGION = [-66, 24, -125, 49]

# Colour-bar range (km²/yr) — adjust per analysis type
SLOPE_VIS = {"min": -5, "max": 5, "palette": ["#d73027", "#EFEFEF", "#4575b4"]}

# Violin-plot y-axis range (km²/yr)
Y_DISPLAY_MIN, Y_DISPLAY_MAX = -15, 15

# Output
OUTPUT_PATH = f"./figures/{ANALYSIS_TYPE}_Trend.png"

# Ensure output directories exist
for _dir in ["./figures", "./results"]:
    os.makedirs(_dir, exist_ok=True)

# =====================================================================
# 1. Load data
# =====================================================================
merged_df = None

if os.path.exists(MK_CSV_MERGED):
    print(f"Reading merged CSV: {MK_CSV_MERGED}")
    merged_df = pd.read_csv(MK_CSV_MERGED)
elif os.path.exists(MK_CSV_BASIC):
    print("Building geo-info from GEE (may take ~1 min) …")
    mk_results = pd.read_csv(MK_CSV_BASIC)
    names = mk_results["Name"].tolist()
    fc = ee.FeatureCollection(GMBA_ASSET_ID).filter(ee.Filter.inList("Name", names))
    dem = ee.Image("USGS/SRTMGL1_003")

    def _add_props(feat):
        elev = dem.reduceRegion(ee.Reducer.mean(), feat.geometry(), 1000).get("elevation")
        lat  = feat.geometry().centroid().coordinates().get(1)
        return feat.set({"mean_elevation": elev, "centroid_lat": lat})

    fc_props = fc.map(_add_props)
    props = fc_props.reduceColumns(
        ee.Reducer.toList(3), ["Name", "centroid_lat", "mean_elevation"]
    ).get("list").getInfo()
    geo_df = pd.DataFrame(props, columns=["Name", "centroid_lat", "mean_elevation"])
    merged_df = pd.merge(mk_results, geo_df, on="Name", how="inner")
    merged_df.to_csv(MK_CSV_MERGED, index=False)
else:
    raise FileNotFoundError(
        f"Neither {MK_CSV_MERGED} nor {MK_CSV_BASIC} found."
    )

# =====================================================================
# 2. Significant-mountain geometries (simplified for plotting)
# =====================================================================
GMBA_FC = ee.FeatureCollection(GMBA_ASSET_ID)

sig_names = merged_df.loc[merged_df["sig_010"] == 1, "Name"].tolist()


def _geojson_to_shapely(geojson, tolerance=0.02):
    """Convert GEE GeoJSON to simplified Shapely geometries."""
    if geojson is None:
        return []
    geoms = []
    def _proc(g):
        try:
            return shape(g).simplify(tolerance)
        except Exception:
            return None
    gtype = geojson.get("type")
    if gtype == "GeometryCollection":
        for g in geojson.get("geometries", []):
            s = _proc(g)
            if s:
                geoms.append(s)
    elif gtype in ("Polygon", "MultiPolygon"):
        s = _proc(geojson)
        if s:
            geoms.append(s)
    return geoms


sig_geometries = []
if sig_names:
    print(f"Fetching {len(sig_names)} significant-mountain geometries …")
    try:
        sig_fc = GMBA_FC.filter(ee.Filter.inList("Name", sig_names))
        sig_geometries = _geojson_to_shapely(sig_fc.geometry().getInfo())
    except Exception as e:
        print(f"  Warning: {e}")

# =====================================================================
# 3. Prepare GEE visualisation image
# =====================================================================
# Convert slope to km² where needed
slope_divisor = 1e6 if "slope" in merged_df.columns and merged_df["slope"].abs().max() > 1000 else 1

mk_features = []
for _, row in merged_df.iterrows():
    val = row["slope"] / slope_divisor if not pd.isna(row.get("slope")) else 0
    mk_features.append(ee.Feature(None, {"Name": row["Name"], "mean": val}))

mk_fc = ee.FeatureCollection(mk_features)
join = ee.Join.saveFirst(matchKey="mk_data", outer=True)
joined = join.apply(GMBA_FC, mk_fc,
                    ee.Filter.equals(leftField="Name", rightField="Name"))


def _extract(f):
    mk = ee.Feature(f.get("mk_data"))
    return f.set("mean", ee.Algorithms.If(f.get("mk_data"), mk.get("mean"), 0))


result_fc = joined.map(_extract)
slope_img = result_fc.reduceToImage(["mean"], ee.Reducer.first())
final_viz = slope_img.visualize(**SLOPE_VIS).clip(
    ee.Geometry.Rectangle([-66, 24, -125, 49])
)

# =====================================================================
# 4. Data prep for violin plots
# =====================================================================
merged_df = merged_df.dropna(subset=["mean_elevation", "centroid_lat", "slope"])
merged_df["slope_km2"] = merged_df["slope"] / slope_divisor

elev_bins   = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 9000]
elev_labels = ["<500", "500–1000", "1000–1500", "1500–2000",
               "2000–2500", "2500–3000", "3000–3500", ">3500"]
merged_df["elev_group"] = pd.cut(merged_df["mean_elevation"],
                                  bins=elev_bins, labels=elev_labels)

lat_bins   = [24, 30, 35, 40, 45, 50]
lat_labels = ["24–30°N", "30–35°N", "35–40°N", "40–45°N", "45–50°N"]
merged_df["lat_group"] = pd.cut(merged_df["centroid_lat"],
                                 bins=lat_bins, labels=lat_labels)

# =====================================================================
# 5. Plotting
# =====================================================================
rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.weight": "bold", "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

COLORS = {"accent": "#e74c3c", "bg_light": "#f8f9fa", "text": "#2c3e50"}

fig = plt.figure(figsize=(16, 12), dpi=300, facecolor="white")
gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], width_ratios=[1, 1, 0.15],
                      hspace=0.25, wspace=0.24,
                      left=0.08, right=0.92, top=0.95, bottom=0.08)

# ----- (a) Map -----
proj = ccrs.PlateCarree()
ax_map = fig.add_subplot(gs[0, :], projection=proj)
cartoee.add_layer(ax_map, final_viz, region=USA_REGION, scale=10000)

if sig_geometries:
    for g in sig_geometries:
        ax_map.add_geometries([g], crs=proj, facecolor="none",
                              edgecolor="#1a1a1a", hatch="///",
                              linewidth=0.6, alpha=0.85, zorder=10)

ax_map.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor="#B0B0B0", linewidth=0.6, linestyle="--", alpha=0.7)
ax_map.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#CECECE", linewidth=0.6)

shp = shapereader.natural_earth("50m", "cultural", "admin_0_countries")
usa = [r.geometry for r in shapereader.Reader(shp).records()
       if r.attributes.get("NAME") == "United States of America"]
ax_map.add_geometries(usa, proj, facecolor="none", edgecolor="#A0A0A0",
                      linewidth=1.2, zorder=12)

gl = ax_map.gridlines(draw_labels=True, linewidth=0.2, color="gray",
                      alpha=0.4, linestyle=":")
gl.top_labels = gl.right_labels = False
gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
gl.xlabel_style = gl.ylabel_style = {
    "size": 14, "color": "#333", "weight": "bold", "family": "Times New Roman"
}

# Capsule colour-bar
cmap = mcolors.LinearSegmentedColormap.from_list("s", SLOPE_VIS["palette"], 256)
norm = mcolors.TwoSlopeNorm(SLOPE_VIS["min"], 0, SLOPE_VIS["max"])
rx, ry, rw, rh = 0.58, 0.05, 0.40, 0.22
ax_map.add_patch(plt.Rectangle((rx, ry), rw, rh, transform=ax_map.transAxes,
                 facecolor="white", alpha=0.9, edgecolor="#999", lw=1, zorder=15))
ax_cb = inset_axes(ax_map, "85%", "18%", loc="lower center",
                   bbox_to_anchor=(rx + 0.025, ry + 0.12, rw - 0.05, rh),
                   bbox_transform=ax_map.transAxes, borderpad=0)
cb = mcolorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, orientation="horizontal")
cb.set_ticks([-4, -2, 0, 2, 4])

# Capsule-shaped outline with clip path
capsule = patches.FancyBboxPatch(
    (0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.5",
    transform=ax_cb.transAxes, fc="none", ec="#000", lw=1.2, zorder=30)
ax_cb.add_patch(capsule)
for art in ax_cb.get_children():
    if isinstance(art, (plt.matplotlib.image.AxesImage,
                        plt.matplotlib.collections.Collection)):
        art.set_clip_path(capsule)

cb.set_label("Snow Area Trend (km²/yr)", family="Times New Roman",
             weight="bold", size=16, labelpad=5)
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=14, width=1.2, length=5, direction="out", pad=4)
for lb in cb.ax.xaxis.get_ticklabels():
    lb.set_family("Times New Roman")
    lb.set_weight("bold")
ax_cb.patch.set_alpha(0)

legend_p = patches.Patch(facecolor="white", edgecolor="#333", hatch="///",
                         label="Significant (p < 0.10)")
ax_map.legend(handles=[legend_p], loc="lower left", bbox_to_anchor=(0.01, 0.02),
              fontsize=15, frameon=True, facecolor="white", edgecolor="#999",
              prop={"family": "Times New Roman", "weight": "bold"})

label = "Permanent" if ANALYSIS_TYPE == "PSA" else "Seasonal"
ax_map.set_title(
    f"(a) Spatial Distribution of {label} Snow Area Trends",
    fontsize=20, fontweight="bold", family="Times New Roman",
    loc="left", pad=12, color=COLORS["text"],
)


# ----- Helper: draw violin panel -----
def _draw_violin(ax, group_col, group_labels, panel_label, xlabel):
    grps = merged_df.groupby(group_col, observed=True)["slope_km2"].apply(list).to_dict()
    data, labels_valid = [], []
    for lb in group_labels:
        if lb in grps and grps[lb]:
            data.append(grps[lb])
            labels_valid.append(lb)
    if not data:
        return

    pos = list(range(len(labels_valid)))
    clipped = [np.clip(d, Y_DISPLAY_MIN, Y_DISPLAY_MAX) for d in data]
    vp = ax.violinplot(clipped, pos, widths=0.7, showmeans=False,
                       showmedians=False, showextrema=False)
    cmap_v = plt.cm.YlOrBr if "Elev" in xlabel else plt.cm.Blues
    cols = cmap_v(np.linspace(0.25, 0.75, len(labels_valid)))
    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(cols[i])
        pc.set_edgecolor("#666")
        pc.set_alpha(0.35)

    np.random.seed(42)
    for i, (lb, d) in enumerate(zip(labels_valid, data)):
        arr = np.array(d)
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        ax.vlines(i, max(q1, Y_DISPLAY_MIN), min(q3, Y_DISPLAY_MAX),
                  color="#333", lw=4, zorder=3)
        if Y_DISPLAY_MIN <= med <= Y_DISPLAY_MAX:
            ax.scatter(i, med, color="white", s=40, zorder=4,
                       edgecolor="#333", lw=1.5)
        jit = np.random.uniform(-0.18, 0.18, len(arr))
        ok = (arr >= Y_DISPLAY_MIN) & (arr <= Y_DISPLAY_MAX)
        ax.scatter(i + jit[ok], arr[ok], color=cols[i], alpha=0.7,
                   s=25, edgecolor="white", lw=0.4, zorder=2)
        lo = arr < Y_DISPLAY_MIN
        hi = arr > Y_DISPLAY_MAX
        if lo.sum():
            ax.scatter(i + jit[lo], np.full(lo.sum(), Y_DISPLAY_MIN * 0.93),
                       marker="v", color=COLORS["accent"], s=45, zorder=5,
                       edgecolor="white", lw=0.5)
        if hi.sum():
            ax.scatter(i + jit[hi], np.full(hi.sum(), Y_DISPLAY_MAX * 0.93),
                       marker="^", color=COLORS["accent"], s=45, zorder=5,
                       edgecolor="white", lw=0.5)
        n_out = int(lo.sum() + hi.sum())
        txt = f"n={len(d)}\n({n_out} out)" if n_out else f"n={len(d)}"
        ax.text(i, Y_DISPLAY_MAX * 1.05, txt, ha="center", fontsize=12,
                fontfamily="Times New Roman", fontweight="bold", color="#555", va="bottom")

    ax.axhline(0, color="#999", ls="--", lw=1.2, alpha=0.7, zorder=1)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels_valid, rotation=25, ha="right")
    ax.set_ylim(Y_DISPLAY_MIN * 1.18, Y_DISPLAY_MAX * 1.18)
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold", family="Times New Roman")
    ax.set_ylabel("Snow Area Trend (km²/yr)", fontsize=14, fontweight="bold",
                  family="Times New Roman")
    ax.set_title(panel_label, fontsize=18, fontweight="bold",
                 family="Times New Roman", loc="left", pad=12, color=COLORS["text"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontfamily("Times New Roman")

    # Kruskal-Wallis test
    if len(data) > 1:
        h, p = stats.kruskal(*[g for g in data if g])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(0.97, 0.03, f"Kruskal-Wallis\nH = {h:.2f}\np = {p:.3f} {sig}",
                transform=ax.transAxes, fontsize=12, va="bottom", ha="right",
                fontfamily="Times New Roman", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor=COLORS["bg_light"],
                          alpha=0.95, edgecolor="#ccc"))


# ----- (b) Elevation -----
ax_elev = fig.add_subplot(gs[1, 0])
_draw_violin(ax_elev, "elev_group", elev_labels,
             "(b) Trend Distribution by Elevation", "Elevation Range (m)")

# ----- (c) Latitude -----
ax_lat = fig.add_subplot(gs[1, 1])
_draw_violin(ax_lat, "lat_group", lat_labels,
             "(c) Trend Distribution by Latitude", "Latitude Range")

# ----- Legend panel -----
ax_leg = fig.add_subplot(gs[1, 2])
ax_leg.axis("off")
ax_leg.legend(
    handles=[
        Line2D([], [], marker="v", color="w", markerfacecolor=COLORS["accent"],
               ms=12, label="Outlier\n(below range)", markeredgecolor="white"),
        Line2D([], [], marker="^", color="w", markerfacecolor=COLORS["accent"],
               ms=12, label="Outlier\n(above range)", markeredgecolor="white"),
        Line2D([], [], marker="o", color="w", markerfacecolor="white",
               ms=12, label="Median", markeredgecolor="#333", markeredgewidth=1.5),
    ],
    loc="center left", fontsize=13, frameon=True, facecolor="white",
    edgecolor="#ccc", labelspacing=1.2,
    prop={"family": "Times New Roman", "weight": "bold"},
    bbox_to_anchor=(-0.5, 0.5),
)

# =====================================================================
# 6. Save
# =====================================================================
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUTPUT_PATH}")
plt.show()
