"""
09_scf_overview_figure.py
=========================
Four-panel composite figure summarising Snow Cover Frequency (SCF)
across U.S. mountain regions (2018-2024).

Panels:
  (a) Mean SCF map + mountain-range box plots
  (b) SCF trend map (Sen's Slope) + marginal profiles + box plots
  (c) Zoom-in: mean SCF for a selected mountain range
  (d) Zoom-in: SCF trend for the same range

Mountain ranges are colour-coded by mean elevation.

Dependencies: ee, geemap, cartopy, matplotlib, scipy

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
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib import rcParams
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shapereader
import matplotlib.pyplot as plt
from geemap import cartoee
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# =====================================================================
# CONFIGURATION
# =====================================================================
# TODO: Replace with your GEE project ID
try:
    ee.Initialize(project="YOUR_PROJECT")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="YOUR_PROJECT")

# TODO: Replace with your GEE asset paths
ASSET_FOLDER     = "projects/YOUR_PROJECT/assets/SCF_Fishnet"
STUDY_AREA_ASSET = "projects/YOUR_PROJECT/assets/GMBA_USA_clipped"

# Temporal range
YEARS = list(range(2018, 2025))

# Map extent [east, south, west, north]
USA_REGION = [-66, 24, -125, 49]
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = -125, 24, -66, 49

# Mountain ranges (bounding boxes for statistics & annotation)
MOUNTAIN_RANGES = {
    "Cascades":      {"bounds": [-123.5, 39.0, -119.5, 49.0],
                      "full_name": "Cascade Range"},
    "Sierra Nevada": {"bounds": [-121.5, 35.0, -117.5, 40.5],
                      "full_name": "Sierra Nevada"},
    "N. Rockies":    {"bounds": [-117.5, 42.0, -108.0, 49.0],
                      "full_name": "Northern Rocky Mountains"},
    "C. Rockies":    {"bounds": [-111.5, 36.5, -104.5, 43.0],
                      "full_name": "Central Rocky Mountains"},
    "S. Rockies":    {"bounds": [-108.5, 31.0, -104.0, 36.5],
                      "full_name": "Southern Rocky Mountains"},
    "Appalachians":  {"bounds": [-84.5, 33.0, -69.0, 47.5],
                      "full_name": "Appalachian Mountains"},
}

# Which range to zoom into for panels (c)/(d)
ZOOM_KEY = "N. Rockies"
ZOOM_LABEL = MOUNTAIN_RANGES[ZOOM_KEY].get("full_name", ZOOM_KEY)
ZB = MOUNTAIN_RANGES[ZOOM_KEY]["bounds"]
ZOOM_REGION = [ZB[2], ZB[1], ZB[0], ZB[3]]

# Colour-bar settings
MEAN_VIS  = {"min": 0, "max": 80,
             "palette": ["#EFEFEF", "#abd9e9", "#74add1", "#4575b4"]}
SLOPE_VIS = {"min": -5.0, "max": 5.0,
             "palette": ["#d73027", "#EFEFEF", "#4575b4"]}

OUTPUT_PATH = "./figures/SCF_Combined_4Panel.png"
os.makedirs("./figures", exist_ok=True)

# =====================================================================
# 1. Data preparation
# =====================================================================
study_region = ee.FeatureCollection(STUDY_AREA_ASSET)

assets = ee.data.listAssets({"parent": ASSET_FOLDER})["assets"]
image_ids = [a["id"] for a in assets if a["type"] == "IMAGE"]
combined_img = ee.ImageCollection(image_ids).mosaic()

scf_bands = [f"SCF_{y}" for y in YEARS]

# Mean SCF
mean_img = (combined_img.select(scf_bands)
            .reduce(ee.Reducer.mean()).rename("SCF_mean")
            .clip(study_region))

# Sen's Slope trend
def _stack(year):
    bn = ee.String("SCF_").cat(ee.Number(year).format("%d"))
    scf = combined_img.select([bn]).rename("SCF")
    t = ee.Image.constant(ee.Number(year)).rename("time").float()
    return t.addBands(scf)

col = ee.ImageCollection(ee.List(YEARS).map(_stack))
slope_img = (col.select(["time", "SCF"])
             .reduce(ee.Reducer.sensSlope())
             .select("slope").rename("slope"))

dem = ee.Image("USGS/SRTMGL1_003").select("elevation")

# Visualisation layers
land_white = ee.Image.constant(1).visualize(palette=["#FFFFFF"]).clip(study_region)
land_gray  = ee.Image.constant(1).visualize(palette=["#EFEFEF"]).clip(study_region)

mean_layer  = mean_img.visualize(**MEAN_VIS)
trend_layer = slope_img.visualize(**SLOPE_VIS)

final_mean  = land_white.blend(mean_layer).clip(study_region)
final_trend = land_gray.blend(trend_layer).clip(study_region)

# Colour maps
cmap_scf   = mcolors.LinearSegmentedColormap.from_list("mean", MEAN_VIS["palette"], 256)
norm_scf   = mcolors.Normalize(MEAN_VIS["min"], MEAN_VIS["max"])
cmap_slope = mcolors.LinearSegmentedColormap.from_list("slope", SLOPE_VIS["palette"], 256)
norm_slope = mcolors.TwoSlopeNorm(SLOPE_VIS["min"], 0, SLOPE_VIS["max"])

# =====================================================================
# 2. Mountain-range statistics
# =====================================================================
print("Computing mountain-range statistics …")

stats_mean  = {}
stats_trend = {}
elev_values = {}

for name, info in MOUNTAIN_RANGES.items():
    b = info["bounds"]
    geom = study_region.geometry().intersection(ee.Geometry.Rectangle(b))

    pm = mean_img.reduceRegion(ee.Reducer.percentile([5, 25, 50, 75, 95]),
                               geom, 1000, maxPixels=1e9, bestEffort=True).getInfo()
    pt = slope_img.reduceRegion(ee.Reducer.percentile([5, 25, 50, 75, 95]),
                                geom, 1000, maxPixels=1e9, bestEffort=True).getInfo()
    me = (dem.reduceRegion(ee.Reducer.mean(), geom, 1000, maxPixels=1e9,
                           bestEffort=True).get("elevation").getInfo() or 0)

    for d, prefix, target in [(pm, "SCF_mean", stats_mean),
                               (pt, "slope",    stats_trend)]:
        target[name] = {f"p{p}": (d.get(f"{prefix}_p{p}") or 0) for p in [5, 25, 50, 75, 95]}
        target[name]["mean_elev"] = me

    elev_values[name] = me
    print(f"  {name:16s}  median SCF={stats_mean[name]['p50']:5.1f}%"
          f"  trend={stats_trend[name]['p50']:+.3f}%/yr  elev={me:.0f} m")

# Elevation colour scale
e_min = min(elev_values.values()) * 0.9
e_max = max(elev_values.values()) * 1.1
cmap_elev = mcolors.LinearSegmentedColormap.from_list(
    "elev", ["#7CB9A8", "#A8D5BA", "#F5E6CC", "#F5C28C", "#E8945A", "#D35D3A"], 256)
norm_elev = mcolors.Normalize(e_min, e_max)

# =====================================================================
# 3. Marginal slope statistics
# =====================================================================
print("Computing marginal slope profiles …")

lon_step, lat_step = 1.0, 0.5
lons = np.arange(LON_MIN, LON_MAX, lon_step)
lats = np.arange(LAT_MIN, LAT_MAX, lat_step)

lon_means = []
for lon in lons:
    v = slope_img.reduceRegion(
        ee.Reducer.mean(), ee.Geometry.Rectangle([lon, LAT_MIN, lon + lon_step, LAT_MAX]),
        5000, maxPixels=1e9).get("slope").getInfo()
    lon_means.append(v if v is not None else np.nan)
lon_means = np.array(lon_means)

lat_means = []
for lat in lats:
    v = slope_img.reduceRegion(
        ee.Reducer.mean(), ee.Geometry.Rectangle([LON_MIN, lat, LON_MAX, lat + lat_step]),
        5000, maxPixels=1e9).get("slope").getInfo()
    lat_means.append(v if v is not None else np.nan)
lat_means = np.array(lat_means)

# Smooth
sigma = 1.0
for arr in [lon_means, lat_means]:
    valid = ~np.isnan(arr)
    if valid.any():
        arr[valid] = gaussian_filter1d(arr[valid], sigma)

lon_centres = lons + lon_step / 2
lat_centres = lats + lat_step / 2

# =====================================================================
# 4. Plotting
# =====================================================================
rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                 "font.weight": "bold"})
proj = ccrs.PlateCarree()

fig = plt.figure(figsize=(18, 28), dpi=300)

# Natural Earth country boundaries
shp = shapereader.natural_earth("50m", "cultural", "admin_0_countries")
usa_geoms = [r.geometry for r in shapereader.Reader(shp).records()
             if r.attributes.get("NAME") == "United States of America"]


def _map_decor(ax):
    """Add common map decorations."""
    ax.add_feature(cfeature.STATES.with_scale("50m"),
                   edgecolor="#B0B0B0", lw=0.6, ls="--", alpha=0.7)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                   edgecolor="#CECECE", lw=0.6)
    ax.add_geometries(usa_geoms, proj, facecolor="none",
                      edgecolor="#A0A0A0", lw=1.2, zorder=12)


def _add_mountain_boxes(ax, stats_dict):
    """Draw mountain-range rectangles colour-coded by elevation."""
    for nm, info in MOUNTAIN_RANGES.items():
        b = info["bounds"]
        c = cmap_elev(norm_elev(stats_dict[nm]["mean_elev"]))
        lw = 4.0 if nm == ZOOM_KEY else 2.8
        ax.add_patch(patches.Rectangle(
            (b[0], b[1]), b[2] - b[0], b[3] - b[1],
            transform=proj, facecolor="none", edgecolor=c,
            lw=lw, alpha=0.9, zorder=13))
        ax.text(b[0] + (b[2] - b[0]) * 0.05, b[1] + (b[3] - b[1]) * 0.08,
                nm, transform=proj, fontsize=10, fontweight="bold",
                family="Times New Roman", color="#333", zorder=15)


def _capsule_cbar(ax, cmap_, norm_, ticks, label,
                  rx=0.55, ry=0.08, rw=0.42, rh=0.18):
    """Draw a capsule-shaped colour-bar inside the map axes."""
    ax.add_patch(plt.Rectangle((rx, ry), rw, rh, transform=ax.transAxes,
                 facecolor="white", alpha=0.9, edgecolor="#999", lw=1, zorder=15))
    ax_ins = inset_axes(ax, "88%", "15%", loc="lower center",
                        bbox_to_anchor=(rx, ry + 0.09, rw, rh),
                        bbox_transform=ax.transAxes, borderpad=0)
    cb = mcolorbar.ColorbarBase(ax_ins, cmap=cmap_, norm=norm_, orientation="horizontal")
    cb.set_ticks(ticks)
    cap = patches.FancyBboxPatch((0, 0), 1, 1,
          boxstyle="round,pad=0,rounding_size=0.5",
          transform=ax_ins.transAxes, fc="none", ec="#000", lw=1.2, zorder=30)
    ax_ins.add_patch(cap)
    for art in ax_ins.get_children():
        if isinstance(art, (plt.matplotlib.image.AxesImage,
                            plt.matplotlib.collections.Collection)):
            art.set_clip_path(cap)
    cb.set_label(label, family="Times New Roman", weight="bold", size=14, labelpad=5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=12, width=1.2, length=5)
    ax_ins.patch.set_alpha(0)


def _boxplot_panel(ax, stats_dict, xlabel, title, xlim=None):
    """Horizontal box plots for mountain ranges sorted by elevation."""
    items = sorted(stats_dict.items(), key=lambda x: x[1]["mean_elev"], reverse=True)
    names = [i[0] for i in items]
    sdata = [i[1] for i in items]
    bdata = [{"med": s["p50"], "q1": s["p25"], "q3": s["p75"],
              "whislo": s["p5"], "whishi": s["p95"], "fliers": []} for s in sdata]
    bp = ax.bxp(bdata, positions=np.arange(len(names)), vert=False,
                patch_artist=True, widths=0.55, showfliers=False,
                boxprops=dict(lw=1.5), whiskerprops=dict(lw=1.2, ls="--"),
                capprops=dict(lw=1.2), medianprops=dict(lw=2.5, color="#333"))
    for p, s in zip(bp["boxes"], sdata):
        p.set_facecolor(cmap_elev(norm_elev(s["mean_elev"])))
        p.set_alpha(0.85); p.set_edgecolor("#333")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold", family="Times New Roman")
    ax.tick_params(labelsize=13)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_family("Times New Roman"); lb.set_weight("bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", ls=":", alpha=0.5, color="gray")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10, family="Times New Roman")


# ====================== Panel (a) ======================
print("Panel (a) …")
ax_a = fig.add_axes([0.03, 0.62, 0.60, 0.28], projection=proj)
cartoee.add_layer(ax_a, final_mean, region=USA_REGION)
_map_decor(ax_a)
_add_mountain_boxes(ax_a, stats_mean)

gl = ax_a.gridlines(draw_labels=True, lw=0.2, color="gray", alpha=0.4, ls=":")
gl.top_labels = gl.right_labels = False
gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
gl.xlabel_style = gl.ylabel_style = {
    "size": 14, "color": "#333", "weight": "bold", "family": "Times New Roman"}

_capsule_cbar(ax_a, cmap_scf, norm_scf, [20, 40, 60],
              "Mean Snow Cover Frequency (%)")

fig.text(0.03, 0.905, "(a)", fontsize=20, fontweight="bold", family="Times New Roman")
fig.text(0.065, 0.905, "Mean Snow Cover Frequency (2018–2024)",
         fontsize=16, fontweight="bold", family="Times New Roman")

fig.canvas.draw()
pos_a = ax_a.get_position()
ax_ba = fig.add_axes([pos_a.x1 + 0.06, pos_a.y0, 0.20, pos_a.height])
_boxplot_panel(ax_ba, stats_mean, "SCF (%)", "SCF by Mountain Range", xlim=(0, 70))

# Elevation mini-colorbar for panel (a) boxplot
ax_el_a = fig.add_axes([pos_a.x1 + 0.06 + 0.20 - 0.08,
                         pos_a.y0 + pos_a.height - 0.025, 0.07, 0.01])
cb_el_a = mcolorbar.ColorbarBase(ax_el_a, cmap=cmap_elev, norm=norm_elev,
                                  orientation="horizontal")
cb_el_a.set_ticks([int(e_min / 0.9), int((e_min / 0.9 + e_max / 1.1) / 2),
                   int(e_max / 1.1)])
cb_el_a.ax.tick_params(labelsize=8, length=3, pad=2)
cb_el_a.set_label("Elev (m)", fontsize=9, family="Times New Roman", labelpad=2)
cb_el_a.outline.set_linewidth(0.8)

# ====================== Panel (b) ======================
print("Panel (b) …")
ax_b = fig.add_axes([0.05, 0.30, 0.55, 0.25], projection=proj)
cartoee.add_layer(ax_b, final_trend, region=USA_REGION)
_map_decor(ax_b)
_add_mountain_boxes(ax_b, stats_trend)

gl = ax_b.gridlines(draw_labels=True, lw=0.2, color="gray", alpha=0.4, ls=":")
gl.top_labels = gl.right_labels = False
gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
gl.xlabel_style = gl.ylabel_style = {
    "size": 14, "color": "#333", "weight": "bold", "family": "Times New Roman"}

_capsule_cbar(ax_b, cmap_slope, norm_slope, [-4, -2, 0, 2, 4],
              "SCF Trend (Slope %/yr)")

fig.text(0.03, 0.585, "(b)", fontsize=20, fontweight="bold", family="Times New Roman")
fig.text(0.065, 0.585, "SCF Trend (Sen's Slope, 2018–2024)",
         fontsize=16, fontweight="bold", family="Times New Roman")

# Marginal profiles
fig.canvas.draw()
bb = ax_b.get_position()
gap = 0.008
lc, fc_p, fc_n = "#555", "#d4e6f1", "#f5d7d7"

# Top (longitude)
ax_top = fig.add_axes([bb.x0, bb.y1 + gap, bb.width, 0.06])
lp = np.where(lon_means >= 0, lon_means, 0)
ln = np.where(lon_means <  0, lon_means, 0)
ax_top.fill_between(lon_centres, 0, lp, alpha=0.7, color=fc_p)
ax_top.fill_between(lon_centres, 0, ln, alpha=0.7, color=fc_n)
ax_top.plot(lon_centres, lon_means, color=lc, lw=1.5, marker="o", ms=2,
            markerfacecolor="white", markeredgecolor=lc, markeredgewidth=0.6)
ax_top.axhline(0, color="#999", lw=0.8)
ax_top.set_xlim(LON_MIN, LON_MAX)
ya = max(abs(np.nanmin(lon_means)), abs(np.nanmax(lon_means))) * 1.3
ax_top.set_ylim(-ya, ya)
ax_top.set_ylabel("Slope\n(%/yr)", fontsize=11, fontweight="bold")
ax_top.tick_params(axis="x", labelbottom=False, length=0)
ax_top.spines["top"].set_visible(False); ax_top.spines["right"].set_visible(False)

# Right (latitude)
ax_rt = fig.add_axes([bb.x1 + gap, bb.y0, 0.04, bb.height])
lp2 = np.where(lat_means >= 0, lat_means, 0)
ln2 = np.where(lat_means <  0, lat_means, 0)
ax_rt.fill_betweenx(lat_centres, 0, lp2, alpha=0.7, color=fc_p)
ax_rt.fill_betweenx(lat_centres, 0, ln2, alpha=0.7, color=fc_n)
ax_rt.plot(lat_means, lat_centres, color=lc, lw=1.5, marker="o", ms=2,
           markerfacecolor="white", markeredgecolor=lc, markeredgewidth=0.6)
ax_rt.axvline(0, color="#999", lw=0.8)
ax_rt.set_ylim(LAT_MIN, LAT_MAX)
xa = max(abs(np.nanmin(lat_means)), abs(np.nanmax(lat_means))) * 1.3
ax_rt.set_xlim(-xa, xa)
ax_rt.set_xlabel("Slope\n(%/yr)", fontsize=10, fontweight="bold")
ax_rt.tick_params(axis="y", labelleft=False, length=0)
ax_rt.spines["top"].set_visible(False); ax_rt.spines["right"].set_visible(False)

# Box plot for trend
x_max_b = max(abs(stats_trend[n]["p5"]) for n in stats_trend)
x_max_b = max(x_max_b, max(abs(stats_trend[n]["p95"]) for n in stats_trend)) * 1.2
ax_bb = fig.add_axes([bb.x1 + gap + 0.04 + 0.04, bb.y0, 0.17, bb.height])
_boxplot_panel(ax_bb, stats_trend, "Trend (%/yr)", "Trend by Mountain Range",
               xlim=(-x_max_b, x_max_b))
ax_bb.axvline(0, color="#999", lw=1)

# Elevation mini-colorbar for panel (b) boxplot
ax_el_b = fig.add_axes([bb.x1 + gap + 0.04 + 0.04 + 0.17 - 0.06,
                         bb.y0 + bb.height - 0.025, 0.07, 0.01])
cb_el_b = mcolorbar.ColorbarBase(ax_el_b, cmap=cmap_elev, norm=norm_elev,
                                  orientation="horizontal")
cb_el_b.set_ticks([int(e_min / 0.9), int((e_min / 0.9 + e_max / 1.1) / 2),
                   int(e_max / 1.1)])
cb_el_b.ax.tick_params(labelsize=8, length=3, pad=2)
cb_el_b.set_label("Elev (m)", fontsize=9, family="Times New Roman", labelpad=2)
cb_el_b.outline.set_linewidth(0.8)

# ====================== Panel (c) — Zoom mean ======================
print("Panel (c) …")
ax_c = fig.add_axes([0.05, 0.03, 0.40, 0.22], projection=proj)
cartoee.add_layer(ax_c, final_mean, region=ZOOM_REGION)
_map_decor(ax_c)
gl = ax_c.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4, ls=":")
gl.top_labels = gl.right_labels = False
gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
gl.xlabel_style = gl.ylabel_style = {
    "size": 12, "color": "#333", "weight": "bold", "family": "Times New Roman"}

ax_cbc = inset_axes(ax_c, "40%", "4%", loc="lower right",
                     bbox_to_anchor=(0, 0.06, 0.95, 1), bbox_transform=ax_c.transAxes)
cb_c = mcolorbar.ColorbarBase(ax_cbc, cmap=cmap_scf, norm=norm_scf, orientation="horizontal")
cb_c.set_ticks([0, 20, 40, 60, 80])
cb_c.set_label("SCF (%)", fontsize=11, weight="bold", labelpad=3)
cb_c.ax.tick_params(labelsize=10, length=3)

fig.text(0.05, 0.26, "(c)", fontsize=20, fontweight="bold", family="Times New Roman")
fig.text(0.085, 0.26, f"Mean SCF — {ZOOM_LABEL}",
         fontsize=15, fontweight="bold", family="Times New Roman")

# ====================== Panel (d) — Zoom trend ======================
print("Panel (d) …")
ax_d = fig.add_axes([0.52, 0.03, 0.40, 0.22], projection=proj)
cartoee.add_layer(ax_d, final_trend, region=ZOOM_REGION)
_map_decor(ax_d)
gl = ax_d.gridlines(draw_labels=True, lw=0.3, color="gray", alpha=0.4, ls=":")
gl.top_labels = gl.right_labels = False
gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
gl.xlabel_style = gl.ylabel_style = {
    "size": 12, "color": "#333", "weight": "bold", "family": "Times New Roman"}

ax_dbc = inset_axes(ax_d, "40%", "4%", loc="lower right",
                     bbox_to_anchor=(0, 0.06, 0.95, 1), bbox_transform=ax_d.transAxes)
cb_d = mcolorbar.ColorbarBase(ax_dbc, cmap=cmap_slope, norm=norm_slope, orientation="horizontal")
cb_d.set_ticks([-4, -2, 0, 2, 4])
cb_d.set_label("Trend (%/yr)", fontsize=11, weight="bold", labelpad=3)
cb_d.ax.tick_params(labelsize=10, length=3)

fig.text(0.52, 0.26, "(d)", fontsize=20, fontweight="bold", family="Times New Roman")
fig.text(0.555, 0.26, f"SCF Trend — {ZOOM_LABEL}",
         fontsize=15, fontweight="bold", family="Times New Roman")

# ====================== Save ======================
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUTPUT_PATH}")
plt.show()
