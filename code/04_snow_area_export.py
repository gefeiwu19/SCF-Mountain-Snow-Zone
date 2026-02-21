"""
04_snow_area_export.py
======================
Compute slope-corrected annual snow zone area for each GMBA mountain.

Supports both Seasonal Snow Area (SSA) and Permanent Snow Area (PSA)
via a single ANALYSIS_TYPE parameter.  Snow Cover Frequency (SCF)
rasters produced by 02_scf_annual_export.js are used as input.

Zone definitions (adjustable via THRESHOLDS dict):
  SSA :  5 <= SCF < 85   (seasonal snow coverage)
  PSA :  85 <= SCF        (permanent / near-permanent snow)

Slope correction:
  True pixel area = pixelArea / cos(slope_in_radians)
  where slope is derived from SRTM 30 m DEM.

Output: GEE FeatureCollection asset (one row per mountain × year).

Author:      Gefei Wu
Affiliation: Zhejiang University
Date:        2026-02-21
License:     MIT
"""

import math
import ee

# ======================================================================
# CONFIGURATION — edit these before running
# ======================================================================
# TODO: Replace 'YOUR_PROJECT' with your GEE project ID
GEE_PROJECT = "YOUR_PROJECT"

# Analysis type: "SSA" (seasonal) or "PSA" (permanent)
ANALYSIS_TYPE = "SSA"

# SCF thresholds for each zone type
THRESHOLDS = {
    "SSA": {"scf_min": 5,  "scf_max": 85},   # 5 <= SCF < 85
    "PSA": {"scf_min": 85, "scf_max": 101},   # SCF >= 85  (101 = effectively no cap)
}

# TODO: Replace with your GEE asset paths
ASSET_FOLDER     = "projects/YOUR_PROJECT/assets/SCF_Fishnet"
STUDY_AREA_ASSET = "projects/YOUR_PROJECT/assets/GMBA_USA_clipped"
OUTPUT_FC_ID     = f"projects/YOUR_PROJECT/assets/GMBA_Yearly_{ANALYSIS_TYPE}_Stats"

YEARS = list(range(2018, 2025))

# ======================================================================
# GEE initialisation
# ======================================================================
try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

print("GEE initialised.")

# ======================================================================
# Data loading
# ======================================================================
study_region = ee.FeatureCollection(STUDY_AREA_ASSET)

assets = ee.data.listAssets({"parent": ASSET_FOLDER})["assets"]
image_ids = [a["id"] for a in assets if a["type"] == "IMAGE"]
area_images = ee.ImageCollection(image_ids)
print(f"SCF images found: {len(image_ids)}")

# DEM and slope-corrected pixel area
# True ground area = nominal pixel area / cos(slope)
dem = ee.Image("USGS/SRTMGL1_003").clip(study_region)
slope_rad = ee.Terrain.slope(dem).multiply(math.pi / 180)
actual_pixel_area = ee.Image.pixelArea().divide(slope_rad.cos())

# ======================================================================
# Compute snow zone area per mountain per year
# ======================================================================
thresh = THRESHOLDS[ANALYSIS_TYPE]
scf_min = thresh["scf_min"]
scf_max = thresh["scf_max"]

print(f"\nAnalysis type : {ANALYSIS_TYPE}")
print(f"SCF range     : [{scf_min}, {scf_max})")
print(f"Years         : {YEARS[0]}–{YEARS[-1]}")
print(f"Output asset  : {OUTPUT_FC_ID}")
print()

all_results = []

for year in YEARS:
    band_name = f"SCF_{year}"
    scf = area_images.select(band_name).mosaic()

    # Apply SCF threshold mask
    mask = scf.gte(scf_min).And(scf.lt(scf_max))
    area_band = actual_pixel_area.updateMask(mask)

    # Sum area within each mountain polygon
    yearly_stats = area_band.reduceRegions(
        collection=study_region,
        reducer=ee.Reducer.sum(),
        scale=30,
    )

    # Attach year and analysis type metadata
    yearly_stats = yearly_stats.map(
        lambda f, y=year: f.set("year", y).set("analysis_type", ANALYSIS_TYPE)
    )
    all_results.append(yearly_stats)
    print(f"  {year} — reduceRegions complete")

combined = ee.FeatureCollection(all_results).flatten()

# ======================================================================
# Export
# ======================================================================
task = ee.batch.Export.table.toAsset(
    collection=combined,
    description=f"GMBA_Yearly_{ANALYSIS_TYPE}_Stats_Export",
    assetId=OUTPUT_FC_ID,
)
task.start()
print(f"\n{ANALYSIS_TYPE} area export task started → {OUTPUT_FC_ID}")
print("Check the Tasks tab in the GEE Code Editor for progress.")
