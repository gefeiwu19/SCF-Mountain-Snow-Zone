# Data Sources

This folder is intentionally left empty in the repository because the raw
data files are too large for GitHub. Below are the data sources and
instructions for obtaining them.

## Required GEE Assets

| Asset | Description | Source |
|-------|-------------|--------|
| `GMBA_USA_clipped` | GMBA Mountain Inventory v1.2, clipped to CONUS | [GMBA](https://ilias.unibe.ch/goto_ilias3_unibe_cat_1000515.html) |
| `weather_stations` | SNOTEL station point locations | [NRCS SNOTEL](https://www.nrcs.usda.gov/wps/portal/wcc/home/snowClimateMonitoring/snowpack/snotelSiteData/) |
| `SCF_Fishnet` | Fishnet grid clipped to GMBA extent | Generated in GIS preprocessing |

## SNOTEL Ground-Truth Data

Station observations (daily SWE) are downloaded from the
[NRCS Air & Water Database](https://wcc.sc.egov.usda.gov/nwcc/inventory).

Snow Water Equivalent (SWE) binarisation threshold: **0.4 inch** (≈ 1 cm).

| File pattern | Content | Source |
|-------------|---------|--------|
| `station_obs/*.csv` | Daily SWE, temperature, precipitation per station | NRCS SNOTEL |
| `remote_sensing/*.csv` | NDSI, RED, above_snowline sampled at station points | Extracted via script 03 |
| `scf_frequency_bins/*.csv` | SCF histogram bins per GMBA mountain | Exported from GEE |
| `snowline_results/*.csv` | Annual max/min snowline per mountain | Exported via script 01 |

## Public Datasets (accessed directly in GEE)

| Dataset | GEE Collection ID |
|---------|-------------------|
| Sentinel-2 Level-2A SR | `COPERNICUS/S2_SR` |
| Landsat 8 Collection 2 T1 TOA | `LANDSAT/LC08/C02/T1_TOA` |
| SRTM DEM 30m | `USGS/SRTMGL1_003` |

## How to Use

1. Download the raw data from the links above.
2. Upload to your GEE project as Assets.
3. Update the `CONFIG.assets` paths in each script under `code/`.
