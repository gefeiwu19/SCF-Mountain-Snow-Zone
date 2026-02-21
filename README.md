# SCF-Mountain-Snow-Zone

Multi-source satellite snow cover frequency (SCF) mapping and trend analysis for U.S. mountain regions (2018–2024).

**Paper:** *Delineating the Mountain Snow Zone: A Snow Cover Frequency Framework for Characterizing Seasonal Snow Dynamics from Sentinel-2 and Landsat-8* — submitted to *GIScience and Remote Sensing*

**Author:** Gefei Wu · Zhejiang University

---

## Algorithm Overview

Snow cover is detected using a **Theia-inspired two-pass algorithm** applied to Sentinel-2 SR and Landsat-8 TOA imagery:

| Pass | NDSI threshold (S2 / L8) | RED threshold | Condition |
|------|--------------------------|---------------|-----------|
| 1 — Conservative | 0.25 / 0.30 | 0.20 | Applied everywhere |
| 2 — Relaxed      | 0.09 / 0.12 | 0.04 | Applied only above snowline |

Sensor fusion prioritises Sentinel-2; Landsat-8 fills non-overlapping dates. Annual **Snow Cover Frequency (SCF)** = snow days / valid days. Three snow zones are classified from SCF: non-snow, seasonal (SSA), and permanent (PSA). Trends are computed with **Sen's Slope** and tested via **Mann-Kendall**.

## Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATA ACQUISITION                              │
│  Sentinel-2 SR  ·  Landsat-8 TOA  ·  SRTM DEM  ·  SNOTEL SWE  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │  01 Snowline  │  │  02 SCF Map   │  │  03 Station   │
  │  Extraction   │  │  (Theia 2-pass│  │  Validation   │
  │  (GEE JS)     │  │   + cloud QA) │  │  (GEE JS)     │
  └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
          │                  │                  │
          ▼                  ▼                  ▼
  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
  │  08 SER Trend │  │  04 Area      │  │  05 NDSI      │
  │  (5-panel     │  │  Export       │  │  Validation   │
  │   figure)     │  │  (SSA / PSA)  │  │  (ROC + dist) │
  └───────────────┘  └───────┬───────┘  └───────────────┘
                             │
                    ┌────────┼────────┐
                    ▼        ▼        ▼
            ┌────────────┐  ┌──────────────┐
            │ 06 Zone    │  │ 07 Snow Area │
            │ Threshold  │  │ Trend Figure │
            │ (Otsu/     │  │ (map+violin) │
            │  K-means)  │  └──────────────┘
            └────────────┘
                    │
                    ▼
            ┌──────────────┐
            │ 09 SCF       │
            │ Overview     │
            │ (4-panel)    │
            └──────────────┘
```

---

## Repository Structure

```
SCF-Mountain-Snow-Zone/
├── README.md
├── LICENSE                              # MIT
├── .gitignore
├── requirements.txt                     # Python dependencies
├── code/
│   ├── 01_snowline_annual_summary.js    # GEE: annual snowline extraction
│   ├── 02_scf_annual_export.js          # GEE: annual SCF rasters (Theia)
│   ├── 03_scf_station_validation.js     # GEE: station-level SCF extraction
│   ├── 04_snow_area_export.py           # GEE-Python: SSA/PSA area by mountain
│   ├── 05_ndsi_threshold_validation.py  # NDSI ROC validation vs SNOTEL
│   ├── 06_scf_zone_classification.py    # Optimal SCF zone thresholds
│   ├── 07_snow_area_trend_figure.py     # Figure: PSA/SSA trend map + violins
│   ├── 08_ser_trend_figure.py           # Figure: snowline range trend
│   └── 09_scf_overview_figure.py        # Figure: SCF 4-panel overview
├── data/
│   └── README.md                        # Data sources & download instructions
└── figures/
    └── README.md                        # Figure descriptions & script mapping
```

## Scripts

| # | Script | Platform | Purpose |
|---|--------|----------|---------|
| 01 | `01_snowline_annual_summary.js` | GEE Code Editor | Extract annual max/min snowline elevations per GMBA mountain (S2 + L8) |
| 02 | `02_scf_annual_export.js` | GEE Code Editor | Compute annual SCF rasters using Theia two-pass snow detection with cloud masking |
| 03 | `03_scf_station_validation.js` | GEE Code Editor | Daily snow detection with temporal interpolation; extract SCF at SNOTEL stations |
| 04 | `04_snow_area_export.py` | GEE Python API | Compute slope-corrected SSA or PSA area per mountain-year (configurable) |
| 05 | `05_ndsi_threshold_validation.py` | Python | Validate NDSI thresholds against SNOTEL SWE (ROC + distribution analysis) |
| 06 | `06_scf_zone_classification.py` | Python | Determine optimal SCF thresholds (Otsu / K-means / adaptive) for zone classification |
| 07 | `07_snow_area_trend_figure.py` | Python + GEE | Composite figure: snow area trend map + violin plots (supports PSA and SSA) |
| 08 | `08_ser_trend_figure.py` | Python + GEE | Composite figure: SER / SLA_max / SLA_min trend analysis (5 panels) |
| 09 | `09_scf_overview_figure.py` | Python + GEE | 4-panel figure: mean SCF + trend maps with marginal profiles |

## Usage

### GEE JavaScript scripts (01–03)

1. Upload required assets to your GEE project (see [`data/README.md`](data/README.md))
2. Open the script in the [GEE Code Editor](https://code.earthengine.google.com/)
3. Update the `CONFIG` object at the top — replace `YOUR_PROJECT` with your GEE project ID
4. Run the script; start export tasks from the **Tasks** panel

### Python scripts (04–09)

```bash
# Install all dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine (required for scripts 04, 07-09)
earthengine authenticate
```

1. Update the `CONFIG` / `CONFIGURATION` section at the top of each script
2. Replace `YOUR_PROJECT` with your GEE project ID
3. Update local file paths (`./data/...`) to point to your data directory

### Key parameters

| Parameter | Value | Script(s) |
|-----------|-------|-----------|
| SWE binarisation threshold | 0.4 inch (≈ 1 cm) | 05 |
| SSA SCF range | 5–85% | 04, 06 |
| PSA SCF range | ≥ 85% | 04, 06 |
| Temporal interpolation window | ± 5 days | 02, 03 |
| Elevation bin width | 100 m | 01, 02, 03 |
| S2 cloud pre-filter | CLOUDY_PIXEL_PERCENTAGE < 80% | 02, 03 |
| L8 cloud pre-filter | CLOUD_COVER < 80% | 02, 03 |

## Data Availability

See [`data/README.md`](data/README.md) for complete data source documentation, including GEE asset descriptions, SNOTEL station data, and download instructions.

## Citation

If you use this code, please cite:

```bibtex
@article{wu2026snow,
  title   = {Delineating the Mountain Snow Zone: A Snow Cover Frequency
             Framework for Characterizing Seasonal Snow Dynamics
             from Sentinel-2 and Landsat-8},
  author  = {Wu, Gefei},
  journal = {GIScience and Remote Sensing},
  year    = {2026}
}
```

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

Gefei Wu — Zhejiang University
