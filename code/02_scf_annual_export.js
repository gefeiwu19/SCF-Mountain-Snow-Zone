/**
 * =============================================================================
 * 02_scf_annual_export.js
 * =============================================================================
 * Title:       Annual Snow Cover Frequency (SCF) Computation and Export
 * Description: Implements the full Theia-like two-pass snow detection algorithm
 *              to compute pixel-level annual SCF for GMBA mountain regions in
 *              the contiguous US (2018–2024). Results are exported as multi-band
 *              GEE Assets (one image per year, with SCF, snow_days, valid_days).
 *
 * Algorithm:
 *   Pass 1 — Conservative snow detection using strict NDSI/Red thresholds.
 *   Pass 2 — Relaxed thresholds applied only above the monthly snowline
 *            (derived from Pass 1) to recover under-detected snow pixels.
 *   Cloud masking follows a sensor-specific pipeline with dark-cloud recovery.
 *   Sentinel-2 images take priority on overlapping dates; Landsat 8 fills gaps.
 *
 * Reference:
 *   Gascoin, S., et al. (2019). A snow cover climatology for the Pyrenees
 *   from MODIS snow products. Hydrology and Earth System Sciences, 23(3).
 *
 * Author:      Gefei Wu
 * Affiliation: Zhejiang University
 * Date:        2026-02-21
 * License:     MIT
 *
 * Dependencies:
 *   - Google Earth Engine (JavaScript API)
 *   - SRTM DEM (USGS/SRTMGL1_003)
 *   - Sentinel-2 Level-2A Surface Reflectance (COPERNICUS/S2_SR)
 *   - Landsat 8 Collection 2 Tier 1 TOA (LANDSAT/LC08/C02/T1_TOA)
 *
 * Required Assets (user must upload to their own GEE project):
 *   - GMBA Mountain Inventory v2.0 clipped to CONUS
 *   See data/README.md for download links and preprocessing steps.
 *
 * Usage:
 *   1. Upload required assets to your GEE project.
 *   2. Update CONFIG.assets and CONFIG.export.assetPrefix below.
 *   3. Run in GEE Code Editor; export tasks appear in the Tasks panel.
 * =============================================================================
 */

// =============================================================================
// CONFIGURATION
// =============================================================================
var CONFIG = {
  // ---- GEE Asset Paths ----
  // TODO: Replace 'YOUR_PROJECT' with your GEE project ID.
  assets: {
    mountainRegions: 'projects/YOUR_PROJECT/assets/GMBA_USA_clipped'
  },

  // ---- Public Satellite Collections (no changes needed) ----
  satellite: {
    sentinel2: 'COPERNICUS/S2_SR',
    landsat8:  'LANDSAT/LC08/C02/T1_TOA'
  },

  // ---- Algorithm Parameters ----
  params: {
    startYear: 2018,
    endYear:   2024,
    scale:     30,             // Output spatial resolution (m)

    // Elevation binning
    elevBinWidth: 100,         // Elevation band width for snowline search (m)

    // Snowline detection thresholds
    snowFracThreshold:   0.10, // Min snow fraction per elevation band
    globalSnowThreshold: 0.001,// Min global snow fraction
    clearSkyThreshold:   0.10, // Min clear-sky fraction per band

    // Neighborhood smoothing radii (pixels)
    smoothRadius_S2: 4,
    smoothRadius_L8: 8,

    // Cloud restoration thresholds (reflectance)
    darkCloudReflectance: 0.30,// Max red reflectance for dark cloud
    brightThreshold:      0.10,// Min red reflectance to restore to cloud

    // Sentinel-2 snow thresholds
    s2_ndsi_pass1: 0.25,  s2_red_pass1: 0.20,
    s2_ndsi_pass2: 0.09,  s2_red_pass2: 0.04,

    // Landsat 8 snow thresholds
    l8_ndsi_pass1: 0.30,  l8_red_pass1: 0.20,
    l8_ndsi_pass2: 0.12,  l8_red_pass2: 0.04,

    // Terrain (reserved for slope correction)
    minCosSlope: 0.1
  },

  // ---- Export Settings ----
  // TODO: Replace 'YOUR_PROJECT' with your GEE project ID.
  export: {
    assetPrefix: 'projects/YOUR_PROJECT/assets/SCF_Annual/SCF_'
  }
};

// =============================================================================
// LOAD DATA
// =============================================================================
var GMBA_USA = ee.FeatureCollection(CONFIG.assets.mountainRegions);
var S2  = ee.ImageCollection(CONFIG.satellite.sentinel2);
var L8  = ee.ImageCollection(CONFIG.satellite.landsat8);
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation');

var studyArea = GMBA_USA.geometry();

// Unpack frequently used parameters
var SCALE = CONFIG.params.scale;
var dz    = CONFIG.params.elevBinWidth;
var fs    = CONFIG.params.snowFracThreshold;
var ft    = CONFIG.params.globalSnowThreshold;
var fct   = CONFIG.params.clearSkyThreshold;
var rf_s2 = CONFIG.params.smoothRadius_S2;
var rf_l8 = CONFIG.params.smoothRadius_L8;
var rD    = CONFIG.params.darkCloudReflectance;
var rB    = CONFIG.params.brightThreshold;

// =============================================================================
// SENTINEL-2 PREPROCESSING
// =============================================================================

/**
 * Preprocess Sentinel-2 SR image: extract spectral bands and compute NDSI.
 * Reflectance values are scaled from DN to [0, 1].
 */
function preprocessS2(img) {
  var green = img.select('B3').divide(10000).rename('GREEN');
  var red   = img.select('B4').divide(10000).rename('RED');
  var nir   = img.select('B8').divide(10000).rename('NIR');
  var swir  = img.select('B11').divide(10000).rename('SWIR');
  var scl   = img.select('SCL');
  var ndsi  = green.subtract(swir).divide(green.add(swir)).rename('NDSI');
  return ndsi.addBands([red, nir, swir, green, scl]);
}

/**
 * Generate cloud mask from Sentinel-2 Scene Classification Layer (SCL).
 * Flagged classes: 2 (dark area), 3 (cloud shadow), 8/9 (cloud), 10 (cirrus).
 */
function getS2CloudMask(img) {
  var scl = img.select('SCL');
  return scl.eq(8).or(scl.eq(9))
            .or(scl.eq(10))
            .or(scl.eq(3))
            .or(scl.eq(2))
            .rename('s2_cloud');
}

/** High-altitude cloud mask (cirrus only, SCL class 10). */
function getS2HighCloudMask(img) {
  return img.select('SCL').eq(10).rename('high_cloud');
}

/** Cloud shadow mask (SCL classes 2 and 3). */
function getS2CloudShadowMask(img) {
  var scl = img.select('SCL');
  return scl.eq(3).or(scl.eq(2)).rename('cloud_shadow');
}

/**
 * Dark cloud mask: cloud pixels where neighborhood-smoothed red reflectance
 * is below the dark threshold. These are likely true clouds (not bright snow).
 */
function getS2DarkCloudMask(img, s2Cloud) {
  var redSmoothed = img.select('RED').reduceNeighborhood({
    reducer: ee.Reducer.mean(),
    kernel: ee.Kernel.square(rf_s2, 'pixels')
  });
  return s2Cloud.and(redSmoothed.lt(rD)).rename('dark_cloud');
}

/**
 * Build the composite cloud mask for Sentinel-2.
 * Pass-1 cloud = high cloud OR shadow OR (cloud AND NOT dark cloud).
 * This preserves bright cloud pixels that may actually be snow.
 */
function processS2CloudMask(img) {
  var s2Cloud     = getS2CloudMask(img);
  var highCloud   = getS2HighCloudMask(img);
  var cloudShadow = getS2CloudShadowMask(img);
  var darkCloud   = getS2DarkCloudMask(img, s2Cloud);
  var pass1Cloud  = highCloud.or(cloudShadow).or(s2Cloud.and(darkCloud.not()));

  return ee.Dictionary({
    's2_cloud':     s2Cloud,
    'high_cloud':   highCloud,
    'cloud_shadow': cloudShadow,
    'dark_cloud':   darkCloud,
    'pass1_cloud':  pass1Cloud
  });
}

/** Pass-1 snow detection for Sentinel-2 (strict thresholds). */
function s2Pass1(img) {
  return img.select('NDSI').gte(CONFIG.params.s2_ndsi_pass1)
            .and(img.select('RED').gte(CONFIG.params.s2_red_pass1))
            .rename('pass1');
}

/** Pass-2 snow detection for Sentinel-2 (relaxed thresholds, above snowline). */
function s2Pass2(img, snowline) {
  return img.select('NDSI').gte(CONFIG.params.s2_ndsi_pass2)
            .and(img.select('RED').gte(CONFIG.params.s2_red_pass2))
            .and(dem.gte(snowline))
            .rename('pass2');
}

// =============================================================================
// LANDSAT 8 PREPROCESSING
// =============================================================================

/**
 * Preprocess Landsat 8 TOA image: extract spectral bands and compute NDSI.
 */
function preprocessL8(img) {
  var green  = img.select('B3').rename('GREEN');
  var red    = img.select('B4').rename('RED');
  var nir    = img.select('B5').rename('NIR');
  var swir   = img.select('B6').rename('SWIR');
  var cirrus = img.select('B9').rename('CIRRUS');
  var qa     = img.select('QA_PIXEL');
  var ndsi   = green.subtract(swir).divide(green.add(swir)).rename('NDSI');
  return ndsi.addBands([red, nir, swir, green, cirrus, qa]);
}

/**
 * Generate cloud mask from Landsat 8 QA_PIXEL band.
 * Uses bit flags: 1 (dilated cloud), 2 (cirrus), 3 (cloud), 4 (shadow).
 */
function getL8CloudMask(img) {
  var qa = img.select('QA_PIXEL').toInt();
  return qa.bitwiseAnd(1 << 3).neq(0)
           .or(qa.bitwiseAnd(1 << 4).neq(0))
           .or(qa.bitwiseAnd(1 << 2).neq(0))
           .or(qa.bitwiseAnd(1 << 1).neq(0))
           .rename('l8_cloud');
}

/** High-altitude cloud mask using both QA cirrus bit and Band 9. */
function getL8HighCloudMask(img) {
  var qa = img.select('QA_PIXEL').toInt();
  var qaCirrus   = qa.bitwiseAnd(1 << 2).neq(0);
  var cirrusBand = img.select('CIRRUS').gt(0.01);
  return qaCirrus.or(cirrusBand).rename('high_cloud');
}

/** Cloud shadow mask from QA_PIXEL. */
function getL8CloudShadowMask(img) {
  var qa = img.select('QA_PIXEL').toInt();
  return qa.bitwiseAnd(1 << 4).neq(0).rename('cloud_shadow');
}

/** Dark cloud mask for Landsat 8 (same logic as S2). */
function getL8DarkCloudMask(img, l8Cloud) {
  var redSmoothed = img.select('RED').reduceNeighborhood({
    reducer: ee.Reducer.mean(),
    kernel: ee.Kernel.square(rf_l8, 'pixels')
  });
  return l8Cloud.and(redSmoothed.lt(rD)).rename('dark_cloud');
}

/** Build composite cloud mask for Landsat 8. */
function processL8CloudMask(img) {
  var l8Cloud     = getL8CloudMask(img);
  var highCloud   = getL8HighCloudMask(img);
  var cloudShadow = getL8CloudShadowMask(img);
  var darkCloud   = getL8DarkCloudMask(img, l8Cloud);
  var pass1Cloud  = highCloud.or(cloudShadow).or(l8Cloud.and(darkCloud.not()));

  return ee.Dictionary({
    'l8_cloud':     l8Cloud,
    'high_cloud':   highCloud,
    'cloud_shadow': cloudShadow,
    'dark_cloud':   darkCloud,
    'pass1_cloud':  pass1Cloud
  });
}

/** Pass-1 snow detection for Landsat 8 (strict thresholds). */
function l8Pass1(img) {
  return img.select('NDSI').gte(CONFIG.params.l8_ndsi_pass1)
            .and(img.select('RED').gte(CONFIG.params.l8_red_pass1))
            .rename('pass1');
}

/** Pass-2 snow detection for Landsat 8 (relaxed thresholds, above snowline). */
function l8Pass2(img, snowline) {
  return img.select('NDSI').gte(CONFIG.params.l8_ndsi_pass2)
            .and(img.select('RED').gte(CONFIG.params.l8_red_pass2))
            .and(dem.gte(snowline))
            .rename('pass2');
}

// =============================================================================
// SNOWLINE COMPUTATION (Histogram-based)
// =============================================================================

/**
 * Compute snowline elevation using binned DEM histograms.
 *
 * Searches elevation bands either from low-to-high or high-to-low to find
 * the first band meeting both snow fraction (>= fs) and clear-sky fraction
 * (>= fct) criteria.
 *
 * @param {ee.Image} pass1Mask       - Binary Pass-1 snow mask.
 * @param {ee.Image} validMask       - Binary clear-sky mask (1 = cloud-free).
 * @param {ee.Geometry} geom         - Region of interest.
 * @param {String} searchDirection   - 'lowToHigh' or 'highToLow'.
 * @returns {ee.Number} Snowline elevation (m), or 9999 if undetermined.
 */
function computeSnowlineHistogram(pass1Mask, validMask, geom, searchDirection) {
  var maskedSnow = pass1Mask.and(validMask);
  var snowDEM = dem.updateMask(maskedSnow);

  var stats = snowDEM.reduceRegion({
    reducer: ee.Reducer.minMax().combine(ee.Reducer.count(), '', true),
    geometry: geom,
    scale: SCALE,
    maxPixels: 1e9,
    bestEffort: true
  });

  var count = ee.Number(stats.get('elevation_count'));
  var hasSnow = count.gt(50);

  var snowline = ee.Number(ee.Algorithms.If(hasSnow, (function() {
    var minZ    = ee.Number(stats.get('elevation_min')).divide(dz).floor().multiply(dz);
    var maxZ    = ee.Number(stats.get('elevation_max')).divide(dz).ceil().multiply(dz);
    var range   = maxZ.subtract(minZ).max(dz);
    var numBins = range.divide(dz).ceil().add(1).toInt();

    var demBinned = dem.subtract(minZ).divide(dz).floor().toInt().rename('bin');
    var validRange = dem.gte(minZ).and(dem.lt(maxZ.add(dz)));
    var demBinnedMasked = demBinned.updateMask(validRange);

    // Build histograms: total pixels, clear-sky pixels, snow pixels
    var totalHist = demBinnedMasked.reduceRegion({
      reducer: ee.Reducer.fixedHistogram(0, numBins, numBins),
      geometry: geom, scale: SCALE, maxPixels: 1e9, bestEffort: true
    });
    var validHist = demBinnedMasked.updateMask(validMask).reduceRegion({
      reducer: ee.Reducer.fixedHistogram(0, numBins, numBins),
      geometry: geom, scale: SCALE, maxPixels: 1e9, bestEffort: true
    });
    var snowHist = demBinnedMasked.updateMask(maskedSnow).reduceRegion({
      reducer: ee.Reducer.fixedHistogram(0, numBins, numBins),
      geometry: geom, scale: SCALE, maxPixels: 1e9, bestEffort: true
    });

    var totalArray = ee.Array(totalHist.get('bin'));
    var validArray = ee.Array(ee.Algorithms.If(
      ee.Algorithms.IsEqual(validHist.get('bin'), null),
      totalArray.multiply(0), validHist.get('bin')
    ));
    var snowArray = ee.Array(ee.Algorithms.If(
      ee.Algorithms.IsEqual(snowHist.get('bin'), null),
      totalArray.multiply(0), snowHist.get('bin')
    ));

    // Extract count columns and compute fractions
    var totalCounts = totalArray.slice(1, 1, 2).project([0]);
    var validCounts = validArray.slice(1, 1, 2).project([0]);
    var snowCounts  = snowArray.slice(1, 1, 2).project([0]);

    var one = ee.Array([1]).repeat(0, numBins);
    var clearFractions = validCounts.divide(totalCounts.max(one));
    var snowFractions  = snowCounts.divide(validCounts.max(one));

    var clearList = clearFractions.toList();
    var snowList  = snowFractions.toList();
    var indices   = ee.List.sequence(0, numBins.subtract(1));

    // Search for the qualifying elevation bin
    var resultBin = ee.Number(ee.Algorithms.If(
      ee.Algorithms.IsEqual(searchDirection, 'highToLow'),
      indices.reverse().iterate(function(idx, result) {
        idx = ee.Number(idx).toInt();
        result = ee.Number(result);
        var snowFrac  = ee.Number(snowList.get(idx));
        var clearFrac = ee.Number(clearList.get(idx));
        var meets = snowFrac.gte(fs).and(clearFrac.gte(fct));
        return ee.Number(ee.Algorithms.If(meets, idx, result));
      }, ee.Number(-1)),
      indices.iterate(function(idx, result) {
        idx = ee.Number(idx).toInt();
        result = ee.Number(result);
        var snowFrac  = ee.Number(snowList.get(idx));
        var clearFrac = ee.Number(clearList.get(idx));
        var meets = snowFrac.gte(fs).and(clearFrac.gte(fct));
        return ee.Number(ee.Algorithms.If(result.eq(-1).and(meets), idx, result));
      }, ee.Number(-1))
    ));

    return ee.Number(ee.Algorithms.If(
      resultBin.gte(0),
      ee.Algorithms.If(
        ee.Algorithms.IsEqual(searchDirection, 'highToLow'),
        minZ.add(resultBin.multiply(dz)),
        minZ.add(resultBin.subtract(2).max(0).multiply(dz))
      ),
      9999
    ));
  })(), 9999));

  return snowline;
}

// =============================================================================
// MONTHLY SNOW MASK COMPUTATION (Full Theia Pipeline)
// =============================================================================

/**
 * Compute daily snow and valid-pixel masks for one month, merging S2 and L8.
 *
 * Workflow per sensor:
 *   1. Build a monthly median mosaic to derive the monthly snowline.
 *   2. For each unique acquisition date, apply two-pass snow detection.
 *   3. Perform cloud restoration: reclassify non-snow bright pixels as cloud.
 *   4. Merge sensors (S2 takes priority on overlapping dates).
 *
 * @param {Number} year  - Calendar year.
 * @param {Number} month - Month (1–12).
 * @param {ee.Geometry} geom - Region of interest.
 * @returns {ee.Dictionary} Keys: 'snowSum', 'validSum', 'obsCount'.
 */
function computeMonthlySnowMasks(year, month, geom) {
  var startDate = ee.Date.fromYMD(year, month, 1);
  var endDate   = startDate.advance(1, 'month');

  // ---- Sentinel-2 ----
  var s2Col = S2.filterBounds(geom)
                .filterDate(startDate, endDate)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
                .select(['B3', 'B4', 'B8', 'B11', 'SCL']);

  var s2Dates = s2Col.aggregate_array('system:time_start')
                     .map(function(t) { return ee.Date(t).format('YYYY-MM-dd'); })
                     .distinct();

  // Derive monthly snowline from S2 median mosaic (high-to-low search)
  var s2MonthlySnowline = ee.Number(ee.Algorithms.If(
    s2Col.size().gt(0),
    (function() {
      var mosaic     = preprocessS2(s2Col.median().clip(geom));
      var cloudMasks = processS2CloudMask(mosaic);
      var validMask  = ee.Image(cloudMasks.get('pass1_cloud')).not();
      var pass1      = s2Pass1(mosaic).and(validMask);
      return computeSnowlineHistogram(pass1, validMask, geom, 'highToLow');
    })(),
    9999
  ));

  // Process each S2 date: two-pass detection + cloud restoration
  var s2Daily = s2Dates.map(function(dateStr) {
    var d      = ee.Date(dateStr);
    var dayCol = s2Col.filterDate(d, d.advance(1, 'day'));
    var img    = preprocessS2(dayCol.median().clip(geom));

    var cloudMasks = processS2CloudMask(img);
    var validMask  = ee.Image(cloudMasks.get('pass1_cloud')).not().unmask(0);
    var pass1      = s2Pass1(img).and(validMask).unmask(0);

    var pass2 = ee.Image(ee.Algorithms.If(
      s2MonthlySnowline.lt(9999),
      s2Pass2(img, s2MonthlySnowline).and(validMask).unmask(0),
      ee.Image.constant(0)
    ));

    var combined = pass1.or(pass2).rename('snow');

    // Cloud restoration: re-flag bright non-snow pixels as cloud
    var originalCloud  = ee.Image(cloudMasks.get('s2_cloud'));
    var highCloud      = ee.Image(cloudMasks.get('high_cloud'));
    var cloudShadow    = ee.Image(cloudMasks.get('cloud_shadow'));
    var redBand        = img.select('RED');
    var restoreToCloud = originalCloud.and(combined.not()).and(redBand.gte(rB));
    var finalCloud     = highCloud.or(cloudShadow).or(restoreToCloud);
    var finalValid     = finalCloud.not().unmask(0).rename('valid');

    return ee.Image.cat([combined.rename('snow'), finalValid])
             .set('date', dateStr).set('sensor', 'S2');
  });

  // ---- Landsat 8 ----
  var l8Col = L8.filterBounds(geom)
                .filterDate(startDate, endDate)
                .filter(ee.Filter.lt('CLOUD_COVER', 80))
                .select(['B3', 'B4', 'B5', 'B6', 'B9', 'QA_PIXEL']);

  var l8Dates = l8Col.aggregate_array('system:time_start')
                     .map(function(t) { return ee.Date(t).format('YYYY-MM-dd'); })
                     .distinct();

  // Derive monthly snowline from L8 median mosaic (low-to-high search)
  var l8MonthlySnowline = ee.Number(ee.Algorithms.If(
    l8Col.size().gt(0),
    (function() {
      var mosaic     = preprocessL8(l8Col.median().clip(geom));
      var cloudMasks = processL8CloudMask(mosaic);
      var validMask  = ee.Image(cloudMasks.get('pass1_cloud')).not();
      var pass1      = l8Pass1(mosaic).and(validMask);
      return computeSnowlineHistogram(pass1, validMask, geom, 'lowToHigh');
    })(),
    9999
  ));

  // Process each L8 date
  var l8Daily = l8Dates.map(function(dateStr) {
    var d      = ee.Date(dateStr);
    var dayCol = l8Col.filterDate(d, d.advance(1, 'day'));
    var img    = preprocessL8(dayCol.median().clip(geom));

    var cloudMasks = processL8CloudMask(img);
    var validMask  = ee.Image(cloudMasks.get('pass1_cloud')).not().unmask(0);
    var pass1      = l8Pass1(img).and(validMask).unmask(0);

    var pass2 = ee.Image(ee.Algorithms.If(
      l8MonthlySnowline.lt(9999),
      l8Pass2(img, l8MonthlySnowline).and(validMask).unmask(0),
      ee.Image.constant(0)
    ));

    var combined = pass1.or(pass2).rename('snow');

    var originalCloud  = ee.Image(cloudMasks.get('l8_cloud'));
    var highCloud      = ee.Image(cloudMasks.get('high_cloud'));
    var cloudShadow    = ee.Image(cloudMasks.get('cloud_shadow'));
    var redBand        = img.select('RED');
    var restoreToCloud = originalCloud.and(combined.not()).and(redBand.gte(rB));
    var finalCloud     = highCloud.or(cloudShadow).or(restoreToCloud);
    var finalValid     = finalCloud.not().unmask(0).rename('valid');

    return ee.Image.cat([combined.rename('snow'), finalValid])
             .set('date', dateStr).set('sensor', 'L8');
  });

  // ---- Merge: S2 priority, L8 fills remaining dates ----
  var l8OnlyDates = l8Dates.removeAll(s2Dates);
  var l8OnlyDaily = l8OnlyDates.map(function(dateStr) {
    var matches = ee.ImageCollection(l8Daily).filter(ee.Filter.eq('date', dateStr));
    return ee.Image(matches.first());
  });
  var merged = ee.ImageCollection(s2Daily.cat(l8OnlyDaily));

  return ee.Dictionary({
    'snowSum':  merged.select('snow').sum(),
    'validSum': merged.select('valid').sum(),
    'obsCount': merged.size()
  });
}

// =============================================================================
// ANNUAL SCF COMPUTATION
// =============================================================================

/**
 * Compute annual Snow Cover Frequency (SCF) for a given region.
 *
 * SCF = (total snow days / total valid observation days) * 100 [%]
 * Pixels with zero valid observations are assigned SCF = -1 (no data).
 *
 * @param {Number} year          - Calendar year.
 * @param {ee.Geometry} geom     - Region of interest.
 * @returns {ee.Image} Three-band image: SCF (%), snow_days, valid_days.
 */
function computeAnnualSCF(year, geom) {
  var months = ee.List.sequence(1, 12);

  var monthlyResults = months.map(function(m) {
    return computeMonthlySnowMasks(year, m, geom);
  });

  var totalSnowDays = ee.ImageCollection(monthlyResults.map(function(d) {
    return ee.Image(ee.Dictionary(d).get('snowSum'));
  })).sum().rename('snow_days');

  var totalValidDays = ee.ImageCollection(monthlyResults.map(function(d) {
    return ee.Image(ee.Dictionary(d).get('validSum'));
  })).sum().rename('valid_days');

  var scf = totalSnowDays.divide(totalValidDays.max(1))
                         .multiply(100)
                         .rename('SCF');
  scf = scf.where(totalValidDays.eq(0), -1);

  return scf.addBands(totalSnowDays).addBands(totalValidDays).clip(geom);
}

// =============================================================================
// MULTI-BAND STACK (Optional: all years in one image)
// =============================================================================
var years = ee.List.sequence(CONFIG.params.startYear, CONFIG.params.endYear);

var annualSCFImages = years.map(function(year) {
  year = ee.Number(year).toInt();
  var scfImage = computeAnnualSCF(year, studyArea);

  var scfBand       = scfImage.select('SCF').rename(ee.String('SCF_').cat(ee.String(year)));
  var snowDaysBand  = scfImage.select('snow_days').rename(ee.String('SnowDays_').cat(ee.String(year)));
  var validDaysBand = scfImage.select('valid_days').rename(ee.String('ValidDays_').cat(ee.String(year)));

  return scfBand.addBands(snowDaysBand).addBands(validDaysBand);
});

var scfStack = ee.ImageCollection(annualSCFImages).toBands();

// Clean auto-generated band name prefixes (e.g., "0_SCF_2018" -> "SCF_2018")
var bandNames    = scfStack.bandNames();
var newBandNames = bandNames.map(function(name) {
  return ee.String(name).replace('\\d+_', '');
});
scfStack = scfStack.rename(newBandNames);

print('Multi-band image band names:', scfStack.bandNames());

// =============================================================================
// EXPORT (per-year images to GEE Asset)
// =============================================================================
years.evaluate(function(yearList) {
  yearList.forEach(function(year) {
    var scfImage = computeAnnualSCF(year, studyArea);

    Export.image.toAsset({
      image:       scfImage,
      description: 'SCF_' + year + '_GMBA_USA',
      assetId:     CONFIG.export.assetPrefix + year + '_GMBA_USA',
      scale:       SCALE,
      region:      studyArea,
      maxPixels:   1e13,
      pyramidingPolicy: { '.default': 'mean' }
    });
  });
});

// =============================================================================
// VISUALIZATION (test a single year)
// =============================================================================
var testYear = 2020;
var testSCF  = computeAnnualSCF(testYear, studyArea);

Map.centerObject(GMBA_USA, 5);
Map.addLayer(GMBA_USA, { color: 'blue' }, 'GMBA USA');
Map.addLayer(testSCF.select('SCF'), {
  min: 0, max: 100,
  palette: ['brown', 'orange', 'yellow', 'cyan', 'blue', 'white']
}, 'SCF ' + testYear);

print('Export tasks created. Run them from the Tasks panel.');
print('Bands per year: SCF (%), snow_days, valid_days');
