/********************************************************************
 * 03_scf_station_validation.js
 * ============================================================
 * Multi-source Snow Cover Detection & Station-level SCF Extraction
 *
 * Purpose:
 *   Filter GMBA mountain regions that contain weather stations,
 *   compute daily snow cover using Sentinel-2 + Landsat-8 fusion
 *   with temporal interpolation (Theia two-pass algorithm), then
 *   extract annual Snow Cover Frequency (SCF) at each station
 *   location for ground-truth validation.
 *
 * Data priority per day:
 *   S2 same-day > L8 same-day > S2 interpolated (≤5 d) >
 *   L8 interpolated (≤5 d) > no data
 *
 * Note on cloud handling:
 *   This script deliberately omits per-pixel cloud masking
 *   (unlike 02_scf_annual_export.js) because:
 *     (a) extraction is at station points, not wall-to-wall maps;
 *     (b) temporal interpolation fills cloudy days;
 *     (c) collection-level CLOUDY_PIXEL_PERCENTAGE pre-filter
 *         already removes heavily clouded scenes.
 *   For full cloud masking logic, see script 02.
 *
 * Outputs:
 *   CSV files exported to Google Drive with per-station SCF,
 *   snow days, and valid observation days.
 *
 * Author:      Gefei Wu
 * Affiliation: Zhejiang University
 * Date:        2026-02-21
 * License:     MIT
 ********************************************************************/

// =================================================================
// CONFIGURATION — update these before running
// =================================================================
var CONFIG = {
  // Study area: GMBA mountain polygons (CONUS subset)
  // TODO: Upload your GMBA shapefile to GEE and paste the asset path
  gmbaAsset: 'projects/YOUR_PROJECT/assets/GMBA_USA_clipped',

  // Weather station points (FeatureCollection)
  //   Required columns: "Longitude", "Latitude", "Station Id"
  // TODO: Upload your station CSV/shapefile to GEE
  stationAsset: 'projects/YOUR_PROJECT/assets/weather_stations',

  // Temporal range
  startYear: 2018,
  endYear:   2024,

  // Elevation bin width for snowline calculation (m)
  elevBinSize: 100,

  // Minimum snow fraction per elevation bin to count as "snowy"
  minSnowFractionPerBin: 0.10,

  // Maximum gap (days) for temporal interpolation
  maxGapDays: 5,

  // Sentinel-2 Theia two-pass NDSI / RED thresholds
  //   Pass 1 — conservative (applied everywhere)
  //   Pass 2 — relaxed      (applied only above snowline)
  S2_NDSI_PASS1: 0.25,  S2_RED_PASS1: 0.20,
  S2_NDSI_PASS2: 0.09,  S2_RED_PASS2: 0.04,

  // Landsat-8 Theia two-pass NDSI / RED thresholds
  L8_NDSI_PASS1: 0.30,  L8_RED_PASS1: 0.20,
  L8_NDSI_PASS2: 0.12,  L8_RED_PASS2: 0.04,

  // Export settings
  exportFolder: 'Mountain_Station_Snow_Frequency'
};

// =================================================================
// Section 1 — Load datasets
// =================================================================
var GMBA_USA = ee.FeatureCollection(CONFIG.gmbaAsset);
var stations = ee.FeatureCollection(CONFIG.stationAsset);
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation');

// Pre-filter: discard scenes with > 80 % cloud cover
var S2_SR = ee.ImageCollection('COPERNICUS/S2_SR')
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80));

var L8_TOA = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
               .filter(ee.Filter.lt('CLOUD_COVER', 80));

// =================================================================
// Section 2 — Filter mountains that contain stations
// =================================================================

/**
 * Attach point geometries to station features.
 * Expects columns "Longitude" and "Latitude" in the feature properties.
 *
 * @param {ee.FeatureCollection} stationsFC  Station feature collection
 * @returns {ee.FeatureCollection} Stations with Point geometries
 */
function prepareStations(stationsFC) {
  return stationsFC.map(function (f) {
    var lon = ee.Number(f.get('Longitude'));
    var lat = ee.Number(f.get('Latitude'));
    return f.setGeometry(ee.Geometry.Point([lon, lat]));
  });
}

var preparedStations = prepareStations(stations);

/**
 * Return mountains that spatially contain at least one station.
 *
 * @param {ee.FeatureCollection} mountains   GMBA mountain polygons
 * @param {ee.FeatureCollection} stationsFC  Station points with geometry
 * @returns {ee.FeatureCollection} Filtered mountain polygons
 */
function filterMountainsWithStations(mountains, stationsFC) {
  var stationsUnion = stationsFC.geometry();
  return mountains.filterBounds(stationsUnion)
    .filter(ee.Filter.intersects('.geo', stationsUnion));
}

var studyArea = filterMountainsWithStations(GMBA_USA, preparedStations);

print('=== Mountain filtering results ===');
print('Original mountain count:', GMBA_USA.size());
print('Mountains with stations:', studyArea.size());
print('Total stations:', preparedStations.size());

// =================================================================
// Section 3 — Preprocessing functions
// =================================================================

/**
 * Sentinel-2 SR → NDSI + RED.
 * Divides SR bands by 10 000 to convert to surface reflectance.
 *
 * @param {ee.Image} img  Sentinel-2 SR image (B3, B4, B11)
 * @returns {ee.Image} Two-band image: NDSI, RED
 */
function preprocessS2(img) {
  var red  = img.select('B4').divide(10000).rename('RED');
  var ndsi = img.normalizedDifference(['B3', 'B11']).rename('NDSI');
  return ndsi.addBands(red);
}

/**
 * Landsat-8 TOA → NDSI + RED.
 * TOA reflectance values are already in [0, 1].
 *
 * @param {ee.Image} img  Landsat-8 TOA image (B3, B4, B6)
 * @returns {ee.Image} Two-band image: NDSI, RED
 */
function preprocessL8(img) {
  var red  = img.select('B4').rename('RED');
  var ndsi = img.normalizedDifference(['B3', 'B6']).rename('NDSI');
  return ndsi.addBands(red);
}

// =================================================================
// Section 4 — Pass 1 / Pass 2 snow masks
// =================================================================

/** S2 Pass 1: conservative thresholds, applied everywhere. */
function s2Pass1Mask(img) {
  return img.select('NDSI').gte(CONFIG.S2_NDSI_PASS1)
    .and(img.select('RED').gte(CONFIG.S2_RED_PASS1))
    .rename('pass1')
    .updateMask(img.select('NDSI'));
}

/** S2 Pass 2: relaxed thresholds, applied only above snowline. */
function s2Pass2Mask(img, snowline) {
  return img.select('NDSI').gte(CONFIG.S2_NDSI_PASS2)
    .and(img.select('RED').gte(CONFIG.S2_RED_PASS2))
    .and(dem.gte(snowline))
    .rename('pass2')
    .updateMask(img.select('NDSI'));
}

/** L8 Pass 1: conservative thresholds, applied everywhere. */
function l8Pass1Mask(img) {
  return img.select('NDSI').gte(CONFIG.L8_NDSI_PASS1)
    .and(img.select('RED').gte(CONFIG.L8_RED_PASS1))
    .rename('pass1')
    .updateMask(img.select('NDSI'));
}

/** L8 Pass 2: relaxed thresholds, applied only above snowline. */
function l8Pass2Mask(img, snowline) {
  return img.select('NDSI').gte(CONFIG.L8_NDSI_PASS2)
    .and(img.select('RED').gte(CONFIG.L8_RED_PASS2))
    .and(dem.gte(snowline))
    .rename('pass2')
    .updateMask(img.select('NDSI'));
}

// =================================================================
// Section 5 — Snowline calculation
// =================================================================

/**
 * Compute snowline elevation for one image within a mountain region.
 *
 * Algorithm:
 *   1. Build elevation bins (100 m steps) between min and max
 *      snow-covered elevations.
 *   2. For each bin, compute the fraction of pixels that are snow.
 *   3. Search bins from HIGH to LOW; the lowest bin whose snow
 *      fraction >= minSnowFractionPerBin is the snowline.
 *   4. If no snow detected, return 9999 (sentinel value).
 *
 * @param {ee.Image}    snowMask     Binary pass-1 snow mask (0/1)
 * @param {ee.Geometry} mountainGeom Mountain polygon geometry
 * @param {number}      scale        Pixel resolution (m), e.g. 20 for S2
 * @param {string}      bandName     Name of the snow band in snowMask
 * @returns {ee.Number} Snowline elevation (m) or 9999
 */
function computeSnowline(snowMask, mountainGeom, scale, bandName) {
  var mask0 = snowMask.unmask(0);
  var snowDEM = dem.updateMask(mask0);

  var minDEM_raw = snowDEM.reduceRegion({
    reducer: ee.Reducer.min(), geometry: mountainGeom,
    scale: scale, maxPixels: 1e13
  }).get('elevation');

  var maxDEM_raw = snowDEM.reduceRegion({
    reducer: ee.Reducer.max(), geometry: mountainGeom,
    scale: scale, maxPixels: 1e13
  }).get('elevation');

  var hasSnow = ee.Algorithms.If(
    ee.Algorithms.IsEqual(minDEM_raw, null), false,
    ee.Algorithms.If(ee.Algorithms.IsEqual(maxDEM_raw, null), false, true)
  );

  var snowline = ee.Algorithms.If(hasSnow,
    (function () {
      var minDEM = ee.Number(minDEM_raw);
      var maxDEM = ee.Number(maxDEM_raw);
      var dz = CONFIG.elevBinSize;

      // Create elevation bins, compute snow fraction per bin
      var binStats = ee.List.sequence(minDEM, maxDEM, dz).map(function (z) {
        z = ee.Number(z);
        var binMask = dem.gte(z).and(dem.lt(z.add(dz)));
        var snowFrac = ee.Number(mask0.updateMask(binMask).reduceRegion({
          reducer: ee.Reducer.mean(), geometry: mountainGeom,
          scale: scale, maxPixels: 1e13
        }).get(bandName));
        return ee.Dictionary({ z: z, snowFrac: snowFrac });
      });

      // Reverse (high→low), keep bins with sufficient snow, take the last
      // (i.e., lowest elevation) one → snowline
      var validBins = binStats.reverse().map(function (d) {
        d = ee.Dictionary(d);
        return ee.Algorithms.If(
          ee.Number(d.get('snowFrac')).gte(CONFIG.minSnowFractionPerBin),
          d.get('z'), null
        );
      }).removeAll([null]);

      return ee.Algorithms.If(validBins.size().gt(0), validBins.get(-1), 9999);
    })(),
    9999
  );
  return ee.Number(snowline);
}

// =================================================================
// Section 6 — Snow detection (Pass 1 + Pass 2 combined)
// =================================================================

/**
 * Combine pass-1 and pass-2 snow masks into a single binary snow image.
 * Pass 2 is only applied when a valid snowline (< 9999) exists.
 *
 * @param {ee.Image}    processedImg Preprocessed NDSI+RED image
 * @param {ee.Number}   snowline     Snowline elevation from pass 1
 * @param {Function}    pass1Fn      Pass-1 mask function
 * @param {Function}    pass2Fn      Pass-2 mask function
 * @returns {ee.Image}  Binary snow image (band: 'snow')
 */
function detectSnow(processedImg, snowline, pass1Fn, pass2Fn) {
  var s1 = pass1Fn(processedImg).unmask(0);
  var snowlineNum = ee.Number(snowline);
  var s2 = ee.Image(0).rename('pass2');
  s2 = ee.Image(ee.Algorithms.If(
    snowlineNum.lt(9999),
    pass2Fn(processedImg, snowlineNum).unmask(0),
    s2
  ));
  return s1.or(s2).rename('snow');
}

// =================================================================
// Section 7 — Single-day processing
// =================================================================

/**
 * Process all S2 images for one day over a mountain.
 * Composites via median if multiple images exist.
 *
 * @param {ee.ImageCollection} s2Col        S2 collection (pre-filtered)
 * @param {ee.Geometry}        mountainGeom Mountain polygon
 * @param {string}             dateStr      'YYYY-MM-dd'
 * @returns {ee.Image|null} Snow image or null if no data
 */
function processS2Day(s2Col, mountainGeom, dateStr) {
  var dateStart = ee.Date(dateStr);
  var dayImages = s2Col.filterDate(dateStart, dateStart.advance(1, 'day'))
    .filterBounds(mountainGeom).select(['B3', 'B4', 'B8', 'B11']);

  return ee.Algorithms.If(dayImages.size().gt(0), (function () {
    var processed = preprocessS2(dayImages.median().clip(mountainGeom));
    var snowline = computeSnowline(s2Pass1Mask(processed), mountainGeom, 20, 'pass1');
    return detectSnow(processed, snowline, s2Pass1Mask, s2Pass2Mask)
      .set('date', dateStr).set('source', 'S2').set('snowline', snowline);
  })(), null);
}

/**
 * Process all L8 images for one day over a mountain.
 *
 * @param {ee.ImageCollection} l8Col        L8 collection (pre-filtered)
 * @param {ee.Geometry}        mountainGeom Mountain polygon
 * @param {string}             dateStr      'YYYY-MM-dd'
 * @returns {ee.Image|null} Snow image or null if no data
 */
function processL8Day(l8Col, mountainGeom, dateStr) {
  var dateStart = ee.Date(dateStr);
  var dayImages = l8Col.filterDate(dateStart, dateStart.advance(1, 'day'))
    .filterBounds(mountainGeom).select(['B3', 'B4', 'B5', 'B6']);

  return ee.Algorithms.If(dayImages.size().gt(0), (function () {
    var processed = preprocessL8(dayImages.median().clip(mountainGeom));
    var snowline = computeSnowline(l8Pass1Mask(processed), mountainGeom, 30, 'pass1');
    return detectSnow(processed, snowline, l8Pass1Mask, l8Pass2Mask)
      .set('date', dateStr).set('source', 'L8').set('snowline', snowline);
  })(), null);
}

/**
 * Find the nearest image within ±maxDays of a target date.
 * Returns a Dictionary with keys 'image' and 'gap'.
 *
 * @param {ee.ImageCollection} col          Sensor collection
 * @param {ee.Geometry}        mountainGeom Mountain polygon
 * @param {ee.Date}            targetDate   Date to search around
 * @param {number}             maxDays      Maximum gap in days
 * @param {Array<string>}      bands        Band names to select
 * @returns {ee.Dictionary} {image: ee.Image|null, gap: ee.Number}
 */
function findNearestImage(col, mountainGeom, targetDate, maxDays, bands) {
  var millis = targetDate.millis();
  var before = col.filterBounds(mountainGeom)
    .filterDate(targetDate.advance(-maxDays, 'day'), targetDate)
    .select(bands).sort('system:time_start', false);
  var after = col.filterBounds(mountainGeom)
    .filterDate(targetDate.advance(1, 'day'), targetDate.advance(maxDays + 1, 'day'))
    .select(bands).sort('system:time_start', true);

  var beforeImg = before.first();
  var afterImg  = after.first();

  var beforeGap = ee.Algorithms.If(
    ee.Algorithms.IsEqual(beforeImg, null), ee.Number(999),
    ee.Number(millis).subtract(ee.Number(ee.Image(beforeImg).get('system:time_start')))
      .divide(86400000).abs()
  );
  var afterGap = ee.Algorithms.If(
    ee.Algorithms.IsEqual(afterImg, null), ee.Number(999),
    ee.Number(ee.Image(afterImg).get('system:time_start')).subtract(millis)
      .divide(86400000).abs()
  );

  return ee.Dictionary(ee.Algorithms.If(
    ee.Number(beforeGap).lte(ee.Number(afterGap)),
    ee.Dictionary({ image: beforeImg, gap: beforeGap }),
    ee.Dictionary({ image: afterImg,  gap: afterGap })
  ));
}

/**
 * Daily composite main function.
 * Priority: S2 today → L8 today → S2 interp (≤5 d) → L8 interp (≤5 d) → null
 *
 * @param {string}             dateStr      'YYYY-MM-dd'
 * @param {ee.Geometry}        mountainGeom Mountain polygon
 * @param {ee.ImageCollection} s2Col        Sentinel-2 collection
 * @param {ee.ImageCollection} l8Col        Landsat-8 collection
 * @returns {ee.Image|null} Best available snow image for the day
 */
function processSingleDay(dateStr, mountainGeom, s2Col, l8Col) {
  var targetDate = ee.Date(dateStr);
  var maxDays = CONFIG.maxGapDays;

  // Same-day composites
  var s2Today = processS2Day(s2Col, mountainGeom, dateStr);
  var l8Today = processL8Day(l8Col, mountainGeom, dateStr);

  // S2 temporal interpolation
  var s2Near = findNearestImage(s2Col, mountainGeom, targetDate, maxDays,
                                ['B3', 'B4', 'B8', 'B11']);
  var s2Interp = ee.Algorithms.If(
    ee.Number(s2Near.get('gap')).lte(maxDays),
    ee.Algorithms.If(ee.Algorithms.IsEqual(s2Near.get('image'), null), null,
      (function () {
        var proc = preprocessS2(ee.Image(s2Near.get('image')).clip(mountainGeom));
        var sl = computeSnowline(s2Pass1Mask(proc), mountainGeom, 20, 'pass1');
        return detectSnow(proc, sl, s2Pass1Mask, s2Pass2Mask)
          .set('date', dateStr).set('source', 'S2_interp')
          .set('gap_days', s2Near.get('gap')).set('snowline', sl);
      })()
    ), null
  );

  // L8 temporal interpolation
  var l8Near = findNearestImage(l8Col, mountainGeom, targetDate, maxDays,
                                ['B3', 'B4', 'B5', 'B6']);
  var l8Interp = ee.Algorithms.If(
    ee.Number(l8Near.get('gap')).lte(maxDays),
    ee.Algorithms.If(ee.Algorithms.IsEqual(l8Near.get('image'), null), null,
      (function () {
        var proc = preprocessL8(ee.Image(l8Near.get('image')).clip(mountainGeom));
        var sl = computeSnowline(l8Pass1Mask(proc), mountainGeom, 30, 'pass1');
        return detectSnow(proc, sl, l8Pass1Mask, l8Pass2Mask)
          .set('date', dateStr).set('source', 'L8_interp')
          .set('gap_days', l8Near.get('gap')).set('snowline', sl);
      })()
    ), null
  );

  // Priority cascade: S2 > L8 > S2_interp > L8_interp > null
  return ee.Algorithms.If(ee.Algorithms.IsEqual(s2Today, null),
    ee.Algorithms.If(ee.Algorithms.IsEqual(l8Today, null),
      ee.Algorithms.If(ee.Algorithms.IsEqual(s2Interp, null),
        ee.Algorithms.If(ee.Algorithms.IsEqual(l8Interp, null),
          null, l8Interp),
        s2Interp),
      l8Today),
    s2Today);
}

// =================================================================
// Section 8 — Annual SCF computation
// =================================================================

/**
 * Build a daily ImageCollection for one mountain-year.
 * Extends the date range by ±maxGapDays for interpolation.
 *
 * @param {ee.Feature} mountain  Mountain polygon feature
 * @param {number}     year      Calendar year (e.g. 2020)
 * @returns {ee.ImageCollection} Daily snow images (nulls removed)
 */
function processMountainYear(mountain, year) {
  var geom = mountain.geometry();
  var start = ee.Date.fromYMD(year, 1, 1);
  var end   = ee.Date.fromYMD(year, 12, 31);
  var ext   = CONFIG.maxGapDays + 1;

  // Extend temporal window for interpolation look-ahead/back
  var s2Year = S2_SR.filterDate(start.advance(-ext, 'day'), end.advance(ext, 'day'));
  var l8Year = L8_TOA.filterDate(start.advance(-ext, 'day'), end.advance(ext, 'day'));

  var numDays = end.difference(start, 'day').add(1);
  var dates = ee.List.sequence(0, numDays.subtract(1)).map(function (d) {
    return start.advance(d, 'day').format('YYYY-MM-dd');
  });

  var results = dates.map(function (dateStr) {
    return processSingleDay(dateStr, geom, s2Year, l8Year);
  });
  return ee.ImageCollection(results.removeAll([null]));
}

/**
 * Compute annual SCF image: snow frequency, snow days, valid days.
 *
 * @param {ee.Feature} mountain  Mountain polygon feature
 * @param {number}     year      Calendar year
 * @returns {ee.Image} Three-band image (snow_frequency, snow_days, valid_days)
 */
function computeAnnualSCF(mountain, year) {
  var geom = mountain.geometry();
  var daily = processMountainYear(mountain, year);

  var validDays = daily.select('snow').count().rename('valid_days');
  var snowDays  = daily.select('snow').sum().rename('snow_days');
  var scf       = snowDays.divide(validDays).rename('snow_frequency');

  return scf.addBands(snowDays).addBands(validDays)
    .clip(geom).set('mountain', mountain.get('Name')).set('year', year);
}

// =================================================================
// Section 9 — Station-level SCF extraction & export
// =================================================================

/**
 * Get stations located inside a given mountain polygon.
 *
 * @param {ee.Feature}           mountain   Mountain polygon feature
 * @param {ee.FeatureCollection} stationsFC Station point features
 * @returns {ee.FeatureCollection} Stations within the mountain
 */
function getStationsInMountain(mountain, stationsFC) {
  return stationsFC.filterBounds(mountain.geometry());
}

/**
 * Sample SCF raster at a single station point.
 *
 * @param {ee.Feature} station      Station feature (with geometry)
 * @param {ee.Image}   annualImg    Annual SCF image from computeAnnualSCF
 * @param {number}     year         Calendar year
 * @param {ee.String}  mountainName Mountain name string
 * @returns {ee.Feature} Point-less feature with SCF attributes
 */
function extractStationSCF(station, annualImg, year, mountainName) {
  var sampled = annualImg.reduceRegion({
    reducer: ee.Reducer.first(), geometry: station.geometry(),
    scale: 30, maxPixels: 1e9
  });
  return ee.Feature(null, {
    Station_Id:      station.get('Station Id'),
    Latitude:        station.get('Latitude'),
    Longitude:       station.get('Longitude'),
    Year:            year,
    Snow_Frequency:  sampled.get('snow_frequency'),
    Snow_Days:       sampled.get('snow_days'),
    Valid_Days:      sampled.get('valid_days'),
    Mountain_Name:   mountainName
  });
}

/**
 * Process all stations × all years for one mountain.
 *
 * @param {ee.Feature}           mountain   Mountain polygon
 * @param {ee.FeatureCollection} stationsFC Prepared station features
 * @param {number}               startY     First year (inclusive)
 * @param {number}               endY       Last year (inclusive)
 * @returns {ee.FeatureCollection} One row per station per year
 */
function processMountainStations(mountain, stationsFC, startY, endY) {
  var name = mountain.get('Name');
  var mtStations = getStationsInMountain(mountain, stationsFC);
  var years = ee.List.sequence(startY, endY);

  var allResults = years.map(function (year) {
    year = ee.Number(year).toInt();
    var annualImg = computeAnnualSCF(mountain, year);
    var stList = mtStations.toList(mtStations.size());
    return stList.map(function (s) {
      return extractStationSCF(ee.Feature(s), annualImg, year, name);
    });
  });
  return ee.FeatureCollection(allResults.flatten());
}

// =================================================================
// Section 10 — Export functions
// =================================================================

var CSV_COLUMNS = ['Station_Id', 'Latitude', 'Longitude', 'Year',
                   'Snow_Frequency', 'Snow_Days', 'Valid_Days', 'Mountain_Name'];

/** Export one CSV per mountain (each contains all stations × years). */
function exportByMountain() {
  studyArea.aggregate_array('Name').evaluate(function (names) {
    print('Starting per-mountain export. Count:', names.length);
    names.forEach(function (name) {
      var safeName = name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '');
      var mountain = studyArea.filter(ee.Filter.eq('Name', name)).first();
      var results = processMountainStations(
        mountain, preparedStations, CONFIG.startYear, CONFIG.endYear
      );
      Export.table.toDrive({
        collection: results,
        description: 'StationSCF_' + safeName,
        folder: CONFIG.exportFolder,
        fileNamePrefix: 'StationSCF_' + safeName,
        fileFormat: 'CSV',
        selectors: CSV_COLUMNS
      });
      print('Export task created: ' + name);
    });
    print('All tasks created — run them from the Tasks panel.');
  });
}

/** Export a single merged CSV for all mountains. */
function exportAllMerged() {
  var mountainList = studyArea.toList(studyArea.size());
  var allResults = mountainList.map(function (mt) {
    return processMountainStations(
      ee.Feature(mt), preparedStations, CONFIG.startYear, CONFIG.endYear
    );
  });
  var merged = ee.FeatureCollection(allResults).flatten();

  Export.table.toDrive({
    collection: merged,
    description: 'All_Mountains_StationSCF_' + CONFIG.startYear + '_' + CONFIG.endYear,
    folder: CONFIG.exportFolder,
    fileNamePrefix: 'All_Mountains_StationSCF_' + CONFIG.startYear + '_' + CONFIG.endYear,
    fileFormat: 'CSV',
    selectors: CSV_COLUMNS
  });
  print('Merged CSV export task created.');
}

/** Export annual SCF raster images to Drive. */
function exportAllImages() {
  studyArea.aggregate_array('Name').evaluate(function (names) {
    names.forEach(function (name) {
      var safeName = name.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '');
      var mountain = studyArea.filter(ee.Filter.eq('Name', name)).first();
      for (var y = CONFIG.startYear; y <= CONFIG.endYear; y++) {
        Export.image.toDrive({
          image: computeAnnualSCF(mountain, y),
          description: 'SCF_' + safeName + '_' + y,
          folder: 'Snow_Frequency_Fused',
          fileNamePrefix: 'SCF_' + safeName + '_' + y,
          region: mountain.geometry(),
          scale: 30, crs: 'EPSG:4326', maxPixels: 1e13
        });
      }
    });
  });
}

// =================================================================
// Section 11 — Visualization & execution
// =================================================================

Map.centerObject(studyArea, 5);
Map.addLayer(GMBA_USA, { color: 'gray' },  'All GMBA mountains', false);
Map.addLayer(studyArea, { color: 'blue' },  'Mountains with stations');
Map.addLayer(preparedStations, { color: 'red' }, 'Weather stations');

print('');
print('=== Usage ===');
print('exportByMountain()  — one CSV per mountain');
print('exportAllMerged()   — single merged CSV');
print('exportAllImages()   — SCF raster images');

// Uncomment ONE of the following to run:
exportByMountain();
// exportAllMerged();
// exportAllImages();
