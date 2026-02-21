/**
 * =============================================================================
 * 01_snowline_annual_summary.js
 * =============================================================================
 * Title:       Annual Snowline Elevation Extraction for GMBA Mountain Regions
 * Description: Computes per-image snowline elevation for each GMBA-defined
 *              mountain area in the contiguous US, using both Sentinel-2 and
 *              Landsat 8 imagery (2018–2024). Results are exported as CSV
 *              files in batches to Google Drive.
 *
 * Algorithm:   For each satellite image, snow pixels are identified via NDSI
 *              thresholding. The snowline is defined as the lowest elevation
 *              band where the snow-covered fraction of clear-sky pixels
 *              exceeds a specified threshold (fs).
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
 *   - Weather station shapefile (polygons)
 *   - Weather station point locations (lon/lat)
 *   See data/README.md for download links and preprocessing steps.
 *
 * Usage:
 *   1. Upload required assets to your GEE project.
 *   2. Update CONFIG.assets paths below to point to your uploads.
 *   3. Adjust CONFIG.params if needed.
 *   4. Run in GEE Code Editor; export tasks appear in the Tasks panel.
 * =============================================================================
 */

// =============================================================================
// CONFIGURATION
// =============================================================================
var CONFIG = {
  // ---- GEE Asset Paths ----
  // TODO: Replace 'YOUR_PROJECT' with your GEE project ID.
  //       e.g., 'projects/ee-johndoe/assets/GMBA_USA_clipped'
  assets: {
    stationsShape:   'projects/YOUR_PROJECT/assets/station_shapefile',
    mountainRegions: 'projects/YOUR_PROJECT/assets/GMBA_USA_clipped',
    stationPoints:   'projects/YOUR_PROJECT/assets/station_longitude_latitude',
    fishnet:         'projects/YOUR_PROJECT/assets/GMBA_fishnet_clip'
  },

  // ---- Public Satellite Collections (no changes needed) ----
  satellite: {
    landsat8:  'LANDSAT/LC08/C02/T1_TOA',
    sentinel2: 'COPERNICUS/S2_SR'
  },

  // ---- Algorithm Parameters ----
  params: {
    startYear: 2018,
    endYear:   2024,
    batchSize: 10,             // Number of mountains per export batch
    elevBinWidth: 100,         // Elevation band width (m)
    snowFracThreshold: 0.10,   // Min snow fraction within an elevation band
    globalSnowThreshold: 0.001,// Min snow fraction over entire mountain
    clearSkyThreshold: 0.10,   // Min clear-sky fraction within an elevation band
    s2CloudMax: 80,            // Max cloud cover (%) for Sentinel-2 pre-filter
    l8CloudMax: 80             // Max cloud cover (%) for Landsat 8 pre-filter
  },

  // ---- Export Settings ----
  export: {
    folder: 'GMBA_Snowline_Results',
    columns: [
      'MountainID', 'MountainName', 'Year', 'Sensor',
      'ValidDays', 'Max_Snowline', 'Max_Date', 'Min_Snowline', 'Min_Date'
    ]
  }
};

// =============================================================================
// LOAD ASSETS
// =============================================================================
var stationsShape = ee.FeatureCollection(CONFIG.assets.stationsShape);
var allMountains  = ee.FeatureCollection(CONFIG.assets.mountainRegions);
var stationPoints = ee.FeatureCollection(CONFIG.assets.stationPoints);

var L8  = ee.ImageCollection(CONFIG.satellite.landsat8);
var S2  = ee.ImageCollection(CONFIG.satellite.sentinel2);
var dem = ee.Image('USGS/SRTMGL1_003').select('elevation');

// Unpack frequently used parameters
var dz  = CONFIG.params.elevBinWidth;
var fs  = CONFIG.params.snowFracThreshold;
var ft  = CONFIG.params.globalSnowThreshold;
var fct = CONFIG.params.clearSkyThreshold;

var totalCount = allMountains.size();
print('Total mountain regions:', totalCount);

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Extract a stable ID and human-readable name from a mountain feature.
 * Tries multiple common attribute names for robustness across different
 * versions of the GMBA inventory.
 *
 * @param {ee.Feature} mountain - A single GMBA mountain feature.
 * @returns {ee.Dictionary} With keys 'MountainID' and 'MountainName'.
 */
function getMountainIdName(mountain) {
  var d = mountain.toDictionary();
  var keys = d.keys();
  var sysIndex = mountain.get('system:index');

  var mid = ee.Algorithms.If(keys.contains('GMBA_V2_ID'), d.get('GMBA_V2_ID'),
            ee.Algorithms.If(keys.contains('GMBA_ID'), d.get('GMBA_ID'),
            ee.Algorithms.If(keys.contains('ID'), d.get('ID'),
            ee.Algorithms.If(keys.contains('OBJECTID'), d.get('OBJECTID'),
            ee.Algorithms.If(keys.contains('FID'), d.get('FID'),
            sysIndex)))));

  var mname = ee.Algorithms.If(keys.contains('MapName'), d.get('MapName'),
              ee.Algorithms.If(keys.contains('Name'), d.get('Name'),
              ee.Algorithms.If(keys.contains('NAME'), d.get('NAME'),
              ee.Algorithms.If(keys.contains('Mountain'), d.get('Mountain'),
              'Unknown'))));

  return ee.Dictionary({ 'MountainID': mid, 'MountainName': mname });
}

// =============================================================================
// SATELLITE PREPROCESSING
// =============================================================================

/**
 * Preprocess a Sentinel-2 SR image: compute NDSI, generate snow and cloud masks.
 *
 * Snow detection: NDSI >= 0.25 AND Red >= 0.20 (reflectance units).
 * Cloud mask:     SCL classes 3 (cloud shadow), 8 (cloud medium probability),
 *                 9 (cloud high probability), 10 (thin cirrus).
 *
 * @param {ee.Image} img - Sentinel-2 SR image with bands B3, B4, B11, SCL.
 * @returns {ee.Image} Original image with added 'snow' and 'cloud' bands.
 */
function preprocessS2(img) {
  var green = img.select('B3').divide(10000);
  var red   = img.select('B4').divide(10000);
  var swir  = img.select('B11').divide(10000);
  var ndsi  = green.subtract(swir).divide(green.add(swir)).rename('NDSI');

  var scl = img.select('SCL');
  var isCloud = scl.eq(3).or(scl.eq(8)).or(scl.eq(9)).or(scl.eq(10));
  var isSnow  = ndsi.gte(0.25).and(red.gte(0.20));

  return img.addBands([isSnow.rename('snow'), isCloud.rename('cloud')]);
}

/**
 * Preprocess a Landsat 8 TOA image: compute NDSI, generate snow and cloud masks.
 *
 * Snow detection: NDSI >= 0.30 AND Red >= 0.20 (TOA reflectance).
 * Cloud mask:     QA_PIXEL bits 2 (cirrus), 3 (cloud), 4 (cloud shadow).
 *
 * @param {ee.Image} img - Landsat 8 TOA image with bands B3, B4, B6, QA_PIXEL.
 * @returns {ee.Image} Original image with added 'snow' and 'cloud' bands.
 */
function preprocessL8(img) {
  var green = img.select('B3');
  var red   = img.select('B4');
  var swir  = img.select('B6');
  var ndsi  = green.subtract(swir).divide(green.add(swir));

  var qa = img.select('QA_PIXEL');
  var cloud = qa.bitwiseAnd(1 << 3).neq(0)       // Cloud
              .or(qa.bitwiseAnd(1 << 4).neq(0))   // Cloud shadow
              .or(qa.bitwiseAnd(1 << 2).neq(0));   // Cirrus

  var isSnow = ndsi.gte(0.30).and(red.gte(0.20));

  return img.addBands([isSnow.rename('snow'), cloud.rename('cloud')]);
}

// =============================================================================
// CORE SNOWLINE COMPUTATION
// =============================================================================

/**
 * Compute the snowline elevation for a single satellite image over a region.
 *
 * Method:
 *   1. Divide the DEM into elevation bands of width `dz`.
 *   2. For each band, compute the fraction of clear-sky pixels classified as snow.
 *   3. Starting from the lowest band, find the first band where:
 *      - snow fraction >= fs, AND
 *      - clear-sky fraction >= fct.
 *   4. Return that band's lower boundary minus 2*dz as the snowline.
 *   5. Return 9999 if no valid snowline is found.
 *
 * @param {ee.Image} snowImg  - Binary snow mask (1 = snow).
 * @param {ee.Image} cloudImg - Binary cloud mask (1 = cloud).
 * @param {ee.Geometry} geom  - Region of interest.
 * @param {Number} scale      - Spatial resolution in meters.
 * @returns {ee.Number} Snowline elevation (m), or 9999 if not determinable.
 */
function computeSnowlineOptimized(snowImg, cloudImg, geom, scale) {
  var valid = cloudImg.not().rename('valid');
  var snow  = snowImg.and(valid).rename('snow');
  var total = ee.Image.constant(1).rename('total');

  // Assign each pixel to an elevation band
  var demZones = dem.divide(dz).floor().multiply(dz).rename('zone');
  var stack = snow.addBands(valid).addBands(total).addBands(demZones);

  // Aggregate snow, valid, and total pixel counts per elevation band
  var stats = stack.reduceRegion({
    reducer: ee.Reducer.sum().repeat(3).group({
      groupField: 3,
      groupName: 'elevation'
    }),
    geometry: geom,
    scale: scale,
    maxPixels: 1e10,
    tileScale: 4
  });

  var safeStats = ee.Dictionary(stats).combine(
    ee.Dictionary({ 'groups': [] }), false
  );
  var groups = ee.List(safeStats.get('groups'));

  return ee.Algorithms.If(groups.size().eq(0), 9999, (function() {

    // Compute global snow fraction across all bands
    var totalStats = groups.map(function(d) {
      d = ee.Dictionary(d);
      var sums = ee.List(d.get('sum'));
      return ee.Dictionary({
        'snow':  ee.Number(sums.get(0)),
        'valid': ee.Number(sums.get(1))
      });
    });

    var sums = ee.Dictionary(totalStats.iterate(function(c, a) {
      c = ee.Dictionary(c);
      a = ee.Dictionary(a);
      return a
        .set('s', ee.Number(a.get('s')).add(ee.Number(c.get('snow'))))
        .set('v', ee.Number(a.get('v')).add(ee.Number(c.get('valid'))));
    }, ee.Dictionary({ 's': 0, 'v': 0 })));

    var vTotal = ee.Number(sums.get('v'));
    var sTotal = ee.Number(sums.get('s'));
    var totalFrac = ee.Algorithms.If(vTotal.gt(0), sTotal.divide(vTotal), 0);

    return ee.Algorithms.If(ee.Number(totalFrac).lt(ft), 9999, (function() {

      var groupsFC = ee.FeatureCollection(groups.map(function(item) {
        item = ee.Dictionary(item);
        var elev = ee.Number(item.get('elevation'));
        var valList = ee.List(item.get('sum'));
        var s = ee.Number(valList.get(0));
        var v = ee.Number(valList.get(1));
        var t = ee.Number(valList.get(2));

        var snFrac = ee.Algorithms.If(v.gt(0), s.divide(v), ee.Number(0));
        var clFrac = ee.Algorithms.If(t.gt(0), v.divide(t), ee.Number(0));

        return ee.Feature(null, {
          'elevation': elev,
          'snowFrac':  snFrac,
          'clearFrac': clFrac
        });
      }));

      var sortedFC = groupsFC.sort('elevation', true);
      var validZones = sortedFC.filter(
        ee.Filter.and(
          ee.Filter.gte('snowFrac', fs),
          ee.Filter.gte('clearFrac', fct)
        )
      );

      return ee.Algorithms.If(
        validZones.size().gt(0),
        ee.Number(validZones.first().get('elevation')).subtract(dz * 2),
        9999
      );
    })());
  })());
}

// =============================================================================
// PER-MOUNTAIN, PER-YEAR PROCESSING
// =============================================================================

/**
 * Process one mountain region for one year with one sensor.
 * Computes per-image snowline, then extracts the annual max and min
 * snowline elevations with their corresponding dates.
 *
 * @param {ee.Feature} mountain - GMBA mountain feature.
 * @param {Number} year         - Calendar year (e.g. 2020).
 * @param {String} sensor       - 'S2' for Sentinel-2 or 'L8' for Landsat 8.
 * @returns {ee.Feature} Summary feature with snowline statistics.
 */
function processMountainYear(mountain, year, sensor) {
  mountain = ee.Feature(mountain);
  var idName = getMountainIdName(mountain);
  var y = ee.Number(year).toInt();
  var geom = mountain.geometry();

  var col;
  if (sensor === 'S2') {
    col = S2.filterBounds(geom)
            .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31))
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', CONFIG.params.s2CloudMax))
            .map(preprocessS2);
  } else {
    col = L8.filterBounds(geom)
            .filterDate(ee.Date.fromYMD(y, 1, 1), ee.Date.fromYMD(y, 12, 31))
            .filter(ee.Filter.lt('CLOUD_COVER', CONFIG.params.l8CloudMax))
            .map(preprocessL8);
  }

  // Compute snowline for each image
  var daily = col.map(function(img) {
    var sl = computeSnowlineOptimized(
      img.select('snow'), img.select('cloud'), geom, 30
    );
    return ee.Feature(null, {
      'sl':   sl,
      'date': img.date().format('YYYY-MM-dd')
    });
  });

  var validResults = daily.filter(ee.Filter.lt('sl', 9000));
  var count = validResults.size();

  var maxF = ee.Feature(ee.Algorithms.If(
    count.gt(0),
    validResults.sort('sl', false).first(),
    ee.Feature(null)
  ));
  var minF = ee.Feature(ee.Algorithms.If(
    count.gt(0),
    validResults.sort('sl', true).first(),
    ee.Feature(null)
  ));

  return ee.Feature(null, {
    'MountainID':    idName.get('MountainID'),
    'MountainName':  idName.get('MountainName'),
    'Year':          y,
    'Sensor':        sensor,
    'ValidDays':     count,
    'Max_Snowline':  ee.Algorithms.If(count.gt(0), maxF.get('sl'), null),
    'Max_Date':      ee.Algorithms.If(count.gt(0), maxF.get('date'), null),
    'Min_Snowline':  ee.Algorithms.If(count.gt(0), minF.get('sl'), null),
    'Min_Date':      ee.Algorithms.If(count.gt(0), minF.get('date'), null)
  });
}

// =============================================================================
// BATCH EXPORT
// =============================================================================

var years = ee.List.sequence(CONFIG.params.startYear, CONFIG.params.endYear);
var mountainsList = allMountains.toList(10000);

totalCount.evaluate(function(total) {
  var numBatches = Math.ceil(total / CONFIG.params.batchSize);
  print('==========================================================');
  print('Mountains: ' + total +
        ' | Batch size: ' + CONFIG.params.batchSize +
        ' | Total batches: ' + numBatches);
  print('==========================================================');

  for (var i = 0; i < numBatches; i++) {
    var startIdx = i * CONFIG.params.batchSize;
    var endIdx   = (i + 1) * CONFIG.params.batchSize;
    var batchMountains = ee.FeatureCollection(
      mountainsList.slice(startIdx, endIdx)
    );

    // --- Sentinel-2 ---
    var summaryS2 = batchMountains.map(function(m) {
      return ee.FeatureCollection(years.map(function(y) {
        return processMountainYear(m, y, 'S2');
      }));
    }).flatten();

    Export.table.toDrive({
      collection:     summaryS2,
      description:    'GMBA_S2_Batch_' + i,
      fileNamePrefix: 'GMBA_USA_S2_Batch_' + i,
      folder:         CONFIG.export.folder,
      fileFormat:     'CSV',
      selectors:      CONFIG.export.columns
    });

    // --- Landsat 8 ---
    var summaryL8 = batchMountains.map(function(m) {
      return ee.FeatureCollection(years.map(function(y) {
        return processMountainYear(m, y, 'L8');
      }));
    }).flatten();

    Export.table.toDrive({
      collection:     summaryL8,
      description:    'GMBA_L8_Batch_' + i,
      fileNamePrefix: 'GMBA_USA_L8_Batch_' + i,
      folder:         CONFIG.export.folder,
      fileFormat:     'CSV',
      selectors:      CONFIG.export.columns
    });
  }

  print('Created ' + (numBatches * 2) + ' export tasks ' +
        '(' + numBatches + ' S2 + ' + numBatches + ' L8)');
  print('Go to the Tasks panel and click "Run all" or run individually.');
});

// =============================================================================
// VISUALIZATION
// =============================================================================
Map.centerObject(allMountains, 5);
Map.addLayer(allMountains, { color: 'red' }, 'GMBA USA Mountains');
Map.addLayer(stationsShape, { color: 'blue' }, 'Weather Stations', false);
