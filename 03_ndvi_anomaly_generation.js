// =========================================================================
// File 03: MODIS NDVI Monthly Anomaly Generation Pipeline
// Framework: De-seasoning and Climatology Baseline (2000-2025)
// =========================================================================

var iran = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na', 'Iran'));

// 1. Load MODIS NDVI (MOD13Q1 / 250m) and filter by timeline
var modisNDVI = ee.ImageCollection("MODIS/061/MOD13Q1")
  .filterDate('2000-01-01', '2025-12-31')
  .select('NDVI');

// 2. Generate Monthly Aggregates to match ERA5-Land temporal resolution
var years = ee.List.sequence(2000, 2025);
var months = ee.List.sequence(1, 12);

var monthlyNDVI = ee.ImageCollection(years.map(function(y) {
  return months.map(function(m) {
    var start = ee.Date.fromYMD(y, m, 1);
    var end = start.advance(1, 'month');
    var monthlyMean = modisNDVI.filterDate(start, end).mean();
    return monthlyMean.multiply(0.0001) // Apply MODIS scale factor
      .rename('NDVI_mean')
      .set('system:time_start', start.millis())
      .set('year', y).set('month', m);
  });
}).flatten());

// 3. Compute Long-term Monthly Climatology (Baseline for De-seasoning)
var monthsInList = ee.List.sequence(1, 12);
var climatologyMeans = ee.ImageCollection.fromImages(monthsInList.map(function(m) {
  var meanImage = monthlyNDVI.filter(ee.Filter.eq('month', m)).mean();
  return meanImage.rename('NDVI_clim_mean').set('month', m);
}));

// 4. Extract NDVI Anomalies (According to Equation in Section 2-3-3)
var ndviAnomalyCollection = monthlyNDVI.map(function(img) {
  var m = img.get('month');
  var climMean = climatologyMeans.filter(ee.Filter.eq('month', m)).first();
  var anomaly = img.select('NDVI_mean').subtract(climMean)
    .rename('NDVI_anomaly');
  return anomaly.copyProperties(img, ['system:time_start', 'year', 'month']);
});

// 5. Export Multi-band Anomaly Raster Stack scaled to ERA5 grid (~11km)
var ndviAnomalyStack = ndviAnomalyCollection.toBands();
Export.image.toDrive({
  image: ndviAnomalyStack,
  description: 'Iran_NDVI_Anomaly_TimeSeries_2000_2025',
  folder: 'Drought_Propagation_Project',
  fileNamePrefix: 'Iran_NDVI_Anom_312months',
  region: iran.geometry(),
  scale: 11132, 
  maxPixels: 1e13
});
