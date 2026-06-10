// =========================================================================
// Script 1: Dynamic Ecosystem-Specific Root-Zone Soil Moisture (RZSM) Extraction
// Location: Iran (Multi-temporal 2000-2025)
// Framework: Dynamic baseline matching for Non-stationary Climate Analysis
// =========================================================================

// 1. Define Spatial Boundary
var iran = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.eq('country_na', 'Iran'));

// 2. Loop through Years and Months to apply dynamic weightings based on matching Land Cover
var years = ee.List.sequence(2000, 2025);
var months = ee.List.sequence(1, 12);

var rzsmImages = years.map(function(y) {
  return months.map(function(m) {
    var start = ee.Date.fromYMD(y, m, 1);
    var end = start.advance(1, 'month');
    
    // Load monthly ERA5-Land Soil Moisture Layers
    var sm = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
      .filterDate(start, end).mean().clip(iran);
      
    var sm1 = sm.select('volumetric_soil_water_layer_1'); // 0-7 cm
    var sm2 = sm.select('volumetric_soil_water_layer_2'); // 7-28 cm
    var sm3 = sm.select('volumetric_soil_water_layer_3'); // 28-100 cm
    
    // Define Dynamic matching year for MODIS (bounded between product limits 2001-2024)
    var lcYear = ee.Number(y).max(2001).min(2024);
    var lc = ee.ImageCollection("MODIS/061/MCD12Q1")
      .filterDate(ee.Date.fromYMD(lcYear, 1, 1), ee.Date.fromYMD(lcYear, 12, 31))
      .first().select('LC_Type1').clip(iran);
      
    // Create Ecosystem Binary Masks (IGBP Classification Scheme)
    var forestMask = lc.eq(1).or(lc.eq(2)).or(lc.eq(3)).or(lc.eq(4)).or(lc.eq(5));
    var grassMask = lc.eq(10).or(lc.eq(9));
    var cropMask = lc.eq(12).or(lc.eq(14)); // Annual Croplands + Mosaic
    
    // Apply Ecosystem-Specific Weights (According to Article Table 1)
    var rzsmForest = sm1.multiply(0.20).add(sm2.multiply(0.30)).add(sm3.multiply(0.50));
    var rzsmGrass  = sm1.multiply(0.60).add(sm2.multiply(0.30)).add(sm3.multiply(0.10));
    var rzsmCrop   = sm1.multiply(0.50).add(sm2.multiply(0.30)).add(sm3.multiply(0.20));
    
    // Blend layers dynamically into a unified continuous surface
    var rzsm = rzsmForest.updateMask(forestMask)
      .blend(rzsmGrass.updateMask(grassMask))
      .blend(rzsmCrop.updateMask(cropMask));
      
    var imgId = ee.String('RZSM_').concat(ee.Number(y).format('%d')).concat('_').concat(ee.Number(m).format('%02d'));
    return rzsm.rename('RZSM')
      .set('system:time_start', start.millis())
      .set('year', y)
      .set('month', m)
      .set('system:id', imgId);
  });
}).flatten();

var rzsmCollection = ee.ImageCollection(rzsmImages);

// 3. Export Multi-band Stack to Google Drive for Python Analytics Pipeline
var rzsmStack = rzsmCollection.toBands();
Export.image.toDrive({
  image: rzsmStack,
  description: 'Iran_Dynamic_RZSM_TimeSeries_2000_2025',
  folder: 'Drought_Propagation_Project',
  fileNamePrefix: 'Iran_RZSM_312months',
  region: iran.geometry(),
  scale: 11132, // Resampled to match ERA5-Land native grid resolution (~11km)
  maxPixels: 1e13
});
