// =========================================================================
// Script 1: Ecosystem-Specific Root-Zone Soil Moisture (RZSM)
// Study area: Iran
// Period: 2000-2025
// Data: ERA5-Land Monthly Aggregated + MODIS MCD12Q1
// =========================================================================


// -------------------------------------------------------------------------
// 1. Define Study Area
// -------------------------------------------------------------------------

var iran = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
  .filter(ee.Filter.eq('country_na', 'Iran'));


// -------------------------------------------------------------------------
// 2. Define Analysis Period
// -------------------------------------------------------------------------

var years = ee.List.sequence(2000, 2025);
var months = ee.List.sequence(1, 12);


// -------------------------------------------------------------------------
// 3. Calculate Monthly Ecosystem-Specific RZSM
// -------------------------------------------------------------------------

var rzsmImages = years.map(function(y) {

  return months.map(function(m) {

    var start = ee.Date.fromYMD(y, m, 1);
    var end = start.advance(1, 'month');


    // ---------------------------------------------------------------------
    // 3.1 Load ERA5-Land monthly soil moisture
    // ---------------------------------------------------------------------

    var sm = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
      .filterDate(start, end)
      .mean()
      .clip(iran);

    // ERA5-Land soil moisture layers:
    // Layer 1: 0-7 cm
    // Layer 2: 7-28 cm
    // Layer 3: 28-100 cm

    var sm1 = sm.select(
      'volumetric_soil_water_layer_1'
    );

    var sm2 = sm.select(
      'volumetric_soil_water_layer_2'
    );

    var sm3 = sm.select(
      'volumetric_soil_water_layer_3'
    );


    // ---------------------------------------------------------------------
    // 3.2 Match each month to the corresponding annual land-cover map
    // ---------------------------------------------------------------------
    //
    // MCD12Q1 Collection 6.1 is available from 2001 onward.
    // Therefore:
    //   2000 uses the 2001 land-cover map.
    //   2001-2024 use the corresponding annual map.
    //   2025 uses the latest available 2024 map.
    // ---------------------------------------------------------------------

    var lcYear = ee.Number(y)
      .max(2001)
      .min(2024);

    var lc = ee.ImageCollection("MODIS/061/MCD12Q1")
      .filterDate(
        ee.Date.fromYMD(lcYear, 1, 1),
        ee.Date.fromYMD(lcYear.add(1), 1, 1)
      )
      .first()
      .select('LC_Type1')
      .clip(iran);


    // ---------------------------------------------------------------------
    // 3.3 Define ecosystem masks using the IGBP classification
    // ---------------------------------------------------------------------

    // Forest classes: 1-5
    var forestMask = lc.eq(1)
      .or(lc.eq(2))
      .or(lc.eq(3))
      .or(lc.eq(4))
      .or(lc.eq(5));

    // Grassland / savanna-related classes used in the analysis
    var grassMask = lc.eq(10)
      .or(lc.eq(9));

    // Cropland classes
    var cropMask = lc.eq(12)
      .or(lc.eq(14));


    // ---------------------------------------------------------------------
    // 3.4 Calculate ecosystem-specific RZSM
    // ---------------------------------------------------------------------
    //
    // Forest:
    //   0-7 cm    = 0.20
    //   7-28 cm   = 0.30
    //   28-100 cm = 0.50
    //
    // Grassland:
    //   0-7 cm    = 0.60
    //   7-28 cm   = 0.30
    //   28-100 cm = 0.10
    //
    // Cropland:
    //   0-7 cm    = 0.50
    //   7-28 cm   = 0.30
    //   28-100 cm = 0.20
    // ---------------------------------------------------------------------

    var rzsmForest = sm1.multiply(0.20)
      .add(sm2.multiply(0.30))
      .add(sm3.multiply(0.50));

    var rzsmGrass = sm1.multiply(0.60)
      .add(sm2.multiply(0.30))
      .add(sm3.multiply(0.10));

    var rzsmCrop = sm1.multiply(0.50)
      .add(sm2.multiply(0.30))
      .add(sm3.multiply(0.20));


    // ---------------------------------------------------------------------
    // 3.5 Combine ecosystem-specific RZSM surfaces
    // ---------------------------------------------------------------------

    var rzsm = rzsmForest
      .updateMask(forestMask)
      .blend(rzsmGrass.updateMask(grassMask))
      .blend(rzsmCrop.updateMask(cropMask));


    // ---------------------------------------------------------------------
    // 3.6 Set temporal metadata
    // ---------------------------------------------------------------------

    var imgId = ee.String('RZSM_')
      .concat(ee.Number(y).format('%d'))
      .concat('_')
      .concat(ee.Number(m).format('%02d'));

    return rzsm
      .rename('RZSM')
      .set('system:time_start', start.millis())
      .set('year', y)
      .set('month', m)
      .set('landcover_year', lcYear)
      .set('system:id', imgId);

  });

}).flatten();


// Convert list to ImageCollection
var rzsmCollection = ee.ImageCollection(rzsmImages);


// -------------------------------------------------------------------------
// 4. Export Monthly RZSM Time Series
// -------------------------------------------------------------------------

var rzsmStack = rzsmCollection.toBands();

Export.image.toDrive({
  image: rzsmStack,
  description: 'Iran_Dynamic_RZSM_TimeSeries_2000_2025',
  folder: 'Drought_Propagation_Project',
  fileNamePrefix: 'Iran_RZSM_312months',
  region: iran.geometry(),
  scale: 11132,
  maxPixels: 1e13
});
