// =========================================================================
// Script 10A: Ecosystem-Specific Weighting Sensitivity - GEE
//
// Purpose:
//   Generate RZSM datasets for the baseline and four alternative
//   ecosystem-specific weighting scenarios.
//
// Data:
//   ERA5-Land Monthly Aggregated
//   MODIS MCD12Q1 Collection 6.1
//
// Study period:
//   2000-2025
//
// Soil layers:
//   Layer 1: 0-7 cm
//   Layer 2: 7-28 cm
//   Layer 3: 28-100 cm
//
// Ecosystems:
//   Forest
//   Grassland
//   Cropland
//
// Scenarios:
//   Baseline
//   Forest Deep
//   Forest Shallow
//   Grass Shallow
//   Crop Deep
//
// The same land-cover matching procedure used in Script 1 is retained.
// =========================================================================


// -------------------------------------------------------------------------
// 1. Study area
// -------------------------------------------------------------------------

var iran = ee.FeatureCollection(
  "USDOS/LSIB_SIMPLE/2017"
).filter(
  ee.Filter.eq('country_na', 'Iran')
);


// -------------------------------------------------------------------------
// 2. Analysis period
// -------------------------------------------------------------------------

var years = ee.List.sequence(
  2000,
  2025
);

var months = ee.List.sequence(
  1,
  12
);


// -------------------------------------------------------------------------
// 3. Weighting scenarios
//
// Order:
//   [0-7 cm, 7-28 cm, 28-100 cm]
// -------------------------------------------------------------------------

var scenarios = {

  Baseline: {
    forest:  [0.20, 0.30, 0.50],
    grass:   [0.60, 0.30, 0.10],
    crop:    [0.50, 0.30, 0.20]
  },

  Forest_Deep: {
    forest:  [0.15, 0.30, 0.55],
    grass:   [0.60, 0.30, 0.10],
    crop:    [0.50, 0.30, 0.20]
  },

  Forest_Shallow: {
    forest:  [0.25, 0.30, 0.45],
    grass:   [0.60, 0.30, 0.10],
    crop:    [0.50, 0.30, 0.20]
  },

  Grass_Shallow: {
    forest:  [0.20, 0.30, 0.50],
    grass:   [0.70, 0.20, 0.10],
    crop:    [0.50, 0.30, 0.20]
  },

  Crop_Deep: {
    forest:  [0.20, 0.30, 0.50],
    grass:   [0.60, 0.30, 0.10],
    crop:    [0.40, 0.30, 0.30]
  }

};


// -------------------------------------------------------------------------
// 4. Function to calculate one scenario
// -------------------------------------------------------------------------

function buildScenario(
  forestWeights,
  grassWeights,
  cropWeights
) {

  var images = years.map(function(y) {

    return months.map(function(m) {

      var start = ee.Date.fromYMD(
        y,
        m,
        1
      );

      var end = start.advance(
        1,
        'month'
      );


      // ---------------------------------------------------------------
      // 4.1 ERA5-Land
      // ---------------------------------------------------------------

      var sm = ee.ImageCollection(
        "ECMWF/ERA5_LAND/MONTHLY_AGGR"
      )
      .filterDate(
        start,
        end
      )
      .mean()
      .clip(iran);


      var sm1 = sm.select(
        'volumetric_soil_water_layer_1'
      );

      var sm2 = sm.select(
        'volumetric_soil_water_layer_2'
      );

      var sm3 = sm.select(
        'volumetric_soil_water_layer_3'
      );


      // ---------------------------------------------------------------
      // 4.2 Match annual land cover
      //
      // Same logic as Script 1:
      // 2000 -> 2001
      // 2001-2024 -> corresponding year
      // 2025 -> 2024
      // ---------------------------------------------------------------

      var lcYear = ee.Number(y)
        .max(2001)
        .min(2024);


      var lc = ee.ImageCollection(
        "MODIS/061/MCD12Q1"
      )
      .filterDate(
        ee.Date.fromYMD(
          lcYear,
          1,
          1
        ),
        ee.Date.fromYMD(
          lcYear.add(1),
          1,
          1
        )
      )
      .first()
      .select(
        'LC_Type1'
      )
      .clip(iran);


      // ---------------------------------------------------------------
      // 4.3 Ecosystem masks
      // ---------------------------------------------------------------

      var forestMask = lc.eq(1)
        .or(lc.eq(2))
        .or(lc.eq(3))
        .or(lc.eq(4))
        .or(lc.eq(5));


      var grassMask = lc.eq(9)
        .or(lc.eq(10));


      var cropMask = lc.eq(12)
        .or(lc.eq(14));


      // ---------------------------------------------------------------
      // 4.4 Apply scenario-specific weights
      // ---------------------------------------------------------------

      var forestRZSM =
        sm1.multiply(
          forestWeights[0]
        )
        .add(
          sm2.multiply(
            forestWeights[1]
          )
        )
        .add(
          sm3.multiply(
            forestWeights[2]
          )
        );


      var grassRZSM =
        sm1.multiply(
          grassWeights[0]
        )
        .add(
          sm2.multiply(
            grassWeights[1]
          )
        )
        .add(
          sm3.multiply(
            grassWeights[2]
          )
        )
        );


      var cropRZSM =
        sm1.multiply(
          cropWeights[0]
        )
        .add(
          sm2.multiply(
            cropWeights[1]
          )
        )
        .add(
          sm3.multiply(
            cropWeights[2]
          )
        );


      // ---------------------------------------------------------------
      // 4.5 Combine ecosystems
      // ---------------------------------------------------------------

      var rzsm =
        forestRZSM
        .updateMask(
          forestMask
        )
        .blend(
          grassRZSM.updateMask(
            grassMask
          )
        )
        .blend(
          cropRZSM.updateMask(
            cropMask
          )
        );


      return rzsm
        .rename('RZSM')
        .set(
          'system:time_start',
          start.millis()
        )
        .set(
          'year',
          y
        )
        .set(
          'month',
          m
        )
        .set(
          'landcover_year',
          lcYear
        );

    });

  }).flatten();


  return ee.ImageCollection(
    images
  );
}


// -------------------------------------------------------------------------
// 5. Build all five scenarios
// -------------------------------------------------------------------------

var baseline = buildScenario(
  scenarios.Baseline.forest,
  scenarios.Baseline.grass,
  scenarios.Baseline.crop
);


var forestDeep = buildScenario(
  scenarios.Forest_Deep.forest,
  scenarios.Forest_Deep.grass,
  scenarios.Forest_Deep.crop
);


var forestShallow = buildScenario(
  scenarios.Forest_Shallow.forest,
  scenarios.Forest_Shallow.grass,
  scenarios.Forest_Shallow.crop
);


var grassShallow = buildScenario(
  scenarios.Grass_Shallow.forest,
  scenarios.Grass_Shallow.grass,
  scenarios.Grass_Shallow.crop
);


var cropDeep = buildScenario(
  scenarios.Crop_Deep.forest,
  scenarios.Crop_Deep.grass,
  scenarios.Crop_Deep.crop
);


// -------------------------------------------------------------------------
// 6. Export helper
// -------------------------------------------------------------------------

function exportScenario(
  collection,
  name
) {

  var stack =
    collection.toBands();

  Export.image.toDrive({

    image: stack,

    description:
      'Iran_WeightSensitivity_' +
      name,

    folder:
      'Drought_Propagation_Project',

    fileNamePrefix:
      'Iran_RZSM_' +
      name,

    region:
      iran.geometry(),

    scale:
      11132,

    maxPixels:
      1e13

  });

}


// -------------------------------------------------------------------------
// 7. Export all scenarios
// -------------------------------------------------------------------------

exportScenario(
  baseline,
  'Baseline'
);


exportScenario(
  forestDeep,
  'ForestDeep'
);


exportScenario(
  forestShallow,
  'ForestShallow'
);


exportScenario(
  grassShallow,
  'GrassShallow'
);


exportScenario(
  cropDeep,
  'CropDeep'
);


// -------------------------------------------------------------------------
// End
// -------------------------------------------------------------------------

print(
  'Weighting sensitivity exports created.'
);

print(
  'Scenarios:',
  [
    'Baseline',
    'ForestDeep',
    'ForestShallow',
    'GrassShallow',
    'CropDeep'
  ]
);
