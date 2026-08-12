// =========================================================================
// Script 3: MODIS NDVI Monthly Anomaly Generation
//
// Product: MODIS/061/MOD13A2
// Period: 2000-2025
//
// Purpose:
//   1. Load MODIS NDVI
//   2. Apply MODIS scale factor
//   3. Aggregate NDVI observations to monthly means
//   4. Calculate calendar-month NDVI climatology
//   5. Calculate monthly NDVI anomalies
//
// Spatial processing:
//   Monthly NDVI anomalies are exported at a nominal 5-km scale.
//   Grid matching to the SSMI grid is performed later in Python.
// =========================================================================


// -------------------------------------------------------------------------
// 1. Define Iran boundary
// -------------------------------------------------------------------------

var iran = ee.FeatureCollection(
  "USDOS/LSIB_SIMPLE/2017"
).filter(
  ee.Filter.eq("country_na", "Iran")
);


// -------------------------------------------------------------------------
// 2. Load MODIS NDVI
// -------------------------------------------------------------------------
//
// MOD13A2 provides NDVI observations at 1-km spatial resolution
// and 16-day temporal intervals.
// -------------------------------------------------------------------------

var modisNDVI = ee.ImageCollection(
  "MODIS/061/MOD13A2"
)
.filterDate(
  "2000-01-01",
  "2025-12-31"
)
.select("NDVI");


// -------------------------------------------------------------------------
// 3. Generate Monthly NDVI Means
// -------------------------------------------------------------------------
//
// All available MODIS NDVI observations within each calendar month
// are averaged to obtain one monthly NDVI image.
// -------------------------------------------------------------------------

var years = ee.List.sequence(2000, 2025);
var months = ee.List.sequence(1, 12);

var monthlyNDVI = ee.ImageCollection(
  years.map(function(y) {

    return months.map(function(m) {

      var start = ee.Date.fromYMD(y, m, 1);
      var end = start.advance(1, "month");

      var monthlyMean = modisNDVI
        .filterDate(start, end)
        .mean()
        .multiply(0.0001)
        .rename("NDVI");

      return monthlyMean
        .clip(iran)
        .set("system:time_start", start.millis())
        .set("year", y)
        .set("month", m);

    });

  }).flatten()
);


// -------------------------------------------------------------------------
// 4. Calculate Calendar-Month NDVI Climatology
// -------------------------------------------------------------------------
//
// A separate climatological mean is calculated for each calendar month.
// For example, all January observations from 2000-2025 are used to
// calculate the January climatological mean.
// -------------------------------------------------------------------------

var climatology = ee.ImageCollection.fromImages(

  months.map(function(m) {

    var climatologyMean = monthlyNDVI
      .filter(ee.Filter.eq("month", m))
      .mean()
      .rename("NDVI_climatology");

    return climatologyMean
      .set("month", m);

  })

);


// -------------------------------------------------------------------------
// 5. Calculate Monthly NDVI Anomalies
// -------------------------------------------------------------------------
//
// NDVI anomaly = Monthly NDVI - Calendar-month climatological mean
// -------------------------------------------------------------------------

var ndviAnomalies = monthlyNDVI.map(function(img) {

  var month = img.get("month");

  var climatologyMean = climatology
    .filter(ee.Filter.eq("month", month))
    .first();

  var anomaly = img
    .select("NDVI")
    .subtract(climatologyMean)
    .rename("NDVI_anomaly");

  return anomaly
    .clip(iran)
    .copyProperties(
      img,
      [
        "system:time_start",
        "year",
        "month"
      ]
    );

});


// -------------------------------------------------------------------------
// 6. Convert Monthly NDVI Anomalies to Multi-band Stack
// -------------------------------------------------------------------------

var ndviAnomalyStack = ndviAnomalies.toBands();


// -------------------------------------------------------------------------
// 7. Export NDVI Anomaly Stack
// -------------------------------------------------------------------------
//
// The nominal export scale is 5 km.
// Final grid matching to the SSMI grid is performed later in Python.
// -------------------------------------------------------------------------

Export.image.toDrive({

  image: ndviAnomalyStack,

  description:
    "Iran_NDVI_Anomaly_2000_2025",

  folder:
    "Drought_Propagation_Project",

  fileNamePrefix:
    "Iran_NDVI_Anom_312months",

  region:
    iran.geometry(),

  scale:
    5000,

  maxPixels:
    1e13

});


print(
  "Monthly NDVI anomaly generation completed."
);
