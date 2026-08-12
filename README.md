# Drought Propagation Analysis

This repository contains the Google Earth Engine and Python scripts
used to analyze drought propagation from root-zone soil moisture
to vegetation dynamics across Iran during 2000–2025.

The workflow combines ecosystem-specific root-zone soil moisture
(RZSM), standardized soil moisture indices, MODIS NDVI anomalies,
lagged correlation analysis, spatial propagation mapping,
state-dependent analysis, sensitivity analyses, and moving-baseline
drought characterization.

---

## Workflow

### 1. Dynamic Ecosystem-Specific RZSM Generation

`01_dynamic_rzsm_generation.js`

Google Earth Engine is used to extract monthly ERA5-Land soil
moisture from three depth layers:

- 0–7 cm
- 7–28 cm
- 28–100 cm

Annual MODIS MCD12Q1 land-cover maps are used to identify forests,
grasslands, and croplands. Ecosystem-specific weighting factors are
then applied to generate monthly root-zone soil moisture.

**Output:**

`Iran_RZSM_312months.tif`

↓

### 2. Drought Indices and Drought Event Identification

`02_drought_indices_and_propagation.py`

The monthly RZSM dataset is used to calculate:

- SSMI1
- SSMI3
- drought events based on the Theory of Runs

SSMI3 is used as the primary drought indicator for the propagation
analysis.

**Outputs:**

- SSMI1
- SSMI3
- drought-event statistics

↓

### 3. MODIS NDVI Anomaly Generation

`03_ndvi_anomaly_generation.js`

MODIS MOD13A2 NDVI observations are aggregated to monthly values.
A monthly climatology is then used to calculate NDVI anomalies.

**Output:**

`Iran_NDVI_Anom_312months.tif`

↓

### 4. Ecosystem-Scale Drought Propagation Analysis

`04_ecosystem_propagation_analysis.py`

Ecosystem-level SSMI3 and NDVI anomaly time series are extracted
for forests, grasslands, and croplands.

Pearson correlations are calculated for propagation lags from
0 to 12 months.

**Output:**

Figure 5

↓

### 5. Pixel-Wise Drought Propagation Mapping

`05_pixelwise_propagation_mapping.py`

Pixel-wise lagged correlations are calculated to identify:

- dominant propagation lag
- maximum propagation correlation

**Outputs:**

Figures 6–7

↓

### 6. State-Dependent Propagation Analysis

`06_state_dependent_propagation.py`

Drought propagation is evaluated under different antecedent
soil-moisture states:

- Wet
- Normal
- Dry

**Output:**

Figure 8

↓

### 7. Threshold Sensitivity Analysis

`07_threshold_sensitivity_analysis.py`

The sensitivity of drought propagation to alternative drought
thresholds is evaluated.

**Output:**

Figure 9

↓

### 8. Moving-Baseline Drought Characterization

`08_nonstationary_drought_framework.py`

A moving climatological baseline is used to characterize drought
under evolving climatic conditions.

The analysis compares stationary and moving-baseline
standardization.

**Outputs:**

Figures 10–11  
Table 3

↓

### 9. Moving-Baseline Window Sensitivity

`09_window_sensitivity_analysis.py`

The robustness of the moving-baseline framework is evaluated using
alternative baseline windows:

- 8 years
- 10 years
- 12 years

The resulting propagation statistics are compared to assess the
stability of the estimated relationships.

**Outputs:**

Window-sensitivity results and associated manuscript tables/figures.

↓

### 10. Ecosystem-Specific Weighting Sensitivity

#### 10A. Google Earth Engine

`10A_weighting_sensitivity.js`

Five ecosystem-specific RZSM weighting scenarios are generated:

- Baseline
- Forest Deep
- Forest Shallow
- Grass Shallow
- Crop Deep

**Outputs:**

Scenario-specific RZSM datasets.

#### 10B. Python Analysis

`10B_weighting_sensitivity.py`

For each weighting scenario, the complete processing chain is
repeated:

RZSM → 3-month aggregation → SSMI3 → ecosystem averaging →
lagged correlation with NDVI anomalies

Peak correlation and dominant propagation lag are then compared
with the baseline scenario.

**Outputs:**

- Weighting sensitivity table
- Lag-by-lag sensitivity results
- Sensitivity curves

---

## Overall Workflow

ERA5-Land Soil Moisture  
(0–7, 7–28, 28–100 cm)

↓

MODIS MCD12Q1 Land Cover

↓

Ecosystem-Specific Weighting

↓

RZSM

↓

3-Month Aggregation

↓

SSMI3

↓

MODIS NDVI

↓

Monthly NDVI Anomalies

↓

Lagged Soil Moisture–Vegetation Correlation

↓

Ecosystem-Scale Analysis  
+  
Pixel-Wise Analysis  
+  
State-Dependent Analysis

↓

Sensitivity Analyses

- Drought threshold sensitivity
- Moving-baseline window sensitivity
- Ecosystem-specific weighting sensitivity

↓

Final Drought Propagation Results

---

## Data Sources

### ERA5-Land

Monthly soil moisture from three soil layers is obtained from the
ERA5-Land Monthly Aggregated dataset through Google Earth Engine.

### MODIS MCD12Q1

Annual land-cover data are used to identify the major ecosystem
classes and construct ecosystem-specific RZSM masks.

### MODIS MOD13A2

MODIS NDVI observations are aggregated to monthly values and used
to calculate vegetation anomalies.

---

## Study Period

2000–2025

## Study Area

Iran

## Main Ecosystems

- Forests
- Grasslands
- Croplands

## Main Drought Indicator

SSMI3

## Propagation Lag

0–12 months

Lag 0 represents a concurrent response within the same month.

---

## Software

- Google Earth Engine
- Python 3.x
- Google Colab

### Main Python packages

- xarray
- rioxarray
- pandas
- NumPy
- SciPy
- Matplotlib

---

## Reproducibility

Google Earth Engine scripts are provided for data extraction and
preprocessing. Python scripts perform the subsequent statistical
analysis, spatial processing, sensitivity analysis, and generation
of manuscript outputs.

Users should run the scripts in the indicated order because the
output of each stage is used as input for subsequent analyses.
