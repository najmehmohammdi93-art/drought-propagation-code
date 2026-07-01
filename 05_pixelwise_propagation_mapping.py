# =========================================================================
# File 05: Pixel-wise Drought Propagation Mapping Engine
# Produces Dominant Lag Map (Fig 6) & Maximum Correlation Map (Fig 7)
# Reference Temporal Event: Extreme Drought 2010-2011 (Section 2-3-4 Criteria)
# =========================================================================

import xarray as xr
import numpy as np
from scipy.stats import pearsonr

# 1. Load arrays and slice specifically around the Reference Severe Drought Event (2010-2011)
ssmi3 = xr.open_rasterio("Iran_SSMI3_312months.tif").rename({'band': 'time'})
ndvi_anom = xr.open_rasterio("Iran_NDVI_Anom_312months.tif").rename({'band': 'time'})

# Filter timeline to event scope (2010-11-01 to 2011-12-31)
event_ssmi = ssmi3.sel(time=slice('2010-11-01', '2011-12-31'))
event_ndvi = ndvi_anom.sel(time=slice('2010-11-01', '2011-12-31'))

# 2. Allocate Blank Matrix Templates matching spatial resolution
y_shape, x_shape = ssmi3.shape[1], ssmi3.shape[2]
dominant_lag_map = np.full((y_shape, x_shape), np.nan)
max_corr_map = np.full((y_shape, x_shape), np.nan)

max_lag = 6 # Bounded according to Section 2-3-4 setup

print("Running heavy pixel-wise multi-dimensional cross-correlation grid loop...")
for y in range(y_shape):
    for x in range(x_shape):
        pixel_ssmi = event_ssmi[:, y, x].values
        pixel_ndvi = event_ndvi[:, y, x].values
        
        # Skip operations on masked border regions
        if np.isnan(pixel_ssmi).all() or np.isnan(pixel_ndvi).all():
            continue
            
        lags_r = []
        for lag in range(max_lag + 1):
            if lag == 0:
                r, _ = pearsonr(pixel_ssmi, pixel_ndvi)
            else:
                r, _ = pearsonr(pixel_ssmi[:-lag], pixel_ndvi[lag:])
            lags_r.append(r if not np.isnan(r) else -1)
            
        # Extract Maximum Trajectory
        highest_r = max(lags_r)
        best_lag = lags_r.index(highest_r)
        
        dominant_lag_map[y, x] = best_lag
        max_corr_map[y, x] = highest_r

# 3. Construct spatial GeoTIFF objects back from numpy arrays
dominant_lag_xr = xr.DataArray(dominant_lag_map, coords=[ssmi3.y, ssmi3.x], dims=['y', 'x'])
max_corr_xr = xr.DataArray(max_corr_map, coords=[ssmi3.y, ssmi3.x], dims=['y', 'x'])

dominant_lag_xr.rio.to_raster("Fig6_Dominant_Lag_Map.tif")
max_corr_xr.rio.to_raster("Fig7_Max_Correlation_Map.tif")
print("Spatial Raster Maps Generated Successfully (Figures 6 & 7 Exported).")
