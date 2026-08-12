# =========================================================================
# Script 5: Pixel-wise Drought Propagation Mapping
#
# Purpose:
#   Calculate pixel-wise:
#     1. Dominant drought propagation lag
#     2. Maximum SSMI3-NDVI correlation
#
# Analysis period: 2000-2025
# Propagation lags: 0-12 months
#
# Outputs:
#   Figure 4 - Dominant propagation lag map
#   Figure 5 - Maximum propagation correlation map
# =========================================================================


import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from scipy.stats import pearsonr


# -------------------------------------------------------------------------
# 1. Load SSMI3 and NDVI anomaly datasets
# -------------------------------------------------------------------------

print("Loading SSMI3 and NDVI anomaly datasets...")

ssmi3_stack = rxr.open_rasterio(
    "Iran_SSMI3_312months.tif"
)

ndvi_anom_stack = rxr.open_rasterio(
    "Iran_NDVI_Anom_312months.tif"
)


# -------------------------------------------------------------------------
# 2. Define monthly time dimension
# -------------------------------------------------------------------------

timeline = pd.date_range(
    start="2000-01-01",
    end="2025-12-01",
    freq="MS"
)

if ssmi3_stack.sizes["band"] != len(timeline):
    raise ValueError(
        "SSMI3 does not contain 312 monthly observations."
    )

if ndvi_anom_stack.sizes["band"] != len(timeline):
    raise ValueError(
        "NDVI anomaly dataset does not contain 312 monthly observations."
    )


ssmi3 = (
    ssmi3_stack
    .assign_coords(band=timeline)
    .rename({"band": "time"})
)

ndvi_anom = (
    ndvi_anom_stack
    .assign_coords(band=timeline)
    .rename({"band": "time"})
)


# -------------------------------------------------------------------------
# 3. Check spatial dimensions
# -------------------------------------------------------------------------

if (
    ssmi3.sizes["y"] != ndvi_anom.sizes["y"]
    or ssmi3.sizes["x"] != ndvi_anom.sizes["x"]
):
    raise ValueError(
        "SSMI3 and NDVI anomaly grids have different spatial dimensions."
    )


# -------------------------------------------------------------------------
# 4. Define maximum propagation lag
# -------------------------------------------------------------------------

max_lag = 12

lags = np.arange(
    0,
    max_lag + 1
)


# -------------------------------------------------------------------------
# 5. Prepare output arrays
# -------------------------------------------------------------------------

y_size = ssmi3.sizes["y"]
x_size = ssmi3.sizes["x"]

dominant_lag_map = np.full(
    (y_size, x_size),
    np.nan,
    dtype=np.float32
)

max_corr_map = np.full(
    (y_size, x_size),
    np.nan,
    dtype=np.float32
)


# -------------------------------------------------------------------------
# 6. Pixel-wise lagged correlation
# -------------------------------------------------------------------------
#
# Lag 0:
#   Concurrent monthly variability.
#
# Positive lag:
#   NDVI response follows the preceding SSMI3 condition.
#
# For each pixel, the dominant lag is the lag associated with the
# maximum Pearson correlation coefficient.
# -------------------------------------------------------------------------

print(
    "Calculating pixel-wise correlations for lags 0-12 months..."
)


for y in range(y_size):

    if y % 25 == 0:
        print(
            f"Processing row {y + 1} of {y_size}..."
        )

    for x in range(x_size):

        pixel_ssmi = ssmi3[:, y, x].values
        pixel_ndvi = ndvi_anom[:, y, x].values


        # ---------------------------------------------------------------
        # Pairwise valid observations
        # ---------------------------------------------------------------

        valid = (
            np.isfinite(pixel_ssmi)
            &
            np.isfinite(pixel_ndvi)
        )

        if valid.sum() < 3:
            continue


        # ---------------------------------------------------------------
        # Calculate lagged correlations
        # ---------------------------------------------------------------

        r_values = []

        for lag in lags:

            if lag == 0:

                x_data = pixel_ssmi
                y_data = pixel_ndvi

            else:

                x_data = pixel_ssmi[:-lag]
                y_data = pixel_ndvi[lag:]


            # Pairwise valid values for this lag
            valid_lag = (
                np.isfinite(x_data)
                &
                np.isfinite(y_data)
            )

            x_valid = x_data[valid_lag]
            y_valid = y_data[valid_lag]


            # Pearson correlation requires at least 3 observations
            if len(x_valid) < 3:

                r_values.append(np.nan)
                continue


            # Avoid zero-variance cases
            if (
                np.std(x_valid) == 0
                or np.std(y_valid) == 0
            ):

                r_values.append(np.nan)
                continue


            r, _ = pearsonr(
                x_valid,
                y_valid
            )

            r_values.append(r)


        r_values = np.asarray(
            r_values,
            dtype=np.float32
        )


        # ---------------------------------------------------------------
        # Select dominant lag
        # ---------------------------------------------------------------

        if np.all(np.isnan(r_values)):
            continue


        peak_index = np.nanargmax(
            r_values
        )

        dominant_lag_map[y, x] = (
            lags[peak_index]
        )

        max_corr_map[y, x] = (
            r_values[peak_index]
        )


# -------------------------------------------------------------------------
# 7. Convert outputs to xarray DataArrays
# -------------------------------------------------------------------------

dominant_lag_xr = xr.DataArray(
    dominant_lag_map,
    coords=[
        ssmi3.y,
        ssmi3.x
    ],
    dims=[
        "y",
        "x"
    ],
    name="dominant_lag"
)

max_corr_xr = xr.DataArray(
    max_corr_map,
    coords=[
        ssmi3.y,
        ssmi3.x
    ],
    dims=[
        "y",
        "x"
    ],
    name="maximum_correlation"
)


# -------------------------------------------------------------------------
# 8. Preserve spatial reference
# -------------------------------------------------------------------------

dominant_lag_xr = (
    dominant_lag_xr
    .rio.write_crs(
        ssmi3.rio.crs
    )
)

max_corr_xr = (
    max_corr_xr
    .rio.write_crs(
        ssmi3.rio.crs
    )
)


# -------------------------------------------------------------------------
# 9. Export Figure 4 and Figure 5 raster products
# -------------------------------------------------------------------------

dominant_lag_xr.rio.to_raster(
    "Fig4_Dominant_Propagation_Lag.tif",
    dtype="float32"
)

max_corr_xr.rio.to_raster(
    "Fig5_Maximum_Propagation_Correlation.tif",
    dtype="float32"
)


print(
    "Pixel-wise drought propagation maps generated successfully."
)
