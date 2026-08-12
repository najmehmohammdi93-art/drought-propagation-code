# =========================================================================
# Script 2: Standardized Soil Moisture Index (SSMI1 and SSMI3)
#
# Language: Python 3.x
# Dependencies: xarray, rioxarray, pandas, numpy, matplotlib
#
# Purpose:
#   1. Load monthly ecosystem-specific RZSM
#   2. Construct SSMI1 using calendar-month standardization
#   3. Construct SSMI3 from 3-month accumulated RZSM
#   4. Compare SSMI1 and SSMI3
#
# Study period: 2000-2025
# Primary drought indicator: SSMI3
# =========================================================================


import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# 1. Load RZSM Data
# -------------------------------------------------------------------------

print("Loading monthly RZSM raster stack...")

rzsm_stack = rxr.open_rasterio(
    "Iran_RZSM_312months.tif"
)


# -------------------------------------------------------------------------
# 2. Assign Continuous Monthly Time Dimension
# -------------------------------------------------------------------------

# 312 monthly observations:
# January 2000 - December 2025

time_coords = pd.date_range(
    start="2000-01-01",
    end="2025-12-01",
    freq="MS"
)

if rzsm_stack.sizes["band"] != len(time_coords):
    raise ValueError(
        f"Expected {len(time_coords)} monthly bands, "
        f"but found {rzsm_stack.sizes['band']}."
    )

rzsm = (
    rzsm_stack
    .assign_coords(band=time_coords)
    .rename({"band": "time"})
)


# -------------------------------------------------------------------------
# 3. SSMI1: One-Month Standardized Soil Moisture Index
# -------------------------------------------------------------------------
#
# For each pixel and each calendar month:
#
#   SSMI1 = (RZSM - monthly_mean) / monthly_std
#
# Monthly statistics are calculated separately for each calendar month
# to remove the recurring seasonal cycle.
# -------------------------------------------------------------------------

print("Calculating SSMI1...")

monthly_mean_1m = (
    rzsm
    .groupby("time.month")
    .mean("time", skipna=True)
)

monthly_std_1m = (
    rzsm
    .groupby("time.month")
    .std("time", skipna=True)
)

ssmi1 = (
    (rzsm.groupby("time.month") - monthly_mean_1m)
    / monthly_std_1m
)

ssmi1 = ssmi1.drop_vars(
    "month",
    errors="ignore"
)

ssmi1.name = "SSMI1"


# -------------------------------------------------------------------------
# 4. Three-Month RZSM Aggregation
# -------------------------------------------------------------------------
#
# A trailing 3-month moving average is used to represent cumulative
# soil-moisture conditions and hydrological memory.
#
# The first two months are excluded because a complete 3-month window
# is required.
# -------------------------------------------------------------------------

print("Calculating 3-month RZSM aggregation...")

rzsm_3m = (
    rzsm
    .rolling(
        time=3,
        center=False,
        min_periods=3
    )
    .mean()
)


# -------------------------------------------------------------------------
# 5. SSMI3: Three-Month Standardized Soil Moisture Index
# -------------------------------------------------------------------------
#
# The 3-month aggregated RZSM is standardized separately for each
# calendar month.
# -------------------------------------------------------------------------

print("Calculating SSMI3...")

monthly_mean_3m = (
    rzsm_3m
    .groupby("time.month")
    .mean("time", skipna=True)
)

monthly_std_3m = (
    rzsm_3m
    .groupby("time.month")
    .std("time", skipna=True)
)

ssmi3 = (
    (rzsm_3m.groupby("time.month") - monthly_mean_3m)
    / monthly_std_3m
)

ssmi3 = ssmi3.drop_vars(
    "month",
    errors="ignore"
)

ssmi3.name = "SSMI3"


# -------------------------------------------------------------------------
# 6. Save SSMI Outputs
# -------------------------------------------------------------------------

print("Saving SSMI datasets...")

ssmi1.to_netcdf(
    "Iran_SSMI1_2000_2025.nc"
)

ssmi3.to_netcdf(
    "Iran_SSMI3_2000_2025.nc"
)


# -------------------------------------------------------------------------
# 7. Spatially Averaged SSMI1 and SSMI3
# -------------------------------------------------------------------------
#
# RZSM is already masked to the three ecosystems in Script 1.
# Therefore, NaN values outside the analyzed vegetation classes are
# excluded from the spatial mean.
# -------------------------------------------------------------------------

ssmi1_mean = ssmi1.mean(
    dim=["y", "x"],
    skipna=True
).to_pandas()

ssmi3_mean = ssmi3.mean(
    dim=["y", "x"],
    skipna=True
).to_pandas()


# -------------------------------------------------------------------------
# 8. Compare SSMI1 and SSMI3
# -------------------------------------------------------------------------

print("Generating SSMI1 vs SSMI3 comparison figure...")

plt.figure(figsize=(14, 5))

plt.plot(
    ssmi1_mean.index,
    ssmi1_mean.values,
    linewidth=1.0,
    label="SSMI1"
)

plt.plot(
    ssmi3_mean.index,
    ssmi3_mean.values,
    linewidth=1.5,
    label="SSMI3"
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1.0
)

plt.xlabel("Year")
plt.ylabel("Standardized Soil Moisture Index")
plt.title(
    "Comparison of SSMI1 and SSMI3 across Iran (2000–2025)"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "Fig1_SSMI1_vs_SSMI3.tif",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Script 2 completed successfully.")
