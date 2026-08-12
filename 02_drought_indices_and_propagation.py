# =========================================================================
# Script 2: Standardized Soil Moisture Indices and Drought Event Analysis
#
# Purpose:
#   1. Load monthly ecosystem-specific RZSM
#   2. Construct SSMI1 using calendar-month standardization
#   3. Construct SSMI3 from a trailing 3-month RZSM average
#   4. Identify drought events using the Theory of Runs
#   5. Calculate drought duration, peak intensity, and cumulative severity
#   6. Compare SSMI1 and SSMI3
#
# Study period:
#   2000-2025
#
# Primary drought indicator:
#   SSMI3
#
# Drought threshold:
#   SSMI3 <= -1.0
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
#
# 312 monthly observations:
# January 2000 - December 2025
# -------------------------------------------------------------------------

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
# For each pixel, soil moisture is standardized separately for each
# calendar month to remove the seasonal cycle.
# -------------------------------------------------------------------------

print("Calculating SSMI1...")


monthly_mean_1m = (
    rzsm
    .groupby("time.month")
    .mean(
        "time",
        skipna=True
    )
)


monthly_std_1m = (
    rzsm
    .groupby("time.month")
    .std(
        "time",
        skipna=True
    )
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
# A trailing 3-month moving average is used to represent accumulated
# soil-moisture conditions and short-term hydrological memory.
#
# The first two months do not have a complete 3-month window.
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
# The 3-month RZSM series is standardized separately for each calendar
# month.
# -------------------------------------------------------------------------

print("Calculating SSMI3...")


monthly_mean_3m = (
    rzsm_3m
    .groupby("time.month")
    .mean(
        "time",
        skipna=True
    )
)


monthly_std_3m = (
    rzsm_3m
    .groupby("time.month")
    .std(
        "time",
        skipna=True
    )
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
# RZSM contains only the selected vegetated ecosystem classes.
# NaN values outside these areas are excluded from the spatial mean.
# -------------------------------------------------------------------------

ssmi1_mean = (
    ssmi1
    .mean(
        dim=["y", "x"],
        skipna=True
    )
    .to_pandas()
)


ssmi3_mean = (
    ssmi3
    .mean(
        dim=["y", "x"],
        skipna=True
    )
    .to_pandas()
)


# -------------------------------------------------------------------------
# 8. Theory of Runs: Drought Event Identification
# -------------------------------------------------------------------------
#
# A drought event is defined as a consecutive sequence of months during
# which spatially averaged SSMI3 remains at or below -1.0.
#
# For each event:
#   - Start date
#   - End date
#   - Duration
#   - Peak SSMI3
#   - Cumulative severity
#   - Drought category
# are calculated.
# -------------------------------------------------------------------------

print("Identifying drought events using Theory of Runs...")


drought_threshold = -1.0


is_drought = (
    ssmi3_mean <= drought_threshold
)


drought_events = []


in_event = False
start_date = None


for date, value in is_drought.items():

    # -------------------------------------------------------------
    # Start of a drought event
    # -------------------------------------------------------------

    if value and not in_event:

        start_date = date
        in_event = True


    # -------------------------------------------------------------
    # End of a drought event
    # -------------------------------------------------------------

    elif not value and in_event:

        end_date = (
            date
            - pd.DateOffset(months=1)
        )


        event_values = ssmi3_mean.loc[
            start_date:end_date
        ].dropna()


        duration = len(
            event_values
        )


        peak_ssmi3 = (
            event_values.min()
        )


        cumulative_severity = (
            event_values.sum()
        )


        # ---------------------------------------------------------
        # Drought classification
        # ---------------------------------------------------------

        if peak_ssmi3 <= -2.0:

            category = "Extreme"

        elif peak_ssmi3 <= -1.5:

            category = "Severe"

        else:

            category = "Moderate"


        drought_events.append(
            [
                start_date,
                end_date,
                duration,
                peak_ssmi3,
                cumulative_severity,
                category
            ]
        )


        in_event = False
        start_date = None


# -------------------------------------------------------------------------
# 9. Close Event if It Extends to December 2025
# -------------------------------------------------------------------------

if in_event:

    end_date = ssmi3_mean.index[-1]


    event_values = ssmi3_mean.loc[
        start_date:end_date
    ].dropna()


    duration = len(
        event_values
    )


    peak_ssmi3 = (
        event_values.min()
    )


    cumulative_severity = (
        event_values.sum()
    )


    if peak_ssmi3 <= -2.0:

        category = "Extreme"

    elif peak_ssmi3 <= -1.5:

        category = "Severe"

    else:

        category = "Moderate"


    drought_events.append(
        [
            start_date,
            end_date,
            duration,
            peak_ssmi3,
            cumulative_severity,
            category
        ]
    )


# -------------------------------------------------------------------------
# 10. Create Drought Event Table
# -------------------------------------------------------------------------

events_df = pd.DataFrame(
    drought_events,
    columns=[
        "Start",
        "End",
        "Duration_months",
        "Peak_SSMI3",
        "Cumulative_Severity",
        "Category"
    ]
)


print("\nIdentified drought events:")
print(
    events_df.to_string(
        index=False
    )
)


events_df.to_csv(
    "Drought_Events_Table.csv",
    index=False
)


# -------------------------------------------------------------------------
# 11. SSMI1 vs SSMI3 Comparison
# -------------------------------------------------------------------------

print(
    "Generating SSMI1 vs SSMI3 comparison figure..."
)


plt.figure(
    figsize=(14, 5)
)


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


plt.axhline(
    -1.0,
    linestyle=":",
    linewidth=1.0,
    label="Drought threshold (-1.0)"
)


plt.xlabel(
    "Year"
)


plt.ylabel(
    "Standardized Soil Moisture Index"
)


plt.title(
    "Comparison of SSMI1 and SSMI3 across Iran (2000–2025)"
)


plt.grid(
    alpha=0.3
)


plt.legend()


plt.tight_layout()


plt.savefig(
    "Fig2_SSMI1_vs_SSMI3.tif",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# -------------------------------------------------------------------------
# 12. Final Summary
# -------------------------------------------------------------------------

print("\n" + "=" * 70)
print(
    "SCRIPT 2 COMPLETED SUCCESSFULLY"
)
print("=" * 70)

print(
    f"Total drought events identified: "
    f"{len(events_df)}"
)

print(
    "Primary drought indicator: SSMI3"
)

print(
    "Drought threshold: SSMI3 <= -1.0"
)

print("=" * 70)
