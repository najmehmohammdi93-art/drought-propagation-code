# =========================================================================
# Script 4: Ecosystem-Scale Lagged Cross-Correlation
#
# Purpose:
#   Calculate ecosystem-level drought propagation between SSMI3
#   and NDVI anomalies for lags 0-12 months.
#
# Output:
#   1. Pearson correlation coefficient for each lag
#   2. Dominant propagation lag
#   3. Maximum correlation coefficient
#   4. Figure 3: Ecosystem-specific lagged correlation curves
#
# Ecosystems:
#   Forest
#   Grassland
#   Cropland
# =========================================================================


import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# -------------------------------------------------------------------------
# 1. Load input datasets
# -------------------------------------------------------------------------

print("Loading SSMI3, NDVI anomalies, and ecosystem mask...")

ssmi3_stack = rxr.open_rasterio(
    "Iran_SSMI3_312months.tif"
)

ndvi_anom_stack = rxr.open_rasterio(
    "Iran_NDVI_Anom_312months.tif"
)

ecosystem_mask = rxr.open_rasterio(
    "Iran_Ecosystem_Mask.tif"
).isel(band=0)


# -------------------------------------------------------------------------
# 2. Assign monthly time dimension
# -------------------------------------------------------------------------

timeline = pd.date_range(
    start="2000-01-01",
    end="2025-12-01",
    freq="MS"
)


if ssmi3_stack.sizes["band"] != len(timeline):
    raise ValueError(
        "SSMI3 does not contain 312 monthly bands."
    )

if ndvi_anom_stack.sizes["band"] != len(timeline):
    raise ValueError(
        "NDVI anomaly dataset does not contain 312 monthly bands."
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
# 3. Extract ecosystem-level spatial means
# -------------------------------------------------------------------------
#
# IMPORTANT:
# The ecosystem IDs must correspond to the coding used in
# Iran_Ecosystem_Mask.tif.
#
# Current assumed coding:
#   1 = Forest
#   3 = Grassland
#   4 = Cropland
#
# Verify this raster before final publication.
# -------------------------------------------------------------------------

ECOSYSTEM_IDS = {
    "Forest": 1,
    "Grassland": 3,
    "Cropland": 4
}


def get_ecosystem_series(data, mask, ecosystem_id):

    masked_data = data.where(
        mask == ecosystem_id
    )

    series = (
        masked_data
        .mean(
            dim=["y", "x"],
            skipna=True
        )
        .to_pandas()
    )

    return series


# -------------------------------------------------------------------------
# 4. Calculate lagged Pearson correlations
# -------------------------------------------------------------------------
#
# Lag 0:
#   Concurrent monthly variability.
#
# Positive lag:
#   Vegetation response follows the preceding soil-moisture condition.
# -------------------------------------------------------------------------


def calculate_lagged_correlations(
    forcing,
    response,
    max_lag=12
):

    # Align the two time series first
    combined = pd.concat(
        [
            forcing.rename("SSMI3"),
            response.rename("NDVI")
        ],
        axis=1
    ).dropna()

    forcing = combined["SSMI3"].values
    response = combined["NDVI"].values

    correlations = []

    for lag in range(max_lag + 1):

        if lag == 0:

            x = forcing
            y = response

        else:

            x = forcing[:-lag]
            y = response[lag:]

        # Check that enough observations remain
        if len(x) < 3:

            correlations.append(np.nan)

        else:

            r, _ = pearsonr(x, y)
            correlations.append(r)

    return np.array(correlations)


# -------------------------------------------------------------------------
# 5. Calculate ecosystem-specific propagation curves
# -------------------------------------------------------------------------

print(
    "Calculating lagged correlations for lags 0-12 months..."
)


results = {}

for ecosystem, ecosystem_id in ECOSYSTEM_IDS.items():

    ssmi_series = get_ecosystem_series(
        ssmi3,
        ecosystem_mask,
        ecosystem_id
    )

    ndvi_series = get_ecosystem_series(
        ndvi_anom,
        ecosystem_mask,
        ecosystem_id
    )

    r_vector = calculate_lagged_correlations(
        ssmi_series,
        ndvi_series,
        max_lag=12
    )

    lags = np.arange(0, 13)

    # Dominant lag = lag with maximum correlation
    peak_index = np.nanargmax(r_vector)

    peak_lag = lags[peak_index]
    peak_r = r_vector[peak_index]

    results[ecosystem] = {
        "lags": lags,
        "r": r_vector,
        "peak_lag": peak_lag,
        "peak_r": peak_r
    }

    print(
        f"{ecosystem}: "
        f"Peak r = {peak_r:.3f}, "
        f"Dominant lag = {peak_lag} month(s)"
    )


# -------------------------------------------------------------------------
# 6. Save numerical results
# -------------------------------------------------------------------------

result_rows = []

for ecosystem, result in results.items():

    for lag, r in zip(
        result["lags"],
        result["r"]
    ):

        result_rows.append(
            [
                ecosystem,
                lag,
                r
            ]
        )


results_df = pd.DataFrame(
    result_rows,
    columns=[
        "Ecosystem",
        "Lag_months",
        "Pearson_r"
    ]
)


results_df.to_csv(
    "Ecosystem_Lagged_Correlation_Results.csv",
    index=False
)


# -------------------------------------------------------------------------
# 7. Save dominant propagation results
# -------------------------------------------------------------------------

peak_df = pd.DataFrame(
    [
        {
            "Ecosystem": ecosystem,
            "Peak_r": result["peak_r"],
            "Dominant_lag_months": result["peak_lag"]
        }
        for ecosystem, result in results.items()
    ]
)


peak_df.to_csv(
    "Ecosystem_Dominant_Propagation_Lag.csv",
    index=False
)


# -------------------------------------------------------------------------
# 8. Generate Figure 3
# -------------------------------------------------------------------------

plt.figure(
    figsize=(8, 5.5)
)


for ecosystem, result in results.items():

    plt.plot(
        result["lags"],
        result["r"],
        marker="o",
        linewidth=2,
        label=(
            f"{ecosystem} "
            f"(Peak lag = {result['peak_lag']} "
            f"months, r = {result['peak_r']:.3f})"
        )
    )


plt.axhline(
    0,
    linestyle=":",
    linewidth=1
)


plt.xlabel(
    "Propagation lag (months)",
    fontsize=11
)

plt.ylabel(
    "Correlation coefficient (r)",
    fontsize=11
)


plt.xticks(
    np.arange(0, 13, 1)
)


plt.grid(
    alpha=0.2
)


plt.legend(
    fontsize=9
)


plt.tight_layout()


plt.savefig(
    "Fig3_Ecosystem_Lagged_Correlation.tif",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


print(
    "\nEcosystem-scale lagged correlation analysis completed."
)
