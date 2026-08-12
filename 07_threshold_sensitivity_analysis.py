# =========================================================================
# File 07: Threshold Sensitivity Analysis and Stability Testing Framework
#
# Purpose:
#   Evaluate the sensitivity of drought propagation to alternative
#   drought-severity thresholds.
#
# Thresholds:
#   SSMI3 < -0.5   Moderate drought / primary threshold
#   SSMI3 < -0.75  Moderate-to-severe drought
#   SSMI3 < -1.0   Conventional severe drought
#
# Ecosystems:
#   Forest
#   Grassland
#   Cropland
#
# Lags:
#   0-12 months
#
# Outputs:
#   - Threshold_Sensitivity_Results.csv
#   - Threshold_Sensitivity_Peaks.csv
#   - Fig9_Threshold_Sensitivity.tif
# =========================================================================


import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# -------------------------------------------------------------------------
# 1. Load SSMI3, NDVI anomaly and ecosystem mask
# -------------------------------------------------------------------------

print("Loading datasets...")

ssmi3_stack = rxr.open_rasterio(
    "Iran_SSMI3_312months.tif"
)

ndvi_stack = rxr.open_rasterio(
    "Iran_NDVI_Anom_312months.tif"
)

ecosystem_mask = rxr.open_rasterio(
    "Iran_Ecosystem_Mask.tif"
).isel(band=0)


# -------------------------------------------------------------------------
# 2. Define monthly timeline
# -------------------------------------------------------------------------

timeline = pd.date_range(
    start="2000-01-01",
    end="2025-12-01",
    freq="MS"
)


if ssmi3_stack.sizes["band"] != len(timeline):
    raise ValueError(
        "SSMI3 dataset does not contain 312 monthly observations."
    )

if ndvi_stack.sizes["band"] != len(timeline):
    raise ValueError(
        "NDVI dataset does not contain 312 monthly observations."
    )


ssmi3 = (
    ssmi3_stack
    .assign_coords(band=timeline)
    .rename({"band": "time"})
)

ndvi = (
    ndvi_stack
    .assign_coords(band=timeline)
    .rename({"band": "time"})
)


# -------------------------------------------------------------------------
# 3. Ecosystem definitions
# -------------------------------------------------------------------------
#
# Assumed ecosystem IDs:
#   1 = Forest
#   3 = Grassland
#   4 = Cropland
#
# Verify these IDs against Iran_Ecosystem_Mask.tif.
# -------------------------------------------------------------------------

ECOSYSTEMS = {
    "Forest": 1,
    "Grassland": 3,
    "Cropland": 4
}


# -------------------------------------------------------------------------
# 4. Alternative drought thresholds
# -------------------------------------------------------------------------

THRESHOLDS = {
    "SSMI3 < -0.5": -0.5,
    "SSMI3 < -0.75": -0.75,
    "SSMI3 < -1.0": -1.0
}


# -------------------------------------------------------------------------
# 5. Extract ecosystem-level time series
# -------------------------------------------------------------------------

def get_ecosystem_series(
    data,
    mask,
    ecosystem_id
):

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
# 6. Calculate threshold-dependent lagged correlations
# -------------------------------------------------------------------------

def calculate_threshold_correlations(
    forcing,
    response,
    threshold,
    max_lag=12,
    min_samples=5
):

    results = []

    # Align the two time series
    combined = pd.concat(
        [
            forcing.rename("SSMI3"),
            response.rename("NDVI")
        ],
        axis=1
    )

    combined = combined.dropna()

    forcing = combined["SSMI3"].values
    response = combined["NDVI"].values


    # -------------------------------------------------------------
    # Identify drought observations according to threshold
    # -------------------------------------------------------------

    drought_mask = forcing < threshold


    # -------------------------------------------------------------
    # Calculate lagged correlations
    # -------------------------------------------------------------

    for lag in range(max_lag + 1):

        if lag == 0:

            x = forcing
            y = response
            state_mask = drought_mask

        else:

            # Soil moisture at time t
            x = forcing[:-lag]

            # Vegetation response at t + lag
            y = response[lag:]

            # Drought state is defined at forcing time t
            state_mask = drought_mask[:-lag]


        # Valid paired observations
        valid = (
            state_mask
            &
            np.isfinite(x)
            &
            np.isfinite(y)
        )


        x_valid = x[valid]
        y_valid = y[valid]

        n = len(x_valid)


        # ---------------------------------------------------------
        # Statistical safeguards
        # ---------------------------------------------------------

        if n < min_samples:

            r = np.nan
            p = np.nan

        elif (
            np.std(x_valid) == 0
            or np.std(y_valid) == 0
        ):

            r = np.nan
            p = np.nan

        else:

            r, p = pearsonr(
                x_valid,
                y_valid
            )


        results.append(
            {
                "Lag_months": lag,
                "Pearson_r": r,
                "p_value": p,
                "n": n
            }
        )


    return pd.DataFrame(results)


# -------------------------------------------------------------------------
# 7. Run threshold analysis for all ecosystems
# -------------------------------------------------------------------------

all_results = []

print(
    "Running threshold sensitivity analysis..."
)


for ecosystem, ecosystem_id in ECOSYSTEMS.items():

    print(
        f"\nProcessing {ecosystem}..."
    )


    # Extract ecosystem time series
    ecosystem_ssmi = get_ecosystem_series(
        ssmi3,
        ecosystem_mask,
        ecosystem_id
    )

    ecosystem_ndvi = get_ecosystem_series(
        ndvi,
        ecosystem_mask,
        ecosystem_id
    )


    # -------------------------------------------------------------
    # Run all thresholds
    # -------------------------------------------------------------

    for threshold_name, threshold_value in THRESHOLDS.items():

        print(
            f"  {threshold_name}"
        )


        threshold_results = (
            calculate_threshold_correlations(
                ecosystem_ssmi,
                ecosystem_ndvi,
                threshold=threshold_value,
                max_lag=12
            )
        )


        threshold_results.insert(
            0,
            "Threshold",
            threshold_name
        )

        threshold_results.insert(
            0,
            "Ecosystem",
            ecosystem
        )


        all_results.append(
            threshold_results
        )


# -------------------------------------------------------------------------
# 8. Combine all results
# -------------------------------------------------------------------------

results_df = pd.concat(
    all_results,
    ignore_index=True
)


# -------------------------------------------------------------------------
# 9. Save complete numerical results
# -------------------------------------------------------------------------

results_df.to_csv(
    "Threshold_Sensitivity_Results.csv",
    index=False
)


# -------------------------------------------------------------------------
# 10. Extract peak correlation and corresponding lag
# -------------------------------------------------------------------------

peak_results = []


for ecosystem in ECOSYSTEMS.keys():

    for threshold_name in THRESHOLDS.keys():

        subset = results_df[
            (results_df["Ecosystem"] == ecosystem)
            &
            (results_df["Threshold"] == threshold_name)
        ].copy()


        subset = subset.dropna(
            subset=["Pearson_r"]
        )


        if len(subset) == 0:

            continue


        # Maximum positive correlation
        peak_idx = (
            subset["Pearson_r"]
            .idxmax()
        )

        peak = subset.loc[peak_idx]


        peak_results.append(
            {
                "Ecosystem": ecosystem,
                "Threshold": threshold_name,
                "Peak_r": peak["Pearson_r"],
                "Peak_lag_months": peak["Lag_months"],
                "p_value": peak["p_value"],
                "n": peak["n"]
            }
        )


peak_df = pd.DataFrame(
    peak_results
)


peak_df.to_csv(
    "Threshold_Sensitivity_Peaks.csv",
    index=False
)


# -------------------------------------------------------------------------
# 11. Generate Figure 9
# -------------------------------------------------------------------------
#
# One panel for each ecosystem.
# Each panel contains the three drought thresholds.
# -------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
    sharey=True
)


ecosystem_order = [
    "Forest",
    "Grassland",
    "Cropland"
]


threshold_styles = {
    "SSMI3 < -0.5": {
        "color": "orange",
        "marker": "o",
        "linestyle": "-"
    },

    "SSMI3 < -0.75": {
        "color": "red",
        "marker": "s",
        "linestyle": "--"
    },

    "SSMI3 < -1.0": {
        "color": "darkred",
        "marker": "^",
        "linestyle": "-."
    }
}


for ax, ecosystem in zip(
    axes,
    ecosystem_order
):

    for threshold_name in THRESHOLDS.keys():

        subset = results_df[
            (results_df["Ecosystem"] == ecosystem)
            &
            (results_df["Threshold"] == threshold_name)
        ].sort_values(
            "Lag_months"
        )


        style = threshold_styles[
            threshold_name
        ]


        ax.plot(
            subset["Lag_months"],
            subset["Pearson_r"],
            marker=style["marker"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            label=threshold_name
        )


    ax.axhline(
        0,
        color="gray",
        linestyle=":",
        linewidth=1
    )


    ax.set_title(
        ecosystem,
        fontsize=13,
        fontweight="bold"
    )


    ax.set_xlabel(
        "Lag (months)",
        fontsize=11
    )


    ax.set_xticks(
        np.arange(13)
    )


    ax.grid(
        alpha=0.3
    )


axes[0].set_ylabel(
    "Correlation coefficient (r)",
    fontsize=11
)


axes[-1].legend(
    fontsize=9,
    loc="best"
)


fig.suptitle(
    "Sensitivity of Drought Propagation to Drought Severity Threshold",
    fontsize=14,
    fontweight="bold"
)


plt.tight_layout()


plt.savefig(
    "Fig9_Threshold_Sensitivity.tif",
    dpi=600,
    bbox_inches="tight"
)


plt.show()


print(
    "\nThreshold sensitivity analysis completed successfully."
)
