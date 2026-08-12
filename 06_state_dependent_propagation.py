# =========================================================================
# Script 6: State-Dependent Drought Propagation Analysis
#
# Purpose:
#   Evaluate SSMI3-NDVI propagation separately under:
#       1. Dry conditions
#       2. Normal conditions
#       3. Wet conditions
#
# Ecosystems:
#   Forest
#   Grassland
#   Cropland
#
# States:
#   Dry    : SSMI3 < -0.5
#   Normal : -0.5 <= SSMI3 <= 0.5
#   Wet    : SSMI3 > 0.5
#
# Propagation lags:
#   0-12 months
#
# Outputs:
#   - State_Dependent_Lagged_Correlations.csv
#   - Figure 8: State-dependent propagation heatmap
# =========================================================================


import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


# -------------------------------------------------------------------------
# 1. Load SSMI3, NDVI anomaly, and ecosystem mask
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

if ndvi_stack.sizes["band"] != len(timeline):
    raise ValueError(
        "NDVI anomaly data do not contain 312 monthly observations."
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
# 3. Ecosystem coding
# -------------------------------------------------------------------------
#
# Verify that these IDs correspond to Iran_Ecosystem_Mask.tif.
#
# Current assumed coding:
#   1 = Forest
#   3 = Grassland
#   4 = Cropland
# -------------------------------------------------------------------------

ECOSYSTEMS = {
    "Forest": 1,
    "Grassland": 3,
    "Cropland": 4
}


# -------------------------------------------------------------------------
# 4. Define moisture states
# -------------------------------------------------------------------------

def classify_state(ssmi):

    if ssmi > 0.5:
        return "Wet"

    elif ssmi >= -0.5:
        return "Normal"

    else:
        return "Dry"


# -------------------------------------------------------------------------
# 5. Extract ecosystem-level time series
# -------------------------------------------------------------------------

def get_ecosystem_series(
    data,
    mask,
    ecosystem_id
):

    masked = data.where(
        mask == ecosystem_id
    )

    return (
        masked
        .mean(
            dim=["y", "x"],
            skipna=True
        )
        .to_pandas()
    )


# -------------------------------------------------------------------------
# 6. Calculate state-dependent lagged correlations
# -------------------------------------------------------------------------

def state_lagged_correlation(
    forcing,
    response,
    max_lag=12
):

    results = []

    # Align time series first
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

    for state in [
        "Dry",
        "Normal",
        "Wet"
    ]:

        # -------------------------------------------------------------
        # State classification is based on the forcing SSMI3 at time t.
        # The corresponding vegetation response occurs at t + lag.
        # -------------------------------------------------------------

        if state == "Dry":

            state_mask = forcing < -0.5

        elif state == "Normal":

            state_mask = (
                (forcing >= -0.5)
                &
                (forcing <= 0.5)
            )

        else:

            state_mask = forcing > 0.5


        for lag in range(max_lag + 1):

            if lag == 0:

                x = forcing
                y = response
                state_lag_mask = state_mask

            else:

                x = forcing[:-lag]
                y = response[lag:]

                # State is determined from the soil-moisture
                # condition at the forcing time t.
                state_lag_mask = state_mask[:-lag]


            # Pairwise finite observations
            valid = (
                state_lag_mask
                &
                np.isfinite(x)
                &
                np.isfinite(y)
            )

            x_valid = x[valid]
            y_valid = y[valid]

            n = len(x_valid)


            if n < 3:

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
                    "State": state,
                    "Lag_months": lag,
                    "Pearson_r": r,
                    "p_value": p,
                    "n": n
                }
            )


    return pd.DataFrame(results)


# -------------------------------------------------------------------------
# 7. Run analysis for all ecosystems
# -------------------------------------------------------------------------

all_results = []

print(
    "Running state-dependent propagation analysis..."
)


for ecosystem, ecosystem_id in ECOSYSTEMS.items():

    print(
        f"Processing {ecosystem}..."
    )

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

    ecosystem_results = (
        state_lagged_correlation(
            ecosystem_ssmi,
            ecosystem_ndvi,
            max_lag=12
        )
    )

    ecosystem_results.insert(
        0,
        "Ecosystem",
        ecosystem
    )

    all_results.append(
        ecosystem_results
    )


results_df = pd.concat(
    all_results,
    ignore_index=True
)


# -------------------------------------------------------------------------
# 8. Save complete numerical results
# -------------------------------------------------------------------------

results_df.to_csv(
    "State_Dependent_Lagged_Correlations.csv",
    index=False
)


# -------------------------------------------------------------------------
# 9. Identify peak correlation for each ecosystem/state
# -------------------------------------------------------------------------

peak_rows = []

for ecosystem in ECOSYSTEMS.keys():

    for state in [
        "Dry",
        "Normal",
        "Wet"
    ]:

        subset = results_df[
            (results_df["Ecosystem"] == ecosystem)
            &
            (results_df["State"] == state)
        ].copy()

        subset = subset.dropna(
            subset=["Pearson_r"]
        )

        if len(subset) == 0:
            continue

        peak_idx = (
            subset["Pearson_r"]
            .idxmax()
        )

        peak = subset.loc[peak_idx]

        peak_rows.append(
            {
                "Ecosystem": ecosystem,
                "State": state,
                "Peak_r": peak["Pearson_r"],
                "Peak_lag_months": peak["Lag_months"],
                "p_value": peak["p_value"],
                "n": peak["n"]
            }
        )


peak_df = pd.DataFrame(
    peak_rows
)


peak_df.to_csv(
    "State_Dependent_Peak_Results.csv",
    index=False
)


# -------------------------------------------------------------------------
# 10. Prepare Figure 8 data
# -------------------------------------------------------------------------

state_order = [
    "Dry",
    "Normal",
    "Wet"
]

ecosystem_order = [
    "Forest",
    "Grassland",
    "Cropland"
]


heatmap_rows = []
labels = []

for ecosystem in ecosystem_order:

    for state in state_order:

        subset = results_df[
            (results_df["Ecosystem"] == ecosystem)
            &
            (results_df["State"] == state)
        ].sort_values(
            "Lag_months"
        )

        heatmap_rows.append(
            subset["Pearson_r"].values
        )

        labels.append(
            f"{ecosystem}-{state}"
        )


heatmap_data = np.array(
    heatmap_rows
)


# -------------------------------------------------------------------------
# 11. Generate Figure 8
# -------------------------------------------------------------------------

lags = np.arange(13)

plt.figure(
    figsize=(13, 7)
)

im = plt.imshow(
    heatmap_data,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-0.35,
    vmax=0.35
)


plt.yticks(
    np.arange(len(labels)),
    labels,
    fontsize=11
)

plt.xticks(
    np.arange(13),
    lags,
    fontsize=11
)


plt.xlabel(
    "Lag (months)",
    fontsize=12
)

plt.ylabel(
    "Ecosystem and climate state",
    fontsize=12
)


cbar = plt.colorbar(
    im
)

cbar.set_label(
    "Correlation coefficient",
    fontsize=11
)


# Separate ecosystems visually
plt.axhline(
    2.5,
    linewidth=1
)

plt.axhline(
    5.5,
    linewidth=1
)


plt.title(
    "State-Dependent Drought Propagation Across Iranian Ecosystems",
    fontsize=14,
    fontweight="bold"
)


plt.tight_layout()


plt.savefig(
    "Fig8_State_Dependent_Propagation.tif",
    dpi=600,
    bbox_inches="tight"
)


plt.show()


print(
    "State-dependent propagation analysis completed successfully."
)
