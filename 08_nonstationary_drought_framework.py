# =========================================================================
# File 08: Moving-Baseline Drought Characterization
#
# Primary moving-baseline configuration:
#   10-year trailing monthly climatology
#
# Purpose:
#   1. Calculate stationary SSMI3 using the fixed reference period
#   2. Calculate moving-baseline SSMI3 using a 10-year trailing window
#   3. Compare drought propagation under the two standardization methods
#   4. Calculate ecosystem-specific lagged correlations
#
# Ecosystems:
#   Forest
#   Grassland
#   Cropland
#
# Analysis period:
#   2000-2025
#
# Evaluation period:
#   2015-2025
#
# Propagation lags:
#   0-12 months
#
# Primary moving-baseline window:
#   10 years
#
# Outputs:
#   - MovingBaseline_10yr_Results.csv
#   - MovingBaseline_10yr_Peak_Results.csv
#   - Fig10_Stationary_vs_MovingBaseline.tif
#   - Fig11_Propagation_Comparison_10yr.tif
# =========================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray as rxr
from scipy.stats import pearsonr


# -------------------------------------------------------------------------
# 1. Load RZSM, NDVI anomaly, and ecosystem mask
# -------------------------------------------------------------------------

print("Loading datasets...")

rzsm_stack = rxr.open_rasterio(
    "Iran_RZSM_312months.tif"
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


if rzsm_stack.sizes["band"] != len(timeline):
    raise ValueError(
        "RZSM dataset does not contain 312 monthly observations."
    )

if ndvi_stack.sizes["band"] != len(timeline):
    raise ValueError(
        "NDVI anomaly dataset does not contain 312 monthly observations."
    )


rzsm_stack = (
    rzsm_stack
    .assign_coords(band=timeline)
    .rename({"band": "time"})
)

ndvi_stack = (
    ndvi_stack
    .assign_coords(band=timeline)
    .rename({"band": "time"})
)


# -------------------------------------------------------------------------
# 3. Ecosystem definitions
# -------------------------------------------------------------------------
#
# These IDs must match the ecosystem mask generated in Script 1.
#
# Current coding:
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
# 4. Extract ecosystem-level RZSM and NDVI time series
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
# 5. Calculate stationary SSMI3
# -------------------------------------------------------------------------
#
# The stationary reference uses the fixed 2000-2014 period.
#
# Standardization is performed separately for each calendar month.
#
# Therefore:
#
#   January 2015 is standardized against January 2000-2014
#   February 2015 is standardized against February 2000-2014
#   ...
# -------------------------------------------------------------------------

def calculate_stationary_ssmi3(
    rzsm_series,
    reference_start="2000-01-01",
    reference_end="2014-12-01"
):

    df = pd.DataFrame({
        "RZSM": rzsm_series
    })

    df["Year"] = df.index.year
    df["Month"] = df.index.month

    reference = df.loc[
        reference_start:reference_end
    ]

    monthly_mean = (
        reference
        .groupby("Month")["RZSM"]
        .mean()
    )

    monthly_std = (
        reference
        .groupby("Month")["RZSM"]
        .std()
    )

    ssmi3 = []

    for date, row in df.iterrows():

        month = date.month

        mu = monthly_mean.get(
            month,
            np.nan
        )

        sigma = monthly_std.get(
            month,
            np.nan
        )

        if (
            pd.isna(row["RZSM"])
            or pd.isna(mu)
            or pd.isna(sigma)
            or sigma == 0
        ):

            ssmi3.append(np.nan)

        else:

            ssmi3.append(
                (row["RZSM"] - mu) / sigma
            )

    return pd.Series(
        ssmi3,
        index=rzsm_series.index,
        name="SSMI3"
    )


# -------------------------------------------------------------------------
# 6. Calculate 10-year trailing moving-baseline SSMI3
# -------------------------------------------------------------------------
#
# For each month:
#
#   current RZSM
#       ↓
#   same calendar month in previous 10 years
#       ↓
#   calculate mean and standard deviation
#       ↓
#   standardize current RZSM
#
# Example:
#
#   January 2015
#       uses January 2005-2014
#
#   January 2025
#       uses January 2015-2024
#
# No future observations are used.
# -------------------------------------------------------------------------

def calculate_moving_baseline_ssmi3(
    rzsm_series,
    window_years=10,
    min_years=8
):

    df = pd.DataFrame({
        "RZSM": rzsm_series
    })

    df["Year"] = df.index.year
    df["Month"] = df.index.month

    moving_ssmi = []

    for date, row in df.iterrows():

        current_year = date.year
        current_month = date.month

        # -------------------------------------------------------------
        # Previous years only
        # -------------------------------------------------------------

        history_start = current_year - window_years
        history_end = current_year - 1

        history = df[
            (df["Month"] == current_month)
            &
            (df["Year"] >= history_start)
            &
            (df["Year"] <= history_end)
        ]["RZSM"].dropna()


        # -------------------------------------------------------------
        # Require sufficient historical observations
        # -------------------------------------------------------------

        if len(history) < min_years:

            moving_ssmi.append(
                np.nan
            )

            continue


        mu = history.mean()
        sigma = history.std()


        if (
            pd.isna(row["RZSM"])
            or pd.isna(sigma)
            or sigma == 0
        ):

            moving_ssmi.append(
                np.nan
            )

        else:

            moving_ssmi.append(
                (row["RZSM"] - mu) / sigma
            )


    return pd.Series(
        moving_ssmi,
        index=rzsm_series.index,
        name=f"MovingSSMI3_{window_years}yr"
    )


# -------------------------------------------------------------------------
# 7. Lagged correlation function
# -------------------------------------------------------------------------
#
# Lag 0:
#   concurrent monthly response
#
# Positive lag:
#   NDVI response follows the preceding soil-moisture condition.
# -------------------------------------------------------------------------

def calculate_lagged_correlations(
    forcing,
    response,
    max_lag=12
):

    combined = pd.concat(
        [
            forcing.rename("Forcing"),
            response.rename("Response")
        ],
        axis=1
    ).dropna()

    forcing = combined["Forcing"]
    response = combined["Response"]

    r_values = []

    for lag in range(
        max_lag + 1
    ):

        if lag == 0:

            x = forcing
            y = response

        else:

            x = forcing.iloc[:-lag]
            y = response.iloc[lag:]


        valid = (
            np.isfinite(x.values)
            &
            np.isfinite(y.values)
        )

        x_valid = x.values[valid]
        y_valid = y.values[valid]


        if len(x_valid) < 3:

            r_values.append(
                np.nan
            )

            continue


        if (
            np.std(x_valid) == 0
            or np.std(y_valid) == 0
        ):

            r_values.append(
                np.nan
            )

            continue


        r, _ = pearsonr(
            x_valid,
            y_valid
        )

        r_values.append(
            r
        )


    return np.asarray(
        r_values,
        dtype=float
    )


# -------------------------------------------------------------------------
# 8. Analysis for all ecosystems
# -------------------------------------------------------------------------

all_results = []
peak_results = []


for ecosystem, ecosystem_id in ECOSYSTEMS.items():

    print(
        f"\nProcessing {ecosystem}..."
    )


    # -------------------------------------------------------------
    # Extract ecosystem time series
    # -------------------------------------------------------------

    rzsm_series = get_ecosystem_series(
        rzsm_stack,
        ecosystem_mask,
        ecosystem_id
    )

    ndvi_series = get_ecosystem_series(
        ndvi_stack,
        ecosystem_mask,
        ecosystem_id
    )


    # -------------------------------------------------------------
    # Calculate stationary SSMI3
    # -------------------------------------------------------------

    stationary_ssmi3 = (
        calculate_stationary_ssmi3(
            rzsm_series
        )
    )


    # -------------------------------------------------------------
    # Calculate 10-year moving-baseline SSMI3
    # -------------------------------------------------------------

    moving_ssmi3 = (
        calculate_moving_baseline_ssmi3(
            rzsm_series,
            window_years=10,
            min_years=8
        )
    )


    # -------------------------------------------------------------
    # Restrict analysis to 2015-2025
    # -------------------------------------------------------------

    evaluation_start = "2015-01-01"
    evaluation_end = "2025-12-01"


    stationary_eval = stationary_ssmi3.loc[
        evaluation_start:evaluation_end
    ]

    moving_eval = moving_ssmi3.loc[
        evaluation_start:evaluation_end
    ]

    ndvi_eval = ndvi_series.loc[
        evaluation_start:evaluation_end
    ]


    # -------------------------------------------------------------
    # Lagged correlations
    # -------------------------------------------------------------

    stationary_r = (
        calculate_lagged_correlations(
            stationary_eval,
            ndvi_eval,
            max_lag=12
        )
    )

    moving_r = (
        calculate_lagged_correlations(
            moving_eval,
            ndvi_eval,
            max_lag=12
        )
    )


    lags = np.arange(
        0,
        13
    )


    # -------------------------------------------------------------
    # Extract peak metrics
    # -------------------------------------------------------------

    stationary_peak_index = (
        np.nanargmax(
            stationary_r
        )
    )

    moving_peak_index = (
        np.nanargmax(
            moving_r
        )
    )


    stationary_peak_r = (
        stationary_r[
            stationary_peak_index
        ]
    )

    moving_peak_r = (
        moving_r[
            moving_peak_index
        ]
    )


    stationary_peak_lag = (
        lags[
            stationary_peak_index
        ]
    )

    moving_peak_lag = (
        lags[
            moving_peak_index
        ]
    )


    delta_r = (
        moving_peak_r
        -
        stationary_peak_r
    )


    # -------------------------------------------------------------
    # Store lag-by-lag results
    # -------------------------------------------------------------

    for lag, r in zip(
        lags,
        stationary_r
    ):

        all_results.append(
            {
                "Ecosystem": ecosystem,
                "Baseline": "Stationary",
                "Window_years": np.nan,
                "Lag_months": lag,
                "Pearson_r": r
            }
        )


    for lag, r in zip(
        lags,
        moving_r
    ):

        all_results.append(
            {
                "Ecosystem": ecosystem,
                "Baseline": "Moving",
                "Window_years": 10,
                "Lag_months": lag,
                "Pearson_r": r
            }
        )


    # -------------------------------------------------------------
    # Store peak results
    # -------------------------------------------------------------

    peak_results.append(
        {
            "Ecosystem": ecosystem,
            "Stationary_peak_r": stationary_peak_r,
            "Stationary_peak_lag": stationary_peak_lag,
            "Moving_10yr_peak_r": moving_peak_r,
            "Moving_10yr_peak_lag": moving_peak_lag,
            "Delta_peak_r": delta_r
        }
    )


    print(
        f"{ecosystem}:"
    )

    print(
        f"  Stationary: "
        f"r = {stationary_peak_r:.3f}, "
        f"lag = {stationary_peak_lag}"
    )

    print(
        f"  Moving 10-year: "
        f"r = {moving_peak_r:.3f}, "
        f"lag = {moving_peak_lag}"
    )

    print(
        f"  Delta r = {delta_r:+.3f}"
    )


# -------------------------------------------------------------------------
# 9. Save numerical results
# -------------------------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    "MovingBaseline_10yr_Results.csv",
    index=False
)


peak_df = pd.DataFrame(
    peak_results
)

peak_df.to_csv(
    "MovingBaseline_10yr_Peak_Results.csv",
    index=False
)


# -------------------------------------------------------------------------
# 10. Figure 10
#
# Stationary vs moving-baseline drought index.
#
# The forest, grassland and cropland series are shown separately.
# -------------------------------------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 9),
    sharex=True
)


for ax, ecosystem in zip(
    axes,
    ECOSYSTEMS.keys()
):

    ecosystem_id = ECOSYSTEMS[
        ecosystem
    ]


    rzsm_series = get_ecosystem_series(
        rzsm_stack,
        ecosystem_mask,
        ecosystem_id
    )


    stationary_ssmi3 = (
        calculate_stationary_ssmi3(
            rzsm_series
        )
    )


    moving_ssmi3 = (
        calculate_moving_baseline_ssmi3(
            rzsm_series,
            window_years=10,
            min_years=8
        )
    )


    ax.plot(
        stationary_ssmi3.loc[
            "2015-01-01":"2025-12-01"
        ],
        linewidth=1.5,
        label="Stationary SSMI3"
    )


    ax.plot(
        moving_ssmi3.loc[
            "2015-01-01":"2025-12-01"
        ],
        linewidth=1.5,
        label="10-year moving-baseline SSMI3"
    )


    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )


    ax.set_ylabel(
        "Index"
    )


    ax.set_title(
        ecosystem
    )


    ax.grid(
        alpha=0.25
    )


axes[-1].set_xlabel(
    "Year"
)


axes[0].legend(
    loc="upper right"
)


fig.suptitle(
    "Stationary and 10-Year Moving-Baseline SSMI3",
    fontsize=14,
    fontweight="bold"
)


plt.tight_layout()


plt.savefig(
    "Fig10_Stationary_vs_MovingBaseline.tif",
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# -------------------------------------------------------------------------
# 11. Figure 11
#
# Compare propagation curves obtained from stationary and
# 10-year moving-baseline SSMI3.
# -------------------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
    sharey=True
)


for ax, ecosystem in zip(
    axes,
    ECOSYSTEMS.keys()
):

    stationary = results_df[
        (results_df["Ecosystem"] == ecosystem)
        &
        (results_df["Baseline"] == "Stationary")
    ].sort_values(
        "Lag_months"
    )


    moving = results_df[
        (results_df["Ecosystem"] == ecosystem)
        &
        (results_df["Baseline"] == "Moving")
    ].sort_values(
        "Lag_months"
    )


    ax.plot(
        stationary["Lag_months"],
        stationary["Pearson_r"],
        marker="o",
        linewidth=2,
        label="Stationary"
    )


    ax.plot(
        moving["Lag_months"],
        moving["Pearson_r"],
        marker="s",
        linewidth=2,
        label="Moving 10-year"
    )


    ax.axhline(
        0,
        linestyle=":",
        linewidth=1
    )


    ax.set_title(
        ecosystem,
        fontsize=12,
        fontweight="bold"
    )


    ax.set_xlabel(
        "Lag (months)"
    )


    ax.set_xticks(
        np.arange(13)
    )


    ax.grid(
        alpha=0.25
    )


axes[0].set_ylabel(
    "Correlation coefficient (r)"
)


axes[0].legend()


fig.suptitle(
    "Drought Propagation under Stationary and Moving-Baseline SSMI3",
    fontsize=14,
    fontweight="bold"
)


plt.tight_layout()


plt.savefig(
    "Fig11_Propagation_Comparison_10yr.tif",
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# -------------------------------------------------------------------------
# 12. Final summary
# -------------------------------------------------------------------------

print("\n" + "=" * 70)
print(
    "10-YEAR MOVING-BASELINE ANALYSIS COMPLETED"
)
print("=" * 70)

print(
    peak_df.to_string(
        index=False
    )
)

print("=" * 70)
