# =========================================================================
# Script 09: Moving-Baseline Window Sensitivity Analysis
#
# Purpose:
#   Compare stationary SSMI3 with moving-baseline SSMI3 calculated using
#   trailing 8-, 10-, and 12-year monthly climatological windows.
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
# Moving-baseline method:
#   For each month, only observations from the previous N years of the
#   same calendar month are used to calculate the mean and standard
#   deviation.
#
# Windows tested:
#   8 years
#   10 years
#   12 years
#
# Propagation lags:
#   0-12 months
#
# Outputs:
#   - MovingBaseline_Window_Sensitivity.csv
#   - MovingBaseline_Peak_Sensitivity.csv
#   - Figure 8: Stationary vs moving-baseline SSMI3
#   - Figure 9: Propagation curves for 8-, 10-, and 12-year baselines
# =========================================================================


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray as rxr
from scipy.stats import pearsonr


# -------------------------------------------------------------------------
# 1. Load datasets
# -------------------------------------------------------------------------

print("Loading RZSM, NDVI anomaly, and ecosystem mask...")


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
        "RZSM stack does not contain 312 monthly observations."
    )


if ndvi_stack.sizes["band"] != len(timeline):
    raise ValueError(
        "NDVI anomaly stack does not contain 312 monthly observations."
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
#
# These IDs must match the ecosystem mask generated in Script 01.
# -------------------------------------------------------------------------

ECOSYSTEMS = {
    "Forest": 1,
    "Grassland": 3,
    "Cropland": 4
}


# -------------------------------------------------------------------------
# 4. Extract ecosystem-level time series
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
# 5. Stationary SSMI3
#
# Fixed reference period:
#   2000-2014
#
# Standardization is performed separately for each calendar month.
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

    ssmi_values = []

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

            ssmi_values.append(np.nan)

        else:

            ssmi_values.append(
                (row["RZSM"] - mu) / sigma
            )

    return pd.Series(
        ssmi_values,
        index=rzsm_series.index,
        name="SSMI3_Stationary"
    )


# -------------------------------------------------------------------------
# 6. Moving-baseline SSMI3
#
# For each month:
#
#   current RZSM
#       ↓
#   previous N years
#       ↓
#   same calendar month only
#       ↓
#   monthly mean and standard deviation
#       ↓
#   standardized current value
#
# No future observations are used.
# -------------------------------------------------------------------------

def calculate_moving_baseline_ssmi3(
    rzsm_series,
    window_years,
    min_years=None
):

    if min_years is None:
        min_years = window_years

    df = pd.DataFrame({
        "RZSM": rzsm_series
    })

    df["Year"] = df.index.year
    df["Month"] = df.index.month

    moving_values = []

    for date, row in df.iterrows():

        current_year = date.year
        current_month = date.month

        history_start = (
            current_year - window_years
        )

        history_end = (
            current_year - 1
        )

        history = df[
            (df["Month"] == current_month)
            &
            (df["Year"] >= history_start)
            &
            (df["Year"] <= history_end)
        ]["RZSM"].dropna()

        # -------------------------------------------------------------
        # Require a complete historical window
        # -------------------------------------------------------------

        if len(history) < min_years:

            moving_values.append(
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

            moving_values.append(
                np.nan
            )

        else:

            moving_values.append(
                (row["RZSM"] - mu) / sigma
            )

    return pd.Series(
        moving_values,
        index=rzsm_series.index,
        name=f"SSMI3_Moving_{window_years}yr"
    )


# -------------------------------------------------------------------------
# 7. Lagged correlation
#
# Lag 0:
#   concurrent soil-moisture and vegetation anomalies.
#
# Positive lag:
#   vegetation response follows the preceding soil-moisture anomaly.
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

        x = x.values[valid]
        y = y.values[valid]

        if len(x) < 3:

            r_values.append(
                np.nan
            )

            continue

        if (
            np.std(x) == 0
            or np.std(y) == 0
        ):

            r_values.append(
                np.nan
            )

            continue

        r, _ = pearsonr(
            x,
            y
        )

        r_values.append(
            r
        )

    return np.asarray(
        r_values,
        dtype=float
    )


# -------------------------------------------------------------------------
# 8. Define moving-baseline windows
# -------------------------------------------------------------------------

WINDOWS = [
    8,
    10,
    12
]

LAGS = np.arange(
    0,
    13
)


# -------------------------------------------------------------------------
# 9. Run sensitivity analysis
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
    # Stationary SSMI3
    # -------------------------------------------------------------

    stationary_ssmi3 = (
        calculate_stationary_ssmi3(
            rzsm_series
        )
    )

    stationary_eval = (
        stationary_ssmi3
        .loc["2015-01-01":"2025-12-01"]
    )

    ndvi_eval = (
        ndvi_series
        .loc["2015-01-01":"2025-12-01"]
    )

    stationary_r = (
        calculate_lagged_correlations(
            stationary_eval,
            ndvi_eval,
            max_lag=12
        )
    )

    stationary_peak_index = (
        np.nanargmax(
            stationary_r
        )
    )

    stationary_peak_r = (
        stationary_r[
            stationary_peak_index
        ]
    )

    stationary_peak_lag = (
        LAGS[
            stationary_peak_index
        ]
    )

    # -------------------------------------------------------------
    # Save stationary lag-by-lag values
    # -------------------------------------------------------------

    for lag, r in zip(
        LAGS,
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

    # -------------------------------------------------------------
    # Moving-baseline windows
    # -------------------------------------------------------------

    for window in WINDOWS:

        print(
            f"  Moving baseline: {window} years"
        )

        moving_ssmi3 = (
            calculate_moving_baseline_ssmi3(
                rzsm_series,
                window_years=window
            )
        )

        moving_eval = (
            moving_ssmi3
            .loc["2015-01-01":"2025-12-01"]
        )

        moving_r = (
            calculate_lagged_correlations(
                moving_eval,
                ndvi_eval,
                max_lag=12
            )
        )

        moving_peak_index = (
            np.nanargmax(
                moving_r
            )
        )

        moving_peak_r = (
            moving_r[
                moving_peak_index
            ]
        )

        moving_peak_lag = (
            LAGS[
                moving_peak_index
            ]
        )

        delta_r = (
            moving_peak_r
            -
            stationary_peak_r
        )

        # ---------------------------------------------------------
        # Store lag-by-lag results
        # ---------------------------------------------------------

        for lag, r in zip(
            LAGS,
            moving_r
        ):

            all_results.append(
                {
                    "Ecosystem": ecosystem,
                    "Baseline": "Moving",
                    "Window_years": window,
                    "Lag_months": lag,
                    "Pearson_r": r
                }
            )

        # ---------------------------------------------------------
        # Store peak results
        # ---------------------------------------------------------

        peak_results.append(
            {
                "Ecosystem": ecosystem,
                "Window_years": window,
                "Stationary_peak_r": stationary_peak_r,
                "Stationary_peak_lag": stationary_peak_lag,
                "Moving_peak_r": moving_peak_r,
                "Moving_peak_lag": moving_peak_lag,
                "Delta_peak_r": delta_r
            }
        )

        print(
            f"    Peak r = {moving_peak_r:.3f}, "
            f"Lag = {moving_peak_lag}, "
            f"Delta r = {delta_r:+.3f}"
        )


# -------------------------------------------------------------------------
# 10. Save complete lag-by-lag results
# -------------------------------------------------------------------------

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    "MovingBaseline_Window_Sensitivity.csv",
    index=False
)


# -------------------------------------------------------------------------
# 11. Save peak results
# -------------------------------------------------------------------------

peak_df = pd.DataFrame(
    peak_results
)

peak_df.to_csv(
    "MovingBaseline_Peak_Sensitivity.csv",
    index=False
)


# -------------------------------------------------------------------------
# 12. Create Table 3-style summary
# -------------------------------------------------------------------------

table3 = peak_df[
    [
        "Ecosystem",
        "Stationary_peak_r",
        "Stationary_peak_lag",
        "Window_years",
        "Moving_peak_r",
        "Moving_peak_lag",
        "Delta_peak_r"
    ]
].copy()


table3 = table3.sort_values(
    [
        "Ecosystem",
        "Window_years"
    ]
)


table3.to_csv(
    "Table3_MovingBaseline_Sensitivity.csv",
    index=False
)


# -------------------------------------------------------------------------
# 13. Figure 8
#
# Stationary vs moving-baseline SSMI3.
# The three moving windows are shown for each ecosystem.
# -------------------------------------------------------------------------

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 10),
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

    stationary = (
        calculate_stationary_ssmi3(
            rzsm_series
        )
        .loc["2015-01-01":"2025-12-01"]
    )

    ax.plot(
        stationary.index,
        stationary.values,
        linewidth=1.5,
        label="Stationary"
    )

    for window in WINDOWS:

        moving = (
            calculate_moving_baseline_ssmi3(
                rzsm_series,
                window_years=window
            )
            .loc["2015-01-01":"2025-12-01"]
        )

        ax.plot(
            moving.index,
            moving.values,
            linewidth=1.2,
            label=f"Moving {window}-year"
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        ecosystem,
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        "SSMI3"
    )

    ax.grid(
        alpha=0.25
    )


axes[-1].set_xlabel(
    "Year"
)


axes[0].legend(
    ncol=4,
    loc="upper center"
)


fig.suptitle(
    "Stationary and Moving-Baseline SSMI3",
    fontsize=14,
    fontweight="bold"
)


plt.tight_layout()


plt.savefig(
    "Fig8_Stationary_vs_MovingBaseline.tif",
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# -------------------------------------------------------------------------
# 14. Figure 9
#
# Lagged propagation curves for the three moving-baseline windows.
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

    # -------------------------------------------------------------
    # Stationary
    # -------------------------------------------------------------

    stationary = results_df[
        (results_df["Ecosystem"] == ecosystem)
        &
        (results_df["Baseline"] == "Stationary")
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

    # -------------------------------------------------------------
    # Moving baselines
    # -------------------------------------------------------------

    for window in WINDOWS:

        moving = results_df[
            (results_df["Ecosystem"] == ecosystem)
            &
            (results_df["Baseline"] == "Moving")
            &
            (results_df["Window_years"] == window)
        ].sort_values(
            "Lag_months"
        )

        ax.plot(
            moving["Lag_months"],
            moving["Pearson_r"],
            marker="s",
            linewidth=1.8,
            label=f"{window}-year"
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
        LAGS
    )

    ax.grid(
        alpha=0.25
    )


axes[0].set_ylabel(
    "Correlation coefficient (r)"
)


axes[0].legend()


fig.suptitle(
    "Sensitivity of Drought Propagation to Moving-Baseline Length",
    fontsize=14,
    fontweight="bold"
)


plt.tight_layout()


plt.savefig(
    "Fig9_MovingBaseline_Window_Sensitivity.tif",
    dpi=600,
    bbox_inches="tight"
)


plt.show()


# -------------------------------------------------------------------------
# 15. Print final summary
# -------------------------------------------------------------------------

print("\n" + "=" * 75)
print(
    "MOVING-BASELINE WINDOW SENSITIVITY ANALYSIS COMPLETED"
)
print("=" * 75)

print(
    table3.to_string(
        index=False
    )
)

print("=" * 75)

print(
    "Tested windows: 8, 10, and 12 years"
)

print(
    "Evaluation period: 2015-2025"
)

print(
    "Propagation lags: 0-12 months"
)

print("=" * 75)
