# =========================================================================
# File 08: Non-Stationary Drought Characterization Framework
#
# Implements a 10-Year Trailing Monthly Climatology
# for calculation of the Non-Stationary Standardized
# Soil Moisture Index (NSSI3).
#
# Monthly standardization is performed using the
# previous 10 years of the same calendar month.
#
# This approach removes seasonal effects while
# allowing climatological reference conditions
# to evolve through time.
#
# Used for Figure 10, Figure 11, and Table 3.
# =========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# -------------------------------------------------------------------------
# 1. Load Data
# -------------------------------------------------------------------------

rzsm_series = pd.read_csv(
    "grassland_rzsm_3m.csv",
    index_col=0,
    parse_dates=True
).iloc[:, 0]

ndvi_anom = pd.read_csv(
    "grassland_ndvi.csv",
    index_col=0,
    parse_dates=True
).iloc[:, 0]

# -------------------------------------------------------------------------
# 2. Build Non-Stationary SSI3 (NSSI3)
#    10-Year Trailing Monthly Climatology
# -------------------------------------------------------------------------

df = pd.DataFrame({
    "time": rzsm_series.index,
    "rzsm": rzsm_series.values
})

df["year"] = df["time"].year
df["month"] = df["time"].month

nssi_values = []

for _, row in df.iterrows():

    yr = row["year"]
    mn = row["month"]

    history = df[
        (df["month"] == mn) &
        (df["year"] >= yr - 10) &
        (df["year"] < yr)
    ]["rzsm"].dropna()

    # Require sufficient historical observations
    if len(history) < 8:
        nssi_values.append(np.nan)
        continue

    mu = history.mean()
    sigma = history.std()

    if sigma == 0:
        nssi_values.append(np.nan)
    else:
        nssi_values.append(
            (row["rzsm"] - mu) / sigma
        )

df["NSSI3"] = nssi_values

nssi3_all = pd.Series(
    df["NSSI3"].values,
    index=df["time"]
).dropna()

# -------------------------------------------------------------------------
# 3. Validation Period (2015–2025)
# -------------------------------------------------------------------------

validation_scope = slice(
    "2015-01-01",
    "2025-12-01"
)

nssi3 = nssi3_all.loc[validation_scope]

ndvi_validation = ndvi_anom.loc[
    validation_scope
]

# -------------------------------------------------------------------------
# 4. Stationary SSI3 Reference
#    Fixed Baseline: 2000–2014
# -------------------------------------------------------------------------

ref_period = rzsm_series.loc[
    "2000-01-01":"2014-12-31"
]

ref_mean = ref_period.mean()
ref_std = ref_period.std()

ssi3 = (
    (rzsm_series - ref_mean)
    / ref_std
).loc[validation_scope]

# -------------------------------------------------------------------------
# 5. Lagged Correlation Analysis
# -------------------------------------------------------------------------

def lagged_correlations(
    forcing,
    response,
    max_lag=12
):

    r_values = []

    common_index = (
        forcing.index.intersection(
            response.index
        )
    )

    forcing = forcing.loc[
        common_index
    ]

    response = response.loc[
        common_index
    ]

    for lag in range(max_lag + 1):

        if lag == 0:

            r = pearsonr(
                forcing,
                response
            )[0]

        else:

            r = pearsonr(
                forcing.iloc[:-lag],
                response.iloc[lag:]
            )[0]

        r_values.append(r)

    return r_values

# -------------------------------------------------------------------------
# 6. Extract Peak Metrics
# -------------------------------------------------------------------------

def peak_parameters(
    forcing,
    response
):

    r_values = lagged_correlations(
        forcing,
        response
    )

    peak_r = np.max(r_values)

    peak_lag = np.argmax(r_values)

    return peak_r, peak_lag, r_values

ssi_peak_r, ssi_peak_lag, ssi_curve = (
    peak_parameters(
        ssi3,
        ndvi_validation
    )
)

nssi_peak_r, nssi_peak_lag, nssi_curve = (
    peak_parameters(
        nssi3,
        ndvi_validation
    )
)

# -------------------------------------------------------------------------
# 7. Console Output
# -------------------------------------------------------------------------

print("=" * 70)
print("GRASSLAND DROUGHT PROPAGATION ANALYSIS")
print("=" * 70)

print(
    f"SSI3 Peak r = {ssi_peak_r:.3f} "
    f"(Lag = {ssi_peak_lag} months)"
)

print(
    f"NSSI3 Peak r = {nssi_peak_r:.3f} "
    f"(Lag = {nssi_peak_lag} months)"
)

print(
    f"Delta r = "
    f"{nssi_peak_r - ssi_peak_r:+.3f}"
)

print("=" * 70)

# -------------------------------------------------------------------------
# 8. Figure 10
# -------------------------------------------------------------------------

plt.figure(
    figsize=(12,5)
)

plt.plot(
    ssi3.index,
    ssi3,
    label="SSI3",
    linewidth=2
)

plt.plot(
    nssi3.index,
    nssi3,
    label="NSSI3",
    linewidth=2
)

plt.axhline(
    0,
    color="black",
    linestyle="--"
)

plt.legend()

plt.title(
    "Stationary vs Non-Stationary Drought Index"
)

plt.ylabel("Index Value")

plt.tight_layout()

plt.show()

# -------------------------------------------------------------------------
# 9. Figure 11 Curve
# -------------------------------------------------------------------------

lags = np.arange(13)

plt.figure(
    figsize=(7,5)
)

plt.plot(
    lags,
    ssi_curve,
    marker="o",
    linewidth=2,
    label="SSI3"
)

plt.plot(
    lags,
    nssi_curve,
    marker="s",
    linewidth=2,
    label="NSSI3"
)

plt.axhline(
    0,
    color="black",
    linestyle="--"
)

plt.xlabel("Lag (months)")
plt.ylabel("Correlation (r)")
plt.legend()

plt.title(
    "Drought Propagation Comparison"
)

plt.tight_layout()

plt.show()
