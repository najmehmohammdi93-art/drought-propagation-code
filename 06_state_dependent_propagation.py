# =========================================================================
# File 06: State-Dependent Propagation Analytics Engine
# Segregates baseline by Wet (>0.5), Normal (-0.5 to 0.5), and Dry (< -0.5)
# Generates multi-panel state comparison curves matching Figure 8
# =========================================================================

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# 1. Load Ecosystem Time series extracted from File 04
# Loading Forest as primary example for scripting completeness
ssmi3_series = pd.read_csv("forest_ssmi3.csv", index_col=0, parse_dates=True).iloc[:,0]
ndvi_series = pd.read_csv("forest_ndvi.csv", index_col=0, parse_dates=True).iloc[:,0]

# 2. Partition Temporal Indexes based on Hydroclimatic Background States
wet_indices = ssmi3_series > 0.5
normal_indices = (ssmi3_series >= -0.5) & (ssmi3_series <= 0.5)
dry_indices = ssmi3_series < -0.5

def get_state_lagged_r(forcing, response, structural_mask, max_lag=12):
    r_vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            f_slice = forcing[structural_mask]
            r_slice = response[structural_mask]
        else:
            # Shift temporal response index to account for delay mechanisms
            f_slice = forcing[:-lag][structural_mask[:-lag]]
            r_slice = response[lag:][structural_mask[:-lag]]
            
        if len(f_slice) > 5: # Threshold limit to prevent sample sizing errors
            r, _ = pearsonr(f_slice, r_slice)
            r_vals.append(r)
        else:
            r_vals.append(np.nan)
    return r_vals

forest_wet_r = get_state_lagged_r(ssmi3_series, ndvi_series, wet_indices)
forest_norm_r = get_state_lagged_r(ssmi3_series, ndvi_series, normal_indices)
forest_dry_r = get_state_lagged_r(ssmi3_series, ndvi_series, dry_indices)

# 3. Create Multi-state Analysis Curves Plot (Figure 8 Structure)
lags = np.arange(0, 13)
plt.figure(figsize=(7, 5))
plt.plot(lags, forest_wet_r, marker='o', color='blue', linestyle='-', label='Wet State (SSMI3 > 0.5)')
plt.plot(lags, forest_norm_r, marker='s', color='green', linestyle='--', label='Normal State (-0.5 <= SSMI3 <= 0.5)')
plt.plot(lags, forest_dry_r, marker='^', color='red', linestyle='-.', label='Dry State (SSMI3 < -0.5)')
plt.axhline(0, color='black', alpha=0.3)
plt.title('Forest State-Dependent Propagation Response (Figure 8 Match)')
plt.xlabel('Lag Intervals (Months)')
plt.ylabel('Correlation (r)')
plt.legend()
plt.grid(alpha=0.2)
plt.savefig("Fig8_Forest_State_Dependent.tif", dpi=300)
plt.show()
