# =========================================================================
# File 07: Threshold Sensitivity Analysis and Stability Testing Framework
# Iterates and contrasts correlations under alternative drought definitions (Fig 9)
# =========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

ssi3 = pd.read_csv("cropland_ssi3.csv", index_col=0, parse_dates=True).iloc[:,0]
ndvi = pd.read_csv("cropland_ndvi.csv", index_col=0, parse_dates=True).iloc[:,0]

thresholds = [-0.5, -0.75, -1.0]
colors = ['orange', 'red', 'darkred']
plt.figure(figsize=(8, 5))

for ts, col in zip(thresholds, colors):
    # Isolate timestamps where conditions drop below targeted severe filters
    active_mask = ssi3 < ts
    
    r_sensitivity = []
    for lag in range(13):
        if lag == 0:
            r, _ = pearsonr(ssi3[active_mask], ndvi[active_mask])
        else:
            r, _ = pearsonr(ssi3[:-lag][active_mask[:-lag]], ndvi[lag:][active_mask[:-lag]])
        r_sensitivity.append(r)
        
    plt.plot(np.arange(13), r_sensitivity, marker='o', color=col, linewidth=2, label=f'Threshold Condition (SSI3 < {ts})')

plt.axhline(0, color='gray', linestyle=':')
plt.title('Cropland Sensitivity Analysis Curves (Figure 9 Match)')
plt.xlabel('Lag (months)')
plt.ylabel('Correlation Coefficient (r)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("Fig9_Threshold_Sensitivity_Croplands.tif", dpi=300)
plt.show()
