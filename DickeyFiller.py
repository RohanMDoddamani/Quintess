import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Generate a stationary time series (White Noise)
np.random.seed(42)
n = 100
white_noise = np.random.normal(0, 1, n)

# Perform Augmented Dickey-Fuller (ADF) test
adf_result = adfuller(white_noise)

print(f"ADF Statistic: {adf_result[0]}")
print(f"P-Value: {adf_result[1]}")
if adf_result[1] < 0.05:
    print("Stationary process (Reject Null Hypothesis)")
else:
    print("Non-Stationary process (Fail to Reject Null Hypothesis)")
