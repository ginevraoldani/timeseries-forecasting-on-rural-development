import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima.utils import ndiffs

def test_stationarity(series: pd.Series, verbose=False) -> bool:
    """
    Checks stationarity using the Augmented Dickey-Fuller (ADF) test.
    It evaluates both 'constant' (level stationarity) and 'constant + trend' 
    (trend stationarity) hypotheses to avoid false negatives on macroeconomic data.

    Args:
        series (pd.Series): The time series to test.
        verbose (bool): If True, prints the p-values of the tests.

    Returns:
        bool: True if the series is stationary (rejects the null hypothesis of 
            non-stationarity: p-value < 0.05 in at least one test), False otherwise.
    """
    clean_series = series.dropna()
    
    # Test with Constant (checks for level stationarity)
    # autolag='AIC' automatically selects the optimal lag length
    res_c = adfuller(clean_series, regression='c', autolag='AIC')
    p_val_c = res_c[1]
    
    # Test with Constant + Linear Trend (checks for trend stationarity)
    res_ct = adfuller(clean_series, regression='ct', autolag='AIC')
    p_val_ct = res_ct[1]
    
    if verbose:
        print(f"ADF (c) p-val: {p_val_c:.4f} | ADF (ct) p-val: {p_val_ct:.4f}")

    # If at least one test rejects the null hypothesis (Non-Stationary), 
    # we consider the series stationary.
    return (p_val_c < 0.05) or (p_val_ct < 0.05)

def find_integration_order(series: pd.Series, max_d=2) -> int:
    """
    Determines the optimal order of integration (d) using a hybrid approach 
    combining the Box-Jenkins variance criterion and the KPSS test.
    
    Methodology:
    1. Variance Criterion: Selects 'd' that minimizes the standard deviation of the series 
    (Box & Jenkins, 1970), penalizing over-differencing.
    2. KPSS Test: Uses the Hyndman-Khandakar algorithm (2008) to statistically 
    test for stationarity.
    
    Args:
        series (pd.Series): The time series to analyze.
        max_d (int, optional): Maximum number of differentiations to test. Defaults to 2.
        
    Returns:
        int: The optimal integration order 'd' (0, 1, or 2).
    """
    y = series.dropna()
    
    # CRITERION 1: Minimum Variance (Box-Jenkins)
    # We calculate the standard deviation for d=0, d=1, and d=2.
    # The 'd' that minimizes variance is often the most parsimonious choice.
    std_d0 = y.std()
    std_d1 = y.diff().dropna().std()
    
    # Calculate d=2 only if possible (series length permitting)
    if len(y) > 3:
        std_d2 = y.diff().diff().dropna().std()
    else:
        std_d2 = float('inf') # Penalize d=2 on very short series

    # Find d with minimum variance
    stds = [std_d0, std_d1, std_d2]
    best_d_var = np.argmin(stds)
    
    # CRITERION 2: KPSS Test (Hyndman-Khandakar)
    # Standard algorithmic check used in auto.arima
    try:
        kpss_d = ndiffs(y, alpha=0.05, test='kpss', max_d=max_d)
    except:
        kpss_d = 1 # Safe fallback if test fails on edge cases
        
    print(f"  StdDev (d=0): {std_d0:.4f}")
    print(f"  StdDev (d=1): {std_d1:.4f}")
    print(f"  StdDev (d=2): {std_d2:.4f}")
    print(f"  -> Optimal d (Variance Rule): {best_d_var}")
    print(f"  -> Optimal d (KPSS Test):     {kpss_d}")
    
    # FINAL DECISION LOGIC
    # Priority is given to the Variance Rule as it is more robust on short/noisy data,
    # preventing over-differencing (d=2) which destroys signal.
    final_d = best_d_var
    
    # Correction 1: Conservative approach for d=2
    # If Variance suggests 2 but KPSS is happy with 1, stick to 1 to preserve information.
    if final_d == 2 and kpss_d == 1: 
        final_d = 1
        
    # Correction 2: Handling mild trends
    # If Variance suggests 0 (flat variance) but KPSS detects a stochastic trend, trust KPSS.
    if final_d == 0 and kpss_d == 1: 
        final_d = 1
    
    # Ensure we don't exceed the user-defined max_d
    final_d = min(final_d, max_d)

    print(f"  => FINAL DECISION: d={final_d}")
    return final_d