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
        bool: True if the series is stationary (p-value < 0.05 in either test), 
            False otherwise.
    """
    clean_series = series.dropna()
    
    # Test with Constant (checks for level stationarity)
    # autolag='AIC' automatically selects the optimal lag length
    res_c = adfuller(clean_series, regression='c', autolag='AIC')
    p_val_c = res_c[1]
    
    # Test with Constant + Linear Trend (checks for trend stationarity)
    # Crucial for economic series that grow over time (e.g., GDP)
    res_ct = adfuller(clean_series, regression='ct', autolag='AIC')
    p_val_ct = res_ct[1]
    
    if verbose:
        print(f"  ADF (c) p-val: {p_val_c:.4f} | ADF (ct) p-val: {p_val_ct:.4f}")

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
    
    # --- CRITERION 1: Minimum Variance (Box-Jenkins) ---
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
    
    # --- CRITERION 2: KPSS Test (Hyndman-Khandakar) ---
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
    
    # --- FINAL DECISION LOGIC ---
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

# def difference_series(series: pd.Series, d=1):
#     """
#     Applies 'd' orders of differentiation to make the series stationary.
#     Stores the initial values required to reverse the transformation.

#     Args:
#         series (pd.Series): The original time series.
#         d (int, optional): The integration order found via find_integration_order. Default is 1.

#     Returns:
#         tuple: (diff_series, initial_values)
#             - diff_series (pd.Series): The stationary, differenced series.
#             - initial_values (list): The starting values needed for inverse transformation.
#     """
#     if d == 0:
#         print("Series already stationary (d=0). No differencing applied.")
#         return series, []
    
#     print(f"Applying {d} order(s) of differencing...")
#     diff_series = series.copy()
#     initial_values = []
    
#     for _ in range(d):
#         # Save the first value before differencing (anchor for reconstruction)
#         initial_values.append(diff_series.iloc[0])        
#         diff_series = diff_series.diff().dropna()
        
#     return diff_series, initial_values

# def inverse_difference_series(pred_series: pd.Series, initial_values: list, d=1) -> pd.Series:
#     """
#     Reconstructs the original series (or forecasts) from the differenced values
#     using iterative cumulative summation.

#     Args:
#         pred_series (pd.Series): The differenced series (predictions or actuals).
#         initial_values (list): The anchor values from the original scale.
#                             - Use values from difference_series() for training reconstruction.
#                             - Use [last_observed_value] for future forecasting.
#         d (int): The order of integration used.

#     Returns:
#         pd.Series: The reconstructed series on the original scale.
#     """
#     if d == 0:
#         return pred_series
        
#     reconstructed = pred_series.copy()
    
#     # We apply cumulative sum iteratively in reverse order of differencing.
#     # Example: If d=2, we first integrate back to d=1, then to d=0.
#     for start_val in reversed(initial_values):
#         # We prepend the start_val to perform the cumulative sum
#         # Note: We reconstruct assuming 'start_val' is the point strictly preceding 'pred_series'
#         temp_series = pd.concat([pd.Series([start_val], index=[pred_series.index[0]]), reconstructed])
        
#         # Calculate cumsum ignoring the index temporarily to ensure continuity
#         # then re-apply the correct index (dropping the first dummy element)
#         cumsum_values = temp_series.values.cumsum()
#         reconstructed = pd.Series(cumsum_values[1:], index=pred_series.index)
        
#     return reconstructed