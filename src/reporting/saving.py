import src.evaluation as eval
import pandas as pd
import os
import json
from datetime import datetime
from src.config import (
    PERFORMANCE_FILE, 
    PARAMS_FILE, 
    RESIDUALS_FILE, 
    PREDICTIONS_FILE
)

def _save_to_excel(filepath, new_data_dict, keys_to_match):
    """
    Saves a dictionary (record) to Excel, handling overwrites and new columns automatically.
    
    If 'new_data_dict' contains keys that are not yet in the Excel file (e.g., specific parameters),
    Pandas will automatically add those columns and fill missing values for other rows with NaN.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    new_df = pd.DataFrame([new_data_dict])
    
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_excel(filepath)
            
            if not existing_df.empty:
                # Identify rows to overwrite based on Primary Keys
                condition = pd.Series([True] * len(existing_df))
                for key in keys_to_match:
                    if key in existing_df.columns:
                        condition &= (existing_df[key] == new_data_dict.get(key))
                
                # Remove old rows matching the key
                existing_df = existing_df[~condition]
                
                # Concatenate (Pandas handles column alignment/new columns)
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                final_df = new_df
        except Exception as e:
            print(f"Error reading {filepath}: {e}. Creating new file.")
            final_df = new_df
    else:
        final_df = new_df

    final_df.to_excel(filepath, index=False)
    
def _report_params(base_info, params, path=PARAMS_FILE):
    """
    Saves hyperparameters. Expands the params dict into separate columns.
    """
    if not params:
        # If no params (e.g. Naive), we still save the row to track it exists
        data_to_save = base_info.copy()
    else:
        # MERGE: base_info + params
        # Example: {'indicator': 'GDP', ...} + {'p': 1, 'q': 1}
        # Result: {'indicator': 'GDP', ..., 'p': 1, 'q': 1}
        data_to_save = {**base_info, **params}
    
    _save_to_excel(
        path, 
        data_to_save, 
        keys_to_match=["indicator", "model", "configuration"]
    )

def _report_performance(base_info, y_true, y_pred, training_time=None, path=PERFORMANCE_FILE):
    """
    Calculates metrics and saves them to the performance log.
    Includes a timestamp.
    """
    # Calculate metrics using your centralized logic
    metrics = eval.compute_errors(y_true, y_pred)
    
    data_to_save = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **base_info,  # Keys
        "training_time_sec": training_time,
        **metrics     # Expands RMSE, MAE, MAPE, R2
    }
    
    _save_to_excel(
        path, 
        data_to_save, 
        keys_to_match=["indicator", "model", "configuration"]
    )

def _report_residuals(base_info, y_true, y_pred, path=RESIDUALS_FILE):
    """
    Calculates residuals diagnostics and saves them.
    """
    diagnostics = eval.compute_residual_diagnostics(y_true, y_pred)
    
    if diagnostics:
        data_to_save = {
            **base_info,
            **diagnostics
        }
        
        _save_to_excel(
            path, 
            data_to_save, 
            keys_to_match=["indicator", "model", "configuration"]
        )
        
def _report_predictions(base_info, years, y_true, y_pred):
    """
    Saves the time series of predictions.
    Note: This file grows long (Long Format).
    """
    # Create a temporary DF
    pred_df = pd.DataFrame({
        'year': years,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    # Add metadata columns to every row
    for k, v in base_info.items():
        pred_df[k] = v
        
    # Reorder columns for readability (Keys first)
    cols = list(base_info.keys()) + ['year', 'y_true', 'y_pred']
    pred_df = pred_df[cols]
    
    # Saving logic for predictions is slightly different:
    # We remove all previous rows for this specific model/indicator and append the new series
    if os.path.exists(PREDICTIONS_FILE):
        try:
            existing_df = pd.read_excel(PREDICTIONS_FILE)
            # Filter out old predictions for this specific combo
            mask = (existing_df['indicator'] == base_info['indicator']) & \
                (existing_df['model'] == base_info['model']) & \
                (existing_df['configuration'] == base_info['configuration'])
            existing_df = existing_df[~mask]
            
            final_df = pd.concat([existing_df, pred_df], ignore_index=True)
        except:
            final_df = pred_df
    else:
        final_df = pred_df
        
    final_df.to_excel(PREDICTIONS_FILE, index=False)

def save_experiment_results(
    indicator, 
    model_name, 
    configuration, 
    y_train, # Not used for saving, but kept for interface consistency if needed
    y_test, 
    y_pred, 
    years_test, # Necessary for plotting/saving predictions against time
    params=None, 
    training_time=None
):
    """
    Main entry point. Call this function at the end of your notebook loop.
    It distributes the data to the specific reporting functions.
    """
    
    # Define Primary Key (Centralized)
    base_info = {
        "indicator": indicator,
        "model": model_name,
        "configuration": configuration
    }
    
    print(f"Saving results for {model_name} - {indicator}...")

    try:
        _report_params(base_info, params)
        _report_performance(base_info, y_test, y_pred, training_time)
        _report_residuals(base_info, y_test, y_pred)
        
        # Ensure array alignment for predictions
        # (Assuming years_test, y_test, y_pred have same length)
        if len(years_test) == len(y_pred):
            _report_predictions(base_info, years_test, y_test, y_pred)
        else:
            print(f"Skipping prediction save: Length mismatch (Years: {len(years_test)}, Pred: {len(y_pred)})")

        print("Save complete.")
        
    except Exception as e:
        print(f"Error during saving: {e}")