import pandas as pd
import os
import traceback
from collections.abc import MutableMapping
from datetime import datetime
import src.evaluation as eval
from src.config import (
    PERFORMANCE_FILE, 
    PARAMS_FILE, 
    RESIDUALS_FILE, 
    PREDICTIONS_TEST_FILE,
    PREDICTIONS_FUTURE_FILE
)

# --- HELPERS ---

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def _save_to_excel_row(filepath, new_data_dict, keys_to_match):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    clean_dict = {k: str(v) if isinstance(v, (list, tuple, dict)) else v for k, v in new_data_dict.items()}
    new_df = pd.DataFrame([clean_dict])
    
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_excel(filepath)
            if not existing_df.empty:
                condition = pd.Series([True] * len(existing_df))
                for key in keys_to_match:
                    if key in existing_df.columns:
                        condition &= (existing_df[key].astype(str) == str(new_data_dict.get(key)))
                existing_df = existing_df[~condition]
                final_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                final_df = new_df
        except Exception:
            final_df = new_df
    else:
        final_df = new_df
    final_df.to_excel(filepath, index=False)

def _save_to_excel_bulk(filepath, new_df, base_info):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_excel(filepath)
            # Rimuovi vecchi dati per questo modello
            mask = (existing_df['indicator'] == base_info['indicator']) & \
                (existing_df['model'] == base_info['model']) & \
                (existing_df['configuration'] == base_info['configuration'])
            existing_df = existing_df[~mask]
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            print(f"File corrotto o illeggibile: {filepath}. Rigenero il file.")
            final_df = new_df
    else:
        final_df = new_df
    final_df.to_excel(filepath, index=False)

# --- REPORTING FUNCTIONS ---

def _report_params(base_info, params):
    flat_params = flatten_dict(params) if isinstance(params, dict) else {"params_str": str(params)} if params else {}
    _save_to_excel_row(PARAMS_FILE, {**base_info, **flat_params}, list(base_info.keys()))

def _report_performance(base_info, y_true, y_pred, training_time):
    metrics = eval.compute_errors(y_true, y_pred)
    data = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **base_info, "training_time_sec": training_time, **metrics}
    _save_to_excel_row(PERFORMANCE_FILE, data, list(base_info.keys()))

def _report_residuals(base_info, y_true, y_pred):
    diagnostics = eval.compute_residual_diagnostics(y_true, y_pred)
    if diagnostics:
        _save_to_excel_row(RESIDUALS_FILE, {**base_info, **diagnostics}, list(base_info.keys()))

def _report_test_predictions(base_info, years, y_true, y_pred):
    # SAFETY: Convertiamo tutto in liste/array semplici per rompere dipendenze dagli indici Pandas
    years_safe = list(years) if hasattr(years, 'tolist') else list(years)
    y_true_safe = y_true.tolist() if hasattr(y_true, 'tolist') else list(y_true)
    y_pred_safe = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)

    df = pd.DataFrame({'year': years_safe, 'y_true': y_true_safe, 'y_pred': y_pred_safe})
    for k, v in base_info.items(): df[k] = v
    
    cols = list(base_info.keys()) + ['year', 'y_true', 'y_pred']
    _save_to_excel_bulk(PREDICTIONS_TEST_FILE, df[cols], base_info)

def _report_future_forecasts(base_info, future_df):
    if future_df is None or future_df.empty: return
    df = future_df.copy()
    rename_map = {'Year': 'year', 'Pred': 'y_pred', 'Lower': 'lower_ci', 'Upper': 'upper_ci', 'lower': 'lower_ci', 'upper': 'upper_ci', 'pred': 'y_pred'}
    df = df.rename(columns=rename_map)
    for k, v in base_info.items(): df[k] = v
    
    expected = list(base_info.keys()) + ['year', 'y_pred']
    optional = ['lower_ci', 'upper_ci']
    final_cols = expected + [c for c in optional if c in df.columns]
    _save_to_excel_bulk(PREDICTIONS_FUTURE_FILE, df[final_cols], base_info)

# --- MAIN ---

def save_experiment_results(indicator, model_name, configuration, y_test, y_pred, years_test, y_train=None, params=None, training_time=None, future_predictions=None):
    base_info = {"indicator": indicator, "model": model_name, "configuration": configuration}
    print(f"Saving results for {model_name} | {indicator}...")
    try:
        _report_params(base_info, params)
        _report_performance(base_info, y_test, y_pred, training_time)
        _report_residuals(base_info, y_test, y_pred)
        if len(years_test) == len(y_pred):
            _report_test_predictions(base_info, years_test, y_test, y_pred)
        if future_predictions is not None:
            _report_future_forecasts(base_info, future_predictions)
        print("Save complete.")
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()