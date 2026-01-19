import pandas as pd
import os
import traceback
from collections.abc import MutableMapping
from datetime import datetime
import src.evaluation as eval
from src.config import (
    INTEGRATION_FILE,
    LEADERBOARD_FILE,
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

# def _save_to_excel_row(filepath, new_data_dict, keys_to_match):
#     os.makedirs(os.path.dirname(filepath), exist_ok=True)
#     clean_dict = {k: str(v) if isinstance(v, (list, tuple, dict)) else v for k, v in new_data_dict.items()}
#     new_df = pd.DataFrame([clean_dict])
    
#     if os.path.exists(filepath):
#         try:
#             existing_df = pd.read_excel(filepath)
#             if not existing_df.empty:
#                 condition = pd.Series([True] * len(existing_df))
#                 for key in keys_to_match:
#                     if key in existing_df.columns:
#                         condition &= (existing_df[key].astype(str) == str(new_data_dict.get(key)))
#                 existing_df = existing_df[~condition]
#                 final_df = pd.concat([existing_df, new_df], ignore_index=True)
#             else:
#                 final_df = new_df
#         except Exception:
#             final_df = new_df
#     else:
#         final_df = new_df
#     final_df.to_excel(filepath, index=False)

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

# # --- REPORTING FUNCTIONS ---

# def _report_params(base_info, params):
#     flat_params = flatten_dict(params) if isinstance(params, dict) else {"params_str": str(params)} if params else {}
#     _save_to_excel_row(PARAMS_FILE, {**base_info, **flat_params}, list(base_info.keys()))

# def _report_performance(base_info, y_true, y_pred, training_time):
#     metrics = eval.compute_errors(y_true, y_pred)
#     data = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **base_info, "training_time_sec": training_time, **metrics}
#     _save_to_excel_row(PERFORMANCE_FILE, data, list(base_info.keys()))

# def _report_residuals(base_info, y_true, y_pred):
#     diagnostics = eval.compute_residual_diagnostics(y_true, y_pred)
#     if diagnostics:
#         _save_to_excel_row(RESIDUALS_FILE, {**base_info, **diagnostics}, list(base_info.keys()))

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
    if isinstance(future_df, pd.Series):
        # Assumiamo che il nome della series sia la predizione
        # e l'indice sia l'anno (se è un DatetimeIndex, estraiamo l'anno)
        df = future_df.to_frame(name='y_pred')
        if hasattr(df.index, 'year'):
            df['year'] = df.index.year
        else:
            df['year'] = df.index
        df.reset_index(drop=True, inplace=True)
    else:
        df = future_df.copy()
    rename_map = {
        'Year': 'year',
        'Pred': 'y_pred',
        'Lower': 'lower_ci',
        'Upper': 'upper_ci',
        'lower': 'lower_ci',
        'upper': 'upper_ci',
        'pred': 'y_pred'
    }
    df = df.rename(columns=rename_map)
    for k, v in base_info.items(): df[k] = v
    
    expected = list(base_info.keys()) + ['year', 'y_pred']
    optional = ['lower_ci', 'upper_ci']
    final_cols = expected + [c for c in optional if c in df.columns]
    _save_to_excel_bulk(PREDICTIONS_FUTURE_FILE, df[final_cols], base_info)

# --- MAIN ---

def save_experiment_results(indicator, model_name, configuration, y_test, y_pred, years_test, y_train=None, params=None, training_time=None, future_predictions=None):
    print(f"Saving results for {model_name} | {indicator}...")
    try:
        all_metrics = {}
        if params:
            all_metrics.update(params)
        if training_time:
            all_metrics['train_time'] = round(training_time, 4)
        has_test_data = False
        if hasattr(y_test, '__len__') and len(y_test) > 0:
            has_test_data = True
            perf_metrics = eval.compute_errors(y_test, y_pred)
            all_metrics.update(perf_metrics)
            try:
                res_metrics = eval.compute_residual_diagnostics(y_test, y_pred)
                all_metrics.update(res_metrics)
                is_valid = res_metrics.get('ljung_box_pvalue', 0) > 0.05
                all_metrics['Statistical_Valid'] = 'YES' if is_valid else 'NO'
            except Exception as e:
                print(f"Warning: Could not compute residuals: {e}")

        update_leaderboard(
            indicator=indicator,
            model_name=model_name,
            metrics=all_metrics,
            filepath=LEADERBOARD_FILE
        )
        
        if has_test_data and len(years_test) == len(y_pred):
            base_info = {"indicator": indicator, "model": model_name}
            _report_test_predictions(base_info, years_test, y_test, y_pred)
        
        if future_predictions is not None:
            base_info = {"indicator": indicator, "model": model_name}
            _report_future_forecasts(base_info, future_predictions)
        print("Save complete.")
    except Exception as e:
        print(f"ERROR in save_experiment_results: {e}")
        traceback.print_exc()

def update_leaderboard(indicator, model_name, metrics, filepath=LEADERBOARD_FILE):
    """
    Aggiorna la tabella comparativa (Leaderboard) salvando i risultati in orizzontale.
    
    Args:
        indicator (str): Nome dell'indicatore (es. 'population_growth').
        model_name (str): Prefisso del modello (es. 'MA', 'AR', 'Baseline').
        metrics (dict): Metriche da salvare (es. {'RMSE': 0.05, 'LjungBox': 0.8}).
        filepath (str): Nome del file excel.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if os.path.exists(filepath):
        try:
            df = pd.read_excel(filepath, index_col=0) 
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if indicator not in df.index:
        df.loc[indicator] = pd.Series(dtype='object')
    
    for key, value in metrics.items():
        col_name = f"{model_name}_{key}"
        df.loc[indicator, col_name] = value

    df = df.reindex(sorted(df.columns), axis=1)

    try:
        df.to_excel(filepath)
        print(f"Leaderboard updated: {indicator} | {model_name}")
    except PermissionError:
        print(f"ERROR: close {filepath}")

def save_master_config(df_config, filepath=INTEGRATION_FILE):
    """
    Salva il dataframe di configurazione (Integration Orders) in Excel.
    Sovrascrive il file se esiste per garantire che la configurazione sia sempre
    l'ultima versione calcolata.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        df_config.to_excel(filepath)
        print(f"Configuration successfully saved to: {filepath}")
    except Exception as e:
        print(f"ERROR saving configuration: {e}")
        traceback.print_exc()