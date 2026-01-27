import pandas as pd
import os
import traceback
import src.evaluation as eval
from src.config import LEADERBOARD_FILE, INTEGRATION_FILE

def update_leaderboard(indicator, model_name, metrics, filepath=LEADERBOARD_FILE):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        try:
            df = pd.read_excel(filepath, index_col=0)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    
    if df.empty:
        df = pd.DataFrame(index=[indicator])
        df.index.name = 'Indicator'
    elif indicator not in df.index:
        df.loc[indicator] = pd.Series(dtype='object')

    for key, value in metrics.items():
        col_name = f"{model_name}_{key}"
        if isinstance(value, (list, tuple)):
            df.loc[indicator, col_name] = str(value)
        else:
            df.loc[indicator, col_name] = value
    try:
        df.to_excel(filepath)
        print(f"Leaderboard updated: {indicator} | {model_name}")
    except PermissionError:
        print(f"ERROR: close {filepath}.")

def save_experiment_results(indicator, model_name, configuration, y_test, y_pred, years_test, y_train=None, params=None, training_time=None):
    print(f"Saving results for {model_name} | {indicator}...")
    try:
        all_metrics = {}
        
        if isinstance(configuration, dict):
            all_metrics.update(configuration)
        else:
            all_metrics['config'] = str(configuration)
            
        if params: all_metrics.update(params)
        if training_time: all_metrics['train_time'] = round(training_time, 4)
        
        has_test_data = hasattr(y_test, '__len__') and len(y_test) > 0
        
        if has_test_data:
            perf_metrics = eval.compute_errors(y_test, y_pred)
            all_metrics.update(perf_metrics)
            try:
                res_metrics = eval.compute_residual_diagnostics(y_test, y_pred)
                all_metrics.update(res_metrics)
                is_valid = res_metrics.get('ljung_box_pvalue', 0) > 0.05
                all_metrics['valid'] = 'YES' if is_valid else 'NO'
            except Exception as e:
                print(f"Warning: Residual calc failed: {e}")

        update_leaderboard(indicator, model_name, all_metrics)
        print("Save leaderboard complete.")
    except Exception as e:
        print(f"ERROR inside save_experiment_results for leaderboard: {e}")
        traceback.print_exc()

def save_master_config(df_config, filepath=INTEGRATION_FILE):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        df_config.to_excel(filepath)
        print(f"Configuration successfully saved to: {filepath}")
    except Exception as e:
        print(f"ERROR saving configuration: {e}")
        traceback.print_exc()