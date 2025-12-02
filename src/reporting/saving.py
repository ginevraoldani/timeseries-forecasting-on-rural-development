import pandas as pd
import os
import json
from src.config import (
    PERFORMANCE_FILE, 
    PARAMS_FILE, 
    RESIDUALS_FILE, 
    PREDICTIONS_FILE
)

def _save_to_excel(filepath, new_data_dict, keys_to_match):
    """
    Salva su Excel evitando duplicati. 
    keys_to_match: lista di colonne che identificano univocamente l'esperimento 
                (es. ['Indicator', 'Model', 'Configuration'])
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    new_df = pd.DataFrame([new_data_dict])
    
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_excel(filepath)
            
            if not existing_df.empty:
                # Creiamo una maschera per trovare le righe che matchano le chiavi
                condition = pd.Series([True] * len(existing_df))
                
                for key in keys_to_match:
                    if key in new_df.columns and key in existing_df.columns:
                        # Filtra dove le colonne sono uguali
                        new_val = new_data_dict[key]
                        condition = condition & (existing_df[key] == new_val)
                
                # Rimuoviamo le righe vecchie che matchano (così inseriamo quella nuova pulita)
                existing_df = existing_df[~condition]
            
            # Concatena (Vecchio pulito + Nuovo)
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            
        except Exception as e:
            print(f"Errore lettura Excel ({e}). Creo nuovo file.")
            final_df = new_df
    else:
        final_df = new_df
        
    sort_cols = [k for k in keys_to_match if k in final_df.columns]
    if sort_cols:
        final_df = final_df.sort_values(by=sort_cols)

    final_df.to_excel(filepath, index=False)

def log_experiment_results(indicator, model_type, config_name, metrics, best_params=None, residuals_stats=None):
    """
    Salva metriche, parametri e statistiche residui nei rispettivi file Excel.
    """
    
    # FILE ERRORI: Performance (RMSE, MAE...) -> results/errors/model_performances.xlsx
    perf_data = {
        "Indicator": indicator,
        "Model": model_type,
        "Configuration": config_name,
        "RMSE": metrics.get("RMSE"),
        "MAE": metrics.get("MAE"),
        "MAPE": metrics.get("MAPE")
    }
    _save_to_excel(PERFORMANCE_FILE, perf_data, keys_to_match=["Indicator", "Model", "Configuration"])
    
    # FILE ERRORI: Residui Stats -> results/errors/residuals.xlsx
    if residuals_stats:
        res_data = {
            "Indicator": indicator,
            "Model": f"{model_type}_{config_name}",
            "Mean_Residual": residuals_stats.get('mean'),
            "Std_Residual": residuals_stats.get('std')
            # Aggiungi qui Ljung-Box p-value se lo hai calcolato
        }
        _save_to_excel(RESIDUALS_FILE, res_data, keys_to_match=["Indicator", "Model", "Configuration"])
    
    # FILE LOGS: Parametri Ottimizzati -> results/logs/optimized_params.xlsx
    if best_params:
        # Convertiamo in stringa se è un dizionario (per MLP), altrimenti lasciamo così
        params_val = json.dumps(best_params) if isinstance(best_params, dict) else str(best_params)
        
        param_data = {
            "Indicator": indicator,
            "Model": model_type,
            "Config": config_name,
            "Best_Params": params_val
        }
        _save_to_excel(PARAMS_FILE, param_data, keys_to_match=["Indicator", "Model", "Configuration"])
    print(f"Log salvati per {indicator} (Perf, Params, Res)")
    
def save_future_forecasts(indicator, model_type, config_name, years, y_true, y_pred):
    """
    Salva la sequenza temporale delle predizioni.
    Va in -> results/logs/predictions.csv
    """
    df = pd.DataFrame({
        'Indicator': indicator,
        'Model': model_type,
        'Config': config_name,
        'Year': years,
        'y_true': y_true,
        'y_pred': y_pred
    })
    
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    
    # Append mode su CSV
    file_exists = os.path.exists(PREDICTIONS_FILE)
    df.to_csv(PREDICTIONS_FILE, mode='a', header=not file_exists, index=False)
    print("Predizioni accodate al CSV.")