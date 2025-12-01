import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
import scipy.stats as stats
from src.config import DEFAULT_PATH, SAFE_VAR_NAME

def load_data(filepath=DEFAULT_PATH):
    """
    Carica i dati World Bank, li traspone (anni sulle righe) 
    e gestisce i valori mancanti.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File non trovato: {filepath}")
    
    df = pd.read_excel(filepath)
    if 'Indicator Name' not in df.columns:
        raise ValueError("Il DataFrame deve contenere una colonna 'Indicator Name'")
    
    year_cols = [col for col in df.columns if str(col).isdigit()]
    df_long = df.melt(id_vars='Indicator Name', value_vars=year_cols,
                    var_name='Year', value_name='Value')
    
    df_wide = df_long.pivot(index='Year', columns='Indicator Name', values='Value').reset_index()
    df_wide['Year'] = df_wide['Year'].astype(int)
    df_wide = df_wide.sort_values('Year').reset_index(drop=True)
    df_wide.columns.name = None
    
    unusable_cols = [col for col in df_wide.columns if col in UNUSABLE]
    if unusable_cols:
        df_wide = df_wide.drop(columns=unusable_cols)
    print(f"Dataset caricato: {df_wide.shape[0]} anni, {df_wide.shape[1]} variabili.")
    return df_wide

def plot_results(df, train, test, prediction, model_name, variable_name, calc_rmse):
    """
    Plotta i risultati e salva l'immagine in una cartella specifica.
    
    Parametri:
    - train: serie di training
    - test: serie di test (reale)
    - prediction: serie predetta dal modello
    - model_name: nome del modello (es. 'ARIMA', 'MLP')
    - variable_name: nome della variabile (es. 'GDP_Growth')
    """
    
    save_folder = f"results/plots/{model_name}"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Cartella creata: {save_folder}")
        
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(train['Year'], train['Value'], '-', color=COLORS['train'], label='Train')
    ax.plot(test['Year'], test['Value'], '-', color=COLORS['test'], label='Test')
    ax.plot(test['Year'], prediction, '--', color=COLORS['pred'], label=f'Predicted ({model_name})')
    ax.set_title(f'{variable_name} - {model_name} (RMSE: {calc_rmse:.2f}%)')
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    test_start = int(test['Year'].min())
    test_end = int(test['Year'].max())
    ax.axvspan(test_start - 0.5, test_end + 0.5, color='#808080', alpha=0.2)
    ax.legend(loc='best')
    plt.title(f'{model_name} forecast: {variable_name}')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    xticks = np.arange(min_year, max_year + 1, 5)
    plt.xticks(xticks, [str(y) for y in xticks])
    
    safe_var_name = SAFE_VAR_NAME.get(variable_name, variable_name[:3])
    filename = f"{model_name}_{safe_var_name}.png"
    full_path = os.path.join(save_folder, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {full_path}")
    plt.show()
    plt.close()

def analyze_residuals(y_true, y_pred, model_name, indicator_name):
    """
    Calcola e plotta l'analisi dei residui:
    1. Line plot dei residui (per vedere trend residui)
    2. Istogramma (per vedere normalità)
    3. ACF Plot (per vedere autocorrelazione non catturata)
    """
    residuals = y_true - y_pred
    
    # Creazione figura con 3 subplot
    fig = plt.figure(figsize=(12, 8))
    layout = (2, 2)
    ax1 = plt.subplot2grid(layout, (0, 0), colspan=2) # Line plot largo sopra
    ax2 = plt.subplot2grid(layout, (1, 0))            # Istogramma sotto sx
    ax3 = plt.subplot2grid(layout, (1, 1))            # ACF sotto dx
    
    # A. Residui nel tempo
    ax1.plot(residuals, color='purple', linewidth=1.5)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'Residui nel tempo: {indicator_name} ({model_name})')
    ax1.set_ylabel('Errore')
    ax1.grid(True, alpha=0.3)
    
    # B. Istogramma (Distribuzione)
    ax2.hist(residuals, bins=10, color='gray', edgecolor='black', alpha=0.7, density=True)
    ax2.set_title('Distribuzione dei Residui')
    # Aggiungi una curva normale ideale per confronto
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax2.plot(x, p, 'k', linewidth=2, label='Normale')
    ax2.legend()
    
    # C. ACF Plot (Autocorrelazione)
    # lags=min(10, len(residuals)-1) evita errori se hai pochi dati di test
    sm.graphics.tsa.plot_acf(residuals, ax=ax3, lags=min(10, len(residuals)//2 - 1), zero=False)
    ax3.set_title('Autocorrelazione dei Residui (ACF)')
    
    plt.tight_layout()
    plt.show()
    plt.close() # Chiude per non intasare la memoria
    
    return {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals)
    }

def evaluate_forecast(y_true, y_pred):
    """
    Calcola RMSE, MAE e un MAPE 'sicuro' (gestisce divisione per zero).
    y_true: Array dei valori reali
    y_pred: Array dei valori predetti
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Evitiamo divisione per zero: calcoliamo MAPE solo dove y_true != 0
    mask = y_true != 0
    if np.any(mask):
        # Calcolo standard MAPE: mean(|(true - pred) / true|) * 100
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan  # Se tutti i valori sono 0, il MAPE è impossibile
        
    # Se il MAPE è enorme (>1000%), è probabile che ci siano valori vicini allo zero.
    # In quel caso, meglio restituire NaN o un cap, per non rovinare i grafici.
    if mape > 5000: 
        mape = np.nan

    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MAPE": round(mape, 2) if not np.isnan(mape) else None
    }

def save_metrics(model_name, variable_name, metrics, filepath=RESULTS_PATH):
    """
    Salva le metriche di performance in un file Excel cumulativo.
    Se la combinazione (Indicator, Model) esiste già, la aggiorna.
    
    Parametri:
    - model_name: Nome del modello
    - variable_name: Nome della variabile
    - metrics: Dizionario con le metriche (output di evaluate_forecast)
    - filepath: Percorso del file Excel
    """
    
    new_data = {
        "Indicator": variable_name,
        "Model": model_name,
        "RMSE": metrics.get("RMSE"),
        "MAE": metrics.get("MAE"),
        "MAPE": metrics.get("MAPE")
    }
    new_row = pd.DataFrame([new_data])

    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    if os.path.exists(filepath):
        try:
            df = pd.read_excel(filepath)
            
            # 4. Logica "Smart Update":
            # Rimuovi la vecchia entry per questo specifico Modello e Variabile (se esiste)
            # così evitiamo duplicati se rilanci il codice.
            mask = (df['Indicator'] == variable_name) & (df['Model'] == model_name)
            df = df[~mask]
            
            # Aggiungi la nuova riga
            df = pd.concat([df, new_row], ignore_index=True)
            
        except PermissionError:
            print(f"ERRORE: Chiudi il file Excel '{filepath}' prima di salvare!")
            return
        except Exception as e:
            print(f"Errore durante la lettura del file risultati: {e}")
            return
    else:
        # Se il file non esiste, inizia con la nuova riga
        df = new_row

    # 5. Ordina per Variabile e poi per RMSE (così vedi subito il migliore)
    if 'RMSE' in df.columns:
        df = df.sort_values(by=['Indicator', 'RMSE'])

    try:
        df.to_excel(filepath, index=False)
        print(f"Metriche salvate per {model_name} su {variable_name}")
    except PermissionError:
        print(f"ERRORE CRITICO: Impossibile salvare. Il file '{filepath}' è aperto in Excel? Chiudilo!")