import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

DEFAULT_PATH = "C:/Users/oldan/Desktop/RuralDevelopment/progetto-tirocinio/data/processed_italy_data.xlsx"
RESULTS_PATH = "C:/Users/oldan/Desktop/RuralDevelopment/progetto-tirocinio/results/models_performance.xlsx"

UNUSABLE = [
    "Rural population living in areas where elevation is below 5 meters (% of total population)",
    "Access to electricity, rural (% of rural population)",
    "Surface area (sq. km)",
    "Rural land area (sq. km)",
    "Land area (sq. km)",
    "Average precipitation in depth (mm per year)",
    "Agricultural irrigated land (% of total agricultural land)",
    "Rural land area where elevation is below 5 meters (% of total land area)",
    "Rural land area where elevation is below 5 meters (sq. km)"
]

SAFE_VAR_NAME = {
    "Rural population (% of total population)" : "population_percent",
    "Rural population growth (annual %)" : "population_growth",
    "Rural population" : "population_abs",
    "Employment in agriculture (% of total employment)" : "employment_tot",
    "Employment in agriculture, male (% of male employment)" : "employment_male",
    "Employment in agriculture, female (% of female employment)" : "employment_female",
    "Forest area (% of land area)" : "forestarea_percent",
    "Forest area (sq. km)" : "forestarea_abs",
    "Agricultural land (% of land area)" : "agriland_percent",
    "Agricultural land (sq. km)" : "agriland_abs",
    "Arable land (% of land area)" : "arableland_percent",
    "Arable land (hectares per person)" : "arableland_person",
    "Arable land (hectares)" : "arableland_abs",
    "Land under cereal production (hectares)" : "cerealland_abs",
    "Permanent cropland (% of land area)" : "cropland_percent",
    "Annual freshwater withdrawals, agriculture (% of total freshwater withdrawal)" : "withdrawals_percent",
    "Fertilizer consumption (kilograms per hectare of arable land)" : "fertilizer_abs",
    "Fertilizer consumption (% of fertilizer production)" : "fertilizer_percent",
    "Livestock production index (2014-2016 = 100)" : "livestock_production_index",
    "Food production index (2014-2016 = 100)" : "food_production_index",
    "Crop production index (2014-2016 = 100)" : "crop_production_index",
    "Cereal production (metric tons)" : "cereal_production",
    "Cereal yield (kg per hectare)" : "cerealyield_abs",
    "Agriculture, forestry, and fishing, value added (% of GDP)" : "valueadded_percent",
    "Agriculture, forestry, and fishing, value added (current US$)" : "valueadded_dollars",
    "Agricultural raw materials exports (% of merchandise exports)" : "exports_percent",
    "Agricultural raw materials imports (% of merchandise imports)" : "imports_percent"    
}

COLORS = {
    'train': 'black',
    'test': '#1f77b4',
    'pred': '#d62728',
    'pred_multi': ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
}

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