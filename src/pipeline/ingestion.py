import pandas as pd
import os
from src.config import RAW_DATA_FILE, UNUSABLE

def load_data(filepath=RAW_DATA_FILE):
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