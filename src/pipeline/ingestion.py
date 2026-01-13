import pandas as pd
import os
from src.config import RAW_DATA_FILE, UNUSABLE, SAFE_VAR_NAMES

def load_data(filepath=RAW_DATA_FILE):
    """
    Loads the Italy-specific World Bank dataset, reshapes it to a time series format,
    applies variable renaming, and sets a proper DatetimeIndex.

    Args:
        filepath (str): path to the processed Excel file (Italy data).

    Returns:
        pd.DataFrame: a cleaned dataframe with DatetimeIndex (Annual) and renamed indicators.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_excel(filepath)
    
    if 'Indicator Name' not in df.columns:
        raise ValueError("The DataFrame must contain an 'Indicator Name' column.")
    
    # Identify Year Columns
    year_cols = [col for col in df.columns if str(col).isdigit()]
    if not year_cols:
        raise ValueError("No year columns found in the dataset.")

    # Reshape: Melt (Wide to Long)
    df_long = df.melt(
        id_vars='Indicator Name', 
        value_vars=year_cols,
        var_name='Year', 
        value_name='Value'
    )
    
    # Reshape: Pivot (Long to Wide Time Series)
    # Index: Year, Columns: Indicator Name
    df_wide = df_long.pivot(index='Year', columns='Indicator Name', values='Value')
    
    # Drop Unusable Columns (Using UNUSABLE from config)
    cols_to_drop = [col for col in df_wide.columns if col in UNUSABLE]
    if cols_to_drop:
        df_wide = df_wide.drop(columns=cols_to_drop)
        print(f"Dropped {len(cols_to_drop)} unusable indicators.")

    # Rename Columns (Using SAFE_VAR_NAMES from config)
    if SAFE_VAR_NAMES:
        df_wide = df_wide.rename(columns=SAFE_VAR_NAMES)
    
    # Setup DatetimeIndex -> convertiamo l'anno (int) in datetime (1° Gennaio dell'anno)
    df_wide.index = pd.to_datetime(df_wide.index, format='%Y')
    
    # Impostiamo esplicitamente la frequenza 'YS' (Year Start).
    df_wide = df_wide.asfreq('YS')
    
    # Ordiniamo l'indice per sicurezza
    df_wide = df_wide.sort_index()
    
    # Rimuoviamo il nome dell'indice 'Year' e delle colonne 'Indicator Name' per pulizia
    df_wide.index.name = None
    df_wide.columns.name = None

    print(f"Dataset loaded: {df_wide.shape[0]} years (from {df_wide.index.min().year} to {df_wide.index.max().year}), {df_wide.shape[1]} variables.")
    
    return df_wide