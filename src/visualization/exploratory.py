import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import scipy.stats as stats
import textwrap
import matplotlib.dates as mdates
from src.config import set_path, set_filename
from src.config import COLORS, SAFE_VAR_NAMES, REVERSE_VAR_NAMES, PLOTS_DIR, LINE_STYLES, PLOT_CONFIG

plt.rcParams.update(PLOT_CONFIG)

def plot_exploratory_time_series(df, folder_name, x_col='Year', columns=None, save_plots=False):
    """
    Plots time series data for exploratory analysis using standardized project styles.
    This function adheres to the configurations defined in config.py (COLORS, LINE_STYLES)
    to ensure visual consistency across the project.

    Args:
        df (pd.DataFrame): input DataFrame.
        folder_name (str): sub-folder name within PLOTS_DIR for saving.
        x_col (str): column name for X-axis. Defaults to 'Year'. If missing, index is used.
        columns (list, optional): list of columns to plot. If None, selects all numeric cols.
        save_plots (bool): whether to save the plots to disk.
    """
    if x_col in df.columns:
        use_index = False
        x_label = x_col
    else:
        use_index = True
        x_label = "Year"

    if columns is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        columns = [c for c in numeric_cols if c != x_col]

    save_folder = None
    if save_plots: save_folder = set_path(folder_name, PLOTS_DIR)

    print(f"Plotting {len(columns)} variables...")
    for col in columns:
        series = df[col].dropna()
        if series.empty:
            print(f"Skipping empty: {col}")
            continue
        if use_index: x_plot = series.index
        else: x_plot = df.loc[series.index, x_col]
        
        plt.figure(figsize=(10, 6))
        plt.plot(
            x_plot, 
            series, 
            color=COLORS.get('train', 'blue'),
            linestyle=LINE_STYLES.get('real', '-'),
            marker='o', 
            markersize=4,
            label=col
        )
        
        long_name = REVERSE_VAR_NAMES.get(col, col)
        title_text = "\n".join(textwrap.wrap(f"EDA: {long_name}", width=60))
        
        plt.title(title_text, fontsize=14, fontweight='bold')
        plt.xlabel(x_label)
        plt.ylabel("Value")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc='best')

        if save_plots and save_folder:
            filename = set_filename(col, "EDA")
            full_path = os.path.join(save_folder, filename)
            try:
                plt.savefig(full_path, dpi=300, bbox_inches='tight')
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"Error saving {col}: {e}")
        
        plt.show() 
        plt.close()

def plot_sanity_check(df, folder_name="00_DIFFCHECK", save_plots=True):
    save_folder = None
    if save_plots: save_folder = set_path(folder_name, PLOTS_DIR)
    
    for col in df.columns:
        if df[col].dropna().empty: continue
        series = df[col].dropna()
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        fig.suptitle(f"Sanity Check (Integration Order): {col}", fontsize=16, fontweight='bold')
        
        # --- RIGA 1: D=0 (Originale) ---
        axes[0, 0].plot(series, color='blue')
        axes[0, 0].set_title(f"Original Series (d=0)")
        axes[0, 0].grid(True, alpha=0.3)
        # Lags limitati alla lunghezza serie per evitare errori su serie corte
        lags = min(20, len(series)//2 - 1) 
        plot_acf(series, ax=axes[0, 1], lags=lags, title="ACF Original (d=0)")
        
        # --- RIGA 2: D=1 (Differenziata) ---
        diff_series = series.diff().dropna()
        axes[1, 0].plot(diff_series, color='green')
        axes[1, 0].set_title(f"1st Difference (d=1)")
        axes[1, 0].grid(True, alpha=0.3)
        
        if len(diff_series) > 2:
            lags_d1 = min(20, len(diff_series)//2 - 1)
            plot_acf(diff_series, ax=axes[1, 1], lags=lags_d1, title="ACF d=1")
        
        # --- RIGA 3: D=2 (Differenziata due volte) ---
        # Prendiamo la differenza della differenza
        diff2_series = diff_series.diff().dropna()
        axes[2, 0].plot(diff2_series, color='red') # Rosso per evidenziare il rischio
        axes[2, 0].set_title(f"2nd Difference (d=2)")
        axes[2, 0].grid(True, alpha=0.3)
        
        if len(diff2_series) > 2:
            lags_d2 = min(20, len(diff2_series)//2 - 1)
            plot_acf(diff2_series, ax=axes[2, 1], lags=lags_d2, title="ACF d=2")
        
        plt.tight_layout()
        
        filename = f"DIFFCHECK_{col}.png"
        save_path = os.path.join(save_folder, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"Check plot saved: {save_path}")