import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import scipy.stats as stats
import textwrap
import matplotlib.dates as mdates
from src.config import COLORS, SAFE_VAR_NAMES, REVERSE_VAR_NAMES, PLOTS_DIR, LINE_STYLES, PLOT_CONFIG

plt.rcParams.update(PLOT_CONFIG)

def set_path(model_name, DIR):
    save_folder = os.path.join(DIR, str(model_name))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Cartella creata: {save_folder}")
    return save_folder

def set_filename(variable_name, model_name):
    filename = f"{str(model_name)}_{variable_name}.png"
    return filename

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
    if save_plots:
        save_folder = set_path(folder_name, PLOTS_DIR)

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

def plot_forecast(train, test, variable_name, model_name, folder_name, prediction=None, baseline=None, baseline_name="Baseline", rmse=None, save_plot=True):
    """
    Plots the forecast results comparing Train, Test, Baseline (optional), and Prediction (optional).
    Handles DatetimeIndex automatically.
    
    Args:
        train (pd.DataFrame or pd.Series): training data with DatetimeIndex.
        test (pd.DataFrame or pd.Series): test data with DatetimeIndex.
        variable_name (str): short name of the variable (key in SAFE_VAR_NAMES).
        model_name (str): name of the model (e.g., 'ARIMA', 'LSTM', 'Naive').
        folder_name (str): sub-folder name within PLOTS_DIR for saving.
        prediction (pd.Series/array, optional): model predictions.
        baseline (pd.Series/array, optional): abseline predictions (e.g., Hist Mean).
        baseline_name (str): label for the baseline legend.
        rmse (float, optional): RMSE value to display in the title.
        save_plot (bool): whether to save the figure to disk.
    """
    save_folder = set_path(folder_name, PLOTS_DIR)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(train.index, train['Value'], 
            label='Train', color=COLORS.get('train', 'black'), marker='.', linewidth=2)
    
    ax.plot(test.index, test['Value'], 
            label='Test (Real)', color=COLORS.get('test', 'blue'), marker='.', linewidth=2)
    
    # Connetti ultimo valore train con primo valore test
    last_train_idx = train.index[-1]
    last_train_val = train['Value'].iloc[-1]
    first_test_idx = test.index[0]
    first_test_val = test['Value'].iloc[0]
    ax.plot([last_train_idx, first_test_idx], [last_train_val, first_test_val], 
            color=COLORS.get('test', 'blue'), linewidth=2)
    
    if baseline is not None:
        y_vals = baseline if isinstance(baseline, (pd.Series, list, np.ndarray)) else baseline
        
        ax.plot(test.index, y_vals, 
                linestyle=LINE_STYLES.get('baseline', '--'), 
                label=f'{baseline_name}', 
                color=COLORS.get('baseline', 'gray'), alpha=0.8)
        
        # Connetti ultimo valore train con prima predizione baseline
        first_baseline_val = y_vals[0] if isinstance(y_vals, (list, np.ndarray)) else y_vals.iloc[0]
        ax.plot([last_train_idx, first_test_idx], [last_train_val, first_baseline_val], 
                linestyle=LINE_STYLES.get('baseline', '--'), 
                color=COLORS.get('baseline', 'gray'), alpha=0.8)

    if prediction is not None:
        y_vals = prediction if isinstance(prediction, (pd.Series, list, np.ndarray)) else prediction
        
        ax.plot(test.index, y_vals, 
                linestyle=LINE_STYLES.get('pred', '--'), 
                label=f'Pred ({model_name})', 
                color=COLORS.get('pred', 'red'), linewidth=2)
        
        # Connetti ultimo valore train con prima predizione
        first_pred_val = y_vals[0] if isinstance(y_vals, (list, np.ndarray)) else y_vals.iloc[0]
        ax.plot([last_train_idx, first_test_idx], [last_train_val, first_pred_val], 
                linestyle=LINE_STYLES.get('pred', '--'), 
                color=COLORS.get('pred', 'red'), linewidth=2)

    ax.axvspan(train.index.max(), test.index.max(), color='#d3d3d3', alpha=0.2)

    long_name = REVERSE_VAR_NAMES.get(variable_name, variable_name)
    title_text = f"{model_name} Forecast: {long_name}"
    
    if rmse is not None:
        title_text += f"\n(RMSE: {rmse:.4f})"
    
    ax.set_title("\n".join(textwrap.wrap(title_text, width=70)), fontsize=14, fontweight='bold')
    ax.set_ylabel("Value")
    ax.set_xlabel("Year")
    
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.4)
    
    if save_plot:
        filename = set_filename(variable_name, model_name)
        full_path = os.path.join(save_folder, filename)
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {full_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
            
    plt.show()
    plt.close()

def plot_future_forecasts(history_series, future_df, variable_name, model_name, folder_name, save_plots=True):
    """
    Plotta la serie storica completa e la proiezione futura (2025-2030).
    
    Args:
        history_series (pd.Series): La serie completa dei dati storici (fino al 2024).
        future_df (pd.DataFrame): DataFrame con colonne ['year', 'pred'] e opzionalmente ['lower_ci', 'upper_ci'].
        variable_name (str): Nome dell'indicatore.
        model_name (str): Nome del modello.
    """
    if future_df is None or future_df.empty:
        print(f"Skipping future plot for {variable_name}: No future data.")
        return

    save_folder = set_path(folder_name, PLOTS_DIR)

    plt.figure(figsize=(12, 6))
    
    if hasattr(history_series.index, 'year'): x_hist = history_series.index.year
    else: x_hist = history_series.index
        
    plt.plot(x_hist, history_series.values, label='Historical Data', color=COLORS.get('train', 'black'), marker='.', linewidth=2)
    
    last_year = x_hist.max()
    last_val = history_series.values[-1]
    
    connect_x = [last_year, future_df['year'].iloc[0]]
    connect_y = [last_val, future_df['pred'].iloc[0]]
    plt.plot(connect_x, connect_y, color=COLORS.get('pred_model', 'red'), linestyle='--', linewidth=2)

    plt.plot(future_df['year'], future_df['pred'], label=f'Forecast {model_name}', 
            color=COLORS.get('pred_model', 'red'), linestyle='--', linewidth=2, markersize=4)
    
    # Intervallo di Confidenza (Se esiste)
    if 'lower_ci' in future_df.columns and 'upper_ci' in future_df.columns:
        plt.fill_between(
            future_df['year'], 
            future_df['lower_ci'], 
            future_df['upper_ci'], 
            color=COLORS.get('pred_model', 'red'), alpha=0.15, label='95% Confidence Interval'
        )
    
    plt.axvspan(last_year, 2030, color='#d3d3d3', alpha=0.2)
    
    long_name = REVERSE_VAR_NAMES.get(variable_name, variable_name)
    title_text = f"Future Forecast ({last_year}-2030): {long_name}"
    plt.title("\n".join(textwrap.wrap(title_text, width=70)), fontsize=14, fontweight='bold')
    
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.4)
    
    if save_plots:
        filename = set_filename(variable_name, f"future{model_name}")
        full_path = os.path.join(save_folder, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Future Plot saved: {full_path}")
        
    plt.show()
    plt.close()

def plot_baseline_comparison(train, test, future_years, predictions_dict, variable_name, save_plot=True):
    """
    Plotta: Storia + Test Reale + Predizioni Test (vari modelli) + Futuro (vari modelli).
    """
    # Setup
    folder_name = "00_Baselines_Comparison"
    save_folder = set_path(folder_name, PLOTS_DIR) # Assicurati che PLOTS_DIR sia importato
    
    plt.figure(figsize=(14, 7))
    
    # 1. Dati Reali (Train e Test)
    plt.plot(train.index.year, train['Value'], label='Train Data', color='black', linewidth=2)
    plt.plot(test.index.year, test['Value'], label='Test Data (Ground Truth)', color='gray', linewidth=2, alpha=0.7)
    
    # Colori per i modelli
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blu, Arancio, Verde, Rosso
    
    # 2. Loop sui modelli (Dizionario: { 'NomeModello': (pred_test, pred_future) })
    for i, (model_name, (pred_test, pred_future)) in enumerate(predictions_dict.items()):
        color = colors[i % len(colors)]
        
        # A. Plot Test Prediction (Tratteggiato)
        # Allineamento asse X
        plt.plot(test.index.year, pred_test.values, 
                label=f'{model_name} (Test)', 
                color=color, linestyle='--', linewidth=1.5)
        
        # B. Plot Future Prediction (Punteggiato)
        if pred_future is not None:
            # Linea di connessione (Ultimo punto Test -> Primo Futuro) per continuità
            connect_x = [test.index.year[-1], future_years[0]]
            connect_y = [pred_test.values[-1], pred_future.values[0]]
            plt.plot(connect_x, connect_y, color=color, linestyle=':', linewidth=1)
            
            # Plot vero e proprio
            plt.plot(future_years, pred_future.values, 
                    # label=f'{model_name} (2030)', # Non mettiamo label doppia per pulizia
                    color=color, linestyle=':', linewidth=2, marker='.', markersize=4)

    # 3. Formattazione
    plt.axvline(x=test.index.year[0], color='gray', linestyle='-', alpha=0.3)
    plt.axvline(x=future_years[0], color='black', linestyle='-', alpha=0.5, label='Future Start')
    
    plt.title(f"Baseline Models Comparison: {variable_name}", fontsize=14, fontweight='bold')
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Legenda fuori dal grafico
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    if save_plot:
        filename = f"COMPARE_{variable_name}.png"
        plt.savefig(os.path.join(save_folder, filename), dpi=300)
        
    plt.show()
    plt.close()


def plot_augmented(variable_name, df_step, df_jitter, x_train_vals, y_train_vals):
    """ plots original time series 
    + step function augmented time series (blue) 
    + linear interpolation with jitter augmented time series (orange)

    Args:
        df_step (pd.DataFrame): DataFrame ('Year', 'Value') augmented through step function
        df_jitter (pd.DataFrame): DataFrame ('Year', 'Value') augmented through linear interpolation with jitter
        x_train_vals (_type_): _description_
        y_train_vals (_type_): _description_
    """
    save_folder = set_path("AUGMENTATION", PLOTS_DIR)
    plt.figure(figsize=(12, 6))
    plt.plot(x_train_vals, y_train_vals, '.', label='Original', color='black', markersize=8)
    plt.plot(df_step['Year'], df_step['Value'], '-', label='Step Function', alpha=0.7)
    plt.plot(df_jitter['Year'], df_jitter['Value'], '--', label='Linear + Jittering', alpha=0.7)
    plt.title(f"Data Augmentation - {variable_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    safe_var_name = SAFE_VAR_NAMES.get(variable_name, variable_name[:3])
    filename = f"AUG_{safe_var_name}.png"
    full_path = os.path.join(save_folder, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {full_path}")
    plt.show()
    plt.close()

def plot_residuals(residuals, model_name, variable_name):
    """
    Prende un array di residui e genera la dashboard diagnostica (Line, Hist, ACF).
    Salva automaticamente usando la logica interna.
    """
    save_folder = set_path("RESIDUALS", PLOTS_DIR)
    fig = plt.figure(figsize=(10, 8))
    layout = (2, 2)
    ax1 = plt.subplot2grid(layout, (0, 0), colspan=2)
    ax2 = plt.subplot2grid(layout, (1, 0))
    ax3 = plt.subplot2grid(layout, (1, 1))
    
    # Residui nel tempo
    ax1.plot(residuals, color='purple', linewidth=1.5)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'Residuals: {variable_name} ({model_name})')
    ax1.grid(True, alpha=0.3)
    
    # Istogramma
    ax2.hist(residuals, bins=15, color='gray', edgecolor='black', alpha=0.7, density=True)
    ax2.set_title('Distribution')
    
    # Curva normale teorica
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax2.plot(x, p, 'k', linewidth=2, label='Normal')
    ax2.legend()
    
    # ACF Plot
    # Gestione errori se i residui sono troppo pochi per l'ACF
    if len(residuals) > 2:
        lags = min(10, len(residuals)//2 - 1)
        sm.graphics.tsa.plot_acf(residuals, ax=ax3, lags=lags, zero=False)
        ax3.set_title('Autocorrelation (ACF)')
    else:
        ax3.text(0.5, 0.5, "Insufficient data for ACF", ha='center')
        
    filename = set_filename(variable_name, model_name)
    full_path = os.path.join(save_folder, filename)
    try:
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato in: {full_path}")
    except Exception as e:
        print(f"errore salvataggio: {e}")
    plt.tight_layout()
    plt.close()

def plot_shallow_nn_preds(variable_name, results_dict, orig_aug_subsets, model_name):
    """
    Plotta le predizioni della Shallow CNN confrontando le diverse strategie di augmentation.
    Include anche il grafico della Loss per diagnosticare il training.
    """
    save_folder = set_path(model_name, PLOTS_DIR)
    
    full_years = orig_aug_subsets[variable_name]['full_orig']['Year'].values
    full_values = orig_aug_subsets[variable_name]['full_orig']['Value'].values
    
    test_years = orig_aug_subsets[variable_name]['orig_test']['Year'].values[3:]
    # test_real = orig_aug_subsets['orig_test']['Value'].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    ax1.plot(full_years, full_values, color=COLORS['train'], linestyle=LINE_STYLES['real'], label='Real', marker='.')

    styles = {
        'orig':   {'color': COLORS['pred_orig'],  'ls': LINE_STYLES['real'],  'label': 'Original'},
        'step':   {'color': COLORS['pred_step'], 'ls': LINE_STYLES['aug'], 'label': 'Step Aug'},
        'jitter': {'color': COLORS['pred_jitter'],   'ls': LINE_STYLES['aug'], 'label': 'Jitter Aug'}
    }

    if variable_name in results_dict:
        indicator_res = results_dict[variable_name]
        
        for aug_type, res_data in indicator_res.items():
            if aug_type not in styles: continue
            
            y_pred = res_data['pred_real'].flatten() # Appiattiamo (N,1) -> (N,)
            loss_history = res_data['history']

            ax1.plot(test_years, y_pred, 
                    color=styles[aug_type]['color'], 
                    linestyle=styles[aug_type]['ls'], 
                    linewidth=2, 
                    label=styles[aug_type]['label'])
            
            ax2.plot(loss_history, 
                    color=styles[aug_type]['color'], 
                    linestyle=styles[aug_type]['ls'], 
                    label=f"{styles[aug_type]['label']} (Epoche: {len(loss_history)})")

    ax1.set_title(f"Forecast Comparison with {model_name}", fontsize=14)
    ax1.set_ylabel("Value")
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    test_start = int(test_years.min())
    test_end = int(test_years.max())
    ax1.axvspan(test_start - 0.5, test_end + 0.5, color='#808080', alpha=0.2)
    
    ax2.set_title("Training Loss Convergence", fontsize=14)
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss (MSE)")
    ax2.set_yscale('log') # Scala logaritmica spesso aiuta a vedere meglio la convergenza
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f'{model_name} Analysis: {variable_name}', fontsize=16, y=0.93, fontweight='bold')

    filename = set_filename(variable_name, model_name)
    full_path = os.path.join(save_folder, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {full_path}")
    plt.show()
    plt.close()

def plot_nn_preds(variable_name, predictions_dict, train_df, val_df, test_df, model_name):
    save_folder = os.path.join(PLOTS_DIR, str(model_name))

    full_df = np.concatenate([
        train_df.values,
        val_df.values,
        test_df.values
    ])
    
    if variable_name not in predictions_dict:
        print(f"Skipping plot for {variable_name}: No predictions found.")
        return
    data = predictions_dict[variable_name]

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    styles = {
        'orig':       {'color': COLORS['pred_orig'], 'ls': LINE_STYLES['baseline'], 'label': 'Pred - Orig'},
        'step_aug':   {'color': COLORS['pred_step'], 'ls': LINE_STYLES['aug'], 'label': 'Pred - Step Aug'},
        'jitter_aug': {'color': COLORS['pred_jitter'], 'ls': LINE_STYLES['aug'],  'label': 'Pred - Jitter Aug'}
    }
    
    years_full = full_df[:, 0]
    values_full = full_df[:, 1]
    
    for i, sampler in enumerate(['Random', 'TPE']):
        ax = axes[i]
        sampler_data = data.get(sampler, {})
        
        ax.plot(years_full, values_full, '-', marker='.', color=COLORS['train'], label='Train')
        
        if sampler_data:
            for ds_type, res in sampler_data.items():
                years_pred = res.get('years', [])
                y_pred = res.get('y_pred', [])

                rmse = None
                metrics = res.get('metrics')
                if metrics and isinstance(metrics, dict) and 'RMSE' in metrics:
                    rmse = metrics['RMSE']

                style = styles.get(ds_type, {})
                label_base = style.get('label', ds_type)
                label_txt = f"{label_base} (RMSE: {rmse:.2f})" if rmse is not None else label_base

                step = max(1, len(full_df) // 10)
                xticks = years_full[::step]
                ax.set_xticks(xticks)
                ax.set_xticklabels([int(x) for x in xticks])
                ax.tick_params(axis='x', which='both', labelbottom=True)
                test_start = int(years_pred.min())
                test_end = int(years_pred.max())
                ax.axvspan(test_start, test_end, color='#808080', alpha=0.1)
                ax.plot(years_pred,
                        y_pred,
                        color=style.get('color', 'C0'),
                        linestyle=style.get('ls', '-'),
                        linewidth=2,
                        label=label_txt)
        
        ax.set_title(f"Hyperparameter Tuning Method: {sampler}", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'{variable_name} - {model_name} predictions', fontsize=16, fontweight='bold', y=0.93)
        
    safe_var_name = SAFE_VAR_NAMES.get(variable_name, variable_name[:3])
    filename = f"{model_name}_{safe_var_name}.png"
    
    if not os.path.isabs(save_folder):
        save_folder = os.path.abspath(save_folder)
        
    if not os.path.exists(save_folder):
        try:
            os.makedirs(save_folder)
            print(f"DEBUG: Cartella creata: {save_folder}")
        except Exception as e:
            print(f"ERRORE: Impossibile creare la cartella {save_folder}. Motivo: {e}")
            return

    full_path = os.path.join(save_folder, filename)
    try:
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Grafico salvato in: {full_path}")
    except Exception as e:
        print(f"errore salvataggio: {e}")
    plt.show()
    plt.close()