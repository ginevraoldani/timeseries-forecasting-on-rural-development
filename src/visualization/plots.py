import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import scipy.stats as stats
from src.config import COLORS, SAFE_VAR_NAME, PLOTS_DIR, LINE_STYLES, PLOT_CONFIG

plt.rcParams.update(PLOT_CONFIG)

def plot_results(df, train, test, baseline, prediction, baseline_model, model_name, variable_name, calc_rmse):
    """
    Plotta i risultati e salva l'immagine in una cartella specifica.
    
    Parametri:
    - train: serie di training
    - test: serie di test (reale)
    - prediction: serie predetta dal modello
    - model_name: nome del modello (es. 'ARIMA', 'MLP')
    - variable_name: nome della variabile (es. 'GDP_Growth')
    """
    save_folder = os.path.join(PLOTS_DIR, str(model_name))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Cartella creata: {save_folder}")
        
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(train['Year'], train['Value'], '-', color=COLORS['train'], label='Train')
    ax.plot(test['Year'], test['Value'], '-', color=COLORS['test'], label='Test')
    ax.plot(test['Year'], baseline, '--', color=COLORS['pred_baseline1'], label=f'Baseline ({baseline_model})')
    ax.plot(test['Year'], prediction, '--', color=COLORS['pred_model'], label=f'Predicted ({model_name})')
    ax.set_title(f'{variable_name} - {model_name} (RMSE: {calc_rmse:.2f}%)', y=0.93)
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
    save_folder = os.path.join(PLOTS_DIR, "augmentation")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print(f"Cartella creata: {save_folder}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_train_vals, y_train_vals, '.', label='Original', color='black', markersize=8)
    plt.plot(df_step['Year'], df_step['Value'], '-', label='Step Function', alpha=0.7)
    plt.plot(df_jitter['Year'], df_jitter['Value'], '--', label='Linear + Jittering', alpha=0.7)
    plt.title(f"Data Augmentation - {variable_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    safe_var_name = SAFE_VAR_NAME.get(variable_name, variable_name[:3])
    filename = f"augment_{safe_var_name}.png"
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
    save_folder = os.path.join(PLOTS_DIR, "residuals")
    fig = plt.figure(figsize=(10, 8))
    layout = (2, 2)
    ax1 = plt.subplot2grid(layout, (0, 0), colspan=2)
    ax2 = plt.subplot2grid(layout, (1, 0))
    ax3 = plt.subplot2grid(layout, (1, 1))
    
    # Residui nel tempo
    ax1.plot(residuals, color='purple', linewidth=1.5)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'Residui: {variable_name} ({model_name})')
    ax1.grid(True, alpha=0.3)
    
    # Istogramma
    ax2.hist(residuals, bins=15, color='gray', edgecolor='black', alpha=0.7, density=True)
    ax2.set_title('Distribuzione')
    
    # Curva normale teorica
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax2.plot(x, p, 'k', linewidth=2, label='Normale')
    ax2.legend()
    
    # ACF Plot
    # Gestione errori se i residui sono troppo pochi per l'ACF
    if len(residuals) > 2:
        lags = min(10, len(residuals)//2 - 1)
        sm.graphics.tsa.plot_acf(residuals, ax=ax3, lags=lags, zero=False)
        ax3.set_title('Autocorrelazione (ACF)')
    else:
        ax3.text(0.5, 0.5, "Dati insufficienti per ACF", ha='center')
        
    safe_var_name = SAFE_VAR_NAME.get(variable_name, variable_name[:3])
    filename = f"{str(model_name).lower()}_{safe_var_name}.png"
    
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
    plt.tight_layout()
    plt.close()
    
def plot_nn_predictions(variable_name, predictions_dict, train_df, val_df, test_df, model_name):
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
                ax.plot(years_pred, y_pred, color=style.get('color', 'C0'),
                        linestyle=style.get('ls', '-'), linewidth=2, label=label_txt)
        
        ax.set_title(f"Hyperparameter Tuning Method: {sampler}", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        fig.suptitle(f'{variable_name} - {model_name} predictions', fontsize=16, fontweight='bold', y=0.93)
        
    safe_var_name = SAFE_VAR_NAME.get(variable_name, variable_name[:3])
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