import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import scipy.stats as stats
from src.config import COLORS, SAFE_VAR_NAME, PLOTS_DIR, COLORS, LINE_STYLES, PLOT_CONFIG

# Applica lo stile globale
plt.rcParams.update(PLOT_CONFIG)

def plot_augmented(df_step, df_jitter, x_train_vals, y_train_vals):
    """ plots original time series 
    + step function augmented time series (blue) 
    + linear interpolation with jitter augmented time series (orange)

    Args:
        df_step (pd.DataFrame): DataFrame ('Year', 'Value') augmented through step function
        df_jitter (pd.DataFrame): DataFrame ('Year', 'Value') augmented through linear interpolation with jitter
        x_train_vals (_type_): _description_
        y_train_vals (_type_): _description_
    """
    plt.figure(figsize=(12, 6))
    plt.plot(x_train_vals, y_train_vals, '.', label='Original', color='black', markersize=8)
    plt.plot(df_step['Year'], df_step['Value'], '-', label='Step Function', alpha=0.7)
    plt.plot(df_jitter['Year'], df_jitter['Value'], '--', label='Linear + Jittering', alpha=0.7)
    plt.title("Comparison between Data Augmentation Techniques")
    plt.legend()
    plt.grid(True)
    plt.show()

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

def plot_residuals(residuals, model_name, indicator_name, save_folder="residuals"):
    """
    Prende un array di residui e genera la dashboard diagnostica (Line, Hist, ACF).
    Salva automaticamente usando la logica interna.
    """

    fig = plt.figure(figsize=(10, 8))
    layout = (2, 2)
    ax1 = plt.subplot2grid(layout, (0, 0), colspan=2)
    ax2 = plt.subplot2grid(layout, (1, 0))
    ax3 = plt.subplot2grid(layout, (1, 1))
    
    # A. Residui nel tempo
    ax1.plot(residuals, color='purple', linewidth=1.5)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_title(f'Residui: {indicator_name} ({model_name})')
    ax1.grid(True, alpha=0.3)
    
    # B. Istogramma
    ax2.hist(residuals, bins=15, color='gray', edgecolor='black', alpha=0.7, density=True)
    ax2.set_title('Distribuzione')
    
    # Curva normale teorica
    xmin, xmax = ax2.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
    ax2.plot(x, p, 'k', linewidth=2, label='Normale')
    ax2.legend()
    
    # C. ACF Plot
    # Gestione errori se i residui sono troppo pochi per l'ACF
    if len(residuals) > 2:
        lags = min(10, len(residuals)//2 - 1)
        sm.graphics.tsa.plot_acf(residuals, ax=ax3, lags=lags, zero=False)
        ax3.set_title('Autocorrelazione (ACF)')
    else:
        ax3.text(0.5, 0.5, "Dati insufficienti per ACF", ha='center')
    
    plt.tight_layout()
    plt.show()