import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
import textwrap
import seaborn as sns
from src.config import set_path, set_filename
from src.config import COLORS, SAFE_VAR_NAMES, REVERSE_VAR_NAMES, PLOTS_DIR, LINE_STYLES
matplotlib.use('Agg')

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
            label='Train', color=COLORS.get('train', 'grey'), marker='.', linewidth=2)
    
    ax.plot(test.index, test['Value'], 
            label='Test (Real)', color=COLORS.get('test_real', 'black'), marker='.', linewidth=2)
    
    last_train_idx = train.index[-1]
    last_train_val = train['Value'].iloc[-1]
    first_test_idx = test.index[0]
    first_test_val = test['Value'].iloc[0]
    ax.plot([last_train_idx, first_test_idx], [last_train_val, first_test_val], 
            color=COLORS.get('test_real', 'black'), linewidth=2)
    
    if baseline is not None:
        y_vals = baseline if isinstance(baseline, (pd.Series, list, np.ndarray)) else baseline
        ax.plot(test.index, y_vals, 
                linestyle=LINE_STYLES.get('baseline', '--'), 
                label=f'Baseline ({baseline_name})', 
                color=COLORS.get('baseline', 'gray'))
        
        first_baseline_val = y_vals[0] if isinstance(y_vals, (list, np.ndarray)) else y_vals.iloc[0]
        ax.plot([last_train_idx, first_test_idx], [last_train_val, first_baseline_val], 
                linestyle=LINE_STYLES.get('baseline', '--'), 
                color=COLORS.get('baseline', 'gray'))

    if prediction is not None:
        y_vals = prediction if isinstance(prediction, (pd.Series, list, np.ndarray)) else prediction
        ax.plot(test.index, y_vals, 
                linestyle=LINE_STYLES.get('pred', '--'), 
                label=f'Pred ({model_name})', 
                color=COLORS.get('pred', 'red'), linewidth=2)
        
        first_pred_val = y_vals[0] if isinstance(y_vals, (list, np.ndarray)) else y_vals.iloc[0]
        ax.plot([last_train_idx, first_test_idx], [last_train_val, first_pred_val], 
                linestyle=LINE_STYLES.get('pred', '--'), 
                color=COLORS.get('pred', 'red'), linewidth=2)

    ax.axvspan(train.index.max(), test.index.max(), color='#C3C3C3', alpha=0.3)

    long_name = REVERSE_VAR_NAMES.get(variable_name, variable_name)
    title_text = f"{model_name} Forecast: {long_name}"
    
    ax.set_title("\n".join(textwrap.wrap(title_text, width=70)), fontsize=14, fontweight='bold')
    ax.set_ylabel("Value")
    ax.set_xlabel("Year")
    
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if save_plot:
        filename = set_filename(variable_name, model_name)
        full_path = os.path.join(save_folder, filename)
        try:
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {full_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    plt.close()

# ========================================================================================================

def plot_future_forecasts(full_history, future_pred, baseline_pred, variable_name, model_name, baseline_name, folder_name, save_plots=True):
    """
    Plotta la serie storica completa e la proiezione futura (2025-2030).
    
    Args:
        full_history (pd.Series): complete serie of historical data.
        future_pred (pd.DataFrame): DataFrame containing ['year', 'pred'] columns and optionally ['lower_ci', 'upper_ci'].
        baseline_pred (pd.DataFrame):
        variable_name (str): indicator name.
        model_name (str): name of the model of the future predictions.
        baseline_name (str):
        folder_name (str): name of the folder where plots will be saved.
    """
    if future_pred is None or future_pred.empty:
        print(f"Skipping future plot for {variable_name}: No future data.")
        return
    save_folder = set_path(folder_name, PLOTS_DIR)

    plt.figure(figsize=(12, 6))
    
    if hasattr(full_history.index, 'year'): x_hist = full_history.index.year
    else: x_hist = full_history.index
        
    plt.plot(x_hist, full_history.values, label='History', color=COLORS.get('train', 'black'), marker='.', linewidth=2)
    
    last_year = x_hist.max()
    last_val = full_history.values[-1]
    
    connect_x_future = [last_year, future_pred['year'].iloc[0]]
    connect_y_future = [last_val, future_pred['pred'].iloc[0]]
    plt.plot(connect_x_future, connect_y_future, color=COLORS.get('pred_model', 'red'), linestyle='--', linewidth=2)
    plt.plot(future_pred['year'], future_pred['pred'], label=f'Forecast {model_name}', 
            color=COLORS.get('pred_model', 'red'), linestyle='--', linewidth=2, markersize=4)
    
    connect_x_baseline = [last_year, baseline_pred['year'].iloc[0]]
    connect_y_baseline = [last_val, baseline_pred['pred'].iloc[0]]
    plt.plot(connect_x_baseline, connect_y_baseline, color=COLORS.get('baseline', 'grey'), linestyle='--', linewidth=2)
    plt.plot(baseline_pred['year'], baseline_pred['pred'], label=f'Baseline ({baseline_name})', 
            color=COLORS.get('baseline', 'grey'), linestyle='--', linewidth=2, markersize=4)
    
    # Intervallo di Confidenza (Se esiste)
    if 'lower_ci' in future_pred.columns and 'upper_ci' in future_pred.columns:
        plt.fill_between(
            future_pred['year'], 
            future_pred['lower_ci'], 
            future_pred['upper_ci'], 
            color=COLORS.get('pred_model', 'red'), alpha=0.15, label='95% Confidence Interval'
        )
    
    plt.axvspan(last_year, 2030, color='#C3C3C3', alpha=0.3)
    
    long_name = REVERSE_VAR_NAMES.get(variable_name, variable_name)
    title_text = f"Future Forecast ({last_year}-2030): {long_name}"
    plt.title("\n".join(textwrap.wrap(title_text, width=70)), fontsize=14, fontweight='bold')
    
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if save_plots:
        filename = set_filename(variable_name, f"future{model_name}")
        full_path = os.path.join(save_folder, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Future Plot saved: {full_path}")
    plt.close()
    
# ========================================================================================================

def plot_nn_preds(variable_name, predictions_dict, train_df, val_df, test_df, model_name, folder_name):
    save_folder = os.path.join(PLOTS_DIR, folder_name)

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
                ax.axvspan(test_start, test_end, color='#C3C3C3', alpha=0.3)
                
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
        
        fig.suptitle(f'{variable_name} - {model_name} predictions', fontsize=14, fontweight='bold', y=0.93)
        
    filename = f"{model_name}_{variable_name}.png"
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
    plt.close()

# ========================================================================================================

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
    save_folder = set_path("07_AUGMENTATION", PLOTS_DIR)
    plt.figure(figsize=(12, 6))
    plt.plot(x_train_vals, y_train_vals, '.', label='Original', color='black', markersize=8)
    plt.plot(df_step['Year'], df_step['Value'], '-', label='Step Function', alpha=0.7)
    plt.plot(df_jitter['Year'], df_jitter['Value'], '--', label='Linear + Jittering', alpha=0.7)
    plt.title(f"Data Augmentation - {variable_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"AUG_{variable_name}.png"
    full_path = os.path.join(save_folder, filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Grafico salvato in: {full_path}")
    plt.close()

# ========================================================================================================

# mae_last = mean_absolute_error(test_data['Value'], test_data['pred_last_value'])
# mae_arimax = mean_absolute_error(test_data['Value'], test_data['pred_ARIMAX'])

# mse_last = mean_squared_error(test_data['Value'], test_data['pred_last_value'])
# mse_arimax = mean_squared_error(test_data['Value'], test_data['pred_ARIMAX'])

# print(f"Metriche di errore per {target}:")
# print(f"  MAE -> Last Value: {mae_last:.5f}, ARIMAX{order}: {mae_arimax:.5f}")
# print(f"  MSE -> Last Value: {mse_last:.5f}, ARIMAX{order}: {mse_arimax:.5f}")

# results_metrics = pd.DataFrame({
#     'Method': ['Last Value', f'ARIMAX{order}'],
#     'MSE': [mse_last, mse_arimax],
#     'MAE': [mae_last, mae_arimax]
# })

# fig, ax = plt.subplots(figsize=(8, 5))
# x = np.arange(len(results_metrics['Method']))
# width = 0.35

# bar1 = ax.bar(x - width/2, results_metrics['MSE'], width, label='MSE')
# bar2 = ax.bar(x + width/2, results_metrics['MAE'], width, label='MAE')

# ax.set_xticks(x)
# ax.set_xticklabels(results_metrics['Method'])
# ax.set_title(f"Error on predictions for {target}")
# ax.set_ylabel("Error Value")
# ax.legend()
# ax.grid(alpha=0.3, axis='y')

# def add_labels(bars):
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height:.5f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=9)

# add_labels(bar1)
# add_labels(bar2)

# plt.tight_layout()
# plt.show()


def plot_residuals(y_true, y_pred, variable_name, model_name, folder_name, save_plots=True):
    if y_true is None or y_pred is None or len(y_true) == 0: return

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred

    save_folder = set_path(folder_name, PLOTS_DIR)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Diagnostic: {variable_name} ({model_name})', fontsize=16, fontweight='bold')
    ax = axes.ravel()

    # 1. Residuals over Time
    ax[0].plot(residuals, color='purple', linewidth=1.5)
    ax[0].axhline(0, color='black', linestyle='--', linewidth=1)
    ax[0].set_title("1. Residuals over Time (Homoscedasticity)")
    ax[0].set_ylabel("Residuals")
    ax[0].grid(True, alpha=0.3)

    # 2. Distribution (Histogram)
    ax[1].hist(residuals, bins=15, density=True, color='purple', alpha=0.6, edgecolor='black')
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = ax[1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax[1].plot(x, p, 'k', linewidth=2, label=f'Norm (μ={mu:.2f})')
    ax[1].set_title("2. Residual Distribution (Normality)")
    ax[1].legend(loc='upper right', fontsize='small')

    # 3. ACF Plot
    lags_to_show = min(20, len(residuals)//2 - 1)
    if lags_to_show > 1:
        plot_acf(residuals, ax=ax[2], lags=lags_to_show, title="3. Autocorrelation (Whiteness)", fft=True, zero=False)
    else:
        ax[2].text(0.5, 0.5, "Not enough data", ha='center')

    # 4. scatter plot
    ax[3].scatter(y_true, y_pred, alpha=0.6, color='tab:blue', edgecolors='k', s=40)
    lims = [
        np.min([ax[3].get_xlim(), ax[3].get_ylim()]),  # min of both axes
        np.max([ax[3].get_xlim(), ax[3].get_ylim()]),  # max of both axes
    ]
    ax[3].plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Perfect Fit')
    ax[3].set_xlabel('Actual Values')
    ax[3].set_ylabel('Predicted Values')
    ax[3].set_title("4. Actual vs Predicted (Linearity)")
    ax[3].legend()
    ax[3].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    if save_plots:
        filename = set_filename(variable_name, f"RESID{model_name}")
        full_path = os.path.join(save_folder, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Diagnostics saved: {full_path}")
    
    plt.close()