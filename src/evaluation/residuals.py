import numpy as np
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox

# i RESIDUI sono l'errore del modello istante per istante (realtà - previsione)

# dovrebbero essere WHITE NOISE ->  1. media == 0
#                                   2. non avere schemi ripetitivi
#                                   3. essere casuali

# SHAPIRO-WILK TEST vede se residui si distribuiscono secondo curva a campana (gaussiana)
#       -> H0 : i dati sono normali (seguono gaussiana),
#       p < 0.05 : rifiuto ipotesi = residui NON sono normali (male per intervalli di confidenza)
#    !! p > 0.05 : accetto ipotesi = residui sono normali

# LJUNG-BOX TEST verifica se c'è relazione tra errore di oggi e errore di ieri (= autocorrelazione)
# se c'è autocorrelazione nei residui, significa che c'è ancora informazione nei dati che il modello non ha usato.
# speriamo che errori siano casuali, random, non vogliamo che abbiano pattern nascosti che non sono stati catturati dal modello.
#       -> H0 : dati sono casuali (no autocorrelazione),
#       p < 0.05 : rifiuto ipotesi = c'è autocorrelazione (modello è incompleto o migliorabile)
#    !! p > 0.05 : accetto ipotesi = residui sono rumore bianco

def compute_residual_diagnostics(y_true, y_pred):
    """
    Computes residuals and performs statistical diagnostic tests for model evaluation.
    
    Automatically manages:
    - list/array input ( -> flattens).
    - NaN deletion.
    
    Performs:
    1. residual calculation = (y_true - y_pred).
    2. descriptive statistics = mean, std
    3. Shapiro-Wilk test = normality of residuals
    4. Ljung-Box test = autocorrelation of residuals / white noise

    Args:
        y_true (array-like): real values.
        y_pred (array-like): predicted values.

    Returns:
        dict: dictionary containing keys ['residual_mean', 'residual_std', 'shapiro_wilk_pvalue', 'ljung_box_pvalue']
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    residuals = y_true_clean - y_pred_clean

    if len(residuals) < 3:
        return {
            "residual_mean": np.nan,
            "residual_std": np.nan,
            "shapiro_wilk_pvalue": np.nan,
            "ljung_box_pvalue": np.nan
        }

    res_mean = np.mean(residuals)
    res_std = np.std(residuals)

    try:
        _, p_shapiro = shapiro(residuals)
    except Exception:
        p_shapiro = np.nan

    try:
        # Regola empirica per i lag: min(10, N/5)
        lags = min(10, len(residuals) // 5)
        if lags < 1: lags = 1
        
        lb_res = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        p_lb = lb_res['lb_pvalue'].iloc[0]
    except Exception:
        p_lb = np.nan

    return {
        "residual_mean": round(res_mean, 5),
        "residual_std": round(res_std, 5),
        "shapiro_wilk_pvalue": round(p_shapiro, 5) if not np.isnan(p_shapiro) else None,
        "ljung_box_pvalue": round(p_lb, 5) if not np.isnan(p_lb) else None
    }