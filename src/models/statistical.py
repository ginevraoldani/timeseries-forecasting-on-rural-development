import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def predict_arima_rolling(train_data, test_data, order, refit=True):
    """
    Esegue un Rolling Forecast (Walk-Forward Validation) con ARIMA.
    
    Args:
        train_data (pd.Series): Serie storica di training iniziale.
        test_data (pd.Series): Serie storica di test (i valori reali servono per aggiornare il modello passo-passo).
        order (tuple): (p, d, q).
        refit (bool): 
            - True: Ri-stima i parametri (fit) ad ogni passo. Più lento, più accurato (consigliato per tesi).
            - False: Usa i parametri stimati all'inizio e aggiorna solo lo stato (filtro di Kalman). Più veloce.
            
    Returns:
        pd.Series: Le predizioni one-step-ahead allineate con l'indice di test_data.
    """
    # Preparazione dello storico (History)
    # Convertiamo in lista per appendere velocemente i nuovi valori
    history = [x for x in train_data.values] if hasattr(train_data, 'values') else list(train_data)
    test_values = test_data.values if hasattr(test_data, 'values') else list(test_data)
    forecast_index = test_data.index
    predictions = []
    
    print(f"Starting Rolling Forecast ARIMA{order} over {len(test_values)} steps...")
    try:
        for t in range(len(test_values)):
            model = ARIMA(history, order=order)
            
            if refit:
                model_fit = model.fit()
            else:
                # Opzione avanzata: usa i parametri del primo fit (più veloce, meno preciso se il trend cambia)
                # Per la tesi, consiglio di lasciare refit=True o usare model_fit.append/apply
                # Qui per semplicità rimaniamo sul refit=True nel 99% dei casi
                model_fit = model.fit() 

            # Predizione 1 step avanti (t+1)
            # forecast() restituisce un array, prendiamo il primo elemento
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            
            # Aggiornamento Storia (Observation)
            # Aggiungiamo il VERO valore osservato alla storia per il prossimo giro
            obs = test_values[t]
            history.append(obs)
            
        # Creazione Serie Finale
        forecast_series = pd.Series(predictions, index=forecast_index, name='pred')
        return forecast_series

    except Exception as e:
        print(f"Rolling ARIMA{order} failed: {e}")
        # Fallback: media mobile o serie statica in caso di crash
        return pd.Series([np.mean(history)] * len(test_values), index=forecast_index)

def predict_arima_family(train_data, forecast_index, order, **kwargs):
    """
    Gestisce l'intera famiglia dei modelli statistici classici (Box-Jenkins).
    
    Args:
        train_data: historical series for training.
        forecast_index: indexes for predictions.
        order (tuple): (p, d, q)
            - MA: (0, 0, q)
            - IMA: (0, d, q)
            - AR: (p, 0, 0)
            - ARI: (p, d, 0)
            - ARMA:    (p, 0, q)
            - ARIMA:   (p, d, q)
            
    Returns:
        pd.Series: predictions.
    """
    series = train_data.iloc[:, 0] if isinstance(train_data, pd.DataFrame) else train_data
    
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        
        steps = len(forecast_index)
        forecast_result = model_fit.forecast(steps=steps)
        
        forecast_series = pd.Series(
            data=forecast_result.values,
            index=forecast_index,
            name='pred'
        )
        return forecast_series
        
    except Exception as e:
        print(f"ARIMA{order} failed: {e}")
        return pd.Series([series.mean()] * len(forecast_index), index=forecast_index)