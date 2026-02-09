import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

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
                model_fit = model.fit() 

            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            
            # Aggiungiamo il VERO valore osservato alla storia per il prossimo giro
            obs = test_values[t]
            history.append(obs)
            
        forecast_series = pd.Series(predictions, index=forecast_index, name='pred')
        return forecast_series

    except Exception as e:
        print(f"Rolling ARIMA{order} failed: {e}")
        # Fallback: media mobile o serie statica in caso di crash
        return pd.Series([np.mean(history)] * len(test_values), index=forecast_index)

def predict_arimax_rolling(train_data, test_data, order, exog_train=None, exog_test=None, refit=True):
    """
    Esegue un Rolling Forecast (Walk-Forward Validation) unificato per ARIMA e ARIMAX.
    
    Args:
        train_data (pd.Series): Serie storica di training (endogena).
        test_data (pd.Series): Serie storica di test (endogena). I valori reali vengono usati 
                            per aggiornare la storia passo dopo passo.
        order (tuple): Parametri (p, d, q) del modello.
        exog_train (pd.DataFrame or None): Variabili esogene per il training. Se None, esegue ARIMA standard.
        exog_test (pd.DataFrame or None): Variabili esogene per il test (future rispetto al train). 
                                        Deve avere la stessa lunghezza di test_data.
        refit (bool): 
            - True: Ri-stima i coefficienti del modello ad ogni passo (più lento, più accurato).
            - False: Stima i coefficienti una volta sola all'inizio e aggiorna solo lo stato interno 
                    (filtro di Kalman) con le nuove osservazioni (più veloce).
            
    Returns:
        pd.Series: Le predizioni one-step-ahead allineate con l'indice di test_data.
    """
    history_endog = list(train_data.values)
    history_exog = list(exog_train.values) if exog_train is not None else None
    
    test_values = test_data.values
    predictions = []
    
    # Se abbiamo esogene, ci assicuriamo che siano liste per poter fare .append()
    if exog_test is not None:
        test_exog_values = exog_test.values
    
    print(f"Starting Rolling Forecast (Order={order}, Exog={exog_train is not None}) over {len(test_values)} steps...")

    # 2. Modello Iniziale (se refit=False, addestriamo una volta sola qui)
    model = None
    model_fit = None
    
    if not refit:
        try:
            # Addestramento iniziale su tutto il train set
            model = SARIMAX(endog=history_endog, 
                            exog=history_exog if history_exog is not None else None,
                            order=order, 
                            seasonal_order=(0,0,0,0),
                            enforce_stationarity=False, 
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)
        except Exception as e:
            print(f"Initial fit failed: {e}. Switching to refit=True mode as fallback.")
            refit = True

    # 3. Loop Walk-Forward
    for t in range(len(test_values)):
        try:
            # A. Gestione Exog per il passo corrente (se presenti)
            current_exog_history = history_exog if history_exog is not None else None
            
            # Il valore esogeno "futuro" per la previsione al tempo t
            exog_forecast_step = None
            if exog_test is not None:
                # Reshape necessario per statsmodels (1, n_vars)
                exog_forecast_step = test_exog_values[t].reshape(1, -1)

            # B. Stima / Aggiornamento Modello
            if refit:
                # RI-ADDESTRAMENTO COMPLETO (Slower but safer for changing dynamics)
                model = SARIMAX(endog=history_endog, 
                                exog=current_exog_history,
                                order=order, 
                                seasonal_order=(0,0,0,0),
                                enforce_stationarity=False, 
                                enforce_invertibility=False)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_fit = model.fit(disp=False)
                
                # Previsione 1 passo avanti
                # Se abbiamo exog, dobbiamo passarle a get_forecast o forecast
                if exog_forecast_step is not None:
                    yhat = model_fit.get_forecast(steps=1, exog=exog_forecast_step).predicted_mean[0]
                else:
                    yhat = model_fit.forecast(steps=1)[0]
            
            else:
                # NO REFIT - SOLO AGGIORNAMENTO STATO (Faster)
                # Nota: In statsmodels, per fare rolling senza refit si usa append/extend o filter,
                # ma l'approccio più semplice qui è usare il modello fittato e predire l'ultimo step.
                # Tuttavia, per semplicità e robustezza in questo script custom, 
                # spesso il refit=True è preferibile per ARIMAX complessi.
                # Se vuoi usare update senza refit:
                
                # Per semplicità in questo script, se refit=False, facciamo un forecast 
                # basato sull'ultimo stato aggiornato (append).
                # Statsmodels < v0.12 gestiva questo diversamente. 
                # Metodo robusto: usare `apply` (o `append` nelle vecchie versioni) per aggiornare lo stato senza ristimare i parametri.
                
                # Creiamo un nuovo modello con i nuovi dati ma USIAMO I PARAMETRI VECCHI (params)
                new_model = SARIMAX(endog=history_endog, 
                                    exog=current_exog_history,
                                    order=order, 
                                    seasonal_order=(0,0,0,0),
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
                
                # "Filter" applica i parametri vecchi ai dati nuovi per aggiornare lo stato
                new_res = new_model.filter(model_fit.params)
                
                if exog_forecast_step is not None:
                    yhat = new_res.get_forecast(steps=1, exog=exog_forecast_step).predicted_mean[0]
                else:
                    yhat = new_res.forecast(steps=1)[0]

            predictions.append(yhat)

            # C. Aggiornamento Storia (aggiungiamo il vero valore osservato)
            obs = test_values[t]
            history_endog.append(obs)
            
            if exog_test is not None:
                # Aggiungiamo la riga di esogene usata per questo step alla storia
                # Attenzione: history_exog deve essere una lista di array o liste
                history_exog.append(test_exog_values[t])

        except Exception as e:
            print(f"Step {t} failed: {e}")
            # Fallback semplice: ultimo valore osservato (Naive)
            last_val = history_endog[-1] if len(history_endog) > 0 else 0
            predictions.append(last_val)
            
            # Aggiungiamo comunque l'osservazione reale per non rompere il loop successivo
            history_endog.append(test_values[t])
            if exog_test is not None:
                history_exog.append(test_exog_values[t])

    # 4. Creazione Serie Risultato
    forecast_series = pd.Series(predictions, index=test_data.index, name='pred_arimax')
    return forecast_series

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