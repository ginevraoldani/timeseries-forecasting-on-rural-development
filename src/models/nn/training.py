import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping        # type: ignore

def train_model(model, X_train, y_train, X_val, y_val, batch_size, epochs=50, patience=10, extra_callbacks=None):
    callbacks_list = [
        EarlyStopping(
            monitor='val_loss', 
            patience=patience, 
            restore_best_weights=True,
            verbose=0
        )
    ]
    if extra_callbacks:
        callbacks_list.extend(extra_callbacks)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        verbose=0
    )
    return history, model