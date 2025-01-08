import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input

from tkan import save_json, save_csv
from tkan import TKAN


def create_data(data_pth):

    data = pd.read_csv(f"{data_pth}/MackyG17.csv")

    train_size = int(data.shape[0] * 0.4)
    val_size = int(data.shape[0] * 0.1)

    data_train = data.iloc[:train_size].reset_index(drop=True)
    data_val = data.iloc[train_size:(val_size+train_size)].reset_index(drop=True)
    data_test = data.iloc[(val_size+train_size):].reset_index(drop=True)

    data_train.to_csv(f"{data_pth}/train.csv")
    data_val.to_csv(f"{data_pth}/val.csv")
    data_test.to_csv(f"{data_pth}/test.csv")

    return data_train.squeeze(), data_val.squeeze(), data_test.squeeze()


N_MAX_EPOCHS = 1 #500
BATCH_SIZE = 2
early_stopping_callback = lambda: keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=10,
    mode="min",
    restore_best_weights=True,
    start_from_epoch=1
)

callbacks = lambda: [early_stopping_callback(), keras.callbacks.TerminateOnNaN()]


def mackeyGlass(data, sequence_length):

    X, y = [], []

    for idx in range(data.shape[0] - sequence_length):
        X.append(data[idx:idx + sequence_length])
        y.append(data[idx + sequence_length])

    X, y = np.array(X), np.array(y)

    return np.expand_dims(X, axis=2), y


def train(dir_path, sequence_length, data_train, data_val):

    X_train, y_train = mackeyGlass(data_train, sequence_length)
    X_val, y_val = mackeyGlass(data_val, sequence_length)

    X_train = np.concatenate((X_train, X_val), axis=0)
    y_train = np.concatenate((y_train, y_val), axis=0)

    model = Sequential([
        Input(shape=X_train.shape[1:]),
        TKAN(100, return_sequences=True),
        TKAN(100, sub_kan_output_dim=20, sub_kan_input_dim=20, return_sequences=False),
        Dense(units=1, activation='linear')
    ], name='TKAN')

    optimizer = keras.optimizers.Adam(0.05)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_MAX_EPOCHS, validation_split=0.2,
              callbacks=callbacks(), shuffle=True, verbose=False)

    model.save(f'{dir_path}/tkan_model.keras')


def test(dir_path, sequence_length, data_test):

    X_test, y_test = mackeyGlass(data_test, sequence_length)

    model = Sequential([
        Input(shape=X_test.shape[1:]),
        TKAN(100, return_sequences=True),
        TKAN(100, sub_kan_output_dim=20, sub_kan_input_dim=20, return_sequences=False),
        Dense(units=2, activation='linear')
    ], name='TKAN')

    model = load_model(f'{dir_path}/tkan_model.keras')

    preds = model.predict(X_test, batch_size=X_test.shape[0], verbose=False)

    def test_model(true, preds, x):

        data_test = {'x_0': x.squeeze()[:, 0],
                     'x_1': x.squeeze()[:, 1],
                     'true_label': true.squeeze(),
                     'pred_label': preds.squeeze()}

        save_csv(data_test, f'{dir_path}/data_test.csv')

        # Start computing regression metrics
        rmse = np.sqrt(np.mean(np.square(np.subtract(true, preds))))
        mae = np.mean(np.abs(np.subtract(true, preds)))
        mape = np.mean(np.abs(np.subtract(true, preds) / np.array(true))) * 100

        # Calculate residuals
        residuals = np.subtract(true, preds)

        # Plot true vs predicted
        plt.figure(figsize=(10, 5))

        # True vs Predicted Plot
        plt.figure(figsize=(6, 6))
        plt.scatter(true, preds, color='blue', alpha=0.7)
        plt.plot([min(true), max(true)], [min(true), max(true)],
                 color='red', linestyle='--', linewidth=1.5)
        plt.title("True vs Predicted")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{dir_path}/true_vs_predicted.png", dpi=600)
        plt.close()

        # Residual Distribution Plot
        plt.figure(figsize=(6, 6))
        plt.hist(residuals, bins=10, alpha=0.7)
        plt.title("Residual Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{dir_path}/residual_distribution.png", dpi=600)
        plt.close()

        return rmse, mae, mape

    rmse, mae, mape = test_model(y_test, preds, x=X_test)

    print(f"RMSE  : {rmse:.3f}")
    print(f"MAE   : {mae:.3f}")
    print(f"MAPE  : {mape:.3f}")

    metrics_test = {'rmse': float(rmse),
                    'mae': float(mae),
                    'mape': float(mape)}

    save_json(metrics_test, f'{dir_path}/metrics_test.json')
