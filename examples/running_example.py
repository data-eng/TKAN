class MinMaxScaler:
    def __init__(self, feature_axis=None, minmax_range=(0, 1)):
        """
        Initialize the MinMaxScaler.
        Args:
        feature_axis (int, optional): The axis that represents the feature dimension if applicable.
                                      Use only for 3D data to specify which axis is the feature axis.
                                      Default is None, automatically managed based on data dimensions.
        """
        self.feature_axis = feature_axis
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.minmax_range = minmax_range  # Default range for scaling (min, max)

    def fit(self, X):
        import numpy as np
        """
        Fit the scaler to the data based on its dimensionality.
        Args:
        X (np.array): The data to fit the scaler on.
        """
        if X.ndim == 3 and self.feature_axis is not None:  # 3D data
            axis = tuple(i for i in range(X.ndim) if i != self.feature_axis)
            self.min_ = np.min(X, axis=axis)
            self.max_ = np.max(X, axis=axis)
        elif X.ndim == 2:  # 2D data
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
        elif X.ndim == 1:  # 1D data
            self.min_ = np.min(X)
            self.max_ = np.max(X)
        else:
            raise ValueError("Data must be 1D, 2D, or 3D.")

        self.scale_ = self.max_ - self.min_
        return self

    def transform(self, X):
        """
        Transform the data using the fitted scaler.
        Args:
        X (np.array): The data to transform.
        Returns:
        np.array: The scaled data.
        """
        X_scaled = (X - self.min_) / self.scale_
        X_scaled = X_scaled * (self.minmax_range[1] - self.minmax_range[0]) + self.minmax_range[0]
        return X_scaled

    def fit_transform(self, X):
        """
        Fit to data, then transform it.
        Args:
        X (np.array): The data to fit and transform.
        Returns:
        np.array: The scaled data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """
        Inverse transform the scaled data to original data.
        Args:
        X_scaled (np.array): The scaled data to inverse transform.
        Returns:
        np.array: The original data scale.
        """
        X = (X_scaled - self.minmax_range[0]) / (self.minmax_range[1] - self.minmax_range[0])
        X = X * self.scale_ + self.min_
        return X


def generate_data(df, sequence_length, n_ahead=1):

    import numpy as np

    # Case without known inputs
    scaler_df = df.copy().shift(n_ahead).rolling(24 * 14).median()
    tmp_df = df.copy() / scaler_df
    tmp_df = tmp_df.iloc[24 * 14 + n_ahead:].fillna(0.)
    scaler_df = scaler_df.iloc[24 * 14 + n_ahead:].fillna(0.)

    def prepare_sequences(df, scaler_df, n_history, n_future):
        X, y, y_scaler = [], [], []
        num_features = df.shape[1]

        # Iterate through the DataFrame to create sequences
        for i in range(n_history, len(df) - n_future + 1):
            # Extract the sequence of past observations
            X.append(df.iloc[i - n_history:i].values)
            # Extract the future values of the first column
            y.append(df.iloc[i:i + n_future, 0:1].values)
            y_scaler.append(scaler_df.iloc[i:i + n_future, 0:1].values)

        X, y, y_scaler = np.array(X), np.array(y), np.array(y_scaler)
        return X, y, y_scaler

    # Prepare sequences
    X, y, y_scaler = prepare_sequences(tmp_df, scaler_df, sequence_length, n_ahead)

    # Split the dataset into training and testing sets
    train_test_separation = int(len(X) * 0.8)
    X_train_unscaled, X_test_unscaled = X[:train_test_separation], X[train_test_separation:]
    y_train_unscaled, y_test_unscaled = y[:train_test_separation], y[train_test_separation:]
    y_scaler_train, y_scaler_test = y_scaler[:train_test_separation], y_scaler[train_test_separation:]

    # Generate the data
    X_scaler = MinMaxScaler(feature_axis=2)
    X_train = X_scaler.fit_transform(X_train_unscaled)
    X_test = X_scaler.transform(X_test_unscaled)

    y_scaler = MinMaxScaler(feature_axis=2)
    y_train = y_scaler.fit_transform(y_train_unscaled)
    y_test = y_scaler.transform(y_test_unscaled)

    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    return X_scaler, X_train, X_test, X_train_unscaled, X_test_unscaled, y_scaler, y_train, y_test, y_train_unscaled, y_test_unscaled, y_scaler_train, y_scaler_test

import os
BACKEND = 'torch'  # You can use any backend here
os.environ['KERAS_BACKEND'] = BACKEND

import numpy as np
import pandas as pd

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Input

from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from tkan import TKAN

import time

keras.utils.set_random_seed(1)

N_MAX_EPOCHS = 1000
BATCH_SIZE = 128
early_stopping_callback = lambda: keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=10,
    mode="min",
    restore_best_weights=True,
    start_from_epoch=6,
)
lr_callback = lambda: keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.25,
    patience=5,
    mode="min",
    min_delta=0.00001,
    min_lr=0.000025,
    verbose=0,
)

callbacks = lambda: [early_stopping_callback(), lr_callback(), keras.callbacks.TerminateOnNaN()]


def train():

    df = pd.read_parquet('data.parquet')
    df = df[(df.index >= pd.Timestamp('2020-01-01')) & (df.index < pd.Timestamp('2023-01-01'))]
    assets = ['BTC', 'ETH', 'ADA', 'XMR', 'EOS', 'MATIC', 'TRX', 'FTM', 'BNB', 'XLM', 'ENJ', 'CHZ', 'BUSD', 'ATOM', 'LINK', 'ETC', 'XRP', 'BCH', 'LTC']
    df = df[[c for c in df.columns if 'quote asset volume' in c and any(asset in c for asset in assets)]]
    df.columns = [c.replace(' quote asset volume', '') for c in df.columns]

    n_aheads = [1] #, 3, 6, 9, 12, 15]

    results = {n_ahead: [] for n_ahead in n_aheads}
    results_rmse = {n_ahead: [] for n_ahead in n_aheads}
    time_results = {n_ahead: [] for n_ahead in n_aheads}
    for n_ahead in n_aheads:
        sequence_length = max(45, 5 * n_ahead)
        X_scaler, X_train, X_test, X_train_unscaled, X_test_unscaled, y_scaler, y_train, y_test, y_train_unscaled, y_test_unscaled, y_scaler_train, y_scaler_test = generate_data(
            df, sequence_length, n_ahead)

        for run in range(1):

            model = Sequential([
                Input(shape=X_train.shape[1:]),
                TKAN(100, return_sequences=True),
                TKAN(100, sub_kan_output_dim=20, sub_kan_input_dim=20, return_sequences=False),
                Dense(units=n_ahead, activation='linear')
            ], name='TKAN')

            optimizer = keras.optimizers.Adam(0.001)
            model.compile(optimizer=optimizer, loss='mean_squared_error') #, jit_compile=True)
            if run == 0:
                model.summary()

            # Fit the model
            start_time = time.time()

            model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_MAX_EPOCHS, validation_split=0.2,
                      callbacks=callbacks(), shuffle=True, verbose=False)

            end_time = time.time()

            model.save('./tkan_model.keras')

            time_results[n_ahead].append(end_time - start_time)
            # Evaluate the model on the test set
            preds = model.predict(X_test, verbose=False)
            r2 = r2_score(y_true=y_test, y_pred=preds)
            print(end_time - start_time, r2)
            rmse = root_mean_squared_error(y_true=y_test, y_pred=preds)
            results[n_ahead].append(r2)
            results_rmse[n_ahead].append(rmse)

            del model
            del optimizer

    print('R2 scores')
    print('Means:')
    print({n_ahead: np.mean(results[n_ahead]) for n_ahead in n_aheads})
    print({n_ahead: np.mean(results_rmse[n_ahead]) for n_ahead in n_aheads})
    print('Std:')
    print({n_ahead: np.std(results[n_ahead]) for n_ahead in n_aheads})
    print({n_ahead: np.std(results_rmse[n_ahead]) for n_ahead in n_aheads})
    print('Training Times')
    print({n_ahead: np.mean(time_results[n_ahead]) for n_ahead in n_aheads})
    print({n_ahead: np.std(time_results[n_ahead]) for n_ahead in n_aheads})

# train()
