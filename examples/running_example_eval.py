import os

BACKEND = 'torch' #'jax'  # You can use any backend here
os.environ['KERAS_BACKEND'] = BACKEND

import numpy as np
import pandas as pd

import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Input

from running_example import generate_data

from tkan import TKAN


keras.utils.set_random_seed(1)


df = pd.read_parquet('data.parquet')
df = df[(df.index >= pd.Timestamp('2020-01-01')) & (df.index < pd.Timestamp('2023-01-01'))]
assets = ['BTC', 'ETH', 'ADA', 'XMR', 'EOS', 'MATIC', 'TRX', 'FTM', 'BNB', 'XLM', 'ENJ', 'CHZ', 'BUSD', 'ATOM', 'LINK', 'ETC', 'XRP', 'BCH', 'LTC']
df = df[[c for c in df.columns if 'quote asset volume' in c and any(asset in c for asset in assets)]]
df.columns = [c.replace(' quote asset volume', '') for c in df.columns]


n_aheads = [1] #, 3, 6, 9, 12, 15]

for n_ahead in n_aheads:
    sequence_length = max(45, 5 * n_ahead)
    _, _, X_test, _, _, _, _, y_test, _, _, _, _ = generate_data(df, sequence_length, n_ahead)

    model = Sequential([
        Input(shape=X_test.shape[1:]),
        TKAN(100, return_sequences=True),
        TKAN(100, sub_kan_output_dim=20, sub_kan_input_dim=20, return_sequences=False),
        Dense(units=n_ahead, activation='linear')
    ], name='TKAN')

    model = load_model('./tkan_model.keras')

    preds = model.predict(X_test, batch_size=X_test.shape[0], verbose=False)

    dict_cp_0 = dict()
    for key, val in model.layers[0].cell.save_io.items():
        if val:
            val_numpy = np.array(val)
            val_numpy = val_numpy.reshape(-1, val_numpy.shape[2])
            # val_numpy = val_numpy[-1, :, :].squeeze()
            dict_cp_0[f'{key}_concat'] = val_numpy

    dict_cp_1 = dict()
    for key, val in model.layers[1].cell.save_io.items():
        if val:
            val_numpy = np.array(val)
            val_numpy = val_numpy.reshape(-1, val_numpy.shape[2])
            # val_numpy = val_numpy[-1, :, :].squeeze()
            dict_cp_1[f'{key}_concat'] = val_numpy

    for key, value in dict_cp_0.items():
        df = pd.DataFrame(value)
        df.to_csv(f"./df_{key}.csv")

    for key, value in dict_cp_1.items():
        df = pd.DataFrame(value)
        df.to_csv(f"./df_{key}.csv")



