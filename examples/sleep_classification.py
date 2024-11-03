import os
BACKEND = 'jax' # You can use any backend here
os.environ['KERAS_BACKEND'] = BACKEND

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Input

from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from tkan import TKAN, get_dataframes, create_dfs, get_boas_data, split_data, extract_weights, get_dir

import time

keras.utils.set_random_seed(1)

N_MAX_EPOCHS = 2
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


results = []
results_rmse = []
time_results = []

samples, chunks = 7680, 32
seq_len = samples // chunks

bitbrain_dir = get_dir('..', 'data')
raw_dir = get_dir('..', 'data', 'raw')

get_boas_data(base_path=bitbrain_dir, output_path=raw_dir)

datapaths = split_data(dir=raw_dir, train_size=43, test_size=13)

train_df, test_df = get_dataframes(datapaths, seq_len=seq_len, exist=True)
_, weights = extract_weights(train_df, label_col='majority')

classes = list(weights.keys())

datasets = create_dfs(dataframes=(train_df, test_df))

X_train, y_train = datasets[0]
X_test, y_test = datasets[1]

# Find the maximum length of lists in the entire DataFrame
max_length = X_train.applymap(len).max().max()

# Define a function to pad each list with zeros to match max_length
def pad_list(lst):
    return np.pad(lst, (0, max_length - len(lst)), 'constant')

# Apply padding to each list in the DataFrame
for col in X_train.columns:
    X_train[col] = X_train[col].apply(pad_list)
    X_test[col] = X_test[col].apply(pad_list)

# Convert the entire DataFrame to a NumPy array
X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

X_train = np.stack([np.stack(row) for row in X_train])
X_test = np.stack([np.stack(row) for row in X_test])
y_train = np.stack([np.stack(row) for row in y_train])
y_test = np.stack([np.stack(row) for row in y_test])

model = Sequential([
    Input(shape=X_train.shape[1:]),
    TKAN(10, return_sequences=True),
    TKAN(10, sub_kan_output_dim=20, sub_kan_input_dim=20, return_sequences=False),
    Dense(units=1, activation='linear')
], name='TKAN')

optimizer = keras.optimizers.Adam(0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error', jit_compile=True)
model.summary()

# Fit the model
start_time = time.time()
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_MAX_EPOCHS, validation_split=0.2,
                    callbacks=callbacks(), shuffle=True, verbose=False)
end_time = time.time()
time_results.append(end_time - start_time)

# Evaluate the model on the test set
preds = model.predict(X_test, verbose=False)
r2 = r2_score(y_true=y_test, y_pred=preds)
print(end_time - start_time, r2)
rmse = root_mean_squared_error(y_true=y_test, y_pred=preds)
results.append(r2)
results_rmse.append(rmse)

del model
del optimizer

print('R2 scores')
print('Means:')
print(np.mean(results))
print(np.mean(results_rmse))
print('Std:')
print(np.std(results))
print(np.std(results_rmse))
print('Training Times')
print(np.mean(time_results))
print(np.std(time_results))
