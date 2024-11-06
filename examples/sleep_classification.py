import os
import time
BACKEND = 'jax' # You can use any backend here
os.environ['KERAS_BACKEND'] = BACKEND

import jax.numpy as jnp

import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Input

from sklearn.metrics import f1_score

from tkan import TKAN, get_dataframe, create_df, get_boas_data, split_data, extract_weights, get_dir


keras.utils.set_random_seed(1)

N_MAX_EPOCHS = 1
BATCH_SIZE = 128

early_stopping_callback = lambda: keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=10,
    mode="min",
    restore_best_weights=True,
    start_from_epoch=5,
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

samples, chunks = 7680, 1
seq_len = samples // chunks

bitbrain_dir = get_dir('..', 'data')
raw_dir = get_dir('..', 'data', 'raw')

get_boas_data(base_path=bitbrain_dir, output_path=raw_dir)

datapaths = split_data(dir=raw_dir, train_size=1, test_size=1)
# datapaths = split_data(dir=raw_dir, train_size=46, test_size=5)

train_path, test_path = datapaths[0], datapaths[1]

exist = True
train_df = get_dataframe(train_path, name="train", seq_len=seq_len, exist=exist)

global weights
_, weights = extract_weights(train_df, label_col='majority_class')

classes = list(weights.keys())

print(list(weights.values()))
weights = jnp.array(list(weights.values()))

X_train, y_train = create_df(df=train_df)

n_classes = len(classes)

model = Sequential([
    Input(shape=X_train.shape[1:]),
    TKAN(2, return_sequences=True),
    TKAN(2, sub_kan_output_dim=5, sub_kan_input_dim=5, return_sequences=False),
    Dense(units=n_classes, activation='softmax')
], name='TKAN')


def weighted_categorical_crossentropy(class_weights):
    def loss(y_true, y_pred):
        # Clip y_pred to avoid log(0) errors
        y_pred = jnp.clip(y_pred, 1e-7, 1 - 1e-7)

        # Calculate categorical cross-entropy loss
        cce = -jnp.sum(y_true * jnp.log(y_pred), axis=-1)

        # Get the weights for each class
        weights = jnp.sum(y_true * class_weights, axis=-1)  # Weight each sample by true class

        # Calculate the weighted loss for each sample
        weighted_loss = cce * weights
        return jnp.mean(weighted_loss)  # Return the mean weighted loss

    return loss


optimizer = keras.optimizers.Adam(0.001)
# model.compile(optimizer=optimizer, loss=weighted_categorical_crossentropy(class_weights=weights), jit_compile=True)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', jit_compile=True, metrics=['categorical_accuracy'])
model.summary()

# Fit the model
start_time = time.time()
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_MAX_EPOCHS, validation_split=0.2,
                    callbacks=callbacks(), shuffle=True, verbose=False)
end_time = time.time()
time_results.append(end_time - start_time)

# Begin Evaluation
test_df = get_dataframe(test_path, name="test", seq_len=seq_len, exist=False)
X_test, y_test = create_df(df=test_df)

# Evaluate the model on the test set
y_pred = model.predict(X_test, verbose=False)
print(f'Training time {end_time - start_time}')

del model
del optimizer

y_pred = np.argmax(y_pred, axis=1)

f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_per_class = f1_score(y_test, y_pred, average=None)  # F1 score for each class

print(f"Macro F1 Score: {f1_macro}")
print(f"Micro F1 Score: {f1_micro}")
print(f"Weighted F1 Score: {f1_weighted}")
print(f"F1 Score per class: {f1_per_class}")
