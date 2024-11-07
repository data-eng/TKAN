import os

BACKEND = 'jax'  # You can use any backend here
os.environ['KERAS_BACKEND'] = BACKEND

import jax.numpy as jnp
import numpy as np

import keras
from keras.models import Sequential, load_model

from sklearn.metrics import f1_score

from tkan import get_dataframe, create_df, load_json, get_path


samples, chunks = 7680, 1
seq_len = samples // chunks

test_path = get_path('..', '..', 'data', 'proc', filename='test.csv')

path = get_path('..', '..', 'data', filename='weights.json')
weights = load_json(path)
classes = list(weights.keys())
weights = jnp.array(list(weights.values()))
n_classes = len(classes)

# Begin Evaluation
exist = True
test_df = get_dataframe(test_path, name="test", seq_len=seq_len, exist=exist)
X_test, y_test = create_df(df=test_df)

model = load_model('tkan_model.keras')

# Evaluate the model on the test set
y_pred = model.predict(X_test, verbose=False)

print(model)

del model

y_pred = np.argmax(y_pred, axis=1)

f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_per_class = f1_score(y_test, y_pred, average=None)  # F1 score for each class

print(f"Macro F1 Score: {f1_macro}")
print(f"Micro F1 Score: {f1_micro}")
print(f"Weighted F1 Score: {f1_weighted}")
print(f"F1 Score per class: {f1_per_class}")
