import keras
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, accuracy_score
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input

from tkan import save_json, save_csv
from tkan import TKAN


N_MAX_EPOCHS = 500
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


def pendulumTimeSeriesDataset(data, sequence_length, indices):

    features = data[['angular_displacement', 'angular_velocity']]
    labels_1 = data['label_1']
    labels_2 = data['label_2']

    X, y = [], []

    for idx in range(len(features) - sequence_length + 1):

        X.append(features[idx:idx + sequence_length])

        y_1 = labels_1[idx + sequence_length - 1]
        y_2 = labels_2[idx + sequence_length - 1]

        y.append(np.array([y_1, y_2]))

    X, y = np.array(X), np.array(y)

    return X[indices], y[indices]


def train(dir_path, sequence_length):
    exps = "../../KANs"

    data = pd.read_csv(f'{exps}/experiments/data/pendulum.csv')

    train_indices = np.load(f'{exps}/experiments/data/train_indices_{sequence_length}.npy')
    val_indices = np.load(f'{exps}/experiments/data/val_indices_{sequence_length}.npy')

    train_indices = np.concatenate((train_indices, val_indices))

    X_train, y_train = pendulumTimeSeriesDataset(data, sequence_length, indices=train_indices)

    model = Sequential([
        Input(shape=X_train.shape[1:]),
        TKAN(100, return_sequences=True),
        TKAN(100, sub_kan_output_dim=20, sub_kan_input_dim=20, return_sequences=False),
        Dense(units=2, activation='linear')
    ], name='TKAN')

    optimizer = keras.optimizers.Adam(0.05)
    model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(from_logits=True))

    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=N_MAX_EPOCHS, validation_split=0.2,
              callbacks=callbacks(), shuffle=True, verbose=False)

    model.save(f'{dir_path}/tkan_model.keras')


def test(dir_path, sequence_length):
    exps = "../../KANs"

    data = pd.read_csv(f'{exps}/experiments/data/pendulum.csv')
    test_indices = np.load(f'{exps}/experiments/data/test_indices_{sequence_length}.npy')
    X_test, y_test = pendulumTimeSeriesDataset(data, sequence_length, indices=test_indices)

    model = Sequential([
        Input(shape=X_test.shape[1:]),
        TKAN(100, return_sequences=True),
        TKAN(100, sub_kan_output_dim=20, sub_kan_input_dim=20, return_sequences=False),
        Dense(units=2, activation='linear')
    ], name='TKAN')

    model = load_model(f'{dir_path}/tkan_model.keras')

    preds = model.predict(X_test, batch_size=X_test.shape[0], verbose=False)
    preds = np.where(expit(preds) <= 0.5, 0, 1)

    def test_model(true, preds, x):

        y1 = {"all_labels": [], "all_preds": []}
        y2 = {"all_labels": [], "all_preds": []}

        categorical = {"y1": "Energy Label", "y2": "Equilibrium label",
                       "y1_pos": "Energy Increasing", "y2_pos": "Close to equilibrium",
                       "y1_neg": "Energy Decreasing", "y2_neg": "Far from equilibrium"}

        y1["all_labels"] = true[:, 0]
        y1["all_preds"] = preds[:, 0]

        y2["all_labels"] = true[:, 1]
        y2["all_preds"] = preds[:, 1]

        all_labels_ = [np.array(y1["all_labels"]), np.array(y2["all_labels"])]
        all_preds_ = [np.array(y1["all_preds"]), np.array(y2["all_preds"])]

        data_test = {'x': [matrix.tolist() for matrix in x],
                     'true_label_1': all_labels_[0].astype(int),
                     'true_label_2': all_labels_[1].astype(int),
                     'pred_label_1': all_preds_[0].astype(int),
                     'pred_label_2': all_preds_[1].astype(int)}

        save_csv(data_test, f'{dir_path}/data_test.csv')

        f1, precision, recall, accuracy = dict(), dict(), dict(), dict()

        for i, (all_labels, all_preds) in enumerate(zip(all_labels_, all_preds_)):
            f1[f'y{i + 1}'] = f1_score(all_labels, all_preds)
            precision[f'y{i + 1}'] = precision_score(all_labels, all_preds)
            recall[f'y{i + 1}'] = recall_score(all_labels, all_preds)
            accuracy[f'y{i + 1}'] = accuracy_score(all_labels, all_preds)

            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=[f"{categorical[f'y{i + 1}_neg']}", f"{categorical[f'y{i + 1}_pos']}"],
                        yticklabels=[f"{categorical[f'y{i + 1}_neg']}", f"{categorical[f'y{i + 1}_pos']}"],)
            plt.title(f"Confusion Matrix ({categorical[f'y{i + 1}']})")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.tight_layout()
            plt.savefig(f'{dir_path}/cm_y{i + 1}.png', dpi=600)

        return f1, precision, recall, accuracy

    f1, precision, recall, accuracy = test_model(y_test, preds, x=X_test)
    print(f"F1 Score  : f1(y1) = {f1['y1']:.3f}, f1(y2) = {f1['y2']:.3f}")
    print(f"Precision : precision(y1) = {precision['y1']:.3f}, precision(y2) = {precision['y2']:.3f}")
    print(f"Recall    : recall(y1) = {recall['y1']:.3f}, recall(y2) = {recall['y2']:.3f}")
    print(f"Accuracy  : accuracy(y1) = {accuracy['y1']:.3f}, accuracy(y2) = {accuracy['y2']:.3f}")

    metrics_test = {'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy}

    save_json(metrics_test, f'{dir_path}/metrics_test.json')
