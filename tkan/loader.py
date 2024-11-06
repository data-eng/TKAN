import pandas as pd
import numpy as np
import os
import glob
import mne
import ast

from collections import Counter
from .utils import get_path, robust_normalize, save_json


def get_boas_data(base_path, output_path):
    """
    Retrieve and combine EEG and event data for subjects from the Bitbrain dataset. For each subject, 
    this function reads EEG signals from an EDF file and event data from a TSV file, combines them, 
    and saves the result as a CSV file in the specified output directory.

    :param base_path: Path to the base directory containing subject folders.
    :param output_path: Path to the directory where combined data will be saved.
    """
    for subject_folder in glob.glob(os.path.join(base_path, 'sub-*')):
        subject_id = os.path.basename(subject_folder)
        eeg_folder = os.path.join(subject_folder, 'eeg')

        output_file = os.path.join(output_path, f'{subject_id}.csv')

        if os.path.exists(output_file):
            continue

        if not os.path.exists(eeg_folder):
            print(f'No EEG folder found for {subject_id}. Skipping.')
            continue

        eeg_file_pattern = os.path.join(eeg_folder, f'{subject_id}_task-Sleep_acq-headband_eeg.edf')
        events_file_pattern = os.path.join(eeg_folder, f'{subject_id}_task-Sleep_acq-psg_events.tsv')

        try:
            raw = mne.io.read_raw_edf(eeg_file_pattern, preload=True)
            x_data = raw.to_data_frame()

            print(f'x_data shape for {subject_id}: {x_data.shape}')
            print(f'x_data sample:\n{x_data.head()}')

        except Exception as e:
            print(f'Error loading EEG data for {subject_id}: {e}')
            continue

        try:
            y_data = pd.read_csv(events_file_pattern, delimiter='\t')

            print(f'y_data shape for {subject_id}: {y_data.shape}')
            print(f'y_data sample:\n{y_data.head()}')

        except Exception as e:
            print(f'Error loading events data for {subject_id}: {e}')
            continue

        y_expanded = pd.DataFrame(index=x_data.index, columns=y_data.columns)

        for _, row in y_data.iterrows():
            begsample = row['begsample'] - 1
            endsample = row['endsample'] - 1

            y_expanded.loc[begsample:endsample] = row.values

        combined_data = pd.concat([x_data, y_expanded], axis=1)

        combined_data.to_csv(output_file, index=False)
        print(f'Saved combined data for {subject_id} to {output_file}')


def split_data(dir, train_size=57, test_size=1):
    """
    Split the CSV files into training, and test sets.

    :param dir: Directory containing the CSV files.
    :param train_size: Number of files for training.
    :param test_size: Number of files for testing.
    :return: Tuple of lists containing CSV file paths for train and test sets.
    """
    paths = [get_path(dir, filename=file) for file in os.listdir(dir)]
    print(f'Found {len(paths)} files in directory: {dir} ready for splitting.')

    train_paths = paths[:train_size]
    test_paths = paths[train_size:train_size + test_size]

    print(f'Splitting complete!')

    return (train_paths, test_paths)


def load_file(path):
    """
    Load data from a CSV file.
    :param path: Path to the CSV file.
    :return: Tuple (X, t, y) where X contains EEG features, t contains time, and y contains labels.
    """
    df = pd.read_csv(path)

    X = df[['HB_1', 'HB_2']].values
    t = df['time'].values
    y = df['majority'].values

    return X, t, y


def combine_data(paths, seq_len=240, normalize=False):
    """
    Combine data from multiple CSV files into a dataframe, processing sequences and removing invalid rows.

    :param paths: List of file paths to CSV files.
    :param seq_len: Sequence length for grouping data.
    :return: Combined dataframe after processing.
    """
    dataframes = []
    total_removed_majority = 0

    print(f'Combining data from {len(paths)} files.')

    for path in paths:
        X, _, y = load_file(path)

        df = pd.DataFrame(X, columns=['HB_1', 'HB_2'])
        df['majority'] = y

        df['seq_id'] = (np.arange(len(df)) // seq_len) + 1
        df['night'] = int(os.path.basename(path).split('-')[1].split('.')[0])

        rows_before_majority_drop = df.shape[0]
        df.drop(df[df['majority'] == 8].index, inplace=True)
        total_removed_majority += (rows_before_majority_drop - df.shape[0])

        dataframes.append(df)

    df = pd.concat(dataframes, ignore_index=True)
    print(f'Combined dataframe shape: {df.shape}')

    print(f'Removed {total_removed_majority} rows with majority value -1.')
    
    rows_before_nan_drop = df.shape[0]
    df.dropna(inplace=True)
    print(f'Removed {rows_before_nan_drop - df.shape[0]} rows with NaN values.')

    assert not df.isna().any().any(), 'NaN values found in the dataframe!'

    stats_path = get_path('..', '..', 'data', filename='stats.json')
    if normalize:
        df = robust_normalize(df, exclude=['night', 'seq_id', 'time', 'majority'], path=stats_path)

    return df


def get_dataframe(path, seq_len, name, exist):
    """
    Create or load dataframes for training, and testing.

    :param path: File path for training, and testing.
    :param exist: Boolean flag indicating if the dataframes already exist.
    :param name: Split name (e.g. train, test)
    :return: Tuple of dataframes for train, validation, and test sets.
    """
    weights = None

    print(f'Creating dataframe for {name}ing.')

    proc_path = get_path('..', 'data', 'proc', filename=f'{name}.csv')

    if exist:
        # df = pd.read_csv(proc_path)
        df = pd.read_json(proc_path)
        print(f'Loaded existing dataframe from {proc_path}.')
    else:
        df = combine_data(path, seq_len)

        # Convert integer columns to smallest possible type
        for col in df.select_dtypes(include='int').columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')

        # Convert float columns to smallest possible type
        for col in df.select_dtypes(include='float').columns:
            df[col] = pd.to_numeric(df[col], downcast='float')

        df = df.groupby(['seq_id', 'night']).agg(list).reset_index()

        def most_common(lst):
            return Counter(lst).most_common(1)[0][0]

        df['majority_class'] = df['majority'].apply(most_common).astype(int)

        df = df.drop(['seq_id', 'night', 'majority'], axis=1)

        # df.to_csv(proc_path, index=False)
        df.to_json(proc_path, orient='records')
        print(f'Saved {name} dataframe to {proc_path}.')

    print(f'Dataframes for {name}ing are ready!')

    return df


def extract_weights(df, label_col):
    """
    Calculate class weights from the training dataframe to handle class imbalance, and save them to a file.

    :param df: Dataframe containing the training data.
    :param label_col: The name of the column containing class labels.
    :return: A tuple containing a dictionary of class weights and a list of class labels if mapping is enabled.
    """
    occs = df[label_col].value_counts().to_dict()
    inverse_occs = {key: 1e-10 for key in occs.keys()}

    for key, value in occs.items():
        inverse_occs[int(key)] = 1 / (value + 1e-10)

    weights = {key: value / sum(inverse_occs.values()) for key, value in inverse_occs.items()}
    weights = dict(sorted(weights.items()))

    new_weights = {i: weights[key] for i, key in enumerate(weights.keys())}

    path = get_path('..', '..', 'data', filename='weights.json')
    save_json(data=new_weights, filename=path)

    return weights, new_weights


def create_df(df):
    """
    Create dataset for the specified dataframes (e.g. training, testing).
    :param df: Dataframe.
    :return: Dataset.
    """

    X = ['HB_1', 'HB_2']
    y = ['majority_class']

    X_train, y_train = df[X], df[y]

    '''
    X_train['HB_1'] = X_train['HB_1'].apply(lambda x: np.array(ast.literal_eval(x)))
    X_train['HB_2'] = X_train['HB_2'].apply(lambda x: np.array(ast.literal_eval(x)))
    '''
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()

    X_train = np.stack([np.stack(row) for row in X_train])
    y_train = np.stack([np.stack(row) for row in y_train])

    print(f'Dataset created successfully!')

    return X_train, y_train
