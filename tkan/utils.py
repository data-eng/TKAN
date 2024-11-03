import os
import json


def get_stats(df):
    """
    Compute mean, standard deviation, median, and IQR for each column in the DataFrame.

    :param df: DataFrame containing the data to compute statistics for.
    :return: Dictionary containing statistics for each column.
    """
    stats = {}

    for col in df.columns:
        series = df[col]

        mean = series.mean()
        std = series.std()
        median = series.median()
        iqr = series.quantile(0.75) - series.quantile(0.25)

        stats[col] = {
            'mean': mean,
            'std': std,
            'median': median,
            'iqr': iqr
        }

    return stats

def save_json(data, filename):
    """
    Save data to a JSON file.

    :param data: Dictionary containing the data to save.
    :param filename: Name of the file to save the data into.
    """
    with open(filename, 'w') as f:
        json.dump(data, f)

def robust_normalize(df, exclude, path):
    """
    Normalize data using robust scaling (median and IQR) from precomputed stats.

    :param df: DataFrame containing the data to normalize.
    :param exclude: List of columns to exclude from normalization.
    :param path: File path to save the computed statistics.
    :return: Processed DataFrame with normalized data.
    """
    newdf = df.copy()

    stats = get_stats(df)
    save_json(data=stats, filename=path)

    for col in df.columns:
        if col not in exclude:
            median = stats[col]['median']
            iqr = stats[col]['iqr']

            newdf[col] = (df[col] - median) / (iqr if iqr > 0 else 1)

    return newdf

def get_path(*dirs, filename):
    """
    Construct a full file path by combining directory paths and a filename.

    :param dirs: List of directory paths.
    :param filename: Name of the file.
    :return: Full path to the file.
    """
    dir_path = get_dir(*dirs)
    path = os.path.join(dir_path, filename)

    return path


def get_dir(*sub_dirs):
    """
    Retrieve or create a directory path based on the script's location and the specified subdirectories.

    :param sub_dirs: List of subdirectories to append to the script's directory.
    :return: Full path to the directory.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dir = os.path.join(script_dir, *sub_dirs)

    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir
