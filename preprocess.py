import numpy as np
import pandas as pd
import torch


def resample_time_resolution(df: pd.DataFrame, time_resolution: pd.Timedelta=pd.Timedelta('1min')):
    resampled_df = df.resample(rule=time_resolution).mean()
    return resampled_df


def split_dataframe(df: pd.DataFrame, split_ratio=0.75):
    data_len = len(df)
    
    if isinstance(split_ratio, float):
        split_ratio = np.clip(split_ratio, 0, 1)
        split_idx = int(data_len * split_ratio)
        train_df = df[:split_idx]
        test_df = df[split_idx:]
        return train_df, test_df
    
    elif isinstance(split_ratio, list):
        split_ratio = np.array(split_ratio) / np.sum(split_ratio)

        if len(split_ratio) == 2:
            split_idx = int(data_len * split_ratio[0])
            train_df = df[:split_idx]
            test_df = df[split_idx:]
            return train_df, test_df
        
        elif len(split_ratio) == 3:
            split_idx1 = int(data_len * split_ratio[0])
            split_idx2 = -int(data_len * split_ratio[2])
            train_df = df[:split_idx1]
            valid_df = df[split_idx1:split_idx2]
            test_df = df[split_idx2:]
            return train_df, valid_df, test_df


def standardize_dataframe(train_mean: pd.Series, train_std: pd.Series, train_df: pd.DataFrame, test_df: pd.DataFrame, valid_df=None):
    train_df = (train_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    if valid_df is None:
        return train_df, test_df
    else:
        valid_df = (valid_df - train_mean) / train_std
        return train_df, test_df, valid_df
    
def fill_nan(with_nan_dfs, fill_nan_value=0):
    dfs = []
    for with_nan_df in with_nan_dfs:
        if isinstance(fill_nan_value, str):
            if fill_nan_value == 'mean':
                fill_nan_value = with_nan_df.mean()
            elif fill_nan_value == 'median':
                fill_nan_value = with_nan_df.median()
        df = with_nan_df.fillna(fill_nan_value)
        dfs.append(df)
    return dfs


def preprocess_dataframe(df, time_resolution: pd.Timedelta=pd.Timedelta('1min'), split_ratio=0.75, fill_nan_value=0):
    rsp_df = resample_time_resolution(df, time_resolution)
    train_sdf, test_sdf = split_dataframe(rsp_df, split_ratio)

    train_mean = train_sdf.mean()
    train_std = train_sdf.std()
    train_zdf, test_zdf = standardize_dataframe(train_mean, train_std, train_sdf, test_sdf)

    train_df, test_df = fill_nan([train_zdf, test_zdf], fill_nan_value)
    return train_df, test_df, train_mean, train_std


def make_tensordata(train_df, test_df, input_col_names=None, label_col_names=['Kp'], input_window=1, label_window=1, offset=1):
    input_col_names = input_col_names
    if input_col_names is None:
        input_col_names = [name for name in train_df.columns]

    label_col_names = label_col_names
    if label_col_names is None:
        label_col_names = [name for name in train_df.columns]

    train_data_len = len(train_df)
    test_data_len = len(test_df)

    train_inputs = []
    train_labels = []
    for i in range(train_data_len - input_window - label_window - offset):
        input = train_df[input_col_names][i : i + input_window].to_numpy()
        label = train_df[label_col_names][i + input_window + offset - 1 : i + input_window + offset - 1 + label_window].to_numpy()
        train_inputs.append(input)
        train_labels.append(label)
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)

    test_inputs = []
    test_labels = []
    for i in range(test_data_len - input_window - label_window - offset):
        input = test_df[input_col_names][i : i + input_window].to_numpy()
        label  = test_df[label_col_names][i + input_window + offset - 1 : i + input_window + offset - 1 + label_window].to_numpy()
        test_inputs.append(input)
        test_labels.append(label)
    test_inputs = np.array(test_inputs)
    test_labels = np.array(test_labels)

    return torch.tensor(train_inputs), torch.tensor(train_labels), torch.tensor(test_inputs), torch.tensor(test_labels)