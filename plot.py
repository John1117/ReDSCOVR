import numpy as np
import matplotlib.pyplot as plt
import torch as tc
from torch import nn


def plot_demo(
        df, 
        train_mean,
        train_std,
        input_col_names=None, 
        label_col_names=['Kp'], 
        plot_col_name='Kp',
        time_resolution='1min',
        input_window=1, 
        label_window=1, 
        offset=0, 
        max_demos=3, 
        model=None
    ):

    data_len = len(df)
    max_demos = min(max_demos, data_len)

    demo_indices = np.random.choice(data_len, size=max_demos, replace=False)

    input_col_names = input_col_names
    if input_col_names is None:
        input_col_names = [name for name in df.columns]

    label_col_names = label_col_names
    if label_col_names is None:
        label_col_names = [name for name in df.columns]

    plot_col_name = plot_col_name
    if plot_col_name is None:
        plot_col_name = label_col_names[0]

    mean = train_mean[plot_col_name]
    std = train_std[plot_col_name]

    if model is not None:
        inputs = []
        for i in demo_indices:
            input = df[input_col_names][i:i+input_window].to_numpy()
            inputs.append(input)
        inputs = tc.tensor(np.array(inputs))
        preds = model(inputs).detach().numpy() * std + mean

    input_indices = np.arange(-input_window + 1, 0 + 1)
    label_indices = np.arange(offset, offset + label_window)
    total_indices = np.arange(-input_window+1, offset+label_window)

    series = df[plot_col_name] * std + mean

    for j, i in enumerate(demo_indices):
        plt.figure(figsize=(20, 7.5))
        plt.plot(input_indices, series[i:i+input_window], 'k.', ms=20, label='Input')
        if plot_col_name in label_col_names:
            plt.plot(label_indices, series[i+input_window+offset:i+input_window+offset+label_window], 'ko', mfc='w', ms=20, label='Label')
            if model is not None:
                plt.plot(label_indices, preds[j, :, :], 'b.', ms=20, label='Pred')
        plt.xticks(ticks=total_indices, labels=total_indices, fontsize=20)
        plt.yticks(ticks=range(10), labels=range(10), fontsize=20)
        plt.xlabel(f'Time shift (x{time_resolution})', fontsize=30)
        plt.ylabel(plot_col_name, fontsize=30)
        
        plt.legend(fontsize=20) 
        plt.grid()
        plt.show()


def plot_series_prediction(df, input_col_names, mean, std, model, start_index, end_index, input_window, label_window, offset):
    loss_fn = nn.MSELoss()
    input_indices = slice(start_index, end_index - offset - label_window + 1)
    inputs = tc.tensor(df[input_col_names][input_indices].to_numpy()).unsqueeze(0)

    predict_indices = slice(start_index + offset, end_index)
    labels = tc.tensor(df['Kp'][predict_indices].to_numpy()) #.unsqueeze(0)
    predicts = model(inputs, return_series=True).squeeze()
    loss = loss_fn(predicts, labels)

    predict_ts = df.index[predict_indices]
    predicts = predicts.detach().numpy() * std + mean
    plt.plot(predict_ts, predicts, 'b-', label='Series prediction')
    return np.sqrt(loss.squeeze().detach().numpy())

def plot_batch_prediction(df, input_col_names, mean, std, model, start_index, end_index, input_window, label_window, offset):
    loss_fn = nn.MSELoss()
    predict_start_indices = range(start_index, end_index - offset - label_window + 1)
    predict_ts = []
    predicts = []
    loss_running_sum = 0
    for i in predict_start_indices:
        predict_index = slice(i + input_window + offset - 1, i + input_window + offset - 1 + label_window)
        predict_t = df.index[predict_index]
        predict_ts.append(predict_t)

        input = tc.tensor(df[input_col_names][i : i + input_window].to_numpy()).unsqueeze(0)
        label = tc.tensor(df['Kp'][predict_index].to_numpy()).squeeze()
        predict = model(input, return_series=False).squeeze()

        loss = loss_fn(predict, label)
        loss_running_sum += loss
        predict_pt = predict.detach().numpy() * std + mean
        predicts.append(predict_pt)
    plt.plot(predict_ts, predicts, 'g-', label='Batch prediction')
    return np.sqrt(np.sum(loss_running_sum.squeeze().detach().numpy())/len(predict_start_indices))

def plot_series(
        df,
        train_mean,
        train_std,
        input_col_names=None, 
        label_col_names=['Kp'], 
        plot_col_name='Kp', 
        start=None, 
        end=None,
        input_window=1, 
        label_window=1, 
        offset=0, 
        model=None,
        predict_type='both',
    ):
    
    data_len = len(df)

    input_col_names = input_col_names
    if input_col_names is None:
        input_col_names = [name for name in df.columns]

    label_col_names = label_col_names
    if label_col_names is None:
        label_col_names = [name for name in df.columns]

    plot_col_name = plot_col_name
    if plot_col_name is None:
        plot_col_name = label_col_names[0]

    start_index = 0
    if isinstance(start, str):
        start_index = df.index.get_indexer([start], method='nearest')[0]
    elif isinstance(start, int):
        if start < 0:
            start_index = start + data_len
        else:
            start_index = start
    
    end_index = data_len
    if isinstance(end, str):
        end_index = df.index.get_indexer([end], method='nearest')[0]
    elif isinstance(end, int):
        if end < 0:
            end_index = end + data_len
        else:
            end_index = end

    mean = train_mean[plot_col_name]
    std = train_std[plot_col_name]
    series = df[plot_col_name] * std + mean
    
    data_indices = slice(start_index, end_index)
    data_ts = df.index[data_indices]
    data_pts = series[data_indices]

    plt.figure(figsize=(20, 10))

    plt.plot(data_ts, data_pts, 'k-', mfc='w', label='Data')

    title = ''
    if model is not None and plot_col_name in label_col_names:
        if predict_type == 'series':
            series_rmse = plot_series_prediction(df, input_col_names, mean, std, model, start_index, end_index, input_window, label_window, offset)
            title = f'Series RMSE = {series_rmse:.7f}'
        elif predict_type == 'batch':
            batch_rmse = plot_batch_prediction(df, input_col_names, mean, std, model, start_index, end_index, input_window, label_window, offset)
            title = f'Batch RMSE = {batch_rmse:.7f}'
        elif predict_type == 'both':
            series_rmse = plot_series_prediction(df, input_col_names, mean, std, model, start_index, end_index, input_window, label_window, offset)
            batch_rmse = plot_batch_prediction(df, input_col_names, mean, std, model, start_index, end_index, input_window, label_window, offset)
            title = f'RMSE (series, batch) = ({series_rmse:.7f}, {batch_rmse:.7f})'

    plt.title(title, fontsize=20)
    plt.legend(fontsize=20)
    plt.xlabel('Time', fontsize=40)
    plt.ylabel(plot_col_name, fontsize=40)
    plt.xticks(fontsize=20, rotation=-60)
    plt.yticks(ticks=range(10), labels=range(10), fontsize=20)
    plt.ylim(-0.2, data_pts.max()+1)
    plt.grid()
    plt.show()