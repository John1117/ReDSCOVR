# %%
import numpy as np
import pandas as pd
from preprocess import preprocess_dataframe, make_tensordata
from plot import plot_demo, plot_series
from model import train_model, LSTMModel


# %%
# Data loading
dtype = np.float64
years = range(2016, 2017)

dfs = []
for year in years:
    year_df = pd.read_csv(f'data/{year}_1min_data.csv', index_col='t', parse_dates=True, dtype=dtype)
    dfs.append(year_df) 
orig_df = pd.concat(dfs)


# %%
# Data Preprocessing
time_resolution = '15min'
split_ratio=0.75
fill_nan_val=0

train_df, test_df, train_mean, train_std = preprocess_dataframe(orig_df, pd.Timedelta(time_resolution), split_ratio, fill_nan_val)

# %%
# Transform pd.Dataframe to torch.Tensor
input_window = 24 #int(pd.Timedelta('12hr') / time_resolution)
offset = 3 #int(pd.Timedelta('2hr') / time_resolution)
label_window = 1

train_inputs, train_labels, test_inputs, test_labels = make_tensordata(
    train_df, 
    test_df, 
    label_col_names=['Kp'], 
    input_window=input_window, 
    label_window=label_window, 
    offset=offset
)

# %%
# Model initialization
hidden_size = 64
lstm_model = LSTMModel(hidden_size=hidden_size, feature_size=len(train_df.columns), input_window=input_window, label_window=label_window)

plot_series(
    test_df, 
    train_mean, 
    train_std, 
    start=-30, 
    end=None, 
    input_window=input_window, 
    label_window=label_window, 
    offset=offset, 
    model=lstm_model,
    predict_type='both'
)

# %%
# Model training
train_model(train_inputs, train_labels, test_inputs, test_labels, model=lstm_model, lr=1e-6, n_epoch=10, batch_size=1024)

plot_series(
    test_df, 
    train_mean, 
    train_std, 
    start=-300, 
    end=None, 
    input_window=input_window, 
    label_window=label_window, 
    offset=offset, 
    model=lstm_model,
    predict_type='both'
)

# %%
# Model testing
plot_series(
    test_df, 
    train_mean, 
    train_std, 
    start=-300, 
    end=None, 
    input_window=input_window, 
    label_window=label_window, 
    offset=offset, 
    model=lstm_model,
    predict_type='batch'
)

plot_demo(
    test_df, 
    train_mean, 
    train_std, 
    label_col_names=['Kp'], 
    plot_col_name='Kp',
    time_resolution=time_resolution,
    input_window=input_window, 
    label_window=label_window, 
    offset=offset,
    max_demos=1,
    model=lstm_model
)