# ReDSCOVR: Solar Storm Prediction

This project, developed during the 2023 NASA Hackathon, used Long Short Term Memory (LSTM) model to predict the Kp-index, an indicator of solar storms. The input feature of model was designed according to physical insights and Fast Fourier Transform (FFT) analysis. To interpret the model, gradient analysis and set-to-nan methods were applied to assess the importance of each input feature. For more information, please refer to our NASA Hackathon [website](https://www.spaceappschallenge.org/2023/find-a-team/redscovr/) and [demo-slides](https://github.com/John1117/ReDSCOVR/blob/main/ReDSCOVR%20Demo%20Slides.pdf).

## Features

- **Kp-Index Prediction**: Predicts solar storm activity using LSTM model.
- **Physics Insight**: Transform magnitude of magnetic field from magnetic vector data.
- **Fast Fourier Transform (FFT)**: Transform time series data to frequency-domain to capture the periodic information.
- **Nan Indicator**: An indicator to show whether data is available or not to maximize the data usage.
- **Input Importance Analysis**: Gradient and set-to-NaN methods highlight key features.

## Project Structure

- `data_processing/`: Scripts for handling and preprocessing input data, including FFT transformations.
- `model/`: LSTM model implementation for predicting Kp-index.
- `analysis/`: Scripts for gradient and set-to-NaN input importance analysis.
- `visualization/`: Functions for plotting model predictions and analysis results.

## Getting Started

### Prerequisites
- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scipy` (for FFT)
- `torch`
