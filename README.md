# ReDSCOVR: Solar Storm Prediction

This project, developed during the 2023 NASA Hackathon, used Long Short Term Memory (LSTM) model to predict the Kp-index, an indicator of solar storms. The input feature of model was designed according to physical insights and Fast Fourier Transform (FFT) analysis. To interpret the model, gradient analysis and set-to-nan methods were applied to assess the importance of each input feature. For more information, please refer to our NASA Hackathon [website](https://www.spaceappschallenge.org/2023/find-a-team/redscovr/) and [demo-slides](https://github.com/John1117/ReDSCOVR/blob/main/ReDSCOVR%20Demo%20Slides.pdf).

## Table of Content
- [Introduction](#redscovr-solar-storm-prediction)
- [Getting Start](#getting-start)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
- [References](#references)
    - [Trained model](#trained-model)
    - [Training Data](#training-data)

## Getting Start

### Prerequisites
- Python 3.x
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `torch`

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/John1117/ReDSCOVR.git
    ```

2. Install the dependencies using:
    ```bash
    cd ReDSCOVR
    pip install -r requirements.txt
    ```

### Usage
If you would like to either load my trained model or train your own LSTM model for solar storm prediction, please see `model_training_and_testing.ipynb`. I put my trained model and training data at the section [References](#references) below.

## References
These two files are located in my Google Drive. Please feel free to use them.

### Trained model
[`trained_model.zip`](https://drive.google.com/file/d/1RUGFixPMmrhhh5sQd5q_I5sWrB8M2nOL/view?usp=share_link) includes my trained LSTM model.

### Training Data
[`model_data.zip`](https://drive.google.com/file/d/1iJzjYvMGAl11gXTWroaPGZ96vhA9VGuQ/view?usp=share_link) contains the organized [DSCOVR](https://www.ngdc.noaa.gov/dscovr/portal/index.html#/) and [GFZ](https://www.gfz-potsdam.de/en/section/geomagnetism/data-products-services/geomagnetic-kp-index) data which I used to train my LSTM model.