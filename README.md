## About this app

#### Main routine: 
continuously queries PETR4's stock prices dataset to show its values during the business hours.

#### Daily routine: 
queries data over a set period of time:
- to forecast its values over the next days using a LSTM Neural Network (LNN).
- execute a performance anaisys based on its absolute value.

#### Occasional:
if the LSTM Neural Network is underperforming, it might need to be retrained. This routine doens't run atomaticaly, it need to be executed by ruinning the command:
```
python machine_learning_model.py
```

##### Note:
suden fluctuatios on the market caused by external forces may contribute to the erro, yet, it doesnt mean the network is underperforming, since its base on the closing value, its possible that it just need time to adjust.

 

## How to run this app
(The following instructions apply to Posix/bash. Windows users should check [here](https://docs.python.org/3/library/venv.html).)

First, clone this repository and open a terminal inside the root folder, create and activate a new virtual environment (recommended) by running the following:
```
python3 -m venv myvenv
source myvenv/bin/activate
```
Install the requirements:
```
pip install -r requirements.txt
```
Run the app:
```
python app.py
```

## Graphics and plots
#### 1.
![image_1.png](https://github.com/PedroFrias/forecasting_stockprices/blob/main/assets/imgs/image_1.png)
Candlestick showing PETR4's price changes during business hours. Updates every minute.

#### 2.
![image_2.png](https://github.com/PedroFrias/forecasting_stockprices/blob/main/assets/imgs/image_2.png)
Error between the predicted and actual value over the last N samples.

#### 3.
![image_2.png](https://github.com/PedroFrias/forecasting_stockprices/blob/main/assets/imgs/image_3.png)
Predictions for the next days.

Open a browser at http://127.0.0.1:8050
