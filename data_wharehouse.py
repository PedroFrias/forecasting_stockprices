# Python Libs.:
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pickle

# Local Libs.:
from utils import get_data

"""
Section dedicated for all the data handling
"""

MODEL = load_model('model/stocks_values_predictions6220.h5')
DATE, CLOSE = None, None


def predict_values(delta=0):

    """
    forecast a prediction for the stock value for the next n days using a LSTM Neural Network over a
    standard normalizated - f(x) = y, where y = [-1, 1] -. The output is a compose version of the real
    values of the last days with the predicted valeus for the next days.
    """

    ## globlas, variables, constants and objects
    #globals
    global DATE
    global CLOSE

    # variables
    date = datetime.today() - timedelta(1 + delta)

    # constants
    N_STEPS, N_PREDICTIONS = 36, 3

    # object
    predictions = []
    scaler = StandardScaler()

    ## Hadling inputs
    # Load data from Yahoo Finance - last 100 days
    inputs = get_data('PETR4.SA', [date - timedelta(100), date], '1d')

    # Dataframe manipulations
    length = len(inputs)
    dates = pd.to_datetime(inputs['Date'])

    # dates forecast
    dates = pd.date_range(
        list(dates)[-1] + pd.DateOffset(1),
        periods=N_PREDICTIONS,
        freq='b'
    ).tolist()

    inputs = inputs[['Date', 'Close']].set_index(
        pd.DatetimeIndex(inputs['Date'].values)
    )  # sets 'Date' collunm as index (1)
    inputs = inputs.drop('Date', axis=1)  # since (1), 'Date' is now redundant - drops.

    DATE = dates[0]
    CLOSE = inputs['Close']

    # the last 15 values will be shown couple with the predictions
    outputs = pd.DataFrame(inputs.tail(15))
    # Real values markers
    outputs['Color'] = 'white'
    outputs['Fill'] = 'white'

    # tempering data for the model
    inputs = scaler.fit_transform(inputs).transpose()[:, length-N_STEPS:length]
    inputs = inputs.tolist()

    # predicting the next N_PREDICTIONS values
    for i in range(0, N_PREDICTIONS):
        prediction, inputs = make_prediction(
            MODEL,
            np.array(inputs)
        )
        predictions.append(scaler.inverse_transform(prediction)[0][0])

    # Predicted values markers
    color = 'lime' if predictions[-1] - predictions[0] > 0 else 'fuchsia'
    fill = 'palegreen' if predictions[-1] - predictions[0] > 0 else 'violet'

    predictions = pd.DataFrame({
        'Date': np.array(dates),
        'Close': np.array(predictions),
        'Color': color,
        'Fill': fill
    })
    predictions = predictions.set_index(
        pd.DatetimeIndex(predictions['Date'].values)
    )
    predictions = predictions.drop('Date', axis=1)

    outputs = [outputs, predictions]
    outputs = pd.concat(outputs)


def make_prediction(model, inputs):
    """

    :param model:
    LSMT Neureal Network model

    :param inputs:
    A list with the last N valeu used to predict N + 1

    :return
    A list where the first value is removed and the predicted one is appended - output(N) > input(N+1).
    """

    # predicts the next value
    prediction = model.predict(inputs, verbose=0)

    # extract the first value and append the new one
    output = inputs.tolist()[0]
    output.pop(0)
    output.append(prediction[0][0])

    return prediction, [output]


def performance():
    """
    Calcs the absolute error between a predicted value and real one over the last N days. With this is possible
    to see if the network is underperfoming. Note.: sudden fluctuations may contribute with the error, yet it doesn't mean
    it need to be refitted - external forces can't be comprehend with this model.
    """
    global CLOSE

    with open("data/predictions.pickle", "rb") as data:
        data = pickle.load(data)

    # data = np.array(data).transpose()
    # dates = np.array(data[0].tolist()
    # values = data[1].astype(float)

    if DATE != data[-1][0]:

        dataframe = pd.DataFrame()

        predictions = np.array(data).transpose()
        predictions = predictions[1].astype(float)
        length = [len(predictions), len(CLOSE)]

        absolute_error = abs(predictions[length[0] - length[1]:length[0]] - CLOSE)
        absolute_error = np.round(absolute_error, 1)
        dataframe['Error'] = absolute_error
        occourence = dataframe['Error'].value_counts()

        occourence.to_csv("data/performance.csv")


if __name__ == '__main__':
    predict_values()
    performance()










