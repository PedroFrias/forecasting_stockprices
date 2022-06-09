# Python Libs.:
import pandas as pd
import numpy as np
import os.path
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.optimizers.optimizer_v2.adam import Adam
from keras import backend as bk
from random import random

# Local Libs.:
from utils import get_data
from models import build_model
from datetime import timedelta, datetime


def main(train=False):
    ## Load data
    date = datetime.today()
    dataframe = get_data('PETR3.SA', [date - timedelta(365 * 10), date], '1d')
    # dataframe['Date'] = pd.to_datetime(dataframe['Date'], format='%Y-%m-%d')
    data = np.array(dataframe['Close'])

    ## pre processing: data frequency using FFT
    line_count = len(dataframe)

    ## Dataframe manipulations
    dataframe_diff = dataframe[['Date', 'Close']].set_index(pd.DatetimeIndex(dataframe['Date'].values))
    dataframe_diff = dataframe_diff.drop('Date', axis=1)

    # Data normalization
    scaler = StandardScaler()
    dataframe_scaled = scaler.fit_transform(dataframe_diff)

    line_count_train = round(line_count * 0.75)
    train = dataframe_scaled[:line_count_train]
    test = dataframe_scaled[line_count_train:line_count]

    steps = 30
    inputs_train, outputs_train = split_dataframe(train, steps)
    inputs_test, outputs_test = split_dataframe(test, steps)

    inputs_train = inputs_train.reshape(inputs_train.shape[0], inputs_train.shape[1], 1)
    inputs_test = inputs_test.reshape(inputs_test.shape[0], inputs_test.shape[1], 1)

    # Traning
    model = build_model(steps)
    model.summary()
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['accuracy']
    )

    validation = model.fit(
        inputs_train,
        outputs_train,
        validation_data=(
            inputs_test,
            outputs_test
            ),
        epochs=100,
        batch_size=15,
        verbose=2
        )

    model.save(f'model/stocks_values_predictions{int(random()*10000)}.h5')


def split_dataframe(dataframe, steps):
    data_train, data_test = [], []

    for i in range(len(dataframe) - steps - 1):
        to_train = dataframe[i:(i + steps), 0]
        data_train.append(to_train)
        data_test.append(dataframe[i + steps, 0])

    return np.array(data_train), np.array(data_test)


if __name__ == '__main__':
    main(True)
