# Python Libs.:
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense

# Local Libs.:


def build_model(steps):
    return Sequential([
            LSTM(35, return_sequences=True, input_shape=(steps, 1)),
            LSTM(35, return_sequences=True),

            LSTM(35),
            Dropout(0.25),
            Dense(1)

    ])
