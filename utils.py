import yfinance as yf
import pickle
import pandas as pd


def get_data(share, period, periodicity):
    ticket = yf.Ticker(share)
    dataframe = pd.DataFrame(ticket.history(
        interval=periodicity,
        start=period[0],
        end=period[1]
    ))

    dataframe = dataframe[dataframe['Close'] > 0]

    dataframe = dataframe.reset_index()
    for i in ['Open', 'High', 'Close', 'Low']:
        dataframe[i] = dataframe[i].astype('float64')
    return dataframe


def open_file(file):
    with open(file, "rb") as un_pickle:
        return pickle.load(un_pickle)


