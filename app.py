# Python Libs.:
import os
import pathlib
import numpy as np
import pandas as pd
from datetime import date, timedelta, datetime
import dash
from dash import dcc
from dash import html
from plotly.graph_objects import *
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from scipy.stats import rayleigh
import pickle

# Local Libs.:
from utils import get_data, open_file


GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 1000 * 60 * 2)


def frequencies():

    dataframe = pd.read_csv("data/performance.csv")

    fig = dcc.Graph(
        id='frequency_series-graph',
        figure=Figure(
            data=[
                Bar(
                    x=dataframe["Unnamed: 0"],
                    y=dataframe["Error"] / sum(dataframe['Error']),
                    marker={
                        'color': 'violet',
                        'line': {
                            'color': 'fuchsia',
                            'width': 0.5
                        }
                    },
                )
            ],

            layout=Layout(
                height=400,
                paper_bgcolor='#31343a',
                plot_bgcolor='#31343a',
                xaxis={
                    'color': 'white',
                    'showgrid': False,
                    'tickformat': '.1f',
                    "zeroline": False,
                },
                yaxis={
                    'color': 'white',
                    'showgrid': False,
                    'tickformat': '.0%',
                    "zeroline": False
                }
            )
        )
    )

    return fig


def predict():

    dataframe = pd.read_csv("data/predictions.csv")

    fig = dcc.Graph(
        id='predic_stocks-graph',
        figure=Figure(
            data=[
                Scatter(
                    x=dataframe['Unnamed: 0'],
                    y=dataframe['Close'],
                    marker={
                        'color': dataframe['Color'],
                        'size': 10
                    },
                    line={
                        'color': "white"
                    },
              )
            ],

            layout=Layout(
                height=550,
                paper_bgcolor='#31343a',
                plot_bgcolor='#31343a',
                xaxis={
                    'color': 'white',
                    'showgrid': False,
                    'tickformat': '%d/%m',
                    "zeroline": False
                },
                yaxis={
                    'color': 'white',
                    'showgrid': False,
                    'tickformat': '.2f',
                    "zeroline": False
                }
            )
        )
    )

    return fig


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "PETR4 Live"

server = app.server

app.layout = html.Div(
    [
        # header
        html.Div(
            [
                html.Div(
                    [
                        html.H4(f"PETROBRÁS (PETR4)", className="app__header__title"),
                        html.P(
                            "Esta aplicação coleta continuamente os dados das ações da PETROBRÁS para fazer previsões usando um rede neural LSTM.",
                            className="app__header__title--grey",
                            style={"color": "white", "font-size": 14}
                        ),
                    ],
                    className="app__header__desc",
                ),
                html.Div(
                    [
                        html.A(
                            html.Button("SOURCE CODE", className="link-button", style={"color": "white"}),
                            href="https://github.com/PedroFrias",
                        ),
                        html.A(
                            html.Button("LINKED.IN", className="link-button", style={"color": "white"}),
                            href="https://www.linkedin.com/in/pedro-henrique-abem-athar-frias-a48526119/",
                        ),
                    ],
                    className="app__header__logo",
                ),
            ],
            className="app__header",
        ),
        html.Div(
            [
                # live stocks
                html.Div(
                    [
                        html.Div(
                            [html.H6("PREÇO DAS AÇÕES", className="graph__title")]
                        ),

                        dcc.Graph(
                            id="live_stocks-graph"
                        ),

                        dcc.Interval(
                            id="live_stocks-update",
                            interval=int(GRAPH_INTERVAL),
                            n_intervals=0,
                        ),
                    ],
                    className="two-thirds column live__stocks__container",
                ),

                html.Div(
                    [
                        # correlation
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "ERRO ABSOLUTO DA REDE",
                                            className="graph__title",
                                        )
                                    ]
                                ),
                                html.Div(
                                    children=frequencies(),
                                )
                            ],
                            className="graph__container first",
                        ),

                        # prediction
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            f"PREVISÃO PARA OS PRÓXIMOS DIAS", className="graph__title"
                                        )
                                    ]
                                ),
                                html.Div(
                                    children=predict(),
                                ),
                            ],
                            className="graph__container second",
                        ),
                    ],
                    className="one-third column histogram__direction",
                ),
            ],
            className="app__content",
        ),
    ],
    className="app__container",
)


@app.callback(
    Output(
        component_id="live_stocks-graph",
        component_property="figure"
    ),
    Input(
        component_id="live_stocks-update",
        component_property="n_intervals"
    )
)
def livestocks_candlestickChart(interval):

    date_final_point = datetime.now()

    # account for business hours
    if datetime.now().hour < 7:
        date_final_point = datetime.today() - timedelta(1)
        date_final_point = date_final_point.replace(hour=18, minute=1)

    date_start_point = date_final_point.replace(hour=7, minute=0)

    dataframe = get_data('PETR4.SA', [date_start_point, date_final_point], '1m')

    fig = Figure(
        data=[
            Candlestick(
                x=dataframe['Datetime'],
                open=dataframe['Open'],
                high=dataframe['High'],
                low=dataframe['Low'],
                close=dataframe['Close'],
                increasing={'line': {'color': 'lime'}},
                decreasing={'line': {'color': 'fuchsia'}}
        )],
        layout=Layout(
            height=1000,
            paper_bgcolor='#31343a',
            plot_bgcolor='#31343a',
            xaxis={'color': 'white', 'showgrid': False, 'tickformat': '%d/%m %H:%M', "zeroline": True},
            yaxis={'color': 'white', 'showgrid': False, 'tickformat': '.1f'}
        )
    )

    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
