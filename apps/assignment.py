from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import pathlib
from app import app

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

# owner: shivp Kaggle. Source: https://data.mendeley.com/datasets
# dataset was modified. Original data: https://www.kaggle.com/shivkp/customer-behaviour
# dfg = pd.read_csv(DATA_PATH.joinpath("..."))

layout = html.Div([
    ])