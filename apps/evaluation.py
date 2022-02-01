import pathlib
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import torch
import pandas as pd
import dash.dcc as download
from apps.Help import parse_contents
from app import app
import dash_bootstrap_components as dbc

TEXT = html.Div([
    html.H1('Work in progress'),
    html.Div([
        html.P("We will use this page to evaluate the performance of our fact checking assignment."
               " We will compare the results of our clustering assignments with what the fact checkers"
               " would have decided to be assigned to. "
               )
    ])
])

BODY = dbc.Container(
    [dbc.Row([dbc.Col(TEXT)], align="center", style={"marginTop": 30})
     ],
    className="mt-12",
)
layout = html.Div([BODY])