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
        html.P("Basically, the idea is that a fact checker could decide to follow a specific hashtag on twitter,"
               " and the app will send an email to the fact checker if the tweets is highly similar to one of the"
               "clusters he has been assigned to. On this page the fact checker will log in and enter"
               " the hashtags he wants to follow."
               )
    ])
])

BODY = dbc.Container(
    [dbc.Row([dbc.Col(TEXT)], align="center", style={"marginTop": 30})
     ],
    className="mt-12",
)
layout = html.Div([BODY])