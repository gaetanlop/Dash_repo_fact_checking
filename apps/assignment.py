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

# PATH = pathlib.Path(__file__).parent
# DATA_PATH = PATH.joinpath("../datasets").resolve()
#
# embeddings = torch.load(DATA_PATH.joinpath("embeddings_example.pt"))
# embeddings_fact_checker = torch.load(DATA_PATH.joinpath("fact_checker_embeddings_example.pt"))
# df = pd.read_csv(DATA_PATH.joinpath("example_dataset.csv"))

CSV_LOADER = dcc.Upload(
    id='upload-data',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files')
    ]),
    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
    },
    multiple=False
)

layout = html.Div([
    dbc.Row([
        dbc.Col(CSV_LOADER, width=12)
    ]),
    dcc.Loading(
        html.Div(id='output-users')
    ),
    dcc.Download(id="download-dataframe-csv")
])


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True
)
def func(n_clicks):
    from apps.Help import my_data
    return download.send_data_frame(my_data.to_csv, "assignments.csv")


@app.callback(
    Output('output-users', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = parse_contents(list_of_contents, list_of_names)
        if children is not None:
            return children
        else:
            return [()]
    return [()]

