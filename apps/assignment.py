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

TEXT = html.Div([
    html.H1('Page description'),
    html.Div([
        html.P("The fact checker can now input its profile (a csv file with all the articles he has written, and all"
               " the tweets he has worked on). Based on that and the document dropped in the page"
               " 'Tweets analysis', the fact checker will be assigned a specific cluster (the one that is the most"
               " related to its knowledge). You can try this page by putting a fake csv file (one is already "
               "downloaded in the app for demonstration purposes). It returns a csv file with all the tweets assigned"
               " to the fact checker."
               )
    ])
])

BODY = dbc.Container([
    dbc.Row([
        dbc.Col(TEXT, width=12)
    ], style={"marginTop": 30}),
    dbc.Row([
        dbc.Col(CSV_LOADER, width=12)
    ], style={"marginTop": 30}),
    dcc.Loading(
        html.Div(id='output-users')
    ),
    dcc.Download(id="download-dataframe-csv")
], className="mt-12")

layout = html.Div([BODY])


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

