import pathlib
import dash_table
import dash
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import torch
import pandas as pd
import pickle
from apps.Help import wordcloud_one_cluster, generate_wordcloud_one_cluster, fact_checking_assignment

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("datasets").resolve()
embeddings = torch.load(DATA_PATH.joinpath("embeddings_example.pt"))
embeddings_fact_checker = torch.load(DATA_PATH.joinpath("fact_checker_embeddings_example.pt"))
df = pd.read_csv(DATA_PATH.joinpath("example_dataset.csv"))

FACT_CHECKER_LOADER = dcc.Upload(
        id='fact_checker_data',
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
        # Allow multiple files to be uploaded
        multiple=False
    )

DATA_LOADER = dcc.Upload(
        id='data_loader',
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
        # Allow multiple files to be uploaded
        multiple=False
    )

app.layout = dbc.Row([
    dbc.Col(FACT_CHECKER_LOADER, width = 6),
    dbc.Col(DATA_LOADER, width = 6)
])
if __name__ == "__main__":
    app.run_server(debug = True)