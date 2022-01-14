import pathlib
import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
import torch
import pandas as pd
import base64
import io
import plotly.graph_objects as go
import dash.dcc as download

from apps.Help import fact_checking_assignment

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("datasets").resolve()
embeddings = torch.load(DATA_PATH.joinpath("embeddings_example.pt"))
embeddings_fact_checker = torch.load(DATA_PATH.joinpath("fact_checker_embeddings_example.pt"))
df = pd.read_csv(DATA_PATH.joinpath("example_dataset.csv"))

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

app.layout = html.Div([
    dbc.Row([
        dbc.Col(CSV_LOADER, width=12)
    ]),
    dcc.Loading(
        html.Div(id='output-users')
    ),
    dcc.Download(id="download-dataframe-csv")
])


def parse_contents(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                temp = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8', errors='ignore')))
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                temp = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])
        global my_data
        my_cluster, my_data, my_fig = fact_checking_assignment(embeddings, embeddings_fact_checker, df, n_clusters=5)
        plot_number = go.Figure(go.Indicator(value=my_cluster, title="You have been assigned to cluster number"))

        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(id='my_number', figure=plot_number), width=6),
                dbc.Col([
                    html.H5("Cluster Explanability"),
                    html.Img(id='my_wordcloud', src=my_fig, width="100%")
                ],
                    width=6,
                    align="center")
            ]),
            dbc.Row([
                html.Button("Download CSV", id="btn_csv"),
            ])
        ])

    else:
        return [()]


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True
)
def func(n_clicks):
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


if __name__ == "__main__":
    app.run_server(debug=True)
