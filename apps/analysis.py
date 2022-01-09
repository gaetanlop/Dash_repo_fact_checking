import dash_table
from dash import dcc
from dash import html
import torch
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import pathlib
from app import app
import pickle
import dash_bootstrap_components as dbc
# from apps.Help import dim_reduction, TFIDF_emb, run_fc, score_table

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()

# df = pd.read_csv(DATA_PATH.joinpath("example_dataset.csv"))
# embeddings = torch.load(DATA_PATH.joinpath("embeddings_example.pt"))
# umap_emb = dim_reduction(embeddings)
# doc_term_matrix, vectorizer = TFIDF_emb(df)
#
# smaller = 4
# higher = 11
#
# n_clusters = [i for i in range(smaller, higher)]
# models = ["_" for i in range(higher)]
# dfs = ["_" for i in range(higher)]
# dfs_tfidf = ["_" for i in range(higher)]
# contingency_tables = ["_" for i in range(higher)]
# embeddings_plots = ["_" for i in range(higher)]
# wordcloud_urls = ["_" for i in range(higher)]
# wordcloud_urls_tfidf = ["_" for i in range(higher)]
#
# for cluster in n_clusters:
#     models[cluster], dfs[cluster], contingency_tables[cluster], embeddings_plots[cluster], wordcloud_urls[
#         cluster] = run_fc(df, umap_emb=umap_emb, n_clusters=cluster)
#     dfs_tfidf[cluster], wordcloud_urls_tfidf = run_fc(df, doc_term_matrix=doc_term_matrix, n_clusters=cluster,
#                                                       vectorizer=vectorizer)
#
# scores_bert = score_table(dfs, my_clusters=[i for i in range(smaller, higher)], embeddings=umap_emb)
# scores_tfidf = score_table(dfs_tfidf, my_clusters=[i for i in range(smaller, higher)], doc_term_matrix=doc_term_matrix)
# scores_bert = scores_bert.reset_index().rename({"index": "Metrics"}, axis=1)
# scores_tfidf = scores_tfidf.reset_index().rename({"index": "Metrics"}, axis=1)

# for a faster implementation we will directly load the files
scores_bert = pd.read_csv(DATA_PATH.joinpath("scores_bert.csv"))
scores_tfidf = pd.read_csv(DATA_PATH.joinpath("scores_tfidf.csv"))
with open(DATA_PATH.joinpath("wordcloud_urls"), "rb") as fp:   # Unpickling
    wordcloud_urls = pickle.load(fp)


SBert_cluster = [
    dbc.CardHeader(html.H5("SBert clustering")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                [   html.P("Metrics depending on the number of clusters"),
                                    dash_table.DataTable(
                                        id='table',
                                        columns=[{"name": i, "id": i} for i in scores_bert.columns],
                                        data=scores_bert.to_dict('records'),
                                        style_table={'overflowX': 'auto'}
                                    )
                                ],
                                md=12
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.P("Choose the appropriate number of clusters"),
                                    dcc.Slider(
                                                id='slider_ncluster',
                                                min=4,
                                                max=10,
                                                marks={str(i): str(i) for i in [i for i in range(4, 11)]},
                                                value=4),
                                    html.Img(id='wordcloud_image_SBert', src="", width = "100%")
                                ],
                                md=12,
                            ),

                        ],
                    ),

                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

TFIDF_cluster = [
    dbc.CardHeader(html.H5("TFIDF clustering")),
    dbc.CardBody(
        [
            dcc.Loading(
                id="",
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                [   html.P("Metrics depending on the number of clusters"),
                                    dash_table.DataTable(
                                        id='table',
                                        columns=[{"name": i, "id": i} for i in scores_tfidf.columns],
                                        data=scores_tfidf.to_dict('records'),
                                        style_table={'overflowX': 'auto'}
                                    )
                                ],
                                md=12
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.P("Choose the appropriate number of clusters"),
                                    dcc.Slider(
                                                id='slider_ncluster_tfidf',
                                                min=4,
                                                max=10,
                                                marks={str(i): str(i) for i in [i for i in range(4, 11)]},
                                                value=4),
                                    html.Img(id='wordcloud_image_TFIDF', src="", width = "100%")
                                ],
                                md=12,
                            ),

                        ],
                    ),

                ],
                type="default",
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

MY_LOADER = dcc.Upload(
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
        # Allow multiple files to be uploaded
        multiple=False
    )


BODY = dbc.Container(
    [
        dbc.Row([dbc.Col(MY_LOADER),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(SBert_cluster)),], style={"marginTop": 30}),
        dbc.Row([dbc.Col(dbc.Card(TFIDF_cluster)),], style={"marginTop": 30}),
    ],
    className="mt-12",
)
layout = html.Div([BODY])

# layout = html.Div([
#     html.H1('Twitter Disinformation analysis'),
#     html.H4('Number of clusters'),
#     html.H4('SBert Analysis'),
#     dcc.Slider(
#         id='slider_ncluster',
#         min=4,
#         max=10,
#         marks={str(i): str(i) for i in [i for i in range(4, 11)]},
#         value=4),
#     dash_table.DataTable(
#         id='table',
#         columns=[{"name": i, "id": i} for i in scores_bert.columns],
#         data=scores_bert.to_dict('records') ),
#     html.Img(id='wordcloud_image_SBert', src=""),
#     html.H4('TF-IDF Analysis')
# ])

@app.callback(
    Output('wordcloud_image_SBert', 'src'),
    Input('slider_ncluster', 'value'))
def update_figure(n_cluster):
    return wordcloud_urls[n_cluster]

@app.callback(
    Output('wordcloud_image_TFIDF', 'src'),
    Input('slider_ncluster_tfidf', 'value'))
def update_figure(n_cluster):
    return wordcloud_urls[n_cluster]
