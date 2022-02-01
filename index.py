from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
# Connect to main app.py file
from app import app
from app import server
# Connect to your app pages
from apps import real_time, assignment, analysis, evaluation

NAVBAR = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink(
                "Tweets analysis",
                href='/apps/analysis',
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Fact checking assignment",
                href='/apps/assignment',
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "Real Time tweets analysis",
                href='/apps/real_time',
            )
        ),
    ],
    brand="Twitter disinformation analysis using clustering techniques",
    brand_href="#",
    color="dark",
    dark=True,
)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    NAVBAR,
    html.Div(id='page-content', children=[])
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/analysis':
        return analysis.layout
    if pathname == '/apps/assignment':
        return assignment.layout
    if pathname == '/apps/real_time':
        return real_time.layout
    if pathname == '/apps/evaluation':
        return evaluation.layout
    else:
        return assignment.layout


if __name__ == '__main__':
    app.run_server(debug=True)