# -*- coding: utf-8 -*-

# future import statements
from __future__ import print_function
from __future__ import division

# version information
__project__ = "EXPLORE"
__author__  = "ACRI-ST"
__modifiers__ = '$Author: N. Cox $'
__date__ = '$Date: 2021-10-12 $'
__version__ = '$Rev: 1.0 $'
__license__ = '$Apache 2.0 $'

from dash import dcc
from dash import html
#import dash_core_components as dcc
#import dash_html_components as html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

#from flask_caching import Cache

import globals

from sda import sda
#from cube_cut import cube_cut
#from load_cube import load_cube
#from reddening import reddening

from tab1d import *
from tab2d import *
from tab3i import *

""" main figure layout """

layout=html.Div([
    dcc.Store(id='store-lab', storage_type='session'),
    #dcc.Store(id='1d-data-store', storage_type='session'),
    dcc.Store(id='signal'),
    dcc.Location(id='url', refresh=False),
    html.Div([
        html.Div([
            html.Pre("v1.0"),
            dcc.Link(
                html.Img(
                    src=sda.get_asset_url('gtomo_text.png'),
                    style={
                        'width':'100px',
                        'height':'auto',
                        'float':'left'
                    },
                ),
                href='https://explore-platform.eu',
                style={'display':'inline-block', 'height':'80px'},
            ),                 
            html.H3("EXPLORE: G-Tomo", 
                style={'display':'inline-block','height':'80px','float':'center','margin-left':'100px', 'margin-right':'100px', 'padding-top':'15px'},
            ),
            html.Div([
                dcc.Dropdown(
                    id='cube-dropdown',
                    options=[
                        {'label':'Resolution 25pc (sampling: 10pc - extent: 3 kpc)', 'value': 'cube1'},
                        {'label':'Resolution 50pc (sampling: 20pc - extent: 5 kpc)', 'value': 'cube2'},
                        #{'label':'Cube 3 - Best of both', 'value': 'cube3'},
                    ],
                    value='cube1',
                    clearable=False,
                    style={'width':'400px', 'font-size':'12px', 'padding-top':'10px'},                
                ),
                #html.Pre("selected cube: "),
                html.Div(id='cubeselection'),
            ], style={'display':'inline-block', 'height':'80px', 'float':'right', 'left-margin':'200px'}),
        ], style={'width':'100%', 'display':'flex', 'justify-content':'center'}),
        html.Br(),
        # html.Div([
        #     html.Button("Download CSV", id="btn_csv"),
        #     dcc.Download(id="download-dataframe-csv"),
        # ]),
        # html.Div(id='3d-log', children=[]),
        html.Div([
            dcc.Tabs(
                id='tabs', 
                value='None', 
                children=[
                    dcc.Tab(label='Extinction profiles', value='tab-1'),
                    dcc.Tab(label='Extinction maps', value='tab-2'),
                    dcc.Tab(label='Information', value='tab-3'),
                ],
            ),       
        ]),
        dcc.Loading(
            id='load-tab',
            type='circle',
            fullscreen=False,
            color='#534998',
            children=[
                html.Div(
                        id='tabs-content', style={'width': '100%', 'float': 'left'},
                        children=[],
                ),
            ],
        ),
    ]),
    html.Div(
    id='funder',
    children=[
        html.Img(
            src=sda.get_asset_url('flag_yellow_low.jpg'),
            style={
                'width':'75px',
                'height':'auto',
                'float':'left',
                'right-margin':'20px'
            },
        ),
        html.Br(),
        html.Pre("EXPLORE. This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No 101004214."),
    ], style={'width':'100%', 'float':'left', 'top-margin':'20px'}),
])


@sda.callback(
    [Output('tabs-content', 'children')],
    [Input('tabs', 'value')], [State('store-lab', 'data')]
)
def update_tab(tab, store_lab):

    #print('store content', store_lab)

    if tab == 'None':
        raise PreventUpdate

    if tab == 'tab-1':
        tab_content = html.Div([ 
                            content_1d(),
                        ], 
                        id='1d')

    elif tab == 'tab-2':
        tab_content = html.Div([
                            content_2d(),
                        ], id='2d')

    elif tab == 'tab-3':
        tab_content = html.Div([
                            content_info(),
                        ], id='3d')
    
    return [tab_content]

@sda.callback(
    [Output('tab1d-content', 'children')],
    [Input('tabs-1d', 'value')], 
)
def update_tab1d(tab):

    if tab == 'single':
        tab_content = [html.Div([search_box])]

    elif tab == 'bulk':
        tab_content = [bulk_box]

    return tab_content

@sda.callback(
    [Output('cubeselection', 'children')],
    [Input('cube-dropdown', 'value')]
)
def load_cube(cubename):
    
    ctx = dash.callback_context

    #print(cubename)

    if ctx.triggered:
        print('triggered')
        #globals.initialise(cubename)

    if cubename == 'cube1':
        selectedcube = "Cube: Resolution 25pc (sampling: 10pc - extent: 3 kpc)"
    if cubename == 'cube2':
        selectedcube = "Cube: Resolution 50pc (sampling: 20pc - extent: 5 kpc)"
    if cubename == None:
        selectedcube = 'none'

    return [html.Div(
                html.Pre(selectedcube),
                #html.Pre(str(cubename)),
                #html.Pre(str(globals.headers['resolution_values'])),
            )]


# @sda.callback(
#     [Output('3d-log', 'children')],
#     #Output("download-dataframe-csv", "data")],
#     Input("btn_csv", "n_clicks"),
#     prevent_initial_call=True,
# )
# def save3(n_clicks):
#     if n_clicks==0:
#         print('update prevented')
#         raise PreventUpdate

#     else:

#         sv = dict(content="Hello world!", filename="hello.txt")

#         err_msg = "clicks: "+str(n_clicks)
#         log = html.Div([
#                         html.Br(),
#                         html.Pre(err_msg),
#                         html.Br(),
#                     ])

#     return [log]