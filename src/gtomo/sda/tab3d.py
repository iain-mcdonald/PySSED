# -*- coding: utf-8 -*-

# future import statements
from __future__ import print_function
from __future__ import division
from astropy.coordinates.sky_coordinate import SkyCoord

# version information
__project__ = "EXPLORE"
__author__  = "ACRI-ST"
__modifiers__ = '$Author: N. Cox $'
__date__ = '$Date: 2021-10-12 $'
__version__ = '$Rev: 1.0 $'
__license__ = '$Apache 2.0 $'

import os
import io
import base64
import datetime
import sys
from collections import OrderedDict

import astropy.units as u
import numpy as np

import dash
#import dash_table
#import dash_html_components as html
#import dash_core_components as dcc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go

import globals

from lab import lab
#from cube_cut import cube_cut
#from load_cube import load_cube
from cube import sub_cube as subcube

""" functions """

""" creat figure to display a 3d data set (volume rendered)
"""
def create_3d_plot(cube, axes):
    import plotly.graph_objects as go
    import numpy as np

    err_message=""
    
    if (cube.size > 2e8): ## too big/small??
        err_message="size of cube too large!"
        return 
   
    #values=np.log10(cube.flatten())
    values =cube.flatten()

    X,Y,Z=np.meshgrid(axes[0],axes[1],axes[2]) # original too large. needs sub-cube!

    #x,y=np.meshgrid(sub_axes[0],sub_axes[1])
    #z=np.zeros(x.shape)

    # tune "opacityscale" is "uniform", "extremes", "min", "max"
    # possible to use custom opacity scale, as "opacityscale=[[-0.5,1], [-0.2,0], [0.2,0], [0.5, 1]]"
    # mapping scalar values to relative opacity values (between 0 and 1) the max opacity is given by
    # opacity keyword. Can be used to make some ranges completely transparent
    caps_on = False
    
    # enable/disable caps (color coded surfaces on the sides of the visualisation domain):
    if caps_on == True:
        caps=dict(x_show=True, y_show=True, z_show=True, x_fill=1)
    else:
        caps=dict(x_show=False, y_show=False, z_show=False)

    layout3d=go.Layout(
        autosize=False,
        width=800,
        height=1000,
        # margin=go.layout.Margin(
        #     l=10,
        #     r=10,
        #     b=10,
        #     t=10,
        #     pad=4,
        # )
    )

    fig = go.Figure(data=go.Volume(
                        x=X.flatten(),
                        y=Y.flatten(),
                        z=Z.flatten(),
                        value=values,
                        #isomin=np.nanmin(values),
                        #isomax=1.1*np.nanmax(values),
                        opacity=0.1,
                        opacityscale="uniform",
                        caps = caps,
                        surface_count=30,
                        #surface=[1,30],
                        colorscale='Plasma',  #RdBu
                        ),
                    layout=layout3d,
                    )
    return fig, err_message

#values3d=sub_cube
#X,Y,Z=np.meshgrid(axes[0],axes[1],axes[2])

# X, Y, Z = np.mgrid[-8:8:40j, -8:8:40j, -8:8:40j]
# values3d = np.sin(X*Y*Z) / (X*Y*Z) #will sub cube

# # enable/disable caps (color coded surfaces on the sides of the visualisation domain):
caps_on = True
if caps_on == True:
    caps=dict(x_show=True, y_show=True, z_show=True, x_fill=1)
else:
    caps=dict(x_show=False, y_show=False, z_show=False)


fig3d = go.Figure(data=go.Volume(
                    #x=X.flatten(),
                    #y=Y.flatten(),
                    #z=Z.flatten(),
                    #value=values3d.flatten(),
                    isomin=0.1,
                    isomax=0.8,
                    opacity=0.1,
                    opacityscale="uniform",
                    caps = caps,
                    surface_count=17,
                    colorscale='RdBu'
                    ),
                )



def content_3d():
    content_3d = html.Div([
        html.Div([
            html.Div([
                html.H5('Select subcube (fixed size X pc)'), # fixed size of the subcube
                html.Label('Enter Ra-Dec-Distance (galactic)', style={'margin-left':'10px'}),
                html.Br(),
                dcc.Input(id='3d-ra', type='number', placeholder="RA [-360..+360] deg", size='15'),
                dcc.Input(id='3d-dec', type='number', placeholder="DEC [-360..+360] deg", size='15'),
                #dcc.Input(id='radius', type='number', placeholder="RADIUS [0..10] arcmin", size='15'),
                dcc.Input(id='3d-distance', type='number', placeholder="DISTANCE [0..1000] pc", size='15'),
                html.Button(id='3d-get-cube', n_clicks=0, children='Extract Cube', style={'margin-left':'10px'}),
            ], style={'padding':'5px','borderWidth': '2px', 'borderStyle': 'dashed'}),
            #
            html.Div([
                html.H5('Gaia target filter'),
                html.Br(),
                html.Label('Query row limit: '),
                dcc.Input(id='Maxrows', type='number', placeholder="[rows]", size='10'),
                html.Br(),
                html.Label('G magnitude limit: '),
                dcc.Input(id='Gmag', type='number', placeholder="[Gmag]", size='10'),
                html.Br(),
                #html.Label('Distance limit: '),
                #dcc.Input(id='distance_gaia', type='number', placeholder="[mas]", size='10'),
                #html.Br(),
                html.Label('Distance uncertainty (max): '),
                dcc.Input(id='error_gaia', type='number', placeholder="[error %]", size='10'),
                html.Br(),
                html.Hr(),
                html.H6('Alternative:'),
                html.Label('Upload target list (ra-dec-distance)'),
                dcc.Upload(
                    id='upload-file',
                    children=[html.Button(['upload'])],
                    multiple=False
                ),
                html.Br(),
                html.Button(id='plot-targets', n_clicks=0, children='Plot targets'),
            ], style={'padding':'5px','borderWidth': '2px', 'borderStyle': 'dashed'}),
        ], style={'width':'30%', 'display':'inline-block', 'margin':'5px'} ),
        #
        html.Div([
            html.Div([
                html.Div([
                    html.Label('Load saved graph data'),
                    html.Button(id='load-saved-data', n_clicks=0, children='Load', style={'margin-left':'10px'}),
                ], style={'display':'inline-block'}),
                html.Div([
                    html.Label('Save graph data'),
                    html.Button(id='save-gaia-3d', n_clicks=0, children='Save', style={'margin-left':'10px'}),
                    dcc.Download(id='download-dataframe'),
                ], style={'display':'inline-block'}),
                html.Div([
                    html.Label('Connect to SAMP'),
                    html.Button(id='connect-samp', n_clicks=0, children='SAMP', style={'margin-left':'10px'}),
                ], style={'display':'inline-block'}),
                html.Div([
                    html.Label('Send to VO tool'),
                    html.Button(id='send-vo', n_clicks=0, children='Send VO', style={'margin-left':'10px'}),
                ], style={'display':'inline-block'}),
            ], style={'padding':'5px'}),
            html.Hr(),
            html.Br(),
            html.Div(id='temp3'),
            html.Br(),
            html.Div([
                html.H6("Extinction cube at x,y,z"),
            ], id='cube-coord'),
            dcc.Graph(id='3d-graph', figure=fig3d, config={"doubleClick": "reset"}),
            # 3d volume-rendering plot [xyz] of the cube with Gaia targets
            # ra-dec-distance --> xyz (cartesian cube center coordinate)
            # extract cube at xyz with size n
            # how to search for all Gaia targets in this given volume? 
            # probably simply do a box-square search centred on ra-dec coordinate 
            # and then filter for distances 'close' to the subcub center
            # Limit query result to e.g. 1000(0) results, distance filter reduced the number of targets further
        ], style={'width':'60%', 'padding':'5px', 'margin':'5px', 'display':'inline-block', 'vertical-align':'top', 'borderWidth': '2px', 'borderStyle': 'dashed'} ),
        html.Div([
        ])
            #dcc.Download(
            #    id='3d-download',
            #    children=[html.Button(['Save data'])],
            #),
            # in callback:
            # Input("button", "n_clicks")
            # Output("donwload-dataframe", "data")
            # return dcc.send_data_frame(df.to_csv, filename='some_name.csv')
            # return dcc.send_file("path_to_file")
    ])

    return content_3d


@lab.callback(
    [Output("3d-graph","figure"),
     Output('temp3', 'children'),
    ],
    [Input("3d-get-cube", "n_clicks"),
    ],
    [State("3d-ra", "value"),
     State("3d-dec", "value"),
     State("3d-distance", "value"),
    ],
)
def update_3d_graph(n_clicks,ra,dec,distance):

    if not n_clicks: #((n_clicks == 0) or (n_clicks == None)):
        raise PreventUpdate

    ctx = dash.callback_context

    if not ctx.triggered: 
        button_id = None
        raise PreventUpdate

    import json
    ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered,
        'inputs': ctx.inputs,
        #'outputs': ctx.outputs_list,
    }, indent=2)

    err_message=None

    if ra is None:
        sc=None
        err_message = "coordinate-distance not correct"
    if dec is None:
        sc=None
        err_message = "coordinate-distance not correct"
    if distance is None:
        sc=None
        err_message = "coordinate-distance not correct"
    else:
        sc="coord"
        sc = SkyCoord(ra*u.deg, dec*u.deg, frame='galactic')
        #sub_cube, sub_axes = subcube(globals.cube, globals.axes, sc=sc, step=5.0, size_pc=100.0, center_distance=distance)
        
    sub_cube = globals.cube[620:660,620:660,60:100]
    sub_axes=[globals.axes[0][620:660],globals.axes[1][620:660],globals.axes[2][60:100]]

    fig3d, err_message = create_3d_plot(sub_cube, sub_axes)

    #     # fig3d=None

    # X, Y, Z = np.mgrid[-8:8:40j, -8:8:40j, -8:8:40j]
    # values3d = np.sin(X*Y*Z) / (X*Y*Z) #will sub cube
    # caps=dict(x_show=True, y_show=True, z_show=True, x_fill=1)

    # fig3d = go.Figure(data=go.Volume(
    #         x=X.flatten(),
    #         y=Y.flatten(),
    #         z=Z.flatten(),
    #         value=values3d.flatten(),
    #         isomin=0.1,
    #         isomax=0.8,
    #         opacity=0.1,
    #         opacityscale="uniform",
    #         caps = caps,
    #         surface_count=17,
    #         colorscale='RdBu'
    #     ))


    msg=html.Div([
        html.Pre("try 40"),
        html.Pre(err_message),
        #html.Pre(str(fig3d)),
        #html.Pre(str(ctx_msg)),
        #html.Pre(str((globals.axes[2].shape))),
        #html.Pre(str(sub_cube)),
        #html.Pre(str(sub_axes[0])),
    ])

    return [fig3d, msg]
