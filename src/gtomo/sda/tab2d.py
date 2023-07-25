# -*- coding: utf-8 -*-

# future import statements
from __future__ import print_function
from __future__ import division

# version information
__project__ = "EXPLORE"
__author__  = "ACRI-ST"
__modifiers__ = '$Author: N. Cox $'
__date__ = '$Date: 2022-03-21 $'
__version__ = '$Rev: 1.0 $'
__license__ = '$Apache 2.0 $'

import pandas as pd
import numpy as np

#import dash_core_components as dcc
#import dash_html_components as html
from dash import dcc
from dash import html

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go

import globals

from sda import sda
from cube_cut import cube_cut
#from load_cube import load_cube

""" functions """

def update_planar(X,Y,Z):
    newfig = go.Figure(data=
    go.Contour(
        z=Z,
        x=X,
        y=Y,
        name='2d map',
        colorscale='Cividis_r', #'Inferno_r', #
        colorbar=dict(
            title='A(550nm) mag/pc (log10)',
            titleside="right",
            ),
        connectgaps=True, 
        line_smoothing=0.85,
        hoverinfo='all',
        #hovertemplate="x: %{x:$.1f}, y: %{y:$.1f}, z: %{z:$.1f}",
        zmin=-4,
        zmax=-1.5,
        contours=dict(
            start=-3.5,
            end=-2,
            size=0.5,
        ),
        contours_coloring='heatmap', # can also be 'lines', or 'none',
        
    ),
    layout_xaxis_range=[np.nanmin(X),np.nanmax(X)],
    #layout_yaxis_range=[-5000,5000],
    layout=fig_layout_2d)
    #newfig.update_layout(yaxis_range=[-3000,3000], xaxis_range=[-3000,3000])
    return newfig

""" layout figures tab 2d """

fig_layout_2d = dict(
        margin={'l': 5, 'b': 5, 't': 5, 'r': 5},
        legend={'x': 0.8, 'y': 0.1},
        hovermode='closest',
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        yaxis=dict(
            scaleanchor='x',
            scaleratio=1,
        )
    )

# create data for first map:

fig2d = go.Figure(data =
     go.Contour(
        z=[[0,0],[0,0]],
        name='2d map',
        colorscale='Inferno_r', #'Cividis_r',
        colorbar=dict(
            title='A(550nm) mag/pc (log10)',
            titleside="right",
            ),
        connectgaps=True, 
        line_smoothing=0.85,
        hoverinfo='all',
        #hovertemplate="x: %{x:$.1f}, y: %{y:$.1f}, z: %{z:$.1f}",
        zmin=-3.5,
        zmax=-2,
        contours=dict(
            start=-3.5,
            end=-2,
            size=0.5,
        ),
        contours_coloring='heatmap', # can also be 'lines', or 'none',
    ),
    layout=fig_layout_2d)

#fig2d.update_layout(xaxis_range=[-5000,5000])
#fig2d.update_layout(yaxis_range=[-5000,5000], xaxis_range=[-5000,5000], yaxis=dict(scaleanchor='x'))
#fig2d['layout']['yaxis']['scaleanchor']='x'
fig2d.update_yaxes(
    scaleanchor = "x",
    scaleratio = 1,
)

def content_2d():
    content_2d = html.Div([
        #dcc.Store(id='memory'),
        html.Br(),
        html.Div([
            html.Div([
                html.Label('Frame ', style={'float':'left', 'margin-right':'10px'}),
                dcc.Dropdown(
                    id='2d-set-frame',
                    style={'width':'40%', 'float':'left'},
                    options=[
                        {'label': 'Galactic', 'value': 'galactic'},
                        #{'label': 'ICRS', 'value': 'icrs'},
                    ], 
                    value='galactic'),
                html.Br(),
            ], style={'width':'100%', 'align': 'left'}),
            html.Div([
                html.Div([
                    html.Br(),
                    html.H5('Spherical coordinates of the map origin'),
                    html.Br(),
                    html.Label('Longitude of origin', style={'margin-left':'10px'}),
                    dcc.Input(id='2d-set-lon-origin', type='number', placeholder="[0,360]", size='10'),
                    html.Br(),
                    html.Label('Latitude of origin ', style={'margin-left':'10px'}),
                    dcc.Input(id='2d-set-lat-origin', type='number', placeholder="[-90,90]", size='10'),
                    html.Br(),
                    html.Label('Distance of origin ', style={'margin-left':'10px'}),
                    dcc.Input(id='2d-set-distance-origin', type='number', placeholder='>0', size='10'),
                    #html.Br(),
                ], style={'width':'50%', 'display':'inline-block'}),
                html.Div([
                    html.H5('Polar coordinates of the direction defining the normal to the plane of the image'),
                    html.Br(),
                    html.Label('Longitude of normal', style={'margin-left':'10px'}),
                    dcc.Input(id='2d-set-lon-normal', type='number', placeholder="[0,360]", size='10'),
                    html.Br(),
                    html.Label('Latitude of normal', style={'margin-left':'10px'}),
                    dcc.Input(id='2d-set-lat-normal', type='number', placeholder="[-90,90]", size='10'),
                    #html.Br(),
                ], style={'width':'50%', 'display':'inline-block'}),
            ], style={'width':'100%', 'display':'inline-block'}),
            html.Br(),
            html.Button(id='2d-submit-button', n_clicks=0, children='Submit', style={'margin-left':'10px'}),
            html.Label(" Please note that it may take a little while to compute the new planar slice. Please be patient."),
        ], id='input', style={'width':'100%', 'margin':'5px','borderWidth': '2px'}),
        html.Div([  
            html.Button(id='save-gaia-2d', n_clicks=0, children='Download csv', style={'margin-left':'10px'}),
            dcc.Download(id='download-map-csv'), 
        ]),
        html.Div(id='log2', children=[]),
        html.Div([
            html.Hr(),
            html.Div(id='2d-log', children=[]),
            #html.Hr(),
            html.Div(id='2d-vars', children=[]),
            html.Hr(),
            dcc.Graph(id='2d-graph', figure=fig2d, config={"doubleClick": "reset"}),
        ], id='output', style={'width': '100%', 'margin':'5px', 'float': 'bottom', 'borderWidth': '2px'} ), 

    ], id='2d', style={'width':'100%', 'display':'inline-block', 'margin':'auto', 'padding-left':'10%', 'padding-right':'10%'})

    return content_2d

""" callbacks tab 2d """



### callback to save 2d maps. Format: x,y,z
@sda.callback(
    [Output('log2', 'children'),
    Output("download-map-csv", "data")],
    [Input("save-gaia-2d", "n_clicks")],
    #State("signal", "data"),
    [State('2d-graph', 'figure')],
    prevent_initial_call=True,
)
def save2d(n_clicks, figure):
    if n_clicks==0:
        print('update prevented')
        raise PreventUpdate

    # elif n_clicks is None:
    #     raise PreventUpdate

    else:
        #df = global_store(value)
        data = np.column_stack((np.linspace(0,6,num=4), np.linspace(0,6,num=4), np.logspace(-5.0,-2.0, num=4) ))
        df2 = pd.DataFrame(columns=["x", "y", 'z'], data=data)

        try:
            map2d=[]
            xx=np.asarray(figure['data'][0]['x'])
            yy=np.asarray(figure['data'][0]['y'])
            zz=np.asarray(figure['data'][0]['z'])
            xm,ym=np.meshgrid(xx,yy)
            for i in range(len(xx)):
                for j in range(len(yy)):
                    map2d.append([xx[i], yy[j], 10**zz[j,i]])
            #df2 = pd.DataFrame({'x':xm, 'y':ym, 'z':zz})
            #stacked = pd.Panel(a.swapaxes(1,2)).to_frame().stack().reset_index()
            #stacked.columns = ['x', 'y', 'z', 'value']
            #stacked.to_csv('stacked.csv', index=False)
            map2d2 = np.array(map2d)
            df2 = pd.DataFrame(map2d2, columns=['x','y','z'])

            err_msg2='no error'
        except:
            err_msg2='error!'


    err_msg = "clicks: "+str(n_clicks)
    log = html.Div([
                    #html.Br(),
                    #html.Pre(err_msg2),
                    #html.Br(),
                    #html.Pre('test 22'),
                    #html.Pre(str((xx.shape))),
                    #html.Pre(str((yy.shape))),
                    #html.Pre(str((zz.shape))),
                    #html.Pre(str(map2d)),
                    #html.Pre(str(map2d2.shape)),
                    #html.Pre(str(df2)),
                    #html.Pre(str((xm))),
                    #html.Pre(str((ym))),
                    #html.Pre(str(type(figure['data'][0]['x']))),
                    #html.Pre(str(type(figure['data'][0]['y']))),
                    #html.Pre(str(type(figure['data'][0]['z']))),
                    #html.Br(),
                ])


    return [log, dcc.send_data_frame(df2.to_csv, filename="my2dmap.csv", index=False)]
    #return dict(content='Saved the data!', filename='saved_data.txt')

@sda.callback(
    [Output('2d-graph', 'figure'),
     Output('2d-vars', 'children'),
     Output('2d-log', 'children'),
     #Output('memory', 'data'),
    ],
    [Input('2d-submit-button', 'n_clicks'),
     Input("save-gaia-2d", "n_clicks")],
    [State('2d-set-frame', 'value'), 
     State('2d-set-lon-origin', 'value'),
     State('2d-set-lat-origin', 'value'),
     State('2d-set-distance-origin', 'value'),
     State('2d-set-lon-normal', 'value'),
     State('2d-set-lat-normal', 'value'),
     State('cube-dropdown', 'value'), 
     ], 
    )
def on_submit_2d(clicks, clicks_save,frame,lon_orig,lat_orig,dist_orig,lon_norm,lat_norm, cubename):


    if clicks==0 or clicks is None:
        parsed_input_log = html.Div([])

        ulon = ulat = unlon = unlat = 'deg'
        udist = 'pc'
        lon_norm = -270.0
        lat_norm = -90.0

        result, X1, Y1, Z1 = cube_cut(globals.cube25, globals.hw25, globals.step25, globals.points25, globals.s25, 0.0, ulon, 0.0, ulat, 'galactic', 0.0, udist, lon_norm, unlon, lat_norm, unlat)

        parsed_input = html.Div([
            html.Label(result['title']),
            html.Br(),
            html.Label(result['xTitle']),
            html.Br(),
            html.Label(result['yTitle']),
        ])

        newfig = update_planar(X1,Y1,Z1)

        return [newfig, parsed_input, parsed_input_log]
        #raise PreventUpdate

    # elif clicks is None:
    #     raise PreventUpdate

    else:

        #df2d = pd.DataFrame()

        inputs = [lon_orig, lat_orig, dist_orig, lon_norm, lat_norm]

        if all(v is not None for v in inputs):

            if (dist_orig <= -1):
                parsed_input_log = html.Div([
                    html.Label('Distance should be greater than zero')
                ])
            else:
                parsed_input_log = html.Div([
                    #html.Pre(str(df)),
                    #html.Label(str(frame)),
                    #html.Label(str(lon_orig)),
                    #html.Label(str(lat_orig)),
                    #html.Label(str(dist_orig)),
                    #html.Label(str(lon_norm)),
                    #html.Label(str(lat_norm)),
                    #html.Br(),
                    #html.Label(str(globals.headers50['resolution_values'])),
                    #html.Label(str(globals.headers25['resolution_values'])),
                    #html.Label(str(cubename)),
                    ])
        else:
            parsed_input_log = html.Div([
                html.Label('All values required')
            ])

        newfig = None

        if all(v is not None for v in inputs):
            try:
                ulon = ulat = unlon = unlat = 'deg'
                udist = 'pc'
                #print(cube.shape, hw, step, lon_orig, ulon, lat_orig, ulat, frame, dist_orig, udist, lon_norm, unlon, lat_norm, unlat)
                #result, X, Y, Z = cube_cut(globals.cube, globals.hw, globals.step, globals.points, globals.s, lon_orig, ulon, lat_orig, ulat, frame, dist_orig, udist, lon_norm, unlon, lat_norm, unlat)

                if cubename == 'cube1':
                    result, X, Y, Z = cube_cut(globals.cube25, globals.hw25, globals.step25, globals.points25, globals.s25, lon_orig, ulon, lat_orig, ulat, frame, dist_orig, udist, lon_norm, unlon, lat_norm, unlat)
                if cubename == 'cube2':
                    result, X, Y, Z = cube_cut(globals.cube50, globals.hw50, globals.step50, globals.points50, globals.s50, lon_orig, ulon, lat_orig, ulat, frame, dist_orig, udist, lon_norm, unlon, lat_norm, unlat)    
               
                #renew parsed input:
                parsed_input = html.Div([
                    html.Label(result['title']),
                    html.Br(),
                    html.Label(result['xTitle']),
                    html.Br(),
                    html.Label(result['yTitle']),
                    html.Br(),
                    #html.Label(''),
                ])

                newfig = update_planar(X,Y,Z)

                #np.dstack((X,Y,Z))
                #df2d = pd.DataFrame({'x':X, 'y':Y, 'z':Z})

            except:
                #print('fail')
                pass 

        else:

            pass

        return [newfig, parsed_input, parsed_input_log]
