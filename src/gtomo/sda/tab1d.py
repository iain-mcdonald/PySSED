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

import io
import base64
import datetime
import pandas as pd
import json

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
#from astroquery.gaia import Gaia

import scipy.interpolate as spi

import dash
#import dash_html_components as html
#import dash_core_components as dcc
#import dash_table
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objects as go

import globals

from sda import sda
from reddening import reddening

from flask import Flask
from flask_caching import Cache

CACHE_CONFIG = {
    'DEBUG': True,
    'CACHE_TYPE': 'FileSystemCache',
    'CACHE_DIR': '/tmp/',
    'CACHE_DEFAULT_TIMEOUT': 600,
    'CACHE_THRESHOLD': 10, #max number of items the cache will store
}
cache = Cache()
cache.init_app(sda.server, config=CACHE_CONFIG)

#We can clear the cache:
#  cache.clear()

""" figure layouts """

def fig_layout_1d(title):
    return dict(
        xaxis={
            'type': 'linear', 
            'title': 'distance (pc)',
            'zeroline': False,
            'showline': True,
            'showticklabels': True,
            'color': "#534998",
            },
        yaxis={
            'title': 'A(550nm) Extinction (mag)',
            'zeroline': False,
            'showticklabels': True,
            'color': "#534998",
            },
        #line=dict(color="#C68F0A"),
        margin={'l': 60, 'b': 60, 't': 30, 'r': 10},
        legend={'orientation':'h', 'yanchor':'bottom', 'y':0.95, 'xanchor':'right', 'x':1},
        title=str(title),
        hovermode='closest',
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
        )


fig_dict = {
    'data': [
        dict(
            mode='lines',
        )
    ],
    'layout': fig_layout_1d(""),
}

fig1d = go.Figure(fig_dict)


search_box = html.Div([
            html.H5("Step 1: Enter Target Coordinates or Name"),
            html.Div([
                #html.Label('Coordinate Frame: '),
                dcc.Checklist(
                    id='check-lb',
                    options=[
                        {'label': 'Check box if you use Galactic coordinates (otherwise Equatorial)', 'value': "1"},
                    ],
                ),
                html.Br(),
                html.Label('Coordinates: '),
                html.Br(),
                html.Label('RA/l (0..360 degrees)= '),
                dcc.Input(id='set-ra', value='', type='text', placeholder="[0,360]"),
                html.Br(),
                html.Label('Dec/b (-90..90 degrees)= '),
                dcc.Input(id='set-dec', value='', type='text', placeholder="[-90,90]"),
                html.Br(),
                html.Label("OR"),
                html.Br(),
                html.Label("Name: "),
                dcc.Input(id='set-target', value='', type='text', placeholder="Target", size='16'),
                html.Br(),
                dcc.Checklist(
                    id='resolve-check',
                    options=[
                        {'label': 'Check box if you want to resolve input target name', 'value': "1"},
                    ],
                ),
                html.Hr(),
                html.Div(id='resolve-out'),
                html.Label("Distance"),
                dcc.Input(id='set-distance', value='100.0', type='text', placeholder="Distance", size='16'),
            ], style={'padding':'3px', 'margin':'10px', 'borderWidth': '2px'}),
            html.Hr(),
            html.Button(id='submit-button-1d-single', n_clicks=0, children='Submit'),
            html.Hr(),
            html.Div(id='upload-file'),
            html.Div(id='submit-button-1d-bulk'),
    ], style={'width': '100%', 'float': 'left', 'margin':'10px'})

bulk_box = html.Div([
            html.H5("Step 1: Upload Target List"),
            html.Div([
                html.Label('Input list (see format in Information tab)'),
                dcc.Upload(
                    id='upload-file',
                    children=[html.Button(['Upload file'])],
                    multiple=False
                ),
                html.Br(),
                dcc.Checklist(
                    id='check-lb',
                    options=[
                        {'label': 'Check box if you use Galactic coordinates (otherwise Equatorial)', 'value': "1"},
                    ],
                ),
                html.Br(),
                dcc.Checklist(
                    id='resolve-check',
                    options=[
                        {'label': 'Check box if you want to resolve list of target names', 'value': "1"},
                    ],
                ),
            ], style={'padding':'3px', 'margin':'10px', 'borderWidth': '2px'}),
            html.Hr(),
            html.Button(id='submit-button-1d-bulk', n_clicks=0, children='Submit'),
            html.Hr(),
            html.H6('Parsing result input file'),
            html.Div(id='uploaded-targets', style={'width':'100%', 'padding':'10px'}),
            html.Div(id='submit-button-1d-single'),
            html.Div(id='set-ra'),
            html.Div(id='set-dec'),
            html.Div(id='set-distance'),
            html.Div(id='resolve-out'),
            html.Div(id='set-target'),
    ], style={'width': '100%', 'float': 'left', 'margin':'10px'})

def content_1d():
    content_1d = html.Div([
        html.Div([
            dcc.Tabs(
                id='tabs-1d', 
                value='single', 
                children=[
                    dcc.Tab(label='Single Target', value='single'),
                    dcc.Tab(label='Bulk Upload', value='bulk'),
                ]
            ),       
            dcc.Loading(
                id='load-1d',
                type='circle',
                fullscreen=False,
                children=[
                    html.Div(
                            id='tab1d-content', 
                            children=[],
                    ),
                ],
            ),
        ], style={'width': '33.33%', 'display':'inline-block', 'margin-right':'5px'}),
        html.Div([
            html.Br(),
            #html.Label('Results'),
            html.Br(),
            html.Div([   #add download button to save 1d-profiles (either from single target or bulk processing)
                html.Button(id='save-gaia-1d', n_clicks=0, children='Download CSV', style={'margin-left':'0px'}),
                dcc.Download(id='download-profiles-csv'), 
            ]),
            html.Br(),
            html.Div(id='log1', children=[]),
            html.Hr(),
            html.Div(id='1d-status'),
            html.Hr(),
            html.Div(id='dropdown-div'),
            html.Hr(),
            dcc.Graph(id='1d-plot', figure=fig1d, config={"doubleClick": "reset"}),
            dcc.Graph(id='1d-plot-c', figure=fig1d, config={"doubleClick": "reset"}),
            html.Div(id='temp'),
            html.Div(id='temp2'),
        ], style={'width': '60%','display':'inline-block', 'margin-right':'5px'})
    ])

    return content_1d

""" callbacks 1d """

### callback to save 1d calculated profiles + gaia info. Format of output file TBD
@sda.callback(
    [Output('log1', 'children'),
     Output("download-profiles-csv", "data")],
    [Input("save-gaia-1d", "n_clicks")],
    [State("signal", "data"),
     State('cube-dropdown', 'value')],
    prevent_initial_call=True,
)
def save1d(n_clicks, signal_data, cubename):

    if n_clicks==0:
        print('update prevented')
        raise PreventUpdate

    else:
        mykeys = []
        mylabel = []
        seltype = 0
        mydict = {}
        err_message = 'no error'
        df = pd.DataFrame()

        try:
            results = global_store(signal_data)
        except:
            results = None
            err_msg = 'error on reading store'

        try:
            err_message = 'no error'
            #for key in np.arange(len(results.keys())):
            for key in results.keys():

                #print(str(key))

                mykeys.append(key)

                selected = results[str(key)]
                seltype = type(selected['xvalues'])
                try:
                    if cubename == 'cube1':
                        error_band_dens = selected['yval_errdens_interpol']
                        error_band_ext = selected['yval_errext_interpol']

                    if cubename == 'cube2':
                        error_band_dens = selected['yval_errdens']
                        error_band_ext = selected['yval_errext']
                except:
                    err_message = 'error here'

                mylabel.append(str(key)+"_x")
                columnx = str(key)+"_x"
                columny = str(key)+"_y"
                columnyc = str(key)+"_yc"
                columnerry = str(key)+"_erry"
                columnerryc = str(key)+"_erryc"

                mydict[columnx] = selected['xvalues']
                mydict[columny] = selected['yvalues']
                mydict[columnyc] = selected['yvalues_cumul']
                mydict[columnerry] = error_band_dens
                mydict[columnerryc] = error_band_ext
                
                #df[str(key)+"_"+str(selected['ra'][0])+"_"+str(selected['dec'][0])+"_x"] = selected['xvalues']
                del(selected)

            df = pd.DataFrame.from_dict(mydict, orient='index')
            df = df.transpose()

            #err_message = 'no error'

        except:
            err_message = 'error'

        #data = np.column_stack((np.arange(10), np.arange(10) * 2))
        #df = pd.DataFrame(columns=["a column", "another column"], data=data)

    err_msg = "clicks: "+str(n_clicks)

    log = html.Div([
                    #html.Br(),
                    #html.Pre(err_msg),
                    #html.Pre(err_message),
                    #html.Br(),
                    #html.Pre('counter 2'),
                    #html.Pre(cubename),
                    #html.Pre(str(df)),
                    #html.Pre(str(results.keys())),
                    #html.Pre(str(results)),
                    #html.Pre(str(results[str(2)])),
                    #html.Pre(str(mydict)),
                    #html.Pre(str(mykeys)),
                    #html.Pre(str(mylabel)),
                    #html.Pre(str(seltype)),

                ])
    return [log, dcc.send_data_frame(df.to_csv, filename="my1dprofiles.csv", index=False)]


@sda.callback(Output('uploaded-targets', 'children'),
              Input('upload-file', 'contents'),
              State('upload-file', 'filename'),
              State('upload-file', 'last_modified') )
def update_targets(content, filename, moddate):
    if content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)

        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                test=[{'name': i, 'id': i } for i in df.columns]
                #add check to make sure its' either 1 or 2 col input with target names or coordinates
        except:
            return html.Div(['Error in processing this file; please ensure to provide csv file (1 col target names or 2 col coordinates; possible with extra distance columns)'])

        return html.Div([
            html.H5(filename),
            html.H6(datetime.datetime.fromtimestamp(moddate)),
            dash_table.DataTable(
                id='datatable-bulk',
                data=df.to_dict('records'),
                columns=[{'name': i, 'id': i } for i in df.columns],
                #filter_action='native',
                #editable=True,
                #row_selectable='multi',
                #row_deletable=True,
                #selected_rows=[],
                page_action='native',
                page_current=0,
                page_size=10,
                fixed_rows={'headers': True},
                style_table={'height': '300px', 'overlfowY': 'auto'}
            ),
            html.Hr(),
            #
            #html.Pre(content[0:200] + '...', style={
            #    'whiteSpace':'pre-wrap',
            #    'wordBreak': 'break-all'
            #})
        ], style={'width':'90%', 'padding':'5px'}) 
 

@cache.memoize()
def global_store(input_dict):
    import numpy as np
    #performs the expensive computation+queries on list of SkyCoord

    input_dict = json.loads(input_dict)
    step_pc=input_dict['step_pc']
    ra=input_dict['ra']
    dec=input_dict['dec']
    sc=SkyCoord(ra*u.deg, dec*u.deg)

    dist = input_dict['dist']
    #dist_min = input_dict['dist_min']
    #dist_max = input_dict['dist_max']

    cubename = input_dict['cubename']

    xvalues=None
    yvalues=None
    yvalues_cumul=None
    yval_errdens=None
    yval_errext=None
    yval_errdens_interpol=None
    yval_errext_interpol=None
    cube_id=None

    err_message = ''

    results = {}

    try:
        for i in range(len(sc)):

            try:
                key=str(i)
            except:
                err_message = 'error key idx'
                
            # extract the cumulative and differential extinction values:
            #xvalues, yvalues_cumul, yvalues = reddening(sc[i], cube=globals.cube, axes=globals.axes, max_axes=globals.max_axes,step_pc=step_pc)

            try:                
                # extract the error on the extinction density; used for the differential extinction plot
                xval_errdens, ycumul, yval_errdens = reddening(sc[i], cube=globals.cube_errdens, axes=globals.axes_errdens, max_axes=globals.max_axes_errdens,step_pc=step_pc)
                # extract the error on the intergrated extinction; used for the cumulative extinction plot
                xval_errext, ycumul_err, yval_errext = reddening(sc[i], cube=globals.cube_errext, axes=globals.axes_errext, max_axes=globals.max_axes_errext,step_pc=step_pc)
            except:
                err_message = 'error read error cubes'

            if cubename == 'cube1':
                cube_id = "explore_cube_density_values_025pc_v1.h5"
                xvalues, yvalues_cumul, yvalues = reddening(sc[i], cube=globals.cube25, axes=globals.axes25, max_axes=globals.max_axes25,step_pc=step_pc)

                try:
                    finterpd = spi.interp1d(xval_errdens, yval_errdens)
                    yval_errdens_interpol = finterpd(xvalues)

                    finterpe = spi.interp1d(xval_errext, yval_errext)
                    yval_errext_interpol = finterpe(xvalues)
                    
                except:
                    err_message = 'error cube 1'

            if cubename == 'cube2':
                cube_id = "explore_cube_density_values_050pc_v1.h5"
                try:
                    xvalues, yvalues_cumul, yvalues = reddening(sc[i], cube=globals.cube50, axes=globals.axes50, max_axes=globals.max_axes50,step_pc=step_pc)
                except:
                    err_message = "error cube 2"

           
            #### INTERPOLATE TO GET EXTINCTION AT DIST, DIST_MIN, DIST_MAX + ERROR_EXTINCTION_DIST
            try:
                ext_dist = np.interp(dist, xvalues, yvalues_cumul)
                err_ext_dist = np.interp(dist, xval_errext, yval_errext)
                #ext_dist_min = np.interp(dist_min, xvalues, yvalues_cumul)
                #ext_dist_max = np.interp(dist_max, xvalues, yvalues_cumul)
            except:
                err_message = 'error interpol'

            err_message_red=str(cube_id)

            results[key] = {'data': str(cube_id),
                            'ra': ra[i],
                            'dec': dec[i],
                            'xvalues': xvalues,
                            'yvalues': yvalues,
                            'yvalues_cumul': yvalues_cumul,
                            'yval_errdens': yval_errdens,
                            'yval_errext': yval_errext,
                            'dist': dist,
                            #'dist_min': dist_min,
                            #'dist_max': dist_max,
                            'ext_dist': ext_dist,
                            'err_ext_dist': err_ext_dist,
                            #'ext_dist_min': ext_dist_min,
                            #'ext_dist_max': ext_dist_max,
                            'yval_errdens_interpol': yval_errdens_interpol,
                            'yval_errext_interpol': yval_errext_interpol,
                            'err_msg': err_message,
                            }
    except:
        #err_message2 = 'calc error'
        results[key] = {'xvalues': 4,
                        'yvalues': 4,
                        'yvalues_cumul': 0,
                        'yval_errdens': 0,
                        'yval_errext': 0,
                        'dist': 0,
                        'dist_min': 0,
                        'dist_max': 0,
                        'ext_dist': 0,
                        'err_ext_dist': 0,
                        'ext_dist_min': 0,
                        'ext_dist_max': 0,
                        'err_msg': err_message,
                        }

    return results



@sda.callback(
    [Output('signal', 'data'),
     Output('1d-status', 'children'),
     Output('dropdown-div', 'children'),
    ],
    [Input('submit-button-1d-single', 'n_clicks'),
    Input('submit-button-1d-bulk', 'n_clicks'),
    ],
    [State('set-ra', 'value'), 
    State('set-dec', 'value'),
    State('check-lb', 'value'),
    State('resolve-check', 'value'),
    State('set-target', 'value'),
    State('set-distance', 'value'),
    State('tabs', 'value'),
    State('upload-file', 'contents'),
    State('upload-file', 'filename'),
    State('cube-dropdown', 'value'), 
    ],
)
def compute_value(n_clicks1,n_clicks2,lon,lat,checklb,resolve,target,dist,tab,csv_content,csv_filename,cubename):
    # compute value and send a signal when done
    err_message=''
    sc=None
    input_dict = {}
    input_dict2 = {}
    droplist = html.Div(id='1d-dropdown')
    targets=[]

    target=str(target)

    if ((n_clicks1 == 0) or (n_clicks1 == None)) and ((n_clicks2 == 0) or (n_clicks2 == None)):
        raise PreventUpdate

    ctx = dash.callback_context

    if not ctx.triggered: 
        button_id = None
        raise PreventUpdate

    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    import json
    ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered,
        'inputs': ctx.inputs,
        #'outputs': ctx.outputs_list,
    }, indent=2)

    if (button_id == 'submit-button-1d-single'):
        try:
            lon=[np.float(lon)]
            lat=[np.float(lat)]
            tmp_lon = lon
            tmp_lat = lat
            #dist_min = float(dist)*0.9
            #dist_max = float(dist)*1.1
        except:
            #if ((type(lon) == int) or (type(lat) == float)):
            #err_message = "Please enter valid coordinates"
            lon=None
            lat=None

        try:
            targets.append(target)
        except:
            targets.append(None)


    if (button_id == 'submit-button-1d-bulk'):
        lon=None
        lat=None

        if csv_filename is not None:
            err_message = '' #str(csv_filename)
            if 'csv' in csv_filename:
                try:
                    content_type, content_string = csv_content.split(',')
                    decoded = base64.b64decode(content_string)
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

                    if len(df.columns) == 1:
                        #try to use as target names
                        resolve = 1
                        targets = df[df.columns[0]]
                        #target = df.iloc[:,0].tolist()
                        dist = None
                        #dist_min = float(dist)*0.9
                        #dist_max = float(dist)*1.1

                    elif len(df.columns) == 2:
                        #try to use as coordinates: lon, lat
                        df_lon = df.iloc[:,0]
                        df_lat = df.iloc[:,1]
                        dist = None
                        # dist_min = 0.0
                        # dist_max = 0.0
                        resolve = 0
                        lon=df_lon.to_numpy()
                        lat=df_lat.to_numpy()
                        #err_message = str(lon)
                        targets = [None]

                    # elif len(df.columns) == 3:
                    #     #try to use as coordinates: lon, lat
                    #     df_lon = df.iloc[:,0]
                    #     df_lat = df.iloc[:,1]
                    #     dist = (df.iloc[:,2]).to_numpy()
                    #     # dist_min = 0.0 
                    #     # dist_max = 0.0
                    #     resolve = 0
                    #     lon=df_lon.to_numpy()
                    #     lat=df_lat.to_numpy()
                    #     err_message = str(lon)
                    #     target = [None]

                    # elif len(df.columns) == 5:
                    #     #try to use as coordinates: lon, lat
                    #     df_lon = df.iloc[:,0]
                    #     df_lat = df.iloc[:,1]
                    #     dist = (df.iloc[:,2]).to_numpy()
                    #     dist_min = (df.iloc[:,3]).to_numpy()
                    #     dist_max = (df.iloc[:,4]).to_numpy()
                    #     resolve = 0
                    #     lon=df_lon.to_numpy()
                    #     lat=df_lat.to_numpy()
                    #     err_message = str(lon)
                    #     target = [None]

                    else:
                        err_message = 'warning: no valid csv file detected; 0 or >2 columns in CSV'
                
                    #err_message = 'csv read ok'
                except:
                    err_message = 'warning: no valid csv file detected; parsing failed'
        else:
            err_message = "please first upload target list"


    if not resolve:

        if ((lon is not None) and (lat is not None)):

            if checklb:
                sc=SkyCoord(lon*u.deg, lat*u.deg, frame='galactic')
                sc=sc.transform_to('icrs')
                #err_message = "coord gal: "+str(sc)

                err_message = "Galactic coordinate converted to ICRS: "+str(sc)

            if not checklb: 
                sc=SkyCoord(lon*u.deg, lat*u.deg, frame='icrs') 
                #err_message = "coord icrs: "+str(sc)

        
        else:
            err_message = "Please provide valid lon/lat values (not None)"
            
    if resolve:

        try:
            ra=[]
            dec=[]
            ntarg=len(targets)

            for targ in range(ntarg):
                tmp=SkyCoord.from_name(targets[targ], frame='icrs') 
                ra.append(tmp.ra)
                dec.append(tmp.dec)

            sc=SkyCoord(ra, dec, frame='icrs')

            err_message = "Coordinates resolved: "+str(sc)

        except:

            err_message = "Target not resolved. Please enter a valid target name." #+str(sc)

    if not sc:
        err_message2 = 'No SkyCoord provided'  

    else:
        #coord_string = "used coordinates: "+str(sc.ra)+"_"+str(sc.dec)
        x_err_message = "run global_store()"
        
        input_dict['index'] = np.arange(len(sc)).tolist()
        input_dict['ra'] = (sc.ra.value).tolist()
        input_dict['dec'] = (sc.dec.value).tolist()
        input_dict['dist'] = dist
        #input_dict['dist_min'] = dist_min
        #input_dict['dist_max'] = dist_max
        input_dict['step_pc'] = 5
        input_dict['cubename'] = cubename

        if resolve == 1:
            input_dict['target'] = targets.tolist()

        input_dict2 = json.dumps(input_dict)
        
        try:
            global_store(input_dict2) ## this is the expensive computation to compute 1d sighltines 
        except:
            err_message = "error with global store calculation"

    if not resolve:
        if sc is not None:
            #options=[{'label': str(sc[i].ra)+"_"+str(sc[i].dec), 'value': str(sc[i].ra)+"_"+str(sc[i].dec)} for i in range(len(sc))]
            droplist = dcc.Dropdown(
                id='1d-dropdown',
                options=[{'label': str(i)+"_"+str(sc[i].ra)+"_"+str(sc[i].dec), 'value': str(i)+"_"+str(sc[i].ra)+"_"+str(sc[i].dec)} for i in range(len(sc))],
                clearable=True,
            )
    else:
        if target is not None:
            #err_message = "creating pulldown for target names"
            #options=[{'label': target[i], 'value': target[i]} for i in range(len(target))]
            droplist = dcc.Dropdown(
                id='1d-dropdown',
                options=[{'label': str(i)+"_"+targets[i], 'value': str(i)+"_"+targets[i]} for i in range(len(targets))],
                clearable=True,
            )

    div_status=html.Div([
        #html.Label("Status:"),
        #html.Br(),
        html.Pre(err_message),
        #html.Pre(str(input_dict2)),
        #html.Br(),
        #html.Pre("distance min: ", str(dist_min)),
        #html.Br(),
        #html.Pre(str(cubename)),
        #html.Pre(str(input_dict['target'])),
        #html.Pre(str(options)),
        #html.Pre(str(droplist)),
        html.H5("Step 2: Select target from dropdown list below to plot the 1d profiles"), #+str(json.loads(input_dict2)))
    ])

    return [input_dict2, div_status, droplist]


# callback to update plot 1d
# (also shows output/data in temp Div)
@sda.callback(
    [
    Output('1d-plot', 'figure'),
    Output('1d-plot-c', 'figure'),
    Output('temp', 'children')
    ],
    [Input('signal', 'data'),
     Input('1d-dropdown', 'value')
    ],
    State('cube-dropdown', 'value'), 
)
def selected_entry(signal_data, dropdownvalue, cubename):
    import numpy as np
    
    if dropdownvalue is None:
        raise PreventUpdate

    ctx = dash.callback_context

    if not ctx.triggered: 

        raise PreventUpdate

    title='Extinction-distance profile'
    titlec='Extinction-distance profile (cumulative)'
    #get data from 'global_store'

    try:
        results = global_store(signal_data)
        ###results = json.loads(results_json)
        idx=dropdownvalue.split("_")[0]
        ###selected = results[str(dropdownvalue)]
        selected = results[str(idx)]
        error='no error'
    except:
        error='error'

    #data=[]
    #datac=[]

    if selected['yvalues'] is not None:

        error_band_dens = selected['yval_errdens']
        if cubename == 'cube1':
            error_band_dens = selected['yval_errdens_interpol']

        data = go.Scatter(
                    x=selected['xvalues'],
                    y=selected['yvalues'],
                    mode='lines',
                    line=dict(
                        color="rgb(83,73,152)",
                        width=2.0
                    ),
                    opacity=1.0,
                    customdata=error_band_dens, #selected['yval_errdens'],
                    hovertemplate='d:%{x:.1f}<br>Ext:%{y:.4f}<br>err:%{customdata:.5f}',
                    name=str(dropdownvalue)+" (ISM)",
                    )

        data_upper = go.Scatter(
                        x=selected['xvalues'],
                        y=selected['yvalues']+error_band_dens,
                        mode='lines',
                        marker=dict(color='#efd088'),
                        opacity=0.5,
                        line=dict(width=0),
                        showlegend=False,
        )

        data_lower = go.Scatter(
                        x=selected['xvalues'],
                        y=selected['yvalues']-error_band_dens,
                        mode='lines',
                        marker=dict(color='#efd088'),
                        line=dict(width=0),
                        opacity=0.5,
                        fillcolor='#efd088',
                        fill='tonexty',
                        showlegend=False,
        )

        # try:
        #     dist_point1 = go.Scatter(
        #                     x=np.array(float(selected['dist'])),
        #                     y=np.array(float(selected['ext_dist'])),
        #                     error_y=dict(
        #                         type='data',
        #                         array=np.array(float(selected['err_ext_dist'])),
        #                         visible=True,
        #                         color='green'),
        #                     marker=dict(color='green', size=8),
        #                     mode='markers',
        #                     name='distance point',
        #     )

        # except:
        #     dist_point1 = go.Scatter()

    newfig = go.Figure([data_upper, data_lower, data])
    #newfig = go.Figure([data])
    newfig.update_layout(fig_layout_1d("Differential extinction"))

    if selected['yvalues_cumul'] is not None:
        msg='test'
        error_band_ext = selected['yval_errext']
        if cubename == 'cube1':
            error_band_ext = selected['yval_errext_interpol']

        #if cubename == 'cube2':
        #    error_band_ext = selected['yval_errext']

        datac = go.Scatter(
                    x=selected['xvalues'],
                    y=selected['yvalues_cumul'],
                    mode='lines',
                    line=dict(
                        color="rgb(83,73,152)",
                        width=2.0
                    ),
                    opacity=1.0,
                    customdata=error_band_ext,
                    hovertemplate='d:%{x:.1f}<br>Ext:%{y:.2f}<br>err:%{customdata:.3f}',
                    name=str(dropdownvalue)+" (ISM)",
                    )

        datac_upper = go.Scatter(
                        x=selected['xvalues'],
                        y=selected['yvalues_cumul']+error_band_ext,
                        mode='lines',
                        marker=dict(color='#efd088'),
                        opacity=0.5,
                        line=dict(width=0),
                        showlegend=False,
        )

        datac_lower = go.Scatter(
                        x=selected['xvalues'],
                        y=selected['yvalues_cumul']-error_band_ext,
                        mode='lines',
                        marker=dict(color='#efd088'),
                        line=dict(width=0),
                        opacity=0.5,
                        fillcolor='#efd088',
                        fill='tonexty',
                        showlegend=False,
        )

        try:
            dist_point = go.Scatter(
                            x=np.array(float(selected['dist'])),
                            y=np.array(float(selected['ext_dist'])),
                            error_y=dict(
                                type='data',
                                array=np.array(float(selected['err_ext_dist'])),
                                visible=True,
                                color='green'),
                            marker=dict(color='green', size=15),
                            mode='markers',
                            name='distance point',
            )

        except:
            dist_point = go.Scatter()

        #     dist_minmax = go.Scatter(
        #                     x=np.array([float(selected['dist_min']), float(selected['dist_max'])]),
        #                     y=np.array([float(selected['ext_dist_min']), float(selected['ext_dist_max'])]),
        #                     marker=dict(color='red', size=8),
        #                     mode='markers',
        #     )

        #     error = 'dist point ok!'
        # except:
        #     error = 'error in dist_point'    


    try: 
        newfigc = go.Figure([datac_upper, datac_lower, datac, dist_point])
    except:
        newfigc = go.Figure([datac_upper, datac_lower, datac])
    #newfigc = go.Figure([datac]) 
    #newfigc = go.Figure()
    newfigc.update_layout(fig_layout_1d("Cumulative extinction"))

    #newfig = go.Figure()
    
    msg = html.Div([
        #html.Pre(str(dropdownvalue)),
        #html.Pre(str(idx)),
        #html.Pre(str(signal_data)),
        #html.Pre('results: '+str(results.keys())),
        #html.Pre('xvalues: '+str((selected['xvalues']))),
        #html.Pre('err_message: '+str((selected['err_message']))),
        #html.Pre('len(xvalues): '+str(len(selected['xvalues']))),
        #html.Pre('len(yvalues_cumul): '+str(len(selected['yvalues_cumul']))),
        #html.Pre('len(yval_errdens): '+str(len(selected['yval_errdens']))),
        #html.Pre('len(yval_errext): '+str(len(selected['yval_errext']))),
        #html.Pre('len(yval_errdens_interpol): '+str(len(selected['yval_errdens_interpol']))),
        #html.Pre('len(yval_errext_interpol): '+str(len(selected['yval_errext_interpol']))),
        #html.Pre('data: '+str(data)),
        #html.Pre('error extinct: '+str(selected['yval_errext'])),
        #html.Pre("error density: "+str(selected['yval_errdens'])),
        #html.Pre('results: '+str(results[str(idx)])),
        #html.Pre('error: '+str(error)),
        #html.Pre('err_ext_band: '+str(len(error_band_ext))),
        #html.Pre('err_dens_band: '+str(len(error_band_dens))),
        #html.Pre('cubename: '+str(cubename)),
        #html.Pre(str(selected['err_msg'])),
        # html.Pre('type dist :'+str(type(np.array(float(selected['dist']))))),
        # html.Pre('type ext dist :'+str(type(np.array(float(selected['ext_dist']))))),
        # html.Pre('type err ext dist :'+str(type(np.array(float(selected['err_ext_dist']))))),
        # html.Pre('ext dist :'+str(selected['ext_dist'])),
        # html.Pre('err ext dist :'+str(selected['err_ext_dist'])),
        # html.Pre('type xvalues: '+str(type(selected['xvalues'])))
    ])

    return [newfig, newfigc, msg]

