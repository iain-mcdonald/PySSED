# -*- coding: utf-8 -*-

# future import statements
from __future__ import print_function
from __future__ import division

# version information
__project__ = "EXPLORE"
__author__  = "ACRI-ST"
__modifiers__ = '$Author: N. Cox $'
__date__ = '$Date: 2021-10-22 $'
__version__ = '$Rev: 1.0 $'
__license__ = '$Apache 2.0 $'

import os

# check that the Lab data (.h5) is available

try:
    lab_data = os.environ.get('LAB_DATA')
except:
    print('env var not available?')

#from dash import dcc
from dash import html
#from dash.dependencies import Input, Output

# Import all Lab modules

import globals

from sda import sda
import sda_main

#from reddening import reddening
#from load_cube import load_cube
#from cube_cut import cube_cut

# test for the avaialble folders:
service_app_data = os.environ.get('SERVICE_APP_DATA')
app_data_files = os.listdir(service_app_data)

sda.layout = sda_main.layout

# setup "complete" layout for validation
sda.validation_layout = html.Div([
    sda_main.layout,
])


if __name__ == '__main__':
    #import os
    globals.initialise()
    #print('header cube version: ', globals.headers['resolution_values'])

    try:
        port = os.environ.get('dash_port')
        debug = os.environ.get('dash_debug')
        #print(port, debug)
        sda.run_server(host='0.0.0.0', debug=False, port=port, threaded=False, processes=1)
    except:
        sda.run_server(host='0.0.0.0', debug=False, port=8050, threaded=False, processes=1)
