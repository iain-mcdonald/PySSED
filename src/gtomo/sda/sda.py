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

import os
import dash
import dash_bootstrap_components as dbc

external_stylesheets = [dbc.themes.YETI]
# define Trebuchet MS font in custom stylesheet in assets/

sda = dash.Dash(__name__, #suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets,
    requests_pathname_prefix = os.environ.get('PATH_PREFIX', '/'),
    meta_tags = [
        {"name": "viewport",
         "content": "width=device-width, initial-scale=1.0"},
    ],
)

sda.scripts.config.serve_locally = True
sda.title = "EXPLORE: G-Tomo" 
sda.config.suppress_callback_exceptions = True

server = sda.server

