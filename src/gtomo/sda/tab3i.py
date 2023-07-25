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

#import dash_html_components as html
#import dash_core_components as dcc
from dash import dcc
from dash import html

#from dash.dependencies import Input, Output, State
from dash import Dash, Input, Output, State
from dash.exceptions import PreventUpdate

import pandas as pd
import globals
from sda import sda

def content_info():
    content_info = html.Div([
        html.Div([
            html.H5('References and additional information to use the dust extinction cubes'),
            dcc.Markdown('''            
            Details on the 3d dust map reconstruction are given in 
            [Lallement et al. 2022](https://arxiv.org/abs/2203.01627)

            #### Available 3d dust extinction maps 

            ##### Highest resolution
            This 3d map is recommended for users interested in detailed structures in the galactic 
            dust distribution relative close to the Sun.

            * Extent: 3 kpc
            * Sampling: 10 pc
            * Resolution: 25 pc

            ##### Largest distance
            This 3d map is recommend for users interested primarily in larger structures in the 
            galactic dust distribution traced to further distances from the Sun.

            * Extent: 5 kpc
            * Sampling: 20 pc
            * Resolution: 50 pc

            #### 1D distance-extinction density

            ##### Single target 1d profiles

            You can enter either sky coordinates (equatorial or galactic) or the target name 
            (resolved with CDS/Simbad) to retrieve line-of-sight differential extinction-distance 
            and cumulative extinction-distance curves.
            You can also provide a distance for a marker to be put in the plots.

            ##### Bulk upload 1d profiles

            Target input file (CSV) can be a list of target names (resolved with Simbad) or 
            equatorial/galactic (decimal notation) coordinates.
             
            Two examples are available for download below (use right-click + save link as).
            '''),
            dcc.Link("Target ID example input file", href=sda.get_asset_url('targets.csv')),
            html.Br(),
            dcc.Link("Galactic coordinates example input file:", href=sda.get_asset_url('coord.csv')),
            html.Br(),
            dcc.Markdown('''
            #### 2D planar extinction density maps

            ##### Tips

            For horizontal maps, if increasing longitudes are not in the right sense then the solution
            is to map again using an opposite latitude for the vector perpendicular to the image.

            In case the view is not correct (e.g., view from above or below for maps parallel to the plane) 
            try to invert the perpendicular. For the map parallel to the plane this will change 
            the view from above or below.
            
            #### Export results

            1d profiles and 2d maps can be exported as CSV files.

            ##### 1D export
            
            For each input target the ouput CSV will contain 3 columns: distance, extinction, cumulative extinction

            ##### 2D export

            The 2D export CSV file contains 3 columns: X, Y, Z, where X and Y form the 2d plane and Z gives 
            the log10 extinction value in the XY-plane.

            #### Acknowledgements

            If you use data from this tool we kindly ask you to include a reference to Lallement et al. 2022, ArXiv:2203.01627.
            And include an acknowledgement to the EXPLORE project:

            ***This research has used data, tools or materials developed as part of the EXPLORE 
                    project that has received funding from the European Union’s Horizon 2020 
                    research and innovation programme under grant agreement No 101004214.***

            '''),
            #html.Br(),
            html.Hr(),
            #html.H6("This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 101004214."),
            #html.Br(),
            # dcc.Link(
            #     html.Img(
            #         src=sda.get_asset_url('flag_yellow_low.jpg'),
            #         style={
            #             'width':'100px',
            #             'height':'auto',
            #             'float':'left'
            #         },
            #     ),
            #     href='https://explore-platform.eu',
            # ),                 

        ], style={'width':'100%', 'display':'inline-block', 'margin':'5px', 'padding':'50px'} ),
    ])

    return content_info



