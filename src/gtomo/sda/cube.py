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


import numpy as np
import scipy.interpolate as spi
from astropy import units as u
from astropy.coordinates import SkyCoord

def sub_cube(cube, axes, sc=None, step=5.0, size_pc=100.0, center_distance=0.0):
    """ function that allows to subset the data cube based on direction (coordinates), distance (pc) and size (pc).
    It extracts a sub-cube centred on a certain position in the master cube (defined by distance and sky coordinate).
    It does no resampling, simply recentering and subsetting. Handle issue when sub-cube would lie outside the master cube. 
    """
    #find pixel corresponding to the 3d-location (coordinate/distance) of the
    #desired 'origin'
    
    #print(center_distance)

    # if not sc:
    #     sc = SkyCoord(
    #         0.0,
    #         0.0,
    #         distance=0.0,
    #         unit=(u.deg, u.deg, u.pc),
    #         frame='galactic',
    #     )

    # r = sc.represent_as('cartesian').get_xyz().value
    # # find in axes[0], axes[1], axes[2] values closest to r!
    
    nr_pixels = size_pc / step / 2 #divide by two to have range around the center.
    # check that center+-nr_pixels is not outside of the cube; if so set max.
    #center=[0,0,0]
    center=[600,600,80]
    min_x = center[0] - int(nr_pixels)
    max_x = center[0] + int(nr_pixels)
    min_y = center[0] - int(nr_pixels)
    max_y = center[0] + int(nr_pixels)
    min_z = center[0] - int(nr_pixels)
    max_z = center[0] + int(nr_pixels)

    sub_cube = cube[min_x:max_x+1,min_y:max_y+1,min_z:max_z+1]
    sub_axes = [axes[0][min_x:max_x], axes[1][min_y:max_y], axes[2][min_z:max_z]]

    return sub_cube, sub_axes