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

import numpy as np
import scipy.interpolate as spi
from astropy import units as u
from astropy.coordinates import SkyCoord

def reddening(sc, cube, axes, max_axes, step_pc=5):
    """Calculate Extinction versus distance from Sun.

    Args:
        sc: SkyCoord object

    Kwargs:
        step_pc (int): Incremental distance in parsec

    Returns:
        array: Parsec values.
        array: Extinction A(5500) value obtained with integral of linear extrapolation.

    """

    sc1=SkyCoord(sc, distance = 1 * u.pc)

    coords_xyz = sc1.transform_to('galactic').represent_as('cartesian').get_xyz().value

    # Find the number of parsec I can calculate before go out the cube
    # (exclude divide by 0)
    not0 = np.where(coords_xyz != 0)

    max_pc = np.amin(
        np.abs( np.take(max_axes, not0) / np.take(coords_xyz, not0) ) )

    # Calculate all coordinates to interpolate (use step_pc)

    distances = np.arange(0, max_pc, step_pc)

    sc2 = SkyCoord(
        sc,
        distance=distances)

    sc2 = sc2.transform_to('galactic').represent_as('cartesian')
    coords_xyz = np.array([coord.get_xyz().value for coord in sc2])

    # linear interpolation with coordinates
    interpolation = spi.interpn(
        axes,
        cube,
        coords_xyz,
        method='linear'
    )

    xvalues = np.arange(0, len(interpolation) * step_pc, step_pc)
    yvalues_cumul = np.nancumsum(interpolation) * step_pc
    yvalues = interpolation

    
    return (
        xvalues,
        np.around(yvalues_cumul, decimals=5),
        np.around(yvalues, decimals=5)
        )

