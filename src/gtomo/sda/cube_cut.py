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

def cube_cut(cube, hw, step, points, s, vlong, ulong, vlat, ulat, frame, vdist, udist, vnlong, unlong, vnlat, unlat):
    """Calculate map cut of cube.

    Args:
        cube (float, 3d): 3d cube to be used
        vlong (str or double): Longitude value.
        ulong (str): Longitude unit used in :class:`SkyCoord`.
        vlat (str or double): Latitude value.
        ulat (str): Latitude unit used in :class:`SkyCoord`.
        frame (str): Galactic, icrs, etc
            (values supported by :class:`SkyCoord`).
        vdist (str or double): Distance.
        udist (str): Distance unit used in Skycoord.
        vnlong (str or double): Normal longitude value.
        unlong (str or double): Longitude unit used in :class:`SkyCoord`
            for normal.
        vnlat (str or double): Normal latitude value.
        unlat (str): Latitude unit used in :class:`SkyCoord` for normal.

    Returns:
        img (str):plot of map

    """
    #  Transforming the reference position point into cartesian coordinates:
    sc = SkyCoord(
        vlong,
        vlat,
        distance=vdist,
        unit=(ulong, ulat, udist),
        frame=frame
    )
    r = sc.represent_as('cartesian').get_xyz().value

    #  Getting the normal to the plane in cartesian coordinates:
    wp = SkyCoord(
        vnlong,
        vnlat,
        unit=(unlong, unlat),
        frame=frame
    )
    w = wp.represent_as('cartesian').get_xyz().value

    wp=wp.transform_to('galactic')
    sc=sc.transform_to('galactic')

    #  Creating a direct base (u, v, w) using the normal vector:

    lon = np.radians(wp.l.degree)
    u = [np.sin(lon), -np.cos(lon), 0]
    u_latitude = 0 # arcsin(0) = 0 because u[2] = 0
    u_longitude = np.degrees(np.arctan2(u[1], u[0]))

    #  The last vector is just the results of the vector product of w and u
    #  (the order is important to keep the referential oriented as expected):
    v = np.cross(w, u)
    v_latitude = np.degrees(np.arcsin(v[2]))
    v_longitude = np.degrees(np.arctan2(v[1], v[0]))

    # maximum extension maximal of the slice
    #  Taking the two norm of the vector giving the sun position
    #  (if I have understand well the meaning of hw and step):
    f = np.linalg.norm(2 * hw * step, 2)
    #  Then we construct an array giving the cube border:
    c = np.arange(-f, f, step.min())

    #  Transforming this array into a mesh grid (it consist essentially of a
    #  coordinate matrix):
    cu, cv = np.meshgrid(c, c)

    #  The outer product of two vector u_1 and u_2 is equivalent to do the
    #  matrix multiplication u_1 u_2^T (T:: transpose). Here, the
    #  :func:`numpy.outer` function will flatten the matrix of size
    #  (amin(step), amin(step)) to have a vector of dimension $amin(step)^2$.
    #  Then it will multiple, following the matrix multiplication rule all
    #  coordinates of the argument to get in return a matrix of size
    #  (amin(step)^2, len(u)) (same for v and $r+s$ which should be of the same
    #  length).
    #  You can also search for the definition of the kroenecker product as the
    #  outer product is supposed to be a special case of it.
    #  This should give us a matrix to transform the cube coordinates into the
    #  plane coordinate, if I have understand everything correctly:

    cz = np.outer(cu, u) + np.outer(cv, v) + np.outer(np.ones_like(cu), r + s)

    z = np.ones([c.size * c.size])
    z[:] = None

    ix = np.floor((cz[:, 0]) / step[0])
    iy = np.floor((cz[:, 1]) / step[1])
    iz = np.floor((cz[:, 2]) / step[2])
    w = np.where(
        (ix >= 0) & (ix <= cube.shape[0] - 1) &
        (iy >= 0) & (iy <= cube.shape[1] - 1) &
        (iz >= 0) & (iz <= cube.shape[2] - 1)
    )

    z[w] = spi.interpn(
        points,
        cube,
        (ix[w], iy[w], iz[w]),
        method='linear',
        fill_value=1
    )
    z = np.reshape(z, [c.size, c.size])

    f = np.take(
        np.reshape(cu, [c.size * c.size]),
        w
    )
    wx = np.squeeze(
        np.where(
            (c >= np.amin(f)) & (c <= np.amax(f))
        )
    )
    f = np.take(
        np.reshape(cv, [c.size * c.size]),
        w
    )
    wy = np.squeeze(
        np.where(
            (c >= np.amin(f)) & (c <= np.amax(f))
        )
    )

    smap = z[np.ix_(wy, wx)]
    
    if v_latitude < 0:
        v_latitude = -v_latitude
        v_longitude = (v_longitude + 180)%360
        smap = smap[::-1]

    result = {}
    
    # x,y values for axes, log(z) values for a better representation of contour map
    addforLog0 = 1e-7
    result["addforlog0"] = addforLog0
    result["xTitle"] = "'Left to right towards ⇒ (l=%.1f°,b=%.1f°)'"%(u_longitude, u_latitude)
    result["yTitle"] = "'Bottom to top towards ⇒ (l=%.1f°,b=%.1f°)'"%(v_longitude, v_latitude)
    logsmap = np.log10(smap[::2,::2] + addforLog0)    
    # Calculate color bar values (5 values between [min, max])
    # Use log because log(z), and corresponding true value
    logScaletmp = np.linspace(np.nanmin(logsmap), np.nanmax(logsmap), 5)
    result["logScale"] = np.array2string(logScaletmp, separator=",")
    scaletmp = [format(np.exp(v) - addforLog0, '.2e').replace("e-0", "e-") for v in logScaletmp]
    result["scale"] = "[" + ", ".join(["'%s'"%v for v in scaletmp]) + "]"
    
    result["title"] = "'Origin (l=%.1f°,b=%.1f°), distance=%.1fpc --- Normal to the plane (l=%.1f°,b=%.1f°)'"%(
            sc.l.degree, sc.b.degree,  sc.distance.value,
            wp.l.degree, wp.b.degree)
    
    X=c[wx][::2]
    Y=c[wy][::2]
    Z=logsmap

    return result, X, Y, Z
