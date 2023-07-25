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

""" functions """

def gaia_cone_search(sc, radius_arcmin, row_limit):
    radius = u.Quantity(radius_arcmin, u.arcmin)
    Gaia.ROW_LIMIT=np.int(row_limit)
    j = Gaia.cone_search_async(sc, radius)
    r = j.get_results()
    r.pprint()
    return r

def get_gaia_distances(r):
    r=r[np.where(r['parallax']>0.0)]
    r=r[np.where( np.abs(r['parallax_error']/r['parallax'])<0.5 )] 
    #filter r to remove targets without parallax measurement and where (relative) error is > 50%, i.e. select where abs(parallax_error/parallax)<0.5)
    gaia_distances = 1.0/r["parallax"]*1000.0 #very rudimentary and naive way to get the distance!!!
    return np.round(np.array(gaia_distances),0), r['designation']
