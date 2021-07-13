# Python Stellar SEDs (PySSED) model reduction package
# Author: Iain McDonald
# Description:
# - Create new set of reduced stellar atmosphere models for PySSED
# Use: 
# makemodel.py <model name>
#
# CAUTION: This package can take a day or more to run completely
# and requires separate download and processing of existing models

from sys import argv                        # Allows command-line arguments
from sys import exit                        # Allows graceful quitting
from os import listdir                      # List files in a directory
import numpy as np                          # Required for numerical processing
from astropy.io import votable              # Required for extracting profiles from SVO filter files
from scipy import interpolate               # Required to regrid filter and model data

import pyssed                               # Import routines from main PySSED file

# -----------------------------------------------------------------------------
def makemodel(model,setupfile):
    # Main routine
    errmsg=""

    # Load setup
    if (setupfile==""):
        setupfile="setup.default"
    global pyssedsetupdata      # Share setup parameters across all subroutines
    pyssedsetupdata = np.loadtxt(setupfile, dtype=str, comments="#", delimiter="\t", unpack=False)

    global verbosity        # Output level of chatter
    verbosity=int(pyssedsetupdata[pyssedsetupdata[:,0]=="verbosity",1])
    if (verbosity>=30):
        print ("Setup file loaded.")

    if (verbosity>=30):
        print ("Model type:", model)

    # Load filters
    filtdata=get_filter_list()

    # List the model directory and grab the .dat files
    ls=(listdir("../models/"+model))
    files=[f for f in ls if ".dat" in f]

    # Set up model flux array
    nmodels=len(files)
    modelflux=np.zeros((nmodels,len(filtdata['svoname'])+4))

    # Add headers to file
# Re-introduce these once complete
    headers=np.append(['#teff','logg','metal','alpha'],filtdata['svoname'],axis=0)
    np.savetxt("model-"+model+".dat", np.expand_dims(headers,1), fmt='%s', delimiter=' ', newline=' ')
    
    # Loop over models
    for j in np.arange(nmodels):
#    for j in np.arange(7262,nmodels):
        f=files[j]
        modelflux[j,0]=float(f[1:6])
        modelflux[j,1]=float(f[7:12])
        modelflux[j,2]=float(f[13:18])
        modelflux[j,3]=float(f[19:24])
        print (j+1,"/",nmodels,f,":",float(f[1:6]),float(f[7:12]),float(f[13:18]),float(f[19:24]))
        modelpath="../models/"+model+"/"+f
        modeltable=np.loadtxt(modelpath,dtype=float,delimiter=" ")
        # Convert F_lambda to F_nu
        modeltable[:,1]*=modeltable[:,0]
        modeltable[:,1]*=modeltable[:,0]
        modeltable[:,1]/=2.99792458E+21
        # Loop over filters
        for i in np.arange(len(filtdata['svoname'])):
            filt=filtdata['svoname'][i]
            filepath="../data/filters/"+filt.replace('/','.')+".xml"
            filttable=votable.parse_single_table(filepath).array
            # Extract the model points relevant for that filter
            minw=filttable['Wavelength'][0]
            maxw=filttable['Wavelength'][-1]
            modelsubset=modeltable[(modeltable[:,0]>=minw) & (modeltable[:,0]<=maxw),:]
            # Interpolate filter transmission onto model data
            # (Need to interpolate onto finer data structure to avoid undersampling)
            # Make negative over-interpolations zero, then normalise
            cs=interpolate.interp1d(filttable['Wavelength'],filttable['Transmission'])
            filtinterp=cs(modelsubset[:,0])
            filtinterp[filtinterp<0]=0
            filtinterp/=np.sum(filtinterp)
            # Convolve filter transmission with model
            modelflux[j,i+4]=np.sum(modelsubset[:,1]*filtinterp)

        # Append output file with reduced model
        with open("model-"+model+".dat", "ab") as f:
            f.write(b"\n")
            np.savetxt(f,modelflux[j,:],fmt='%.6e',delimiter=' ',newline=' ')

# -----------------------------------------------------------------------------
# Copied from pyssed.py
def get_filter_list():
    # Load filter list
    filtfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="FilterFile",1])[2:-2]
    filtdata = np.loadtxt(filtfile, dtype=[('catname',object),('filtname',object),('errname',object),('svoname',object),('datatype',object),('dataref',object),('errtype',object),('mindata',float),('maxdata',float),('maxperr',float),('zptcorr',float)], comments="#", delimiter="\t", unpack=False)
    return filtdata

# -----------------------------------------------------------------------------
# If running from the command line
if (__name__ == "__main__"):
    # Parse command line arguments
    cmdargs=argv
    if (len(cmdargs)==1):
        print ("Model name required")
    else:
        model=cmdargs[1]
        setupfile="setup.default"
        if (len(cmdargs)>2):
            setupfile=procargs[-1]
        makemodel(model,setupfile)