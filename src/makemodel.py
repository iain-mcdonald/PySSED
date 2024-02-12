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
import wget                                 # Required to download SVO filter files
from datetime import datetime               # Allows date and time to be printed
from pandas import read_csv                 # Much, much faster than Numpy.loadtxt

import pyssed                               # Import routines from main PySSED file

from dust_extinction.parameter_averages import F99   # Adopted dereddening law
import astropy.units as u                   # Required for astropy/astroquery/dust_extinction interfacing

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
        print ("Running makemodel version 20231115")

    if (verbosity>=30):
        print ("Setup file loaded.")

    if (verbosity>=30):
        print ("Model type:", model)

    # Load filters
    filtdata=get_filter_list()

    # List the model directory and grab the .dat files
    ls=(listdir("../models/"+model))
    files=[f for f in ls if ".dat" in f]

    # Set Av trials
    ext = F99(Rv=3.1) # Adopt standard Fitzpatrick 1999 reddening law
    Avs = np.array([0.1, 3.1, 10., 31.])
    Avnames = np.array(["lo","med","hi","vhi"])

    # Set up model flux and A_lambda arrays
    nmodels=len(files)
    modelflux=np.zeros((nmodels,len(filtdata['svoname'])+5))
    alambda=np.zeros((len(Avs),nmodels,len(filtdata['svoname'])+5))

    # Add headers to files
    headers=np.append(['#teff','logg','metal','alpha','lum'],filtdata['svoname'],axis=0)
    np.savetxt("model-"+model+".dat", np.expand_dims(headers,1), fmt='%s', delimiter=' ', newline=' ')
    for k in np.arange(len(Avs)):
        np.savetxt("alambda-"+model+"-"+Avnames[k]+".dat", np.expand_dims(headers,1), fmt='%s', delimiter=' ', newline=' ')    

    # Extract filter data
    wavelengths=[]
    transmissions=[]
    minw=np.zeros(len(filtdata['svoname']))
    maxw=np.zeros(len(filtdata['svoname']))
    for i in np.arange(len(filtdata['svoname'])):
        filt=filtdata['svoname'][i]
        filepath="../data/filters/"+filt.replace('/','.')+".xml"
        try:
            filttable=votable.parse_single_table(filepath).array
        # Download if it doesn't exist
        except:
            print ("Downloading filter data for",filtdata['svoname'][i])
            url = "http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="+filtdata['svoname'][i]
            filename = wget.download(url, out=filepath)
            filttable=votable.parse_single_table(filepath).array
        wavelengths.append(filttable['Wavelength'])
        transmissions.append(filttable['Transmission'])
        minw[i]=np.min(filttable['Wavelength'])
        maxw[i]=np.max(filttable['Wavelength'])

    try:
        time0=datetime.now()
    except:
        time0=0.
   
    # Loop over models
    for j in np.arange(nmodels):
#    for j in np.arange(7262,nmodels):
        f=files[j]
        modelflux[j,0]=float(f[1:6])
        modelflux[j,1]=float(f[7:12])
        modelflux[j,2]=float(f[13:18])
        modelflux[j,3]=float(f[19:24])
        try:
            time=datetime.now()
            if (j>0):
                avt=((time-time0).seconds+(time-time0).microseconds/1000000.)/j
                eta=(nmodels-j)*(time-time0)/j+datetime.now()
            else:
                avt=0.
                eta=0.
        except:
            time=1.
            avt=0.
            eta=0.
        print (j+1,"/",nmodels,f,":",float(f[1:6]),float(f[7:12]),float(f[13:18]),float(f[19:24]),":",time,avt,"s, ETA:",eta)
        modelpath="../models/"+model+"/"+f
#        modeltable=np.loadtxt(modelpath,dtype=float,delimiter=" ")
        modeltable=read_csv(modelpath,dtype=float,delimiter=" ").values
        # Convert F_lambda to F_nu in W/m^2/Hz
        # Multiply by Angstroms**2 and divide by 3e21 and multiply by 1e26
        # following https://www.stsci.edu/~strolger/docs/UNITS.txt
        modeltable[:,1]*=modeltable[:,0]
        modeltable[:,1]*=modeltable[:,0]
        modeltable[:,1]/=2.99792458E-05
        # Sum flux (Fnu dnu) to get luminosity
        hertz=299792458./(modeltable[:,0]/1.e10)
        dnu=-np.diff(hertz)
        lum=np.sum(modeltable[0:-1,1]*dnu)
        modelflux[j,4]=lum
        # Loop over filters
        for i in np.arange(len(filtdata['svoname'])):
            try:
                # If filter data lies within the bounds of the model (or close enough)
                shortacceptw=(maxw[i]-minw[i])*0.0+minw[i]
                longacceptw=maxw[i]-(maxw[i]-minw[i])*0.0
                if ((modeltable[0,0]<shortacceptw) & (modeltable[-1,0]>longacceptw)):
                    # Quicker to only interpolate over a subset of the model
                    # Rebin that model subset to approximately match the filter resolution
                    modelidx0=np.argmax(modeltable[:,0]>minw[i])
                    modelidx1=np.argmax(modeltable[:,0]>=maxw[i])
                    nmodelpoints=len(modeltable[(modeltable[:,0]>=minw[i]) & (modeltable[:,0]<=maxw[i]),:])
                    binfactor=np.floor(nmodelpoints/len(wavelengths[i])).astype(int)
                    if binfactor > 1:
                        # Select relevant indices
                        centralidx=int((modelidx1-modelidx0)/2.+modelidx0)
                        npointsrequired=np.floor(nmodelpoints/binfactor)*binfactor
                        distfromcentre=npointsrequired/2.
                        newmodelidx0=np.floor(centralidx-distfromcentre).astype(int)
                        newmodelidx1=(newmodelidx0+npointsrequired).astype(int)
                        if newmodelidx1 > len(modeltable[:,0]):
                            newmodelidx1 = len(modeltable[:,0])
                        # Extract those indices from unbinned model
                        unbmodelwave=modeltable[newmodelidx0:newmodelidx1,0]
                        unbmodelflux=modeltable[newmodelidx0:newmodelidx1,1]
                        # Do actual rebinning
                        bmodelwave=unbmodelwave.reshape((unbmodelwave.shape[0]//binfactor,binfactor,-1)).mean(axis=2).mean(1)
                        bmodelflux=unbmodelflux.reshape((unbmodelwave.shape[0]//binfactor,binfactor,-1)).mean(axis=2).mean(1)
                    else:
                        newmodelidx0=modelidx0
                        newmodelidx1=modelidx1
                        if newmodelidx1 == 0:
                            newmodelidx1 = len(modeltable[:,0])
                        bmodelwave=modeltable[newmodelidx0:newmodelidx1,0]
                        bmodelflux=modeltable[newmodelidx0:newmodelidx1,1]

                    # Interpolate filter transmission onto model data
                    # (Need to interpolate onto finer data structure to avoid undersampling)
                    # Make negative over-interpolations zero, then normalise
                    cs=interpolate.interp1d(wavelengths[i],transmissions[i])
                    filtinterp=cs(bmodelwave)
                    filtinterp[filtinterp<0]=0
                    filtinterp/=np.sum(filtinterp)
                    avwave=np.sum(bmodelwave*filtinterp) # Average wavelength
                    # Convolve filter transmission with model
                    modelenergy=bmodelflux*filtinterp
                    modelphotons=modelenergy*bmodelwave/avwave
                    modelflux[j,i+5]=np.sum(modelphotons) # assumes photon counting detector
                    alambda[k,j,0:4]=modelflux[j,0:4]
                    try:
                        unred = np.sum(modelphotons)
                        for k in np.arange(len(Avs)):
                            red = np.sum(modelphotons*ext.extinguish(bmodelwave*u.AA, Av=Avs[k]))
                            alambda[k,j,i+5]=-2.5*np.log10(red/unred)/Avs[k]
                    except Exception as e:
                        if (avwave<10000.):
                            alambda[k,j,i+5]=0.0
                        elif ((avwave>10000.) & (avwave<100000.)):
                            alambda[k,j,i+5]=0.1 # constant Alambda/Av=0.1 up to 10 microns
                        else:
                            alambda[k,j,i+5]=0.1/(avwave/100000.) # decrease Alambda/Av beyond 10 microns
                else:
                    modelflux[j,i+5]=-1.
                    for k in np.arange(len(Avs)):
                        alambda[k,j,0:4]=modelflux[j,0:4]
                        if (avwave<10000.):
                            alambda[k,j,i+5]=0.0
                        elif ((avwave>10000.) & (avwave<100000.)):
                            alambda[k,j,i+5]=0.1 # constant Alambda/Av=0.1 up to 10 microns
                        else:
                            alambda[k,j,i+5]=0.1/(avwave/100000.) # decrease Alambda/Av beyond 10 microns
#                print ("Filter",filtdata['svoname'][i])
#                print ("Binning factor",binfactor)
#                print ("Number of filter points",len(wavelengths[i]))
#                print ("Number of model points",len(bmodelwave))
#                print ("Original number of model points",len(unbmodelwave))
#                print ("Model points indices",newmodelidx0,"--",newmodelidx1)
#                print ("Filter wavelength range",np.min(wavelengths[i]),"--",np.max(wavelengths[i]))
#                print ("Model wavelength range",np.min(bmodelwave),"--",np.max(bmodelwave))
            except:
                print ("!!!!!!!! Fail on filter",filtdata['svoname'][i])
                print ("Binning factor",binfactor)
                print ("Number of filter points",len(wavelengths[i]))
                print ("Number of model points",len(bmodelwave))
                print ("Original number of model points",len(unbmodelwave))
                print ("Model points indices",newmodelidx0,"--",newmodelidx1)
                print ("Original model grid spans",np.min(modeltable[:,0]),np.max(modeltable[:,0]))
                print ("Filter wavelength range",np.min(wavelengths[i]),"--",np.max(wavelengths[i]))
                print ("Model wavelength range",np.min(bmodelwave),"--",np.max(bmodelwave))
                raise

        # Append output file with reduced model
        with open("model-"+model+".dat", "ab") as f:
            f.write(b"\n")
            np.savetxt(f,modelflux[j,:],fmt='%.6e',delimiter=' ',newline=' ')
        for k in np.arange(len(Avs)):
            with open("alambda-"+model+"-"+Avnames[k]+".dat", "ab") as f:
                f.write(b"\n")
                np.savetxt(f,alambda[k,j,:], fmt='%.6e', delimiter=' ', newline=' ')    

# -----------------------------------------------------------------------------
# Copied from pyssed.py
def get_filter_list():
    # Load filter list
    filtfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="FilterFile",1])[2:-2]
    if (verbosity>30):
        print ("Filter file",filtfile)
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
            setupfile=cmdargs[-1]
        makemodel(model,setupfile)