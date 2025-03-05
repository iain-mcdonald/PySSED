# Python Stellar SEDs (PySSED) model reduction package
# Author: Iain McDonald
# Description:
# - Create new set of reduced stellar atmosphere models for PySSED
version="20250221"
#
# Use: 
# makemodel.py <model name> <setup file>
#
# Caution:
# This package can take several hours or more to run completely and requires 
# separate download and processing of existing models. Once run, the main PySSED
# program then needs run with the RecomputeModelGrid flag set. Any new models will
# need pre-processed using the models/reduce.scr script. See manual for details.
#
# Method:
# The program takes pre-processed input stellar model spectra from the SVO and
# convolves them with astronomical filter transmission curves. The output is the
# expected number of photons per second per square metre of stellar surface
# CORRECTED for any post-processing imparted by surveys (e.g. to convert flux
# to magnitude). Reddening look-up tables are also generated at specified Av.
#
# Actual photometry goes through the process:
# Star - f(lambda) - emitted spectrum (ergs / s / cm^2 / Angstrom)
# ISM - A(lambda) - fractional absorption by ISM (unitless)
# Atmosphere+Filter+CCD - T(lambda) - transmission curves (unitless)
# Conversion - photoelectrons per photon (energy -> photons if photon counters)
# Spectral assumption - energy/photoelectrons to inferred flux (<f_lambda|nu>)
# Magnitude conversion - inferred flux to catalogue magnitude
#
# Our model must therefore take the model spectrum and...
# Compute ISM correction [alambda-*.dat tables]
# Account for transmission curves [tables from SVO]
# Convert to photoelectrons (E = hc / lambda)
# Account of an assumed spectrum (e.g. f_lambda / f_Vega)
# Allow conversion to magnitude (not done - PySSED converts mags to flux)
#
# In the zero-reddening case in the VEGAMAG and AB systems:
#     Vegamag      int [ f(lambda) T(lambda) lambda d lambda ]      int [ f_Vega(lambda) T(lambda) lambda d lambda ]
# <f_^       > = ------------------------------------------------ . ------------------------------------------------
#     lambda     int [ f_Vega(lambda) T(lambda) lambda d lambda ]          int [ T(lambda) lambda d lambda ]
#
# then:
# <f_nu> = <f_lambda> c / lambda^2
#
#     AB    int [ f(lambda) T(lambda) lambda d lambda ]
# <f_^  > = -------------------------------------------
#     nu        int [ T(lambda) c/lambda d lambda ]
#
# f_Vega(lambda) is sometimes substituted for a nu^-2 power law in infrared
# surveys (e.g. WISE) or a nu^0 law (e.g. IRAS, even for Vega magnitudes).
# Colour corrections ("K corrections") are needed in these cases.
#
# Thus, we need to:
# STEP 1 : Acquire our models
# STEP 2 : Compute f_lambda and <f_lambda> for Vega
# STEP 3 : Compute <f_lambda^Vegamag> and <f_nu^AB> for all models
# STEP 4 : Compute A(lambda) corrections for all models for all selected A_V.

from sys import argv                        # Allows command-line arguments
from sys import exit                        # Allows graceful quitting
from os import listdir                      # List files in a directory
import numpy as np                          # Required for numerical processing
from astropy.io import votable              # Required for extracting profiles from SVO filter files
from scipy import interpolate               # Required to regrid filter and model data
import wget                                 # Required to download SVO filter files
from datetime import datetime               # Allows date and time to be printed
from pandas import read_csv                 # Much, much faster than Numpy.loadtxt
import warnings                             # Allows dynamic warning suppression
import threading                            # Allows multi-threaded analysis

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
    verbosity=int(pyssedsetupdata[pyssedsetupdata[:,0]=="verbosity",1][0])
    if (verbosity>=30):
        print ("Running makemodel version",version)

    if (verbosity>=30):
        print ("Setup file loaded.")

    if (verbosity>=30):
        print ("Model type:", model)
        
    # Set limits
    modeltefflo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelTeffLo",1][0])
    modelteffhi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelTeffHi",1][0])
    modellogglo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelLoggLo",1][0])
    modellogghi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelLoggHi",1][0])
    modelfehlo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelFeHLo",1][0])
    modelfehhi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelFeHHi",1][0])
    modelafelo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelAFeLo",1][0])
    modelafehi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelAFeHi",1][0])

    # STEP 1: Load filters
    # --------------------
    filtdata=get_filter_list()
    svodata=get_svo_data(filtdata)

    # List the model directory and grab the .dat files
#    ls=(listdir("../models/"+model))
    ls=(listdir("../../testing5/models/"+model))
    files=[f for f in ls if ".dat" in f]

    # Set Av trials
    ext = F99(Rv=3.1) # Adopt standard Fitzpatrick 1999 reddening law
    Avs = np.array([0.1, 3.1, 10., 31.])
    Avnames = np.array(["lo","med","hi","vhi"])

    # Set up model flux and A_lambda arrays
    nmodels=len(files)
    modelflux=np.zeros((nmodels,len(filtdata['svoname'])+5))
    alambda=np.zeros((len(Avs),nmodels,len(filtdata['svoname'])+5))
    
    # Rage quit if no data
    if (nmodels==0):
        print ("CRITICAL FAIL! There are no model files!")
        raise
    else:
        if (verbosity>=30):
            print (nmodels,"model files found.")

    # Add headers to files
    headers=np.append(['#teff','logg','metal','alpha','lum'],filtdata['svoname'],axis=0)
    np.savetxt("model-"+model+".dat", np.expand_dims(headers,1), fmt='%s', delimiter=' ', newline=' ')
    for k in np.arange(len(Avs)):
        np.savetxt("alambda-"+model+"-"+Avnames[k]+".dat", np.expand_dims(headers,1), fmt='%s', delimiter=' ', newline=' ')    

    # Extract filter data
    wavelengths=[]
    transmissions=[]
    etrans=[]
    minw=np.zeros(len(filtdata['svoname']))
    maxw=np.zeros(len(filtdata['svoname']))
    vegazpt=np.empty(len(filtdata['svoname']))
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
        wavelengths.append(filttable['Wavelength'].data)
        transmissions.append(filttable['Transmission'].data)
        etrans.append(transmissions[i]*wavelengths[i])
#        print (wavelengths[i])
#        print (transmissions[i])
#        wavelengths.append(np.logical_not(filttable['Wavelength'].mask).nonzero())
#        transmissions.append(np.logical_not(filttable['Transmission'].mask).nonzero())
        minw[i]=np.min(filttable['Wavelength'])
        maxw[i]=np.max(filttable['Wavelength'])
        vegazpt[i]=float(svodata[svodata['svoname']==filt]['zpt'][0])

    # STEP 2 : Compute f_lambda and <f_lambda> for Vega
    # -------------------------------------------------
    # Vega projected area (viewed pole on, use equatorial radius of 2.726 R_Sun; 2012ApJ...761L...3M):
    vegaradius=(696000000.*2.726)
    vegadist=7.68*30856775814913673.
    vegascaling=(vegaradius/vegadist)**2 # r**-2 scaling
    vegaappradius=(vegaradius/149597870700.)/(vegadist/30856775814913673.)
    vegaangarea=np.pi*vegaappradius**2 # area of Vega in sq. arcsec.
    sunappradius=(696000000./149597870700.)/(vegadist/30856775814913673.)
    sunangarea=np.pi*sunappradius**2 # area of Sun in sq. arcsec.

    # Begin timer
    try:
        time0=datetime.now()
    except:
        time0=0.

    # Calculate <f_lambda^Vega> and <f_nu^Vega>
    print ("Performing Vega calculation...")
    # Vega model taken from here:
    vegamodelfile=pyssedsetupdata[pyssedsetupdata[:,0]=="VegaModelFile",1][0]
    # Set up arrays
    vegaenergy=np.empty((len(filtdata['svoname'])),dtype=object)
    #vegaphotons=np.empty((len(filtdata['svoname'])),dtype=object)
    lambda_eff_vega=np.empty((len(filtdata['svoname'])),dtype=float)
    lambda_mean=np.empty((len(filtdata['svoname'])),dtype=float)
    lambda_phot_vega=np.empty((len(filtdata['svoname'])),dtype=float)
    lambda_ref=np.empty((len(filtdata['svoname'])),dtype=float)
    vegaflux=np.empty((len(filtdata['svoname'])),dtype=object)
    f_lambda_Vega=np.empty((len(filtdata['svoname'])),dtype=object)
    f_nu_Vega=np.empty((len(filtdata['svoname'])),dtype=object)
    f=vegamodelfile
#    modelpath="../models/"+model+"/"+f
    modelpath="../../testing5/models/"+model+"/"+f
    modeltable=read_csv(modelpath,dtype=float,delimiter=" ").values
    # Convert F_lambda to F_nu in W/m^2/Hz
    # Multiply by Angstroms**2 and divide by 3e21 and multiply by 1e26
    # following https://www.stsci.edu/~strolger/docs/UNITS.txt
    #modeltable[:,1]*=modeltable[:,0]
    #modeltable[:,1]*=modeltable[:,0]
    #modeltable[:,1]/=2.99792458E-05
    # Sum flux (Fnu dnu) to get luminosity
    hertz=299792458./(modeltable[:,0]/1.e10)
    dnu=-np.diff(hertz)
    lum=np.sum(modeltable[0:-1,1]*dnu)
    print ("filtdata['svoname'][i],filtdata[i]['dataref'],lambda_eff_vega[i],lambda_mean[i],lambda_phot_vega[i],lambda_ref[i]")
    for i in np.arange(len(filtdata['svoname'])):
        try:
            # Non-zero only if filter data lies within the bounds of the model (or close enough)
            shortacceptw=(maxw[i]-minw[i])*0.0+minw[i]
            longacceptw=maxw[i]-(maxw[i]-minw[i])*0.0
            if ((modeltable[0,0]<shortacceptw) & (modeltable[-1,0]>longacceptw)):
                # Quicker to only interpolate over a subset of the model
                # Rebin that model subset to approximately match the filter resolution
                # bmodelwave = lambda
                # bmodelflux = f(lambda)
                bmodelwave,bmodelflux=binmodel(modeltable,minw[i],maxw[i],wavelengths[i])

                # Interpolate the binned model onto the wavelength grid of the transmission curve
                ct=interpolate.interp1d(bmodelwave,bmodelflux,bounds_error=False,fill_value=0.,assume_sorted=True)
                vegaflux[i]=ct(wavelengths[i])
                dlambda=np.diff(bmodelwave,append=0)
                dlambda=np.where(dlambda>0,dlambda,0)
                nu=(299792458./(bmodelwave/1.e10))

                # Check accuracy of interpolation: these should approximate the wavelengths below
                # Should be good to <0.1%
                #xlambda_eff_vega=np.sum(bmodelwave*nu**1*bmodelflux*Tlambda*dlambda)/np.sum(nu**1*bmodelflux*Tlambda*dlambda)
                #xlambda_mean=np.sum(bmodelwave*nu**1*Tlambda*dlambda)/np.sum(nu**1*Tlambda*dlambda)
                #xlambda_phot_vega=np.sum(bmodelwave**2*nu**1*bmodelflux*Tlambda*dlambda)/np.sum(bmodelwave*nu**1*bmodelflux*Tlambda*dlambda)

                nu=(299792458./(wavelengths[i]/1.e10))
                dlambda=np.diff(wavelengths[i],append=0)
                dlambda=np.where(dlambda>0,dlambda,0)
                
                # Multiplying by wavelength converts between energy and photon counting
                lambda_eff_vega[i]=np.sum(wavelengths[i]*etrans[i]*vegaflux[i]*dlambda)/np.sum(etrans[i]*vegaflux[i]*dlambda)
                lambda_mean[i]=np.sum(wavelengths[i]*etrans[i]*dlambda)/np.sum(etrans[i]*dlambda)
                lambda_phot_vega[i]=np.sum(wavelengths[i]**2*etrans[i]*vegaflux[i]*dlambda)/np.sum(wavelengths[i]*etrans[i]*vegaflux[i]*dlambda)
                lambda_ref[i]=np.sqrt(np.sum(etrans[i]*dlambda)/np.sum(etrans[i]/wavelengths[i]**2*dlambda))
                
                # Perform integrals
                vegaenergy[i]=vegaflux[i]*transmissions[i]*wavelengths[i]*dlambda/np.sum(transmissions[i]*wavelengths[i]*dlambda)
                f_lambda_Vega[i]=np.sum(vegaflux[i]*transmissions[i]*wavelengths[i]*dlambda)/np.sum(transmissions[i]*wavelengths[i]*dlambda)
                f_nu_Vega[i]=f_lambda_Vega[i]*(lambda_ref[i])**2/2.99792458e21*1.e26
                
                # Check against Vega zeropoint
                vegachecklambda=f_lambda_Vega[i]*vegascaling
                vegachecknu=vegachecklambda*(lambda_ref[i])**2/2.99792458e21*1.e26
                
                print ("VEGA:",filtdata['svoname'][i],filtdata[i]['dataref'],lambda_eff_vega[i],lambda_mean[i],lambda_phot_vega[i],lambda_ref[i],"           ",f_lambda_Vega[i],vegachecklambda,"      ",vegazpt[i],vegachecknu)

        except:
            print ("!!!! Vega: fail on filter",filtdata['svoname'][i])
            #print ("Binning factor",binfactor)
            print ("Number of filter points",len(wavelengths[i]))
            print ("Number of model points",len(bmodelwave))
            #print ("Original number of model points",len(unbmodelwave))
            #print ("Model points indices",newmodelidx0,"--",newmodelidx1)
            print ("Original model grid spans",np.min(modeltable[:,0]),np.max(modeltable[:,0]))
            print ("Filter wavelength range",np.min(wavelengths[i]),"--",np.max(wavelengths[i]))
            print ("Model wavelength range",np.min(bmodelwave),"--",np.max(bmodelwave))
            raise

    # Get tabulations of reddening
    reddening=np.empty((len(filtdata['svoname']),len(Avs)),dtype=object)
    for i in np.arange(len(filtdata['svoname'])):
        avwave=np.sum(wavelengths[i]*transmissions[i])/np.sum(transmissions[i])     # Average wavelength
        for k in np.arange(len(Avs)):
            try:
                    reddening[i,k] = ext.extinguish(wavelengths[i]*u.AA, Av=Avs[k])
            except Exception as e:
                if (avwave<10000.):
                    reddening[i,k]=0.0
                elif ((avwave>10000.) & (avwave<100000.)):
                    reddening[i,k]=10**((Avs[k]*0.1)/-2.5) # constant Alambda/Av=0.1 up to 10 microns
                else:
                    reddening[i,k]=10**((Avs[k]*0.1/(avwave/100000.))/-2.5) # decrease Alambda/Av beyond 10 microns
            #print (filtdata['svoname'][i],Avs[k],"    ",np.average(reddening[i,k]),"      ",-2.5*np.log10(np.average(reddening[i,k]))/Avs[k])
            #if ((filtdata['svoname'][i]=="WISE/WISE.W1") & (k==0)):
            #    print (reddening[i,k])
   
    # Loop over models
    print ("Executing main loop...")
    for j in np.arange(nmodels):
#    for j in np.arange(4262,nmodels):
            f=files[j]
            modelflux[j,0]=float(f[1:6])
            modelflux[j,1]=float(f[7:12])
            modelflux[j,2]=float(f[13:18])
            modelflux[j,3]=float(f[19:24])
            if ((modelflux[j,0]>modeltefflo) & (modelflux[j,0]<modelteffhi) & (modelflux[j,1]>modellogglo) & (modelflux[j,1]<modellogghi) & (modelflux[j,2]>modelfehlo) & (modelflux[j,2]<modelfehhi) & (modelflux[j,3]>modelafelo) & (modelflux[j,3]<modelafehi)):
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
    #            modelpath="../models/"+model+"/"+f
                modelpath="../../testing5/models/"+model+"/"+f
        #        modeltable=np.loadtxt(modelpath,dtype=float,delimiter=" ")
                modeltable=read_csv(modelpath,dtype=float,delimiter=" ").values
                # Convert F_lambda to F_nu in W/m^2/Hz
                # Multiply by Angstroms**2 and divide by 3e21 and multiply by 1e26
                # following https://www.stsci.edu/~strolger/docs/UNITS.txt
                #modeltable[:,1]*=modeltable[:,0]
                #modeltable[:,1]*=modeltable[:,0]
                #modeltable[:,1]/=2.99792458E-05
                # Sum flux (Fnu dnu) to get luminosity
                hertz=299792458./(modeltable[:,0]/1.e10)
                dnu=-np.diff(hertz)
                lum=np.sum(modeltable[0:-1,1]*modeltable[0:-1,0]**2/2.99792458E-05*dnu)
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
                            bmodelwave,bmodelflux=binmodel(modeltable,minw[i],maxw[i],wavelengths[i])

                            # Interpolate the binned model onto the wavelength grid of the transmission curve
                            ct=interpolate.interp1d(bmodelwave,bmodelflux,bounds_error=False,fill_value=0.,assume_sorted=True)
                            modelfluxes=ct(wavelengths[i])
                            nu=(299792458./(wavelengths[i]/1.e10))
                            dlambda=np.diff(wavelengths[i],append=0)
                            dlambda=np.where(dlambda>0,dlambda,0)

                            #     Vegamag      int [ f(lambda) T(lambda) lambda d lambda ]      int [ f_Vega(lambda) T(lambda) lambda d lambda ]
                            # <f_^       > = ------------------------------------------------ . ------------------------------------------------
                            #     lambda     int [ f_Vega(lambda) T(lambda) lambda d lambda ]          int [ T(lambda) lambda d lambda ]
                            #
                            #     nu^0      int [ f(lambda) T(lambda) lambda d lambda ]
                            # <f_^      > = -------------------------------------------
                            #     lambda        int [ T(lambda) c/lambda d lambda ]
                            #
                            c=299792458.e10
                            f_lambda_VEGAMAG=(modelfluxes*transmissions[i]*wavelengths[i]*dlambda)/np.sum(vegaflux[i]*transmissions[i]*wavelengths[i]*dlambda)*f_lambda_Vega[i]
                            f_lambda_AB=(modelfluxes*transmissions[i]*wavelengths[i]*dlambda)/np.sum(transmissions[i]/wavelengths[i]*dlambda)/lambda_ref[i]**2
                            f_lambda_VEGAMAG_E=(modelfluxes*etrans[i]*wavelengths[i]*dlambda)/np.sum(vegaflux[i]*etrans[i]*wavelengths[i]*dlambda)*f_lambda_Vega[i]
                            f_lambda_AB_E=(modelfluxes*etrans[i]*wavelengths[i]*dlambda)/np.sum(etrans[i]/wavelengths[i]*dlambda)/lambda_ref[i]**2
                            #print (modelflux[j,0],modelflux[j,1],modelflux[j,2],modelflux[j,3],i,filtdata['svoname'][i],lambda_ref[i],np.sum(f_lambda_VEGAMAG),np.sum(f_lambda_AB),np.sum(f_lambda_VEGAMAG_E),np.sum(f_lambda_AB_E),lambda_ref[i])

                            # <f_nu> = <f_lambda> lambda^2 / c
                            f_nu_VEGAMAG=f_lambda_VEGAMAG*(lambda_ref[i])**2/2.99792458e21*1.e26
                            f_nu_AB=f_lambda_AB*(lambda_ref[i])**2/2.99792458e21*1.e26
                            f_nu_VEGAMAG_E=f_lambda_VEGAMAG_E*(lambda_ref[i])**2/2.99792458e21*1.e26
                            f_nu_AB_E=f_lambda_AB_E*(lambda_ref[i])**2/2.99792458e21*1.e26
                                                      
                            if ("Vega" in filtdata[i]['dataref']) & ("nu0" not in filtdata[i]['dataref']): # Vega atmosphere has been assumed, else flat (nu^0) spectrum assumed
                                if ("E" in filtdata[i]['dataref']):
                                    modelflux[j,i+5]=np.sum(f_nu_VEGAMAG_E)
                                    modelphotons=f_nu_VEGAMAG_E*6.626e-34*299792458.e10/wavelengths[i]
                                else:
                                    modelflux[j,i+5]=np.sum(f_nu_VEGAMAG)
                                    modelphotons=f_nu_VEGAMAG*6.626e-34*299792458.e10/wavelengths[i]
                            else: # AB = nu^0 spectrum assumed
                                if ("E" in filtdata[i]['dataref']):
                                    modelflux[j,i+5]=np.sum(f_nu_AB_E)
                                    modelphotons=f_nu_AB_E*6.626e-34*299792458.e10/wavelengths[i]
                                else:
                                    modelflux[j,i+5]=np.sum(f_nu_AB)
                                    modelphotons=f_nu_AB*6.626e-34*299792458.e10/wavelengths[i]
                            print (modelflux[j,0],modelflux[j,1],modelflux[j,2],modelflux[j,3],i,filtdata['svoname'][i],lambda_ref[i],np.sum(f_nu_VEGAMAG),np.sum(f_nu_AB),-2.5*np.log10(np.sum(f_nu_VEGAMAG)/f_nu_Vega[i]),np.sum(f_nu_VEGAMAG)/f_nu_Vega[i],np.sum(f_nu_VEGAMAG)/np.sum(f_nu_AB),np.sum(f_nu_VEGAMAG_E)/np.sum(f_nu_AB_E),np.sum(f_nu_VEGAMAG_E)/np.sum(f_nu_VEGAMAG))

                            # Data headers
                            alambda[:,j,0:4]=modelflux[j,0:4]
#                            for k in np.arange(len(Avs)):
#                                try:
#                                    alambda[k,j,0:4]=modelflux[j,0:4]
#                                except:
#                                    print ("Modelflux[j,0:4]:",modelflux[j,0:4])
#                                    print ("Alambda[k,j,0:4]:",alambda[k,j,0:4])
#                                    raise
                            unred = np.sum(modelphotons)
                            for k in np.arange(len(Avs)):
                                try:
                                    red = np.sum(modelphotons*reddening[i,k])
                                except:
                                    print ("i,k,Filter,Av,reddening[i,k]:",i,k,filtdata['svoname'][i],Avs[k],reddening[i,k])
                                    raise
                                alambda[k,j,i+5]=-2.5*np.log10(red/unred)/Avs[k]
                        else:
                            modelflux[j,i+5]=-1.
                            alambda[:,j,0:4]=modelflux[j,0:4]
                            for k in np.arange(len(Avs)):
                                alambda[k,j,i+5]=np.average(reddening[i,k])
                    except:
                        print ("!!!!!!!! Fail on filter",filtdata['svoname'][i])
                        #print ("Binning factor",binfactor)
                        print ("Number of filter points",len(wavelengths[i]))
                        print ("Number of model points",len(bmodelwave))
                        #print ("Original number of model points",len(unbmodelwave))
                        #print ("Model points indices",newmodelidx0,"--",newmodelidx1)
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
# Bin a model spectrum
def binmodel(modeltable,minw,maxw,wavelengths):
    modelidx0=np.argmax(modeltable[:,0]>minw)
    modelidx1=np.argmax(modeltable[:,0]>=maxw)
    nmodelpoints=len(modeltable[(modeltable[:,0]>=minw) & (modeltable[:,0]<=maxw),:])
    binfactor=np.floor(nmodelpoints/len(wavelengths)).astype(int)
    if binfactor > 1: # Bin the spectrum
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
    else: # Don't bin the spectrum
        newmodelidx0=modelidx0
        newmodelidx1=modelidx1
        if newmodelidx1 == 0:
            newmodelidx1 = len(modeltable[:,0])
        bmodelwave=modeltable[newmodelidx0:newmodelidx1,0]
        bmodelflux=modeltable[newmodelidx0:newmodelidx1,1]
    return bmodelwave,bmodelflux

# -----------------------------------------------------------------------------
# Copied from pyssed.py
def get_filter_list():
    # Load filter list
    filtfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="FilterFile",1])[2:-2]
    if (verbosity>=30):
        print ("Filter file",filtfile)
    filtdata = np.loadtxt(filtfile, dtype=[('catname',object),('filtname',object),('errname',object),('svoname',object),('datatype',object),('dataref',object),('errtype',object),('mindata',float),('maxdata',float),('maxperr',float),('zptcorr',float)], comments="#", delimiter="\t", unpack=False)
    return filtdata

def get_svo_data(filtdata):
    # Get filter data from the Spanish Virtual Observatory

    # Define filter properties table
    filtprops=np.empty(len(filtdata['svoname']), dtype=[('svoname','U64'),('weff',float),('dw',float),('zpt',float)])
    
    # Loop over filters
    for i in np.arange(len(filtdata['svoname'])):
        # Define path for VOTable file and try to extract
        svoname=filtdata['svoname'][i]
        filepath="../data/filters/"+svoname.replace("/",".",1)+".xml"
        try:
            filt=votable.parse(filepath)
        # Download if it doesn't exist
        except:
            url = "http://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="+svoname
            filename = wget.download(url, out=filepath)
            filt=votable.parse(filepath)
        # Extract and tabulate the required properties
        filtprops[i]['svoname']=svoname
        filtprops[i]['weff']=filt.get_field_by_id('WavelengthEff').value
        filtprops[i]['dw']=filt.get_field_by_id('WidthEff').value
        filtprops[i]['zpt']=filt.get_field_by_id('ZeroPoint').value

    # This would load the filter transmission curves
    #filttable=votable.parse_single_table(filepath).array
    
    return filtprops

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
