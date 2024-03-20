# Python Stellar SEDs (PySSED)
# Author: Iain McDonald
# Description:
# - Download and extract data on multi-wavelength catalogues of astronomical objects/regions of interest
# - Automatically process photometry into one or more stellar SEDs
# - Fit those SEDs with stellar parameters
# Additional thanks to: Peter Scicluna for help with interpolation
# Use:
# python3 pyssed.py <search type> <target> <fit type> [fit parameters] [setup file]
# Run as "python3 pyssed.py" to display options.

# =============================================================================
# START-UP IMPORTS
# =============================================================================
try:
    global speedtest
    global globaltime
    speedtest=False # Use this to print timing information
    from datetime import datetime               # Allows date and time to be printed
    if (speedtest):
        starttime=datetime.now()
        globaltime=starttime
        print ("start:",datetime.now())
    from sys import argv                        # Allows command-line arguments
    from sys import exit                        # Allows graceful quitting
    from sys import exc_info                    # Allows graceful error trapping
    from sys import stdout,stderr               # Allows error printing
    from sys import getsizeof                   # Allows memory testing
    from os import remove                       # Allows removal of old files
    import time                                 # Allows sleep functions
    import warnings                             # Allows dynamic warning suppression
    import tracemalloc                          # Required for memory issues
    tracemalloc.start()
    if (speedtest):
        print ("sys:",datetime.now()-starttime,"s")
    # Astroquery takes a long time - do this first
    from astroquery.simbad import Simbad        # Allow SIMBAD queries
    from astroquery.vizier import Vizier        # Allow VizieR queries
    from astroquery.gaia import Gaia            # Allow Gaia queries
#    from astroquery.utils import TableList      # Encode data from files
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    if (speedtest):
        print ("astroquery:",datetime.now()-starttime,"s")
    import numpy as np                          # Required for numerical processing
    import numpy.lib.recfunctions as rfn        # [For consistency with Visualiser]
    if (speedtest):
        print ("numpy:",datetime.now()-starttime,"s")
    import scipy.optimize as optimize           # Required for fitting
    import scipy.interpolate as interpolate     # Required for model grid interpolation
    from scipy.spatial import cKDTree           # Fastest nearest-neighbour analysis
    from scipy.special import erf               # Error function
    if (speedtest):
        print ("scipy:",datetime.now()-starttime,"s")
    import astropy.units as u                   # Required for astropy/astroquery/dust_extinction interfacing
    from astropy.coordinates import match_coordinates_sky    # Required for cross-matching
    from astropy.coordinates import SkyCoord    # Required for astroquery and B1950->J2000
    from astropy.coordinates import FK5         # Required B1950->J2000 conversion
    from astropy.coordinates import Angle       # Required for sexagesimal conversion
    from astropy.io import votable              # Required for SVO interface
    from astropy.table import Table             # Used to load backup data
    if (speedtest):
        print ("astropy:",datetime.now()-starttime,"s")
    import matplotlib.pyplot as plt             # Required for plotting
    from matplotlib.colors import ListedColormap # Required for colour mapping
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes # Reqd. for subplots
    if (speedtest):
        print ("matplotlib:",datetime.now()-starttime,"s")

    import wget                                 # Required to download SVO filter files
    if (speedtest):
        print ("wget:",datetime.now()-starttime,"s")

    import itertools                            # Required for iterating model data
    if (speedtest):
        print ("itertools:",datetime.now()-starttime,"s")
    import pandas as pd                         # Required for model data interpolation
    if (speedtest):
        print ("pandas:",datetime.now()-starttime,"s")

    from dust_extinction.parameter_averages import F99   # Adopted dereddening law
    #import dustmaps                             # Needed for extinction correction
    #if (speedtest):
    #    print ("dustmaps:",datetime.now()-starttime,"s")
    #import beast                                # Needed for Bayesian fitting
    #if (speedtest):
    #    print ("beast:",datetime.now()-starttime,"s")

    from gtomo.sda.reddening import reddening as gtomo_reddening   # G-Tomo extinction correction
    from gtomo.sda.load_cube import load_cube   # G-Tomo extinction correction

except Exception as e:
    print ("PySSED! Problem importing modules. Additional information:")
    print ("-----------------------------------------------------")
    print ("The following modules are REQUIRED:")
    print ("   sys numpy scipy astropy pandas itertools")
    print ("   matplotlib mpl_toolkits wget astroquery")
    print ("The following modules are REQUIRED for some compoents")
    print ("   dust_extinction - for any reddening correction")
    #print ("   dustmaps - for 2D/3D dust extinction")
    #print ("   beast - for full Bayesian fits")
    print ("   gtomo - an EXPLORE package for dereddening [included]")
    print ("   h5py - file reader to read GTomo files")
    print ("The following modules are OPTIONAL:")
    print ("   datetime - to correctly display timing information")
    print ("   time - to correctly wait for server downtime")
    print ("PySSED will try to valiantly soldier on regardless...")
    print ("-----------------------------------------------------")
    print ("Error information:")
    print_fail (exc_info())

if (speedtest):
    print ("Initialisation took:",datetime.now()-starttime,"s")
    
#exit()
    
# =============================================================================
# COLOUR MESSAGE PRINTING
# =============================================================================
# Colored printing functions for strings that use universal ANSI escape sequences.
# fail: bold red, pass: bold green, warn: bold yellow, 
# info: bold blue, bold: bold white
# from: https://stackoverflow.com/questions/39473297/how-do-i-print-colored-output-with-python-3
def print_fail(message):
    stderr.write('\x1b[1;31;41m' + message.strip() + '\x1b[0m' + '\n')

def print_pass(message):
    stdout.write('\x1b[1;32;42m' + message.strip() + '\x1b[0m' + '\n')

def print_warn(message):
    stderr.write('\x1b[1;33;43m' + message.strip() + '\x1b[0m' + '\n')

def print_info(message):
    stdout.write('\x1b[1;34;44m' + message.strip() + '\x1b[0m' + '\n')

def print_bold(message):
    stdout.write('\x1b[1;37;47m' + message.strip() + '\x1b[0m' + '\n')

# =============================================================================
# ARRAY SQUEEZING
# =============================================================================
# Safe return of first element of array
def reducto(a):
    try:
        a=a[0]
    except:
        pass
    return a
# Safe return of last element of array
def reducta(a):
    try:
        a=a[-1]
    except:
        pass
    return a
# String is float
def isfloat(x):
    try:
        a = float(x)
    except (TypeError, ValueError):
        return False
    else:
        return True
        
# =============================================================================
# COMMAND ARGUMENTS
# =============================================================================
def parse_args(cmdargs):
    # Parse command-line arguments
    error=0
    
    # If no command-line arguments, print usage
    if (len(argv)<=1):
        print ("Use:")
        print (__file__,"<search type> [parameters] <processing type> [parameters] [setup file]")
        print ("<search type> : single, parameters = 'Source name' or 'RA, Dec'")
        print ("<search type> : list, parameters = 'File with sources or RA/Dec'")
        print ("<search type> : cone, parameters = RA, Dec, radius (deg)")
        print ("<search type> : rectangle, parameters = RA, Dec, width, height (deg)")
        print ("<search type> : box, parameters = RA1, Dec1, RA2, Dec2 (deg)")
#        print ("<search type> : volume, parameters = RA, Dec, d, r (deg,pc)")
#        print ("<search type> : criteria, parameters = 'SIMBAD criteria setup file'")
#        print ("<search type> : complex, parameters = 'Gaia criteria setup file'")
#        print ("<search type> : nongaia, parameters = 'Non-gaia criteria setup file'")
        print ("<search type> : uselast")
        print ("<processing type> : none")
        print ("<processing type> : simple, parameters = Fe/H, E(B-V), [mass]")
#        print ("<processing type> : fit, parameters = [priors file]")
#        print ("<processing type> : binary, parameters = [priors file]")
        print ("[setup file] : contains additional properties related to surveys and fitting")
        cmdtype=""
        cmdargs=[]
        proctype=""
        procargs=[]
        error=1
    else:
        print ("PySSED has been asked to...")

    # Parse search type
    if (len(cmdargs)>1):
        cmdtype=cmdargs[1]
        if (cmdtype=="single"):
            print ("(1) Process single object:",cmdargs[2])
            cmdparams=cmdargs[2]
            procargs=cmdargs[3:]
        elif (cmdtype=="list"):
            print ("(1) Process a list of objects")
            procargs=cmdargs[3:]
            cmdparams=cmdargs[2]
            print ("    List of targets in:",cmdparams)
        elif (cmdtype=="cone"):
            print ("(1) Process a cone search",cmdargs[2],cmdargs[3],cmdargs[4])
            cmdparams=cmdargs[2:4]
            procargs=cmdargs[5:]
        elif (cmdtype=="rectangle"):
            print ("(1) Process a rectangle search:",cmdargs[2],cmdargs[3],"-",cmdargs[4],cmdargs[5])
            cmdparams=cmdargs[2:5]
            procargs=cmdargs[6:]
        elif (cmdtype=="box"):
            print ("(1) Process a box search:",cmdargs[2],cmdargs[3],"-",cmdargs[4],cmdargs[5])
            cmdparams=cmdargs[2:5]
            procargs=cmdargs[6:]
        elif (cmdtype=="volume"):
            print ("(1) Process a volume:",cmdargs[2],cmdargs[3],cmdargs[4],cmdargs[5])
            cmdparams=cmdargs[2:5]
            procargs=cmdargs[6:]
        elif (cmdtype=="criteria"):
            print ("(1) Process a set of SIMBAD criteria")
            cmdparams=cmdargs[2]
            procargs=cmdargs[3:]
            print ("    SIMBAD query in:",cmdparams)
        elif (cmdtype=="complex"):
            print ("(1) Process a complex Gaia query")
            cmdparams=cmdargs[2]
            procargs=cmdargs[3:]
            print ("    Gaia query in:",cmdparams)
        elif (cmdtype=="nongaia"):
            print ("(1) Process a set of objects from a non-Gaia source set")
            cmdparams=cmdargs[2]
            procargs=cmdargs[3:]
            print ("    Setup file:",cmdparams)
        elif (cmdtype=="uselast"):
            print ("(1) Use the last set of downloaded data")
            cmdparams=[]
            procargs=cmdargs[3:]
            print ("    Setup file:",cmdparams)
        else:
            print_fail ("ERROR! Search type was:"+cmdtype)
            print ("Expected one of: single, list, cone, rectangle, box, volume, criteria, complex, nongaia, uselast")
            cmdtype=""
            cmdparams=[]
            procargs=[]
            proctype=""
            procparams=[]
            error=1
    else:
        print_fail ("ERROR! No command type specified")
        print ("Expected one of: single, list, cone, rectangle, box, volume, criteria, complex, nongaia")
        cmdtype=""
        cmdparams=[]
        procargs=[]
        proctype=""
        procparams=[]
        error=1

    # Parse processing arguments
    setupfile="setup.default"
    if (len(procargs)>0):
        proctype=procargs[0]
        if (proctype=="none"):
            print ("(2) No processing: only create the SEDs")
            if (len(procargs)>1):
                setupfile=procargs[-1]
            procparams=[]
        elif (proctype=="bb"):
            print ("(2) Perform a blackbody fit")
            if (len(procargs)>1):
                setupfile=procargs[-1]
            procparams=[]
        elif (proctype=="trap"):
            print ("(2) Perform a trapezoidal fit")
            if (len(procargs)>1):
                setupfile=procargs[-1]
            procparams=[]
        elif (proctype=="simple"):
            print ("(2) Perform a simple (fast) fit")
            try:
                setupfile=float(procargs[-1])
            except:
                if (len(procargs)>1):
                    setupfile=procargs[-1]
            procparams=procargs[1:3]
        elif (proctype=="fit" or proctype=="binary"):
            if (proctype=="fit"):
                print ("(2) Perform a full parametric fit")
            if (proctype=="binary"):
                print ("(2) Perform a parametric binary fit")
            if (len(procargs)>2):
                setupfile=procargs[-1]
            if (len(procargs)>1):
                procparams=procargs[1]
            else:
                procparams="constraints.default"
            print ("    Using constraints file:",procparams)
        else:
            print_fail ("ERROR! Processing type was: "+procargs[0])
            print ("Expected one of: none, bb, trap, simple, fit, binary")
            proctype=""
            procparams=[]
            error=1
    else:
        print_fail ("ERROR! No processing type specified")
        print ("Expected one of: none, bb, simple, fit, binary")
        proctype=""
        procparams=[]
        error=1
        
    print ("    Using setup file:",setupfile)
    print ("")

    return cmdtype,cmdparams,proctype,procparams,setupfile,error
    
# =============================================================================
# FILE INTERACTION AND DATA LOADING FUNCTIONS
# =============================================================================
def get_catalogue_list():
    # Load catalogue list
    catfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="SurveyFile",1])[2:-2]
    poserrmult=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PosErrMult",1][0])
    catdata = np.loadtxt(catfile, dtype=[('server',object),('catname',object),('cdstable',object),('idcol',object),('epoch',float),('beamsize',float),('matchr',float)], comments="#", delimiter="\t", unpack=False)
    catdata['matchr']*=poserrmult
    return catdata

# -----------------------------------------------------------------------------
def get_filter_list():
    # Load filter list
    filtfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="FilterFile",1])[2:-2]
    filtdata = np.loadtxt(filtfile, dtype=[('catname',object),('filtname',object),('errname',object),('svoname',object),('datatype',object),('dataref',object),('errtype',object),('mindata',float),('maxdata',float),('maxperr',float),('zptcorr',float)], comments="#", delimiter="\t", unpack=False)
    return filtdata

# -----------------------------------------------------------------------------
def get_ancillary_list():
    # Load list of ancillary data queries
    ancillaryfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncillaryFile",1])[2:-2]
    poserrmult=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PosErrMult",1][0])
    ancillarydata = np.loadtxt(ancillaryfile, dtype=[('server',object),('catname',object),('cdstable',object),('idcol',object),('colname',object),('errname',object),('paramname',object),('units',object),('multiplier',float),('epoch',float),('beamsize',float),('matchr',float),('priority',int)], comments="#", delimiter="\t", unpack=False)
    ancillarydata['matchr']*=poserrmult
    return ancillarydata

# -----------------------------------------------------------------------------
def get_reject_list():
    # Load list of reject reasons
    rejectfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="RejectFile",1])[2:-2]
    rejectdata = np.loadtxt(rejectfile, dtype=[('catname',object),('filtname',object),('column',object),('position',int),('logical',object),('value',object),('result',object),('rejcat',object),('rejcol',object)], comments="#", delimiter="\t", unpack=False)
    return rejectdata

# -----------------------------------------------------------------------------
def get_model_grid():
    # Load grid of precomputed stellar models

    if (verbosity>=20):
        print ("Getting stellar models")
    
    # If required, create grid from reduced stellar model list
    recastmodels=int(pyssedsetupdata[pyssedsetupdata[:,0]=="RecomputeModelGrid",1][0])
    if (recastmodels):
        get_model_list()

    modelcode=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelCode",1])[2:-2]
    
    modelfile="model-"+modelcode+"-recast.dat"
    if (verbosity>=30):
        print ("Using model file:", modelfile)
    #modeldata = np.genfromtxt(modelfile, comments="#", names=True, deletechars=" ~!@#$%^&*()=+~\|]}[{';: ?>,<")
    # Although this loads the data twice, it's much quicker than using np.genfromtxt!
    df = pd.read_csv(modelfile,dtype=float,delimiter=" ")
    modeldata = np.array(pd.read_csv(modelfile,dtype=float,delimiter=" ").to_records(index=False))
    modeldata.dtype.names=df.columns
    
    return modeldata
    
# -----------------------------------------------------------------------------
def get_av_grid():
    # Load extinction grids, like get_model_grid

    if (verbosity>=30):
        print ("Loading extinction grids...")
    # No need to recast models since already run get_model_grid

    modelcode=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelCode",1])[2:-2]

    avfile="alambda-"+modelcode+"-lo-recast.dat"
    df = pd.read_csv(avfile,dtype=float,delimiter=" ")
    avgrid0 = np.array(pd.read_csv(avfile,dtype=float,delimiter=" ").to_records(index=False))
    avgrid0.dtype.names=df.columns
    avgrid = np.expand_dims(avgrid0,0)

    avfile="alambda-"+modelcode+"-med-recast.dat"
    df = pd.read_csv(avfile,dtype=float,delimiter=" ")
    avgrid1 = np.array(pd.read_csv(avfile,dtype=float,delimiter=" ").to_records(index=False))
    avgrid1.dtype.names=df.columns
    avgrid = np.append(avgrid,np.expand_dims(avgrid1,0),axis=0)

    avfile="alambda-"+modelcode+"-hi-recast.dat"
    df = pd.read_csv(avfile,dtype=float,delimiter=" ")
    avgrid2 = np.array(pd.read_csv(avfile,dtype=float,delimiter=" ").to_records(index=False))
    avgrid2.dtype.names=df.columns
    avgrid = np.append(avgrid,np.expand_dims(avgrid2,0),axis=0)

    avfile="alambda-"+modelcode+"-vhi-recast.dat"
    df = pd.read_csv(avfile,dtype=float,delimiter=" ")
    avgrid3 = np.array(pd.read_csv(avfile,dtype=float,delimiter=" ").to_records(index=False))
    avgrid3.dtype.names=df.columns
    avgrid = np.append(avgrid,np.expand_dims(avgrid3,0),axis=0)

    return avgrid

# -----------------------------------------------------------------------------
def get_model_list():
    # Translate list of model photometry into four-dimensional grid
    # This takes some time to do, but results in quicker interpolation later

    if (verbosity>=5):
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print ("Recasting model grid [RecomputeModelGrid>0].")
        print ("Have a coffee break - this may take some time (est. a fraction of an hour per filter).")
        print ("If this takes too long, try restricting the Model*Hi and Model*Lo parameters in the setup file.")

    # Load list of reduced models and extinction data
    # List of filenames should match those generated in makemodel.py
    modelcode=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelCode",1])[2:-2]
    nfiles=5
    modelfiles=np.empty(nfiles,dtype=object)
    recastmodelfiles=np.empty(nfiles,dtype=object)
    modelfiles[0]="model-"+modelcode+".dat"
    recastmodelfiles[0]="model-"+modelcode+"-recast.dat"
    modelfiles[1]="alambda-"+modelcode+"-lo.dat"
    recastmodelfiles[1]="alambda-"+modelcode+"-lo-recast.dat"
    modelfiles[2]="alambda-"+modelcode+"-med.dat"
    recastmodelfiles[2]="alambda-"+modelcode+"-med-recast.dat"
    modelfiles[3]="alambda-"+modelcode+"-hi.dat"
    recastmodelfiles[3]="alambda-"+modelcode+"-hi-recast.dat"
    modelfiles[4]="alambda-"+modelcode+"-vhi.dat"
    recastmodelfiles[4]="alambda-"+modelcode+"-vhi-recast.dat"

    try:
        start = datetime.now() # time object
    except:
        start = 0
    
    for k in np.arange(nfiles):
        modelfile=modelfiles[k]
        recastmodelfile=recastmodelfiles[k]
        if (verbosity>=10):
            print ("File",k+1,"of",nfiles,": recasting",modelfile,"to",recastmodelfile)
    
        modelfiledata = np.genfromtxt(modelfile, comments="#", names=True, deletechars=" ~!@#$%^&*()=+~\|]}[{';: ?>,<")

        # Remove outlying models
        modeltefflo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelTeffLo",1][0])
        modelteffhi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelTeffHi",1][0])
        modellogglo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelLoggLo",1][0])
        modellogghi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelLoggHi",1][0])
        modelfehlo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelFeHLo",1][0])
        modelfehhi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelFeHHi",1][0])
        modelafelo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelAFeLo",1][0])
        modelafehi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelAFeHi",1][0])
        modelfiledata = modelfiledata[(modelfiledata['teff']>=modeltefflo) & (modelfiledata['teff']<=modelteffhi)]
        modelfiledata = modelfiledata[(modelfiledata['logg']>=modellogglo) & (modelfiledata['logg']<=modellogghi)]
        modelfiledata = modelfiledata[(modelfiledata['metal']>=modelfehlo) & (modelfiledata['metal']<=modelfehhi)]
        modelfiledata = modelfiledata[(modelfiledata['alpha']>=modelafelo) & (modelfiledata['alpha']<=modelafehi)]

        # Separate parameters and values
        params=np.stack((modelfiledata['teff'],modelfiledata['logg'],modelfiledata['metal'],modelfiledata['alpha']),axis=1)
        valueselector = modelfiledata.dtype.names[4:]

        # Iterate parameters onto complete grid
        interp_grid_points = np.array(list(itertools.product(np.unique(modelfiledata['teff']),np.unique(modelfiledata['logg']),np.unique(modelfiledata['metal']),np.unique(modelfiledata['alpha']))))
        if (verbosity>=90):
            print ("Original grid contains",len(modelfiledata),"points")
            print ("Complete grid will contain",len(interp_grid_points),"points")

        # Set up output data grid
        interp_data_points = np.zeros((len(interp_grid_points[:,0]),len(valueselector)),dtype=float)
        #print (np.shape(interp_data_points))
        
        # Rescale temperature axis but set rescale=False in interpfn
        # This allows temperature to mostly control the flux, but gives more appropriate weight to log g, [Fe/H] and [alpha/Fe]
        rescaled_params = np.copy(params)
        rescaled_params[:,0]/=100.
        rescaled_grid_points = np.copy(interp_grid_points)
        rescaled_grid_points[:,0]/=100.
    
        # Recast onto rectilinear grid
        for i in np.arange(len(valueselector)): # loop over filters
            if (verbosity>=30):
                try: # fails if datetime not imported or i=0
                    #ftotal=nfiles*len(valueselector)
                    fremaining=(nfiles-k-1)*len(valueselector)+(len(valueselector)-i)
                    fdone=k*len(valueselector)+i
                    now=datetime.now()
                    elapsed=((now-start).seconds+(now-start).microseconds/1000000.)
                    remaining=elapsed/(fdone+1e-6)*fremaining
#                    eta=(len(valueselector)-i)*(now-start)/(i+1e-6)+datetime.now()
                    eta=datetime.now()+(now-start)/(fdone+1e-6)*fremaining
                except:
                    now = 0; elapsed = 0; remaining = 0
                print ("File",k+1,"of",nfiles,", filter", i+1, "of", len(valueselector), "[",now,"], elapsed:",int(elapsed)," ETA:",eta)

            ts=np.unique(modelfiledata['teff'])
            nts=len(ts)
            for j in np.arange(nts):
                mint = np.where(j-3<0,0,j-3)
#                if (ts[j]<3500):
#                    maxt = np.where(j+10>=nts,nts-1,j+10)
                if (ts[j]<3000):
                    maxt = np.where(j+8>=nts,nts-1,j+8)
                else:
                    maxt = np.where(j+3>=nts,nts-1,j+3)
                if (verbosity>=80):
                    print (ts[j])
                snip_params=params[(params[:,0]>=ts[mint]) & (params[:,0]<=ts[maxt])]
                snip_values=modelfiledata[(params[:,0]>=ts[mint]) & (params[:,0]<=ts[maxt])][list(valueselector)[i]]
                if (k==0): # for models
                    interpfn = interpolate.LinearNDInterpolator(snip_params,snip_values,rescale=True, fill_value=0.)
                else: # for Alambda/Av (quicker and need non-zero fill_value)
                    interpfn = interpolate.NearestNDInterpolator(snip_params,snip_values,rescale=True)
                #interpfn = interpolate.LinearNDInterpolator(snip_params,snip_values,rescale=True) # fill nan
                modeldata = interpfn(interp_grid_points[interp_grid_points[:,0]==ts[j]])
                interp_data_points[interp_grid_points[:,0]==ts[j],i]=np.squeeze(modeldata)
            
        if (verbosity>=30):
            print ("Done. Saving recast model file.")
        with open(modelfile, "r") as f:
            header=f.readline()
        with open(recastmodelfile, "w") as f:
            f.write(header)
        with open(recastmodelfile, "a") as f:
            np.savetxt(f,np.append(interp_grid_points,interp_data_points,axis=1), fmt='%s', delimiter=' ')
    if (verbosity>=5):
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print ("Now run shorten-model.scr")
    exit()

    return

# =============================================================================
# SERVER INTERACTION FUNCTIONS
# =============================================================================
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
# -----------------------------------------------------------------------------
def get_gaia_obj(cmdparams):
    # Extract the Gaia DR3 ID from the input data

    errmsg=""
    dr3_data=""
    dr3_obj=""
    maxattempts=int(pyssedsetupdata[pyssedsetupdata[:,0]=="ServerRetries",1][0])
    if (verbosity>=30):
        print ("Fitting single object")
    
    # Parse input parameters: convert to array and try to extract co-ordinates
    obj=str.split(cmdparams,sep=" ")

    # If specific Gaia keywords (DR3,DR2) are used, use the source name
    if (obj[0]=="Gaia"):
        obj=obj[1:]
    if (obj[0]=="DR3" or obj[0]=="DR2"):
        # If using a DR2 identifier, first cross-match to an DR3 source
        if (obj[0]=="DR2"):
            dr2_obj=obj[1]
            query="SELECT dr3_source_id FROM gaiadr3.dr2_neighbourhood WHERE dr2_source_id=" + obj[1]
            job = Gaia.launch_job(query)
            result_table=job.get_results()
            try:
                dr3_obj=result_table['dr3_source_id'][0]
                if (verbosity>40):
                    print ("Gaia DR2 source",obj[1],"is DR3 source",dr3_obj)
            except IndexError:
                if (verbosity>40):
                    print_warn ("No Gaia DR3 object")
                return 0,"No Gaia DR3 counterpart"
                dr3_obj=0.
        else:
            dr3_obj=obj[1]
            #coords=np.zeros((2,),dtype=float)
            if (verbosity>40):
                print ("Using Gaia DR3 source",obj[1])

    # Otherwise, if parsable co-ordinates are used, use those
    else:
        try:
            with warnings.catch_warnings(): # This should fail if a single value or non-numeric - only gives warning in NumPy 1.25
                warnings.filterwarnings("ignore", message="string or file could not be read to its end due to unmatched data")
                coords=np.fromstring(cmdparams,dtype=float,sep=" ")
            if (len(coords)==6):
                coords=np.array([coords[0]*15.+coords[1]/4.+coords[2]/240.,coords[3]+np.sign(coords[3])*(coords[4]/60.+coords[5]/3600.)])
            if (coords[0]!=-999 and coords[1]!=-999 and verbosity>40):
                print ("Using co-ordinates: RA",coords[0],"deg, Dec",coords[1],"deg")
        except:
    # and if not, then ask SIMBAD for a position resolution
            if (verbosity>40):
                print ("Resolving",cmdparams,"using SIMBAD")
            attempts = 0
            result_table = 0
            while attempts < maxattempts:
                try:
                    customSimbad = Simbad()
                    customSimbad.add_votable_fields('pmra','pmdec')
                    result_table = customSimbad.query_object(cmdparams)
                    break
                except:
                    attempts += 1
                    if (verbosity >= 25):
                        print_warn ("Could not connect to SIMBAD server (attempt "+str(attempts)+" of "+str(maxattempts)+") [Simbad.query_object]")
                    try: # wait for server to clear
                        time.sleep(attempts**2)
                    except: # if time not installed don't wait
                        pass
            if (attempts==maxattempts):
                print_fail ("Could not connect to SIMBAD server")
                raise Exception("Could not connect to SIMBAD server")
            try:
                gaiaepoch=float(pyssedsetupdata[pyssedsetupdata[:,0]=="GaiaEpoch",1][0])
                if (verbosity>=99):
                    print ("SIMBAD result table:")
                    print (result_table)
                if (verbosity>60):
                    print (result_table['RA'][0],result_table['DEC'][0])
                rac=np.fromstring(result_table['RA'][0],dtype=float,sep=" ")
                decc=np.fromstring(result_table['DEC'][0],dtype=float,sep=" ")
                try:
                    pmra=result_table['PMRA'][0]
                    pmdec=result_table['PMDEC'][0]
                except:
                    pmra=0; pmdec=0
                try:
                    if (result_table['DEC'][0][0]=="+"):
                        coords2000=[rac[0]*15.+rac[1]/4+rac[2]/240,decc[0]+decc[1]/60+decc[2]/3600]
                    else:
                        coords2000=[rac[0]*15.+rac[1]/4+rac[2]/240,decc[0]-decc[1]/60-decc[2]/3600]
                    pmoffset=[pmra/3.6e6*(gaiaepoch-2000.)/np.cos(coords2000[1]/180*np.pi),pmdec/3.6e6*(gaiaepoch-2000.)]
                    coords=np.add(coords2000,pmoffset)
                    if (verbosity>95):
                        print ("Epoch 2000 co-ordinates: RA",coords2000[0],"deg, Dec",coords2000[1],"deg")
                        print ("PM offset: RA",pmoffset[0],"deg, Dec",pmoffset[1],"deg")
                        print ("Epoch",gaiaepoch,"co-ordinates: RA",coords[0],"deg, Dec",coords[1],"deg")
                except IndexError as e:
                    print_fail ("Received malformed source co-ordinates. Could not identify source.")
                    return 0,"Source co-ordinates error"
            except TypeError as e:
                print_fail ("Source could not be identfied:"+cmdparams)
                return 0,"Source could not be identified"
            if (verbosity>40):
                print ("Using co-ordinates: RA",coords[0],"deg, Dec",coords[1],"deg")

        # Now query the Gaia DR3 database for a match
        dr3_obj=query_gaia_coords(coords[0],coords[1])
        if (verbosity>95):
            print ("Gaia returned:")
            print (dr3_obj)
        # If there's a match...
        if (dr3_obj>0):
            if (verbosity>40):
                print ("Gaia DR3",dr3_obj)
        # If there isn't a match, fall back to Hipparcos
        else:
            if (verbosity>40):
                print ("No Gaia object at that position.")

    #if (verbosity>=40):
    #    print (dr3_data)

    return dr3_obj,errmsg
    
# -----------------------------------------------------------------------------
def get_gaia_data(dr3_obj):
    query="SELECT * FROM gaiadr3.gaia_source WHERE source_id=" + str(dr3_obj)
    if (verbosity>70):
        print (query)
    maxattempts=int(pyssedsetupdata[pyssedsetupdata[:,0]=="ServerRetries",1][0])
    attempts = 0
    while attempts < maxattempts:
        try:
            job = Gaia.launch_job(query)
            break
        except:
            attempts += 1
            print_warn ("Could not connect to Gaia server (attempt "+str(attempts)+" of "+str(maxattempts)+") [Gaia.launch_job]")
            try: # wait for server to clear
                time.sleep(attempts**2)
            except: # if time not installed don't wait
                pass
    if (attempts==maxattempts):
        print_fail ("Could not connect to Gaia server")
        raise Exception("Could not connect to SIMBAD server")

    try:
        q=job.get_results()
    except:
        q=np.array([])

    return q

# -----------------------------------------------------------------------------
def get_gaia_list(cmdparams):
    # Extract data on a list of disparate single objects
    # As get_gaia_single
    # Can be multiplexed one object at a time, subject to not triggering rampaging robot alerts
    return

# -----------------------------------------------------------------------------
def get_gaia_volume(cmdparams):
    # Extract data in a volume around a location
    # Search a volume around a 3D point in space using similar principles
    return

# -----------------------------------------------------------------------------
def get_gaia_simbad_criteria(cmdparams):
    # Extract data on a category of objects using SIMBAD
    # Selection of specific object types within SIMBAD
    # Allow selection within a cone/box/volume
    # Allow restriction by proper motion / parallax
    return

# -----------------------------------------------------------------------------
def get_gaia_query(cmdparams):
    # Extract data on a category of objects using a complex Gaia query
    # Work as list objects
    return

# -----------------------------------------------------------------------------
def fit_nongaia(cmdparams):
    # Extract data from a list of sources not from Gaia
    # Start from a catalogue other than Gaia (e.g. HST, WISE, own observations)
    # Progressive match to Gaia catalogue, then match outwards
    return




# =============================================================================
# SERVER TABLE QUERIES
# =============================================================================
def extract_ra_dec(sourcedata):

    # Extract RA and Dec from a Vizier table
    try:
        newra=sourcedata['RArad']
        newdec=sourcedata['DErad']
        if (verbosity>=98):
            print ("Selected co-ordinates from: RArad/DErad")
    except: # Needed for Tycho
        try:
            newra=sourcedata['_RA.icrs']
            newdec=sourcedata['_DE.icrs']
            if (verbosity>=98):
                print ("Selected co-ordinates from: _RA.icrs/_DE.icrs")
        except: # Needed for SDSS
            try:
                newra=sourcedata['RA_ICRS']
                newdec=sourcedata['DE_ICRS']
                if (verbosity>=98):
                    print ("Selected co-ordinates from: RA_ICRS/DE_ICRS")
            except:
                try:
                    newra=sourcedata['RAJ2000']
                    newdec=sourcedata['DEJ2000']
                    if (verbosity>=98):
                        print ("Selected co-ordinates from: RAJ2000/DEJ2000")
                except:
                    try: # for data from files
                        newra=sourcedata['RA']
                        newdec=sourcedata['Dec']
                        if (verbosity>=98):
                            print ("Selected co-ordinates from: RA,Dec")
                    except: # Needed for Morel
                        try:
                            newra=sourcedata['_RA']
                            newdec=sourcedata['_DE']
                            if (verbosity>=98):
                                print ("Selected co-ordinates from: _RA/_DE")
                        except: # Needed for IRAS
                            try:
                                # Extract
                                newra=sourcedata['RA1950']
                                newdec=sourcedata['DE1950']
                                # Reformat to degrees
                                coords=SkyCoord(ra=newra, dec=newdec, unit='deg', frame=FK5, equinox='B1950.0').transform_to(FK5(equinox='B1950.0'))
                                newra=coords.ra.deg*15.
                                newdec=coords.dec.deg
                                # Convert to J2000
                                coords=SkyCoord(ra=newra, dec=newdec, unit='deg', frame=FK5, equinox='B1950.0').transform_to(FK5(equinox='J2000.0'))
                                newra=coords.ra.deg
                                newdec=coords.dec.deg
                                if (verbosity>=98):
                                    print ("Selected co-ordinates from: RA1950/DE1950")
                            except: # Needed for really old catalogues
                                try:
                                    # Extract
                                    newra=sourcedata['RA1900']
                                    newdec=sourcedata['DE1900']
                                    # Reformat to degrees
                                    coords=SkyCoord(ra=newra, dec=newdec, unit='deg', frame=FK5, equinox='B1900.0').transform_to(FK5(equinox='B1900.0'))
                                    newra=coords.ra.deg*15.
                                    newdec=coords.dec.deg
                                    # Convert to J2000
                                    coords=SkyCoord(ra=newra, dec=newdec, unit='deg', frame=FK5, equinox='B1900.0').transform_to(FK5(equinox='J2000.0'))
                                    newra=coords.ra.deg
                                    newdec=coords.dec.deg
                                    if (verbosity>=98):
                                        print ("Selected co-ordinates from: RA1950/DE1950")
                                except: # Needed for Skymapper
                                    try:
                                        newra=sourcedata['RAICRS']
                                        newdec=sourcedata['DEICRS']
                                        if (verbosity>=98):
                                            print ("Selected co-ordinates from: RAICRS,DEICRS")
                                    except: # Resort to pre-computed values, which undoes PM correction
                                        try:
                                            newra=sourcedata['_RAJ2000']
                                            newdec=sourcedata['_DEJ2000']
                                            if (verbosity>=98):
                                                print ("Selected co-ordinates from: _RAJ2000/_DEJ2000")
                                        except:
                                            print_fail ("Failed to find a co-ordinate system")
    # Convert sexagesimal to degrees
    if ((newra.dtype!="float64") & (newra.dtype!="float32") & (newra.dtype!="float")):
        if (verbosity>=99):
            print_warn ("Converting from sexagesimal!")
            print (newra[0],newra.dtype)
        newra=Angle(np.char.add(newra,np.array(["hours"]))).degree
        newdec=Angle(np.char.add(newdec,np.array(["degrees"]))).degree

    return newra,newdec

# -----------------------------------------------------------------------------
def extract_ra_dec_pm(sourcedata):

    # Extract RA and Dec and proper motion from a Vizier table
    pmtype=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PMCorrType",1][0])
    pmra0=float(pyssedsetupdata[pyssedsetupdata[:,0]=="FixedPMRA",1][0])
    pmdec0=float(pyssedsetupdata[pyssedsetupdata[:,0]=="FixedPMDec",1][0])
    pmracol=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PMRAColID",1])[2:-2]
    pmdeccol=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PMDecColID",1])[2:-2]
    pmraerrcol=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PMRAErrColID",1])[2:-2]
    pmdecerrcol=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PMDecErrColID",1])[2:-2]
    if (verbosity>95):
        print ("PM columns:",pmracol,pmdeccol,pmraerrcol,pmdecerrcol)
        print(np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PMRAColID",1]))
        print(np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PMDecColID",1]))
        print(np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PMRAErrColID",1]))
        print(np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PMDecErrColID",1]))

    try:
        if (verbosity>95):
            print ("Trying Gaia...")
        sourcetype="Gaia"
        sourcera=(sourcedata['ra']).astype(float)
        sourcedec=(sourcedata['dec']).astype(float)
        sourceraerr=(sourcedata['ra_error']).astype(float)/3600000.
        sourcedecerr=(sourcedata['dec_error']).astype(float)/3600000.
        if ((sourcedata['pmra']!="--") and (sourcedata['pmdec']!="--")):
            sourcepmra=(sourcedata['pmra']).astype(float)
            sourcepmdec=(sourcedata['pmdec']).astype(float)
            sourcepmraerr=(sourcedata['pmra_error']).astype(float)
            sourcepmdecerr=(sourcedata['pmdec_error']).astype(float)
        else:
            sourcepmra=0.
            sourcepmdec=0.
            sourcepmraerr=0.
            sourcepmdecerr=0.
        sourceepoch=float(pyssedsetupdata[pyssedsetupdata[:,0]=="GaiaEpoch",1][0])
        if (verbosity>95):
            print ("Gaia data found.")
    except:
        # If that fails, try alternative Gaia format
        try:
            sourcetype="Gaia"
            sourcera=(sourcedata['RA_ICRS']).astype(float)
            sourcedec=(sourcedata['DE_ICRS']).astype(float)
            sourceraerr=(sourcedata['e_RA_ICRS']).astype(float)/3600000.
            sourcedecerr=(sourcedata['e_DE_ICRS']).astype(float)/3600000.
            try:
                #sourcedata['pmRA']=np.nan_to_num(np.genfromtxt(sourcedata['pmRA']))
                #sourcedata['pmDE']=np.nan_to_num(np.genfromtxt(sourcedata['pmDE']))
                #sourcedata['e_pmRA']=np.nan_to_num(np.genfromtxt(sourcedata['e_pmRA']))
                #sourcedata['e_pmDE']=np.nan_to_num(np.genfromtxt(sourcedata['e_pmDE']))
                sourcepmra=(sourcedata['pmRA']).astype(float)
                sourcepmdec=(sourcedata['pmDE']).astype(float)
                sourcepmraerr=(sourcedata['e_pmRA']).astype(float)
                sourcepmdecerr=(sourcedata['e_pmDE']).astype(float)
            except TypeError or AttributeError: # if non-numeric result
                sourcepmra=0.
                sourcepmdec=0.
                sourcepmraerr=0.
                sourcepmdecerr=0.
            sourceepoch=float(pyssedsetupdata[pyssedsetupdata[:,0]=="GaiaEpoch",1][0])
            if (verbosity>95):
                print ("Gaia data found (alt).")
        except:
            # If that fails, try Hipparcos
            try:
                if (verbosity>95):
                    print ("Trying Hipparcos...")
                sourcetype="Hipparcos"
                sourcera=(sourcedata['RArad']).astype(float)
                sourcedec=(sourcedata['DErad']).astype(float)
                sourceraerr=(sourcedata['e_RArad']).astype(float)/3600000.
                sourcedecerr=(sourcedata['e_DErad']).astype(float)/3600000.
                sourcepmra=(sourcedata['pmRA']).astype(float)
                sourcepmdec=(sourcedata['pmDE']).astype(float)
                sourcepmraerr=(sourcedata['e_pmRA']).astype(float)
                sourcepmdecerr=(sourcedata['e_pmDE']).astype(float)
                sourceepoch=1991.25
                if (verbosity>95):
                    print ("Hipparcos data found.")
            except:
                # If that fails, try user-specified file-type data
                try:
                    if (verbosity>95):
                        print ("Trying user-specified data...")
                    sourcetype="User"
                    # Allow different capitalisations
                    try: # Sexagesimal
                        try:
                            rac=np.fromstring(sourcedata['RA'],dtype=float,sep=" ")
                        except:
                            try:
                                rac=np.fromstring(sourcedata['Ra'],dtype=float,sep=" ")
                            except:
                                rac=np.fromstring(sourcedata['ra'],dtype=float,sep=" ")
                        try:
                            decc=np.fromstring(sourcedata['DEC'],dtype=float,sep=" ")
                        except:
                            decc=np.fromstring(sourcedata['Dec'],dtype=float,sep=" ")
                        sourcera=rac[0]*15.+rac[1]/4+rac[2]/240
                        sourcedec=abs(decc[0])/decc[0]*(abs(decc[0])+decc[1]/60+decc[2]/3600)
                    except: # Decimal degrees
                        sourcera=sourcedata['RA']
                        sourcedec=sourcedata['Dec']
                    length=0
                    try:
                        length=len(sourcera)
                    except:
                        pass
                    if (length>0):
                        zero=np.zeros(length)
                    else:
                        zero=0.
                    sourceraerr=zero
                    sourcedecerr=zero
                    if (pmtype==0):
                        sourcepmra=zero
                        sourcepmdec=zero
                        sourcepmraerr=zero
                        sourcepmdecerr=zero
                    elif (pmtype==1):
                        try:
                            sourcepmra=(sourcedata[pmracol]).astype(float)
                            sourcepmdec=(sourcedata[pmdeccol]).astype(float)
                            sourcepmraerr=(sourcedata[pmraerrcol]).astype(float)
                            sourcepmdecerr=(sourcedata[pmdecerrcol]).astype(float)
                        except KeyError:
                            sourcepmra=zero
                            sourcepmdec=zero
                            sourcepmraerr=zero
                            sourcepmdecerr=zero
                    else:
                        try:
                            if (length>0):
                                sourcepmra=np.full(length,pmra0)
                                sourcepmdec=np.full(length,pmdec0)
                            else:
                                sourcepmra=pmra0
                                sourcepmdec=pmdec0
                        except KeyError:
                            sourcepmra=zero
                            sourcepmdec=zero
                        sourcepmraerr=zero
                        sourcepmdecerr=zero
                    sourceepoch=2000.
                    if (verbosity>95):
                        print ("User data parsed.")
                except:
                    if (verbosity>95):
                        print ("Resorting to pre-computed ICRS co-ordinates...")
                    sourcetype="Other"
                    try:
                        sourcera=(sourcedata['_RAJ2000']).astype(float)
                        sourcedec=(sourcedata['_DEJ2000']).astype(float)
                    except:
                        print_fail ("Failure to extract co-ordinates from data")
                        print ("Available columns",sourcedata.dtype.names)
                        raise
                    #try: # if single object, expand array
                    #    foo=len(sourcera)
                    #except:
                    #    sourcera=np.array([sourcera])
                    #    sourcedec=np.array([sourcedec])
                    try:
                        sourceraerr=(sourcedata['e_RAJ2000']).astype(float)/3600000.
                        sourcedecerr=(sourcedata['e_DEJ2000']).astype(float)/3600000.
                    except:
                        try:
                            if (len(sourcera)>0):
                                sourceraerr=np.zeros(len(sourcera),dtype=float)
                                sourcedecerr=np.zeros(len(sourcera),dtype=float)
                        except:
                            sourceraerr=0.
                            sourcedecerr=0.
                    try:
                        sourcepmra=(sourcedata['pmRA']).astype(float)
                        sourcepmdec=(sourcedata['pmDE']).astype(float)
                    except:
                        try:
                            if (len(sourcera)>0):
                                sourcepmra=np.full(len(sourcera),pmra0,dtype=float)
                                sourcepmdec=np.full(len(sourcera),pmdec0,dtype=float)
                        except:
                            sourcepmra=0.
                            sourcepmdec=0.
                    try:
                        sourcepmraerr=(sourcedata['e_pmRA']).astype(float)
                        sourcepmdecerr=(sourcedata['e_pmDE']).astype(float)
                    except:
                        try:
                            if (len(sourcera)>0):
                                sourcepmraerr=np.zeros(len(sourcera),dtype=float)
                                sourcepmdecerr=np.zeros(len(sourcera),dtype=float)
                        except:
                            sourcepmraerr=0.
                            sourcepmdecerr=0.
                    sourceepoch=2000.
                    if (verbosity>95):
                        print ("Other data parsed.")

    sourcepmra=np.nan_to_num(sourcepmra,copy=False,nan=0.0,posinf=0.0,neginf=0.0)
    sourcepmdec=np.nan_to_num(sourcepmdec,copy=False,nan=0.0,posinf=0.0,neginf=0.0)
    sourcepmraerr=np.nan_to_num(sourcepmraerr,copy=False,nan=0.0,posinf=0.0,neginf=0.0)
    sourcepmdecerr=np.nan_to_num(sourcepmdecerr,copy=False,nan=0.0,posinf=0.0,neginf=0.0)
    
    return sourcetype,sourcera,sourcedec,sourceraerr,sourcedecerr,sourcepmra,sourcepmdec,sourcepmraerr,sourcepmdecerr,sourceepoch

# -----------------------------------------------------------------------------
def query_gaia_coords(ra,dec):
    # Query the Gaia archive for a single object
    # Format the co-ordinates
    coord = SkyCoord(ra, dec, unit=(u.degree, u.degree), frame='icrs')
    # Select only one object
    Gaia.ROW_LIMIT = 1
    # Get the default search radius from the setup data
    coneradius=float(pyssedsetupdata[pyssedsetupdata[:,0]=="GaiaCone",1][0])
    width = u.Quantity(coneradius, u.arcsec)
    height = u.Quantity(coneradius, u.arcsec)
    # Get the result
    maxattempts=int(pyssedsetupdata[pyssedsetupdata[:,0]=="ServerRetries",1][0])
    attempts=0
    while attempts < maxattempts:
        try:
            result=Gaia.query_object_async(coordinate=coord, width=width, height=height, verbose=False)
            if (verbosity > 95):
                print ("Gaia.query_object_async returned:")
                print (result)
            # Trap null result
            try:
                dr3_obj=result['SOURCE_ID'][0]
                if (verbosity > 95):
                    print ("DR3_obj:")
                    print (dr3_obj)
                break
            except:
                dr3_obj=0
                break
        except:
            attempts += 1
            if (verbosity >= 25):
                print_warn ("Could not connect to Gaia server (attempt "+str(attempts)+" of "+str(maxattempts)+") [Simbad.query_object]")
            try: # wait for server to clear
                time.sleep(attempts**2)
            except: # if time not installed don't wait
                pass
    if (attempts==maxattempts):
        print_fail ("Failure querying Gaia server, continuing anyway")
        dr3_obj=0

    return dr3_obj

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_vizier_single(cmdparams,sourcedata):

    errmsg=""
    if (verbosity>90):
        print ("Initialising data queries...")
    if (speedtest):
        print ("get_vizier_single:",datetime.now()-globaltime,"s")
    # Get from files for catalogues and filters
    catdata=get_catalogue_list()
    filtdata=get_filter_list()
    rejectdata=get_reject_list()
    # Get data from SVO for filters too
    svodata=get_svo_data(filtdata)
    # Set up ancillary data array
    # (If number of queries == 1, expand dimensions to avoid error)
    ancillary_queries=get_ancillary_list()
    if (np.size(ancillary_queries)==1):
            ancillary_queries=np.expand_dims(ancillary_queries,axis=0)
    ancillary=np.zeros([np.size(ancillary_queries)+5],dtype=[('parameter',object),('catname',object),('colname',object),('value',object),('err',object),('priority',object),('mask',bool)])

    try:
    #if (sourcedata.dtype=="float64"):
        if (verbosity>60):
            print ("Trying to get VizieR data from a set of co-ordinates...")
        sourcetype="User"
        sourcera=sourcedata[0]*15.+sourcedata[1]/4.+sourcedata[2]/240.
        sourcedec=np.sign(sourcedata[3])*(np.abs(sourcedata[3])+sourcedata[4]/60.+sourcedata[5]/3600.)
        sourceraerr=0.
        sourcedecerr=0.
        sourcepmra=0.
        sourcepmdec=0.
        sourcepmraerr=0.
        sourcepmdecerr=0.
        sourceepoch=2000.
    except:
    #else:
        if (verbosity>60):
            print ("Trying to get VizieR data from a list of objects...")
        if (verbosity>70):
            print (sourcedata[0])
        # Get the parameters from the source data
        sourcetype,ra,dec,raerr,decerr,pmra,pmdec,pmraerr,pmdecerr,sourceepoch=extract_ra_dec_pm(sourcedata[0])
        # Now extract the first source
        sourcera=reducto(reducto(ra))
        sourcedec=reducto(reducto(dec))
        sourceraerr=reducto(reducto(raerr))
        sourcedecerr=reducto(reducto(decerr))
        sourcepmra=reducto(reducto(pmra))
        sourcepmdec=reducto(reducto(pmdec))
        sourcepmraerr=reducto(reducto(pmraerr))
        sourcepmdecerr=reducto(reducto(pmdecerr))

    ancillary[0]=('RA',sourcetype,'RA',sourcera,sourceraerr,0,True)
    ancillary[1]=('Dec',sourcetype,'Dec',sourcedec,sourcedecerr,0,True)
    ancillary[2]=('PMRA',sourcetype,'PMRA',sourcepmra,sourcepmraerr,0,True)
    ancillary[3]=('PMDec',sourcetype,'PMDec',sourcepmdec,sourcepmdecerr,0,True)
    if (verbosity>70):
        print ("SED search astrometry:")
        print ("Primary RA:",sourcera,"+/-",sourceraerr)
        print ("Primary Dec:",sourcedec,"+/-",sourcedecerr)
        print ("Primary PMRA:",sourcepmra,"+/-",sourcepmraerr)
        print ("Primary PMDec:",sourcepmdec,"+/-",sourcepmdecerr)
        print ("Primary epoch:",sourceepoch)
    
    #print (sourcedata.dtype.names)

    # Set the default photometric error and get the PM astrometry error controls
    defaulterr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultError",1][0])
    addpmerr=int(pyssedsetupdata[pyssedsetupdata[:,0]=="AddPMErr",1][0])
    pmerrtime=float(pyssedsetupdata[pyssedsetupdata[:,0]=="EpochPMErrYears",1][0])
    warnmissing=int(pyssedsetupdata[pyssedsetupdata[:,0]=="WarnMissingCol",1][0])

    # Loop over catalogues
    catalogues=catdata['catname']
    sed=np.zeros((len(filtdata)),dtype=[('catname','<U20'),('objid','<U32'),('ra','f4'),('dec','f4'),('modelra','f4'),('modeldec','f4'),('svoname','U32'),('filter','U10'),('wavel','f4'),('dw','f4'),('mag','f4'),('magerr','f4'),('flux','f4'),('ferr','f4'),('dered','f4'),('derederr','f4'),('model','f4'),('mask','bool')])
    nfsuccess=0
    if (speedtest):
        print ("get_vizier_single, start of catalogue loop:",datetime.now()-globaltime,"s")
    for catalogue in catalogues:
        server=catdata[catdata['catname']==catalogue]['server']
        if (verbosity>60):
            print ("Catalogue:",catalogue,"on server",server)
        if (server=="Gaia"):
            # Assuming this is a Gaia source, extract the data
            if (sourcetype=="Gaia"):
                ra=sourcera
                dec=sourcedec
                # For the Gaia catalogue itself, extract the BP, RP and G-band data directly
                magkeys=filtdata[filtdata['catname']==catalogue]['filtname']
                errkeys=filtdata[filtdata['catname']==catalogue]['errname']
                newra=sourcera
                newdec=sourcedec
                svokeys=filtdata[filtdata['catname']==catalogue]['svoname']
                for i in (np.arange(len(magkeys))):
                    svokey=svokeys[i]
                    magkey=str(magkeys[i])
                    wavel=float(svodata[svodata['svoname']==svokey]['weff'])
                    dw=float(svodata[svodata['svoname']==svokey]['dw'])
                    zpt=float(svodata[svodata['svoname']==svokey]['zpt'])
                    if (magkey=="BP"):
                        mag=float(sourcedata['phot_bp_mean_mag'])
                        flux=10**(mag/-2.5)*zpt
                        ferr=flux/float(sourcedata['phot_bp_mean_flux_over_error'])
                        err=-2.5*np.log10(1-1/float(sourcedata['phot_bp_mean_flux_over_error']))
                    elif (magkey=="RP"):
                        mag=float(sourcedata['phot_rp_mean_mag'])
                        flux=10**(mag/-2.5)*zpt
                        ferr=flux/float(sourcedata['phot_rp_mean_flux_over_error'])
                        err=-2.5*np.log10(1-1/float(sourcedata['phot_rp_mean_flux_over_error']))
                    else:
                        mag=float(sourcedata['phot_g_mean_mag'])
                        flux=10**(mag/-2.5)*zpt
                        ferr=flux/float(sourcedata['phot_g_mean_flux_over_error'])
                        err=-2.5*np.log10(1-1/float(sourcedata['phot_g_mean_flux_over_error']))
                    sed[nfsuccess]=(catalogue,"None",(newra-sourcera)*3600.,(newdec-sourcedec)*3600.,(ra-sourcera)*3600.,(dec-sourcedec)*3600.,svokey,magkey,wavel,dw,mag,err,flux,ferr,0,0,0,1)
                    nfsuccess+=1
        else:
            ra,dec=pm_calc(sourcera,sourcedec,sourcepmra,sourcepmdec,sourceepoch,float(catdata[catdata['catname']==catalogue]['epoch'][0]))
            if (server=="Vizier"):
                # Correct proper motion and query VizieR
                matchr=float(catdata[catdata['catname']==catalogue]['matchr'][0])
                if (addpmerr > 0):
                    matchr+=np.sqrt(sourcepmraerr**2+sourcepmdecerr**2)/1000.*np.abs(sourceepoch-float(catdata[catdata['catname']==catalogue]['epoch'][0]))
                    matchr+=np.sqrt(sourcepmra**2+sourcepmdec**2)/1000.*pmerrtime
                if (speedtest):
                    querystart=datetime.now()
                vizier_data=query_vizier(cat=str(catdata[catdata['catname']==catalogue]['cdstable'])[2:-2],ra=ra,dec=dec,r=matchr,method="cone")
                if (speedtest):
                    print ("[ get_vizier_single -",catalogue," query:",datetime.now()-querystart,"]")

                if (verbosity>70):
                    print (vizier_data)

                # Only proceed if VizieR has returned some data
                if (vizier_data is not None and len(vizier_data)>0):
                    if (verbosity>80):
                        print (vizier_data[0])
                    # Get RA and Dec from various columns in order of preference
                    newra,newdec=extract_ra_dec(vizier_data[0])
                    newra=reducto(newra) # Only take the first entry
                    newdec=reducto(newdec)
                    if (verbosity>98):
                        print ("Source astrometry:")
                        print ("Source epoch:",float(catdata[catdata['catname']==catalogue]['epoch']))
                        print ("Source RA:",ra,"->",newra,"(",(ra-newra)*3600.,"arcsec)")
                        print ("Source Dec:",dec,"->",newdec,"(",(dec-newdec)*3600.,"arcsec)")
                        print ("Search radius:",matchr,"arcsec")
            else: # Data from file
                photfile=str(catdata[catdata['catname']==catalogue]['cdstable'])[2:-2]
                matchr=float(catdata[catdata['catname']==catalogue]['matchr'][0])
                if (addpmerr > 0):
                    matchr+=np.sqrt(sourcepmraerr**2+sourcepmdecerr**2)/1000.*np.abs(sourceepoch-float(catdata[catdata['catname']==catalogue]['epoch'][0]))
                    matchr+=np.sqrt(sourcepmra**2+sourcepmdec**2)/1000.*pmerrtime
                phot=np.genfromtxt(photfile, delimiter='\t', names=True)
                phot=phot[phot['RA']>=0]
                c=SkyCoord(ra=ra*u.degree,dec=dec*u.degree)
                catcoords=SkyCoord(ra=phot['RA']*u.degree,dec=phot['Dec']*u.degree)
                idx,d2d,d3d=c.match_to_catalog_sky(catcoords)
                if (d2d.arcsec<matchr):
                    vizier_data=np.expand_dims(np.expand_dims(phot[idx],axis=0),axis=1)
                    newra=phot[idx]['RA']
                    newdec=phot[idx]['Dec']
                else:
                    vizier_data=[]
                if (verbosity>60):
                    print ("CATALOGUE = ",catalogue,"; RA,DEC =",ra,dec)
                    print ("Vizier data:",vizier_data)
            if (len(vizier_data)>0): # If any data exists
                svokeys=filtdata[filtdata['catname']==catalogue]['svoname']
                # Get identifier in catalgoue
                idcol=catdata[catdata['catname']==catalogue]['idcol']
                if (idcol=="None"):
                    catid="None"
                else:
                    try:
                        catid=vizier_data[0][idcol[0]][0]
                    except:
                        catid="NotRecognised"
                if (verbosity>90):
                    print ("Object:",catid,"(from column",idcol,")")
                # Identify magnitude and error columns
                magkeys=filtdata[filtdata['catname']==catalogue]['filtname']
                #testdata=vizier_data[0]
                # And extract them from vizier_data
                for i in (np.arange(len(magkeys))):
                    fdata=filtdata[(filtdata['catname']==catalogue) & (filtdata['filtname']==magkeys[i])][0]
                    svokey=fdata['svoname']
                    wavel=float(svodata[svodata['svoname']==svokey]['weff'][0])
                    dw=float(svodata[svodata['svoname']==svokey]['dw'][0])
                    reject_reasons=rejectdata[(rejectdata['catname']==catalogue) & ((rejectdata['filtname']==fdata['filtname']) | (rejectdata['filtname']=="All")) & (rejectdata['rejcat']=="Same") & (rejectdata['rejcol']=="Same")]
                    if (fdata['dataref']=='Vega'):
                        zpt=float(svodata[svodata['svoname']==svokey]['zpt'][0])
                    elif (fdata['dataref']=='AB'):
                        zpt=3631.
                    # Flux should be handled within get_mag_flux
                    try:
                        # Extract data
                        mag,magerr,flux,ferr,mask=get_mag_flux(vizier_data[0],fdata,zpt,reject_reasons)
                        # If more than one datapoint, only use the first one
                        mag=reducto(mag)
                        magerr=reducto(magerr)
                        flux=reducto(flux)
                        ferr=reducto(ferr)
                        mask=reducto(mask)
                        # Force limits
                        if ((fdata['dataref']=='Vega') | (fdata['dataref']=='AB')):
                            if ((mag<fdata['mindata']) | (mag>fdata['maxdata'])):
                                flux=0
                    except:
                        if (warnmissing):
                            print_warn ("Failure to extract data for "+catalogue+" "+fdata['filtname'])
                        flux=0
                    # If the fractional error in the flux is sufficiently small
                    if (flux>0):
                        if (ferr/flux<=fdata['maxperr']/100.*1.000001): # Allow rounding errors
                            sed[nfsuccess]=(catalogue,catid,(newra-sourcera)*3600.,(newdec-sourcedec)*3600.,(ra-sourcera)*3600.,(dec-sourcedec)*3600.,svokey,fdata['filtname'],wavel,dw,mag,magerr,flux,ferr,0,0,0,mask)
                            nfsuccess+=1
    if (speedtest):
        print ("get_vizier_single, end of catalogue loop:",datetime.now()-globaltime,"s")

    # If there is no error in flux, use default error
    sed[sed['ferr']==0]['ferr']==sed[sed['ferr']==0]['flux']*defaulterr

    # Get ancillary information
    if (verbosity>50):
        print ("Getting ancillary data...")
    if (speedtest):
        print ("get_vizier_single, start of ancillary data loop:",datetime.now()-globaltime,"s")
    for i in np.arange(np.size(ancillary_queries)):
        if (verbosity>60):
            print (ancillary_queries[i])
        if ((i>0) & (ancillary_queries[i-1]['cdstable']==ancillary_queries[i]['cdstable'])):
            uselastdata=True # saves querying VizieR if using the same table
        else:
            uselastdata=False
        if (uselastdata==False):
            ra,dec=pm_calc(sourcera,sourcedec,sourcepmra,sourcepmdec,sourceepoch,float(ancillary_queries[i]['epoch']))
            # Correct proper motion and query VizieR
            matchr=float(ancillary_queries[i]['matchr'])
            if (addpmerr > 0):
                matchr+=np.sqrt(sourcepmraerr**2+sourcepmdecerr**2)/1000.*np.abs(sourceepoch-float(catdata[catdata['catname']==catalogue]['epoch'][0]))
            matchr+=np.sqrt(sourcepmra**2+sourcepmdec**2)/1000.*pmerrtime
        datalen=0
        if (ancillary_queries[i]['server']=="Vizier"):
            if (uselastdata==False):
                vizier_data=query_vizier(cat=str(ancillary_queries[i]['cdstable']),ra=ra,dec=dec,r=matchr,method="cone")

            if (verbosity>70):
                print ("CATALOGUE = ",str(ancillary_queries[i]['cdstable']),"; RA,DEC =",ra,dec)
                print (vizier_data)

            # Only proceed if VizieR has returned some data
            try:
                datalen=len(vizier_data)
            except AttributeError:
                if (verbosity>80):
                    print_warn ("Ignoring catalogue with no or bad data")
                datalen=0
            except TypeError:
                if (verbosity>70):
                    print_warn ("Bad data returned by VizieR. Retrying.")
                try:
                    vizier_data=query_vizier(cat=str(ancillary_queries[i]['cdstable']),ra=ra,dec=dec,r=matchr,method="cone")
                except TypeError:
                    print_fail ("Repeated bad data from VizieR. Aborting this entry.")
                    print ("CATALOGUE = ",str(ancillary_queries[i]['cdstable']),"; RA,DEC =",ra,dec)
                datalen=0
            if (datalen>0):
                if (verbosity>80):
                    print (vizier_data[0])
                if (verbosity>98):
                    print (vizier_data[0].keys())
                # Get RA and Dec from various columns in order of preference
                newra,newdec=extract_ra_dec(vizier_data[0])
                newra=reducto(newra) # Only take the first entry
                newdec=reducto(newdec)
        elif (ancillary_queries[i]['server']=="File"):
            datafile=ancillary_queries[i]['cdstable']
            anc=np.genfromtxt(datafile, delimiter='\t', names=True)
            anc=anc[anc['RA']>=0]
            ancdata=[]
            c=SkyCoord(ra=ra*u.degree,dec=dec*u.degree)
            catcoords=SkyCoord(ra=anc['RA']*u.degree,dec=anc['Dec']*u.degree)
            d2d=c.separation(catcoords)
            #ancdata=np.expand_dims(anc[d2d.arcsec<matchr],axis=0)
            ancdata=anc[d2d.arcsec<matchr]
            datalen=len(ancdata)
            if (datalen>0):
                newra=reducto(ancdata['RA'])
                newdec=reducto(ancdata['Dec'])
                vizier_data=np.expand_dims(ancdata,axis=0)
        else:
            try:
                np.loadtxt("This source of data is") # ...not found
            except:
                print_fail ("ERROR! Source of data unrecognised!")
                print ("Received:",ancillary_queries[i]['server'],"; should be Vizier or File")
                raise
        if (datalen>0):
            reasons=rejectdata[(rejectdata['catname']==ancillary_queries[i]['catname']) & ((rejectdata['filtname']==ancillary_queries[i]['colname']) | (rejectdata['filtname']=="All")) & (rejectdata['rejcat']=="Same") & (rejectdata['rejcol']=="Same")]
            vizier_data,mask=reject_test(reasons,vizier_data)
            try:
                if (ancillary_queries[i]['errname']=="None"):
                    err=0
                elif (isfloat(ancillary_queries[i]['errname'])):
                    err=float(ancillary_queries[i]['errname'])
                elif ("/" in ancillary_queries[i]['errname']): # allow confidence limits
                    errs=str.split(ancillary_queries[i]['errname'],sep="/")
                    err=(vizier_data[0][errs[0]][0]-vizier_data[0][errs[1]][0])/2.
                elif (":" in ancillary_queries[i]['errname']): # allow upper/lower errors
                    errs=str.split(ancillary_queries[i]['errname'],sep=":")
                    err=(vizier_data[0][errs[0]][0]+vizier_data[0][errs[1]][0])/2.
                else:
                    err=vizier_data[0][ancillary_queries[i]['errname']][0]
                if (vizier_data[0][ancillary_queries[i]['colname']][0]!="--"):
                    ancillary[i+5]=(ancillary_queries[i]['paramname'],ancillary_queries[i]['catname'],ancillary_queries[i]['colname'],vizier_data[0][ancillary_queries[i]['colname']][0],err,ancillary_queries[i]['priority'],mask)
            except KeyError:
                print_fail ("Key error: entry in ancillary request file does not match VizieR table")
                print ("Query:",ancillary_queries[i])
                print ("Available keys:",vizier_data[0].dtype.names)
                print_fail ("Field in Query must match field in Available. Edit ancillary queries.")
                raise
    if (speedtest):
        print ("get_vizier_single, end of ancillary data loop:",datetime.now()-globaltime,"s")

    # Apply additional rejection criteria
    # -----------------------------------

    # Reject if wavelength too short or long
    minlambda=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MinLambda",1][0])*10000.
    maxlambda=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxLambda",1][0])*10000.
    sed[:]['mask']=np.where(sed[:]['wavel']<minlambda,False,sed[:]['mask'])
    sed[:]['mask']=np.where(sed[:]['wavel']>maxlambda,False,sed[:]['mask'])
    #svodata[svodata['svoname']==svokey]['weff'][0]

    # Apply rejection criteria spanning multiple catalogues
    if (verbosity>60):
        print ("Rejecting data based on flags...")
    try:
        reasons=rejectdata[(rejectdata['rejcat']!="Same") & (rejectdata['position']<0)]
    except:
        reasons=[]
    if (verbosity>80):
        print (len(reasons),"reasons identified")

    if (speedtest):
        print ("get_vizier_single, start of rejection criteria loop:",datetime.now()-globaltime,"s")
    for i in np.arange(len(reasons)):

        if (reasons[i]['column']!="anc"):
            # --- Apply to photometry ---
            # Identify data to test
            testdata=sed[(sed['catname']==reasons[i]['rejcat']) & (sed['filter']==reasons[i]['rejcol'])][reasons[i]['column']]
            n=(sed['catname']==reasons[i]['catname']) & (sed['filter']==reasons[i]['filtname'])
            #if (verbosity>90):
            #    print (reasons[i],len(testdata),len(n))
            compmag=sed[n]['mag']
            compflux=sed[n]['flux']
            logic=reasons[i]['logical']
            logictest=reasons[i]['value']

            if ((len(testdata)>0) & (len(n)>0)):
                action=reject_logic(testdata,compmag,compflux,logic,logictest)
                # If action needs taken then mask or purge
                if (action==1):
                    if (verbosity>80):
                        print ("Triggered:",reasons[i])
                    if (reasons[i]['filtname']=="All"): # All filters, or only one?
                        n=(sed['catname']==reasons[i]['catname'])
                    else:
                        n=(sed['catname']==reasons[i]['catname']) & (sed['filter']==reasons[i]['filtname'])
                    if (reasons[i]['result']=="Mask"): # Mask or reject?
                        sed[:]['mask']=np.where(n==True,False,sed[:]['mask'])
                    else:
                        sed[:]['flux']=np.where(n==True,0,sed[:]['flux'])
                else:
                    if (verbosity>80):
                        print ("Not triggered:",reasons[i])
        else:
            # --- Apply to ancillary data ---
            # Identify data to test
            testdata=ancillary[(ancillary['catname']==reasons[i]['rejcat']) & (ancillary['parameter']==reasons[i]['rejcol'])]['value']
            n=(ancillary['catname']==reasons[i]['catname']) & (ancillary['colname']==reasons[i]['filtname'])
            #if (verbosity>90):
            #    print (reasons[i],len(testdata),len(n))
            compmag=ancillary[n]['value']
            compflux=ancillary[n]['value']
            logic=reasons[i]['logical']
            logictest=reasons[i]['value']

            if ((len(testdata)>0) & (len(n)>0)):
                action=reject_logic(testdata,compmag,compflux,logic,logictest)
                # If action needs taken then mask or purge
                if (action==1):
                    if (verbosity>80):
                        print ("Triggered:",reasons[i])
                    if (reasons[i]['filtname']=="All"): # All filters, or only one?
                        n=(ancillary['catname']==reasons[i]['catname'])
                    else:
                        n=(ancillary['catname']==reasons[i]['catname']) & (ancillary['colname']==reasons[i]['filtname'])
                    if (reasons[i]['result']=="Mask"): # Mask or reject?
                        ancillary[:]['mask']=np.where(n==True,False,ancillary[:]['mask'])
                    else:
                        ancillary[n]['parameter']=np.where(n==True,0,ancillary[:]['parameter'])
                else:
                    if (verbosity>80):
                        print ("Not triggered:",reasons[i])
    if (speedtest):
        print ("get_vizier_single, end of rejection criteria loop and done:",datetime.now()-globaltime,"s")
    
    return sed[sed['flux']>0.],ancillary[ancillary['parameter']!=0],errmsg

# -----------------------------------------------------------------------------
def get_mag_flux(testdata,fdata,zpt,reasons):

    # Set up defaults
    mag=[0]
    magerr=[0]
    flux=[0]
    ferr=[0]
    #defaulterr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultError",1][0])
    defaulterr=fdata['maxperr']/100.

    # Extract keys from filterdata
    magkey=fdata['filtname']
    mindata=fdata['mindata']
    maxdata=fdata['maxdata']

    # Extract data (magnitudes or fluxes)
    try:
        mag=testdata[magkey]
    except KeyError:
        print_fail ("Key error: entry in filter metadata file does not match VizieR table")
        print ("Query:",magkey)
        print ("Test data:",testdata.dtype.names)
        print_fail ("Field in Query must match field in Test data. Edit filter information file.")
        raise

    # Then check it's good by rejecting anything nasty
    vizier_data,mask=reject_test(reasons,testdata)

    # If mag is finite and within the limits specified
    mask=np.where(np.isfinite(mag),mask,False)
    mask=np.where((mag >= mindata) & (mag <= maxdata),mask,False)
    mag=np.where(np.isfinite(mag),mag,0)
    errkey=fdata['errname']
    if (errkey=="None"):
        err=mag*0.+defaulterr # Gets err same size as mag array
    else:
        err=testdata[errkey]
    svokey=fdata['svoname']
    datatype=fdata['datatype']
    errtype=fdata['errtype']

    # Detect magnitudes or fluxes
    if ((datatype=='mag') or (datatype=='nMgy')):
        if (mag.size==1):
            mag=reducto(mag)
            magerr=reducto(err)
            if (np.isscalar(magerr)==False or magerr>1.):
                magerr=defaulterr
        if (datatype=='nMgy'):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
                warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
                ferr=magerr/mag
                magerr=-2.5*np.log10(1.-ferr)
                mag=22.5-2.5*np.log10(mag)
        flux=10**(mag/-2.5)*zpt
        if (errtype=='Same'):
            if (datatype!='nMgy'):
#                try:
                    if (reducto(err)>0):
                        ferr=flux-10**((mag+err)/-2.5)*zpt
                    else:
                        ferr=flux*defaulterr
#                        #print ("TRIGGER:",flux,ferr,mag,err,zpt)
#                except:
#                    print_fail ("Error in flux error assignment")
#                    print ("mag:",mag)
#                    print ("flux:",flux)
#                    print ("zpt:",zpt)
#                    raise
        else: # Perc
            ferr=flux*err/100.
        # Remove fluxes if magnitude is EXACTLY zero
        flux=np.where(mag!=0,flux,0)
        ferr=np.where(mag!=0,ferr,0)
    else: # Assume data type is a Jy-based unit
        # What is read as mag is actually flux, so swap them
        flux=mag
        if (errtype=='Same'):
            ferr=err
        else: # Perc
            ferr=flux*err/100.
        if (datatype=='mJy'):
            flux/=1.e3
            ferr/=1.e3
        elif (datatype=='uJy'):
            flux/=1.e6
            ferr/=1.e6
        elif (datatype=='nJy'):
            flux/=1.e9
            ferr/=1.e9
        # Reduce dimensionality, check for nulls
        try:
            if (len(mag)==1):
                flux=reducto(flux)
                ferr=reducto(ferr)
                if (np.isscalar(flux)==False):
                    flux=0.
                if (np.isscalar(ferr)==False):
                    ferr=0.
                if (flux<=0):
                    flux=0
                    ferr=0
                    mag=np.inf
                    magerr=np.inf
                else:
                    # Perform conversion
                    mag=-2.5*np.log10(flux/zpt)
                    magerr=np.where(ferr>0,2.5*np.log10(1+ferr/flux),0.)
            else:
                # Perform conversion
                mag=-2.5*np.log10(flux/zpt)
                magerr=np.where(ferr>0,2.5*np.log10(1+ferr/flux),0.)
        except TypeError:
            mag=-2.5*np.log10(flux/zpt)
            magerr=np.where(ferr>0,2.5*np.log10(1+ferr/flux),0.)
            
    if (verbosity>98):
        try:
            print ("Flux[0]:",flux[0],ferr[0])
            print ("Mag[0]:",mag[0],magerr[0])
        except:
            print ("Flux:",flux,ferr)
            print ("Mag:",mag,magerr)

    return mag,magerr,flux,ferr,mask

# -----------------------------------------------------------------------------
def reject_test(reasons,inputdata):

    mask=1
    for i in np.arange(len(reasons)):
        # Identify data to test
        if (reasons[i]['position']>=0):
            try:
                testdata=inputdata[reasons[i]['column']][0][reasons[i]['position']]
            except:
                try:
                    testdata=inputdata[reasons[i]['column']][reasons[i]['position']]
                except:
                    print_fail("Fail on rejection testing")
                    print_fail ("Check the following entry in the rejects file:")
                    print (inputdata)
                    print (reasons[i])
                    print (reasons[i]['column'])
                    print (inputdata[reasons[i]['column']])
                    print (inputdata[reasons[i]['column']][0])
        else:
            try:
                testdata=inputdata[reasons[i]['column']][0]
            except:
                try:
                    testdata=inputdata[reasons[i]['column']]
                except:
                    print_fail ("Fail on rejection testing")
                    print_fail ("Check the following entry in the rejects file:")
                    print (inputdata)
                    print (reasons[i])
                    raise
        # Identify whether action needs taken
        action=0
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan.")
            if (reasons[i]['logical']=="="):
                if (str(testdata)==reasons[i]['value']):
                    action=1
            elif (reasons[i]['logical']=="<"):
                if (float(testdata)<float(reasons[i]['value'])):
                    action=1
            elif (reasons[i]['logical']==">"):
                if (float(testdata)>float(reasons[i]['value'])):
                    action=1
            elif (reasons[i]['logical']=="!"):
                if (str(testdata)!=str(reasons[i]['value'])):
                    action=1
            elif (reasons[i]['logical']=="~"):
                if (str(testdata) in str(reasons[i]['value'])):
                    action=1
        # If action needs taken then mask or purge
        if (action==1):
            if (reasons[i]['result']=="Mask"):
                mask=0
                if (verbosity>95):
                    print ("Masked",reasons[i])
            else:
                try:
                    inputdata[reasons[i]['filtname']][0]=-9.e99 # Set to stupidly low flux/mag
                except:
                    inputdata[reasons[i]['filtname']]=-9.e99 # Set to stupidly low flux/mag
        else:
                if (verbosity>95):
                    print ("Not masked",reasons[i])

    return inputdata,mask
 
# -----------------------------------------------------------------------------
def reject_logic(testdata,compmag,compflux,logic,logictest):

    action=0
    if ((len(testdata)>0) & ((len(compflux)>0) | (len(compmag)>0)) & (len(logic)>0) & (len(logictest)>0)):
        # Identify data for comparison if logic has a prefix
        if (logic[0]=="d"): # differential mag
            testdata=testdata-compmag
            logic=logic[1:]
        elif (logic[0]=="D"): # abs diff mag
            testdata=abs(testdata-compmag)
            logic=logic[1:]
        elif (logic[0]=="f"): # flux ratio
            testdata=testdata/compflux
            logic=logic[1:]
        elif (logic[0]=="F"): # abs flux ratio
            if (testdata>compflux):
                testdata=testdata/compflux
            else:
                testdata=compflux/testdata
            logic=logic[1:]
        # Identify whether action needs taken
        testdata=reducto(testdata)
        compmag=reducto(compmag)
        compflux=reducto(compflux)
        if (logic=="="):
            if (str(testdata)==logictest):
                action=1
        elif (logic=="<"):
            if (float(testdata)<float(logictest)):
                action=1
        elif (logic==">"):
            if (float(testdata)>float(logictest)):
                action=1
        elif (logic=="!"):
            if (str(testdata)!=str(logictest)):
                action=1
        elif (logic=="~"):
            if (str(testdata) in str(logictest)):
                action=1

    return action

# -----------------------------------------------------------------------------
def query_vizier(cat,ra,dec,ra2=0,dec2=0,w=0,h=0,r=0,method=""):
    if (verbosity>80):
        print ("Vizier",method,"query for",cat)
    # Query the TAP server at CDS
    maxrows=int(pyssedsetupdata[pyssedsetupdata[:,0]=="VizierRowLimit",1][0])
    maxattempts=int(pyssedsetupdata[pyssedsetupdata[:,0]=="ServerRetries",1][0])
    attempts=0
    while attempts < maxattempts:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",message="Unit 'Sun' not supported by the VOUnit standard.")
                if (method=="cone"):
                    if (verbosity>80):
                        print ("Cone search co-ords", ra, dec, "r=", r)
                    result = Vizier(columns=["**","_RAJ2000","_DEJ2000","+_r"],row_limit=maxrows).query_region(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'), radius=r*u.arcsec, catalog=cat, cache=False)
                elif (method=="rectangle"):
                    if (verbosity>80):
                        print ("Rectangle search co-ords", ra, dec, "w,h=", w, h)
                    result = Vizier(columns=["**","_RAJ2000","_DEJ2000","+_r"],row_limit=maxrows).query_region(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'), width=w*u.deg, height=h*u.deg, catalog=cat, cache=False)
                elif (method=="box"):
                    if (verbosity>80):
                        print ("Box search co-ords", ra, dec, "to", ra2, dec2)
                    result = Vizier(columns=["**","_RAJ2000","_DEJ2000","+_r"],row_limit=maxrows).query_region(SkyCoord(ra=(ra+ra2)/2., dec=(dec+dec2)/2., unit=(u.deg, u.deg), frame='icrs'), width=(ra2-ra)*np.cos(np.deg2rad(dec))*u.deg, height=(dec2-dec)*u.deg, catalog=cat, cache=False)
                break
        except Exception as e:
            if (verbosity>50):
                print ("Error:",e)
            attempts += 1
            print_warn ("Could not connect to VizieR server (attempt "+str(attempts)+" of "+str(maxattempts)+") [Vizier,"+str(method)+"query]")
            try: # wait for server to clear
                time.sleep(attempts**2)
            except: # if time not installed don't wait
                pass
    if (attempts==maxattempts):
        print_fail ("Could not connect to VizieR server")
        raise Exception("Could not connect to VizieR server")
    
    return result

# -----------------------------------------------------------------------------
def plx_correction():
    # Correct the parallax based on the Gaia data properties
    return

# -----------------------------------------------------------------------------
def gmag_correction():
    # Correct the G-band magnitude based on the Gaia data properties
    return

# -----------------------------------------------------------------------------
def pm_calc(ra,dec,pmra,pmdec,epoch,newepoch):
    # Correct position for proper motion to a specific epoch
    # Units: ra, dec in degrees
    #        pmra, pmdec in mas/yr
    #        epochs in years
    dt=newepoch-epoch
    newra=ra+np.cos(np.deg2rad(dec.astype(np.float64)))*pmra*dt/3600.e3
    newdec=dec+pmdec*dt/3600.e3
    
    return newra,newdec

# -----------------------------------------------------------------------------
def get_dist():
    # Get the distance between two points on the sky
    return

# -----------------------------------------------------------------------------
def col_correct():
    # Perform colour correction on photometry
    return

# -----------------------------------------------------------------------------
def estimate_flux():
    # Estimate the flux at a given wavelength based on data at others
    return

# =============================================================================
# OBTAINING SEDS
# =============================================================================
def get_sed_single(cmdparams):
    # Generate an SED
    if (speedtest):
        print ("get_sed_single:",datetime.now()-globaltime,"s")
    dr3_obj=0
    dr3_data=[]
    dr3_obj,errmsg=get_gaia_obj(cmdparams)
    maxattempts=int(pyssedsetupdata[pyssedsetupdata[:,0]=="ServerRetries",1][0])

    # If a Gaia DR3 cross-match can be found...
    if (dr3_obj!=0):
        # Query the Gaia DR3 database for data
        dr3_data=get_gaia_data(dr3_obj)
        if (len(dr3_data)>0):
            # Parse setup data: am I saving this file?
            savegaia=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SavePhot",1][0])
            photfile=str(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotFile",1])[2:-2]
            if (savegaia>0):
                if (verbosity>=60):
                    print ("Saving data to",photfile)
                np.save(photfile,dr3_data)
        # Get data on other filters
    #    if (dr3_obj!=0): #(len(dr3_data)>0):
            if (verbosity>50):
                print ("Getting SED based on Gaia cross-reference")
            sed,ancillary,errmsg=get_vizier_single(cmdparams,dr3_data)
        elif (verbosity>50):
            print ("No data returned from Gaia server for object",dr3_obj)
        if (speedtest):
            print ("Gaia entry processed:",datetime.now()-globaltime,"s")

    if ((dr3_obj == 0) or (dr3_obj == "") or (len(dr3_data)==0)):
        try:
            if (verbosity>=40):
                print ("Could not find a Gaia DR3 cross-match.")
            # Get Hipparcos identifier from SIMBAD
            result_table = Simbad.query_objectids(cmdparams,cache=False)
            hip="00000"
            for i in np.arange(len(result_table['ID'])):
                if ('HIP' in result_table['ID'][i]):
                    hip=result_table['ID'][i]
            if (int(hip[4:])>0):
                # Query Hipparcos identifier for data
                if (verbosity>40):
                    print ("Hipparcos id:",hip)
                attempts=0
                while attempts < maxattempts:
                    try:
                        vizquery=Vizier(columns=["**"]).query_object(hip, catalog="I/311/hip2")
                        break
                    except:
                        attempts += 1
                        print_warn ("Could not connect to Vizier server (attempt "+attempts+" of "+maxattempts+") [Vizier, Hipparcos]")
                        try: # wait for server to clear
                            time.sleep(attempts**2)
                        except: # if time not installed don't wait
                            pass
                if (attempts==maxattempts):
                    print_fail ("Could not connect to VizieR server")
                    raise Exception("Could not connect to VizieR server")
            if (verbosity>50):
                print ("Getting SED based on Hipparcos cross-reference")
            sed,ancillary,errmsg=get_vizier_single(cmdparams,vizquery)
        except: # If no HIP source and/or data
            # Just use co-ordinates: try parsing co-ordinates first, then ask SIMBAD
            if (verbosity>40):
                print ("No Gaia or Hipparcos ID: using fixed co-ordinates only.")
                exit()
            try:
                with warnings.catch_warnings(): # Numpy 1.25 warning trapped by try/except
                    warnings.filterwarnings("ignore",message="string or file could not be read to its end due to unmatched data")
                    coords=np.fromstring(cmdparams,dtype=float,sep=" ")
                if (coords[0]>=0 and coords[0]<=360 and coords[1]>=-90 and coords[1]<=90):
                    if (verbosity>60):
                        print (coords)
                    result_table = coords
            except:
                result_table = Simbad.query_object(cmdparams)
            try:
                if (len(result_table)>0):
                    if (verbosity>90):
                        print (result_table)
                        print ("Obtaining SED")
                    sed,ancillary,errmsg=get_vizier_single(cmdparams,result_table)
                else:
                    if (verbosity>20):
                        print_warn("No useful result in result_table!")
            except TypeError:
                if (verbosity>90):
                    print_warn("No SED found, creating an empty one")
                sed=np.array([])
                #sed,ancillary,errmsg=get_vizier_single(cmdparams,result_table)                #sed=np.zeros((0),dtype=[('catname','<U20'),('objid','<U32'),('ra','f4'),('dec','f4'),('modelra','f4'),('modeldec','f4'),('svoname','U32'),('filter','U10'),('wavel','f4'),('dw','f4'),('mag','f4'),('magerr','f4'),('flux','f4'),('ferr','f4'),('dered','f4'),('derederr','f4'),('model','f4'),('mask','bool')])

                return sed,np.array([]),errmsg
#            except:
#                print_fail("ERROR in source name!")
#                print ("Either no source with that name exists in SIMBAD or, if co-ordinates were")
#                print ("specified, they were either uninterpretable or there were no records at that")
#                print ("location in the specified catalogues.")
#                sed=[]
#                ancillary=[]
#                errmsg="ERROR in source name for "+cmdparams
            
    return sed,ancillary,errmsg
    
# -----------------------------------------------------------------------------
def merge_sed(sed):
    # Merge the flux at a specific wavelength, based either on mean, weighted mean or preferred catalogue
    
    minerr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MinPhotError",1][0])
    wtlimit=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotWeightingLimit",1][0])
    sigmalimit=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotSigmaLimit",1][0])
    
    for filt in np.unique(sed['svoname']):
        # If multiple observations in the same filter...
        if (len(sed[(sed['svoname']==filt) & (sed['mask']==True)])>1):
            if (verbosity>80):
                print ("Merging photometry in",filt)
                print (sed[(sed['svoname']==filt) & (sed['mask']==True)])
            # Extract fluxes and errors
            flux=sed[(sed['svoname']==filt) & (sed['mask']==True)]['flux']
            ferr=sed[(sed['svoname']==filt) & (sed['mask']==True)]['ferr']
            # Reweight small errors
            ferr=np.where(ferr/flux<minerr,minerr*flux,ferr)
            # Calculate mean/median
            if (len(flux)==2):
                mflux=np.average(flux)
            else:
                mflux=np.median(flux)
            # Can these be merged within the weighting tolerance?
            wt=min(ferr/flux)/(ferr/flux)
            mergewt=np.where(wt>wtlimit,True,False)
            # Can these be merged within the sigma tolerance?
            sigma=abs(flux-mflux)/ferr
            mergesigma=np.where(sigma<sigmalimit,True,False)
            # If both tests are ok, they can be merged.
            if (np.sum(mergewt*mergesigma)>1):
                    mergeable=sed[(sed['svoname']==filt) & (sed['mask']==True)][mergewt*mergesigma>0]
                    wt=wt[mergewt*mergesigma>0]
                    if (verbosity>80):
                        print ("Can be merged")
                    # Flag mask to remove merged items
                    for i in np.arange(len(sed)):
                        if (sed[i] in mergeable):
                            sed[i]['mask']=False
                    merged=np.copy(mergeable[0])
                    wt=wt*wt # Get variance as weight instead of error
                    wt/=np.sum(wt) # Normalise weights
                    zpt=merged['flux']*10**(merged['mag']/2.5) # Get zero point, since not available here
                    for i in sed.dtype.names:
                        try:
                            merged[i]=np.sum(mergeable[i]*wt)
                        except:
                            if (len(np.unique(mergeable[i]))>1):
                                merged[i]="Merged"
                    # Correct magnitude, as merged by flux
                    merged['mag']=-2.5*np.log10(merged['flux']/zpt)
                    merged['magerr']=2.5*np.log10(1+merged['ferr']/merged['flux'])
                    if (verbosity>80):
                        print (merged)
                    # Add to SED
                    sed=np.append(sed,merged)
            else:
                if (verbosity>50):
                    print ("Cannot be merged (sigma limit exceeded)")
        # If there are still multiple observations in one filter, choose the first one
        if (len(sed[(sed['svoname']==filt) & (sed['mask']==True)])>1):
            if (verbosity>80):
                print ("Select first:",filt)
                print (sed[(sed['svoname']==filt) & (sed['mask']==True)])
            removeable=sed[(sed['svoname']==filt) & (sed['mask']==True)]
            removeable=removeable[1:]
            for i in np.arange(len(sed)):
                if (sed[i] in removeable):
                    sed[i]['mask']=False

    return sed

# -----------------------------------------------------------------------------
def merge_ancillary(ancillary):
    # Merge the flux at a specific wavelength, based either on mean, weighted mean or preferred catalogue
    
    minerr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MinAncError",1][0])
    wtlimit=float(pyssedsetupdata[pyssedsetupdata[:,0]=="AncWeightingLimit",1][0])
    sigmalimit=float(pyssedsetupdata[pyssedsetupdata[:,0]=="AncSigmaLimit",1][0])

    # Reject null values
    ancillary['mask']=np.where(ancillary['value']=='',False,ancillary['mask'])
    
    for param in np.unique(ancillary['parameter']):
        # If multiple measurements of the same parameter...
        if (len(ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)])>1):
            if (verbosity>80):
                print ("Merging photometry in",param)
                print (ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)])
            # Apply priority criteria
            maxpriority=np.min(ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)]['priority'])
            removeable=ancillary[(ancillary['parameter']==param) & (ancillary['priority'].astype(int)>maxpriority)]
            if (verbosity>80):
                print ("Removing",param,"with priority >",maxpriority)
                print (removeable)
            for i in np.arange(len(ancillary)):
                if (ancillary[i] in removeable):
                    ancillary[i]['mask']=False
        # If there are still multiple measurements of the same parameter...
        if (len(ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)])>1):
            if (verbosity>80):
                print ("Round 2:",param)
                print (ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)])
            # Will fail if non-numeric
            try:
                # Extract fluxes and errors
                value=ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)]['value']
                err=ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)]['err']
                # Reweight small errors
                err=np.where(err/value<minerr,minerr*value,err)
                # Reweight infinite errors
                err=np.where(err==inf,value,err)
                # Calculate mean/median
                if (len(value)==2):
                    mvalue=np.average(value)
                else:
                    mvalue=np.median(value)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",message="invalid value encountered in double_scalars")
                    warnings.filterwarnings("ignore",message="invalid value encountered in true_divide")
                    warnings.filterwarnings("ignore",message="divide by zero encountered in long_scalars")
                    warnings.filterwarnings("ignore",message="divide by zero encountered in true_divide")
                    warnings.filterwarnings("ignore",message="divide by zero encountered in double_scalars")
                    # Can these be merged within the weighting tolerance?
                    wt=min(err/value)/(err/value)
                    mergewt=np.where(wt>wtlimit,True,False)
                    # Can these be merged within the sigma tolerance?
                    sigma=abs(value-mvalue)/err
                    mergesigma=np.where(sigma<sigmalimit,True,False)
                    print (sigma)
                # If both tests are ok, they can be merged.
                if (np.sum(mergewt*mergesigma)>1):
                    mergeable=ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)][mergewt*mergesigma>0]
                    wt=wt[mergewt*mergesigma>0]
                    if (verbosity>80):
                        print ("Can be merged")
                    # Flag mask to remove merged items
                    for i in np.arange(len(ancillary)):
                        if (ancillary[i] in mergeable):
                            ancillary[i]['mask']=False
                    merged=np.copy(mergeable[0])
                    wt=wt*wt # Get variance as weight instead of error
                    wt/=np.sum(wt) # Normalise weights
                    merged['catname']="Merged"
                    merged['colname']="Merged"
                    merged['value']=np.sum(mergeable['value']*wt)
                    merged['err']=np.sum(mergeable['err']*wt)/np.sqrt(np.sum(mergewt*mergesigma))
                    merged['priority']=0
                    merged['mask']=True
                    if (verbosity>80):
                        print (merged)
                    # Add to SED
                    ancillary=np.append(ancillary,merged)
                else:
                    if (verbosity>50):
                        print ("Cannot be merged (sigma limit exceeded)")
            except:
                # Do nothing for non-numeric
                x=0
        # If there are still multiple observations in one filter, choose the first one
        if (len(ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)])>1):
            if (verbosity>80):
                print ("Round 3:",param)
                print (ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)])
            # Apply priority criteria
            removeable=ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)]
            removeable=removeable[1:]
            for i in np.arange(len(ancillary)):
                if (ancillary[i] in removeable):
                    ancillary[i]['mask']=False

    return ancillary

# -----------------------------------------------------------------------------
def get_area_data(cmdargs,cats,outdir):
    # Get data from VizieR catalogues for area searches
    trimbox=int(pyssedsetupdata[pyssedsetupdata[:,0]=="TrimBox",1][0])
    for catname in np.unique(cats['catname']):
        cat=cats[cats['catname']==catname]['cdstable'][0]
        matchr=cats[cats['catname']==catname]['matchr'][0]
        photprocfile=outdir+catname+".npy"
        server=cats[cats['catname']==catname]['server'][0]
        
        if (server=="Vizier"):
            try: # If data
                if (cmdargs[1]=="cone"):
                    catdata=query_vizier(cat=cat,ra=float(cmdargs[2]),dec=float(cmdargs[3]),r=float(cmdargs[4])*3600.+matchr,method=cmdargs[1])
                elif (cmdargs[1]=="rectangle"):
                    catdata=query_vizier(cat=cat,ra=float(cmdargs[2]),dec=float(cmdargs[3]),w=float(cmdargs[4])+2.*matchr/3600.,h=float(cmdargs[5])+2.*matchr/3600.,method=cmdargs[1])
                elif (cmdargs[1]=="box"):
                    catdata=query_vizier(cat=cat,ra=float(cmdargs[2])-matchr/3600.,dec=float(cmdargs[3])-matchr/3600.,ra2=float(cmdargs[4])+matchr/3600.,dec2=float(cmdargs[5])+matchr/3600.,method=cmdargs[1])
                nobj=len(catdata[0]) # intentional fail here if EmptyTable
                if (verbosity >=50):
                    print ("Queried",catname,"- found",len(catdata[0]),"objects ->",photprocfile)
                np.save(photprocfile,catdata[0])
            except IndexError: # If no data
                if (verbosity >=70):
                    print ("Queried",catname,"- found no objects")
                try: # Remove old file...
                    remove(photprocfile)
                except FileNotFoundError: #...if it exists
                    pass
                pass
        else: # data from file
            photfile=str(cats[cats['catname']==catname]['cdstable'])[2:-2]
            phot=np.genfromtxt(photfile, delimiter='\t', names=True)
            phot=phot[phot['RA']>=0]
            catdata=[]
            if (cmdargs[1]=="cone"):
                c=SkyCoord(ra=float(cmdargs[2])*u.degree,dec=float(cmdargs[3])*u.degree)
                catcoords=SkyCoord(ra=phot['RA']*u.degree,dec=phot['Dec']*u.degree)
                d2d=c.separation(catcoords)
                #catdata=np.expand_dims(np.expand_dims(phot[d2d.arcsec<float(cmdargs[4])*3600.+matchr],axis=0),axis=1)
                catdata=np.expand_dims(phot[d2d.arcsec<float(cmdargs[4])*3600.+matchr],axis=0)
                catdata=phot[d2d.arcsec<float(cmdargs[4])*3600.+matchr]
            elif (cmdargs[1]=="rectangle"):
                if (trimbox>0):
                    catdata=phot[(phot['RA']>=float(cmdargs[2])-(float(cmdargs[4])/2.-matchr/3600.)) & (phot['RA']<=float(cmdargs[2])+(float(cmdargs[4])/2.+matchr/3600.)) & (phot['Dec']>=float(cmdargs[3])-float(cmdargs[5])/2.-matchr/3600.) & (phot['Dec']<=float(cmdargs[3])+float(cmdargs[5])/2.+matchr/3600.)]
                else:
                    catdata=phot[(phot['RA']>=float(cmdargs[2])-(float(cmdargs[4])/2.-matchr/3600.)/np.cos(np.deg2rad(phot['Dec']))) & (phot['RA']<=float(cmdargs[2])+(float(cmdargs[4])/2.+matchr/3600.)/np.cos(np.deg2rad(phot['Dec']))) & (phot['Dec']>=float(cmdargs[3])-float(cmdargs[5])/2.-matchr/3600.) & (phot['Dec']<=float(cmdargs[3])+float(cmdargs[5])/2.+matchr/3600.)]
            elif (cmdargs[1]=="box"):
                if (trimbox>0):
                    catdata=phot[(phot['RA']>=float(cmdargs[2])-matchr/3600.) & (phot['RA']<=float(cmdargs[4])+matchr/3600.) & (phot['Dec']>=float(cmdargs[3])-matchr/3600.) & (phot['Dec']<=float(cmdargs[5])+matchr/3600.)]
                else:
                    catdata=phot[(phot['RA']>=float(cmdargs[2])-matchr/3600.) & (phot['RA']<=float(cmdargs[4])+matchr/3600.) & (phot['Dec']>=float(cmdargs[3])-matchr/3600.) & (phot['Dec']<=float(cmdargs[5])+matchr/3600.)]
            if (len(catdata)>0):
                if (verbosity >=50):
                    print ("Queried",catname,"- found",len(catdata[0]),"objects ->",photprocfile)
                np.save(photprocfile,catdata)
            else:
                if (verbosity >=90):
                    print ("Queried",catname,"- found no objects")
                try: # Remove old file...
                    remove(photprocfile)
                except FileNotFoundError: #...if it exists
                    pass
                pass
    return

# -----------------------------------------------------------------------------
def get_sed_multiple(method="",ra1=0.,ra2=0.):
    # Generate a set of SEDs from spatially overlapping data
    # - Get data from Gaia
    # - Get data from CDS
    # - Match the photometry based on PM-corrected position
    # - Deblend if necessary
    # - Merge observations at the same wavelength

    # Philosophy:
    # - Start with the base catalogue
    # - Progressively match against other catalogues in order of matching radius
    # - Match unique cross-matches (one<->one mapping in both catalogues)
    # - Treat other objects as potentially blended and solve as necessary

    if (verbosity>=30):
        print ("Constructing SEDs for an area search...")

    # Getting general catalogue and filter data
    catdata=get_catalogue_list()
    filtdata=get_filter_list()
    rejectdata=get_reject_list()
    svodata=get_svo_data(filtdata)
    # Set up ancillary data array
    # (If number of queries == 1, expand dimensions to avoid error)
    ancillary_queries=get_ancillary_list()
    if (np.size(ancillary_queries)==1):
            ancillary_queries=np.expand_dims(ancillary_queries,axis=0)
    # Identify master catalogue
    mastercat=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PrimaryCatRef",1])[2:-2]
    # Use any existing data?
    usepreviousrun=int(pyssedsetupdata[pyssedsetupdata[:,0]=="UsePreviousRun",1][0])

    # Append master catalogue ID to ancillary queries don't include the master catalogue
    if (len(ancillary_queries['catname'==mastercat])==0):
        if (len(catdata[catdata['catname']==mastercat]['catname'])==0):
            print_fail ("Master catalogue is not in list of catalogues")
            print ("Master catalogue:",mastercat)
            print ("List of catalogues:",catdata['catname'])
            raise
        mastercatdata=np.empty_like(ancillary_queries, shape=1)
        mastercatdata['server']=catdata[catdata['catname']==mastercat]['server']
        mastercatdata['catname']=catdata[catdata['catname']==mastercat]['catname']
        mastercatdata['cdstable']=catdata[catdata['catname']==mastercat]['cdstable']
        mastercatdata['colname']=catdata[catdata['catname']==mastercat]['idcol']
        mastercatdata['errname']=''
        mastercatdata['paramname']='ID'
        mastercatdata['multiplier']=0.
        mastercatdata['epoch']=catdata[catdata['catname']==mastercat]['epoch']
        mastercatdata['beamsize']=catdata[catdata['catname']==mastercat]['beamsize']
        mastercatdata['matchr']=catdata[catdata['catname']==mastercat]['matchr']
        mastercatdata['priority']=0.
        ancillary_queries=np.append(mastercatdata,ancillary_queries,axis=0)

    # Get cross-match data
    if (usepreviousrun>2):
        if (verbosity >=50):
            print ("Loading previous cross-match data")
        # Photometry
        xmatchdatafile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotXMatchDataFile",1])[2:-2]
        xmatchcountsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotXMatchCountsFile",1])[2:-2]
        xmatch=np.loadtxt(xmatchdatafile,dtype=object)
        x=np.loadtxt(xmatchcountsfile,dtype=object)
        photdata=np.hstack((np.expand_dims(x[:,0],axis=1).astype(object),x[:,1:7].astype(float)))
        nmatch=x[:,7:].astype(int)
        # Ancillary
        xmatchdatafile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncXMatchDataFile",1])[2:-2]
        xmatchcountsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncXMatchCountsFile",1])[2:-2]
        xmatchanc=np.loadtxt(xmatchdatafile,dtype=object)
        x=np.loadtxt(xmatchcountsfile,dtype=object)
        ancdata=np.hstack((np.expand_dims(x[:,0],axis=1).astype(object),x[:,1:7].astype(float)))
        anmatch=x[:,7:].astype(int)
    else:
        xmatch,photdata,nmatch=get_xmatches(catdata,"phot")
        xmatchanc,ancdata,anmatch=get_xmatches(ancillary_queries,"anc")
    
    # Use cross-matches to build SEDs
    if (usepreviousrun>3):
        if (verbosity >=50):
            print ("Loading previous SEDs")
        # Photometry
        sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaSEDsFile",1])[2:-2]
        weightsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaWeightsFile",1])[2:-2]
        seds=np.loadtxt(sedsfile,dtype=int)
        weights=np.loadtxt(weightsfile,dtype=float)
        # Ancillary
        sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaAncsFile",1])[2:-2]
        weightsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaAncWeightsFile",1])[2:-2]
        ancs=np.loadtxt(sedsfile,dtype=int)
        ancweights=np.loadtxt(weightsfile,dtype=float)
    else:
        seds,weights=get_areaseds(xmatch,photdata,nmatch,"phot")
        ancs,ancweights=get_areaseds(xmatchanc,ancdata,anmatch,"anc")
    # If area only contains one object
    if (np.ndim(seds)==1):
        seds=np.atleast_2d(seds)
        ancs=np.atleast_2d(ancs)

    # Extract data for the SEDs and write to master SEDs file
    if (usepreviousrun>4):
        if (verbosity >=50):
            print ("Loading previous compiled SEDs")
        # Photometry
        sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledSEDsFile",1])[2:-2]
        compiledseds=np.load(sedsfile,allow_pickle=True)
        # Ancillary
        sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledAncFile",1])[2:-2]
        compiledanc=np.load(sedsfile,allow_pickle=True)
        # Source list
        sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledSourceFile",1])[2:-2]
        sourcedata=np.loadtxt(sedsfile,delimiter="\t",dtype=str)
    else:
        compiledseds,compiledanc,sourcedata=compile_areaseds(seds,weights,photdata,ancs,ancweights,ancdata)

    # Deal with trimming boxes in RA
    if (method=="box"):
        trimbox=int(pyssedsetupdata[pyssedsetupdata[:,0]=="TrimBox",1][0])
        if (trimbox>0):
            if (verbosity>50):
                print ("Trimming box edges to fit RA limits")
            ras=np.array([a[0][3] for a in compiledanc])
            compiledseds=compiledseds[(ras>=ra1) & (ras<ra2)]
            compiledanc=compiledanc[(ras>=ra1) & (ras<ra2)]
            sourcedata=sourcedata[(ras>=ra1) & (ras<ra2)]

    return compiledseds,compiledanc,sourcedata

# -----------------------------------------------------------------------------
def get_xmatches(catdata,matchtype):

    # Get location of master catalogue
    mastercat=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PrimaryCatRef",1])[2:-2]
    phottempdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotTempDir",1])[2:-2]
    mastercatfile=phottempdir+mastercat+".npy"

    # Set location of data
    if (matchtype=="phot"):
        if (verbosity>=50):
            print ("Extracting catalogue data...")
    elif (matchtype=="anc"):
        if (verbosity>=50):
            print ("Extracting ancillary data...")
        phottempdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncTempDir",1])[2:-2]
    else:
        print_fail("ERROR! (Attempting to proceed regardless)")
        print_fail("get_xmatches matchtype should be phot or anc")
        print_fail("received",matchtype)

    # Puts master catalogue first
    pmtype=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PMCorrType",1][0])
    cats=np.unique(catdata['catname'])
    cats=np.append(mastercat,cats[~np.isin(cats,mastercat)])

    # Parse each catalogue and load it into a master list
    usedcats=[]
    for cat in np.unique(catdata['catname']):
        # Open the downloaded data and try extracting data (first RA & Dec)
        try:
            if (cat==mastercat):
                data=np.load(mastercatfile)
            else:
                data=np.load(phottempdir+cat+".npy")
        except FileNotFoundError:
            if (verbosity>=95):
                print ("No file called",phottempdir+cat+".npy","found (perhaps no data)")
            data=[]
        else:
            usedcats=np.append(usedcats,cat)
            if (verbosity>=95):
                print ("Extracted",len(data),"points from",cat)

            ra,dec=extract_ra_dec(data)
            catarray=np.full_like(ra,cat,dtype="object")
            a=np.column_stack([catarray,ra,dec])
            try:
                photdata=np.append(photdata,a,axis=0)
            except: # if it doesn't already exist
                photdata=a
    if (verbosity>=60):
        print ("Extracted",len(photdata),"points from all catalogues")

    # -------------------------------------------------
    # Extract PM data from master catalogue if required
    if (verbosity>=50):
        print ("Assigning proper motions from master catalogue...")
    
    # First check master catalogue is in used catalogues
    if (np.isin(mastercat,usedcats)==False):
        print_fail("Master catalogue "+mastercat+" is not among the catalogues identified as having a non-zero number of sources.")
        print_fail("Execution will likely terminate soon.")
        print ("Catalogues:")
        print (usedcats)
        #raise
    
    if (pmtype==0):
        defaultpmra=0.
        defaultpmdec=0.
    else:
        defaultpmra=float(pyssedsetupdata[pyssedsetupdata[:,0]=="FixedPMRA",1][0])
        defaultpmdec=float(pyssedsetupdata[pyssedsetupdata[:,0]=="FixedPMDec",1][0])
    try:
        sourcedata=np.load(mastercatfile)
    except FileNotFoundError:
        print_fail("Master data catalogue not found!")
        print_fail("Check setup parameters PrimaryCatRef and UsePreviousRun.")
        return
        #raise
    else:
        if (verbosity>=95):
            print ("Extracted",len(sourcedata),"points from",mastercat)
    sourcetype,ra,dec,raerr,decerr,masterpmra,masterpmdec,masterpmraerr,masterpmdecerr,sourceepoch=extract_ra_dec_pm(sourcedata)
    # ...and append to extracted photometric data
    #    Remove nans
    masterpmra[np.isnan(masterpmra)]=0.
    masterpmdec[np.isnan(masterpmdec)]=0.
    masterpmraerr[np.isnan(masterpmraerr)]=0.
    masterpmdecerr[np.isnan(masterpmdecerr)]=0.
    #    Create 1D arrays
    pmra=np.full(len(photdata),defaultpmra)
    pmdec=np.full(len(photdata),defaultpmdec)
    pmraerr=np.zeros(len(photdata))
    pmdecerr=np.zeros(len(photdata))
    #    Set PM info within 1D arrays
    a=(photdata[:,0]==mastercat)
    if (len(pmra[a])>0):
        pmra[a]=masterpmra
        pmdec[a]=masterpmdec
        pmraerr[a]=masterpmraerr
        pmdecerr[a]=masterpmdecerr
    # Append to photometric data
    photdata=np.column_stack([photdata,pmra,pmdec,pmraerr,pmdecerr])
        
    # ----------------------
    # Identify cross-matches
    # For each pair of catalogues, first correct proper motion to mean epoch
    # Compute nth-order matches until n exceeds MaxBlends or no more sources within beam
    # Proper motion data is inherited by new cross-matches
    # Big table is too expensive (100k*100k sources = 1G potential cross-matches)
    # so only record the cross-matches that exist
    if (verbosity>=50):
        print ("Identifying cross-matches...")

    # Puts master catalogue first
    orderedcats=np.append(mastercat,usedcats[~np.isin(usedcats,mastercat)])
    
    # Numbers of cross-matches
    nmatch=np.zeros((len(photdata),len(orderedcats)),dtype=int)

    # Maximum number of blends to consider
    maxblends=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxBlends",1][0])
    
    # Remove duplicate entries from catdata
    for i in np.arange(len(catdata)-1,0,-1):
        if (np.count_nonzero(catdata[0:i]['catname']==catdata[i]['catname'])>0):
            catdata=np.delete(catdata,i,axis=0)
    
    # Loop over catalogue pairs
    for cat1 in orderedcats:
        for cat2 in orderedcats[orderedcats!=cat1]:
            if (verbosity>=70):
                print ("   Processing",cat1,"vs.",cat2)

            # Get catalogue data for each catalogue
            ra1=photdata[photdata[:,0]==cat1,1]
            dec1=photdata[photdata[:,0]==cat1,2]
            pmra1=photdata[photdata[:,0]==cat1,3]
            pmdec1=photdata[photdata[:,0]==cat1,4]
            epoch1=catdata[catdata['catname']==cat1]['epoch']
            ra2=photdata[photdata[:,0]==cat2,1]
            dec2=photdata[photdata[:,0]==cat2,2]
            pmra2=photdata[photdata[:,0]==cat2,3]
            pmdec2=photdata[photdata[:,0]==cat2,4]
            epoch2=catdata[catdata['catname']==cat2]['epoch']

            # ------------------------
            # Proper motion correction
            if (verbosity>=90):
                print ("      Proper motion correction")

            # What proper motion information exists?
            # Calculate fraction of sources with PM in both cats (non-default, non-zero)
            pmfrac1=(((pmra1!=defaultpmra) | (pmdec1!=defaultpmdec)) & ((pmra1!=0.) | (pmdec1!=0.))).sum()/len(pmra1)
            pmfrac2=(((pmra2!=defaultpmra) | (pmdec2!=defaultpmdec)) & ((pmra2!=0.) | (pmdec2!=0.))).sum()/len(pmra2)
            if (verbosity>=97):
                print ("         ",pmfrac1*100.,"% of",cat1,"sources have PM")
                print ("         ",pmfrac2*100.,"% of",cat2,"sources have PM")

            # Calculate appropriate epochs and perform PM correction
            # Four proper motion cases:
            # [1] Proper motions for both catalogues
            # [2] Proper motions for cat1 but not cat2
            # [3] Proper motions for cat2 but not cat1
            # [4] Proper motions for neither catalogue or PMCorr==0
            if ((pmfrac1>0 and pmfrac2>0) or pmtype==0): #1
                epochoffset1=(epoch2-epoch1)/2.
                epochoffset2=(epoch1-epoch2)/2.
                if (verbosity>=99):
                    print ("         Case 1")
            elif (pmfrac1>0): #2
                epochoffset1=(epoch2-epoch1)
                epochoffset2=0.
                if (verbosity>=99):
                    print ("         Case 2")
            elif (pmfrac2>0): #3
                epochoffset1=0.
                epochoffset2=(epoch1-epoch2)
                if (verbosity>=99):
                    print ("         Case 3")
            else: #4
                epochoffset1=0.
                epochoffset2=0.
                if (verbosity>=99):
                    print ("         Case 4")
            # If non-zero, apply these PM offsets to the catalogues
            if (epochoffset1==0.):
                newra1=ra1
                newdec1=dec1
            else:
                newra1,newdec1=pm_calc(ra1,dec1,pmra1,pmdec1,0,epochoffset1)
            if (epochoffset2==0.):
                newra2=ra2
                newdec2=dec2
            else:
                newra2,newdec2=pm_calc(ra2,dec2,pmra2,pmdec2,0,epochoffset2)
            
            # Identify sources within the beam of each source
            if (verbosity>=90):
                print ("      Neighbour matching")
            coords1=SkyCoord(ra=newra1, dec=newdec1, unit='deg')
            coords2=SkyCoord(ra=newra2, dec=newdec2, unit='deg')
            nmatches=999999
            i=0
            while ((nmatches>0) & (i<maxblends) & (i<len(coords2))):
                i+=1
                matchidx,sep2d,sep3d=match_coordinates_sky(coords1,coords2,nthneighbor=i,storekdtree=(cat1.astype(str)+cat2.astype(str)))
                cat1idx=np.nonzero(sep2d.arcsecond<catdata[catdata['catname']==cat1]['beamsize'])[0]
                cat2idx=matchidx[sep2d.arcsecond<catdata[catdata['catname']==cat1]['beamsize']]
                nmatches=len(cat2idx)
                if nmatches>0:
                    nmatch[photdata[:,0]==cat1,orderedcats==cat2]=np.where(sep2d.arcsecond<catdata[catdata['catname']==cat1]['beamsize'],i,nmatch[photdata[:,0]==cat1,orderedcats==cat2])
                    cats1=np.full(len(cat1idx),cat1)
                    cats2=np.full(len(cat2idx),cat2)
                    xmatchdata=np.column_stack([cats1,cat1idx,cats2,cat2idx])
                    try:
                        xmatch=np.append(xmatch,xmatchdata,axis=0)
                    except UnboundLocalError:
                        xmatch=xmatchdata
                    if (verbosity>=98):
                        print ("         ",cat1,cat2,"Iteration:",i,"#matches:",nmatches)
            if (verbosity>=95):
                if nmatches>0:
                    print ("      ",len(xmatch),"matches across all processed catalogues")
                else:
                    print_warn ("      Zero matches across all processed catalogues!")

    if (verbosity>=70):
        print (np.count_nonzero(nmatch==0),"source-catalogue pairs have zero matches")
        print (np.count_nonzero(nmatch==1),"source-catalogue pairs have unique matches")
        print (np.count_nonzero(nmatch>1),"source-catalogue pairs have many matches")
        print (np.count_nonzero(nmatch>=maxblends),"source-catalogue pairs exceed the MaxBlends setting")

    # Plot locations of matches if required
    makeplots=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MakeAreaPlots",1][0])
    makeskyplot=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MakeMatchSkyPlot",1][0])
    plotfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="MatchSkyPlotFile",1])[2:-2]
    if ((makeplots>0) and (makeskyplot>0) and (matchtype=="phot")):
        globalmatchskyplot(plotfile,photdata,nmatch,orderedcats)

    # Save photometric matches
    savexmatch=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SavePhotXMatch",1][0])
    if (matchtype=="phot"):
        xmatchdatafile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotXMatchDataFile",1])[2:-2]
        xmatchcountsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotXMatchCountsFile",1])[2:-2]
    if (matchtype=="anc"):
        xmatchdatafile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncXMatchDataFile",1])[2:-2]
        xmatchcountsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncXMatchCountsFile",1])[2:-2]
    if (savexmatch>0):
        if (verbosity>=70):
            print ("Saving results to",xmatchdatafile)
            print ("Saving results to",xmatchcountsfile)
        np.savetxt(xmatchdatafile,xmatch,delimiter="\t",fmt="%s")
        np.savetxt(xmatchcountsfile,np.hstack((photdata,nmatch)),delimiter="\t",fmt="%s")

    # Match catalogues depending on nmatch products
    # = number of sources lying within the beam of the first and seconds surveys, respectively
    #   0+0 = no matches for that source
    #   1+0 = (outlier) or (assign cross-match)
    #   Many+0 = assign catalogue #1 source as blend
    #   0+1 = (outlier) or (assign cross-match)
    #   1+1 = assign cross-match
    #   Many+1 = (assign cat #1 source as blend) or (assign cross-match)
    #   0+Many = assign catalogue #1 source as blend
    #   1+Many = (assign cat #2 source as blend) or (assign cross-match)
    #   Many+Many = assign both sources as blends
    # Whether result (0) or (1) is chosen depends on RobustLoResMatch and DeblendLoResMatch

    # XXX MEMORY TESTING XXX
    #snapshot = tracemalloc.take_snapshot()
    #top_stats = snapshot.statistics('lineno')
    #for stat in top_stats[:20]:
    #    print(stat)
    #exit()

    return xmatch,photdata,nmatch

# -----------------------------------------------------------------------------
def get_areaseds(xmatch,photdata,nmatch,matchtype):
    # Use cross-match information to construct SEDs over an area

    # Get catalogue list
    mastercat=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PrimaryCatRef",1])[2:-2]
    usedcats=np.unique(photdata[:,0])
    orderedcats=np.append(mastercat,usedcats[~np.isin(usedcats,mastercat)])

    # Get treatment of blends from parameter file
    # -------------------------------------------
    # Treatment of potential matches
    # None+One: If a lo-res catalogue matches a hi-res catalogue, but not vice versa
    # 0 = assign the match
    # 1 = treat as two separate sources
    robustlores=int(pyssedsetupdata[pyssedsetupdata[:,0]=="RobustLoResMatch",1][0])
    # Many+One: If a lo-res catalogue matches several sources,
    # but is a match to only one source in the hi-res catalogue
    # 0 = assign the match
    # 1 = treat as a potential blend
    deblendlores=int(pyssedsetupdata[pyssedsetupdata[:,0]=="DeblendLoResMatch",1][0])
    # Many+None: If a lo-res catalogue matches several sources,
    # but is not a match to any source in the hi-res catalogue
    # 0 = ignore the match
    # 1 = assign to brightest matching source in hi-res catalogue
    # 2 = assign to closest source in hi-res catalogue
    # 3 = assign based on a weighting of 1 and 2
    # 4 = try to deblend between sources
    deblendloresnonmatch=int(pyssedsetupdata[pyssedsetupdata[:,0]=="DeblendLoResNonMatch",1][0])
    # Minimum weight to consider a source as a significant blend
    deblendminwt=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DeblendMinWeight",1][0])
    # Minimum weight of brightest source to activiate deblending
    deblendactwt=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DeblendActivationWeight",1][0])

    # Set up initial SED and weights arrays
    # We'll populate these with the IDs of the crossmatches and the fraction of flux assigned to them
    seds=np.full((np.count_nonzero(photdata[:,0]==mastercat),len(orderedcats)),-1,dtype=int)
    weights=np.zeros((np.count_nonzero(photdata[:,0]==mastercat),len(orderedcats)),dtype=float)
    # ...beginning with those from the master catalogue
    #seds[:,0]=np.nonzero(photdata[:,0]==mastercat)[0]
    seds[:,0]=np.arange(len(np.nonzero(photdata[:,0]==mastercat)[0]))
    weights[:,0]=1.

    # Now loop over catalogues, assigning matches
    for i in np.arange(1,len(orderedcats),1):
        masterids=xmatch[(xmatch[:,0]==mastercat) & (xmatch[:,2]==orderedcats[i]),1].astype(int) # IDs of successful matches from master
        catids=xmatch[(xmatch[:,0]==mastercat) & (xmatch[:,2]==orderedcats[i]),3].astype(int) # IDs of successful matches from matching catalogue
        masterids0=masterids[nmatch[photdata[:,0]==mastercat,i][masterids]==0] # No matches in master
        masterids1=masterids[nmatch[photdata[:,0]==mastercat,i][masterids]==1] # Unique matches in master
        masterids2=masterids[nmatch[photdata[:,0]==mastercat,i][masterids]>1] # Many matches in master
        catids0=catids[nmatch[photdata[:,0]==orderedcats[i],0][catids]==0] # No matches in matching catalogue
        catids1=catids[nmatch[photdata[:,0]==orderedcats[i],0][catids]==1] # Unique matches in matching catalogue
        catids2=catids[nmatch[photdata[:,0]==orderedcats[i],0][catids]>1] # Many matches in matching catalogue
        idmatch01=((xmatch[:,2]==mastercat) & (xmatch[:,0]==orderedcats[i]) & np.isin(xmatch[:,3].astype(int),masterids0) & np.isin(xmatch[:,1].astype(int),catids1)).nonzero()[0] # Index where no cross-match in master, but unique in secondary
        idmatch02=((xmatch[:,2]==mastercat) & (xmatch[:,0]==orderedcats[i]) & np.isin(xmatch[:,3].astype(int),masterids0) & np.isin(xmatch[:,1].astype(int),catids2)).nonzero()[0] # Index where no cross-match in master, multiple in secondary
        idmatch10=((xmatch[:,0]==mastercat) & (xmatch[:,2]==orderedcats[i]) & np.isin(xmatch[:,1].astype(int),masterids1) & np.isin(xmatch[:,3].astype(int),catids0)).nonzero()[0] # Index where unique in master, no match in secondary
        idmatch11=((xmatch[:,0]==mastercat) & (xmatch[:,2]==orderedcats[i]) & np.isin(xmatch[:,1].astype(int),masterids1) & np.isin(xmatch[:,3].astype(int),catids1)).nonzero()[0] # Index where unique cross-match in both
        idmatch12=((xmatch[:,0]==mastercat) & (xmatch[:,2]==orderedcats[i]) & np.isin(xmatch[:,1].astype(int),masterids1) & np.isin(xmatch[:,3].astype(int),catids2)).nonzero()[0] # Index where unique in master, multiple in secondary
        masterid01=xmatch[idmatch01,3].astype(int)
        masterid02=xmatch[idmatch02,3].astype(int)
        masterid10=xmatch[idmatch10,1].astype(int) # corresponding cross-match IDs in master catalogue
        masterid11=xmatch[idmatch11,1].astype(int)
        masterid12=xmatch[idmatch12,1].astype(int)
        xmatch01=xmatch[idmatch01,1].astype(int)
        xmatch02=xmatch[idmatch02,1].astype(int)
        xmatch10=xmatch[idmatch10,3].astype(int) # corresponding cross-match IDs in matching catalogue
        xmatch11=xmatch[idmatch11,3].astype(int)
        xmatch12=xmatch[idmatch12,3].astype(int)
        if (verbosity>90):
            print (mastercat," Zero   <->",orderedcats[i],"Zero,Unique,Many matches = -",len(xmatch01),len(xmatch02))
            print (mastercat," Unique <->",orderedcats[i],"Zero,Unique,Many matches =",len(xmatch10),len(xmatch11),len(xmatch12))
        
        # Assign direct 1-1 matches
        seds[masterid11,i]=xmatch11 # assign cross-match IDs to SEDs
        weights[masterid11,i]=1.
        
        # If a lower-resolution catalogue matches the master catalogue uniquely, apply it
        # even if the master catalogue shows no match
        # (i.e. support for poor astrometry in low-resolution matches at the risk of mis-matching sources which drop out of certain catalogues)
        if (robustlores==0):
            seds[masterid01,i]=xmatch01 # assign cross-match IDs to SEDs
            weights[masterid01,i]=1.

        # If the master catalogue matches a lower-resolution source uniquely,
        # but the master catalogue still matches only that low-resolution source, apply it
        # (i.e. easy support for bright stars with good astrometry, at the risk of mis-matching poor-astrometry sources by chance)
        if (deblendlores==0):
            seds[masterid12,i]=xmatch12 # assign cross-match IDs to SEDs
            weights[masterid12,i]=1.

    if (verbosity>60):
        print (np.count_nonzero(seds[:,1:]),"matches assigned to SEDs (",np.count_nonzero(seds[:,1:])/np.sum(nmatch[photdata[:,0]==mastercat,:])*100.,"% of",mastercat,"cross-matches)")

    # This is now done at the end of the main loop to incorporate modelling data
    #saveseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveSEDs",1][0])
    #if (matchtype=="phot"):
    #    sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaSEDsFile",1])[2:-2]
    #    weightsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaWeightsFile",1])[2:-2]
    #elif (matchtype=="anc"):
    #    sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaAncsFile",1])[2:-2]
    #    weightsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaAncWeightsFile",1])[2:-2]
    #outappend=int(pyssedsetupdata[pyssedsetupdata[:,0]=="OutputAppend",1][0])
    #if (saveseds>0 and outappend=0):
    #    np.savetxt(sedsfile,seds,delimiter="\t",fmt="%s")
    #    np.savetxt(weightsfile,weights,delimiter="\t",fmt="%s")

    return seds,weights

# -----------------------------------------------------------------------------
def compile_areaseds(seds,weights,photdata,ancs,ancweights,ancdata,ra1=0.,ra2=0.):
    # Take cross-matched SED indices and extract data from files
    # Follows a modified process of get_vizier_single

    if (verbosity>=80):
        print ("Loading data and setting parameters")

    # Get catalogue list
    mastercat=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PrimaryCatRef",1])[2:-2]
    phottempdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotTempDir",1])[2:-2]
    mastercatfile=phottempdir+mastercat+".npy"
    usedcats=np.unique(photdata[:,0])
    orderedcats=np.append(mastercat,usedcats[~np.isin(usedcats,mastercat)])
    usedanccats=np.unique(ancdata[:,0])
    orderedanccats=np.append(mastercat,usedanccats[~np.isin(usedanccats,mastercat)])

    # Get from files for catalogues and filters
    catdata=get_catalogue_list()
    filtdata=get_filter_list()
    rejectdata=get_reject_list()
    # Get data from SVO for filters too
    svodata=get_svo_data(filtdata)
    # Set up ancillary data array
    ancillary_queries=get_ancillary_list()
    if (np.size(ancillary_queries)==1):
            ancillary_queries=np.expand_dims(ancillary_queries,axis=0)
    ancillary=np.zeros([np.size(ancillary_queries)+5],dtype=[('parameter',object),('catname',object),('colname',object),('value',object),('err',object),('priority',object),('mask',bool)])
    
    # Load catalogues into memory
    catalogues=np.empty(len(orderedcats),dtype=object)
    catdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotTempDir",1])[2:-2]
    for i in np.arange(len(orderedcats)):
        cataloguedata=np.load(catdir+orderedcats[i]+".npy")
        catalogues[i]=cataloguedata
    anccatalogues=np.empty(len(orderedanccats),dtype=object)
    catdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncTempDir",1])[2:-2]
    for i in np.arange(len(orderedanccats)):
        if (orderedanccats[i]==mastercat):
            cataloguedata=np.load(mastercatfile)
        else:
            cataloguedata=np.load(catdir+orderedanccats[i]+".npy")
        anccatalogues[i]=cataloguedata

    # Set the default photometric error
    defaulterr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultError",1][0])

    # Reject criteria if wavelength too short or long
    minlambda=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MinLambda",1])*10000.
    maxlambda=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxLambda",1])*10000.
    
    # Get the ID columns for each catalogue
    idcol=np.empty(len(orderedcats),dtype=object)
    for i in np.arange(len(orderedcats)):
        idcol[i]=np.squeeze(catdata[catdata['catname']==orderedcats[i]]['idcol'])

    if (verbosity>=80):
        print ("Creating SEDs from cross-matches")

    # Process catalogues into SEDs
    compiledseds=np.zeros(len(seds),dtype=object)
    compiledanc=np.zeros(len(seds),dtype=object)
    sourcedata=np.zeros(len(seds),dtype=object)
    nssuccess=0
    for i in np.arange(len(seds)):
        if (verbosity>=90):
            print ("Creating SED of object",i+1,"of",len(seds))
        nfsuccess=0
        sed=np.zeros((len(filtdata)),dtype=[('catname','<U20'),('objid','<U32'),('ra','f4'),('dec','f4'),('modelra','f4'),('modeldec','f4'),('svoname','U32'),('filter','U10'),('wavel','f4'),('dw','f4'),('mag','f4'),('magerr','f4'),('flux','f4'),('ferr','f4'),('dered','f4'),('derederr','f4'),('model','f4'),('mask','bool')])
        for j in np.arange(len(orderedcats)):
            if (seds[i,j]>=0):
                try:
                    data=catalogues[j][seds[i,j]]
                except IndexError:
                    print (catalogues[j])
                    print (catalogues[j-1])
                    raise
                # Get RA & Dec
                newra,newdec=extract_ra_dec(data)
                if (j==0):
                    sourcera=newra
                    sourcedec=newdec
                    sourcetype,ra,dec,raerr,decerr,pmra,pmdec,pmraerr,pmdecerr,sourceepoch=extract_ra_dec_pm(data)
                # Get ID
                if (idcol[j]=="None"):
                    catid="None"
                else:
                    try:
                        catid=data[int(np.squeeze(np.argwhere((data.dtype.names==idcol[j])==True)))]
                    except:
                        catid="NotRecognised"
                if (j==0):
                    sourcedata[i]=str(catid)
                # Identify magnitude and error columns
                catalogue=orderedcats[j]
                magkeys=filtdata[filtdata['catname']==catalogue]['filtname']
                for k in (np.arange(len(magkeys))):
                    fdata=filtdata[(filtdata['catname']==catalogue) & (filtdata['filtname']==magkeys[k])][0]
                    svokey=fdata['svoname']
                    wavel=float(svodata[svodata['svoname']==svokey]['weff'][0])
                    dw=float(svodata[svodata['svoname']==svokey]['dw'][0])
                    reject_reasons=rejectdata[(rejectdata['catname']==catalogue) & ((rejectdata['filtname']==fdata['filtname']) | (rejectdata['filtname']=="All")) & (rejectdata['rejcat']=="Same") & (rejectdata['rejcol']=="Same")]
                    if (fdata['dataref']=='Vega'):
                        zpt=float(svodata[svodata['svoname']==svokey]['zpt'][0])
                    else:
                        zpt=3631.
                    # Extract data
                    #print (catalogue)
                    #print (data.dtype.names)
                    mag,magerr,flux,ferr,mask=get_mag_flux(data,fdata,zpt,reject_reasons)
                    # If the fractional error in the flux is sufficiently small
                    if (flux>0):
                        if (ferr/flux<fdata['maxperr']/100.):
                            sed[nfsuccess]=(catalogue,catid,(newra-sourcera)*3600.,(newdec-sourcedec)*3600.,(ra-sourcera)*3600.,(dec-sourcedec)*3600.,svokey,fdata['filtname'],wavel,dw,mag,magerr,flux,ferr,0,0,0,mask)
                            nfsuccess+=1
        # Remove zero entries
        sed=sed[sed['flux']>0.]
        # If there is no error in flux, use default error
        sed[sed['ferr']==0]['ferr']==sed[sed['ferr']==0]['flux']*defaulterr
        # Reject if wavelength too short or long
        sed[:]['mask']=np.where(sed[:]['wavel']<minlambda,False,sed[:]['mask'])
        sed[:]['mask']=np.where(sed[:]['wavel']>maxlambda,False,sed[:]['mask'])
        if (nfsuccess>1):
            nssuccess+=1
            compiledseds[i]=sed
            
        # Get ancillary information
        if (verbosity>=94):
            print ("Getting ancillary data of object",i+1,"of",len(seds))
        nfsuccess=5
        ancillary=np.zeros([np.size(ancillary_queries)+5],dtype=[('parameter',object),('catname',object),('colname',object),('value',object),('err',object),('priority',object),('mask',bool)])
        #idcol=np.empty(len(orderedanccats),dtype=object)
        #for i in np.arange(len(orderedanccats)):
        #    idcol[i]=np.squeeze(ancillary[ancillary['catname']==orderedanccats[i]]['idcol'])
        for j in np.arange(len(orderedanccats)):
            if (ancs[i,j]>=0):
                if (verbosity>=95):
                    print ("Catalogue:",j,orderedanccats[j])
                data=anccatalogues[j][ancs[i,j]]
                # Get RA & Dec
                newra,newdec=extract_ra_dec(data)
                if (j==0):
                    sourcera=newra
                    sourcedec=newdec
                    sourcetype,ra,dec,raerr,decerr,pmra,pmdec,pmraerr,pmdecerr,sourceepoch=extract_ra_dec_pm(data)
                    ancillary[0]=('RA',sourcetype,'RA',ra,raerr,0,True)
                    ancillary[1]=('Dec',sourcetype,'Dec',dec,decerr,0,True)
                    ancillary[2]=('PMRA',sourcetype,'PMRA',pmra,pmraerr,0,True)
                    ancillary[3]=('PMDec',sourcetype,'PMDec',pmdec,pmdecerr,0,True)
                    ancillary[4]=('Dist','Adopted','Dist',0,0,0,True)
                # Get ID
                catid="None"
#                if (idcol[j]=="None"):
#                    catid="None"
#                else:
#                    try:
#                        catid=data[np.squeeze(np.argwhere((data.dtype.names==idcol[j])==True))]
#                    except:
#                        catid="NotRecognised"
                # Identify data and error columns
                catalogue=orderedanccats[j]
                anc_queries=ancillary_queries[ancillary_queries['catname']==orderedanccats[j]]
#                print (ancillary_queries['catname'],orderedanccats[j])
                for k in np.arange(np.size(anc_queries)):
                    reasons=rejectdata[(rejectdata['catname']==anc_queries[k]['catname']) & ((rejectdata['filtname']==anc_queries[k]['colname']) | (rejectdata['filtname']=="All")) & (rejectdata['rejcat']=="Same") & (rejectdata['rejcol']=="Same")]
                    #print (reasons)
                    data,mask=reject_test(reasons,data)
                    if (anc_queries[k]['errname']=="None"):
                        err=0
                    elif (isfloat(anc_queries[k]['errname'])):
                        err=float(anc_queries[k]['errname'])
                    elif ("/" in anc_queries[k]['errname']): # allow upper/lower confidence limits
                        errs=str.split(anc_queries[k]['errname'],sep="/")
                        err=(data[errs[0]]-data[errs[1]])/2.
                    elif (":" in anc_queries[k]['errname']): # allow upper/lower errors
                        errs=str.split(anc_queries[k]['errname'],sep=":")
                        err=(data[errs[0]]+data[errs[1]])/2.
                    else:
                        try:
                            err=data[anc_queries[k]['errname']]
                        except ValueError as e:
                            print_fail ("Key error: entry in ancillary request file does not match VizieR table")
                            print ("Query:",anc_queries[k])
                            print ("Available keys:",data.dtype.names)
                            print_fail ("Field in Query must match field in Available. Edit ancillary queries.")
                            raise
#                    if ("BP_RP" in anc_queries[k]['paramname']):
#                    print ("!",anc_queries[k]['colname'], data.dtype.names,data.dtype.names[91],data[91])
#                    print (data[anc_queries[k]['colname']])
                    try:
                        #if (data[anc_queries[k]['colname']]!="--"):
                        ancillary[nfsuccess]=(anc_queries[k]['paramname'],anc_queries[k]['catname'],anc_queries[k]['colname'],data[anc_queries[k]['colname']],err,anc_queries[k]['priority'],mask)
                        if (verbosity>=99):
                            print (nfsuccess,ancillary[k+5])
                        nfsuccess+=1
                    except ValueError as e:
                        if (verbosity>95):
                            print_warn ("Warning:")
                            print (e)
        # Get distance
        # If distance exists, put it in slot 4
        # If no distance exists, put empty distance in slot 4
        ancillary[4][3]=adopt_distance(ancillary)

        ancillary=ancillary[ancillary['parameter']!=0]
        if (nfsuccess>=0):
            nssuccess+=1
            compiledanc[i]=ancillary
    
    # Remove zero entires and save output
    # Not done any more - now at end of main loop so that modelling data can be added and appended
    #if (verbosity>=80):
    #    print ("Saving results")
    #saveseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveSEDs",1][0])
    #sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledSEDsFile",1])[2:-2]
    #ancfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledAncFile",1])[2:-2]
    #sourcefile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledSourceFile",1])[2:-2]
    #if (saveseds>0):
    #    np.save(sedsfile,compiledseds)
    #    np.save(ancfile,compiledanc)
    #    np.savetxt(sourcefile,sourcedata,delimiter="\t",fmt="%s")
    
    return compiledseds,compiledanc,sourcedata

# =============================================================================
# PARAMETER ESTIMATION
# =============================================================================
#def get_gtomo_ebv(dist,ext_dist50,ext_av50,ext_dist25,ext_av25):
def get_gtomo_av(dist,ext_dist50,ext_av50):
    #max_ext_dist25=np.max(ext_dist25)
    max_ext_dist50=np.max(ext_dist50)
    if (dist > max_ext_dist50):
        av=ext_av50[-1]
    #elif (dist > max_est_dist25):
    else:
        f=interpolate.interp1d(ext_dist50,ext_av50)
        av=f(dist)
    #else:
        #f=interpolate.interp1d(ext_dist25,ext_av25)
        #ebv=f(dist)

    return av
    
# -----------------------------------------------------------------------------
def deredden(sed,ancillary,dist,avgrid,gtomo_ebv=-1):


    ebv=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultEBV",1][0])
    rv=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultRV",1][0])
    extmap=pyssedsetupdata[pyssedsetupdata[:,0]=="ExtMap",1]
    xcdetail=int(pyssedsetupdata[pyssedsetupdata[:,0]=="ExtCorrDetail",1][0])

    if (verbosity>80):
        if (extmap == "GTomo"):
            print ("Extinction:",extmap)
        else:
            print ("Extinction:",extmap,"at",gtomo_ebv,"mag")

    if (dist>0.):
        # Perform a simple dereddening of an SED based on a known E(B-V)
        filtdata=get_filter_list()
        ext = F99(Rv=rv) # Adopt standard Fitzpatrick 1999 reddening law
        # If needed, get extinction from 2D/3D dust maps
        if (extmap == "Gontcharov2017"):
            # Gontcharov (2017) is based on 2MASS photometry of giants
            # Extends radially to at least 700 pc
            # Specified in Vizier by integer l, b and per 20pc in radius
            # Conversion: E(B-V) = 1.7033 E(J-K)
            foo=ancillary[(ancillary['parameter']=="RA")]; ra=foo[(foo['mask']==True)]['value']
            foo=ancillary[(ancillary['parameter']=="Dec")]; dec=foo[(foo['mask']==True)]['value']
            coords=SkyCoord(ra*u.deg, dec*u.deg, frame='icrs').galactic
            l=int(coords.l.deg+0.5)
            b=int(coords.b.deg+0.5)
            d=int(dist/20.+0.5)*20
            if (verbosity>90):
                print ("l,b,d =",l,b,d)
            try:
                ebv=Vizier(catalog="J/PAZh/43/521/rlbejk",columns=['**']).query_constraints(R=d,GLON=l,GLAT=b)[0]['E_J-Ks_'][0]*1.7033
            except: 
                if (d>0): # if distance is too large
                    try:
                        ebv=Vizier(catalog="J/PAZh/43/521/rlbejk",columns=['**']).query_constraints(R=700,GLON=l,GLAT=b)[0]['E_J-Ks_'][0]*1.7033
                    except:
                        try:
                            ebv=Vizier(catalog="J/PAZh/43/521/rlbejk",columns=['**']).query_constraints(R=600,GLON=l,GLAT=b)[0]['E_J-Ks_'][0]*1.7033
                        except:
                            ebv=0
                else: # try zero distance/reddening instead
                    ebv=0
        elif (extmap == "GTomo"):
            ebv=gtomo_ebv

        if (xcdetail==0):
            wavel=np.where(sed['wavel']>=1000., sed['wavel'], 1000.) # Prevent overflowing the function
            wavel=np.where(sed['wavel']<=33333., wavel, 33333.)    #
            sed['dered'] = sed['flux']/ext.extinguish(wavel*u.AA, Ebv=ebv)
            sed['derederr'] = sed['ferr']/ext.extinguish(wavel*u.AA, Ebv=ebv)
        elif (xcdetail>0):
            if (verbosity>80):
                print ("Avoiding dereddening now. Detailed treatment during fitting requested.")
            sed['dered'] = sed['flux']
            sed['derederr'] = sed['ferr']

    else:
        ebv=0
        sed['dered'] = sed['flux']
        sed['derederr'] = sed['ferr']

    if ((verbosity>80) & (dist>0)):
        print ("E(B-V)=",ebv,"mag")
    
    return sed,ebv

# -----------------------------------------------------------------------------
def estimate_mass(teff,lum):
    # Estimate a stellar mass based on luminosity
    if ((teff>5500.) | (lum<2.)): 
        if (((teff/7000)**9>lum) & (teff>4000)): # white dwarf
            mass=0.6
        else:
            mass=lum**(1/3.5) # MS star
    elif ((teff<=5500.) & (lum>2500.)): # AGB/supergiant
        mcore=lum/62200.+0.487 # Blocker 1993
        minit=(mcore-0.3569)/0.1197 # Casewell 2009
        mass=(minit*2+mcore)/3.
    else: # RGB
        mass=1
    if (verbosity>80):
        print ("Revised mass =",mass)
    return mass


# -----------------------------------------------------------------------------
def adopt_distance(ancillary):
    # Adopt a distance based on a set of ancillary data

    usenodist=int(pyssedsetupdata[pyssedsetupdata[:,0]=="UseStarsWithNoDist",1][0])
    maxdisterr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxDistError",1][0])
    defaultdist=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultDist",1][0])
    
    # List parallaxes & distances
    # Do this in two stages to avoid numpy/python issues about bitwise/elementwise logic
    foo=ancillary[(ancillary['parameter']=="Parallax")]
    plx=foo[(foo['mask']==True)]['value']
    plxerr=foo[(foo['mask']==True)]['err']
    
    # Sort out nans
    plx[plx!=plx] = 0
    plxerr[plxerr!=plxerr] = 0
    plx=np.nan_to_num(plx,nan=0)
    plxerr=np.nan_to_num(plxerr,nan=0)

    # Convert to distance (with error floor)
    minerr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MinAncError",1][0])
    if isinstance(plx, np.ndarray): # if multiple parallaxes
        if (len(plx)>0):
            plx=plx[plx>0]
            plxdist=1000./plx
            plxdisterr=plxerr/plx*plxdist
            plxdisterr=np.where(plxdisterr>0,plxdisterr,plxdist*minerr)
        else:
            plxdist=[]
            plxdisterr=[]
    else:
        if (len(plx)>0.):
            plxdist=1000./plx
            plxdisterr=plxerr/plx*plxdist
            plxdisterr=np.where(plxdisterr>0,plxdisterr,plxdist*minerr)
        else:
            plxdist=[]
            plxdisterr=[]
    
#    try: # If one parallax
#        if (plx>0.):
#            plxdist=1000./plx
#            plxdisterr=plxerr/plx*plxdist
#            plxdisterr=np.where(plxdisterr>0,plxdisterr,plxdist*minerr)
#        else:
#            plxdist=[]
#            plxdisterr=[]
#    except ValueError:
    
        

    # Extract distances
    foo=ancillary[(ancillary['parameter']=="Distance")]
    d=foo[(foo['mask']==True) & (foo['value']>0.)]['value']
    derr=foo[(foo['mask']==True) & (foo['value']>0.)]['err']
    if (verbosity >= 80):
        print (ancillary)
        print ("Plx:",plx,plxerr)
        print ("Dist[plx]:  ",plxdist,plxdisterr)
        print ("Dist[other]:",d,derr)

    # Identify if parallax or distance has higher priority and remove the other
    if (len(d)>0) & (len(plxdist>0)):
        priplx=ancillary[(ancillary['parameter']=="Parallax")]['priority']
        pridist=ancillary[(ancillary['parameter']=="Distance")]['priority']
        if (np.min(priplx)>np.min(pridist)):
            if (verbosity>=98):
                print ("Using distance:",priplx,pridist)
            plxdist=[]
        elif (np.min(priplx)<np.min(pridist)):
            if (verbosity>=98):
                print ("Using parallax",priplx,pridist)
            d=[]
   
    # Now combine them by weighted error
    derr=np.where(derr>0,derr,d*minerr)
    if ((len(d)>0) & (len(plxdist)>0)):
        dd=np.append(d,plxdist,axis=0)
        dderr=np.append(derr,plxdisterr,axis=0)
        try:
            dist=np.average(dd,weights=1./dderr**2)
        except TypeError:
            print_fail ("TypeError in adopt_distance")
            weights=1./dderr**2
            print (dd,d,plxdist)
            print (dderr,derr,plxdisterr)
            print (weights)
            print (np.shape(dd),np.shape(weights))
            print (ancillary)
            exit()
        ferr=np.min(dderr/dd) # Compute minimum fractional error
    elif (len(d)>0):
        dist=np.average(d,weights=1./derr**2)
        ferr=np.min(derr/dist) # Compute minimum fractional error
    elif (len(plxdist)>0):
        dist=np.average(plxdist,weights=1./plxdisterr**2)
        ferr=np.min(plxerr/plx) # Compute minimum fractional error
    else:
        dist = defaultdist # Default to fixed distance
        ferr=1.
    # XXX Needs coding for selection of best or weighted average
    # XXX Needs coding for return of errors

    if (ferr>maxdisterr):
        if (usenodist>0):
            dist = defaultdist
        else:
            dist = 0.

    # List distances
    if ((verbosity > 60) & (dist>0)):
        print ("Adopted distance:",dist,"pc")

    return dist

    
# =============================================================================
# SED FITTING
# =============================================================================
def sed_fit_bb(sed,ancillary,avdata,ebv):
    # Fit L, T_eff via a blackbody
    # XXX No outlier rejection here
    
    wavel=sed[sed['mask']>0]['wavel']/1.e10
    flux=sed[sed['mask']>0]['dered']
    ferr=sed[sed['mask']>0]['derederr']
    ferr=np.nan_to_num(ferr, nan=1.e6)
    freq=299792458./wavel
    
    dist=adopt_distance(ancillary)
    #foo=ancillary[(ancillary['parameter']=="Distance")]
    #dist=np.squeeze(foo[(foo['mask']==True)]['value'])

    if (dist>0.):

        # Get default start (guess) parameters
        modelstartteff=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartTeff",1][0])
        modelstartlogg=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartLogg",1][0])
        modelstartfeh=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartFeH",1][0])
        modelstartafe=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartAFe",1][0])
        # Get outlier rejection criteria
        maxoutliers=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxOutliers",1][0])
        minoutliers=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxOutliers",1][0])
        outtol=float(pyssedsetupdata[pyssedsetupdata[:,0]=="OutlierTolerance",1][0])
        outchisqmin=float(pyssedsetupdata[pyssedsetupdata[:,0]=="OutlierChiSqMin",1][0])
        priors=get_priors(ancillary)
        if ((modelstartteff>priors[0]['max']) or (modelstartteff<priors[0]['min'])):
            modelstartteff=(priors[0]['max']+priors[0]['min'])/2.
        avcorrtype=int(pyssedsetupdata[pyssedsetupdata[:,0]=="ExtCorrDetail",1][0])
        if (avcorrtype>0):
            sed=deredden2(sed,avdata,ebv,modelstartteff,modelstartlogg,modelstartfeh,modelstartafe)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="invalid value encountered in subtract")

            if (avcorrtype>2):
                teff=optimize.minimize(chisq_model_with_extinction,modelstartteff,args=(np.array(["bb"]),priors,[],[],sed,avdata,ebv,modelstartlogg,modelstartfeh,modelstartafe),method='Nelder-Mead')['x'][0]
            else:
                teff=optimize.minimize(chisq_model,modelstartteff,args=(freq,flux,ferr,np.array(["bb"]),priors,[],[]),method='Nelder-Mead')['x'][0]
            
            bb,fratio,chisq=compute_model(teff,freq,flux,ferr,np.array(["bb"]),priors,[],[])
        
        # Recompute after fitting with all data
    #    wavel=sed['wavel']/1.e10
    #    flux=sed['dered']
    #    ferr=sed['derederr']
    #    ferr=np.nan_to_num(ferr, nan=1.e6)
    #    freq=299792458./wavel
    #    bb,fratio,chisq=compute_model(teff,freq,flux,ferr,np.array(["bb"]),priors,[],[])

        rad=np.sqrt(fratio)
        rsun=rad/0.004652477*dist
        # in arcsec - Sun @ 1 pc = 0.004652477
        lum=(rsun)**2*(teff/5776.)**4

    else: # If no distance

        wavel=sed[sed['mask']>0]['wavel']/1.e10
        bb=np.zeros(len(sed[sed['mask']>0]))
        teff=0.
        rsun=0.
        lum=0.
        chisq=0.
    
    return sed,wavel,bb,teff,rsun,lum,chisq

# -----------------------------------------------------------------------------
def sed_fit_trapezoid(sed,ancillary,avdata,ebv):
    # Fit L via trapezoidal integration

    dist=adopt_distance(ancillary)
    flux1A=sed[sed['mask']>0]['dered'][0]*10**((-10-np.log(sed[sed['mask']>0]['wavel'][0]))*4) # Wien tail
    flux1m=sed[sed['mask']>0]['dered'][0]*10**((np.log(sed[sed['mask']>0]['wavel'][0]))*-2) # R-J tail
    wavel=np.append(np.append(np.array(1e-10),sed[sed['mask']>0]['wavel']/1.e10),np.array(1))
    flux=np.append(np.append(np.array(flux1A),sed[sed['mask']>0]['dered']),np.array(flux1m))
    #ferr=sed[sed['mask']>0]['derederr']
    #ferr=np.nan_to_num(ferr, nan=1.e6)
    #freq=299792458./wavel

    if (dist>0.):
        # Interpolate on log wavelength scale
        modelwavel=np.arange(-3,-10,-0.01)
        modelflux=10**np.interp(modelwavel,np.log10(wavel),np.log10(flux))
        modelfreq=299792458./10**modelwavel
        # Integrate
        bolflux=np.trapz(modelflux,modelfreq) # Jy*Hz = 10^26 W/m^2
        # Multiply by distance
        lum=bolflux*4*np.pi*(dist*3.086e16)**2/1.e26 # Watts
        lum/=3.846e26 # solar luminosities
    else: # If no distance
        lum=0.

    teff=0.
    rsun=0.
    chisq=0.

    model=np.zeros_like(sed['model'])
    wavel=sed['wavel']
    
    return sed,wavel,model,teff,rsun,lum,chisq

# -----------------------------------------------------------------------------
def sed_fit_simple(sed,ancillary,modeldata,avdata,ebv):
    # Fit L, T_eff & log(g) to a set of models
    # (Based on McDonald 2009-2017: Assume fixed d, [Fe/H], E(B-V), R_v)
    # (For fast computation)
    # - Either assume mass or generate using estimate_mass
    # - Perform a simple parametric fit
    # - Return parameters, goodness of fit and residual excess/deficit
    
    dist=adopt_distance(ancillary)
    #foo=ancillary[(ancillary['parameter']=="Distance")]
    #dist=np.squeeze(foo[(foo['mask']==True)]['value'])
    
    if (dist>0.):

        # Get default start (guess) parameters
        avcorrtype=int(pyssedsetupdata[pyssedsetupdata[:,0]=="ExtCorrDetail",1][0])
        fitebv=int(pyssedsetupdata[pyssedsetupdata[:,0]=="FitEBV",1][0])
        fitebvgrid=int(pyssedsetupdata[pyssedsetupdata[:,0]=="TeffEBVGrid",1][0])
        ebvoutliers=int(pyssedsetupdata[pyssedsetupdata[:,0]=="RejectOutliersDuringEBVFit",1][0])
        modelstartteff=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartTeff",1][0])
        modelstartfeh=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartFeH",1][0])
        modelstartlogg=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartLogg",1][0])
        modelstartalphafe=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartAFe",1][0])
        logg=modelstartlogg
        feh=modelstartfeh
        alphafe=modelstartalphafe
        fehfromz=int(pyssedsetupdata[pyssedsetupdata[:,0]=="FeHfromZ",1][0])
        if (fehfromz>0):
            foo=ancillary[(ancillary['parameter']=="RA")]
            ra=foo[(foo['mask']==True)]['value']
            foo=ancillary[(ancillary['parameter']=="Dec")]
            dec=foo[(foo['mask']==True)]['value']
            sc=SkyCoord(ra=ra*u.deg, dec=dec*u.deg, distance=dist*u.pc, frame='icrs')
            z=np.squeeze(np.abs(sc.cartesian.z.value/1000.))
            feh=-np.arctan(z)
            alphafe=np.arctan(z)/5
            # Check for limits
            if (feh<np.min(modeldata['metal'])):
                feh=np.min(modeldata['metal'])
            if (feh>np.max(modeldata['metal'])):
                feh=np.max(modeldata['metal'])
            if (alphafe<np.min(modeldata['alpha'])):
                alphafe=np.min(modeldata['alpha'])
            if (alphafe>np.max(modeldata['alpha'])):
                alphafe=np.max(modeldata['alpha'])
            if (verbosity>=60):
                print ("Adopting metallicity of [Fe/H] =",feh,"[alpha/Fe] =",alphafe,"based on z =",z,"kpc")

        # Get priors
        priors=get_priors(ancillary)
        if ((modelstartteff>priors[0]['max']) or (modelstartteff<priors[0]['min'])):
            print (priors[0]['max'])
            print (priors[0]['min'])
            modelstartteff=(priors[0]['max']+priors[0]['min'])/2.

        # Pre-compute log(g) if required
        iteratelogg=int(pyssedsetupdata[pyssedsetupdata[:,0]=="IterateLogg",1][0])
        precomputelogg=int(pyssedsetupdata[pyssedsetupdata[:,0]=="PrecomputeLogg",1][0])
        mass=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultMass",1][0])
        usemsmass=int(pyssedsetupdata[pyssedsetupdata[:,0]=="UseMSMass",1][0])
        if (precomputelogg > 0):
            sed,bbwavel,bb,modelstartteff,bbrsun,bblum,bbchisq=sed_fit_bb(sed,ancillary,avdata,ebv)
            if (usemsmass>0):
                mass=estimate_mass(modelstartteff,bblum)
            logg=4.44+np.log10(mass/bbrsun**2)
            # Check log(g) not too high/low
            if (logg<np.min(modeldata['logg'])):
                logg=np.min(modeldata['logg'])+0.01
            if (logg>np.max(modeldata['logg'])):
                logg=np.max(modeldata['logg'])-0.01
            if (verbosity>80):
                print ("Revised log g =",logg,"[",np.min(modeldata['logg']),np.max(modeldata['logg']),"]")
            if (modelstartteff<3000.):
                modelstartteff=3000.

        # Get outlier rejection criteria
        maxoutliers=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxOutliers",1][0])
        maxseqoutliers=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxSeqOutliers",1][0])
        minoutliers=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MinOutliers",1][0])
        maxfracoutliers=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxFracOutliers",1][0])
        outtol=float(pyssedsetupdata[pyssedsetupdata[:,0]=="OutlierTolerance",1][0])
        outchisqmin=float(pyssedsetupdata[pyssedsetupdata[:,0]=="OutlierChiSqMin",1][0])
        minlambda=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MinLambda",1][0])
        maxlambda=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxLambda",1][0])

        if (maxoutliers>int(maxfracoutliers*len(sed))):
            maxoutliers=int(maxfracoutliers*len(sed))

        try:
            start = datetime.now() # time object
        except:
            start = 0

        # Extract parameters from SED
        wavel=sed[sed['mask']>0]['wavel']/1.e10
        flux=sed[sed['mask']>0]['dered']
        ferr=sed[sed['mask']>0]['derederr']
        ferr=np.nan_to_num(ferr, nan=1.e6)
        freq=299792458./wavel
        
        # Extract parameters and flux values for observed filters
        oparams=np.stack((modeldata['teff'],modeldata['logg'],modeldata['metal'],modeldata['alpha']),axis=1)
        try:
            ovalues=np.array(modeldata[sed[sed['mask']>0]['svoname']].tolist())
        except KeyError:
            print_fail ("KeyError when extracting stellar models!")
            print ("This can occur if additional filters have been introduced that have not been convolved with the models.")
            print ("Does the first line of the model file [model-<name>-recast.dat] include all entries in the filter input file?")
            print ("If not, run 'python3 makemodel.py <name> [setup file]; python3 pyssed.py single '<any star name>' simple [setup file]; source shorten-model.scr'")
            raise
        
        # Restrict parameters by metallicity
        params,values,femask=model_subset(oparams,ovalues,teff="All",logg="All",feh=feh,alphafe=alphafe)

        # (Re)derive extinction
        if (avcorrtype>0):
            sed=deredden2(sed,avdata,ebv,modelstartteff,logg,feh,alphafe)

        # Perform main fit
        # Lower tolerance here because refining fit done later after outlier rejection
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="invalid value encountered in subtract")

            if (fitebv>0): # Fit extinction and temperature
                if (ebvoutliers==0): # Stop outlier fitting if fitting E(B-V)
                    maxoutliers=0
                if (avcorrtype>2):
                    # For initial setup, set starting simplex to ensure adequate parameter space covered
                    initialsimplex=np.array([[modelstartteff,ebv], [modelstartteff*1.1,ebv], [modelstartteff,ebv*2.]])
                    p=optimize.minimize(fit_teff_ebv,np.array([modelstartteff,ebv*1.]),args=(np.array(["simple",logg,feh,alphafe]),priors,params,values,sed,avdata,logg,feh,alphafe),method='Nelder-Mead',options ={'initial_simplex': initialsimplex}
)
                    teff,ebv=p.x
                    if (verbosity>95):
                        print ("Returned Teff,E(B-V):",teff,ebv)
                    sed=deredden2(sed,avdata,ebv,teff,logg,feh,alphafe)
                else:
                    print_fail ("Extinction fitting attempted for unsupported Av correction type.")
                    print ("Set FitEBV = 0 or ExtCorrDetail > 2")
                    exit()
            else: # Fit only temperature
                if (avcorrtype>2): # revise extinction as we go
                    teff=optimize.minimize(chisq_model_with_extinction,modelstartteff,args=(np.array(["simple",logg,feh,alphafe]),priors,params,values,sed,avdata,ebv,logg,feh,alphafe),method='Nelder-Mead',tol=10)['x'][0]
                    sed=deredden2(sed,avdata,ebv,teff,logg,feh,alphafe)
                else: # use extisting extinction
                    teff=optimize.minimize(chisq_model,modelstartteff,args=(freq,flux,ferr,np.array(["simple",logg,feh,alphafe]),priors,params,values),method='Nelder-Mead',tol=10)['x'][0]
            # Compute final model
            model,fratio,chisq=compute_model(teff,freq,flux,ferr,np.array(["simple",logg,feh,alphafe]),priors,params,values)

        # If no good model can be identified, use a blackbody instead
        if ((np.isnan(chisq)) | (chisq>=1.e99)):
            print_warn("Best-fitting model returned NaN - reverting to blackbody fit")
            sed,modwave,modflux,teff,rad,lum,chisq=sed_fit_bb(sed[sed['mask']>0],ancillary,avdata,ebv)
            foo=np.zeros_like(sed['model']) # use of intermediate foo solves weird bug in sed[sed['mask']>0]['model']=modflux
            foo[sed['mask']>0]=modflux
            sed['model']=foo
            return sed,wavel,modflux,teff,rad,lum,logg,feh,chisq,ebv
        else:
            if (verbosity>=80):
                print ("Beginning fit = Teff,chi^2_r:",teff,chisq)

        if (verbosity>=97):
            print (">>> 1")
            print ("List of",len(sed[sed['mask']>0]['svoname']),"filters accepted for fitting")
            print (sed[sed['mask']>0]['svoname'])
            print ("List of",len(sed[sed['mask']==0]['svoname']),"rejected filters:")
            print (sed[sed['mask']==0]['svoname'])

        # Repeat fit to test removing outliers
        if (verbosity>=50):
            print ("Outlier rejection: testing up to",maxoutliers,"outliers")
        seqoutliers=0
        outsed=np.copy(sed) # Temporary copy of SED, holding the (sequential) masked items under consideration
        testteff=teff
        testmodel=model
        testflux=flux
        for i in np.arange(maxoutliers):
            # Find the factor by which photometry is outlying
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
                outratio=np.where(testmodel>testflux,testmodel/testflux,testflux/testmodel)
            # If that exceeds the minimum tolerance...
            if (np.max(outratio)>outtol):
                # Remove the worst outlier from the SED and repeat the steps above to get the data necessary to fit
                sedidx=np.nonzero(outsed['mask'])[0][np.argmax(outratio)]
                if (verbosity>70):
                    print ("Outlier",i+1,":",outsed[sedidx]['catname'],outsed[sedidx]['filter'],", outlier by factor =",np.max(outratio))
                testsed=np.delete(np.copy(outsed),sedidx) # Another temporary copy, masking the current outlier
                # Rederive extinction
                if (avcorrtype>1):
                    testsed=deredden2(testsed,avdata,ebv,teff,logg,feh,alphafe)
                ntestable=((testsed[testsed['mask']>0]['wavel']>minlambda*10000.) & (testsed[testsed['mask']>0]['wavel']<maxlambda*10000.)).sum()
                if ((ntestable<minoutliers) or (len(testsed[testsed['mask']>0])<=2)):
                    if (verbosity>90):
                        print ("Too few points to try outlier rejection [",ntestable,"<",minoutliers," or ",len(testsed[testsed['mask']>0]),"< 2] -> breaking rejection loop")
                    break
                testwavel=testsed[testsed['mask']>0]['wavel']/1.e10
                testflux=testsed[testsed['mask']>0]['dered']
                testferr=testsed[testsed['mask']>0]['derederr']
                testferr=np.nan_to_num(testferr, nan=1.e6)
                testfreq=299792458./testwavel
                testvalues=np.array(modeldata[testsed[testsed['mask']>0]['svoname']].tolist())
                testvalues=testvalues[femask]
                # Refit the data
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore",message="invalid value encountered in subtract")

                    if (fitebv>0):
                        p=optimize.minimize(fit_teff_ebv,np.array([modelstartteff,ebv]),args=(np.array(["simple",logg,feh,alphafe]),priors,params,testvalues,testsed,avdata,logg,feh,alphafe),method='Nelder-Mead',tol=10)['x'][0]
                        teff,ebv=p.x
                        print ("Outlier",i,"returned:",teff,ebv)
                        sed=deredden2(sed,avdata,ebv,teff,logg,feh,alphafe)
                    elif (avcorrtype>2):
                        teff=optimize.minimize(chisq_model_with_extinction,modelstartteff,args=(np.array(["simple",logg,feh,alphafe]),priors,params,testvalues,testsed,avdata,ebv,logg,feh,alphafe),method='Nelder-Mead',tol=10)['x'][0]
                        sed=deredden2(sed,avdata,ebv,teff,logg,feh,alphafe)
                    else:
                        testteff=optimize.minimize(chisq_model,modelstartteff,args=(testfreq,testflux,testferr,np.array(["simple",logg,feh,alphafe]),priors,params,testvalues),method='Nelder-Mead',tol=0.25)['x'][0]

                    testmodel,testfratio,testchisq=compute_model(testteff,testfreq,testflux,testferr,np.array(["simple",logg,feh,alphafe]),priors,params,testvalues)
                    outsed[sedidx]['mask']=False
                # If the improvement in chi^2 is enough...
                if (chisq/testchisq>outchisqmin**(seqoutliers+1)):
                    if (verbosity>80):
                        print ("Triggered chi^2 reduction (total factor",chisq/testchisq,")")
                    seqoutliers=0
                    teff=testteff
#                    # Iterate log(g) if specified
#                    if (iteratelogg > 0):
#                        spint,foo,bar=compute_model(teff,testfreq,testflux,testferr,np.array(["simple",logg,feh,alphafe]),priors,params,np.expand_dims(modeldata['lum'][femask],axis=1))
#                        fratio=fratio/(6.96e8/dist/3.0857e16)**2
#                        rsun=np.sqrt(fratio)
#                        lum=rsun**2*(teff/5776.)**4
#                        if (usemsmass>0):
#                            mass=estimate_mass(teff,lum)
#                        logg=4.44+np.log10(mass/rsun**2)
#                        if (logg<np.min(modeldata['logg'])):
#                            logg=np.min(modeldata['logg'])+0.001
#                        if (logg>np.max(modeldata['logg'])):
#                            logg=np.max(modeldata['logg'])-0.001
#                        if (verbosity>80):
#                            print ("Revised log g =",logg,"[",np.min(modeldata['logg']),np.max(modeldata['logg']),"]")
                    model=testmodel
                    flux=testflux
                    ferr=testferr
                    chisq=testchisq
                    sed['dered']=outsed['dered']
                    sed['mask']=outsed['mask']
                else:
                    seqoutliers+=1
                    if (verbosity>=90):
                        print ("Chi^2 reduction not triggered (total factor",chisq/testchisq,")")
                    if (seqoutliers>=maxseqoutliers):
                        if (verbosity>=90):
                            print ("Too many sequential outliers without rejections -> breaking rejection loop")
                        break
            else:
                if (verbosity>90):
                    print ("No significant outliers -> breaking rejection loop")
                break
        if (verbosity>=97):
            print (">>> 2")
            print ("List of",len(sed[sed['mask']>0]['svoname']),"filters accepted for fitting")
            print (sed[sed['mask']>0]['svoname'])
            print ("List of",len(sed[sed['mask']==0]['svoname']),"rejected filters:")
            print (sed[sed['mask']==0]['svoname'])
                
        # Extract parameters from updated SED
        wavel=sed[sed['mask']>0]['wavel']/1.e10
        flux=sed[sed['mask']>0]['dered']
        ferr=sed[sed['mask']>0]['derederr']
        ferr=np.nan_to_num(ferr, nan=1.e6)
        freq=299792458./wavel

        # Perform more precise fit to include revised logg
        # Iterate log(g) if specified
        if ((iteratelogg > 0) & (chisq<1.e99)):
            spint,foo,bar=compute_model(teff,freq,flux,ferr,np.array(["simple",logg,feh,alphafe]),priors,params,np.expand_dims(modeldata['lum'][femask],axis=1))
            fratio=fratio/(6.96e8/dist/3.0857e16)**2
            rsun=np.sqrt(fratio)
            lum=rsun**2*(teff/5776.)**4
            if (usemsmass>0):
                mass=estimate_mass(teff,lum)
            logg=4.44+np.log10(mass/rsun**2)
            if (logg<np.min(modeldata['logg'])):
                logg=np.min(modeldata['logg'])+0.001
            if (logg>np.max(modeldata['logg'])):
                logg=np.max(modeldata['logg'])-0.001
            if (verbosity>80):
                print ("Revised log g =",logg,"[",np.min(modeldata['logg']),np.max(modeldata['logg']),"]")

        values=np.array(modeldata[sed[sed['mask']>0]['svoname']].tolist())
        values=values[femask]

        # Perform final refining fit
        if (fitebv>0):
            p=optimize.minimize(fit_teff_ebv,np.array([teff,ebv]),args=(np.array(["simple",logg,feh,alphafe]),priors,params,values,sed,avdata,logg,feh,alphafe),method='Nelder-Mead',tol=0.25)
            teff,ebv=p.x
            if (verbosity>95):
                print ("Returned Teff,E(B-V):",teff,ebv)
            sed=deredden2(sed,avdata,ebv,teff,logg,feh,alphafe)
            if (fitebvgrid>0):
                fit_teff_ebv_grid(teff,np.array(["simple",logg,feh,alphafe]),priors,params,values,sed,avdata,ebv,logg,feh,alphafe)
        elif (avcorrtype>2):
            teff=optimize.minimize(chisq_model_with_extinction,teff,args=(np.array(["simple",logg,feh,alphafe]),priors,params,values,sed,avdata,ebv,logg,feh,alphafe),method='Nelder-Mead',tol=0.25)['x'][0]
            sed=deredden2(sed,avdata,ebv,teff,logg,feh,alphafe)
        else:
            teff=optimize.minimize(chisq_model,teff,args=(freq,flux,ferr,np.array(["simple",logg,feh,alphafe]),priors,params,values),method='Nelder-Mead',tol=0.25)['x'][0]
     
        # Compute final model
        # Repeat computation to get specific intensity of interpolated model [Jy]
        # XXX Can't invoke because of duplicate names in unmasked SED data
        # XXX Would like option to predict other bands
        wavel=sed['wavel']/1.e10
        flux=sed['dered']
        ferr=sed['derederr']
        ferr=np.nan_to_num(ferr, nan=1.e6)
        freq=299792458./wavel
        if (len(np.unique(sed['svoname']))==len(sed['svoname'])):
            values=np.array(modeldata[sed['svoname']][femask].tolist())
        else:
            val=np.zeros((len(params[:,0]),len(sed)),dtype=float)
            for i in np.arange(len(sed['svoname'])):
                val[:,i]=modeldata[sed['svoname'][i]][femask].tolist()
#            values=val.tolist()
            values=val
        model,fratio,chisq=compute_model(teff,freq,flux,ferr,np.array(["simple",logg,feh,alphafe]),priors,params,values)
        
        spint,foo,bar=compute_model(teff,freq,flux,ferr,np.array(["simple",logg,feh,alphafe]),priors,params,np.expand_dims(modeldata['lum'][femask],axis=1))
        # Correct flux ratio to that expected for a 1 Rsun star, find radius
        fratio=fratio/(6.96e8/dist/3.0857e16)**2
        rsun=np.sqrt(fratio)
        # Calculate luminosity (solar lum = 3.828e26 W)
        lum=rsun**2*(teff/5776.)**4
        lum=spint/1e26*4.*np.pi*(6.96e8*rsun)**2/3.828e26
        # Revise mass if needed
        if (usemsmass>0):
            mass=estimate_mass(teff,lum)

        if (verbosity > 90):
            print ("Teff,R,L,chi^2_r:",teff,rsun,lum,chisq)
        if (verbosity > 60):
            print ("SED contains",len(sed),"points after outlier rejection")
        if (verbosity>50 or speedtest):
            try:
                now = datetime.now() # time object
                elapsed = float(now.strftime("%s.%f"))-float(start.strftime("%s.%f"))
            except:
                now = 0; elapsed = 0
            print ("Took",elapsed,"seconds to fit, [",start,"] [",now,"]")

    else: # If no distance

        wavel=sed[sed['mask']>0]['wavel']/1.e10
        model=np.zeros(len(sed[sed['mask']>0]))
        teff=0.
        rsun=0.
        lum=0.
        chisq=0.
        if (verbosity > 90):
            print_warn ("Failed due to invalid distance - ignoring object")
            print (ancillary)
    
    return sed,wavel,model,teff,rsun,lum,logg,feh,chisq,ebv

# -----------------------------------------------------------------------------
def deredden2(sed,avdata,ebv,teff,logg,feh,afe):
    # Deredden algorithm for tabulated Av data

    start=datetime.now()

    avs=[0.1,3.1,10,31] # to match those used in makemodel.py
    rv=3.1
    
    # Set global parameters to avoid unnecessary updating
    global deredden2_teff
    global deredden2_ebv
    deredden2_teff=teff
    deredden2_ebv=ebv
    
    modeldata=avdata[0]
    oparams=np.stack((modeldata['teff'],modeldata['logg'],modeldata['metal'],modeldata['alpha']),axis=1)
    try:
        # Values aren't actually used in this instance - only to get the selector of which grid points are needed
        ovalues=np.array(modeldata[sed[sed['mask']>0]['svoname']].tolist())
    except KeyError:
        print_fail ("KeyError when extracting extinction models!")
        print ("This can occur if additional filters have been introduced that have not been convolved with the models.")
        print ("Do the first lines of the extinction files [almabda-<name>-*-recast.dat] include all entries in the filter input file?")
        print ("If not, run 'python3 makemodel.py <name> [setup file]; python3 pyssed.py single '<any star name>' simple [setup file]; source shorten-model.scr'")
        raise

    foo,bar,selector=model_subset(oparams,ovalues,teff,logg,feh,afe)
    avsubset=avdata[:,selector]
    
    modeldata=avsubset[0]
    params=np.stack((modeldata['teff'],modeldata['logg'],modeldata['metal'],modeldata['alpha']),axis=1)
    interpav=np.empty((len(avsubset),len(sed[sed['mask']>0]['svoname'])),dtype=float)
    for i in np.arange(len(avsubset)):
        modeldata=avsubset[i]
        values=np.array(modeldata[sed[sed['mask']>0]['svoname']].tolist())
        interpav[i]=model_interp(params,values,teff,logg,feh,afe)

    #print (interpav)
    #print (sed[sed['mask']>0]['flux'])
        
    try:
        cs = interpolate.CubicSpline(avs,interpav)
        alambdabyav=cs(ebv)
        extmags=alambdabyav*ebv*rv
        extinction=10**(extmags/-2.5)
        dered=sed[sed['mask']>0]['flux']/extinction
        derederr=sed[sed['mask']>0]['ferr']/extinction
        np.put(sed[:]['dered'],(sed['mask']>0).nonzero(),dered)
        np.put(sed[:]['derederr'],(sed['mask']>0).nonzero(),derederr)
    except:
        if (verbosity>80):
            print_warn ("Assigning interpolated values failed (e.g., because outside range)")

    now=datetime.now()
    elapsed=((now-start).seconds+(now-start).microseconds/1000000.)
    if (verbosity>97):
        print ("Deredden2 [Teff,E(B-V)] =",teff,ebv,"(",elapsed,"seconds)")
    
    return sed

# -----------------------------------------------------------------------------
def model_subset(params,valueselector,teff="All",logg="All",feh="All",alphafe="All"):

    start=datetime.now()

    cutparams=params
    cutvalues=valueselector
    selector=np.ones(len(params),dtype=bool)
    if (teff!="All"):
        teffs=np.unique(params[:,0])
        tlo=teffs[np.searchsorted(teffs,teff,side='left')-1]
        if (tlo>=teffs[-1]):
            tlo=teffs[np.searchsorted(teffs,teff,side='left')-2]
            thi=teffs[np.searchsorted(teffs,teff,side='left')-1]
        else:
            thi=teffs[np.searchsorted(teffs,teff,side='left')]
        selector*=((params[:,0]==tlo) | (params[:,0]==thi))

    if (logg!="All"):
        loggs=np.unique(params[:,1])
        glo=loggs[np.searchsorted(loggs,logg,side='left')-1]
        if (glo>=loggs[-1]):
            glo=loggs[np.searchsorted(loggs,logg,side='left')-2]
            ghi=loggs[np.searchsorted(loggs,logg,side='left')-1]
        else:
            ghi=loggs[np.searchsorted(loggs,logg,side='left')]
        selector*=((params[:,1]==glo) | (params[:,1]==ghi))

    if (feh!="All"):
        fehs=np.unique(params[:,2])
        flo=fehs[np.searchsorted(fehs,feh,side='left')-1]
        if (flo>=fehs[-1]):
            flo=fehs[np.searchsorted(fehs,feh,side='left')-2]
            fhi=fehs[np.searchsorted(fehs,feh,side='left')-1]
        else:
            fhi=fehs[np.searchsorted(fehs,feh,side='left')]
        selector*=((params[:,2]==flo) | (params[:,2]==fhi))

    if (alphafe!="All"):
        afes=np.unique(params[:,3])
        alo=afes[np.searchsorted(afes,alphafe,side='left')-1]
        if (alo>=afes[-1]):
            alo=afes[np.searchsorted(afes,alphafe,side='left')-2]
            ahi=afes[np.searchsorted(afes,alphafe,side='left')-1]
        else:
            ahi=afes[np.searchsorted(afes,alphafe,side='left')]
        selector*=((params[:,3]==alo) | (params[:,3]==ahi))
    
    cutparams=cutparams[selector]
    cutvalues=cutvalues[selector]
    
    now=datetime.now()
    elapsed=((now-start).seconds+(now-start).microseconds/1000000.)
    if (verbosity>97):
        print ("Model subset:",np.shape(params),"->",np.shape(cutparams),";",np.shape(valueselector),"->",np.shape(cutvalues),";",np.sum(selector),"of",len(selector),"(",elapsed,"seconds)")

    return cutparams,cutvalues,selector

# -----------------------------------------------------------------------------
def model_interp(cutparams,cutvalues,teff,logg,feh,alphafe):
    # Interpolate between grid points

    interp_grid_points = np.array([float(teff),float(logg),float(feh),float(alphafe)])
    interp_data_points = np.zeros((len(cutvalues[0,:])))

    values=cutvalues[:,0]
    interpfn = interpolate.griddata(cutparams,cutvalues,interp_grid_points, method='linear', rescale=True)
    df = pd.DataFrame(interpfn)
    modeldata = df.interpolate().to_numpy()
    interp_data_points=np.atleast_1d(np.squeeze(modeldata))
    
    if (verbosity>97):
        print ("Model interpolation:",np.shape(interp_data_points),np.shape(interp_grid_points),np.shape(interp_data_points))

    return interp_data_points

# -----------------------------------------------------------------------------
def chisq_model(teff,freq,flux,ferr,modeltype,priors,params,values):
    # Wrapper to only return the chisq

    model,offset,chisq=compute_model(teff,freq,flux,ferr,modeltype,priors,params,values)

    return chisq

# -----------------------------------------------------------------------------
def chisq_model_with_extinction(teff,modeltype,priors,params,values,sed,avdata,ebv,logg,feh,afe):
    # As chisq model but includes extinction
    # Make use of global deredden2_teff to determine whether extinction correction is needed

    hitol=float(pyssedsetupdata[pyssedsetupdata[:,0]=="HiTDeredTol",1][0])
    lotol=float(pyssedsetupdata[pyssedsetupdata[:,0]=="LoTDeredTol",1][0])
    tolx=float(pyssedsetupdata[pyssedsetupdata[:,0]=="TDeredBound",1][0])
    
    if (teff>tolx):
        deredtol=hitol
    else:
        deredtol=lotol

    # Remove any bad values
    sed=sed[sed['flux']!=np.inf]

    # Only update dereddening if E(B-V) has changed or large Teff difference
    if (ebv==0.):
        sed['dered']=sed['flux']
    elif ((ebv!=deredden2_ebv) | (np.abs(teff-deredden2_teff)>deredtol/ebv)):
        sed=deredden2(sed,avdata,ebv,np.squeeze(teff),logg,feh,afe)
    elif (verbosity>97):
        print ("Not updating dereddening",teff,"~",deredden2_teff)
    sed['derederr']=sed['ferr']*sed['dered']/sed['flux']
    wavel=sed[sed['mask']>0]['wavel']/1.e10
    flux=sed[sed['mask']>0]['dered']
    ferr=sed[sed['mask']>0]['derederr']
    ferr=np.nan_to_num(ferr, nan=1.e6)
    freq=299792458./wavel
    model,offset,chisq=compute_model(teff,freq,flux,ferr,modeltype,priors,params,values)

    return chisq

# -----------------------------------------------------------------------------
def fit_teff_ebv(p,modeltype,priors,params,values,sed,avdata,logg,feh,afe):
    # As chisq_model_with_extinction but packaged for scipy.optimize.minimize
    
    teff=p[0]
    ebv=p[1]
    chisq=chisq_model_with_extinction(teff,modeltype,priors,params,values,sed,avdata,ebv,logg,feh,afe)

    if (verbosity>96):
        print ("Fitting Teff,E(B-V) at chisq:",teff,ebv,chisq)

    return chisq

# -----------------------------------------------------------------------------
def fit_teff_ebv_grid(tefffit,modeltype,priors,params,values,sed,avdata,ebvfit,logg,feh,alphafe):
    # Generates grid
    # NB: does not update log(g), [Fe/H] or [alpha/Fe]

    if (verbosity>80):
        print ("Generating Teff versus E(B-V) goodness-of-fit grid")

    outdir=str(pyssedsetupdata[pyssedsetupdata[:,0]=="TeffEBVGridDir",1])[2:-2]
    tefftype=str(pyssedsetupdata[pyssedsetupdata[:,0]=="TeffGridType",1])[2:-2]
    teffmin=float(pyssedsetupdata[pyssedsetupdata[:,0]=="TeffGridMin",1][0])
    teffmax=float(pyssedsetupdata[pyssedsetupdata[:,0]=="TeffGridMax",1][0])
    teffpts=int(pyssedsetupdata[pyssedsetupdata[:,0]=="TeffGridPoints",1][0])
    ebvtype=str(pyssedsetupdata[pyssedsetupdata[:,0]=="EBVGridType",1])[2:-2]
    ebvmin=float(pyssedsetupdata[pyssedsetupdata[:,0]=="EBVGridMin",1][0])
    ebvmax=float(pyssedsetupdata[pyssedsetupdata[:,0]=="EBVGridMax",1][0])
    ebvpts=int(pyssedsetupdata[pyssedsetupdata[:,0]=="EBVGridPoints",1][0])

    outfile=outdir+"grid.dat"
    with open(outfile, "w") as f:
        f.write("#Teff E(B-V) chi^2\n")
    chisqgrid=np.zeros((teffpts,ebvpts))
    if (tefftype=="logarithmic"):
        teffs=np.logspace(np.log10(teffmin),np.log10(teffmax),teffpts)
    else:
        teffs=np.linspace(teffmin,teffmax,teffpts)
    if (ebvtype=="logarithmic"):
        ebvs=np.logspace(np.log10(ebvmin),np.log10(ebvmax),ebvpts)
    else:
        ebvs=np.linspace(ebvmin,ebvmax,ebvpts)
    for t in np.arange(teffpts):
        if (verbosity>90):
            print ("Teff =",t)
        for e in np.arange(ebvpts):
            teff=teffs[t]
            ebv=ebvs[e]
            chisqgrid[t,e]=chisq_model_with_extinction(teff,modeltype,priors,params,values,sed,avdata,ebv,logg,feh,alphafe)
            if (verbosity>97):
                print ("Teff/E(B-V)/chi:",teff,ebv,chisqgrid[t,e])
            with open(outfile, "a") as f:
                f.write(str(teff)+" "+str(ebv)+" "+str(chisqgrid[t,e])+"\n")

    return

# -----------------------------------------------------------------------------
def compute_model(teff,freq,flux,ferr,modeltype,priors,params,valueselector):
    # Fit T_eff via a model for frequency in Hz
    # Fit in THz, rather than Hz, to avoid overflow
    # Returns flux from an emitter with area 1"^2 in Jy

    # Weight parameters if only fitting around BB peak
    weightfit=int(pyssedsetupdata[pyssedsetupdata[:,0]=="UseWeightedTFit",1][0])
    wtsigma=float(pyssedsetupdata[pyssedsetupdata[:,0]=="WeightedTSigma",1][0])
    wtpower=float(pyssedsetupdata[pyssedsetupdata[:,0]=="WeightedTPower",1][0])
    useerrors=int(pyssedsetupdata[pyssedsetupdata[:,0]=="WeightByErrors",1][0])
    wtoffset=float(pyssedsetupdata[pyssedsetupdata[:,0]=="WeightOffset",1][0])

    teff=np.squeeze(teff)
    if (teff<1.):
        teff=1.
    if (modeltype[0]=="bb"):
        # Blackbody
        # 2 * pi * 1"^2 / 206265^2 = 1 / 6771274157.32
        # 2*h/c^2 = 1.47449944e-50 kg s
        # h/k_B = 4.79924466e-11 K s
        # 1.47449944e-50 kg s / 6771274157.32 K s * 1e26 Jy/K = 2.1775805e-34 Jy kg
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="overflow encountered in exp")
            warnings.filterwarnings("ignore",message="divide by zero encountered in log10")
            model=np.log10(2.1775805e+02*(freq/1e12)**3/(np.exp(4.79924466e-11*freq/teff)-1))
            n=len(freq)
    elif (modeltype[0]=="simple"):
        logg=float(modeltype[1])
        feh=float(modeltype[2])
        alphafe=float(modeltype[3])
        
        if (verbosity>95):
            print ("Testing:",teff,logg,feh,alphafe)
        
        # Restrict grid of models to adjecent points for faster fitting
        # This is the reason for the earlier lengthy recasting process!
        cutparams,cutvalues,selector=model_subset(params,valueselector,teff,logg)
        
        if (verbosity>97):
            print ("Data length:",np.shape(flux))
            print ("Model list length:",np.shape(cutvalues))
            
        if (verbosity>=99):
            print (cutparams)
            print (cutvalues)

        interp_data_points=model_interp(cutparams,cutvalues,teff,logg,feh,alphafe)
        n=len(interp_data_points)
        if (verbosity>=99):
            print ("model",interp_data_points)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
            warnings.filterwarnings("ignore", message="invalid value encountered in log10")
            model=np.log10(interp_data_points)
        if (np.sum(interp_data_points)==0):
            return interp_data_points,1.,9e99
        #if (verbosity>=98):
        #    print (interp_data_points)
        #if (verbosity>=99):
        #    print (flux)
    else:
        print ("Model type not understood:",modeltype[0])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="invalid value encountered in add")
        warnings.filterwarnings("ignore",message="invalid value encountered in true_divide")
        warnings.filterwarnings("ignore",message="divide by zero encountered in log10")
        warnings.filterwarnings("ignore",message="invalid value encountered in log10")
        ferrratio=np.log10(1+ferr/flux)
        flux=np.log10(flux)
        offset=np.median(flux-model)
        model+=offset
    if (verbosity>=99):
        print ("log model",model)
        print ("log flux",flux)
        with np.printoptions(precision=3, suppress=True):
            print ("ratio",10**(flux-model))
    if ((n>1) & (teff>0)):
        # Assign weights
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="invalid value encountered in subtract")
            warnings.filterwarnings("ignore",message="invalid value encountered in scalar divide")
            warnings.filterwarnings("ignore",message="overflow encountered in square")
            wt=np.ones(len(flux))
            if (useerrors>0):
                # factor +1.e-30 to avoid divide by zero
                wt/=np.sqrt((ferrratio/(flux+1.e-30))**2+wtoffset**2)/wtoffset**2
            if (weightfit>0):
                bbpeak=2.897771955e-3/teff
                wavel=2.99792458e8/freq
                wt/=np.exp((np.log10(wavel)-np.log10(bbpeak))**2./wtsigma**2.)**wtpower
                #1/exp([log10(lambda)-log10(0.002898 m K/ T_eff)]^2/sigma^2)^power
                if (verbosity>=99):
                    print ("wt:",wt)
            wtsum=np.sum(wt)
            chisq=np.sum((flux-model)**2*wt)/((n-1)*wtsum)
            #chisq=np.sum((flux-model)**2)/(n-1) # Set unity weighting for all data
    else:
        chisq=9.e99

    #if (verbosity>=95):
    #    print ("     Chisq =            ",chisq)

    # Apply weighting penalty from temperature prior
    if (len(priors)>0):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="divide by zero encountered in double_scalars")
            warnings.filterwarnings("ignore",message="invalid value encountered in subtract")
            oldchisq=chisq
            # Upper temperature limit
            if (priors[0]['maxerr']>0.):
                prioruppersigma=(teff-priors[0]['max'])/priors[0]['maxerr']
                if (prioruppersigma>10.):
                    chisq/=(erf(prioruppersigma)+1.)/2.
            elif (priors[0]['max']>teff):
                chisq=1.e99
            # Lower temperature limit
            if (priors[0]['maxerr']>0.):
                priorlowersigma=(priors[0]['min']-teff)/priors[0]['minerr']
                if (priorlowersigma>10.):
                    chisq/=(erf(priorlowersigma)+1.)/2.
            elif (priors[0]['min']<teff):
                chisq=1.e99

    if (verbosity>=90):
        print ("Teff,chisq =",teff,chisq)
#        if (chisq==np.nan):
#            exit()

    # Return fitted model values if more than one value to fit. Otherwise, return interpolated value.
    if (n>1):
        return 10**model,10**(offset),chisq
    else:
        return np.squeeze(interp_data_points),10**(offset),chisq

# -----------------------------------------------------------------------------
def get_priors(ancillary):
    priors=np.zeros(4,dtype=[('param','<U20'),('min','f4'),('max','f4'),('minerr','f4'),('maxerr','f4')])
    priors[0]=("Temp",1.,1.e38,0.,0.)
    priors[1]=("logg",-10.,10.,0.,0.)
    priors[2]=("[Fe/H]",-10.,10.,0.,0.)
    priors[3]=("[alpha/Fe]",-10.,10.,0.,0.)

    usepriors=float(pyssedsetupdata[pyssedsetupdata[:,0]=="UseInformedPriors",1][0])
    if (usepriors>0):
        priortmax=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PriorTMax",1][0])
        priortmin=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PriorTMin",1][0])
        priortmaxerr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PriorTMaxSoft",1][0])
        priortminerr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PriorTMinSoft",1][0])
        priors[0]=("Temp",priortmin,priortmax,priortminerr,priortmaxerr)

    usetspec=float(pyssedsetupdata[pyssedsetupdata[:,0]=="UsePriorsOnTspec",1][0])
    if (usetspec>0):
        foo=ancillary[(ancillary['parameter']=="Tspec")]
        defaulttspecerr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultTspecError",1][0])
        mintspecerr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MinTspecError",1][0])
        if (len(foo)>0): # if spectroscopic temperature measurement
            if (len(foo)>1): # if more than one spectroscopic measurement
                tspec=foo[(foo['mask']==True)]['value']
                tspecerr=foo[(foo['mask']==True)]['err']
                tspecerr=np.where(tspecerr>0,tspecerr,defaulttspecerr) # assign default to nulls
                tspecerr=tspecerr[tspec>0]
                tspec=tspec[tspec>0]
                tspec=tspec[tspecerr>0]
                tspecerr=tspecerr[tspecerr>0]
                tspecwt=1./tspecerr**2
                tspec=np.sum(tspec*tspecwt)/np.sum(tspecwt)
                tspecerr=1./np.sum(tspecwt)
            else:
                tspec=float(reducto(foo[(foo['mask']==True)]['value']))
                tspecerr=float(reducto(foo[(foo['mask']==True)]['err']))
            if (tspecerr<=mintspecerr):
                tspecerr=mintspecerr
            priortmax=tspec
            priortmin=tspec
            priortmaxerr=tspecerr
            priortminerr=tspecerr
            priors[0]=("Temp",priortmin,priortmax,priortminerr,priortmaxerr)

    return priors

# -----------------------------------------------------------------------------
def sed_fit():
    # Fit T_eff, log(g), [Fe/H], E(B-V), R_v to a set of models
    # based on parallax quoted elsewhere
    # (Full parametric fit using priors and errors)
    # - Define a set of priors from known observations (e.g. Gaia parallax)
    # - Perform MCMC
    # - Return parameters, errors, goodness of fit and residual excess/deficit
    return
    
# -----------------------------------------------------------------------------
def sed_fit_binary():
    # As sed_fit_simple, but using two models
    # - Fit single models
    # - Fit binary model
    # - Returns parameters of both models, including which is a better fit
    return

# -----------------------------------------------------------------------------
def interp_model():
    # Generate a full, interpolated stellar atmosphere model at a desired resolution
    # For comparison against the full data
    return

# -----------------------------------------------------------------------------
def reduce_models():
    # Take a set of gridded model atmospheres
    # Generate a grid of data for them based on filter passbands
    return

# =============================================================================
# SED PLOTTING
# =============================================================================
def globalplots(compiledseds,compiledanc,sourcedata):
    # Make global plots for area searches before SED construction
    if (verbosity>=30):
        print ("Making global plots before SED construction...")

    # Getting general catalogue and filter data
    catdata=get_catalogue_list()
    filtdata=get_filter_list()
    rejectdata=get_reject_list()
    svodata=get_svo_data(filtdata)

    #np.set_printoptions(threshold=1000000,linewidth=300)

    # Location of photometry
    phottempdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotTempDir",1])[2:-2]

    plotdata=[] # Obtain universal data from source catalogues to be used in plots
                # These are not used in pre-production (raw source) plots, not SED-constructed plots
                # like the PM plot, which have their data passed via argument to the sub-routine
    cati=-1
    # For each catalogue...
    for cat in catdata['catname']:
        # Open the downloaded data and try extracting data (first RA & Dec)
        cati+=1
        try:
            data=np.load(phottempdir+cat+".npy")
        except FileNotFoundError:
            data=[]
        else:
            if (verbosity>=90):
                print ("Plotting",len(data),"points from",cat)

            ra,dec=extract_ra_dec(data)
            beamsize=ra*0.+float(catdata[catdata['catname']==cat]['beamsize'][0])

            # Loop over filters...
            for fdata in filtdata[filtdata['catname']==cat]:
                if (verbosity>=95):
                    print ("Filter:",fdata['filtname'])
                    
                # Extract magnitudes and fluxes
                svokey=fdata['svoname']
                wavel=ra*0.+float(svodata[svodata['svoname']==svokey]['weff'][0])
                dw=float(svodata[svodata['svoname']==svokey]['dw'][0])
                reject_reasons=rejectdata[(rejectdata['catname']==cat) & ((rejectdata['filtname']==fdata['filtname']) | (rejectdata['filtname']=="All")) & (rejectdata['rejcat']=="Same") & (rejectdata['rejcol']=="Same")]
                if (fdata['dataref']=='Vega'):
                    zpt=float(svodata[svodata['svoname']==svokey]['zpt'][0])
                else:
                    zpt=3631.
                mag,magerr,flux,ferr,mask=get_mag_flux(data,fdata,zpt,reject_reasons)
                if (verbosity>=95):
                    try:
                        print ("...",len(mag),"points recovered")
                    except:
                        print ("...no points recovered!")

                pdata=np.column_stack((ra,dec,wavel,flux,beamsize,ra*0.+cati))
                #print (flux)
                
                if (len(plotdata)==0):
                    plotdata=pdata
                else:
                    try:
                        plotdata=np.append(plotdata,pdata,axis=0)
                    except ValueError:
                        print_fail("Issue with concatenating points")
                        print (np.shape(plotdata))
                        print (np.shape(pdata))
                        raise

            # ...on second thoughts, just take the first one, otherwise it messes up the nearest neighbour matching
                break

    # Set colourmap and alpha - safer option for matplotlib < version 3.4
    colours = np.zeros((len(plotdata),4))
    plotdata[:,2]/=1e5 # convert wavelength from Angstroms to microns
    # Alpha
    # Clip outliers (<1 nJy um^2, >10000 Jy um^2) and take log
    #plotdata[:,3]*=(plotdata[:,2])**2 # Rayleigh-Jeans tail [1e5 -> microns, avoids limits]
    maxflux=3631.e0 # Jy [opaque]
    minflux=1.e-9 # Jy [transparent]
    plotdata[:,3]=np.nan_to_num(plotdata[:,3])
    plotdata[:,3]=np.where(plotdata[:,3]<minflux,minflux,plotdata[:,3])
    plotdata[:,3]=np.where(plotdata[:,3]>maxflux,maxflux,plotdata[:,3])
    plotdata[:,3]=np.log10(plotdata[:,3])
    colours[:,3]=(plotdata[:,3]-np.min(plotdata[:,3]))/(np.max(plotdata[:,3])-np.min(plotdata[:,3])) # rescale to 0-1
    colours[:,3]=(colours[:,3])**2 # amplify scale for better human response
    colours[:,3]*=0.1+0.9*(1.-plotdata[:,4]/np.max(plotdata[:,4])) # scale opacity with point size
    # Colours
    plotdata[:,2]=np.where(plotdata[:,2]>10.,10.,plotdata[:,2]) # show everything >10 um as red
    logwavel=np.log10(plotdata[:,2])
    pos=(logwavel-np.min(logwavel))/(np.max(logwavel)-np.min(logwavel))
    colours[:,2]=np.where(pos<0.5,1.-2.*pos,0)
    colours[:,1]=np.where(pos<0.5,2.*pos,1.-2.*pos)
    colours[:,0]=np.where(pos<0.5,0,(pos-0.5)*2.)
    # Error fix
    colours=np.nan_to_num(colours)
    colours=np.where(colours>1,1,colours)
    colours=np.where(colours<0,0,colours)

    # Locations of all sources on the sky
    makepointplot=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MakePointPlot",1][0])
    if (makepointplot>0):
        pointplotfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PointPlotFile",1])[2:-2]
        globalpointsplot(pointplotfile,plotdata[:,[0,1,4]],colours)

    # --------------------------
    # Matching radius versus wavelength
    makeclosestplot=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MakeClosestPlot",1][0])
    makeoffsetplot=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MakeOffsetPlot",1][0])
    if ((makeclosestplot>0) | (makeoffsetplot>0)):
        # Common aspects to nearest-neighbour distance and offset plot
        closestplotfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="ClosestPlotFile",1])[2:-2]
        offsetplotfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="OffsetPlotFile",1])[2:-2]
        # Find nearest neighbour
        tree=cKDTree(plotdata[:,0:2]*3600.)
        dists,nn=tree.query(plotdata[:,0:2]*3600.,k=2)
        dists=dists[:,1] # Only get second match, since first is original
        nn=nn[:,1]
        nndata=np.column_stack((dists,plotdata[nn,5],plotdata[nn,4]))
    if (makeclosestplot>0):
        globalclosestplot(closestplotfile,nndata,colours[nn,:],catdata['catname'])
    if (makeoffsetplot>0):
        ra=plotdata[:,0]
        dec=plotdata[:,1]
        offra=(ra-ra[nn])*3600./np.cos(ra)
        offdec=(dec-dec[nn])*3600.
        offsetdata=np.column_stack((offra,offdec,plotdata[nn,4],plotdata[nn,5]))
        globaloffsetplot(offsetplotfile,offsetdata,colours[nn,:])

    # --------------------------

    # Proper motion vectors
    makepmplot=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MakePMPlot",1][0])
    if (makepmplot>0):
        pmplotfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PMPlotFile",1])[2:-2]        
        globalpmplot(pmplotfile,compiledanc,colours)

    return

# -----------------------------------------------------------------------------
def globalpointsplot(plotfile,plotdata,colours):
    # Plot all the downloaded photometric points in the same file
    # plotdata = 2D array of x,y,size
    # colours = 0<(R,G,B,A)<1 array of same length as plotdata
    if (verbosity>=50):
        print ("Plot raw photometry points ->",plotfile)

    # Set up main axes
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.gca().invert_xaxis()
    plt.minorticks_on()
    
    # Plot the RA and Dec
    plt.scatter(plotdata[:,0],plotdata[:,1],s=(plotdata[:,2])/np.log10(len(plotdata[:,2]))+1,color=colours,edgecolors='none',zorder=20)

    # Save the file
    plt.savefig(plotfile,dpi=300)
    plt.close("all")

    return

# -----------------------------------------------------------------------------
def globalclosestplot(plotfile,plotdata,colours,labels):
    # Plot the closest match radii for sources
    # plotdata = 2D array of x,y,size
    # colours = 0<(R,G,B,A)<1 array of same length as plotdata
    if (verbosity>=50):
        print ("Plot matching radius data ->",plotfile)

    # Set up main axes
    plt.xlabel("Matching radius (arcsec)")
    plt.ylabel("Catalogue")
    plt.gca().set_xscale('log')
    #plt.gca().set_yscale('log')
    plt.gca().set_yticks(np.arange(len(labels)))
    plt.gca().set_yticklabels(labels)
    plt.gca().xaxis.grid()
    
    # Spread out data on y axis a bit
    y=plotdata[:,1]+((np.random.random(len(plotdata))-0.5)/1.5) # spread results out a bit to increase visibility
    
    # Calculate medians
    medians=np.zeros(len(labels))
    for i in np.arange(len(labels)):
        if (len(plotdata[plotdata[:,1]==i,0])>0):
            medians[i]=np.median(plotdata[plotdata[:,1]==i,0])

    # Get limits
    catdata=get_catalogue_list()
    
    # Set x-axis ranges
    rnghi=np.max(np.concatenate([plotdata[:,0],medians,catdata['matchr'],[1]]))
    rnglo=np.min(np.concatenate([plotdata[:,0],medians,catdata['matchr'],[0.1]]))
    if (rnglo<0.0001):
        rnglo=0.0001
    plt.xlim([rnglo,rnghi])
    
    # Plot the data
    if (len(y)>10000.):
        plt.scatter(plotdata[:,0],y,s=(plotdata[:,2]+1)/3.,color=colours,edgecolors='none',zorder=20) # small points
    else:
        plt.scatter(plotdata[:,0],y,s=plotdata[:,2]+1,color=colours,edgecolors='none',zorder=20) # big points
    plt.scatter(medians,np.arange(len(labels)),s=24,color="#00000070",linewidths=1.,marker='+',zorder=30)
    plt.scatter(catdata['matchr'],np.arange(len(labels)),s=24,color="#FF0000FF",linewidths=1.,marker='x',zorder=30)

    # Save the file
    plt.savefig(plotfile,dpi=300)
    plt.close("all")

    return

# -----------------------------------------------------------------------------
def globaloffsetplot(plotfile,plotdata,colours):
    # Plot the offset of astrometric matches
    # plotdata = 2D array of x,y,size,cat#
    # colours = 0<(R,G,B,A)<1 array of same length as plotdata
    if (verbosity>=50):
        print ("Plot offset data ->",plotfile)

    # Set up main axes
    plt.xlabel('Offset RA (")')
    plt.ylabel('Offset Dec (")')
    plt.gca().invert_xaxis()
    plt.minorticks_on()

    # Calculate medians
    cats=np.unique(plotdata[:,2])
    medians=np.zeros((len(cats),2))
    for i in np.arange(len(cats)):
        if (len(plotdata[plotdata[:,2]==i,0])>0):
            medians[i,0]=np.median(plotdata[plotdata[:,2]==i,0])
            medians[i,1]=np.median(plotdata[plotdata[:,2]==i,1])
    rng=np.max(np.abs(medians[:,0],medians[:,1]))*3.
    if (rng<0.2):   # minimum range of +/-0.2"
        rng=0.2
    plt.xlim([-rng,rng])
    plt.ylim([-rng,rng])
    
    # Plot the RA and Dec
    plt.scatter(plotdata[:,0],plotdata[:,1],s=plotdata[:,2]+1,color=colours,edgecolors='none',zorder=20)
    plt.scatter(medians[:,0],medians[:,1],s=24,color="#00000070",linewidths=1.,marker='+',zorder=30)

    # Save the file
    plt.savefig(plotfile,dpi=300)
    plt.close("all")

    return

# -----------------------------------------------------------------------------
def globalpmplot(plotfile,compiled_anc,colours):
    # Plot the closest match radii for sources
    # compiled_anc = ancillary dataset
    # colours = 0<(R,G,B,A)<1 array of same length as plotdata
    if (verbosity>=50):
        print ("Plot proper motion data ->",plotfile)

    # Establish percentile level at which to clip proper motions
    clipperc=int(pyssedsetupdata[pyssedsetupdata[:,0]=="PMPlotClipping",1][0])

    # Extract PM and distance data
    pmra=[a[2][3] for a in compiled_anc]
    pmdec=[a[3][3] for a in compiled_anc]
    dist=[a[4][3] for a in compiled_anc]

    # Set up main axes
    plt.xlabel('RA proper motion (mas/yr)')
    plt.ylabel('Dec proper motion (mas/yr)')
    plt.minorticks_on()

    # Calculate range
    rng=np.max([np.percentile(np.abs(pmra),clipperc),np.percentile(np.abs(pmdec),clipperc)])*1.01
    plt.xlim([-rng,rng])
    plt.ylim([-rng,rng])

    # Import colourmap
    newcmp=colourmap(1.)

    # Plot the RA and Dec
    plt.scatter(pmra,pmdec,s=1,c=np.argsort(dist),cmap=newcmp,edgecolors='none',zorder=20)

    # Save the file
    plt.savefig(plotfile,dpi=300)
    plt.close("all")

    return

# -----------------------------------------------------------------------------
def globalmatchskyplot(plotfile,photdata,nmatch,cats):
    # Plot location of matched sources
    if (verbosity>=50):
        print ("Plot matching sources on sky ->",plotfile)
    
    # Set up grid
    #cats=np.unique(photdata[:,0])
    fig, axs = plt.subplots(len(cats)+1, len(cats)+1)

    # Calculate square extent to keep same plot area
    minra=np.min(photdata[:,1])
    maxra=np.max(photdata[:,1])
    mindec=np.min(photdata[:,2])
    maxdec=np.max(photdata[:,2])
    if ((maxdec-mindec) < (maxra-minra)):
        extent=maxra-minra
        mindec=(maxdec+mindec)/2.-extent/2.
        maxdec=(maxdec+mindec)/2.+extent/2.
    else:
        extent=maxdec-mindec
        minra=(maxra+minra)/2.-extent/2.
        maxra=(maxra+minra)/2.+extent/2.

    maxblends=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxBlends",1][0])
    for ax1 in np.arange(len(cats)):
        for ax2 in np.arange(len(cats)):
            if (ax1!=ax2):
                catmask=(photdata[:,0]==cats[ax1])
                matchmask=((photdata[:,0]==cats[ax1]) & (nmatch[:,ax2]==1))
                blendmask=((photdata[:,0]==cats[ax1]) & (nmatch[:,ax2]>1))
                helpmask=((photdata[:,0]==cats[ax1]) & (nmatch[:,ax2]>maxblends))
                catmask=np.where(matchmask,False,catmask) # remove matched items to only leave unmatched
                blendmask=np.where(helpmask,False,blendmask) # remove heavily blended objects to only leave simple blends
                if (verbosity>95):
                    print ("   Plotting:",ax1,ax2,cats[ax1],cats[ax2],np.sum(catmask),np.sum(matchmask),np.sum(blendmask),np.sum(helpmask))
                axs[ax1,ax2].axis(xmin=minra,xmax=maxra)
                axs[ax1,ax2].axis(ymin=mindec,ymax=maxdec)
                if (np.sum(catmask)<10):
                    psize=10
                    opacity="7F"
                elif (np.sum(catmask)<100):
                    psize=3
                    opacity="5F"
                elif (np.sum(catmask)<1000):
                    psize=1
                    opacity="3F"
                else:
                    psize=0.3
                    opacity="1F"
                axs[ax1,ax2].scatter(photdata[catmask,1],photdata[catmask,2],s=psize,color="#0000FF"+opacity,edgecolors='none',zorder=20)
                axs[ax1,ax2].scatter(photdata[matchmask,1],photdata[matchmask,2],s=psize,color="#00FF00"+opacity,edgecolors='none',zorder=20)
                axs[ax1,ax2].scatter(photdata[blendmask,1],photdata[blendmask,2],s=psize,color="#FF7700"+opacity,edgecolors='none',zorder=20)
                axs[ax1,ax2].scatter(photdata[helpmask,1],photdata[helpmask,2],s=psize,color="#FF0000"+opacity,edgecolors='none',zorder=20)

    if (verbosity>80):
        print ("   Plotting labels...")

    # Trim axes and bounds
    for ax in axs.flat:
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set(xlim=(minra,maxra), ylim=(mindec,maxdec), xticks=[], yticks=[], aspect=1)
    fig.tight_layout(pad=0.02)
    
    # Right-hand labels
    ax2=len(cats)
    for ax1 in np.arange(len(cats)):
        axs[ax1,ax2].set(xlim=(0,1), ylim=(0,1), xticks=[], yticks=[], aspect=1)
        axs[ax1,ax2].text(0.0,0.5, cats[ax1], ha='left', va='center')
        axs[ax1,ax2].axis("off") # remove frame
    # Bottom labels
    ax1=len(cats)
    for ax2 in np.arange(len(cats)):
        axs[ax1,ax2].set(xlim=(0,1), ylim=(0,1), xticks=[], yticks=[], aspect=1)
        axs[ax1,ax2].text(0.5,1.0, cats[ax2], ha='center', va='top')
        axs[ax1,ax2].axis("off") # remove frame
    # Diagonal labels
    for ax1 in np.arange(len(cats)):
        axs[ax1,ax1].set(xlim=(0,1), ylim=(0,1), xticks=[], yticks=[], aspect=1)
        axs[ax1,ax1].text(0.5,0.5, cats[ax1], ha='center', va='center')
        axs[ax1,ax1].axis("off") # remove frame
    # Corner labels
    axs[len(cats),len(cats)].axis("off") # remove frame
    axs[len(cats),len(cats)].text(0.0,0.8, "All", ha='left', va='center', color="#0000FF")
    axs[len(cats),len(cats)].text(0.0,0.6, "1:1 match", ha='left', va='center', color="#00FF00")
    axs[len(cats),len(cats)].text(0.0,0.4, "Blends", ha='left', va='center', color="#FF7700")
    axs[len(cats),len(cats)].text(0.0,0.2, "Overblend", ha='left', va='center', color="#FF0000")
    
    # Save the file
    if (verbosity>80):
        print ("   Saving final file...")
    fig.set_size_inches(len(cats)+1, len(cats)+1)
    plt.savefig(plotfile,dpi=600)
    plt.close("all")

    return

# -----------------------------------------------------------------------------
def colourmap(opacity):
    # Set up colour map
    n = 128
    vals = np.ones((n*2, 4))
    vals[:128, 0] = 0
    vals[:128, 1] = np.linspace(0/256, 256/256, n)
    vals[:128, 2] = np.linspace(256/256, 0/256, n)
    vals[128:, 0] = np.linspace(0/256, 256/256, n)
    vals[128:, 1] = np.linspace(256/256, 0/256, n)
    vals[128:, 2] = 0.
    vals[:, 3] = opacity
    newcmp = ListedColormap(vals)

    return newcmp

# -----------------------------------------------------------------------------
def heatmap():
    # Set up colour map
    n = 64
    vals = np.ones((n*4, 4))
    vals[:, 3] = 0.25
    vals[:n, 0] = np.linspace(128/256, 256/256, n)
    vals[:n, 1] = 0.
    vals[:n, 2] = np.linspace(128/256, 0/256, n)
    vals[n:n*2, 0] = np.linspace(256/256, 224/256, n)
    vals[n:n*2, 1] = np.linspace(0/256, 192/256, n)
    vals[n:n*2, 2] = 0.
    vals[n*2:n*3, 0] = np.linspace(256/256, 192/256, n)
    vals[n*2:n*3, 1] = np.linspace(256/256, 192/256, n)
    vals[n*2:n*3, 2] = np.linspace(0/256, 192/256, n)
    vals[n*3:n*4, 0] = np.linspace(192/256, 256/256, n)
    vals[n*3:n*4, 1] = np.linspace(192/256, 256/256, n)
    vals[n*3:n*4, 2] = np.linspace(0/256, 256/256, n)
    newcmp = ListedColormap(vals)

    return newcmp

# -----------------------------------------------------------------------------
def redbluemap():
    # Set up colour map
    n = 64
    vals = np.ones((n*2, 4))
    vals[:n, 0] = np.linspace(0/256, 192/256, n)
    vals[:n, 1] = np.linspace(0/256, 192/256, n)
    vals[:n, 2] = np.linspace(256/256, 192/256, n)
    vals[:n, 3] = np.linspace(256/256, 64/256, n)
    vals[n:n*2, 0] = np.linspace(192/256, 256/256, n)
    vals[n:n*2, 1] = np.linspace(192/256, 0/256, n)
    vals[n:n*2, 2] = np.linspace(192/256, 0/256, n)
    vals[n:n*2, 3] = np.linspace(64/256, 256/256, n)
    newcmp = ListedColormap(vals)

    return newcmp

# -----------------------------------------------------------------------------
def plotsed(sed,modwave,modflux,plotfile,image):
    # Plot the completed individual SEDs to a file
    showinset=int(pyssedsetupdata[pyssedsetupdata[:,0]=="ShowAstrometryInset",1][0])
    fixedminlambda=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PlotLambdaMin",1][0])
    fixedmaxlambda=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PlotLambdaMax",1][0])
    fixedminflux=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PlotFluxMin",1][0])
    fixedmaxflux=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PlotFluxMax",1][0])

    line = {}
    points = {}
    points['catname'] = sed['catname']
    points['svoname'] = sed['svoname']
    points['filter'] = sed['filter']
    fig, ax = plt.subplots(figsize=[5, 4])

    newcmp=colourmap(1.)

    # Set up main axes
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("Flux (Jy)")
    plt.minorticks_on()

    # Plot the model if it exists
    if (len(modwave)>0):
        x=modwave*1.e6
        y=modflux
        indx = x.argsort()
        xs = x[indx]
        ys = y[indx]
        xerr=0
        yerr=0
        if (image == True):
            plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='s',markersize=3,color='#AA33AA80',ecolor='lightgray', elinewidth=1, capsize=0, zorder=5)
            plt.plot(xs,ys,color='#FF00FF20',zorder=0,linewidth=3)

    
    # Plot the observed (reddened data)
    # Plot all the data
    x=sed[sed['mag']!=0]['wavel']/10000
    y=sed[sed['mag']!=0]['flux']
    xerr=0
    yerr=sed[sed['mag']!=0]['ferr']
    yerr=np.abs(yerr)
    if (image == True):
        plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='+',markersize=4,color='lightgray',ecolor='lightgray', elinewidth=1, capsize=0, zorder=10)
        
    # Include non-logarithmic values for Visualiser returns
    if (len(modwave)>0):
        line['model_x'] = xs
        line['model_y'] = ys
        points['wavel'] = sed['wavel']/10000.
        points['flux'] = sed['flux']
        points['ferr'] = sed['ferr']
        points['mag'] = sed['mag']
        points['magerr'] = sed['magerr']
        points['dered'] = sed['dered']
        points['derederr'] = sed['derederr']
        points['dw'] = sed['dw']/2./10000.
        points['mask'] = sed['mask']

    # Clip the plot if data too faint
    maxrange=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxSEDPlotFluxRange",1][0])
    maxflux=np.max(y[y!=np.inf])
    minflux=np.min(y[y>0])
    if ((image==True) & (minflux*maxrange<maxflux)):
        plt.ylim(ymin=maxflux/maxrange)
    # Set plot limits if requested
    if ((fixedminlambda>0.) | (fixedmaxlambda>0.)):
        plt.xlim([fixedminlambda,fixedmaxlambda])
    if ((fixedminflux>0.) | (fixedmaxflux>0.)):
        plt.ylim([fixedminflux,fixedmaxflux])

    # Overplot the unmasked in grey-red
    x=sed[sed['mask']>0]['wavel']/10000
    y=sed[sed['mask']>0]['flux']
    xerr=0
    yerr=0
    if (image==True):
        plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='+',markersize=4,color='#FFAAAA00',ecolor='#FFAAAA00', elinewidth=1, capsize=0, zorder=10)

    # Decide whether to put the inset on top or bottom,
    # based on whether the left or right quarter of the model points are higher
    lhs=np.average(np.log10(y[:int(len(y)/4)]))
    rhs=np.average(np.log10(y[-int(len(y)/4):]))

    # Plot lines up to the new data
    x=sed['wavel']/10000
    y=sed['flux']
    dy=sed['dered']
    for i in np.arange(len(x)):
        plt.arrow(x[i],y[i],0,dy[i]-y[i],color='#00000010', linewidth=1,length_includes_head=True,zorder=11)
    x=sed[sed['mask']>0]['wavel']/10000
    y=sed[sed['mask']>0]['flux']
    dy=sed[sed['mask']>0]['dered']
    if (image==True):
        for i in np.arange(len(x)):
            plt.arrow(x[i],y[i],0,dy[i]-y[i],color='#FF000030', linewidth=1,length_includes_head=True,zorder=11)
    
    # Plot the dereddened data
    # Plot all the data
    x=sed[sed['mag']!=0]['wavel']/10000
    y=sed[sed['mag']!=0]['dered']
    xerr=sed[sed['mag']!=0]['dw']/2/10000
    yerr=sed[sed['mag']!=0]['derederr']
    yerr=np.abs(yerr)
    if (image==True):
        plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='o',markersize=4,color='lightgray',ecolor='lightgray', elinewidth=1, capsize=0, zorder=10)

    # Overplot the unmasked in colour
    x=sed[sed['mask']>0]['wavel']/10000
    y=sed[sed['mask']>0]['dered']
    xerr=sed[sed['mask']>0]['dw']/2/10000
    yerr=sed[sed['mask']>0]['derederr']
    yerr=np.abs(yerr)
    colour=np.log10(x)
    if (image==True):
        plt.scatter(x,y,c=colour,cmap=newcmp,s=16,zorder=20)

    # Include co-ordinates inset
    if (showinset>0):
        # Astrometric data
        x=sed['ra']
        y=sed['dec']
        xerr=0.001
        yerr=0.001
        colour=np.log10(sed['wavel'])
        # Model data
        mx=sed['modelra']
        my=sed['modeldec']
        mxerr=0.001
        myerr=0.001
        # Set location of inset
        if (lhs>rhs):
            axins = ax.inset_axes([0.70, 0.70, 0.25, 0.25])
        else:
            axins = ax.inset_axes([0.70, 0.15, 0.25, 0.25])
        axins.set_xlabel(r'$\Delta\alpha^{\prime\prime}$')
        axins.set_ylabel(r'$\Delta\delta^{\prime\prime}$')
        axins.set_xticks(np.arange(int(min(x)), int(max(x)*+1), 1.),minor=True)
        axins.set_yticks(np.arange(int(min(x)), int(max(x)*+1), 1.),minor=True)
        axins.invert_xaxis() # Reverse RA axis
        axins.plot(mx,my,color='#7F007F3F',linewidth=5,zorder=1) # PM line    
        for i in np.arange(len(x)):
            axins.arrow(x[i],y[i],mx[i]-x[i],my[i]-y[i],color='#FF00FF20', linewidth=3,length_includes_head=True,zorder=11) # Arrows to PM line
        axins.errorbar(mx,my,xerr=mxerr,yerr=myerr,fmt='s',markersize=2,color='#FF7FFF',ecolor='lightgray', elinewidth=1, capsize=0,zorder=21) # Model
        axins.scatter(x,y,c=colour,cmap=newcmp,s=16,zorder=31) # Astrometry
        
    # Visualiser additions
    inset_points={}
    inset_line={}
    if (showinset>0):
        inset_points['catname'] = sed['catname']
        inset_points['svoname'] = sed['svoname']
        inset_points['filter'] = sed['filter']
        inset_points['ra'] = sed['ra']
        inset_points['dec'] = sed['dec']
        inset_line['modelra'] = mx
        inset_line['modeldec'] = my

    # Save the file
    plt.tight_layout()
    if (image==True):
        plt.savefig(plotfile,dpi=150)
        plt.close("all")
    
    # Additions for Visualiser output
    plot = pd.concat([pd.DataFrame(data=points), pd.DataFrame(data=line)], axis=1)
    #plot = plot[['catname','svoname','filter','mask','wavel','flux','ferr','mag','magerr','dered','derederr','model_x','model_y','dw']]
    inset_plot = pd.concat([pd.DataFrame(data=inset_points), pd.DataFrame(data=inset_line)], axis=1)
    
    return plot,inset_plot
    
# -----------------------------------------------------------------------------
def globalpostplots(compiledseds,compiledanc,sourcedata):

    if (verbosity>=60):
        print ("Extracting parameters for final plots")

    # Load in luminosity, temperature and observed-to-model flux excesses
    teff=np.array([(a[a['parameter'].astype(str)=="Teff"]['value'].astype(float)).flatten() for a in compiledanc]).flatten()
    lum=np.array([(a[a['parameter'].astype(str)=="Luminosity"]['value'].astype(float)).flatten() for a in compiledanc]).flatten()
    dist=np.array([(a[a['parameter'].astype(str)=="Dist"]['value'].astype(float)).flatten() for a in compiledanc]).flatten()

    teff0=teff[teff>0]
    lum0=lum[teff>0]
    dist0=dist[teff>0]
    
    colours = np.zeros((len(sourcedata),4))

    if (len(teff0)>0):

        if (verbosity>=50):
            print ("Creating H-R diagram")

        # Set up main axes
        plt.xlabel('Effective temperature (K)')
        plt.ylabel('Luminosity (L_Sun)')
        plt.minorticks_on()

        # Calculate range
        hrdxbuffer=1.01 # padding to avoid points on edge of plot
        hrdybuffer=1.10 # 1 = no padding
        minlum=float(pyssedsetupdata[pyssedsetupdata[:,0]=="HRDMinLum",1][0])
        maxlum=float(pyssedsetupdata[pyssedsetupdata[:,0]=="HRDMaxLum",1][0])
        minteff=float(pyssedsetupdata[pyssedsetupdata[:,0]=="HRDMinTemp",1][0])
        maxteff=float(pyssedsetupdata[pyssedsetupdata[:,0]=="HRDMaxTemp",1][0])
        plt.xlim([np.min(teff0[teff0>minteff])/hrdxbuffer,np.max(teff0[teff0<maxteff])*hrdxbuffer])
        plt.ylim([np.min(lum0[lum0>minlum])/hrdybuffer,np.max(lum0[lum0<maxlum])*hrdybuffer])
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%i'))
        plt.gca().set_yscale('log')
        plt.gca().set_xscale('log')
        plt.gca().invert_xaxis()

        # Import colourmap

        # Plot the Hertzsprung-Russell diagram
        ptsize=np.maximum(25./np.sqrt(len(teff0)),1)
        opacity=np.minimum((float(len(teff0))/100.)**(-1./3.),1)
        newcmp=colourmap(opacity) # set colour map with opacity scaled to data volume
        try:
            plt.scatter(teff0,lum0,s=ptsize,c=np.argsort(dist0),cmap=newcmp,edgecolors='none',zorder=20)
        except ValueError:
            print_warn ("ValueError when plotting H-R diagram. Maybe Teff, lum or dist is a flat array?")
            plt.scatter(teff0,lum0,s=ptsize,edgecolors='none',zorder=20)

        if (verbosity>=50):
            print ("Creating excess diagram")

        # Save the file
        plotfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="HRDPlotFile",1])[2:-2]        
        plt.savefig(plotfile,dpi=300)
        plt.close("all")

        # Plot the flux-excess diagram
        makexsplots=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MakeXSPlots",1][0])
        if (makexsplots>0):
            lumxsplotfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="LumXSPlotFile",1])[2:-2]        
            xsspaceplotfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="XSSpacePlotFile",1])[2:-2]        
            xscorrnplotfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="XSCorrnPlotFile",1])[2:-2]        
            excesslumplot(lumxsplotfile,compiledseds[teff>0],compiledanc[teff>0],teff0)
            excessspaceplot(xsspaceplotfile,compiledseds[teff>0],compiledanc[teff>0],teff0)
            excesscorrnplot(xscorrnplotfile,compiledseds[teff>0],compiledanc[teff>0],teff0)
    
    return

# -----------------------------------------------------------------------------
def setupfiltergrid(compiledseds):

    if (verbosity>=80):
        print ("Identifying used filters")
 
   # Get catalogue data
    filtdata=get_filter_list()
    filts=[]
    for i in np.arange(len(compiledseds)):
        if (isinstance(compiledseds[i],int)==False):
            filts=np.unique(np.append(filts,compiledseds[i]['svoname'],axis=0))

    if (verbosity>=90):
        print (filts)

    if (verbosity>=80):
        print ("Creating frames")

    # Make subframes for each catalogue
    yf=np.ceil(np.sqrt(len(filts)+1)).astype(int)
    xf=np.ceil((len(filts)+1)/yf).astype(int)
    fig, axs = plt.subplots(xf,yf,sharex='col',sharey='row')
    fig.tight_layout(pad=0.00)

    return fig,axs,xf,yf,filts

# -----------------------------------------------------------------------------
def excesslumplot(plotfile,compiledseds,compiledanc,teff):

    # Plot observed-to-model flux excesses
    fig,axs,xf,yf,filts=setupfiltergrid(compiledseds)
    fig.text(0.5, 0.00, 'Dereddened/Modelled flux', ha='center', fontsize=5)
    fig.text(0.00, 0.5, 'Flux (log[Jy])', va='center', rotation='vertical', fontsize=5)

    xr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="LumXSPlotXSRange",1][0])
    for i in np.arange(xf):
        for j in np.arange(yf):
            axs[i,j].axis(xmin=1/xr,xmax=xr)
            axs[i,j].set_xscale('log')
            axs[i,j].xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
            axs[i,j].xaxis.set_minor_formatter(plt.FormatStrFormatter(''))
            if (xr>=2):
                xtic=np.arange(xr,1/xr,-0.5)
            else:
                xtic=np.arange(xr,1/xr,-0.2)
            ytic=np.arange(-10,10,1)
            axs[i,j].set_xticks(xtic)
            axs[i,j].set_yticks(ytic)
            axs[i,j].set_xticks(np.arange(xr,1/xr,-0.1),minor=True)
            #axs[i,j].set_xticklabels(xtic.astype(str), fontsize=5)
            #axs[i,j].set_yticklabels(ytic.astype(str), fontsize=5)
            axs[i,j].tick_params(axis='both', which='major', labelsize=5)
            axs[i,j].tick_params(axis='both', which='minor', labelsize=5)
            axs[i,j].axvline(x=1,c="black",dashes=(5,2),lw=0.3)
    
    # Set colour scale boundaries (log[Teff,K])
    newcmp=heatmap()
    tmin=3.3
    tmax=4.3

    # Plot data - have to go point by point due to different subframes
    if (verbosity>=80):
        print ("Plotting data")
    plotdata=np.zeros((xf,yf,len(compiledseds),3))
    for i in np.arange(len(compiledseds)):
        if (isinstance(compiledseds[i],int)==False):
            anc=compiledanc[i]
            teff=np.squeeze(anc[anc['parameter'].astype(str)=='Teff']['value'])
            if (teff>0):
                for j in np.arange(len(compiledseds[i])):
                    q=compiledseds[i][j]['mask']
                    k=np.squeeze(np.transpose(np.nonzero(filts==compiledseds[i][j]['svoname'])))
                    v=int(k/yf)
                    w=k-v*yf
                    m=compiledseds[i][j]['model']
                    f=compiledseds[i][j]['dered']
                    if (m>0):
                        plotdata[v,w,i,0]=f/m
                    else:
                        plotdata[v,w,i,0]=1.
                    plotdata[v,w,i,1]=f
                    if (q==True):
                        plotdata[v,w,i,2]=teff.astype(float)
                    else:
                        plotdata[v,w,i,2]=-teff.astype(float)
    for v in np.arange(xf):
        for w in np.arange(yf):
            if (v*yf+w<len(filts)):
                npoints=len(plotdata[v,w,plotdata[v,w,:,1]>0.,0])
                axs[v,w].text(0.03,0.97,(filts[v*yf+w].split("/"))[1]+" ("+str(npoints)+")", ha='left', va='top', fontsize=5, transform=axs[v,w].transAxes)
                ptsize=np.maximum(20./np.sqrt(npoints),1)
            # Having to manually set y-axis limits on some plots for unknown reasons
            #axs[v,w].set_ylim(ymin=-5,ymax=4)
            # Fudge to trigger ylim setting
            axs[v,w].scatter(1,-3,s=0,c="#FFFFFF00",cmap=newcmp,edgecolors='none',vmin=tmin,vmax=tmax)
            axs[v,w].scatter(plotdata[v,w,plotdata[v,w,:,2]>0,0],np.log10(plotdata[v,w,plotdata[v,w,:,2]>0,1]),s=ptsize,c=np.log10(plotdata[v,w,plotdata[v,w,:,2]>0,2]),cmap=newcmp,edgecolors='none',vmin=tmin,vmax=tmax)
            axs[v,w].scatter(plotdata[v,w,plotdata[v,w,:,2]<0,0],np.log10(plotdata[v,w,plotdata[v,w,:,2]<0,1]),s=ptsize,c="#AAFFAA77",cmap=newcmp,edgecolors='none',vmin=tmin,vmax=tmax)
 
    # Save the file
    if (verbosity>=80):
        print ("Saving plot")
    plt.savefig(plotfile,dpi=600)
    plt.close("all")

    return

# -----------------------------------------------------------------------------
def excessspaceplot(plotfile,compiledseds,compiledanc,teff):

    # Plot observed-to-model flux excesses in space
    fig,axs,xf,yf,filts=setupfiltergrid(compiledseds)
    fig.text(0.5, 0.00, 'RA', ha='center', fontsize=5)
    fig.text(0.00, 0.5, 'Dec', va='center', rotation='vertical', fontsize=5)

    for i in np.arange(xf):
        for j in np.arange(yf):
            axs[i,j].tick_params(axis='x', labelsize=5)
            axs[i,j].tick_params(axis='y', labelsize=5)
    
    # Set colour scale boundaries (log[Teff,K])
    newcmp=redbluemap()
    tmin=-0.3
    tmax=0.3

    # Plot data - have to go point by point due to different subframes
    if (verbosity>=80):
        print ("Plotting data")
    plotdata=np.zeros((xf,yf,len(compiledseds),3))
    for i in np.arange(len(compiledseds)):
        if (isinstance(compiledseds[i],int)==False):
            anc=compiledanc[i]
            teff=np.squeeze(anc[anc['parameter'].astype(str)=='Teff']['value'])
            ra=np.squeeze(anc[anc['parameter'].astype(str)=='RA']['value'])
            dec=np.squeeze(anc[anc['parameter'].astype(str)=='Dec']['value'])
            if (teff>0):
                for j in np.arange(len(compiledseds[i])):
                    q=compiledseds[i][j]['mask']
                    k=np.squeeze(np.transpose(np.nonzero(filts==compiledseds[i][j]['svoname'])))
                    v=int(k/yf)
                    w=k-v*yf
                    m=compiledseds[i][j]['model']
                    f=compiledseds[i][j]['dered']
                    plotdata[v,w,i,0]=ra
                    plotdata[v,w,i,1]=dec
                    if (m>0):
                        plotdata[v,w,i,2]=f/m
                    else:
                        plotdata[v,w,i,2]=0.
    # Having to manually set y-axis limits on some plots for unknown reasons
    ras=plotdata[:,:,:,0].flatten()
    decs=plotdata[:,:,:,1].flatten()
    ramin=np.min(ras[ras!=0])
    ramax=np.max(ras[ras!=0])
    decmin=np.min(decs[decs!=0])
    decmax=np.max(decs[decs!=0])
    for v in np.arange(xf):
        for w in np.arange(yf):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Attempting to set identical left == right")
                warnings.filterwarnings("ignore", message="Attempting to set identical bottom == top")
                warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
                if (v*yf+w<len(filts)):
                    npoints=len(plotdata[v,w,plotdata[v,w,:,0]!=0.,0])
                    axs[v,w].text(0.03,0.97,(filts[v*yf+w].split("/"))[1]+" ("+str(npoints)+")", ha='left', va='top', fontsize=5, transform=axs[v,w].transAxes)
                    ptsize=np.maximum(20./np.sqrt(npoints),1)
                # Having to manually set y-axis limits on some plots for unknown reasons
                axs[v,w].set_xlim(xmin=ramin,xmax=ramax)
                axs[v,w].set_ylim(ymin=decmin,ymax=decmax)
    #            print (ramin,ramax,decmin,decmax)
                pdata=plotdata[v,w,(plotdata[v,w,:,2]>=0),:]
                axs[v,w].scatter(pdata[:,0],pdata[:,1],s=ptsize,c=np.log10(pdata[:,2]),cmap=newcmp,edgecolors='none',vmin=tmin,vmax=tmax)
                pdata=plotdata[v,w,(plotdata[v,w,:,2]==0),:]
                axs[v,w].scatter(pdata[:,0],pdata[:,1],s=ptsize,c="#00FF00AA",cmap=newcmp,edgecolors='none',vmin=tmin,vmax=tmax)
                axs[v,w].set_xlim(axs[v,w].get_xlim()[::-1])
                #axs[v,w].scatter(plotdata[v,w,plotdata[v,w,:,2]<0,0],np.log10(plotdata[v,w,plotdata[v,w,:,2]<0,1]),s=ptsize,c="#AAFFAA77",cmap=newcmp,edgecolors='none',vmin=tmin,vmax=tmax)
 
    # Save the file
    if (verbosity>=80):
        print ("Saving plot")
    plt.savefig(plotfile,dpi=600)
    plt.close("all")

    return

# -----------------------------------------------------------------------------
def excesscorrnplot(plotfile,compiledseds,compiledanc,teff):
    # Plot correlation of observed/modelled fluxes between filters
    if (verbosity>=80):
        print ("Plot correlation of flux excesses ->",plotfile)
    
    # Set up grid
    filtdata=get_filter_list()
    filts=[]
    for i in np.arange(len(compiledseds)):
        if (isinstance(compiledseds[i],int)==False):
            filts=np.unique(np.append(filts,compiledseds[i]['svoname'],axis=0))

    # Calculate square extent to keep same plot area
    xr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="LumXSPlotXSRange",1][0])
    maxxs=xr
    minxs=1/xr

    # Extract XS data
    if (verbosity>=80):
        print ("Obtaining excess data")
    plotdata=np.zeros((len(filts),len(compiledseds),3))
    for i in np.arange(len(compiledseds)):
        if (isinstance(compiledseds[i],int)==False):
            anc=compiledanc[i]
            teff=np.squeeze(anc[anc['parameter'].astype(str)=='Teff']['value'])
            lum=np.squeeze(anc[anc['parameter'].astype(str)=='Luminosity']['value'])
            ra=np.squeeze(anc[anc['parameter'].astype(str)=='RA']['value'])
            dec=np.squeeze(anc[anc['parameter'].astype(str)=='Dec']['value'])
            if (teff>0):
                for j in np.arange(len(compiledseds[i])):
                    q=compiledseds[i][j]['mask']
                    k=np.squeeze(np.transpose(np.nonzero(filts==compiledseds[i][j]['svoname'])))
                    m=compiledseds[i][j]['model']
                    f=compiledseds[i][j]['dered']
                    plotdata[k,i,0]=teff
                    plotdata[k,i,1]=lum
                    if (m>0):
                        plotdata[k,i,2]=f/m
                    else:
                        plotdata[k,i,2]=0.

    # Restrict filters to those with more than [n] points
    usefilt=np.zeros(len(filts))
    minpoints=int(pyssedsetupdata[pyssedsetupdata[:,0]=="LumXSPlotMinPoints",1][0])
    for k in np.arange(len(filts)):
        npoints=np.count_nonzero(plotdata[k,:,2])
        if (npoints>minpoints):
            usefilt[k]=1
            try:
                bar=np.append(bar,filts[k])
            except:
                bar=filts[k]
    nfilts=np.count_nonzero(usefilt)
    foo=np.zeros((nfilts,len(compiledseds),3))
    for k in np.arange(nfilts):
        foo[k,:,:]=plotdata[filts==bar[k],:,:]
    try:
        filts=bar
        plotdata=foo
    except:
        print_warn ("Luminosity-excess corner plot:")
        print_warn ("Minimum number of points ["+str(minpoints)+"] not found on any filter. Plotting all data.")

    # Plotting data
    if (verbosity>=80):
        print ("Plotting data")
    fig, axs = plt.subplots(len(filts)+1, len(filts)+1)
    newcmp=heatmap()
    tmin=3.3
    tmax=4.3
    for ax1 in np.arange(len(filts)):
        if (verbosity>90):
            print ("   Plotting row:",ax1,filts[ax1])
        for ax2 in np.arange(len(filts)):
            if (ax1!=ax2):
                if (verbosity>95):
                    print ("   Plotting:",ax1,ax2,filts[ax1],filts[ax2])
                axs[ax1,ax2].axis(xmin=minxs,xmax=maxxs)
                axs[ax1,ax2].axis(ymin=minxs,ymax=maxxs)
                axs[ax1,ax2].set_xscale('log')
                axs[ax1,ax2].set_yscale('log')
                axs[ax1,ax2].xaxis.set_major_formatter(plt.FormatStrFormatter(''))
                axs[ax1,ax2].xaxis.set_minor_formatter(plt.FormatStrFormatter(''))
                axs[ax1,ax2].yaxis.set_major_formatter(plt.FormatStrFormatter(''))
                axs[ax1,ax2].yaxis.set_minor_formatter(plt.FormatStrFormatter(''))
                if (xr>=2):
                    xtic=np.arange(xr,1/xr,-0.5)
                else:
                    xtic=np.arange(xr,1/xr,-0.2)
                ytic=xtic
                axs[ax1,ax2].tick_params(axis='x', which='both', direction='in')
                axs[ax1,ax2].tick_params(axis='y', which='both', direction='in')
#                axs[ax1,ax2].set_xticks(xtic)
#                axs[ax1,ax2].set_yticks(ytic)
#                axs[ax1,ax2].set_xticks(np.arange(xr,1/xr,-0.1),minor=True)
#                axs[ax1,ax2].set_yticks(np.arange(xr,1/xr,-0.1),minor=True)
                axs[ax1,ax2].set_xticks([]) # Remove ticks to speed up saving
                axs[ax1,ax2].set_yticks([])
                axs[ax1,ax2].set_xticklabels([], fontsize=1)
                axs[ax1,ax2].set_yticklabels([], fontsize=1)
                n=len(plotdata[ax1,:,2])+len(plotdata[ax2,:,2])
                if (np.sum(n)<10):
                    psize=10
                    opacity="7F"
                elif (np.sum(n)<100):
                    psize=3
                    opacity="5F"
                elif (np.sum(n)<1000):
                    psize=1
                    opacity="3F"
                else:
                    psize=0.3
                    opacity="1F"
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="divide by zero encountered in log10")
                    axs[ax1,ax2].scatter(plotdata[ax1,:,2],plotdata[ax2,:,2],s=psize,c=np.log10(plotdata[ax1,:,0]),cmap=newcmp,edgecolors='none',vmin=tmin,vmax=tmax,zorder=20)

    if (verbosity>80):
        print ("   Plotting labels...")

    # Trim axes and bounds
    for ax in axs.flat:
        ax.axvline(x=1,c="black",lw=0.1,zorder=0) #dashes=(5,2),
        ax.axhline(y=1,c="black",lw=0.1,zorder=0) #dashes=(5,2),
        #ax.invert_xaxis()
        #ax.set(xlim=(minxs,maxxs), ylim=(minxs,maxxs), xticks=[], yticks=[], aspect=1)
#    fig.tight_layout(pad=0.00)
    fig.subplots_adjust(hspace=0,wspace=0,bottom=0.,top=1.,left=0.,right=1.)
    
    # Right-hand labels
    ax2=len(filts)
    for ax1 in np.arange(len(filts)):
        axs[ax1,ax2].set(xlim=(0,1), ylim=(0,1), xticks=[], yticks=[], aspect=1)
        axs[ax1,ax2].text(0.0,0.5, (filts[ax1].split("/"))[1], ha='left', va='center')
        axs[ax1,ax2].axis("off") # remove frame
    # Bottom labels
    ax1=len(filts)
    for ax2 in np.arange(len(filts)):
        axs[ax1,ax2].set(xlim=(0,1), ylim=(0,1), xticks=[], yticks=[], aspect=1)
        axs[ax1,ax2].text(0.5,1.0, (filts[ax2].split("/"))[1], ha='center', va='top')
        axs[ax1,ax2].axis("off") # remove frame
    # Diagonal labels
    for ax1 in np.arange(len(filts)):
        axs[ax1,ax1].set(xlim=(0,1), ylim=(0,1), xticks=[], yticks=[], aspect=1)
        axs[ax1,ax1].text(0.5,0.5, (filts[ax1].split("/"))[1], ha='center', va='center')
        axs[ax1,ax1].axis("off") # remove frame
    # Corner labels
    axs[len(filts),len(filts)].axis("off") # remove frame
#    axs[len(filts),len(filts)].text(0.0,0.8, "All", ha='left', va='center', color="#0000FF")
#    axs[len(filts),len(filts)].text(0.0,0.6, "1:1 match", ha='left', va='center', color="#00FF00")
#    axs[len(filts),len(filts)].text(0.0,0.4, "Blends", ha='left', va='center', color="#FF7700")
#    axs[len(filts),len(filts)].text(0.0,0.2, "Overblend", ha='left', va='center', color="#FF0000")
    
    # Save the file
    if (verbosity>80):
        print ("   Saving final file...")
    fig.set_size_inches(len(filts)+1, len(filts)+1)
    plt.savefig(plotfile,dpi=300)
    plt.close("all")

    return

# -----------------------------------------------------------------------------
#converting master output to valid csv
def convert_master_file(name):
    file = open(name, "r")
    data = list(csv.reader(file, delimiter="\t"))
    file.close()
    num_lines = 10
    index = 2
    header = []
    while index < num_lines:
        line = data[index]
        idx = 0
        while idx < len(line):
            element = line[idx]
            if len(header) > idx:
                header_element = header[idx]
                header_element += '|' + element
                header[idx] = header_element
            else:
                header.append(element)
            idx += 1
        index += 1
    res_data = data[10:]
    result = pd.DataFrame(data=res_data, columns=header)
    return result

# =============================================================================
# MAIN PROGRAMME
# =============================================================================
def pyssed(cmdtype,cmdparams,proctype,procparams,setupfile,handler,total_sources,cmdargs) -> dict:

    # Main routine
    errmsg=""
    version="1.1.dev.20240320"
    try:
        startmain = datetime.now() # time object
        globaltime=startmain
    except:
        startmain = 0

    # Visualiser setup
    task_id = 0
    if (handler != None):
        task_id = handler.task_id
    total_steps = total_sources + 4
    results = {}
    hrd_results = pd.DataFrame(data=[], columns=["Object", "Teff", "Lum", "Rad", "Dist", "Chi^2"])
    if (handler != None):
        handler.submit_status(task_id, "processing", { "stage": 2, "stages": 4, "status": "Loading Setup", "progress": 1 / total_steps, "step": 1, "totalSteps": total_steps })


    # ----------
    # Load setup
    if (setupfile==""):
        setupfile="setup.default"
    global pyssedsetupdata      # Share setup parameters across all subroutines
    try:
        pyssedsetupdata = np.loadtxt(setupfile, dtype=str, comments="#", delimiter="\t", unpack=False)
    except ValueError as e:
        print_fail ("Error loading setup file. Check number of columns is always two and tab-separated.")
        raise

    global verbosity        # Output level of chatter
    verbosity=int(pyssedsetupdata[pyssedsetupdata[:,0]=="verbosity",1][0])
    if (verbosity>=30):
        if (speedtest):
            print ("Setup file loaded:",datetime.now()-startmain,"s")
        else:
            print ("Setup file loaded.")

    # ---------------------------------
    # What type of search are we doing?
    if (handler != None):
        handler.submit_status(task_id, "processing", { "stage": 2, "stages": 4, "status": "Finished Loading Setup", "progress": 2 / total_steps, "step": 2, "totalSteps": total_steps })
    if (cmdtype=="single"):
        searchtype="single" # list of one or more objects
    elif (cmdtype=="list"):
        searchtype="list" # list of one or more objects
    elif (cmdtype=="cone" or cmdtype=="rectangle" or cmdtype=="box" or cmdtype=="volume"):
        searchtype="area" # search on an area of sky
    elif (cmdtype=="criteria" or cmdtype=="complex" or cmdtype=="nongaia" or cmdtype=="uselast"):
        searchtype="none"
        errmsg=("Command type",cmdtype,"not programmed yet")
    else:
        searchtype="error"
        errmsg=("Command type",cmdtype,"not recognised")
    
    # --------------------------------------------------
    # Check whether using existing SEDs/Gaia data or not
    # (the 2:-2 avoids [''] that brackets the array)
    usepreviousrun=int(pyssedsetupdata[pyssedsetupdata[:,0]=="UsePreviousRun",1][0])
    outappend=int(pyssedsetupdata[pyssedsetupdata[:,0]=="OutputAppend",1][0])
    cmdargsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="CmdArgsFile",1])[2:-2]
    sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="SEDsFile",1])[2:-2]
    ancfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncFile",1])[2:-2]
    photdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotTempDir",1])[2:-2]
    ancdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncTempDir",1])[2:-2]
    photfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotFile",1])[2:-2]
    maxdisterr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxDistError",1][0])

    if (usepreviousrun==0):
        np.save(cmdargsfile,cmdargs)
    else:
        lastcmd=np.load(cmdargsfile)
        try:
            issame=(cmdargs==lastcmd).all()
        except ValueError:
            issame=False
        if (issame==False):
            print_fail ("UsePreviousRun!=0 but last source settings do not match these source settings!")
            print ("Previous settings:",lastcmd)
            print ("Current settings:",cmdargs)
            print_fail ("Execution has stopped to prevent mismatch between files and data.")
            print ("Either set UsePreviousRun to zero to create a new run, or revise input parameters.")
            error=1
            return

    if (usepreviousrun>0 and searchtype=="list"):
        print_fail ("The UsePreviousRun flag is not supported when a list of targets is specified.")
        print_fail ("Execution has stopped to prevent mismatch between files and data.")
        error=1
        return

    if (speedtest):
        print ("Parse/initiate command line options:",datetime.now()-startmain,"s")

    # ------------------------------------------------------------
    # Get stellar atmosphere model data and extinction corrections at start of run, if needed
    if (proctype=="simple"):
        modeldata=get_model_grid()
    if (proctype!="none"):
        avcorrtype=int(pyssedsetupdata[pyssedsetupdata[:,0]=="ExtCorrDetail",1][0])
        if (avcorrtype>0):
            avdata=get_av_grid()
        else:
            avdata=[]
    fitebv=int(pyssedsetupdata[pyssedsetupdata[:,0]=="FitEBV",1][0])
    if (fitebv>0):
        print_warn ("Warning! FitEBV is currently untested. Do not use for scientific results.")
    if (speedtest):
        print ("Stellar models:",datetime.now()-startmain,"s")

    # -----------------------------------------------
    # Do all the special stuff for area searches here
    mastercat=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PrimaryCatRef",1])[2:-2]
    if (searchtype=="area"):
        # For area searches, only have to download one set of data, so we do that here.
        # For list searches, we'll defer that to the main loop.
        if (usepreviousrun==0):
            primarycat=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PrimaryCatRef",1])[2:-2]
            try:
                starttime = datetime.now() # time object
            except:
                starttime = 0
            if (verbosity >=20):
                if (cmdtype=="cone"):
                    print ("Downloading data for cone (",cmdargs[2],cmdargs[3],cmdargs[4],")")
                if (cmdtype=="rectangle"):
                    print ("Downloading data for rectangle (",cmdargs[2],cmdargs[3],cmdargs[4],cmdargs[5],")")
                if (cmdtype=="box"):
                    print ("Downloading data for box (",cmdargs[2],cmdargs[3],cmdargs[4],cmdargs[5],")")
                if (cmdtype=="volume"):
                    print ("Downloading data for volume (",cmdargs[2],cmdargs[3],cmdargs[4],cmdargs[5],")")
            if (verbosity >=40):
                print ("Downloading photometry...")
            get_area_data(cmdargs,get_catalogue_list(),photdir)
            if (verbosity >=40):
                print ("Downloading ancillary data...")
            get_area_data(cmdargs,get_ancillary_list(),ancdir)
            try:
                now = datetime.now() # time object
                elapsed = float(now.strftime("%s.%f"))-float(starttime.strftime("%s.%f"))
                if (verbosity>=30):
                    print ("Took",elapsed,"seconds to download data, [",starttime,"] [",now,"]")
            except:
                pass
    
        # For area searches we now want to make SEDs and get the ancillary data
        if (cmdtype!="box"):
            compiledseds,compiledanc,sourcedata=get_sed_multiple()
        else: # Extra parameters to deal with TrimBox
            compiledseds,compiledanc,sourcedata=get_sed_multiple(method="box",ra1=float(cmdargs[2]),ra2=float(cmdargs[4]))

        if (outappend>0 and usepreviousrun<=4):
            if (verbosity >=50):
                print ("Loading previous compiled SEDs for append")
            # Photometry
            sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledSEDsFile",1])[2:-2]
            lastcompiledseds=np.load(sedsfile,allow_pickle=True)
            # Ancillary
            sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledAncFile",1])[2:-2]
            lastcompiledanc=np.load(sedsfile,allow_pickle=True)
            # Source list
            sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledSourceFile",1])[2:-2]
            lastsourcedata=np.loadtxt(sedsfile,delimiter="\t",dtype=str)
        
        # For area searches we now make the initial plots
        makeareaplots=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MakeAreaPlots",1][0])
        if ((makeareaplots>0) and (usepreviousrun<=5)):
            globalplots(compiledseds,compiledanc,sourcedata)
        if (speedtest):
            print ("Area search setup:",datetime.now()-startmain,"s")
    elif (usepreviousrun>=5):
        print_warn ("Nothing to do")

        
    # -------------------------------------------
    # If using a list of sources, load that list
    if (searchtype=="list"):
        listfile= "targets.test" if cmdparams == "" else cmdparams
        sourcedata = np.loadtxt(listfile, dtype=str, comments="#", delimiter="\t", unpack=False)
        if (speedtest):
            print ("List search setup:",datetime.now()-startmain,"s")
    # for single sources, load from the module / command line input
    elif (searchtype=="single"):
        sourcedata = np.expand_dims(np.array(cmdparams),axis=0)

    # -------------------------------------------
    # Restart or append output
    object_counter=0
    object_offset=0
    savemasteroutput=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveMasterOutput",1][0])
    outmasterfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="MasterOutputFile",1])[2:-2]
    outparamfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="OutputParamFile",1])[2:-2]
    outancfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="OutputAncFile",1])[2:-2]
    ancillary_queries=get_ancillary_list()
    ancillary_params=np.array(["#Object","RA","Dec"],dtype="str")
    ancillary_params=np.append(ancillary_params,np.unique(ancillary_queries['paramname']),axis=0)
    outputmasked=int(pyssedsetupdata[pyssedsetupdata[:,0]=="OutputMasked",1][0])
    sep=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="OutputSeparator",1])[2:-2]
    if (sep=="\\\\t"):
        sep="\t"
    if (sep=="\\\\n"):
        sep="\n"
    # Double up ancillary parameters to include errors (except for object name)
    nparams=len(ancillary_params)
    ancillary_params=np.append(ancillary_params,ancillary_params[1:],axis=0)
    for i in np.arange(nparams-1,0,-1):
        ancillary_params[i*2-1]=ancillary_params[i]
        ancillary_params[i*2]="e_"+ancillary_params[i]
    # Load data on filters and catalogues
    filtdata=get_filter_list()
    catdata=get_catalogue_list()
    svodata=get_svo_data(filtdata)
    if (usepreviousrun<5):
        # Append = 0 -> start new file, write headers
        if (outappend==0):
            # Write parameter file header
            with open(outparamfile, "w") as f:
                f.write("#Object"+sep+"Effective temperature"+sep+"Luminosity"+sep+"Radius"+sep+"Distance"+sep+"Chi^2\n")
                f.write("#Name"+sep+"Kelvin"+sep+"Solar luminosities"+sep+"Solar Radii"+"pc"+sep+"\n")
            np.savetxt(outancfile,ancillary_params,fmt="%s",delimiter=sep,newline=sep)
            with open(outancfile, "a") as f:
                f.write("\n")
            # Write master output file header
            if (savemasteroutput>0):
                # Data labels/catalogues
                objectlist=np.array(["#Object"],dtype="str")
                masterlist=np.array(["RA","Dec","PMRA","PMDec"],dtype="str")
                fittedlist=np.array(["Teff","Lum"],dtype="str")
                adoptedlist=np.array(["Distance","E(B-V)","logg","[Fe/H]"],dtype="str")
                statlist=np.array(["chisq","RUWE","GoF","UVXS","IRXS"],dtype="str")
                catlist=np.copy(catdata['catname'])
                filtlist=np.copy(filtdata['catname'])
                ancillarylist=np.copy(ancillary_queries['paramname'])
                masteroutputlabels=np.concatenate((objectlist,masterlist,masterlist,fittedlist,fittedlist,adoptedlist,adoptedlist,statlist,catlist,filtlist,filtlist,filtlist,filtlist,ancillarylist,ancillarylist,ancillarylist))
                # Filters
                objectlist=np.array(["#Object"],dtype="str")
                masterlist=np.array(["-","-","-","-"],dtype="str")
                fittedlist=np.array(["-","-"],dtype="str")
                adoptedlist=np.array(["-","-","-","-"],dtype="str")
                statlist=np.array(["-","-","-","-","-"],dtype="str")
                catlist=np.full(len(catlist),"-")
                filtlist=np.copy(filtdata['svoname'])
                ancillarylist=np.full(len(ancillarylist),"-")
                masteroutputfilters=np.concatenate((objectlist,masterlist,masterlist,fittedlist,fittedlist,adoptedlist,adoptedlist,statlist,catlist,filtlist,filtlist,filtlist,filtlist,ancillarylist,ancillarylist,ancillarylist))
                # Units
                masterlist=np.array(["deg","deg","mas/yr","mas/yr"],dtype="str")
                fittedlist=np.array(["K","LSun"],dtype="str")
                adoptedlist=np.array(["pc","mag","dex","dex"],dtype="str")
                filtlist[:]="Jy"
                statlist[:]="-"
                ancillarylist=ancillary_queries['units']
                masteroutputunits=np.concatenate((objectlist,masterlist,masterlist,fittedlist,fittedlist,adoptedlist,adoptedlist,statlist,catlist,filtlist,filtlist,filtlist,filtlist,ancillarylist,ancillarylist,np.full(len(ancillarylist),"-")))
                # Data type	[OBJECT ADOPTED FITTED ANCILLARY PHOTOMETRY MODEL]
                masteroutputtypes1=np.concatenate((objectlist,np.full(len(masterlist),"Adopted"),np.full(len(masterlist),"Adopted"),np.full(len(fittedlist),"Fitted"),np.full(len(fittedlist),"Fitted"),np.full(len(adoptedlist),"Adopted"),np.full(len(adoptedlist),"Adopted"),np.full(len(statlist),"Statistic"),np.full(len(catlist),"ID"),np.full(len(filtlist),"Photometry"),np.full(len(filtlist),"Photometry"),np.full(len(filtlist),"Dereddened"),np.full(len(filtlist),"Model"),np.full(len(ancillarylist),"Ancillary"),np.full(len(ancillarylist),"Ancillary"),np.full(len(ancillarylist),"Ancillary")))
                # Data type [OBJECT VALUE ERROR SOURCE]
                masteroutputtypes2=np.concatenate((objectlist,np.full(len(masterlist),"Value"),np.full(len(masterlist),"Error"),np.full(len(fittedlist),"Value"),np.full(len(fittedlist),"Error"),np.full(len(adoptedlist),"Value"),np.full(len(adoptedlist),"Error"),np.full(len(statlist),"Value"),np.full(len(catlist),"ID"),np.full(len(filtlist),"Value"),np.full(len(filtlist),"Error"),np.full(len(filtlist),"Value"),np.full(len(filtlist),"Value"),np.full(len(ancillarylist),"Value"),np.full(len(ancillarylist),"Error"),np.full(len(ancillarylist),"Source")))
                # Wavelength, width [Angstroms]
                # A_lambda/Av [value] - not easily accessible, need dust_extinction's F99
                filtwave=np.empty(len(filtdata['svoname']),dtype=float)
                filtdw=np.empty(len(filtdata['svoname']),dtype=float)
                filtalamb=np.empty(len(filtdata['svoname']),dtype=float)
                rv=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultRV",1][0])
                ext = F99(Rv=rv) # Adopt standard Fitzpatrick 1999 reddening law
                for i in np.arange(len(filtdata['svoname'])):
                    svokey=filtdata['svoname'][i]
                    filtwave[i]=int(svodata[svodata['svoname']==svokey]['weff'][0]*10.)/10. # Use .1f to avoid long strings
                    filtdw[i]=int(svodata[svodata['svoname']==svokey]['dw'][0]*10.)/10. # "
                    wavel=np.where(filtwave[i]>=1000., filtwave[i], 1000.) # Prevent overflowing the function
                    wavel=np.where(wavel<=33333., wavel, 33333.)            #
                    filtalamb[i]=-2.5*np.log10(ext.extinguish(wavel*u.AA, Av=1.))
                masteroutputwave=np.concatenate((np.array(["#Wavelength[AA]"],dtype="str"),np.full(len(masterlist)*2,"-"),np.full(len(fittedlist)*2,"-"),np.full(len(adoptedlist)*2,"-"),np.full(len(statlist),"-"),np.full(len(catlist),"-"),filtwave,filtwave,filtwave,filtwave,np.full(len(ancillarylist)*3,"-")))
                masteroutputdw=np.concatenate((np.array(["#Width[AA]"],dtype="str"),np.full(len(masterlist)*2,"-"),np.full(len(fittedlist)*2,"-"),np.full(len(adoptedlist)*2,"-"),np.full(len(statlist),"-"),np.full(len(catlist),"-"),filtdw,filtdw,filtdw,filtdw,np.full(len(ancillarylist)*3,"-")))
                masteroutputalamb=np.concatenate((np.array(["#Alambda"],dtype="str"),np.full(len(masterlist)*2,"-"),np.full(len(fittedlist)*2,"-"),np.full(len(adoptedlist)*2,"-"),np.full(len(statlist),"-"),np.full(len(catlist),"-"),filtalamb,filtalamb,filtalamb,filtalamb,np.full(len(ancillarylist)*3,"-")))
                # Write the master output file headers
                with open(outmasterfile, "w") as f:
                    f.write("#PySSED output version"+version+"\n")                
                    try:
                        f.write("#Created"+datetime.now()+"\n") # time object
                    except:
                        pass
                    f.write("#Input: "+' '.join(str(x) for x in cmdargs)+"\n")
                    #YYY
                    #			RA, Dec (extracted from input catalogue)
                    #			PMRA, PMDec (extracted from input catalogue, additional parameters can be given)
                    #			Parallax [mas] <-> Distance [pc] (used interchangeably)
                    #			Teff/logg/[Fe/H] can be used as a starting point for fitting
                    f.write(sep.join(str(x) for x in masteroutputtypes1)+"\n")
                    f.write(sep.join(str(x) for x in masteroutputlabels)+"\n")
                    f.write(sep.join(str(x) for x in masteroutputfilters)+"\n")
                    f.write(sep.join(str(x) for x in masteroutputtypes2)+"\n")
                    f.write(sep.join(str(x) for x in masteroutputunits)+"\n")
                    f.write(sep.join(str(x) for x in masteroutputwave)+"\n")
                    f.write(sep.join(str(x) for x in masteroutputdw)+"\n")
                    f.write(sep.join(str(x) for x in masteroutputalamb)+"\n")
        # Append = 1 -> add new list to existing (do nothing here)
        # Append = 2 -> check which objects have been done already and remove from sourcedata
        elif (outappend==2):
            if (verbosity >=50):
                print ("Checking existing objects...")
            try:
                processeddata = np.loadtxt(outparamfile, dtype=str, comments="#", delimiter=sep, unpack=False)
            except OSError:
                print_fail("ERROR! No existing datafiles!")
                raise OSError("Existing datafile %s does not exist. Try a different file or use OutputAppend = 0 in setup." % outparamfile)
            try:
                procdat = processeddata[:,0]
            except: # only one object
                procdat = processeddata[0]
            fullsourcedata = np.copy(sourcedata) # take a backup for numerical purposes
            sourcedata = [i for i in sourcedata if i not in procdat]
            if (searchtype=="area"):
                compiledseds = [i for i in compiledseds if i not in procdat]
                compiledanc = [i for i in compiledanc if i not in procdat]
            object_counter+=len(procdat)
            object_offset=len(procdat)
            #object_counter=0
            if (verbosity >=50):
                print ("Existing data:",len(procdat),"objects")
                print ("Input file:   ",len(fullsourcedata),"objects")
                print ("To process:   ",len(sourcedata),"objects")
                print ("Total unique: ",len(sourcedata)+len(procdat),"objects")
            if (searchtype=="area"):
                radec_current=np.c_[np.array([(a[a['parameter'].astype(str)=="RA"]['value'].astype(float)).flatten() for a in compiledanc]).flatten(),np.array([(a[a['parameter'].astype(str)=="Dec"]['value'].astype(float)).flatten() for a in compiledanc]).flatten()]
                radec_last=np.c_[np.array([(a[a['parameter'].astype(str)=="RA"]['value'].astype(float)).flatten() for a in lastcompiledanc]).flatten(),np.array([(a[a['parameter'].astype(str)=="Dec"]['value'].astype(float)).flatten() for a in lastcompiledanc]).flatten()]
                overlap=np.isin(radec_current,radec_last)
                overlap=overlap[:,0]*overlap[:,1]
                idxolap=np.nonzero(overlap)
                overlap=np.isin(radec_current,radec_last[idxolap])
                overlap=overlap[:,0]*overlap[:,1]
                idxolap2=np.nonzero(overlap)
                compiledanc[idxolap2]=lastcompiledanc[idxolap]

        # XXX Append = 2 broken in final plots - need full dataset reloaded

    # Get G-Tomo dereddening if required
    extmap=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="ExtMap",1])[2:-2]
    extmaxangle=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ExtMaxAngle",1][0])
    ext_ra=-360.
    ext_dec=-360.
    if ((extmap=="GTomo") & (searchtype=="area")):
        if (verbosity >=50):
            print ("Getting G-Tomo extinction...")
        #hdf5file25="gtomo/_data/app_data/explore_cube_density_values_025pc_v1.h5"
        #headers25,cube25,axes25,min_axes25,max_axes25,step25,hw25,points25,s25 = load_cube(hdf5file25)
        #ext_dist25, ext_av25, ext_dav25 = gtomo_reddening(sc, cube=cube25, axes=axes25, max_axes=max_axes25, step_pc=5)
        #hdf5file50="gtomo/_data/app_data/explore_cube_density_values_050pc_v1.h5"
        hdf5file50="gtomo/_data/app_data/explore_cube_density_values_050pc_v2.h5"
        headers50,cube50,axes50,min_axes50,max_axes50,step50,hw50,points50,s50 = load_cube(hdf5file50)
        ext_ra=float(cmdargs[2])
        ext_dec=float(cmdargs[3])
        sc=SkyCoord(ext_ra*u.deg, ext_dec*u.deg, frame='icrs')
        ext_dist50, ext_av50, ext_dav50 = gtomo_reddening(sc, cube=cube50, axes=axes50, max_axes=max_axes50, step_pc=5)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Main loop over sources: obtaining and processing SED data
    if (verbosity>=95):
        print ("Entering main loop")
    if (handler != None):
        handler.submit_status(task_id, "processing", { "stage": 2, "stages": 4, "status": f"Starting to process {len(sourcedata)} source{'s' if len(sourcedata) > 1 else ''}", "progress": 3 / total_steps, "step": 3, "totalSteps": total_steps })
    if (speedtest):
        print ("Main routine to main loop:",datetime.now()-startmain,"s")
    if (usepreviousrun>=5):
        if (verbosity>=20):
            print ("Using existing fits")
    else:
        try:
            nobjects=len(sourcedata)
        except TypeError: # only one object
            nobjects=1
            sourcedata=np.expand_dims(sourcedata,axis=0)
            print (len(sourcedata))
        if (verbosity>=20):
            print ("Processing",nobjects,"objects")
        if (nobjects==0):
            if ((searchtype=="single" or searchtype=="list") and outappend==2):
                print_warn ("No objects to process. These objects are already in the list.")
            else:
                print_warn ("No objects to process in this region.")

        mindatapoints=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MinDataPoints",1][0])
            
        for counter, source in enumerate(sourcedata):
            if (handler != None):
                handler.submit_status(task_id, "processing", { "stage": 2, "stages": 4, "status": f"Processing source {source}", "progress": (3 + 1) / total_steps, "step": (3 + counter + 1), "totalSteps": total_steps })
            object_counter+=1
            
            if (verbosity >=50):
                print ("")
            if (verbosity>=10):
                    print ("Processing source #",object_counter-object_offset,"of",nobjects,":",source)
            try:
                startsource = datetime.now() # time object
            except:
                startsource = 0
            globaltime=startsource

            # ----------------
            # SED construction
            if (searchtype=="single" or searchtype=="list"):
            # Get SED for list sources

                # If loading SEDs that are already created from disk...
                if (usepreviousrun>0):
                    if (verbosity>=60):
                        print ("Loading data from",sedsfile)
                        print ("Loading data from",ancfile)
                    sed=np.load(sedsfile,allow_pickle=True)
                    ancillary=np.load(ancfile,allow_pickle=True)
                    if (verbosity>40):
                        print ("Extracting SEDs from pre-existing data")
                # If loading pre-downloaded data from disk...
                elif (usepreviousrun>0):
                    dr3_data=np.load(photfile,allow_pickle=True)
                    if (verbosity>40):
                        print ("Extracting source from pre-existing data")
                # Else if querying online databases
                else:
                    sed,ancillary,errmsg=get_sed_single(source)
                    # Parse setup data: am I saving this file? [Done later]
                    #saveseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveSEDs",1][0])
                    #if (saveseds>0):
                    #    if (verbosity>=60):
                    #        print ("Saving data to",sedsfile)
                    #    np.save(sedsfile,sed)
                    #    if (verbosity>=60):
                    #        print ("Saving data to",ancfile)
                    #    np.save(ancfile,ancillary)
            elif (searchtype=="area"):
            # Get SED for area sources
                sed=compiledseds[object_counter-1]
                ancillary=compiledanc[object_counter-1]
#                if (verbosity>=60):
#                    print (mastercat,"source ID:",source)
            if (speedtest):
                print ("SED creation:",datetime.now()-startsource,"s")

            # Merge/deredden SED and test for bad data
            ebv=0
            dist=0
            try:
                if (len(sed)>0):
                    # Merge duplicated filters in SED
                    sed=merge_sed(sed)
                    # Merge ancillary data to produce final values
                    ancillary=merge_ancillary(ancillary)

                    # Deredden
                    ra=ancillary[(ancillary['parameter']=='RA') & (ancillary['mask']==True)]['value']
                    dec=ancillary[(ancillary['parameter']=='Dec') & (ancillary['mask']==True)]['value']
                    dist=adopt_distance(ancillary)
                    if (extmap=="GTomo"): # GTomo done outside deredden() so execution not repeated
                        gtomo_reextract=False
                        if (searchtype=="area"):
                            dra=float(reducto(ra))-float(ext_ra)
                            ddec=float(reducto(dec))-float(ext_dec)
                            if ((dra>extmaxangle) | (ddec>extmaxangle)):
                                gtomo_reextract=True
                            if (verbosity >=70):
                                print ("Requesting new G-Tomo extinction vector...")
                        if ((searchtype!="area") | (gtomo_reextract==True)):
                            if (verbosity >=70):
                                print ("Getting G-Tomo extinction...")
                            ext_ra=ra
                            ext_dec=dec
                            #hdf5file25="gtomo/_data/app_data/explore_cube_density_values_025pc_v1.h5"
                            #headers25,cube25,axes25,min_axes25,max_axes25,step25,hw25,points25,s25 = load_cube(hdf5file25)
                            #ext_dist25, ext_av25, ext_dav25 = gtomo_reddening(sc, cube=cube25, axes=axes25, max_axes=max_axes25, step_pc=5)
                            hdf5file50="gtomo/_data/app_data/explore_cube_density_values_050pc_v1.h5"
                            hdf5file50="gtomo/_data/app_data/explore_cube_density_values_050pc_v2.h5"
                            headers50,cube50,axes50,min_axes50,max_axes50,step50,hw50,points50,s50 = load_cube(hdf5file50)
                            sc=SkyCoord(ext_ra*u.deg, ext_dec*u.deg, frame='icrs')
                            ext_dist50, ext_av50, ext_dav50 = gtomo_reddening(sc, cube=cube50, axes=axes50, max_axes=max_axes50, step_pc=5)
                        input_ebv=get_gtomo_av(dist,ext_dist50,ext_av50)/3.1 #Rv=3.1 in GTomo
                        #input_ebv=get_gtomo_ebv(dist,ext_dist50,ext_av50,ext_dist25,ext_av25)
                    else:
                        input_ebv=0.
                    sed,ebv=deredden(sed,ancillary,dist,avdata,input_ebv)
                    
                    if ((verbosity>=97) or (verbosity>=70 and searchtype=="single")):
                        print (sed)
                    elif (verbosity>=50):
                        print ("SED contains",len(sed),"points")
                    pass
            except TypeError:
                if (verbosity>=30):
                    print_warn (mastercat+" source ID: "+str(source)+" failed due to bad or no data")
                if (verbosity>=70):
                    print (sed)
                    print (ancillary)
                    pass
            except UnboundLocalError:
                print_fail ("ERROR! SED not defined.")
                if (usepreviousrun > 0):
                    print_fail ("This may be because you are trying to use pre-existing photometry that doesn't exist.")
                    print_fail ("Try setting 'UsePreviousRun 0' in the setup file.")
                raise
            if (speedtest):
                print ("Merge and deredden:",datetime.now()-startsource,"s")

            # --------------
            # SED processing

            teff=0; lum=0; rad=0; logg=0; feh=0
            chisq=0; fitsuccess=0
            oe=0; ruwe=0; gof=0; avoeflux=0; uvxs=0; irxs=0

            # Do we need to process the SEDs? No...?
            if (proctype == "none"):
                if (verbosity > 0):
                    print ("No further processing required.")
                modwave=np.empty((1),dtype=float)
                modflux=np.empty((1),dtype=float)
                fitsuccess=1
            # No data - cannot fit
            elif ((isinstance(sed,int)==True) or (sed.size==0) or (len(sed[sed['mask']==True]))<mindatapoints):
                if (verbosity > 0):
                    print ("Not enough data to fit.")
                modwave=np.empty((1),dtype=float)
                modflux=np.empty((1),dtype=float)
            # Yes: continue to process according to model type
            elif (dist>0.):
                # ...blackbody
                if (proctype == "bb"):
                    if (verbosity > 20):
                        print ("Fitting SED with blackbody...")
                    sed,modwave,modflux,teff,rad,lum,chisq=sed_fit_bb(sed,ancillary,avdata,ebv)
                    logg=-9.99
                    feh=-9.99
                # ...trapezoid
                elif (proctype == "trap"):
                    if (verbosity > 20):
                        print ("Fitting SED with trapezoidal integration...")
                    sed,modwave,modflux,teff,rad,lum,chisq=sed_fit_trapezoid(sed,ancillary,avdata,ebv)
                    logg=-9.99
                    feh=-9.99
                # ...simple SED fit
                elif (proctype == "simple"):
                    if (verbosity > 20):
                        print ("Fitting SED with simple stellar model...")
                    sed,modwave,modflux,teff,rad,lum,logg,feh,chisq,ebv=sed_fit_simple(sed,ancillary,modeldata,avdata,ebv)
                elif (proctype == "fit"):
                    if (verbosity > 20):
                        print ("Fitting SED with full stellar model...")
                elif (proctype == "binary"):
                    if (verbosity > 20):
                        print ("Fitting SED with binary stellar model...")
                # Assuming sensible results are returned...
                if ((teff>0.) or (proctype=="trap")):
                    # Unless processing a single source, add entry to global output files
                    #if (cmdtype!="single"):
                    outparams=str(source)+sep+str(teff)+sep+str(lum)+sep+str(rad)+sep+str(dist)+sep+str(chisq)+"\n"
                    added = pd.DataFrame(data=[[source, str(teff), str(lum), str(rad), str(dist), str(chisq)]], columns=["Object", "Teff", "Lum", "Rad", "Dist", "Chi^2"])
                    hrd_results = pd.concat([hrd_results, added])
                    with open(outparamfile, "a") as f:
                        f.write(outparams)
                    #outanc=str(source)+sep+str(np.squeeze(ancillary[(ancillary['parameter']=='RA') & (ancillary['mask']==True)]['value']))+sep+str(np.squeeze(ancillary[(ancillary['parameter']=='RA') & (ancillary['mask']==True)]['err']))+sep+str(np.squeeze(ancillary[(ancillary['parameter']=='Dec') & (ancillary['mask']==True)]['value']))+sep+str(np.squeeze(ancillary[(ancillary['parameter']=='Dec') & (ancillary['mask']==True)]['err']))
                    outanc=str(source)+sep
                    for param in ancillary_params[1:]:
                        if (len(ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)]['value'])>0):
                            outanc=outanc+sep+str(np.squeeze(ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)]['value']))+sep+str(np.squeeze(ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)]['err']))
                        else:
                            outanc=outanc+sep+"--"+sep+"--"
                    outanc=outanc+"\n"
                    with open(outancfile, "a") as f:
                        f.write(outanc)
                    if (verbosity > 20):
                        print (source,"fitted with Teff=",teff,"K, R=",rad,"Rsun, L=",lum,"Lsun")
                    if (verbosity > 40):
                        print ("...dist=",dist,"pc, E(B-V)=",ebv,"mag, chi^2=",chisq)
                    # Add model flux to SED
                    sed['model']=modflux

                    # Calculate goodness of fit and excess parameters
                    truesed=sed[sed['mask']==True]
                    modelledsed=truesed[truesed['model']>0]
                    if (len(modelledsed)>0):
                        oe=modelledsed['dered']/modelledsed['model']-1
                        fracerr=modelledsed['derederr']/modelledsed['model']
                        ruwe=np.average(np.abs(oe)/fracerr)
                        gof=np.median(np.abs(oe))
                        uvsed=truesed[modelledsed['wavel']<4000.]
                        if (len(uvsed)>0):
                            uvxs=np.average(uvsed['dered']/uvsed['model']-1)
                        else:
                            uvxs=0
                        irsed=modelledsed[modelledsed['wavel']>30000.]
                        if (len(irsed)>0):
                            irxs=np.average(irsed['dered']/irsed['model']-1)
                        else:
                            irxs=0
                        if (verbosity>80):
                            print ("Obs/Exp flux:",oe)
                            print ("fracerr:",fracerr)
                            print ("ruwe:",ruwe)
                            print ("gof:",gof)
                            print ("UVXS:",uvxs)
                            print ("IRXS:",irxs)
                    else:
                        if (verbosity>80):
                            print ("No modelled SED points.")

                    fitsuccess=1
                else:
                    print_warn ("Could not fit data for this object")
            else:
                print_warn ("Invalid distance - abandoning fit for this object (check MaxDistError="+str(maxdisterr)+")")
            if (speedtest):
                print ("Fitted SED:",datetime.now()-startsource,"s")

            # Add fitted parameters to the ancillary data # YYY Add object name to file
            if (ancillary.size==0):
                pass
            else:
                ancempty=np.empty_like(ancillary[0])
                ancempty['catname']="Fitted"
                ancempty['colname']="Fitted"
                ancempty['err']=0.
                ancempty['priority']=0
                ancempty['mask']=True
                if (len(ancillary[ancillary['parameter']=="E(B-V)"])==0):
                    newanc=np.copy(ancempty)
                    newanc['parameter']="E(B-V)"
                    newanc['value']=ebv
                    extmap=pyssedsetupdata[pyssedsetupdata[:,0]=="ExtMap",1]
                    newanc['catname']="Internal"
                    if (fitebv>0):
                        newanc['colname']="Fitted"
                    else:
                        newanc['colname']=extmap
                    ancillary=np.append(ancillary,newanc)
                if (len(ancillary[ancillary['parameter']=="Teff"])==0):
                    newanc=np.copy(ancempty)
                    newanc['parameter']="Teff"
                    newanc['value']=teff
                    ancillary=np.append(ancillary,newanc)
                if (len(ancillary[ancillary['parameter']=="Luminosity"])==0):
                    newanc=np.copy(ancempty)
                    newanc['parameter']="Luminosity"
                    newanc['value']=lum
                    ancillary=np.append(ancillary,newanc)
                if (len(ancillary[ancillary['parameter']=="Radius"])==0):
                    newanc=np.copy(ancempty)
                    newanc['parameter']="Radius"
                    newanc['value']=rad
                    ancillary=np.append(ancillary,newanc)
                if (searchtype=="area"):
                    compiledseds[object_counter-1]=sed
                    compiledanc[object_counter-1]=ancillary
            if (speedtest):
                print ("Ancillary parameters:",datetime.now()-startsource,"s")

            # ---------------------
            # Save and plot results
            maxnobjects=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxIndividualSaves",1][0])

            # Save individual SED to its own file?
            saveeachsed=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveEachSED",1][0])
            sedsdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="SEDsDir",1])[2:-2]
            if (((saveeachsed>1) or ((saveeachsed==1) and (fitsuccess==1))) and (nobjects<maxnobjects)):
                sedfile=sedsdir+source.replace(" ","_")+".sed"
                if (verbosity>=60):
                    print ("Saving SED to",sedfile)
                if ((isinstance(sed,int)==True) or (sed.size==0)):
                    print_warn ("No SED to save")
                else:
                    np.savetxt(sedfile, sed, fmt="%s", delimiter=sep, header=sep.join(sed.dtype.names))
            # Save individual ancillary data to its own file?
            saveeachanc=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveEachAnc",1][0])
            ancdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncDir",1])[2:-2]
            if (((saveeachanc>1) or ((saveeachanc==1) and (fitsuccess==1))) and (nobjects<maxnobjects) and (ancillary.size>0)): #and (usepreviousrun>=3 or searchtype=="area")):
                # When running Visualiser, add "Object" to each row of .anc
                if (handler != None):
                    obj_col = np.full(len(ancillary), source, dtype="U100")
                    ancillary2 = rfn.append_fields(ancillary, "Object", obj_col)
                ancfile=ancdir+source.replace(" ","_")+".anc"
                if (verbosity>=60):
                    print ("Saving ancillary data to",ancfile)
                if (handler != None): # Visualiser
                    np.savetxt(ancfile, ancillary, fmt="%s", delimiter=sep, header=sep.join(ancillary2.dtype.names))
                    obj_pd = pd.DataFrame(data=obj_col, columns=["Object"])
                    anc_pd = pd.DataFrame(data=ancillary, columns=ancillary.dtype.names)
                    out_anc_pd = pd.concat([anc_pd, obj_pd], axis=1)
                    results[source.replace(" ","_")+".anc"]=out_anc_pd
                else: # not Visualiser
                    np.savetxt(ancfile, ancillary, fmt="%s", delimiter=sep, header=sep.join(ancillary.dtype.names))

            # Plot the SED if desired and save to file
            plotseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="PlotSEDs",1][0])
            if (((plotseds>0) and (plotseds>1 or fitsuccess==1)) and (nobjects<maxnobjects)):
                plotdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PlotDir",1])[2:-2]
                plotfile=plotdir+source.replace(" ","_")+".png"
                if ((isinstance(sed,int)==True) or (sed.size==0)):
                    print_warn ("No SED to plot")
                elif (len(sed[sed['mask']==True])>0):
                    if (verbosity>=60):
                        print ("Plotting SED to",plotfile)
                    plot = 0
                    inset_plot = 0
                    if (handler != None):
                        image = False
                    else:
                        image = True
                    if ('modwave' in locals()):
                        if (len(modwave)>0):
                            plot,inset_plot=plotsed(sed,modwave,modflux,plotfile,image)
                    else: # If model is bad / uncomputed
                        plot,inset_plot=plotsed(sed,[],[],plotfile,image)

                    # Visualiser additions
                    if (handler != None):
                        plot_obj = pd.DataFrame(data=np.full(len(plot), source, dtype="U100"), columns=["Object"])
                        inset_obj = pd.DataFrame(data=np.full(len(inset_plot), source, dtype="U100"), columns=["Object"])
                        plot = pd.concat([plot, plot_obj], axis=1)
                        inset_plot = pd.concat([inset_plot, inset_obj], axis=1)
                        plot.to_csv(f"../output/{source.replace(' ', '_')}_Plot.csv", index=False)
                        inset_plot.to_csv(f"../output/{source.replace(' ', '_')}_Plot.csv", index=False)
                        results[f"{source.replace(' ', '_')}_Plot.csv"] = plot
                        results[f"{source.replace(' ', '_')}_Inset_Plot.csv"] = inset_plot

                else:
                    if (verbosity>=20):
                        print_warn ("Not enough data to plot SED!")
            if (verbosity >=30):
                try:
                    now = datetime.now() # time object
                    elapsed = float(now.strftime("%s.%f"))-float(startsource.strftime("%s.%f"))
                    print ("Took",elapsed,"seconds to process",source," [",startsource,"] [",now,"]")
                except:
                    pass
            if (speedtest):
                print ("Saved individual files:",datetime.now()-startsource,"s")
                    
            # Trap error for missing filter data if object has not been correctly fit and OutputAppend==2
            if (handler != None):
                handler.submit_status(task_id, "processing", { "stage": 2, "stages": 4, "status": f"Finished processing {total_sources} { 'source' if total_sources == 1 else 'sources'}", "progress": 1, "step": total_steps, "totalSteps": total_steps })
            try:
                foo=len(filtdata)
            except UnboundLocalError:
                filtdata=get_filter_list()
            if ((savemasteroutput>0) and (filtdata.size>0) and (ancillary.size>0)):
                # Write out to master output file
                objectlist=np.empty(1,dtype="object")
                objectlist[0]=source.replace(" ","_")
                masterlist=np.squeeze(np.array([ancillary[ancillary['colname']=='RA']['value'],ancillary[ancillary['colname']=='Dec']['value'],ancillary[ancillary['colname']=='PMRA']['value'],ancillary[ancillary['colname']=='PMDec']['value']],dtype="a12"))
                mastererror=np.squeeze(np.array([ancillary[ancillary['colname']=='RA']['err'],ancillary[ancillary['colname']=='Dec']['err'],ancillary[ancillary['colname']=='PMRA']['err'],ancillary[ancillary['colname']=='PMDec']['err']],dtype="a12"))
                fittedlist=np.array([reducto(teff),reducto(lum)],dtype="str")
                fittederror=np.array([0,0],dtype="str")
                adoptedlist=np.array([dist,ebv,logg,feh],dtype="str")
                adoptederror=np.array([0,0,0,0],dtype="str")
                statlist=np.array([chisq,ruwe,gof,uvxs,irxs])
                filtlist=np.copy(filtdata['svoname'])
                catlist=np.copy(catdata['catname'])
                objlist=np.zeros(len(catlist),dtype="object")
                fluxlist=np.zeros(len(filtlist),dtype="object")
                ferrlist=np.zeros(len(filtlist),dtype="object")
                deredlist=np.zeros(len(filtlist),dtype="object")
                modellist=np.zeros(len(filtlist),dtype="object")
                if ((outputmasked>0) | (type(sed)==int)):
                    sedcopy=np.copy(sed)
                else:
                    sedcopy=np.copy(sed)[sed['mask']==True]
                for i in np.arange(len(catlist)):
                    cat=catlist[i]
                    try:
                        # Don't use sedcopy here as want to preserve IDs even if flux not used
                        objlist[i]=reducto(sed[sed['catname']==cat]['objid'])
                    except:
                        objlist[i]=""
                for i in np.arange(len(filtlist)):
                    filt=filtlist[i]
                    try:
                        fluxlist[i]=reducto(sedcopy[sedcopy['svoname']==filt]['flux'])
                    except:
                        fluxlist[i]=0.
                    try:
                        ferrlist[i]=reducto(sedcopy[sedcopy['svoname']==filt]['ferr'])
                    except:
                        ferrlist[i]=0.
                    try:
                        deredlist[i]=reducto(sedcopy[sedcopy['svoname']==filt]['dered'])
                    except:
                        deredlist[i]=0.
                    try:
                        modellist[i]=reducto(sedcopy[sedcopy['svoname']==filt]['model'])
                    except:
                        modellist[i]=0.
                if ((outputmasked>0) | (type(ancillary)==int)):
                    ancillarycopy=np.copy(ancillary)
                else:
                    ancillarycopy=np.copy(ancillary)[ancillary['mask']==True]
                ancillarylist=np.copy(ancillary_queries['paramname'])
                ancillarycats=np.copy(ancillary_queries['catname'])
                vallist=np.zeros(len(ancillarylist),dtype="object")
                errlist=np.zeros(len(ancillarylist),dtype="object")
                srclist=np.zeros(len(ancillarylist),dtype="object")
                for i in np.arange(len(ancillarylist)):
                    anc=ancillarylist[i]
                    try:
                        vallist[i]=reducta(ancillarycopy[ancillarycopy['parameter']==anc]['value'])
                    except:
                        vallist[i]=0.
                    try:
                        errlist[i]=reducta(ancillarycopy[ancillarycopy['parameter']==anc]['err'])
                    except:
                        errlist[i]=0.
                    try:
                        srclist[i]=reducta(ancillarycopy[ancillarycopy['parameter']==anc]['catname'])
                    except:
                        srclist[i]=0.
#                masteroutputdata=np.concatenate((objectlist,masterlist.astype(str),mastererror.astype(str),fittedlist,fittederror,adoptedlist,adoptederror,fluxlist,ferrlist,deredlist,modellist,vallist,errlist,srclist.astype(str)))
                masteroutputdata=np.concatenate((objectlist,masterlist.astype(str),mastererror.astype(str),fittedlist,fittederror,adoptedlist,adoptederror,statlist,objlist,fluxlist,ferrlist,deredlist,modellist,vallist,errlist,srclist))
                with open(outmasterfile, "a") as f:
                    f.write(sep.join(str(x) for x in masteroutputdata)+"\n")
                if (handler != None):
                    mastercsv = convert_master_file(outmasterfile)
                    results["masteroutput.csv"] = mastercsv

        sed=np.array([])
        ancillary=np.zeros([np.size(ancillary_queries)+5],dtype=[('parameter',object),('catname',object),('colname',object),('value',object),('err',object),('priority',object),('mask',bool)])
        if (speedtest):
            print ("Finished source:",datetime.now()-startsource,"s")
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # -------------------------------------------
    # Re-save ancillary data including fitted parameters
#    if (proctype!="none"):
    saveseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveSEDs",1][0])
    sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledSEDsFile",1])[2:-2]
    ancfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledAncFile",1])[2:-2]
    sourcefile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AreaCompiledSourceFile",1])[2:-2]
    if ((saveseds>0) and (searchtype=="area")):
        np.save(sedsfile,compiledseds)
        np.save(ancfile,compiledanc)
        np.savetxt(sourcefile,sourcedata,delimiter=sep,fmt="%s")

    # -------------------------------------------
    # Perform final plots if fitting multiple sources
    if (searchtype=="area"):
        if (verbosity>=20):
            print ("Creating final plots")
        globalpostplots(compiledseds,compiledanc,sourcedata)
    elif (usepreviousrun>=5):
        print_warn ("Nothing to do")

    if (verbosity>=20 or speedtest):
        try:
            now = datetime.now() # time object
            elapsed = float(now.strftime("%s.%f"))-float(startmain.strftime("%s.%f"))
            print ("Took",elapsed,"seconds for entire process [",startmain,"] [",now,"]")
        except:
            pass
    
    # Visualiser additions
    if (hrd_results.shape[0] > 0):
        results["hrd.dat"] = hrd_results
    if (handler != None):
        mastercsv = convert_master_file(outmasterfile)
        results["masteroutput.csv"] = mastercsv
        return results

    return errmsg

# -----------------------------------------------------------------------------
# If running from the command line
if (__name__ == "__main__"):
    # Parse command line arguments
    cmdargs=argv
    cmdtype,cmdparams,proctype,procparams,setupfile,error=parse_args(cmdargs)
    if (error!=0):
        exit(1)
    else:
        errmsg="Success!"
        errmsg=pyssed(cmdtype,cmdparams,proctype,procparams,setupfile,None,0,cmdargs)
        if (errmsg!="" and errmsg!="Success!"):
                print_fail ("ERROR!")
        if (errmsg=="Success!"):
            print_pass (errmsg)
        
