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
    speedtest=False
    from datetime import datetime               # Allows date and time to be printed
    if (speedtest):
        print ("start:",datetime.now(),"s")
    from sys import argv                        # Allows command-line arguments
    from sys import exit                        # Allows graceful quitting
    from sys import exc_info                    # Allows graceful error trapping
    if (speedtest):
        print ("sys:",datetime.now(),"s")
    import numpy as np                          # Required for numerical processing
    if (speedtest):
        print ("numpy:",datetime.now(),"s")
    import scipy.optimize as optimize           # Required for fitting
    import scipy.interpolate as interpolate     # Required for model grid interpolation
    if (speedtest):
        print ("scipy:",datetime.now(),"s")
    import astropy.units as u                   # Required for astropy/astroquery/dust_extinction interfacing
    from astropy.coordinates import SkyCoord    # Required for astroquery and B1950->J2000
    from astropy.coordinates import FK5         # Required B1950->J2000 conversion
    from astropy.io import votable              # Required for SVO interface
    if (speedtest):
        print ("astropy:",datetime.now(),"s")
    import matplotlib.pyplot as plt             # Required for plotting
    from matplotlib.colors import ListedColormap # Required for colour mapping
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes # Reqd. for subplots
    if (speedtest):
        print ("matplotlib:",datetime.now(),"s")

    import wget                                 # Required to download SVO filter files
    if (speedtest):
        print ("wget:",datetime.now(),"s")
    from astroquery.simbad import Simbad        # Allow SIMBAD queries
    from astroquery.vizier import Vizier        # Allow VizieR queries
    from astroquery.gaia import Gaia            # Allow Gaia queries
    if (speedtest):
        print ("astroquery:",datetime.now(),"s")

    import itertools                            # Required for iterating model data
    if (speedtest):
        print ("itetools:",datetime.now(),"s")
    import pandas as pd                         # Required for model data interpolation
    if (speedtest):
        print ("pandas:",datetime.now(),"s")

    from dust_extinction.parameter_averages import F99   # Adopted dereddening law
    import dustmaps                             # Needed for extinction correction
    if (speedtest):
        print ("dustmaps:",datetime.now(),"s")
    import beast                                # Needed for Bayesian fitting
    if (speedtest):
        print ("beast:",datetime.now(),"s")
except:
    print ("PySSED! Problem importing modules. Additional information:")
    print (exc_info())
    print ("-----------------------------------------------------")
    print ("The following modules are REQUIRED:")
    print ("   sys numpy scipy astropy pandas itertools")
    print ("   matplotlib mpl_toolkits wget astroquery")
    print ("The following modules are REQUIRED for some compoents")
    print ("   dust_extinction - for any reddening correction")
    print ("   dust_maps - for 2D/3D dust extinction")
    print ("   beast - for full Bayesian fits")
    print ("The following modules are OPTIONAL:")
    print ("   datetime - to correctly display timing information")
    print ("PySSED will try to valiantly soldier on regardless...")
    print ("-----------------------------------------------------")

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
        print ("<search type> : box, parameters = RA1, Dec1, RA2, Dec2 (deg)")
        print ("<search type> : volume, parameters = RA, Dec, d, r (deg,pc)")
        print ("<search type> : criteria, parameters = 'SIMBAD criteria setup file'")
        print ("<search type> : complex, parameters = 'Gaia criteria setup file'")
        print ("<search type> : nongaia, parameters = 'Non-gaia criteria setup file'")
        print ("<search type> : uselast")
        print ("<processing type> : none")
        print ("<processing type> : simple, parameters = Fe/H, E(B-V), [mass]")
        print ("<processing type> : fit, parameters = [priors file]")
        print ("<processing type> : binary, parameters = [priors file]")
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
        if (cmdargs[1]=="single"):
            print ("(1) Process single object:",cmdargs[2])
            cmdparams=cmdargs[2]
            procargs=cmdargs[3:]
        elif (cmdargs[1]=="list"):
            print ("(1) Process a list of objects")
            procargs=cmdargs[3:]
            cmdparams=cmdargs[2]
            print ("    List of targets in:",cmdparams)
        elif (cmdargs[1]=="cone"):
            print ("(1) Process a cone search",cmdargs[2],cmdargs[3],cmdargs[4])
            cmdparams=cmdargs[2:4]
            procargs=cmdargs[5:]
        elif (cmdargs[1]=="box"):
            print ("(1) Process a box search:",cmdargs[2],cmdargs[3],"-",cmdargs[4],cmdargs[5])
            cmdparams=cmdargs[2:5]
            procargs=cmdargs[6:]
        elif (cmdargs[1]=="volume"):
            print ("(1) Process a volume:",cmdargs[2],cmdargs[3],cmdargs[4],cmdargs[5])
            cmdparams=cmdargs[2:5]
            procargs=cmdargs[6:]
        elif (cmdargs[1]=="criteria"):
            print ("(1) Process a set of SIMBAD criteria")
            cmdparams=cmdargs[2]
            procargs=cmdargs[3:]
            print ("    SIMBAD query in:",cmdparams)
        elif (cmdargs[1]=="complex"):
            print ("(1) Process a complex Gaia query")
            cmdparams=cmdargs[2]
            procargs=cmdargs[3:]
            print ("    Gaia query in:",cmdparams)
        elif (cmdargs[1]=="nongaia"):
            print ("(1) Process a set of objects from a non-Gaia source set")
            cmdparams=cmdargs[2]
            procargs=cmdargs[3:]
            print ("    Setup file:",cmdparams)
        elif (cmdargs[1]=="uselast"):
            print ("(1) Use the last set of downloaded data")
            cmdparams=[]
            procargs=cmdargs[3:]
            print ("    Setup file:",cmdparams)
        else:
            print ("ERROR! Search type was:",cmdargs[1])
            print ("Expected one of: single, list, cone, box, volume, criteria, complex, nongaia, uselast")
            cmdtype=""
            cmdparams=[]
            procargs=[]
            proctype=""
            procparams=[]
            error=1
    else:
        print ("ERROR! No command type specified")
        print ("Expected one of: single, list, cone, box, volume, criteria, complex, nongaia")
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
            print ("ERROR! Processing type was:",procargs[0])
            print ("Expected one of: sed, bb, simple, fit, binary")
            proctype=""
            procparams=[]
            error=1
    else:
        print ("ERROR! No processing type specified")
        print ("Expected one of: sed, bb, simple, fit, binary")
        proctype=""
        procargs=[]
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
    poserrmult=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PosErrMult",1])
    catdata = np.loadtxt(catfile, dtype=[('server',object),('catname',object),('cdstable',object),('epoch',float),('beamsize',float),('matchr',float)], comments="#", delimiter="\t", unpack=False)
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
    poserrmult=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PosErrMult",1])
    ancillarydata = np.loadtxt(ancillaryfile, dtype=[('server',object),('catname',object),('cdstable',object),('colname',object),('errname',object),('paramname',object),('multiplier',object),('epoch',float),('beamsize',float),('matchr',float),('priority',int)], comments="#", delimiter="\t", unpack=False)
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
    recastmodels=int(pyssedsetupdata[pyssedsetupdata[:,0]=="RecomputeModelGrid",1])
    if (recastmodels):
        get_model_list()
    
    modelfile="model-"+np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelCode",1])[2:-2]+"-recast.dat"
    if (verbosity>=30):
        print ("Using model file:", modelfile)
    modeldata = np.genfromtxt(modelfile, comments="#", names=True, deletechars=" ~!@#$%^&*()=+~\|]}[{';: ?>,<")

    return modeldata

# -----------------------------------------------------------------------------
def get_model_list():
    # Translate list of model photometry into four-dimensional grid
    # This takes some time to do, but results in quicker interpolation later

    if (verbosity>=5):
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print ("Recasting model grid [RecomputeModelGrid>0].")
        print ("Have a coffee break - this may take some time (est. a fraction of an hour per filter).")
        print ("If this takes too long, try restricting the Model*Hi and Model*Lo parameters in the setup file.")

    # Load list of reduced models
    modelfile="model-"+np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelCode",1])[2:-2]+".dat"
    if (verbosity>=30):
        print ("Using model file:", modelfile)
    modelfiledata = np.genfromtxt(modelfile, comments="#", names=True, deletechars=" ~!@#$%^&*()=+~\|]}[{';: ?>,<")

    # Remove outlying models
    modeltefflo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelTeffLo",1])
    modelteffhi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelTeffHi",1])
    modellogglo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelLoggLo",1])
    modellogghi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelLoggHi",1])
    modelfehlo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelFeHLo",1])
    modelfehhi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelFeHHi",1])
    modelafelo=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelAFeLo",1])
    modelafehi=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelAFeHi",1])
    modelfiledata = modelfiledata[(modelfiledata['teff']>=modeltefflo) & (modelfiledata['teff']<=modelteffhi)]
    modelfiledata = modelfiledata[(modelfiledata['logg']>=modellogglo) & (modelfiledata['logg']<=modellogghi)]
    modelfiledata = modelfiledata[(modelfiledata['metal']>=modelfehlo) & (modelfiledata['metal']<=modelfehhi)]
    modelfiledata = modelfiledata[(modelfiledata['alpha']>=modelafelo) & (modelfiledata['alpha']<=modelafehi)]

    # Separate parameters and values
    params=np.stack((modelfiledata['teff'],modelfiledata['logg'],modelfiledata['metal'],modelfiledata['alpha']),axis=1)
    valueselector = modelfiledata.dtype.names[4:]

    # Iterate parameters onto complete grid
    interp_grid_points = np.array(list(itertools.product(np.unique(modelfiledata['teff']),np.unique(modelfiledata['logg']),np.unique(modelfiledata['metal']),np.unique(modelfiledata['alpha']))))
    
    # Set up output data grid
    interp_data_points = np.zeros((len(interp_grid_points[:,0]),len(valueselector)),dtype=float)
    #print (np.shape(interp_data_points))
    
    try:
        start = datetime.now().time() # time object
    except:
        start = 0
    # Recast onto rectilinear grid
    for i in np.arange(len(valueselector)):
        if (verbosity>=30):
            try:
                now = datetime.now().time() # time object
                elapsed = float(now.strftime("%s.%f"))-float(start.strftime("%s.%f"))
                remaining = elapsed/(i+1e-6)*(len(valueselector)-i)
            except:
                now = 0; elapsed = 0; remaining = 0
            print ("Processing filter", i+1, "of", len(valueselector), "[",now,"], elapsed:",int(elapsed)," remaining:",int(remaining),"sec")
        values=modelfiledata[:][list(valueselector)[i]]
        interpfn = interpolate.griddata(params,values,interp_grid_points, method='linear', rescale=True, fill_value=0.)
        df = pd.DataFrame(interpfn)
        modeldata = df.interpolate().to_numpy()
        interp_data_points[:,i]=np.squeeze(modeldata)
    
    if (verbosity>=30):
        print ("Done. Saving recast model file.")
    recastmodelfile="model-"+np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelCode",1])[2:-2]+"-recast.dat"
    with open(modelfile, "r") as f:
        header=f.readline()
    with open(recastmodelfile, "w") as f:
        f.write(header)
    with open(recastmodelfile, "a") as f:
        np.savetxt(f,np.append(interp_grid_points,interp_data_points,axis=1), fmt='%s', delimiter=' ')
    if (verbosity>=5):
        print ("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

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
    # Extract the Gaia EDR3 ID from the input data

    errmsg=""
    edr3_data=""
    edr3_obj=""
    if (verbosity>=30):
        print ("Fitting single object")
    
    # Parse input parameters: convert to array and try to extract co-ordinates
    obj=np.str.split(cmdparams,sep=" ")

    # If specific Gaia keywords (EDR3,DR2) are used, use the source name
    if (obj[0]=="EDR3" or obj[0]=="Gaia" or obj[0]=="DR2"):
        # If using a DR2 identifier, first cross-match to an EDR3 source
        if (obj[0]=="DR2"):
            dr2_obj=obj[1]
            query="SELECT dr3_source_id FROM gaiaedr3.dr2_neighbourhood WHERE dr2_source_id=" + obj[1]
            job = Gaia.launch_job(query)
            result_table=job.get_results()
            edr3_obj=result_table['dr3_source_id'][0]
            if (verbosity>40):
                print ("Gaia DR2 source",obj[1],"is EDR3 source",edr3_obj)
        else:
            edr3_obj=obj[1]
            #coords=np.zeros((2,),dtype=float)
            if (verbosity>40):
                print ("Using Gaia EDR3 source",obj[1])

    # Otherwise, if parsable co-ordinates are used, use those
    else:
        try:
            coords=np.fromstring(cmdparams,dtype=float,sep=" ")
            if (coords[0]!=-999 and coords[1]!=-999 and verbosity>40):
                print ("Using co-ordates: RA",coords[0],"deg, Dec",coords[1],"deg")
        except:
        # and if not, then ask SIMBAD for a position resolution
            if (verbosity>40):
                print ("Resolving",cmdparams,"using SIMBAD")
            attempts = 0
            while attempts < 3:
                try:
                    result_table = Simbad.query_object(cmdparams)
                    break
                except:
                    attempts += 1
                    if (verbosity >= 25):
                        print ("Could not connect to SIMBAD server (attempt",attempts,"of 3)")
            if (verbosity>60):
                print (result_table['RA'][0],result_table['DEC'][0])
            rac=np.fromstring(result_table['RA'][0],dtype=float,sep=" ")
            decc=np.fromstring(result_table['DEC'][0],dtype=float,sep=" ")
            if (result_table['DEC'][0][0]=="+"):
                coords=[rac[0]*15.+rac[1]/4+rac[2]/240,decc[0]+decc[1]/60+decc[2]/3600]
            else:
                coords=[rac[0]*15.+rac[1]/4+rac[2]/240,decc[0]-decc[1]/60-decc[2]/3600]
            if (verbosity>40):
                print ("Using co-ordinates: RA",coords[0],"deg, Dec",coords[1],"deg")

        # Now query the Gaia EDR3 database for a match
        edr3_obj=query_gaia_coords(coords[0],coords[1])
        # If there's a match...
        if (edr3_obj>0):
            if (verbosity>40):
                print ("Gaia EDR3",edr3_obj)
        # If there isn't a match, fall back to Hipparcos
        else:
            if (verbosity>40):
                print ("No Gaia object at that position.")

    #if (verbosity>=40):
    #    print (edr3_data)

    return edr3_obj,errmsg
    
# -----------------------------------------------------------------------------
def get_gaia_data(edr3_obj):
    query="select * from gaiaedr3.gaia_source where source_id=" + str(edr3_obj)
    attempts = 0
    while attempts < 3:
        try:
            job = Gaia.launch_job(query)
            break
        except:
            attempts += 1
            if (verbosity >= 25):
                print ("Could not connect to Gaia server (attempt",attempts,"of 3)")
    return job.get_results()

# -----------------------------------------------------------------------------
def get_gaia_list(cmdparams):
    # Extract data on a list of disparate single objects
    # As get_gaia_single
    # Can be multiplexed one object at a time, subject to not triggering rampaging robot alerts
    return

# -----------------------------------------------------------------------------
def get_gaia_cone(cmdparams):
    # Extract data in a cone around a position
    # Includes all objects within an area
    # Similar cone search performed on other catalogues
    # Performs automatic best-match photometry (with or without deblending)
    # Can be multiplexed by area
    # Option to restrict sources by proper motion and/or parallax
    # Limit could be inclusive or exclusive of errors, or a simple mean
    return

# -----------------------------------------------------------------------------
def get_gaia_box(cmdparams):
    # Extract data in a box around a position
    # As fit_cone, but for a box
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
# -----------------------------------------------------------------------------
def fit_nongaia(cmdparams):
    # Extract data from a list of sources not from Gaia
    # Start from a catalogue other than Gaia (e.g. HST, WISE, own observations)
    # Progressive match to Gaia catalogue, then match outwards
    return




# =============================================================================
# SERVER TABLE QUERIES
# =============================================================================
def extract_ra_dec(vizier_data):

    # Extract RA and Dec from a Vizier table
    try:
        newra=vizier_data[0]['RArad'][0]
        newdec=vizier_data[0]['DErad'][0]
        if (verbosity>=98):
            print ("Selected co-ordinates from: RArad/DErad")
    except: # Needed for Tycho
        try:
            newra=vizier_data[0]['_RA.icrs'][0]
            newdec=vizier_data[0]['_DE.icrs'][0]
            if (verbosity>=98):
                print ("Selected co-ordinates from: _RA.icrs/_DE.icrs")
        except: # Needed for SDSS
            try:
                newra=vizier_data[0]['RA_ICRS'][0]
                newdec=vizier_data[0]['DE_ICRS'][0]
                if (verbosity>=98):
                    print ("Selected co-ordinates from: RA_ICRS/DE_ICRS")
            except:
                try:
                    newra=vizier_data[0]['RAJ2000'][0]
                    newdec=vizier_data[0]['DEJ2000'][0]
                    if (verbosity>=98):
                        print ("Selected co-ordinates from: RAJ2000/DEJ2000")
                except: # Needed for Morel
                    try:
                        newra=vizier_data[0]['_RA'][0]
                        newdec=vizier_data[0]['_DE'][0]
                        if (verbosity>=98):
                            print ("Selected co-ordinates from: _RA/_DE")
                    except: # Needed for IRAS
                        try:
                            # Extract
                            newra=vizier_data[0]['RA1950'][0]
                            newdec=vizier_data[0]['DE1950'][0]
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
                        except: # Needed for Skymapper
                            try:
                                newra=vizier_data[0]['RAICRS'][0]
                                newdec=vizier_data[0]['DEICRS'][0]
                                if (verbosity>=98):
                                    print ("Selected co-ordinates from: RAICRS,DEICRS")
                            except: # Resort to pre-computed values, which undoes PM correction
                                try:
                                    newra=vizier_data[0]['_RAJ2000'][0]
                                    newdec=vizier_data[0]['_DEJ2000'][0]
                                    if (verbosity>=98):
                                        print ("Selected co-ordinates from: _RAJ2000/_DEJ2000")
                                except:
                                    print ("Failed to find a co-ordinate system")
    return newra,newdec

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def query_gaia_coords(ra,dec):
    # Query the Gaia archive for a single object
    # Format the co-ordinates
    coord = SkyCoord(ra, dec, unit=(u.degree, u.degree), frame='icrs')
    # Select only one object
    Gaia.ROW_LIMIT = 1
    # Get the default search radius from the setup data
    coneradius=float(pyssedsetupdata[pyssedsetupdata[:,0]=="GaiaCone",1])
    width = u.Quantity(coneradius, u.arcsec)
    height = u.Quantity(coneradius, u.arcsec)
    # Get the result
    result=Gaia.query_object_async(coordinate=coord, width=width, height=height, verbose=False)
    # Trap null result
    try:
        edr3_obj=result['source_id'][0]
    except:
        edr3_obj=0

    return edr3_obj

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def get_vizier_single(cmdparams,sourcedata):

    if (verbosity>60):
        print ("Getting VizieR data...")
    if (verbosity>70):
        print (sourcedata[0])

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
    ancillary=np.zeros([np.size(ancillary_queries)+4],dtype=[('parameter',object),('catname',object),('colname',object),('value',object),('err',object),('priority',object),('mask',bool)])
    
    # Get the parameters from the source data
    # Assuming Gaia first
    try:
        if (verbosity>80):
            print ("Trying Gaia...")
        sourcetype="Gaia"
        sourcera=float(sourcedata['ra'][0])
        sourcedec=float(sourcedata['dec'][0])
        sourceraerr=float(sourcedata['ra_error'][0])/3600000.
        sourcedecerr=float(sourcedata['dec_error'][0])/3600000.
        if ((sourcedata['pmra'][0]!="--") and (sourcedata['pmdec'][0]!="--")):
            sourcepmra=float(sourcedata['pmra'][0])
            sourcepmdec=float(sourcedata['pmdec'][0])
            sourcepmraerr=float(sourcedata['pmra_error'][0])
            sourcepmdecerr=float(sourcedata['pmdec_error'][0])
        else:
            sourcepmra=0.
            sourcepmdec=0.
            sourcepmraerr=0.
            sourcepmdecerr=0.
        sourceepoch=float(pyssedsetupdata[pyssedsetupdata[:,0]=="GaiaEpoch",1])
        if (verbosity>80):
            print ("Gaia data found.")
    # If that fails, try Hipparcos
    except:
        try:
            if (verbosity>80):
                print ("Trying Hipparcos...")
            sourcetype="Hipparcos"
            sourcera=float(sourcedata[0]['RArad'][0])
            sourcedec=float(sourcedata[0]['DErad'][0])
            sourceraerr=float(sourcedata[0]['e_RArad'][0])/3600000.
            sourcedecerr=float(sourcedata[0]['e_DErad'][0])/3600000.
            sourcepmra=float(sourcedata[0]['pmRA'][0])
            sourcepmdec=float(sourcedata[0]['pmDE'][0])
            sourcepmraerr=float(sourcedata[0]['e_pmRA'][0])
            sourcepmdecerr=float(sourcedata[0]['e_pmDE'][0])
            sourceepoch=1991.25
            if (verbosity>80):
                print ("Hipparcos data found.")
        except:
            if (verbosity>80):
                print ("Resorting to user-specified data.")
            sourcetype="User"
            rac=np.fromstring(sourcedata['RA'][0],dtype=float,sep=" ")
            decc=np.fromstring(sourcedata['DEC'][0],dtype=float,sep=" ")
            sourcera=rac[0]*15.+rac[1]/4+rac[2]/240
            sourcedec=abs(decc[0])/decc[0]*(abs(decc[0])+decc[1]/60+decc[2]/3600)
            sourceraerr=0.
            sourcedecerr=0.
            sourcepmra=0.
            sourcepmdec=0.
            sourcepmraerr=0.
            sourcepmdecerr=0.
            sourceepoch=2000.
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

    # Set the default photometric error and get the PM astrometry error controls
    defaulterr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultError",1])
    addpmerr=int(pyssedsetupdata[pyssedsetupdata[:,0]=="AddPMErr",1])
    pmerrtime=float(pyssedsetupdata[pyssedsetupdata[:,0]=="EpochPMErrYears",1])

    # Loop over catalogues
    catalogues=catdata['catname']
    sed=np.zeros((len(filtdata)),dtype=[('catname','<U20'),('ra','f4'),('dec','f4'),('modelra','f4'),('modeldec','f4'),('svoname','U32'),('filter','U10'),('wavel','f4'),('dw','f4'),('mag','f4'),('magerr','f4'),('flux','f4'),('ferr','f4'),('dered','f4'),('derederr','f4'),('model','f4'),('mask','bool')])
    nfsuccess=0
    for catalogue in catalogues:
        if (verbosity>60):
            print ("Catalogue:",catalogue)
        server=catdata[catdata['catname']==catalogue]['server']
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
                    sed[nfsuccess]=(catalogue,(newra-sourcera)*3600.,(newdec-sourcedec)*3600.,(ra-sourcera)*3600.,(dec-sourcedec)*3600.,svokey,magkey,wavel,dw,mag,err,flux,ferr,0,0,0,1)
                    nfsuccess+=1
        else:
            ra,dec=pm_calc(sourcera,sourcedec,sourcepmra,sourcepmdec,sourceepoch,float(catdata[catdata['catname']==catalogue]['epoch']))
            if (server=="Vizier"):
                # Correct proper motion and query VizieR
                matchr=float(catdata[catdata['catname']==catalogue]['matchr'])
                if (addpmerr > 0):
                    matchr+=np.sqrt(sourcepmraerr**2+sourcepmdecerr**2)/1000.*np.abs(sourceepoch-float(catdata[catdata['catname']==catalogue]['epoch']))
                    matchr+=np.sqrt(sourcepmra**2+sourcepmdec**2)/1000.*pmerrtime
                vizier_data=query_vizier_cone(str(catdata[catdata['catname']==catalogue]['cdstable'])[2:-2],ra,dec,matchr)

                if (verbosity>70):
                    print (vizier_data)

                # Only proceed if VizieR has returned some data
                if (len(vizier_data)>0):
                    if (verbosity>80):
                        print (vizier_data[0])
                    # Get RA and Dec from various columns in order of preference
                    newra,newdec=extract_ra_dec(vizier_data)
                    if (verbosity>98):
                        print ("Source astrometry:")
                        print ("Source epoch:",float(catdata[catdata['catname']==catalogue]['epoch']))
                        print ("Source RA:",ra,"->",newra,"(",(ra-newra)*3600.,"arcsec)")
                        print ("Source Dec:",dec,"->",newdec,"(",(dec-newdec)*3600.,"arcsec)")
                        print ("Search radius:",matchr,"arcsec")
            else: # Data from file
                photfile=str(catdata[catdata['catname']==catalogue]['cdstable'])[2:-2]
                matchr=float(catdata[catdata['catname']==catalogue]['matchr'])
                if (addpmerr > 0):
                    matchr+=np.sqrt(sourcepmraerr**2+sourcepmdecerr**2)/1000.*np.abs(sourceepoch-float(catdata[catdata['catname']==catalogue]['epoch']))
                    matchr+=np.sqrt(sourcepmra**2+sourcepmdec**2)/1000.*pmerrtime
                phot=np.genfromtxt(photfile, delimiter='\t', names=True)
                phot=phot[phot['RA']>=0]
                c=SkyCoord(ra=ra*u.degree,dec=dec*u.degree)
                catcoords=SkyCoord(ra=phot['RA']*u.degree,dec=phot['Dec']*u.degree)
                idx,d2d,d3d=c.match_to_catalog_sky(catcoords)
                if (d2d.arcsec<matchr):
                    vizier_data=np.expand_dims(np.expand_dims(phot[idx],axis=0),axis=1)
                else:
                    vizier_data=[]
                if (verbosity>60):
                    print ("CATALOGUE = ",catalogue,"; RA,DEC =",ra,dec)
                    print ("Vizier data:",vizier_data)
            if (len(vizier_data)>0): # If any data exists
                svokeys=filtdata[filtdata['catname']==catalogue]['svoname']
                # Identify magnitude and error columns
                magkeys=filtdata[filtdata['catname']==catalogue]['filtname']
                errkeys=filtdata[filtdata['catname']==catalogue]['errname']
                datatypes=filtdata[filtdata['catname']==catalogue]['datatype']
                datarefs=filtdata[filtdata['catname']==catalogue]['dataref']
                errtypes=filtdata[filtdata['catname']==catalogue]['errtype']
                mindatas=filtdata[filtdata['catname']==catalogue]['mindata']
                maxdatas=filtdata[filtdata['catname']==catalogue]['maxdata']
                maxperr=filtdata[filtdata['catname']==catalogue]['maxperr']
                # And extract them from vizier_data
                for i in (np.arange(len(magkeys))):
                    magkey=magkeys[i]
                    mag=vizier_data[0][magkey][0]
                    mindata=mindatas[i]
                    maxdata=maxdatas[i]
                    # Then check it's good by rejecting anything nasty
                    reasons=rejectdata[(rejectdata['catname']==catalogue) & ((rejectdata['filtname']==magkey) | (rejectdata['filtname']=="All")) & (rejectdata['rejcat']=="Same") & (rejectdata['rejcol']=="Same")]
                    vizier_data,mask=reject_test(reasons,vizier_data)
                    # If mag is within the limits specified
                    if (np.isfinite(mag)):
                        if ((mag >= mindata) & (mag <= maxdata)):
                            errkey=errkeys[i]
                            if (errkey=="None"):
                                err=defaulterr
                            else:
                                err=vizier_data[0][errkey][0]
                            svokey=svokeys[i]
                            wavel=float(svodata[svodata['svoname']==svokey]['weff'][0])
                            dw=float(svodata[svodata['svoname']==svokey]['dw'][0])
                            datatype=datatypes[i]
                            errtype=errtypes[i]
                            # Detect Vega or AB magnitudes
                            if (datarefs[i]=='Vega'):
                                zpt=float(svodata[svodata['svoname']==svokey]['zpt'][0])
                            else:
                                zpt=3631.
                            # Detect magnitudes or fluxes
                            if (datatype=='mag'):
                                if (datarefs[i]=='Vega'):
                                    zpt=float(svodata[svodata['svoname']==svokey]['zpt'][0])
                                else:
                                    zpt=3631.
                                flux=10**(mag/-2.5)*zpt
                                if (errtype=='Same'):
                                    ferr=flux-10**((mag+err)/-2.5)*zpt
                                else: # Perc
                                    ferr=flux*err/100.
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
                                mag=-2.5*np.log10(flux/zpt)
                                if (ferr>0):
                                    magerr=2.5*np.log10(1+ferr/flux)
                                else:
                                    magerr=0.
                            # If the fractional error in the flux is sufficiently small
                            if (ferr/flux<maxperr[i]/100.):
                                sed[nfsuccess]=(catalogue,(newra-sourcera)*3600.,(newdec-sourcedec)*3600.,(ra-sourcera)*3600.,(dec-sourcedec)*3600.,svokey,magkey,wavel,dw,mag,err,flux,ferr,0,0,0,mask)
                                nfsuccess+=1

    # If there is no error in flux, use default error
    sed[sed['ferr']==0]['ferr']==sed[sed['ferr']==0]['flux']*defaulterr

    # Get ancillary information
    if (verbosity>50):
        print ("Getting ancillary data...")
    for i in np.arange(np.size(ancillary_queries)):
        if (verbosity>60):
            print (ancillary_queries[i])
        ra,dec=pm_calc(sourcera,sourcedec,sourcepmra,sourcepmdec,sourceepoch,float(ancillary_queries[i]['epoch']))
        if (ancillary_queries[i]['server']=="Vizier"):
            # Correct proper motion and query VizieR
            matchr=float(ancillary_queries[i]['matchr'])
            if (addpmerr > 0):
                matchr+=np.sqrt(sourcepmraerr**2+sourcepmdecerr**2)/1000.*np.abs(sourceepoch-float(catdata[catdata['catname']==catalogue]['epoch']))
                matchr+=np.sqrt(sourcepmra**2+sourcepmdec**2)/1000.*pmerrtime
            vizier_data=query_vizier_cone(str(ancillary_queries[i]['cdstable']),ra,dec,matchr)

            if (verbosity>70):
                print ("CATALOGUE = ",str(ancillary_queries[i]['cdstable']),"; RA,DEC =",ra,dec)
                print (vizier_data)

            # Only proceed if VizieR has returned some data
            if (len(vizier_data)>0):
                if (verbosity>80):
                    print (vizier_data)
                # Get RA and Dec from various columns in order of preference
                newra,newdec=extract_ra_dec(vizier_data)
                reasons=rejectdata[(rejectdata['catname']==ancillary_queries[i]['catname']) & ((rejectdata['filtname']==ancillary_queries[i]['colname']) | (rejectdata['filtname']=="All")) & (rejectdata['rejcat']=="Same") & (rejectdata['rejcol']=="Same")]
                vizier_data,mask=reject_test(reasons,vizier_data)
                if (ancillary_queries[i]['errname']=="None"):
                    err=0
                elif ("/" in ancillary_queries[i]['errname']): # allow upper/lower limits
                    errs=np.str.split(ancillary_queries[i]['errname'],sep="/")
                    err=(vizier_data[0][errs[0]][0]-vizier_data[0][errs[1]][0])/2.
                else:
                    err=vizier_data[0][ancillary_queries[i]['errname']][0]
                ancillary[i+4]=(ancillary_queries[i]['paramname'],ancillary_queries[i]['catname'],ancillary_queries[i]['colname'],vizier_data[0][ancillary_queries[i]['colname']][0],err,ancillary_queries[i]['priority'],mask)
        else:
            raise ("ERROR! CAN ONLY GET ANCILLARY DATA FROM VIZIER FOR NOW, NOT FILES.")
    #magkeys=filtdata[filtdata['catname']==catalogue]['filtname']
    #ancillarydata = np.loadtxt(ancillaryfile, dtype=[('server',object),('catname',object),('cdstabl',object),('colname',object),('errname',object),('paramname',object),('multiplier',object),('epoch',float),('beamsize',float),('matchr',float)], comments="#", delimiter="\t", unpack=False)

    # Apply additional rejection criteria
    # -----------------------------------

    # Reject if wavelength too short or long
    minlambda=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MinLambda",1])*10000.
    maxlambda=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxLambda",1])*10000.
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
                        sed[n]['flux']=np.where(n==True,0,sed[:]['flux'])
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
                print (">>>",testdata,logic,compmag,compflux,logictest,action)
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
    
    return sed[sed['flux']>0.],ancillary[ancillary['parameter']!=0],errmsg

# -----------------------------------------------------------------------------
def reject_test(reasons,vizier_data):

    mask=1
    for i in np.arange(len(reasons)):
        # Identify data to test
        if (reasons[i]['position']>=0):
            testdata=vizier_data[0][reasons[i]['column']][0][reasons[i]['position']]
        else:
            testdata=vizier_data[0][reasons[i]['column']][0]
        # Identify whether action needs taken
        action=0
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
                if (verbosity>80):
                    print ("Masked",reasons[i])
            else:
                vizier_data[0][reasons[i]['filtname']][0]=-9.e99 # Set to stupidly low flux/mag
        else:
                if (verbosity>80):
                    print ("Not masked",reasons[i])

    return vizier_data,mask
 
# -----------------------------------------------------------------------------
def reject_logic(testdata,compmag,compflux,logic,logictest):

    action=0
    if (len(testdata)>0):
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
    if (len(testdata)>0):
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
def query_vizier_cone(cat,ra,dec,r):
    # Query the TAP server at CDS
    maxrows=int(pyssedsetupdata[pyssedsetupdata[:,0]=="VizierRowLimit",1])
    attempts = 0
    while attempts < 3:
        try:
            result = Vizier(columns=["**","_RAJ2000","_DEJ2000","+_r"],row_limit=maxrows).query_region(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'), radius=r*u.arcsec, catalog=cat)
            break
        except:
            attempts += 1
            if (verbosity >= 25):
                print ("Could not connect to VizieR server (attempt",attempts,"of 3)")
    
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
    newra=ra+np.cos(np.deg2rad(dec))*pmra*dt/3600.e3
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

    edr3_obj=0
    edr3_data=[]
    edr3_obj,errmsg=get_gaia_obj(cmdparams)
    # If a Gaia EDR3 cross-match can be found...
    if (edr3_obj!=""):
        # Query the Gaia EDR3 database for data
        edr3_data=get_gaia_data(edr3_obj)
        # Parse setup data: am I saving this file?
        savegaia=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SavePhot",1])
        photfile=str(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotFile",1])[2:-2]
        if (savegaia>0):
            if (verbosity>=60):
                print ("Saving data to",photfile)
            np.save(photfile,edr3_data)
    # Get data on other filters
    if (len(edr3_data)>0):
        if (verbosity>50):
            print ("Getting SED based on Gaia cross-reference")
        sed,ancillary,errmsg=get_vizier_single(cmdparams,edr3_data)
    if ((edr3_obj == 0) | (len(edr3_data)==0)):
        if (verbosity>=40):
            print ("Could not find a Gaia EDR3 cross-match.")
        # Get Hipparcos identifier from SIMBAD
        result_table = Simbad.query_objectids(cmdparams)
        hip="00000"
        for i in np.arange(len(result_table['ID'])):
            if ('HIP' in result_table['ID'][i]):
                hip=result_table['ID'][i]
        if (int(hip[4:])>0):
            # Query Hipparcos identifier for data
            if (verbosity>40):
                print ("Hipparcos id:",hip)
                attempts = 0
                while attempts < 3:
                    try:
                        vizquery=Vizier(columns=["**"]).query_object(hip, catalog="I/311/hip2")
                        break
                    except:
                        attempts += 1
                        if (verbosity >= 25):
                            print ("Could not connect to Gaia server (attempt",attempts,"of 3)")
        try:
            if (verbosity>50):
                print ("Getting SED based on Hipparcos cross-reference")
            sed,ancillary,errmsg=get_vizier_single(cmdparams,vizquery)
        except: # If no HIP source and/or data
            # Just use co-ordinates: try parsing co-ordinates first, then ask SIMBAD
            if (verbosity>40):
                print ("No Gaia or Hipparcos ID: using fixed co-ordinates only.")
            try:
                coords=np.fromstring(cmdparams,dtype=float,sep=" ")
                if (coords[0]>=0 and coords[0]<=360 and coords[1]>=-90 and coords[1]<=90):
                    if (verbosity>60):
                        print (coords)
            except:
                result_table = Simbad.query_object(cmdparams)
            sed,ancillary,errmsg=get_vizier_single(cmdparams,result_table)
            
    return sed,ancillary,errmsg
    
# -----------------------------------------------------------------------------
def merge_sed(sed):
    # Merge the flux at a specific wavelength, based either on mean, weighted mean or preferred catalogue
    
    minerr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MinPhotError",1])
    wtlimit=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotWeightingLimit",1])
    sigmalimit=float(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotSigmaLimit",1])
    
    for filt in np.unique(sed['svoname']):
        # If multiple observations in the same filter...
        if (len(sed[sed['svoname']==filt])>1):
            if (verbosity>80):
                print ("Merging photometry in",filt)
                print (sed[sed['svoname']==filt])
            # Extract fluxes and errors
            flux=sed[sed['svoname']==filt]['flux']
            ferr=sed[sed['svoname']==filt]['ferr']
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
                    mergeable=sed[sed['svoname']==filt][mergewt*mergesigma>0]
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
                    print ("Cannot be merged")
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
    
    minerr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MinAncError",1])
    wtlimit=float(pyssedsetupdata[pyssedsetupdata[:,0]=="AncWeightingLimit",1])
    sigmalimit=float(pyssedsetupdata[pyssedsetupdata[:,0]=="AncSigmaLimit",1])

    # Reject null values
    # XXX Try to look at masked elements, rather than just zero
    ancillary['mask']=np.where(ancillary['value'].astype(bool)==False,False,ancillary['mask'])
    
    for param in np.unique(ancillary['parameter']):
        # If multiple observations in the same filter...
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
        # If there are still multiple observations in the same filter...
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
                err=np.where(err/value<minerr,minerr*value,error)
                # Calculate mean/median
                if (len(value)==2):
                    mvalue=np.average(value)
                else:
                    mvalue=np.median(value)
                # Can these be merged within the weighting tolerance?
                wt=min(err/value)/(err/value)
                mergewt=np.where(wt>wtlimit,True,False)
                # Can these be merged within the sigma tolerance?
                sigma=abs(value-mvalue)/err
                mergesigma=np.where(sigma<sigmalimit,True,False)
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
                    print ("Cannot be merged")
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
    for catname in np.unique(cats['catname']):
        cat=cats[cats['catname']==catname]['cdstable'][0]
        beamsize=cats[cats['catname']==catname]['beamsize'][0]
        try: # If data
            if (cmdargs[1]=="cone"):
                catdata=query_vizier_cone(cat,float(cmdargs[2]),float(cmdargs[3]),float(cmdargs[4])*3600.+beamsize)
            nobj=len(catdata[0]) # intentional fail here if EmptyTable
            photprocfile=outdir+catname+".npy"
            if (verbosity >=50):
                print ("Queried",catname,"- found",len(catdata[0]),"objects ->",photprocfile)
            np.save(photprocfile,catdata[0])
        except: # If no data
            if (verbosity >=90):
                print ("Queried",catname,"- found no objects")
            pass
    return

# -----------------------------------------------------------------------------
def get_sed_multiple():
    # Generate a set of SEDs from spatially overlapping data
    # - Get data from Gaia
    # - Get data from CDS
    # - Match the photometry based on PM-corrected position
    # - Deblend if necessary
    # - Merge observations at the same wavelength
    return

# =============================================================================
# PARAMETER ESTIMATION
# =============================================================================
def deredden(sed,ancillary):

    ebv=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultEBV",1])
    rv=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultRV",1])
    extmap=pyssedsetupdata[pyssedsetupdata[:,0]=="ExtMap",1]

    if (verbosity>80):
        print ("Extinction:",extmap)

    if (ebv!=0):
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
            dist=adopt_distance(ancillary)
            coords=SkyCoord(ra*u.deg, dec*u.deg, frame='icrs').galactic
            l=int(coords.l.deg+0.5)
            b=int(coords.b.deg+0.5)
            d=int(adopt_distance(ancillary)/20.+0.5)*20
            try:
                ebv=Vizier(catalog="J/PAZh/43/521/rlbejk",columns=['**']).query_constraints(R=d,GLON=l,GLAT=b)[0]['E_J-Ks_'][0]*1.7033
            except: 
                if (d>700): # if distance is too large
                    ebv=Vizier(catalog="J/PAZh/43/521/rlbejk",columns=['**']).query_constraints(R=700,GLON=l,GLAT=b)[0]['E_J-Ks_'][0]*1.7033
                else: # try zero distance/reddening instead
                    ebv=0
        wavel=np.where(sed['wavel']>=1000., sed['wavel'], 1000.) # Prevent overflowing the function
        wavel=np.where(sed['wavel']<=33333., wavel, 33333.)    #
        sed['dered'] = sed['flux']/ext.extinguish(wavel*u.AA, Ebv=ebv)
        sed['derederr'] = sed['ferr']/ext.extinguish(wavel*u.AA, Ebv=ebv)
    else:
        sed['dered'] = sed['flux']
        sed['derederr'] = sed['ferr']

    if (verbosity>80):
        print ("E(B-V)=",ebv,"mag")
    
    return sed

# -----------------------------------------------------------------------------
def estimate_mass(teff,lum):
    # Estimate a stellar mass based on luminosity
    if ((teff>5500.) | (lum<2.)): # MS star
        mass=lum**(1/3.5)
    elif ((teff<5500.) | (lum>2500.)): # AGB/supergiant
        mcore=lum/62200.+0.487 # Blocker 1993
        minit=(mcore-0.3569)/0.1197 # Casewell 2009
        mass=(minit*2+mcore)/3.
    if (verbosity>80):
        print ("Revised mass =",mass)
    return mass


# -----------------------------------------------------------------------------
def adopt_distance(ancillary):
    # Adopt a distance based on a set of ancillary data
    
    # List parallaxes & distances
    # Do this in two stages to avoid numpy/python issues about bitwise/elementwise logic
    foo=ancillary[(ancillary['parameter']=="Parallax")]
    plx=foo[(foo['mask']==True)]['value']
    plxerr=foo[(foo['mask']==True)]['err']
    plxdist=1000./plx
    plxdisterr=plxerr/plx*plxdist
    foo=ancillary[(ancillary['parameter']=="Distance")]
    d=foo[(foo['mask']==True)]['value']
    derr=foo[(foo['mask']==True)]['err']
    if (verbosity > 80):
        print ("Plx:",plx,plxerr)
        print ("Dist[plx]:  ",plxdist,plxdisterr)
        print ("Dist[other]:",d,derr)
    
    # Now combine them by weighted error
    minerr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="MinAncError",1])
    plxdisterr=np.where(plxdisterr>0,plxdisterr,plxdist*minerr)
    derr=np.where(derr>0,derr,d*minerr)
    if ((len(d)>0) & (len(plxdist)>0)):
        dd=np.append(d,plxdist,axis=0)
        dderr=np.append(derr,plxdisterr,axis=0)
        dist=np.average(dd,weights=1./dderr**2)
    elif (len(d)>0):
        dist=np.average(d,weights=1./derr**2)
    elif (len(plxdist)>0):
        dist=np.average(plxdist,weights=1./plxdisterr**2)
    else:
        dist = 1 # Default to 1 pc distance
    # XXX Needs coding for selection of best or weighted average

    # List distances
    if (verbosity > 50):
        print ("Adopted distance:",dist,"pc")
    
    return dist

    
# =============================================================================
# SED FITTING
# =============================================================================
def sed_fit_bb(sed,ancillary):
    # Fit L, T_eff via a blackbody
    # XXX Need outlier rejection here too
    
    wavel=sed[sed['mask']>0]['wavel']/1.e10
    flux=sed[sed['mask']>0]['dered']
    ferr=sed[sed['mask']>0]['derederr']
    ferr=np.nan_to_num(ferr, nan=1.e6)
    freq=299792458./wavel
    
    dist=adopt_distance(ancillary)

    # Get default start (guess) parameters
    modelstartteff=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartTeff",1])
    # Get outlier rejection criteria
    maxoutliers=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxOutliers",1])
    outtol=float(pyssedsetupdata[pyssedsetupdata[:,0]=="OutlierTolerance",1])
    outchisqmin=float(pyssedsetupdata[pyssedsetupdata[:,0]=="OutlierChiSqMin",1])

    teff=optimize.minimize(chisq_model,modelstartteff,args=(freq,flux,ferr,np.array(["bb"]),[],[]),method='Nelder-Mead')['x'][0]
    
    bb,fratio,chisq=compute_model(teff,freq,flux,ferr,np.array(["bb"]),[],[])
    rad=np.sqrt(fratio)
    rsun=rad/0.004652477*dist
    # in arcsec - Sun @ 1 pc = 0.004652477
    lum=(rsun)**2*(teff/5776.)**4
    
    # XXX Need outlier rejection
    
    return sed,wavel,bb,teff,rsun,lum,chisq

# -----------------------------------------------------------------------------
def sed_fit_simple(sed,ancillary,modeldata):
    # Fit L, T_eff & log(g) to a set of models
    # (Based on McDonald 2009-2017: Assume fixed d, [Fe/H], E(B-V), R_v)
    # (For fast computation)
    # - Either assume mass or generate using estimate_mass
    # - Perform a simple parametric fit
    # - Return parameters, goodness of fit and residual excess/deficit
    
    dist=adopt_distance(ancillary)

    # Get default start (guess) parameters
    modelstartteff=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartTeff",1])
    modelstartfeh=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartFeH",1])
    modelstartlogg=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartLogg",1])
    modelstartalphafe=float(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelStartAFe",1])
    logg=modelstartlogg
    feh=modelstartfeh
    alphafe=modelstartalphafe

    # Pre-compute log(g) if required
    iteratelogg=int(pyssedsetupdata[pyssedsetupdata[:,0]=="IterateLogg",1])
    precomputelogg=int(pyssedsetupdata[pyssedsetupdata[:,0]=="PrecomputeLogg",1])
    mass=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultMass",1])
    usemsmass=int(pyssedsetupdata[pyssedsetupdata[:,0]=="UseMSMass",1])
    if (precomputelogg > 0):
        sed,bbwavel,bb,modelstartteff,bbrsun,bblum,bbchisq=sed_fit_bb(sed,ancillary)
        if (usemsmass>0):
            estimate_mass(modelstartteff,bblum)
        logg=4.44+np.log10(mass/bbrsun**2)
        # Check log(g) not too high/low
        if (logg<np.min(modeldata['logg'])):
            logg=np.min(modeldata['logg'])+0.001
        if (logg>np.max(modeldata['logg'])):
            logg=np.max(modeldata['logg'])-0.001
        if (verbosity>80):
            print ("Revised log g =",logg,"[",np.min(modeldata['logg']),np.max(modeldata['logg']),"]")

    # Get outlier rejection criteria
    maxoutliers=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxOutliers",1])
    maxseqoutliers=int(pyssedsetupdata[pyssedsetupdata[:,0]=="MaxSeqOutliers",1])
    outtol=float(pyssedsetupdata[pyssedsetupdata[:,0]=="OutlierTolerance",1])
    outchisqmin=float(pyssedsetupdata[pyssedsetupdata[:,0]=="OutlierChiSqMin",1])

    try:
        start = datetime.now().time() # time object
    except:
        start = 0

    # Extract parameters from SED
    wavel=sed[sed['mask']>0]['wavel']/1.e10
    flux=sed[sed['mask']>0]['dered']
    ferr=sed[sed['mask']>0]['derederr']
    ferr=np.nan_to_num(ferr, nan=1.e6)
    freq=299792458./wavel
    
    # Extract parameters and flux values for observed filters
    params=np.stack((modeldata['teff'],modeldata['logg'],modeldata['metal'],modeldata['alpha']),axis=1)
    values=np.array(modeldata[sed[sed['mask']>0]['svoname']].tolist())

    # Perform rough initial fit
    teff=optimize.minimize(chisq_model,modelstartteff,args=(freq,flux,ferr,np.array(["simple",logg,feh,alphafe]),params,values),method='Nelder-Mead',tol=30)['x'][0]
    # Compute final model
    model,fratio,chisq=compute_model(teff,freq,flux,ferr,np.array(["simple",logg,feh,alphafe]),params,values)

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
        outratio=np.where(testmodel>testflux,testmodel/testflux,testflux/testmodel)
        # If that exceeds the minimum tolerance...
        if (np.max(outratio)>outtol):
            # Remove the worst outlier from the SED and repeat the steps above to get the data necessary to fit
            sedidx=np.nonzero(outsed['mask'])[0][np.argmax(outratio)]
            if (verbosity>70):
                print ("Outlier",i+1,":",outsed[sedidx]['catname'],outsed[sedidx]['filter'],", factor =",np.max(outratio))
            testsed=np.delete(np.copy(outsed),sedidx) # Another temporary copy, masking the current outlier
            testwavel=testsed[testsed['mask']>0]['wavel']/1.e10
            testflux=testsed[testsed['mask']>0]['dered']
            testferr=testsed[testsed['mask']>0]['derederr']
            testferr=np.nan_to_num(testferr, nan=1.e6)
            testfreq=299792458./testwavel
            testvalues=np.array(modeldata[testsed[testsed['mask']>0]['svoname']].tolist())
            # Refit the data
            testteff=optimize.minimize(chisq_model,modelstartteff,args=(testfreq,testflux,testferr,np.array(["simple",logg,feh,alphafe]),params,testvalues),method='Nelder-Mead',tol=30)['x'][0]
            testmodel,testfratio,testchisq=compute_model(testteff,testfreq,testflux,testferr,np.array(["simple",logg,feh,alphafe]),params,testvalues)
            outsed[sedidx]['mask']=False
            # If the improvement in chi^2 is enough...
            if (chisq/testchisq>outchisqmin**(seqoutliers+1)):
                if (verbosity>80):
                    print ("Triggered chi^2 reduction (factor",chisq/testchisq,")")
                seqoutliers=0
                teff=testteff
                # Iterate log(g) if specified
                if (iteratelogg > 0):
                    spint,foo,bar=compute_model(teff,testfreq,testflux,testferr,np.array(["simple",logg,feh,alphafe]),params,np.expand_dims(modeldata['lum'],axis=1))
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
                model=testmodel
                flux=testflux
                chisq=testchisq
                sed['mask']=outsed['mask']
            else:
                seqoutliers+=1
                if (verbosity>90):
                    print ("Chi^2 reduction not triggered (factor",chisq/testchisq,")")
                if (seqoutliers>=maxseqoutliers):
                    if (verbosity>90):
                        print ("Too many sequential outliers without rejections -> breaking rejection loop")
                    break
        else:
            if (verbosity>90):
                print ("No significant outliers -> breaking rejection loop")
            break
            
    # Extract parameters from updated SED
    wavel=sed[sed['mask']>0]['wavel']/1.e10
    flux=sed[sed['mask']>0]['dered']
    ferr=sed[sed['mask']>0]['derederr']
    ferr=np.nan_to_num(ferr, nan=1.e6)
    freq=299792458./wavel

    # Repeat computation to get specific intensity of interpolated model [Jy]
    # Perform more precise fit
    values=np.array(modeldata[sed[sed['mask']>0]['svoname']].tolist())
    teff=optimize.minimize(chisq_model,teff,args=(freq,flux,ferr,np.array(["simple",logg,feh,alphafe]),params,values),method='Nelder-Mead',tol=0.25)['x'][0]
    # Compute final model
    model,fratio,chisq=compute_model(teff,freq,flux,ferr,np.array(["simple",logg,feh,alphafe]),params,values)

    spint,foo,bar=compute_model(teff,freq,flux,ferr,np.array(["simple",logg,feh,alphafe]),params,np.expand_dims(modeldata['lum'],axis=1))
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
    if (verbosity >= 50):
        print ("SED contains",len(sed),"points after outlier rejection")
    if (verbosity > 50):
        try:
            now = datetime.now().time() # time object
            elapsed = float(now.strftime("%s.%f"))-float(start.strftime("%s.%f"))
        except:
            now = 0; elapsed = 0
        print ("Took",elapsed,"seconds to fit, [",start,"] [",now,"]")
    
    return sed,wavel,model,teff,rsun,lum,chisq

# -----------------------------------------------------------------------------
def chisq_model(teff,freq,flux,ferr,modeltype,params,values):
    # Fit T_eff via a model for frequency in Hz
    # Fit in THz, rather than Hz, to avoid overflow
    # Returns flux from an emitter with area 1"^2 in Jy

    model,offset,chisq=compute_model(teff,freq,flux,ferr,modeltype,params,values)

    return chisq

# -----------------------------------------------------------------------------
def compute_model(teff,freq,flux,ferr,modeltype,params,valueselector):
    # Fit T_eff via a model for frequency in Hz
    # Fit in THz, rather than Hz, to avoid overflow
    # Returns flux flux from an emitter with area 1"^2 in Jy

    if (modeltype[0]=="bb"):
        # Blackbody
        # 2 * pi * 1"^2 / 206265^2 = 1 / 6771274157.32
        # 2*h/c^2 = 1.47449944e-50 kg s
        # h/k_B = 4.79924466e-11 K s
        # 1.47449944e-50 kg s / 6771274157.32 K s * 1e26 Jy/K = 2.1775805e-34 Jy kg
        model=np.log10(2.1775805e+02*(freq/1e12)**3/(np.exp(4.79924466e-11*freq/teff)-1))
        n=len(freq)
    elif (modeltype[0]=="simple"):
        logg=float(modeltype[1])
        feh=float(modeltype[2])
        alphafe=float(modeltype[3])
        
        # Restrict grid of models to adjecent points for faster fitting
        # This is the reason for the earlier lengthy recasting process!
        teffs=np.unique(params[:,0])
        tlo=teffs[np.searchsorted(teffs,teff,side='left')-1]
        if (tlo>=teffs[-1]):
            tlo=teffs[np.searchsorted(teffs,teff,side='left')-2]
            thi=teffs[np.searchsorted(teffs,teff,side='left')-1]
        else:
            thi=teffs[np.searchsorted(teffs,teff,side='left')]
        loggs=np.unique(params[:,1])
        glo=loggs[np.searchsorted(loggs,logg,side='left')-1]
        if (glo>=loggs[-1]):
            glo=loggs[np.searchsorted(loggs,logg,side='left')-2]
            ghi=loggs[np.searchsorted(loggs,logg,side='left')-1]
        else:
            ghi=loggs[np.searchsorted(loggs,logg,side='left')]
        cutparams=params[((params[:,0]==tlo) | (params[:,0]==thi)) & ((params[:,1]==glo) | (params[:,1]==ghi))]
        cutvalues=valueselector[((params[:,0]==tlo) | (params[:,0]==thi)) & ((params[:,1]==glo) | (params[:,1]==ghi))]

        interp_grid_points = np.array([float(teff),float(logg),float(feh),float(alphafe)])
        interp_data_points = np.zeros((len(cutvalues[0,:])))

        values=cutvalues[:,0]
        interpfn = interpolate.griddata(cutparams,cutvalues,interp_grid_points, method='linear', rescale=True)
        df = pd.DataFrame(interpfn)
        modeldata = df.interpolate().to_numpy()
        n=len(interp_data_points)
        interp_data_points=np.squeeze(modeldata)
        model=np.log10(interp_data_points)
    else:
        print ("Model type not understood:",modeltype[0])

    ferrratio=np.log10(1+ferr/flux)
    flux=np.log10(flux)
    offset=np.median(flux-model)
    model+=offset
    if (n>1):
        #chisq=np.sum((flux-model)**2/ferrratio**2)/(n-1) # Use errors to weight fit
        chisq=np.sum((flux-model)**2)/(n-1) # Set unity weighting for all data
    else:
        chisq=0.

    if (verbosity>=90):
        print ("Teff,chisq =",teff,chisq)

    # Return fitted model values if more than one value to fit. Otherwise, return interpolated value.
    if (n>1):
        return 10**model,10**(offset),chisq
    else:
        return interp_data_points,10**(offset),chisq

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
def plotsed(sed,modwave,modflux,plotfile):
    # Plot the completed SED to a file

    fig, ax = plt.subplots(figsize=[5, 4])

    # Set up colour map
    n = 128
    vals = np.ones((n*2, 4))
    vals[:128, 0] = 0
    vals[:128, 1] = np.linspace(0/256, 256/256, n)
    vals[:128, 2] = np.linspace(256/256, 0/256, n)
    vals[128:, 0] = np.linspace(0/256, 256/256, n)
    vals[128:, 1] = np.linspace(256/256, 0/256, n)
    vals[128:, 2] = 0.
    newcmp = ListedColormap(vals)

    # Set up main axes
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("Flux (Jy)")
    plt.minorticks_on()

    # Plot the model
    x=modwave*1.e6
    y=modflux
    indx = x.argsort()
    xs = x[indx]
    ys = y[indx]
    xerr=0
    yerr=0
    plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='s',markersize=3,color='#AA33AA80',ecolor='lightgray', elinewidth=1, capsize=0, zorder=5)
    plt.plot(xs,ys,color='#FF00FF20',zorder=0,linewidth=3)

    # Decide whether to put the inset on top or bottom,
    # based on whether the left or right quarter of the model points are higher
    lhs=np.average(np.log10(y[:int(len(y)/4)]))
    rhs=np.average(np.log10(y[-int(len(y)/4):]))

    # Plot the observed (reddened data)
    # Plot all the data
    x=sed['wavel']/10000
    y=sed['flux']
    xerr=0
    yerr=sed['ferr']
    plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='+',markersize=4,color='lightgray',ecolor='lightgray', elinewidth=1, capsize=0, zorder=10)

    # Overplot the unmasked in grey-red
    x=sed[sed['mask']>0]['wavel']/10000
    y=sed[sed['mask']>0]['flux']
    xerr=0
    yerr=0
    plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='+',markersize=4,color='#FFAAAA00',ecolor='#FFAAAA00', elinewidth=1, capsize=0, zorder=10)

    # Plot lines up to the new data
    x=sed['wavel']/10000
    y=sed['flux']
    dy=sed['dered']
    for i in np.arange(len(x)):
        plt.arrow(x[i],y[i],0,dy[i]-y[i],color='#00000010', linewidth=1,length_includes_head=True,zorder=11)
    x=sed[sed['mask']>0]['wavel']/10000
    y=sed[sed['mask']>0]['flux']
    dy=sed[sed['mask']>0]['dered']
    for i in np.arange(len(x)):
        plt.arrow(x[i],y[i],0,dy[i]-y[i],color='#FF000030', linewidth=1,length_includes_head=True,zorder=11)
    
    # Plot the dereddened data
    # Plot all the data
    x=sed['wavel']/10000
    y=sed['dered']
    xerr=sed['dw']/2/10000
    yerr=sed['derederr']
    plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='o',markersize=4,color='lightgray',ecolor='lightgray', elinewidth=1, capsize=0, zorder=10)

    # Overplot the unmasked in colour
    x=sed[sed['mask']>0]['wavel']/10000
    y=sed[sed['mask']>0]['dered']
    xerr=sed[sed['mask']>0]['dw']/2/10000
    yerr=sed[sed['mask']>0]['derederr']
    colour=np.log10(x)
    plt.scatter(x,y,c=colour,cmap=newcmp,s=16,zorder=20)

    # Include co-ordinates inset
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

    # Save the file
    plt.savefig(plotfile,dpi=150)
    plt.close("all")
    
    return

# =============================================================================
# MAIN PROGRAMME
# =============================================================================
def pyssed(cmdtype,cmdparams,proctype,procparams,setupfile):
    # Main routine
    errmsg=""
    try:
        startmain = datetime.now().time() # time object
    except:
        startmain = 0

    # Load setup
    if (setupfile==""):
        setupfile="setup.default"
    global pyssedsetupdata      # Share setup parameters across all subroutines
    pyssedsetupdata = np.loadtxt(setupfile, dtype=str, comments="#", delimiter="\t", unpack=False)

    global verbosity        # Output level of chatter
    verbosity=int(pyssedsetupdata[pyssedsetupdata[:,0]=="verbosity",1])
    if (verbosity>=30):
        print ("Setup file loaded.")

    # What type of search are we doing?
    if (cmdargs[1]=="single" or cmdargs[1]=="list"):
        searchtype="list" # list of one or more objects
    elif (cmdargs[1]=="cone" or cmdargs[1]=="box" or cmdargs[1]=="volume"):
        searchtype="area" # search on an area of sky
    elif (cmdargs[1]=="criteria" or cmdargs[1]=="complex" or cmdargs[1]=="nongaia" or cmdargs[1]=="uselast"):
        searchtype="none"
        errmsg=("Command type",cmdtype,"not programmed yet")
    else:
        searchtype="error"
        errmsg=("Command type",cmdtype,"not recognised")
    
    # Check whether using existing SEDs/Gaia data or not
    # (the 2:-2 avoids [''] that brackets the array)
    useexistingseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="UseExistingSEDs",1])
    sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="SEDsFile",1])[2:-2]
    useexistingphot=int(pyssedsetupdata[pyssedsetupdata[:,0]=="UseExistingPhot",1])
    photdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotTempDir",1])[2:-2]
    ancdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncTempDir",1])[2:-2]
    photfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PhotFile",1])[2:-2]

    # If using a list of sources, load that list.
    if (cmdargs[1]=="list"):
        listfile=cmdparams
        sourcedata = np.loadtxt(listfile, dtype=str, comments="#", delimiter="\t", unpack=False)
        outparamfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="OutputParamFile",1])[2:-2]
        outancfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="OutputAncFile",1])[2:-2]
        ancillary_queries=get_ancillary_list()
        ancillary_params=np.array(["#Object","RA","Dec"],dtype="str")
        ancillary_params=np.append(ancillary_params,np.unique(ancillary_queries['paramname']),axis=0)
        outappend=int(pyssedsetupdata[pyssedsetupdata[:,0]=="OutputAppend",1])
        # Append = 0 -> start new file, write headers
        if (outappend==0):
            with open(outparamfile, "w") as f:
                f.write("#Object\tTeff\tLum\tRad\tChi^2\n")
            np.savetxt(outancfile,ancillary_params,fmt="%s",delimiter="\t",newline="\t")
            with open(outancfile, "a") as f:
                f.write("\n")
        # Append = 1 -> add new list to existing (do nothing here)
        # Append = 2 -> check which objects have been done already and remove from sourcedata
        elif (outappend==2):
            if (verbosity >=50):
                print ("Checking existing objects...")
            try:
                processeddata = np.loadtxt(outparamfile, dtype=str, comments="#", delimiter="\t", unpack=False)
            except OSError:
                raise OSError("Existing datafile %s does not exist. Try a different file or use OutputAppend = 0 in setup." % outparamfile)
            procdat = processeddata[:,0]
            sourcedata = [i for i in sourcedata if i not in procdat]
    elif (cmdargs[1]=="single"):
        sourcedata = np.expand_dims(np.array(cmdparams),axis=0)

    # Get stellar atmosphere model data at start of run, if needed
    if (proctype=="simple"):
        modeldata=get_model_grid()
        
    # For area searches, only have to download one set of data, so we do that here.
    # For list searches, we'll defer that to the main loop.
    if ((searchtype=="area") & (useexistingphot==False)):
        primarycat=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PrimaryCatRef",1])[2:-2]
        try:
            starttime = datetime.now().time() # time object
        except:
            starttime = 0
        if (verbosity >=20):
            if (cmdargs[1]=="cone"):
                print ("Downloading data for cone (",cmdargs[2],cmdargs[3],cmdargs[4],")")
            if (cmdargs[1]=="box"):
                print ("Downloading data for box (",cmdargs[2],cmdargs[3],cmdargs[4],cmdargs[5],")")
            if (cmdargs[1]=="volume"):
                print ("Downloading data for volume (",cmdargs[2],cmdargs[3],cmdargs[4],cmdargs[5],")")
        if (verbosity >=40):
            print ("Downloading photometry...")
        get_area_data(cmdargs,get_catalogue_list(),photdir)
        if (verbosity >=40):
            print ("Downloading ancillary data...")
        get_area_data(cmdargs,get_ancillary_list(),ancdir)
        try:
            now = datetime.now().time() # time object
            elapsed = float(now.strftime("%s.%f"))-float(starttime.strftime("%s.%f"))
            if (verbosity>=30):
                print ("Took",elapsed,"seconds to download data, [",starttime,"] [",now,"]")
        except:
            pass
    
    # For area searches we now make the initial plots and form the SEDs
    if (searchtype=="area"):
        # XXX Make plots
        # XXX Form SEDs
        sourcedata=[]
            
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Main loop over sources: obtaining and processing SED data
    if (verbosity>=20):
        print ("Processing",len(sourcedata),"objects")
    for source in sourcedata:
        if (searchtype=="list"):
        # Get SED for list sources
            try:
                startsource = datetime.now().time() # time object
            except:
                startsource = 0
            if (verbosity >=50):
                print ("")
            if (verbosity>=10):
                print ("Processing source:",source)

            # If loading SEDs that are already created from disk...
            if (useexistingseds>0):
                sed=np.load(sedsfile,allow_pickle=True)
                if (verbosity>40):
                    print ("Extracting SEDs from pre-existing data")
            # If loading pre-downloaded data from disk...
            elif (useexistingphot>0):
                edr3_data=np.load(photfile,allow_pickle=True)
                if (verbosity>40):
                    print ("Extracting source from pre-existing data")
            # Else if querying online databases
            else:
                sed,ancillary,errmsg=get_sed_single(source)
                # Parse setup data: am I saving this file?
                saveseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveSEDs",1])
                if (saveseds>0):
                    if (verbosity>=60):
                        print ("Saving data to",sedsfile)
                    np.save(sedsfile,sed)
        elif (searchtype=="area"):
        # Get SED for area sources
            if (verbosity>=60):
                print ("NOT YET PROGRAMMED!")

        # Merge duplicated filters in SED
        sed=merge_sed(sed)
        # Merge ancillary data to produce final values
        ancillary=merge_ancillary(ancillary)

        # Deredden
        sed=deredden(sed,ancillary)
        
        if (verbosity>=70):
            print (sed)
        elif (verbosity>=50):
            print ("SED contains",len(sed),"points")

        # Do we need to process the SEDs? No...?
        if (proctype == "none"):
            if (verbosity > 0):
                print ("No further processing required.")
            modwave=np.empty((1),dtype=float)
            modflux=np.empty((1),dtype=float)
        # No data - cannot fit
        elif (len(sed[sed['mask']==True])<2):
            if (verbosity > 0):
                print ("Not enough data to fit.")
            modwave=np.empty((1),dtype=float)
            modflux=np.empty((1),dtype=float)
        # Yes: continue to process according to model type
        else:
            # ...blackbody
            if (proctype == "bb"):
                if (verbosity > 20):
                    print ("Fitting SED with blackbody...")
                sed,modwave,modflux,teff,rad,lum,chisq=sed_fit_bb(sed,ancillary)
            # ...simple SED fit
            elif (proctype == "simple"):
                if (verbosity > 20):
                    print ("Fitting SED with simple stellar model...")
                sed,modwave,modflux,teff,rad,lum,chisq=sed_fit_simple(sed,ancillary,modeldata)
            elif (proctype == "fit"):
                if (verbosity > 20):
                    print ("Fitting SED with full stellar model...")
            elif (proctype == "binary"):
                if (verbosity > 20):
                    print ("Fitting SED with binary stellar model...")
            # Unless processing a single source, add entry to global output files
            if (cmdargs[1]!="single"):
                outparams=source+"\t"+str(teff)+"\t"+str(lum)+"\t"+str(rad)+"\t"+str(chisq)+"\n"
                with open(outparamfile, "a") as f:
                    f.write(outparams)
                outanc=source+"\t"+str(np.squeeze(ancillary[(ancillary['parameter']=='RA') & (ancillary['mask']==True)]['value']))+"\t"+str(np.squeeze(ancillary[(ancillary['parameter']=='RA') & (ancillary['mask']==True)]['err']))+"\t"+str(np.squeeze(ancillary[(ancillary['parameter']=='Dec') & (ancillary['mask']==True)]['value']))+"\t"+str(np.squeeze(ancillary[(ancillary['parameter']=='Dec') & (ancillary['mask']==True)]['err']))
                for param in ancillary_params:
                    if (len(ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)]['value'])>0):
                        outanc=outanc+"\t"+str(np.squeeze(ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)]['value']))+"\t"+str(np.squeeze(ancillary[(ancillary['parameter']==param) & (ancillary['mask']==True)]['err']))
                    else:
                        outanc=outanc+"\t"+"--"+"\t"+"--"
                outanc=outanc+"\n"
                with open(outancfile, "a") as f:
                    f.write(outanc)
            if (verbosity > 20):
                print (source,"fitted with Teff=",teff,"K, R=",rad,"Rsun, L=",lum,"Lsun with chi^2=",chisq)
        # Add model flux to SED
        modelledsed=sed[sed['mask']==True]
        j=0
        for i in np.arange(len(sed)):
            if (sed[i] in modelledsed):
                sed[i]['model']=modflux[j]
                j+=1

        # Save individual SED to its own file?
        saveeachsed=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveEachSED",1])
        sedsdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="SEDsDir",1])[2:-2]
        if (saveeachsed>0):
            sedfile=sedsdir+source.replace(" ","_")+".sed"
            if (verbosity>=60):
                print ("Saving SED to",sedfile)
            np.savetxt(sedfile, sed, fmt="%s", delimiter="\t", header='\t'.join(sed.dtype.names))
        # Save individual ancillary data to its own file?
        saveeachanc=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveEachAnc",1])
        ancdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="AncDir",1])[2:-2]
        if ((saveeachanc>0) & (useexistingseds==0)):
            ancfile=ancdir+source.replace(" ","_")+".anc"
            if (verbosity>=60):
                print ("Saving ancillary data to",ancfile)
            np.savetxt(ancfile, ancillary, fmt="%s", delimiter="\t", header='\t'.join(ancillary.dtype.names))

        # Plot the SED if desired and save to file
        plotseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="PlotSEDs",1])
        if (plotseds > 0):
            plotdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PlotDir",1])[2:-2]
            plotfile=plotdir+source.replace(" ","_")+".png"
            if (len(sed[sed['mask']==True])>0):
                if (verbosity>=40):
                    print ("Plotting SED to",plotfile)
                plotsed(sed,modwave,modflux,plotfile)
            else:
                if (verbosity>=20):
                    print ("Not enough data to plot SED!",plotfile)
        if (verbosity >=30):
            try:
                now = datetime.now().time() # time object
                elapsed = float(now.strftime("%s.%f"))-float(startsource.strftime("%s.%f"))
                print ("Took",elapsed,"seconds to process",source," [",startsource,"] [",now,"]")
            except:
                pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (verbosity >=20):
        try:
            now = datetime.now().time() # time object
            elapsed = float(now.strftime("%s.%f"))-float(startmain.strftime("%s.%f"))
            print ("Took",elapsed,"seconds for entire process, [",startmain,"] [",now,"]")
        except:
            pass

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
        errmsg=pyssed(cmdtype,cmdparams,proctype,procparams,setupfile)
        if (errmsg!="" and errmsg!="Success!"):
            print ("ERROR!")
        print (errmsg)
        