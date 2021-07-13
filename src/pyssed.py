# Python Stellar SEDs (PySSED)
# Author: Iain McDonald
# Description:
# - Download and extract data on multi-wavelength catalogues of astronomical objects/regions of interest
# - Automatically process photometry into one or more stellar SEDs
# - Fit those SEDs with stellar parameters
# Use:
#

from sys import argv                        # Allows command-line arguments
from sys import exit                        # Allows graceful quitting
import numpy as np                          # Required for numerical processing
import scipy.optimize as optimize           # Required for fitting
import scipy.interpolate as interpolate     # Required for model grid interpolation
import astropy.units as u                   # Required for astroquery and B1950->J2000
from astropy.coordinates import SkyCoord    # Required for astroquery and B1950->J2000
from astropy.coordinates import FK5         # Required B1950->J2000 conversion
from astropy.io import votable              # Required for SVO interface
import matplotlib.pyplot as plt             # Required for plotting
from matplotlib.colors import ListedColormap # Required for colour mapping
from mpl_toolkits.axes_grid1.inset_locator import inset_axes # Reqd. for subplots

import wget                                 # Required to download SVO filter files
from astroquery.simbad import Simbad        # Allow SIMBAD queries
from astroquery.vizier import Vizier        # Allow VizieR queries
from astroquery.gaia import Gaia            # Allow Gaia queries

import itertools

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def parse_args(cmdargs):
    # Parse command-line arguments
    error=0
    
    # If no command-line arguments, print usage
    if (len(argv)<=1):
        print ("Use:")
        print (__file__,"<search type> [parameters] <processing type> [parameters] [setup file]")
        print ("<search type> : single, parameters = 'Source name' or 'RA, Dec'")
        print ("<search type> : list, parameters = 'File with sources or RA/Dec'")
        print ("<search type> : cone, parameters = RA, Dec, radius")
        print ("<search type> : box, parameters = RA1, Dec1, RA2, Dec2")
        print ("<search type> : volume, parameters = RA, Dec, d, r")
        print ("<search type> : criteria, parameters = 'SIMBAD criteria setup file'")
        print ("<search type> : complex, parameters = 'Gaia criteria setup file'")
        print ("<search type> : nongaia, parameters = 'Non-gaia criteria setup file'")
        print ("<search type> : uselast")
        print ("<processing type> : sed")
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
        if (proctype=="sed"):
            print ("(2) Only create the SEDs")
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
    
# -----------------------------------------------------------------------------
def get_catalogue_list():
    # Load catalogue list
    catfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="SurveyFile",1])[2:-2]
    catdata = np.loadtxt(catfile, dtype=[('server',object),('catname',object),('cdstable',object),('epoch',float),('beamsize',float),('matchr',float)], comments="#", delimiter="\t", unpack=False)
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
    ancillarydata = np.loadtxt(ancillaryfile, dtype=[('server',object),('catname',object),('cdstable',object),('colname',object),('errname',object),('paramname',object),('multiplier',object),('epoch',float),('beamsize',float),('matchr',float)], comments="#", delimiter="\t", unpack=False)
    return ancillarydata

# -----------------------------------------------------------------------------
def get_reject_list():
    # Load list of reject reasons
    rejectfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="RejectFile",1])[2:-2]
    rejectdata = np.loadtxt(rejectfile, dtype=[('catname',object),('filtname',object),('column',object),('position',int),('logical',object),('value',object),('result',object),('rejcat',object),('rejcol',object)], comments="#", delimiter="\t", unpack=False)
    return rejectdata

# -----------------------------------------------------------------------------
def get_model_list():
    # Load list of reduced models
    modelfile="model-"+np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="ModelCode",1])[2:-2]+".dat"
    print (modelfile)
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
    # Define as rectilinear grid
    grid_t,grid_g,grid_m,grid_a = np.meshgrid(np.unique(modelfiledata['teff']),np.unique(modelfiledata['logg']),np.unique(modelfiledata['metal']),np.unique(modelfiledata['alpha']),indexing='ij',sparse=True)
    # Separate parameters and values
    params=np.stack((modelfiledata['teff'],modelfiledata['logg'],modelfiledata['metal'],modelfiledata['alpha']),axis=1)
    valueselector = modelfiledata.dtype.names[4:]
    values=modelfiledata[:][list(valueselector)[0]]

    print(grid_t.shape, params.shape, values.shape)
    interp_grid_points = np.array(list(itertools.product(np.unique(modelfiledata['teff']),np.unique(modelfiledata['logg']),np.unique(modelfiledata['metal']),np.unique(modelfiledata['alpha']))))
    print(interp_grid_points)
    print(interp_grid_points.shape)
    #exit()

    # Recast onto rectilinear grid
    # This is where the problem is!
    #interpfn = interpolate.griddata(params,values,(grid_t,grid_g,grid_m,grid_a), method='linear', rescale=True)
    interpfn = interpolate.griddata(params,values,interp_grid_points, method='linear', rescale=True)
    import pandas as pd
    df = pd.DataFrame(interpfn)
    modeldata = df.interpolate().to_numpy()
    #interpfn = interpolate.RegularGridInterpolator((grid_t,grid_g,grid_m,grid_a), values.reshape((grid_t.shape[0], grid_g.shape[1], grid_m.shape[2], grid_a.shape[3])), method='linear', bounds_error=False, fill_value=None)
    #modeldata = interpfn(interp_grid_points)
    print(modeldata)
    print(modeldata.shape)
    print(np.sum(np.isfinite(modeldata)))
    print(interpfn)
    print(interpfn.shape)
    print(np.sum(np.isfinite(interpfn)))
    
    return modeldata

# -----------------------------------------------------------------------------
def get_svo_data(filtdata):

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
            result_table = Simbad.query_object(cmdparams)
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
    job = Gaia.launch_job(query)
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




# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def extract_ra_dec(vizier_data):

    # Extract RA and Dec from a Vizier table
    try:
        newra=vizier_data[0]['RAJ2000'][0]
        newdec=vizier_data[0]['DEJ2000'][0]
    except: # Needed for Hipparcos
        try:
            newra=vizier_data[0]['RArad'][0]
            newdec=vizier_data[0]['DErad'][0]
        except: # Needed for Tycho
            try:
                newra=vizier_data[0]['_RA.icrs'][0]
                newdec=vizier_data[0]['_DE.icrs'][0]
            except: # Needed for SDSS
                try:
                    newra=vizier_data[0]['RA_ICRS'][0]
                    newdec=vizier_data[0]['DE_ICRS'][0]
                except: # Needed for Morel
                    try:
                        newra=vizier_data[0]['_RA'][0]
                        newdec=vizier_data[0]['_DE'][0]
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
                        except: # Needed for Skymapper
                            try:
                                newra=vizier_data[0]['RAICRS'][0]
                                newdec=vizier_data[0]['DEICRS'][0]
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
        print (sourcedata)

    # Get from files for catalogues and filters
    catdata=get_catalogue_list()
    filtdata=get_filter_list()
    rejectdata=get_reject_list()
    # Get data from SVO for filters too
    svodata=get_svo_data(filtdata)
    
    # Get the parameters from the source data
    # Assuming Gaia first
    try:
        sourcetype="Gaia"
        sourcera=float(sourcedata['ra'][0])
        sourcedec=float(sourcedata['dec'][0])
        sourcepmra=float(sourcedata['pmra'][0])
        sourcepmdec=float(sourcedata['pmdec'][0])
        sourceepoch=float(pyssedsetupdata[pyssedsetupdata[:,0]=="GaiaEpoch",1])
    # If that fails, try Hipparcos
    except:
        try:
            sourcetype="Hipparcos"
            sourcera=float(sourcedata['RArad'][0])
            sourcedec=float(sourcedata['DErad'][0])
            sourcepmra=float(sourcedata['pmRA'][0])
            sourcepmdec=float(sourcedata['pmDE'][0])
            sourceepoch=1991.25
        except:
            sourcetype="None"
            rac=np.fromstring(sourcedata['RA'][0],dtype=float,sep=" ")
            decc=np.fromstring(sourcedata['DEC'][0],dtype=float,sep=" ")
            sourcera=rac[0]*15.+rac[1]/4+rac[2]/240
            sourcedec=abs(decc[0])/decc[0]*(abs(decc[0])+decc[1]/60+decc[2]/3600)
            sourcepmra=0.
            sourcepmdec=0.
            sourceepoch=2000.

    # Set the default error
    defaulterr=float(pyssedsetupdata[pyssedsetupdata[:,0]=="DefaultError",1])

    # Loop over catalogues
    catalogues=catdata['catname']
    sed=np.zeros((len(filtdata)),dtype=[('catname','<U20'),('ra','f4'),('dec','f4'),('modelra','f4'),('modeldec','f4'),('svoname','U32'),('filter','U10'),('wavel','f4'),('dw','f4'),('mag','f4'),('magerr','f4'),('flux','f4'),('ferr','f4'),('mask','bool')])
    nfsuccess=0
    for catalogue in catalogues:
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
                    sed[nfsuccess]=(catalogue,(newra-sourcera)*3600.,(newdec-sourcedec)*3600.,(ra-sourcera)*3600.,(dec-sourcedec)*3600.,svokey,magkey,wavel,dw,mag,err,flux,ferr,1)
                    nfsuccess+=1
        else:
            ra,dec=pm_calc(sourcera,sourcedec,sourcepmra,sourcepmdec,sourceepoch,float(catdata[catdata['catname']==catalogue]['epoch']))
            if (server=="Vizier"):
                # Correct proper motion and query VizieR
                vizier_data=query_vizier_cone(str(catdata[catdata['catname']==catalogue]['cdstable'])[2:-2],ra,dec,float(catdata[catdata['catname']==catalogue]['matchr']))

                if (verbosity>60):
                    print ("CATALOGUE = ",catalogue,"; RA,DEC =",ra,dec)
                    print (vizier_data)

                # Only proceed if VizieR has returned some data
                if (len(vizier_data)>0):
                    if (verbosity>80):
                        print (vizier_data[0])
                    # Get RA and Dec from various columns in order of preference
                    newra,newdec=extract_ra_dec(vizier_data)
            else: # Data from file
                photfile=str(catdata[catdata['catname']==catalogue]['cdstable'])[2:-2]
                matchr=float(catdata[catdata['catname']==catalogue]['matchr'])
                phot=np.genfromtxt(photfile, delimiter='\t', names=True)
                c=SkyCoord(ra=ra*u.degree,dec=dec*u.degree)
                catdata=SkyCoord(ra=phot['RA']*u.degree,dec=phot['Dec']*u.degree)
                idx,d2d,d3d=c.match_to_catalog_sky(catdata)
                if (d2d.arcsec<matchr):
                    vizier_data=np.expand_dims(np.expand_dims(phot[idx],axis=0),axis=1)
                else:
                    vizier_data=[]
                if (verbosity>60):
                    print ("CATALOGUE = ",catalogue,"; RA,DEC =",ra,dec)
                    print (vizier_data)
            if (len(vizier_data)>0): # If any data exists
                # Then check it's good by rejecting anything nasty
                reasons=rejectdata[(rejectdata['catname']==catalogue) & (rejectdata['rejcat']=="Same") & (rejectdata['rejcol']=="Same")]
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
                        else:
                            vizier_data[0][reasons[i]['filtname']][0]=-9.e99 # Set to stupidly low flux/mag
                    # CatName	CDSLabel	Column	Position	Logical	Value	Result
                    # Logical = < (less than), = (equal to), ~ (contains), ! (not equal to), > (greater than)
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
                    mask=1
                    magkey=magkeys[i]
                    mag=vizier_data[0][magkey][0]
                    mindata=mindatas[i]
                    maxdata=maxdatas[i]
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
                                magerr=2.5*np.log10(1+ferr/flux)
                            # If the fractional error in the flux is sufficiently small
                            if (ferr/flux<maxperr[i]/100.):
                                sed[nfsuccess]=(catalogue,(newra-sourcera)*3600.,(newdec-sourcedec)*3600.,(ra-sourcera)*3600.,(dec-sourcedec)*3600.,svokey,magkey,wavel,dw,mag,err,flux,ferr,mask)
                                nfsuccess+=1

    # If there is no error in flux, use default error
    sed[sed['ferr']==0]['ferr']==sed[sed['ferr']==0]['flux']*defaulterr

    # Apply rejection criteria spanning multiple catalogues
    if (verbosity>60):
        print ("Rejecting data based on flags...")
    try:
        reasons=rejectdata[(rejectdata['rejcat']!="Same") & (rejectdata['position']<0)]
    except:
        reasons=[]
    for i in np.arange(len(reasons)):
        # Identify data to test
        testdata=sed[(sed['catname']==reasons[i]['rejcat']) & (sed['filter']==reasons[i]['rejcol'])][reasons[i]['column']]
        n=(sed['catname']==reasons[i]['catname']) & (sed['filter']==reasons[i]['filtname'])
        compmag=sed[n]['mag']
        compflux=sed[n]['flux']
        logic=reasons[i]['logical']
        # If comparison data exists
        if ((len(testdata)>0) & len(n)>0):
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
        if ((len(testdata)>0) & len(n)>0):
            action=0
            if (logic=="="):
                if (str(testdata)==reasons[i]['value']):
                    action=1
            elif (logic=="<"):
                if (float(testdata)<float(reasons[i]['value'])):
                    action=1
            elif (logic==">"):
                if (float(testdata)>float(reasons[i]['value'])):
                    action=1
            elif (logic=="!"):
                if (str(testdata)!=str(reasons[i]['value'])):
                    action=1
            elif (logic=="~"):
                if (str(testdata) in str(reasons[i]['value'])):
                    action=1
            # If action needs taken then mask or purge
            if (action==1):
                if (reasons[i]['filtname']=="All"): # All filters, or only one?
                    n=(sed['catname']==reasons[i]['catname'])
                else:
                    n=(sed['catname']==reasons[i]['catname']) & (sed['filter']==reasons[i]['filtname'])
                if (reasons[i]['result']=="Mask"): # Mask or reject?
                    sed[:]['mask']=np.where(n==True,False,sed[:]['mask'])
                else:
                    sed[n]['flux']=np.where(n==True,0,sed[:]['flux'])

    # Get ancillary information
    if (verbosity>60):
        print ("Getting ancillary data...")
    ancillary_queries=get_ancillary_list()
    # If number of queries == 1, expand dimensions to avoid error
    if (np.size(ancillary_queries)==1):
            ancillary_queries=np.expand_dims(ancillary_queries,axis=0)
    ancillary=np.zeros([np.size(ancillary_queries)],dtype=[('parameter',object),('server',object),('catname',object),('value',object),('err',object)])
    for i in np.arange(np.size(ancillary_queries)):
        print (ancillary_queries[i])
        ra,dec=pm_calc(sourcera,sourcedec,sourcepmra,sourcepmdec,sourceepoch,float(ancillary_queries[i]['epoch']))
        if (ancillary_queries[i]['server']=="Vizier"):
            # Correct proper motion and query VizieR
            vizier_data=query_vizier_cone(str(ancillary_queries[i]['cdstable']),ra,dec,float(ancillary_queries[i]['matchr']))

            if (verbosity>60):
                print ("CATALOGUE = ",str(ancillary_queries[i]['cdstable']),"; RA,DEC =",ra,dec)
                print (vizier_data)

            # Only proceed if VizieR has returned some data
            if (len(vizier_data)>0):
                if (verbosity>80):
                    print (vizier_data)
                # Get RA and Dec from various columns in order of preference
                newra,newdec=extract_ra_dec(vizier_data)
                ancillary[i]=(ancillary_queries[i]['paramname'],ancillary_queries[i]['server'],ancillary_queries[i]['catname'],vizier_data[0][ancillary_queries[i]['colname']][0],vizier_data[0][ancillary_queries[i]['errname']][0])
        else:
            raise ("ERROR! CAN ONLY GET ANCILLARY DATA FROM VIZIER FOR NOW, NOT FILES.")
    #magkeys=filtdata[filtdata['catname']==catalogue]['filtname']
    #ancillarydata = np.loadtxt(ancillaryfile, dtype=[('server',object),('catname',object),('cdstabl',object),('colname',object),('errname',object),('paramname',object),('multiplier',object),('epoch',float),('beamsize',float),('matchr',float)], comments="#", delimiter="\t", unpack=False)

    return sed[sed['flux']>0.],ancillary[ancillary['parameter']!=""],errmsg
    
# -----------------------------------------------------------------------------
def query_vizier_cone(cat,ra,dec,r):
    # Query the TAP server at CDS
    result = Vizier(columns=["**"]).query_region(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'), radius=r*u.arcsec, catalog=cat)
    
    return result

# -----------------------------------------------------------------------------
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
def remove_outliers():
    # Reject outliers from an SED based on goodness-of-fit
    return

# -----------------------------------------------------------------------------
def estimate_flux():
    # Estimate the flux at a given wavelength based on data at others
    return

# -----------------------------------------------------------------------------
def merge_flux():
    # Merge the flux at a specific wavelength, based either on mean, weighted mean or preferred catalogue
    return

# -----------------------------------------------------------------------------
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
        savegaia=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveGaia",1])
        gaiafile=str(pyssedsetupdata[pyssedsetupdata[:,0]=="GaiaFile",1])[2:-2]
        if (savegaia>0):
            if (verbosity>=60):
                print ("Saving data to",gaiafile)
            np.save(gaiafile,edr3_data)
        # Get data on other filters
    if (len(edr3_data)>0):
        sed,ancillary,errmsg=get_vizier_single(cmdparams,edr3_data)
    if ((edr3_obj == 0) | (len(edr3_data)==0)):
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
            hip_data=Vizier.query_object(hip, catalog="I/311/hip2")['I/311/hip2']
            sed,ancillary,errmsg=get_vizier_single(cmdparams,hip_data)
        else:
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
def get_sed_multiple():
    # Generate a set of SEDs from spatially overlapping data
    # - Get data from Gaia
    # - Get data from CDS
    # - Match the photometry based on PM-corrected position
    # - Deblend if necessary
    # - Merge observations at the same wavelength
    return

# -----------------------------------------------------------------------------
def get_models():
    # Get a set of input stellar atmosphere models
    return
    
# -----------------------------------------------------------------------------
def estimate_mass():
    # Estimate a stellar mass based on stellar parameters
    return


# -----------------------------------------------------------------------------
def adopt_distance(ancillary):
    # Adopt a distance based on a set of ancillary data
    
    # List parallaxes
    plx=ancillary[ancillary['parameter']=="Parallax"]['value']
    dist=np.ma.mean(1000./plx)
    # XXX Needs coding for selection of best or weighted average

    # List distances
    if (verbosity > 50):
        print ("Adopted distance:",dist,"pc")
    
    return dist

    
# -----------------------------------------------------------------------------
def sed_fit_bb(sed,ancillary):
    # Fit L, T_eff via a blackbody
    
    wavel=sed[sed['mask']>0]['wavel']/1.e10
    flux=sed[sed['mask']>0]['flux']
    ferr=sed[sed['mask']>0]['ferr']
    ferr=np.nan_to_num(ferr, nan=1.e6)
    freq=299792458./wavel
    
    dist=adopt_distance(ancillary)

    teff=optimize.minimize(chisq_model,5776.,args=(freq,flux,ferr,"bb",[],[]),method='Nelder-Mead')['x'][0]
    
    bb,fratio,chisq=compute_model(teff,freq,flux,ferr,"bb",[],[])
    rad=np.sqrt(fratio)
    # in arcsec - Sun @ 1 pc = 0.004652477
    lum=(rad/0.004652477*dist)**2*(teff/5776.)**4
    
    return wavel,bb,teff,rad,lum,chisq

# -----------------------------------------------------------------------------
def sed_fit_simple(sed,ancillary,modeldata):
    # Fit L, T_eff & log(g) to a set of models
    # (Based on McDonald 2009-2017: Assume fixed d, [Fe/H], E(B-V), R_v)
    # (For fast computation)
    # - Either assume mass or generate using estimate_mass
    # - Perform a simple parametric fit
    # - Return parameters, goodness of fit and residual excess/deficit
    
    wavel=sed[sed['mask']>0]['wavel']/1.e10
    flux=sed[sed['mask']>0]['flux']
    ferr=sed[sed['mask']>0]['ferr']
    ferr=np.nan_to_num(ferr, nan=1.e6)
    freq=299792458./wavel
    
    dist=adopt_distance(ancillary)
    # XXX Need to recode these as ancillary data
    feh=0
    ebv=0

    # Extract parameters and flux values for observed filters
    params=np.stack((modeldata['teff'],modeldata['logg'],modeldata['metal'],modeldata['alpha']),axis=1)
    values=modeldata[sed[sed['mask']>0]['svoname']]
    
#    teff=optimize.minimize(chisq_model,5776.,args=(freq,flux,ferr,"simple",modeldata),method='Nelder-Mead')['x'][0]
    teff=5776.

    model,fratio,chisq=compute_model(teff,freq,flux,ferr,"simple",params,values)
    rad=np.sqrt(fratio)
    # in arcsec - Sun @ 1 pc = 0.004652477
    lum=(rad/0.004652477*dist)**2*(teff/5776.)**4
    
    return wavel,model,teff,rad,lum,chisq

# -----------------------------------------------------------------------------
def chisq_model(teff,freq,flux,ferr,modeltype,params,values):
    # Fit T_eff via a model for frequency in Hz
    # Fit in THz, rather than Hz, to avoid overflow
    # Returns flux from an emitter with area 1"^2 in Jy

    if (modeltype=="bb"):
        # Blackbody
        # 2 * pi * 1"^2 / 206265^2 = 1 / 6771274157.32
        # 2*h/c^2 = 1.47449944e-50 kg s
        # h/k_B = 4.79924466e-11 K s
        # 1.47449944e-50 kg s / 6771274157.32 K s * 1e26 Jy/K = 2.1775805e-34 Jy kg
        model=np.log10(2.1775805e+02*(freq/1e12)**3/(np.exp(4.79924466e-11*freq/teff)-1))
    elif (modeltype=="simple"):
        
        print ("This should never be seen")
    ferrratio=np.log10(1+ferr/flux)
    flux=np.log10(flux)
    offset=np.median(flux-model)
    model+=offset
    #chisq=np.sum((flux-model)**2/ferrratio**2) # Use errors to weight fit
    chisq=np.sum((flux-model)**2) # Set unity weighting for all data

    return chisq

# -----------------------------------------------------------------------------
def compute_model(teff,freq,flux,ferr,modeltype,params,values):
    # Fit T_eff via a model for frequency in Hz
    # Fit in THz, rather than Hz, to avoid overflow
    # Returns flux flux from an emitter with area 1"^2 in Jy

    if (modeltype=="bb"):
        # Blackbody
        # 2 * pi * 1"^2 / 206265^2 = 1 / 6771274157.32
        # 2*h/c^2 = 1.47449944e-50 kg s
        # h/k_B = 4.79924466e-11 K s
        # 1.47449944e-50 kg s / 6771274157.32 K s * 1e26 Jy/K = 2.1775805e-34 Jy kg
        model=np.log10(2.1775805e+02*(freq/1e12)**3/(np.exp(4.79924466e-11*freq/teff)-1))
    elif (modeltype=="simple"):
        # Identify nearest model points
#        print (modeldata[['teff','logg','metal','alpha']][0])
#        print (modeldata[0:3][0])
        logg=1
        teff=5776.
        feh=0.1
        alphafe=0.1
        print (params)
        for filt in values.names:
            model=interpolate.RegularGridInterpolator((params,),values['filt'],method="linear",bounds_error=False,fill_value=None)
            print (model)
    else:
        print ("Model type not understood:",modeltype)
    ferrratio=np.log10(1+ferr/flux)
    flux=np.log10(flux)
    offset=np.median(flux-model)
    model+=offset
    #chisq=np.sum((flux-model)**2/ferrratio**2) # Use errors to weight fit
    chisq=np.sum((flux-model)**2) # Set unity weighting for all data

    return 10**model,10**(offset),chisq

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

# -----------------------------------------------------------------------------
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

    # Plot all the data
    x=sed['wavel']/10000
    y=sed['flux']
    xerr=sed['dw']/2/10000
    yerr=sed['ferr']
    plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='o',markersize=4,color='lightgray',ecolor='lightgray', elinewidth=1, capsize=0, zorder=10)

    # Overplot the unmasked in colour
    x=sed[sed['mask']>0]['wavel']/10000
    y=sed[sed['mask']>0]['flux']
    xerr=sed[sed['mask']>0]['dw']/2/10000
    yerr=sed[sed['mask']>0]['ferr']
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
    
    return

# -----------------------------------------------------------------------------
def pyssed(cmdtype,cmdparams,proctype,procparams,setupfile):
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

    # Check whether using existing SEDs/Gaia data or not
    # (the 2:-2 avoids [''] that brackets the array)
    useexistingseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="UseExistingSEDs",1])
    sedsfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="SEDsFile",1])[2:-2]
    useexistingdata=int(pyssedsetupdata[pyssedsetupdata[:,0]=="UseExistingData",1])
    gaiafile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="GaiaFile",1])[2:-2]

    # Obtaining SED data
    if (cmdargs[1]=="single"):
        # If loading SEDs that are already created from disk...
        if (useexistingseds>0):
            sed=np.load(sedsfile,allow_pickle=True)
            if (verbosity>40):
                print ("Extracting SEDs from pre-existing data")
        # If loading pre-downloaded data from disk...
        elif (useexistingdata>0):
            edr3_data=np.load(gaiafile,allow_pickle=True)
            if (verbosity>40):
                print ("Extracting source from pre-existing data")
        # Else if querying online databases
        else:
            sed,ancillary,errmsg=get_sed_single(cmdparams)
            # Parse setup data: am I saving this file?
            saveseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveSEDs",1])
            if (saveseds>0):
                if (verbosity>=60):
                    print ("Saving data to",sedsfile)
                np.save(sedsfile,sed)
        if (verbosity>=80):
            print (sed)

        # XXX NEED OUTLIER REJECTION XXX
        # XXX NEED REDDENING CORRECTION XXX

        # Do we need to process the SEDs? No...?
        if (proctype == "sed"):
            if (verbosity > 0):
                print ("No further processing required.")
            modwave=np.empty((1),dtype=float)
            modflux=np.empty((1),dtype=float)
        # Yes: continue to process according to model type
        elif (proctype == "bb"):
            if (verbosity > 20):
                print ("Fitting SED with blackbody...")
            modwave,modflux,teff,rad,lum,chisq=sed_fit_bb(sed,ancillary)
            if (verbosity > 20):
                print ("...fitted Teff=",teff,"K, R=",rad,"arcsec, L=",lum,"Lsun with chi^2=",chisq)
        elif (proctype == "simple"):
            if (verbosity > 20):
                print ("Fitting SED with simple stellar model...")
            modeldata=get_model_list()
            modwave,modflux,teff,rad,lum,chisq=sed_fit_simple(sed,ancillary,modeldata)
            if (verbosity > 20):
                print ("...fitted Teff=",teff,"K, R=",rad,"arcsec, L=",lum,"Lsun with chi^2=",chisq)
        elif (proctype == "fit"):
            if (verbosity > 20):
                print ("Fitting SED with full stellar model...")
        elif (proctype == "binary"):
            if (verbosity > 20):
                print ("Fitting SED with binary stellar model...")

        # Plot the SED if desired and save to file
        plotseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="PlotSEDs",1])
        if (plotseds > 0):
            plotdir=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="PlotDir",1])[2:-2]
            plotfile=plotdir+cmdparams.replace(" ","_")+".png"
            if (verbosity>=20):
                print ("Plotting SED to",plotfile)
            plotsed(sed,modwave,modflux,plotfile)

    elif (cmdargs[1]=="list"):
        # If loading SEDs that are already created from disk...
        if (useexistingseds>0):
            sed=np.load(sedsfile,allow_pickle=True)
            if (verbosity>40):
                print ("Extracting SEDs from pre-existing data")
        # If loading pre-downloaded data from disk...
        elif (useexistingdata>0):
            edr3_data=np.load(gaiafile,allow_pickle=True)
            if (verbosity>40):
                print ("Extracting source from pre-existing data")
        # Else if querying online databases
        else:
            seds,errmsg=get_sed_list(cmdparams)
            # Parse setup data: am I saving this file?
            saveseds=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveSEDs",1])
            if (saveseds>0):
                if (verbosity>=60):
                    print ("Saving data to",sedsfile)
                np.save(sedsfile,seds)

        errmsg="Remainder not programmed yet"
    elif (cmdargs[1]=="cone"):
        errmsg="Not programmed yet"
    elif (cmdargs[1]=="box"):
        errmsg="Not programmed yet"
    elif (cmdargs[1]=="volume"):
        errmsg="Not programmed yet"
    elif (cmdargs[1]=="criteria"):
        errmsg="Not programmed yet"
    elif (cmdargs[1]=="complex"):
        errmsg="Not programmed yet"
    elif (cmdargs[1]=="nongaia"):
        errmsg="Not programmed yet"
    elif (cmdargs[1]=="uselast"):
        errmsg="Not programmed yet"
    else:
        errmsg=("Command type",cmdtype,"not recognised")
        
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
