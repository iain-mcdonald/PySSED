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
import wget                                 # Required to download SVO filter files
from astroquery.simbad import Simbad        # Allow SIMBAD queries
from astroquery.vizier import Vizier        # Allow VizieR queries
from astroquery.gaia import Gaia            # Allow Gaia queries
import astropy.units as u                   # Required for astroquery
from astropy.coordinates import SkyCoord    # Required for astroquery
from astropy.io import votable              # Required for SVO interface

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
            print ("Expected one of: sed, simple, fit, binary")
            proctype=""
            procparams=[]
            error=1
    else:
        print ("ERROR! No processing type specified")
        print ("Expected one of: sed, simple, fit, binary")
        proctype=""
        procargs=[]
        error=1
        
    print ("    Using setup file:",setupfile)
    print ("")

    return cmdtype,cmdparams,proctype,procparams,setupfile,error
    
# -----------------------------------------------------------------------------
def get_catalogue_list():
    # Load setup
    catfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="SurveyFile",1])[2:-2]
    catdata = np.loadtxt(catfile, dtype=[('catname',object),('cdstable',object),('epoch',float),('beamsize',float),('matchr',float)], comments="#", delimiter="\t", unpack=False)
    return catdata

# -----------------------------------------------------------------------------
def get_filter_list():
    # Load setup
    filtfile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="FilterFile",1])[2:-2]
    filtdata = np.loadtxt(filtfile, dtype=[('catname',object),('filtname',object),('errname',object),('svoname',object),('datatype',object),('dataref',object),('maxmag',float),('minmag',float),('zptcorr',float)], comments="#", delimiter="\t", unpack=False)
    return filtdata

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
            coords=[rac[0]*15.+rac[1]/4+rac[2]/240,abs(decc[0])/decc[0]*(abs(decc[0])+decc[1]/60+decc[2]/3600)]
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
def get_simbad_single(cmdparams,sourcedata):

    if (verbosity>60):
        print (sourcedata)

    # Get from files for catalogues and filters
    catdata=get_catalogue_list()
    filtdata=get_filter_list()
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
    sed=np.array((0,),dtype=object)
    nfsuccess=0
    for catalogue in catalogues:
        if (catalogue!="Gaia"):
            # Correct proper motion and query VizieR
            ra,dec=pm_calc(sourcera,sourcedec,sourcepmra,sourcepmdec,sourceepoch,float(catdata[catdata['catname']==catalogue]['epoch']))
            vizier_data=query_vizier_cone(str(catdata[catdata['catname']==catalogue]['cdstable'])[2:-2],ra,dec,float(catdata[catdata['catname']==catalogue]['matchr']))

            if (verbosity>60):
                print ("CATALOGUE = ",catalogue,"; RA,DEC =",ra,dec)
                print (vizier_data)

            # Only proceed if VizieR has returned some data
            if (len(vizier_data)>0):
                # Identify magnitude and error columns
                magkeys=filtdata[filtdata['catname']==catalogue]['filtname']
                errkeys=filtdata[filtdata['catname']==catalogue]['errname']
                datatypes=filtdata[filtdata['catname']==catalogue]['datatype']
                datarefs=filtdata[filtdata['catname']==catalogue]['dataref']
                # Get RA and Dec from various columns in order of preference
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
                        except: # Needed for Morel
                            newra=vizier_data[0]['_RA'][0]
                            newdec=vizier_data[0]['_DE'][0]
                svokeys=filtdata[filtdata['catname']==catalogue]['svoname']
                # And extract them from vizier_data
                for i in (np.arange(len(magkeys))):
                    magkey=magkeys[i]
                    mag=vizier_data[0][magkey][0]
                    errkey=errkeys[i]
                    if (errkey=="None"):
                        err=defaulterr
                    else:
                        err=vizier_data[0][errkey][0]
                    svokey=svokeys[i]
                    wavel=float(svodata[svodata['svoname']==svokey]['weff'][0])
                    dw=float(svodata[svodata['svoname']==svokey]['dw'][0])
                    datatype=datatypes[i]
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
                        ferr=flux-10**((mag+err)/-2.5)*zpt
                    else: # Assume data type is a Jy-based unit
                        # What is read as mag is actually flux, so swap them
                        flux=mag
                        ferr=err
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
                    sed=np.append(sed,np.array([0]),axis=0)
                    sed[nfsuccess]=(newra,newdec,magkey,wavel,dw,mag,err,flux,ferr)
                    nfsuccess+=1
        else:
            # Assuming this is a Gaia source, extract the data
            if (sourcetype=="Gaia"):
                # For the Gaia catalogue itself, extract the BP, RP and G-band data directly
                magkeys=filtdata[filtdata['catname']==catalogue]['filtname']
                errkeys=filtdata[filtdata['catname']==catalogue]['errname']
                newra=sourcera
                newdec=sourcedec
                svokeys=filtdata[filtdata['catname']==catalogue]['svoname']
                for i in (np.arange(len(magkeys))):
                    svokey=svokeys[i]
                    magkey=magkeys[i]
                    wavel=float(svodata[svodata['svoname']==svokey]['weff'])
                    dw=float(svodata[svodata['svoname']==svokey]['dw'])
                    zpt=float(svodata[svodata['svoname']==svokey]['zpt'])
                    if (magkey=="BP"):
                        mag=float(sourcedata['phot_bp_mean_mag'])
                        flux=10**(mag/-2.5)*zpt
                        ferr=flux/float(sourcedata['phot_bp_mean_flux_over_error'])
                        err=-2.5*np.log(1-1/float(sourcedata['phot_bp_mean_flux_over_error']))
                    elif (magkey=="RP"):
                        mag=float(sourcedata['phot_rp_mean_mag'])
                        flux=10**(mag/-2.5)*zpt
                        ferr=flux/float(sourcedata['phot_rp_mean_flux_over_error'])
                        err=-2.5*np.log(1-1/float(sourcedata['phot_rp_mean_flux_over_error']))
                    else:
                        mag=float(sourcedata['phot_g_mean_mag'])
                        flux=10**(mag/-2.5)*zpt
                        ferr=flux/float(sourcedata['phot_g_mean_flux_over_error'])
                        err=-2.5*np.log(1-1/float(sourcedata['phot_g_mean_flux_over_error']))
                    sed=np.append(sed,np.array([0]),axis=0)
                    sed[nfsuccess]=(newra,newdec,magkey,wavel,dw,mag,err,flux,ferr)
                    nfsuccess+=1

    print (sed)
    
    return errmsg
    
# -----------------------------------------------------------------------------
def query_vizier_cone(cat,ra,dec,r):
    # Query the TAP server at CDS
    result = Vizier.query_region(SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs'), radius=r*u.arcsec, catalog=cat)
    
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
def get_gaia_data():
    # Process the Gaia data query:
    # - Download the data
    # - Perform plx and gmag corrections
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
def get_catalogues():
    # Read list of input catalogues and parameters from a file
    return

# -----------------------------------------------------------------------------
def get_cds_data():
    # Download the relevant data from CDS
    # - Specify catalogues to query
    # - Download data
    # - Parse for error flags and specific constraints (e.g. RUWE)
    # - Perform colour corrections
    # - Convert to wavelength and flux
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
def get_sed_single():
    # Generate an SED
    # - Get data from Gaia
    # - Get data from CDS
    # - Match the photometry based on PM-corrected position
    # - Merge observations at the same wavelength
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

# -----------------------------------------------------------------------------
def get_models():
    # Get a set of input stellar atmosphere models
    return
    
# -----------------------------------------------------------------------------
def estimate_mass():
    # Estimate a stellar mass based on stellar parameters
    return
    
# -----------------------------------------------------------------------------
def sed_fit_simple():
    # Fit L, T_eff & log(g) to a set of models
    # (Based on McDonald 2009-2017: Assume fixed d, [Fe/H], E(B-V), R_v)
    # (For fast computation)
    # - Either assume mass or generate using estimate_mass
    # - Perform a simple parametric fit
    # - Return parameters, goodness of fit and residual excess/deficit
    return

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

    # Obtaining SED data
    if (cmdargs[1]=="single"):
        # If loading pre-downloaded data from disk...
        # (the 2:-2 avoids [''] that brackets the array)
        useexistingdata=int(pyssedsetupdata[pyssedsetupdata[:,0]=="UseExistingData",1])
        gaiafile=np.array2string(pyssedsetupdata[pyssedsetupdata[:,0]=="GaiaFile",1])[2:-2]
        if (useexistingdata>0):
            edr3_data=np.load(gaiafile,allow_pickle=True)
            if (verbosity>40):
                print ("Extracting source from pre-existing data")
        # Else if querying online databases
        else:
            edr3_obj,errmsg=get_gaia_obj(cmdparams)
            # If a Gaia EDR3 cross-match can be found...
            if (edr3_obj > 0):
                # Query the Gaia EDR3 database for data
                edr3_data=get_gaia_single(edr3_obj)
                # Parse setup data: am I saving this file?
                savegaia=int(pyssedsetupdata[pyssedsetupdata[:,0]=="SaveGaia",1])
                if (savegaia>0):
                    if (verbosity>=60):
                        print ("Saving data to",gaiafile)
                    np.save(gaiafile,edr3_data)
                # Get data on other filters
                errmsg=get_simbad_single(cmdparams,edr3_data)
            else:
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
                    errmsg=get_simbad_single(cmdparams,hip_data)
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
                    errmsg=get_simbad_single(cmdparams,result_table)

    elif (cmdargs[1]=="list"):
        errmsg="Not programmed yet"
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
