Python Stellar Spectral Energy Distributions (PySSED) : Iain McDonald
---------------------------------------------------------------------

PySSED is a toolset that allows you to create and model SEDs for stars.

CONTENTS:
[1] Motivation and citations
[2] Basic usage
[3] Outputs

-------------------------------------------------------------------------------
SECTION 1: MOTIVATION AND CITATIONS
-------------------------------------------------------------------------------

The PySSED toolset is designed to allow the user to create, manipulate and fit the spectral energy distributions of stars based on publicly available data. It builds on the methods reported in the following papers:
McDonald et al. (2009) : https://ui.adsabs.harvard.edu/#abs/2009MNRAS.394..831M
McDonald et al. (2012) : https://ui.adsabs.harvard.edu/#abs/2012MNRAS.427..343M
McDonald et al. (2017) : https://ui.adsabs.harvard.edu/#abs/2017MNRAS.471..770M

-------------------------------------------------------------------------------
SECTION 2: BASIC USAGE
-------------------------------------------------------------------------------

The basic usage of PySSED is as follows:
python3 pyssed.py <mode> <targets> <fitting> [setup_file]

<mode>
Only the modes "single" and "list" are supported for now. Future versions will include "cone" and "box".

<targets>
These should be encompassed in quotation marks. They can be any one of:
- The SIMBAD name of a target for <mode>=single
- A set of co-ordinates in decimal degrees or hexadecimal hh:mm:ss +dd:mm:ss.
- A file containing a list of SIMBAD targets or co-ordinates for <mode>=list

<fitting>
Supported fitting methods are:
"None" - no fitting
"bb" - blackbody
"simple" - single stellar atmopshere model

[setup_file]
The setup file containing the desired settings. By default this is src/setup.default

-------------------------------------------------------------------------------
SECTION 3: OUTPUTS
-------------------------------------------------------------------------------

A series of outputs exist from the programme. These are optional but, by default, they are all on. Outputs are contained within the output/ directory. They include:

<object>.sed - ASCII format SED
<object>.anc - ASCII formatted list of ancillary data collected from tables in ancillary.default or specified ancillary file
<object>.png - Graphical plot of SED
hrd.dat - Short list of fundamental parameters for <mode>=list
anc.dat - Short list of ancillary parameters for <mode>=list

The SED file contains labelled columns:
catname = internal name for catalogue
objid = ID within catalogue
ra = Offset in Right Ascension (degrees) from requested source position
dec = Offset in Declination (degrees) from requested source position
modelra = Modelled <ra> based on expected proper motion
modeldec = Modelled <dec> based on expected proper motion
svoname = Spanish Virtual Observatory name for filter
filter = Catalogue name for filter
wavel = Representative wavelength for filter (Angstroms)
dw = Representative width of filter (Angstroms)
mag = Magnitude in specified filter
magerr = Error in <mag>
flux = Flux in specified filter (Jy)
ferr = Error in <flux>
dered = Flux corrected for interstellar reddening
derederr = Error in <dered>
model = Modelled flux according to SED model fit
mask = Binary flag indicating if photometry was used in SED model fitting

The ancillary file contains labelled columns:
parameter = Parameter in question
catname = Catalogue from which data is sourced
colname = Name of column in catalogue
value = Value of catalogue data
err = Error in <value>
priority = Relative importance for determining between measurements of the same data
mask = Binary flag indicating if data was selected as best representing that parameter for this object
