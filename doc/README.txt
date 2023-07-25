Python Stellar Spectral Energy Distributions (PySSED) : Iain McDonald
---------------------------------------------------------------------

PySSED is a toolset that allows you to create and model SEDs for stars.

CONTENTS:
[1] Motivation and citations
[2] Basic usage
[3] Limitations / wish list

-------------------------------------------------------------------------------
SECTION 1: MOTIVATION AND CITATIONS
-------------------------------------------------------------------------------

The PySSED toolset is designed to allow the user to create, manipulate and fit the spectral energy distributions of stars based on publicly available data. It builds on the terminal-based methods in the following papers:
McDonald et al. (2009) : https://ui.adsabs.harvard.edu/#abs/2009MNRAS.394..831M
McDonald et al. (2012) : https://ui.adsabs.harvard.edu/#abs/2012MNRAS.427..343M
McDonald et al. (2017) : https://ui.adsabs.harvard.edu/#abs/2017MNRAS.471..770M

-------------------------------------------------------------------------------
SECTION 2: BASIC USAGE
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
SECTION 3: LIMITATIONS / WISH LIST
-------------------------------------------------------------------------------

- Read in catalogues from files
- No bespoke photometry rejection is allowed
- Only one beam size is allowed per catalogue. This may be limiting when survey wavelength varies substantially, or when the PSF is non-circular, leading to sub-ideal PSF deblending (e.g. Gaia, IRAS, AKARI FIS).

-------------------------------------------------------------------------------
SECTION 4: COMMON PROBLEMS (a.k.a. problems encountered in testing)
-------------------------------------------------------------------------------

Co-ordinates resolved by SIMBAD, but no SED returned.
- Known to occur if a poor source of astrometry is used (e.g. from IRC, IRAS, etc.). Widening the search radii (e.g. setup.default>GaiaCone, catalogues.default>XMatchCone) is possible, but may result in false cross-matches.
- Known to occur for resolved binaries (e.g. "alf Cen A" works, but "alf Cen" does not include proper motion).