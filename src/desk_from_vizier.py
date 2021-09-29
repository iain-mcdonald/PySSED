from astropy.table import Table
import astropy.units as u
import numpy as np
import argparse
import ipdb

from astropy.coordinates import get_icrs_coordinates
from vizier_sed import query_sed
import desk


def get_and_fit_sed(target, distance_kpc):
    """Obtains SED using Morgan Fouesneau's vizier_sed script
    (https://gist.github.com/mfouesneau/6caaae8651a926516a0ada4f85742c95).
    Package then is parsed into a version usable by the DESK
    (https://github.com/s-goldman/Dusty-Evolved-Star-Kit).
    Then fits the SED and generates a SED PDF.

    Parameters
    ----------
    target : str
        Name of target to be resolved by Simbad.
    distance_kpc : float
        Distance to source in kpc.
    """
    # Obtaining photometry
    coords = get_icrs_coordinates(target)
    results = query_sed([coords.ra.value, coords.dec.value])

    output_table = Table(
        (
            results["sed_freq"].to(u.um, equivalencies=u.spectral()).value,
            results["sed_flux"],
            results["sed_filter"],
        ),
        names=("wave", "flux", "filter"),
    )
    output_table.sort("wave")
    csv_name = target.replace(" ", "_") + ".csv"
    output_table.write(csv_name, overwrite=True)

    # DESK commands
    desk.fit(csv_name, distance=distance_kpc[0])
    desk.sed_indiv(flux="Jy")
    print("Done")


# just for parsing command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("distance", nargs="+")
    args = parser.parse_args()
    get_and_fit_sed(args.name, args.distance)
