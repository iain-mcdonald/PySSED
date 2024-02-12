## script to read G-Tomo hdf5 file and convert to FITS

import h5py
import astropy.io.fits as fits
import numpy as np

def load_cube(hdf5file):
    """Load hdf5, calculate axes values corresponding to data.

    (original authors: N. Leclerc, G. Plum, S. Ferron)

    Args:
        hdf5file (str): full path for HDF5 file.

    Returns:
        dict: headers contains in HDF5 file.
        :func:`np.array`: 3D array which contains the extinction value.
        tuple: (x, y, z) where x,y,z contains array of axes
            corresponding to cube values.
        array: value min for x, y, z axes.
        array: value max for x, y, z axes.
        float: value of gridstep size
        float: value of half-width of the cube
        float: points (neeed??)
        float: value of scale (half-width*gridstep)
        step, hw, points, s

    """
    # read hdf5 file
    with h5py.File(hdf5file, 'r') as hf:
        cube = hf['explore/cube_datas'][:]
        dc = hf['explore/cube_datas']
        #cube = hf['stilism/cube_datas'][:]
        #dc = hf['stilism/cube_datas']
        
        headers = {k: v for k, v in dc.attrs.items()}

    sun_position = headers["sun_position"]
    gridstep_values = headers["gridstep_values"]
    new_sun_position = np.append(sun_position[1:],sun_position[0])

    headers["new_sun_position"] = new_sun_position
    headers['values_unit'] = 'A0(550nm)/parsec'

    # Calculate axes for cube value, with sun at position (0, 0, 0)
    min_axes = -1 * new_sun_position * gridstep_values
    max_axes = np.abs(min_axes)
    axes = (
        np.linspace(min_axes[0], max_axes[0], cube.shape[0]),
        np.linspace(min_axes[1], max_axes[1], cube.shape[1]),
        np.linspace(min_axes[2], max_axes[2], cube.shape[2])
    )

    step = np.array(headers["gridstep_values"])
    hw = (np.copy(cube.shape) - 1) / 2.
    points = (
        np.arange(0, cube.shape[0]),
        np.arange(0, cube.shape[1]),
        np.arange(0, cube.shape[2])
    )
    s = hw * step

    return (headers, cube,
        axes, min_axes, max_axes,
        step, hw, points, s)


def create_fits(data, headers, filename):
    #fits creation reverses the order of input ndarray
    #cube.shape(10,20,30) -> fits.shape(30,20,10)
    #therefore swap first and last axis
    swapped = np.swapaxes(data,0,2)
    hdu=fits.PrimaryHDU(swapped)
    hdul=fits.HDUList([hdu])

    hdr = hdu.header
    hdr.comments['NAXIS1'] = 'X axis'
    hdr.comments['NAXIS2'] = 'Y axis'
    hdr.comments['NAXIS3'] = 'Z axis'

    hdr['CREATION'] = (headers['creation_date'], 'Data creation date')
    hdr['CREATOR'] = ("EXPLORE (ACRI-ST)", 'Data creator')
    hdr['VERSION'] = headers['cube_version']
    hdr['STEP'] = (headers['gridstep_values'][0], 'gridstep value (parsec)')
    hdr['RESOL'] = (headers['resolution_values'][0], 'Resolution (parsec)')
    hdr['UNIT'] = (headers['values_unit'], 'Unit of the map')
    hdr['SUN_POSX'] = (headers['new_sun_position'][0], 'X position of the Sun')
    hdr['SUN_POSY'] = (headers['new_sun_position'][1], 'Y position of the Sun')
    hdr['SUN_POSZ'] = (headers['new_sun_position'][2], 'Z position of the Sun')
    #hdr.comments['object'] = 'updated comment'

    hdul.writeto(filename)

    #shorter
    #fits.writeto('out.fits', data, hdr)
    return

__main__:

files_base = ['explore_cube_density_values_050pc_v1', 'explore_cube_density_errors_050pc_v1',
'explore_cube_extinct_errors_050pc_v1', 'explore_cube_density_values_025pc_v1',
'explore_cube_density_errors_050pc_v2', 'explore_cube_density_values_050pc_v2',
'explore_cube_extinct_errors_050pc_v2', 'explore_cube_density_values_025pc_v2',
'explore_cube_density_values_010pc_v2', 'explore_cube_density_values_005pc_v2']

for file_base in files_base:
    file_hdf5 = file_base+".h5"
    file_fits = file_base+".fits"
    headers, cube, axes, min_axes, max_axes, step, hw, points, s = load_cube(file_hdf5)
    create_fits(cube, headers, file_fits)
