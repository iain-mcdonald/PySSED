PySSED version 1.0 installation instructions.

PySSED should work out of the box (see requirements.txt), with the following exceptions:

The G-Tomo data cube is too large for GitHub storage. To use the G-Tomo dereddening, you will need to download this file:
https://figshare.com/ndownloader/files/43618119
and place it in the folder:
src/gtomo/_data/app_data/

If you need to add more filters, you will also need to download the model data files as described in doc/Pyssed_manual.pdf.
