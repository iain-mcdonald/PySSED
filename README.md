<b>PySSED version 1.2</b>

This version accompanies the Gaia All-Sky Parameteres Service (GASPS), an EU OSCARS funded project. Original reference: McDonald et al., 2024, RASTI, 3, 89.

PySSED should work out of the box using Python3 src/pyssed.py (see requirements.txt). However, if you need to add more filters, you will also need to download the model data files. Documentation, including detailed installation instructions, can be found in doc/Pyssed_manual.pdf.

To run the NESS models as in the MNRAS paper, you will need to change the programme setup from the default filters to the Allsky filters. In brief, run:

<tt>cd src/</tt><br>
Edit setup.allsky so that <tt>RecomputeModelGrid = 1</tt>
<tt>
python3 makemodel.py bt-settl setup.allsky
python3 pyssed.py single "Betelgeuse" simple setup.allsky
source shorten-model.scr
</tt>
Edit setup.allsky file so that <tt>RecomputeModelGrid = 0</tt><br>
Then run your command, e.g.: <tt>python3 pyssed.py cone 270. -30. 0.05 setup.allsky</tt><br>
The same process can be used to add other filters - see manual for full details.
</tt>
