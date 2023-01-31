#### OVERVIEW
The following scripts provide basic functions to read & write HDF5 DAS files,
specifically from the Silixa / PRODML 2.1 standard of headers.

Also included are a couple basic functions to filter or downsample a large block
of data in a 2D numpy array.

#### FORMAT:
Files written natively by our Silixa iDAS use the PRODML
standard for headers. Many headers are not needed for our purposes,
and on top of that the raw files contain some proprietary information
about internal serial numbers.

These scripts are designed to read and write the only relevant / useful headers.
Files newly written by "write_das_h5" do NOT consitute the full PRODML standard, 
but at least variable names are consistent. This way, the same reader function can 
be used for raw PRODML HDF5 files and for our custom converted/downsampled files.


#### TODO:
As of Jan 2023, we have not settled on how to best handle channel mapping.
An update will follow.

#### EXAMPLE:
An full example of how to use these scripts is in the Jupyter notebooks, 
but the basic functions might be used as follows:

"""
import datetime
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters

#-- Load
t_start = datetime.strptime('2021/09/27 10:49:20.0', '%Y/%m/%d %H:%M:%S.%f') 
t_end   = datetime.strptime('2021/09/27 10:50:45.0', '%Y/%m/%d %H:%M:%S.%f') 
data, headers, axis = load_das_h5.load_das_custom(t_start, t_end, input_dir = "path/to/dir/")

#-- Filter
data2 = block_filters.block_bandpass(data, 0.1, 10, headers['fs'])

#-- Write
write_das_h5.write_block(data2, headers, "new_filename.h5")
"""


Last updated: Jan 2023
daniel.bowden@erdw.ethz.ch
patrick.paitz@erdw.ethz.ch 
