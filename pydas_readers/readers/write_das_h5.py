"""
Scripts to write HDF5 files, assuming a PRODML-style format.
This does not constitute the entire PRODML standard (i.e., does
not have all necessary fields), but uses compatible variable names.


Daniel Bowden, ETH Zürich
daniel.bowden@erdw.ethz.ch
Last updated: Jan 2023
"""

import numpy as np
from datetime import datetime, timedelta
import glob
import h5py
import pathlib
import sys
import os

l_fields = []
l_attrs = []

def reset_attributes():
    l_fields.clear()
    l_attrs.clear()
    return(None)

def write_block(data2,headers,new_filename):
    reset_attributes()

    with h5py.File(new_filename, "w") as f:
        # The exact structure was meant to match Silixa's prodml format
        # The structure can be learned from the load function with:
        #   print(group,k,f[group].attrs[k])
        #
        # Or with unix "h5dump" ( consider "| head" and "| tail" to see 
        #   parts while avoiding the main data block)

        sdt = h5py.string_dtype('utf-8', 32)       # specify StringDataType
        
        subgroup = f.create_group("Acquisition")
        subgroup.attrs.create("GaugeLength",data=headers['gauge'])
        subgroup_custom = subgroup.create_group("Custom")
        subgroup_custom_user = subgroup_custom.create_group("UserSettings")
        subgroup_custom_user.attrs.create("SpatialResolution",data=headers['dx'])
        subgroup_custom_user.attrs.create("MeasureLength",data=headers['lx'])
        subgroup_custom_user.attrs.create("StartDistance",data=headers['d0'])
        subgroup_custom_user.attrs.create("StopDistance",data=headers['d1'])


        subgroup_custom_system = subgroup_custom.create_group("SystemSettings")
        subgroup_custom_system.attrs.create("FibreLengthMultiplier",data=headers['fm'])


        subgroup_raw = subgroup.create_group("Raw[0]")
        subgroup_raw.attrs.create("OutputDataRate",data=headers['fs'])
        
        # Original sample rate (important for scaling native optical units to proper nano strainrate)
        if('fs_orig' in headers):
            subgroup_raw.attrs.create("OriginalDataRate",data=headers['fs_orig'])

        # Note of whether corrective scaling has been applied yet
        # Raw units = amplitude of 1
        # Scaled units = amplitude scaling => data * 116. / 8192. * fs_orig / 10. 
        if('amp_scaling' in headers):
            subgroup_raw.attrs.create("AmpScaling",data=headers['amp_scaling'])
        else:
            subgroup_raw.attrs.create("AmpScaling",data=1.0)

        # nchan and npts should match dimensions of data (needed to read in data block properly)
        subgroup_raw.attrs.create("NumberOfLoci",data=headers['nchan'])
        subgroup_raw.attrs.create("RawDataUnit", data=headers['unit'].encode('ascii'), dtype=sdt)

        # Raw data block
        # Note these are integers. This could lead to rounding differences, for example 
        #  if one writes data that has been scaled/filtered/downsampled.

        #-- Option 1: Raw Silixa/PRODML uses 16-bit integer ("i2" in python)
        #dataset = subgroup_raw.create_dataset("RawData", data=data2, dtype="i2", chunks=True)

        #-- Option 2: If we've filtered/downsampled/manipulated the data, we might want higher precision
        #--  here is with float32. 
        dataset = subgroup_raw.create_dataset("RawData", data=data2, dtype="f4", chunks=True)
        dataset.attrs.create("Count", data=headers['npts'])

        # Start time and end times, in a particular string format
        sdf = '%Y-%m-%dT%H:%M:%S.%f+00:00'         # ascii format to always use
        dataset.attrs.create("PartStartTime", data=headers['t0'].strftime(sdf).encode('ascii'), dtype=sdt)
        dataset.attrs.create("PartEndTime", data=headers['t1'].strftime(sdf).encode('ascii'), dtype=sdt)


