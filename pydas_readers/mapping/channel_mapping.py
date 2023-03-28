import numpy as np
import pandas as pd

MAPPED_FILENAME = "./pydas_readers/mapping/Channel_mapping_information_catalouge_v4.1.csv"
"""
Hardcoded the CSV file with lat/lon info. Not ideal practice! 
But one can update the path & name with:
  channel_mapping.update_filename("new_path")
"""

def update_filename(new_filename):
    global MAPPED_FILENAME
    MAPPED_FILENAME = new_filename

def print_filename():
    print(MAPPED_FILENAME)

def get_mapping(data_type="raw", chan_spacing=8, d_start=0, d_end=0, nth_channel=1):
    """
    mapping = channel_mapping.get_mapping(data_type="mapped", chan_spacing=8, d_start=0, d_end=0, nth_channel=1)
    :
    :Loads a CSV of channel locations. Most likely the needs of every project are slightly different and so the 
    : column-names and format here will need to change.
    :
    :OUTPUT:
    :The goal is to return a library, e.g. "mapping" with certain fields:
    :   mapping['ii']  -- the index of channels in the CSV file (0:n)
    :   mapping['i0']  -- an index of channels, adjusted that distance=0 is index=0.
    :                       (for the Silixa iDAS, some negative distance is always returned, 
                            as in fiber inside the box. That's ok by itself, but for different epochs,
                            we can have a different negative distance. Things can be synchronized 
                            by using 'i0' instead of 'ii')
        mapping['lat'] -- lat
        mapping['lon'] -- lon
        mapping['dd']  -- linear distance along the fiber
        mapping['dx']  -- channel spacing. This can vary along the line! 
                            (One might set the iDAS to use dx=2 meters, for example, but because
                            the lat/lon mapping is imprecise, a given segment of cable might squish or stretch
                            channels a little bit. So one might have dx=1.9 for one segment and 2.1 for another.)
        mapping['flags'] -- any notes, if desired.
    :
    :INPUTS:
    :As inputs, the field "data_type" might refer to different things:
    :   data_type = "raw" just returns a constant dx and all channels.
    :   data_type = "mapped" returns all channels for which a mapping assignments was possible
    :   data_type = "clean" returns channels for which we are more certain are useable.
    :
    :   chan_spacing -- by default, this might match one's CSV. But in the case the CSV was mapped
                        at a higher resolution (e.g. 2m), but a given datafile uses something different,
                        this will appropriately downsample. Note the longer spacing must be a multiple 
                        of the lower.
        d_start and d_end -- specify a smaller segment of channels, for faster loading of HDF5 data later.
        nth_channel       -- returns only each n'th channel. This is different than chan_spacing, because we
                             need to know the 'ii' or 'i0' index. For nth_channel=4, 'ii' would be 0,4,8,etc.
                             so one can refer to the indices in an HDF5 block.
    :
    :USAGE:
    :The goal is to be able to specify one's distance, channel spacing, etc. to get: mapping['i0'], 
    : and then use this with: load_das_h5.load_das_custom(..., mapchan = mapping['i0'],...)
    : thus loading the exact indices desired from an HDF5 block.
    """

    mapping = dict()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #-- LOAD CSV, PICK TYPE
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #-- Open a pandas dataframe
    df = pd.read_csv(MAPPED_FILENAME,skiprows=10,)

    #-- Case: You want all the data. No lat/lon will be given.
    if(data_type == "raw"):
        mapping['dd'] = df['Distance_IU (m)'].to_numpy()
        mapping['ii'] = df['Chan Number'].to_numpy()

    #-- Case: You want all channels that have a lat/lon assigned
    #--  Note: using values from Distance_Calculated that are not "None", 
    #--  which differs fom the "Keep" flag
    elif(data_type == "mapped"):
        dd = df['Distance_mapped (m)'] 
        mapping['dd'] =  dd[dd != "None"].to_numpy(dtype=float)
        mapping['ii'] =  df['Chan Number'][dd != "None"].to_numpy(dtype=int)
        mapping['i0'] =  df['Chan Number Zeroed'][dd != "None"].to_numpy(dtype=int)
        mapping['dx'] =  df['actual_dx'][dd != "None"].to_numpy(dtype=float)
        mapping['lat'] = df['Latitude'][dd != "None"].to_numpy(dtype=float)
        mapping['lon'] = df['Longitude'][dd != "None"].to_numpy(dtype=float)
        mapping['flags'] = df['Flags/Hammers'][dd != "None"].to_list()

    elif(data_type == "clean"):
        iuse = []
        for i in range(len(df['Keep (True/False)'])):
            if(df['Good - not noise (True/False/None)'][i]=="TRUE" and df['Distance_mapped (m)'][i] != "None"):
                iuse = np.append(iuse,i)

        mapping['dd'] = df['Distance_mapped (m)'][iuse].to_numpy(dtype=float)
        mapping['ii'] =  df['Chan Number'][iuse].to_numpy(dtype=int)
        mapping['i0'] =  df['Chan Number Zeroed'][iuse].to_numpy(dtype=int)
        mapping['dx'] =  df['actual_dx'][iuse].to_numpy(dtype=float)
        mapping['lat'] = df['Latitude'][iuse].to_numpy(dtype=float)
        mapping['lon'] = df['Longitude'][iuse].to_numpy(dtype=float)
        mapping['flags'] = df['Flags/Hammers'][iuse].to_list()     
 

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #-- MODIFICATIONS
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #-- Channel mapping was performed on a 2m resolution. 
    #-- If something longer is used (e.g. 8m), scale it down
    if(chan_spacing != 2):
        factor = int(chan_spacing/2)
        istart=0
        #print(factor)
        mapping['dd'] = mapping['dd'][istart::factor]
        mapping['ii'] = mapping['ii'][istart::factor]/factor
        mapping['ii'] = mapping['ii'].astype(int)
        if(data_type != "raw"):
            mapping['i0'] = mapping['i0'][istart::factor]/factor
            mapping['i0'] = mapping['i0'].astype(int)
            mapping['dx'] = mapping['dx'][::factor]*factor
            mapping['lat'] = mapping['lat'][::factor]
            mapping['lon'] = mapping['lon'][::factor]


    #-- Further channel subdivision requested?
    #--  Define "d_pull" as vector of indices to select
    d_pull = []
    if(nth_channel>1):
        d_pull = np.arange(0, len(mapping['dd']))
        d_pull = d_pull[::nth_channel]

        #-- Apply it
        mapping['dd'] = mapping['dd'][d_pull]
        mapping['ii'] = mapping['ii'][d_pull]
        if(data_type != "raw"):
            mapping['i0'] = mapping['i0'][d_pull]
            mapping['dx'] = mapping['dx'][d_pull]
            mapping['lat'] = mapping['lat'][d_pull]
            mapping['lon'] = mapping['lon'][d_pull]
            #mapping['flags'] = mapping['flags'][d_pull]


    #-- To keep track of distance modifications, just count off from zero.
    mapping['index'] = np.arange(len(mapping['ii']))

    #-- Subset requested?
    if(d_end>0):
        d_pull = []
        id1 = np.searchsorted(mapping['dd'],[d_start,],side='right')[0]   
        id2 = np.searchsorted(mapping['dd'],[d_end,],side='right')[0]-1
        d_pull = np.arange(id1, id2+1)

        #-- Apply it
        mapping['dd'] = mapping['dd'][d_pull]
        mapping['ii'] = mapping['ii'][d_pull]
        mapping['index'] = mapping['index'][d_pull]
        if(data_type != "raw"):
            mapping['i0'] = mapping['i0'][d_pull]
            mapping['dx'] = mapping['dx'][d_pull]
            mapping['lat'] = mapping['lat'][d_pull]
            mapping['lon'] = mapping['lon'][d_pull]
            #mapping['flags'] = mapping['flags'][d_pull]


        
    ##-- Further channel subdivision requested?
    ##--  Define "d_pull" as vector of indices to select
    #d_pull = []
    #if(nth_channel>1):
    #    d_pull = np.arange(0, len(mapping['dd']))
    #    d_pull = d_pull[::nth_channel]

    ##-- Subset requested?
    #if(d_end>0):
    #    id1 = np.argmin(np.abs(mapping['dd']-d_start))
    #    id2 = np.argmin(np.abs(mapping['dd']-d_end))
    #    d_pull = np.arange(id1, id2+1)
    #    if(nth_channel>1):
    #        d_pull = d_pull[::nth_channel]
    #    
    ##-- Apply d_pull, if it exists
    #if(len(d_pull)>0):
    #    mapping['dd'] = mapping['dd'][d_pull]
    #    mapping['ii'] = mapping['ii'][d_pull]
    #    mapping['index'] = mapping['index'][d_pull]
    #    if(data_type != "raw"):
    #        mapping['i0'] = mapping['i0'][d_pull]
    #        mapping['dx'] = mapping['dx'][d_pull]
    #        mapping['lat'] = mapping['lat'][d_pull]
    #        mapping['lon'] = mapping['lon'][d_pull]
    #        #mapping['flags'] = mapping['flags'][d_pull]


    return mapping

def fix_things(headers,axis,mapping):
    headers['d0'] = mapping['dd'][0]
    headers['d1'] = mapping['dd'][-1]
    #headers['dx'] = np.mean(mapping['dx'])
    headers['dx'] = np.mean(mapping['dd'][1:-1] - mapping['dd'][0:-2])/headers['fm']
    axis['dd'] = mapping['dd']

    return headers, axis


