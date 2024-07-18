"""
Scripts to load HDF5 files with DAS data,
specifically those with PRODML-style format.

Files written natively by our Silixa iDAS use the PRODML 2.1
standard for headers. Most are not needed for our purposes,
and on top of that, the raw files contain some proprietary information
and generally should not be shared openly.

These scripts are designed to read and write the only relevant/useful headers.
Files newly written do NOT consitute the full PRODML standard, but at least
variable names are the same, of those we keep. The same reader can be used 
for raw PRODML files or for our converted / stripped files.

Daniel Bowden, ETH Zürich
daniel.bowden@erdw.ethz.ch
Last updated: Mar 2024
"""

import os

import h5py
import numpy as np
from datetime import datetime, timedelta
import glob
import h5py
import os
from re import split

l_fields = []
l_attrs = []

def browse_file_attributes(name):
    l_fields.append(str(name))
    return(None)

def reset_attributes():
    l_fields.clear()
    l_attrs.clear()
    return(None)

def list_headers(file):
   #-- Reset containers for headers      
    reset_attributes()

    with h5py.File(file, "r") as f:
        f.visit(browse_file_attributes)
        for group in l_fields:
            l_k = f[group].attrs.keys()
            for k in l_k:
                l_attrs.append([group,k,f[group].attrs[k]])
                #- Use the print function to see all headers and the general structure
                print(group,k,f[group].attrs[k])    



def load_headers_only(file, verbose=False):
    """
    headers = load_das_h5.load_headers_only( file )

    :Return only the header, metadata, without the large data block
    :OUTPUTS:
    :headers -- dict containing many header information, such as:
    :   fs    -- sample rate in Hz
    :   dx    -- reported channel spacing in meters (may be off by -2%)
    :   lx    -- total fiber length
    :   nchan -- number of channels
    :   npts  -- number of samples
    :   t0    -- datetime object, start time of the file
    :   t1    -- datetime object, end time of the file
    :   d0    -- starting channel distance along fibre (usually the first ~200m are internal to the DAS)
    :   d1    -- end channel distance along fibre
    :   fm    -- fiber multiplier. Requested channel spacing of 1m is actually this distance
    :   unit  -- String of the units of measurment (i.e., strain-rate is (nm/m)/s * Hz/m)
    :   amp_scaling -- by default no scaling; i.e., 1.0
    :                  raw Silixa units are proportional to strain rate, but need scaling
    :                  to be accurate. This variable tracks what scaling value has been
    :                  applied. (see "convert=True" flag on the custom reader below)
    """
    #-- Reset containers for headers      
    reset_attributes()

    with h5py.File(file, "r") as f:
        f.visit(browse_file_attributes)
        for group in l_fields:
            l_k = f[group].attrs.keys()
            for k in l_k:
                l_attrs.append([group,k,f[group].attrs[k]])
                #- Use the print function to see all headers and the general structure
                #print(group,k,f[group].attrs[k])
        headers = dict()
        for attr in l_attrs: 
            if "OutputDataRate" == attr[1]: 
                headers['fs'] = round(attr[-1],ndigits=3)
            if "OriginalDataRate" == attr[1]: 
                headers['fs_orig'] = round(attr[-1],ndigits=3)
            if "SpatialResolution" == attr[1]: 
                #headers['dx'] = int(attr[-1])
                headers['dx'] = float(attr[-1])
            if "MeasureLength" == attr[1]: 
                headers['lx'] = int(attr[-1])
            if "NumberOfLoci" == attr[1]:
                headers['nchan'] = int(attr[-1])
            if "Count" == attr[1]:
                headers['npts'] = int(attr[-1])
            if "PartStartTime" == attr[1]: 
                headers['t0'] = datetime.strptime(attr[-1].decode('ascii'),'%Y-%m-%dT%H:%M:%S.%f+00:00')
                #t0 = UTCDateTime(attr[-1])
                #t0 = t0._get_datetime()
            if "PartEndTime" == attr[1]: 
                headers['t1'] = datetime.strptime(attr[-1].decode('ascii'),'%Y-%m-%dT%H:%M:%S.%f+00:00')
            if "StartDistance" == attr[1]:
                headers['d0'] = float(attr[-1])
            if "OriginalStartDistance" == attr[1]:
                headers['d0_absolute'] = float(attr[-1])
            if "StopDistance" == attr[1]:
                headers['d1'] = float(attr[-1])
            if "FibreLengthMultiplier" == attr[1]:
                headers['fm'] = float(attr[-1])
            if "RawDataUnit" == attr[1]:
                headers['unit'] = attr[-1].decode('ascii')
            if "GaugeLength" == attr[1]:
                headers['gauge'] = float(attr[-1])

            ## Custom added headers, in case files were modified and then re-written
            if "OriginalDataRate" == attr[1]:
                headers['fs_orig'] = float(attr[-1])
            if "AmpScaling" == attr[1]:
                headers['amp_scaling'] = float(attr[-1])
           
        ## Add those custom defined headers if not already present
        if('fs_orig' not in headers.keys()):
            headers['fs_orig'] = headers['fs']
        if('amp_scaling' not in headers.keys()):
            headers['amp_scaling'] = 1.0
        if('d0_absolute' not in headers.keys()):
            headers['d0_absolute'] = headers['d0']
		## Some users may cut the start point (only pull d>0), but we note the "absolute" original
		##  for counting channels later, just in case.

        if(verbose):
            print("Loading headers from: {0}, start: {1}, end: {2}".format(file, headers['t0'], headers['t1']))   

    ## metadata problem: timestamps missing. Check, and get time from filename:
    if(headers['t0'] == datetime(1970,1,1,0,0,0)):
        filesplit = split('UTC_',file)
        file_start = datetime.strptime(filesplit[1],'%Y%m%d_%H%M%S.%f.h5')
        headers['t0'] = file_start
        headers['t1'] = file_start + timedelta(seconds=(30-1/headers['fs']))
        if(verbose):
            print('Header issue, missing timestamp. Filling based on filename:')
            print('{0}'.format(file))
            print('   t0 = {0}'.format(headers['t0']))
            print('   t1 = {0}'.format(headers['t1']))

    return(headers)

def load_file(file, convert=False, return_axis=True, verbose=False):
    """
    data, headers, axis = load_das_h5.load_file( file )

    :Load a single HDF5 file, based solely on the filename
    :OUTPUTS:
    :data    -- 2D numpy array [ num_samples, num_channels ]
    :headers -- dict of header information
    :axis    -- dict of constructed axis vectors: 
    :            - tt         -- timesteps in seconds
    :            - date_times -- absolute datetime objects
    :            - dd         -- channel distances
    """

    headers = load_headers_only(file,verbose=verbose)
    #-- (Using this other function to load headers is cleaner, but currently
    #--   it means the file is opened and closed twice)

    with h5py.File(file, "r") as f:
        data = f["Acquisition/Raw[0]/RawData"][:]


    #-- Convert everything
    if(convert==True):
        #-- DCB note: Silixa raw PRODML files report units as strain rate, even when 
        #--  the following conversion has not yet been applied. Up to the user to mark 
        #--  whether data has actually been converted yet or not...
        #-- We started writing a custom header defining any amplitude scaling that has been 
        #--  applied (i.e., 1.0 if no scaling, some number otherwise)
        scale = True
        if('amp_scaling' in headers):
            #-- Check if amplitude has been scaled yet or is still 1.0
            #--  (close to 1.0, possible rounding errors with read/write)
            if(np.abs(headers['amp_scaling']-1.0)>0.0001):
                 print("WARNING: flag \"convert\" is TRUE, but units are already scaled somehow")
                 print("   Doing nothing regarding conversion.")
                 scale = False
       
        #-- DCB note: the sample rate (fs) used below is the ORIGINAL sample rate
        #--  at which data is recorded. If files have been downsampled, use the custom
        #--  header['fs_orig']
        if(scale):
            if('fs_orig' in headers.keys()):
               fs = headers['fs_orig']

            data = 116. * data / 8192. * fs / 10. 
            headers['amp_scaling'] = 116. / 8192. * fs / 10.
            headers['unit'] = '(nm/m)/s'
            if(verbose):
                print("Converted to strain rate!")


    ##########################
    #-- One could stop here...
    #-- But we'll compute vectors for channel spacing and timing
    if(return_axis):
        axis = dict()

        #-- For the SWP Silixa iDAS, 1m channel spacing is actually ~1.02. See "FibreLengthMultiplier"
        #-- For the SWP Silixa iDAS, the first ~200m of fiber is internally inside the interrogator
        #--    See "StartDistance" for the exact value (which can change depending on settings)
        d0 = headers['d0']
        d1 = headers['d1']
        fm = headers['fm']
        dx = headers['dx']

        #dd = np.arange(d0, d1+dx*fm, dx*fm) 
        #dd = np.arange(d0, d1, dx*fm) 
        #-- NOTE: The end channel location calculated this way can differ from "d1" ("StopDistance") by 
        #--  order 0.01m, especially when accumulated over a long (>30km) fibre. Rounding erorrs?
        #-- Possibly one would need to add/subtract one index to dd to get the dimensions correct.
        #-- Temporary solution? Add 1/2 a sample to the end target of np.arange, to make sure the final sample is reached
        dd = np.arange(d0, d1+dx*fm/2, dx*fm) 
        if(len(dd)!=np.shape(data)[1]):
            print("WARNING. Data returned does not match channel spacing calculated from metadata (dd)")
            print("dd is {0} channels from {1} to {2}, spacing {3}".format(len(dd),dd[0],dd[-1], dd[1]-dd[0]))
            print("Reported metadata: d0: {0}, d1:{1}".format(d0,d1))
            print("data is {0} channels".format(np.shape(data)[1]))
        axis['dd'] = dd

        #-- Set up vectors of time samples
        fs = headers['fs']
        t0 = headers['t0']
        tt = np.arange(0, np.shape(data)[0]/fs, 1.0/fs) 
        if(len(tt) != np.shape(data)[0]):
            tt = tt[0:np.shape(data)[0]]
        date_times = [None]*len(tt)
        for i,t in enumerate(tt):
            #date_times.append((final_t0+t)._get_datetime())
            date_times[i] = t0 + timedelta(seconds=t)
        axis['tt'] = tt
        axis['date_times'] = date_times

    #print(file, t0, t1)     
    if(return_axis):
        return data, headers, axis
    else:
        return data, headers


def make_file_list(t_start, t_end, input_dir, verbose=False):
    """
    consider_files = make_file_list(t_start, t_end, input_dir)
    :
    :Given a time-window, make a list of all files that might be needed to load.
    : -Plan 1: it will consider all things in the given "input_dir" matching *.h5
    : -Plan 2: it will return all h5 files in directories according to the days needed,
    :  assuming directories are formatted like YYYY_MM_DD or YYYYMMDD
    : -Plan 3: it will look for directories with keyword "epoch" somewhere and search
    :  for day directories within that.
    :
    :After making a list of all possible files, it will refine the search parameters to
    : within the nearest +/-1 minute. Files *must* be named like "...YYYYMMDD_HHMM...h5"
    :
    :The final step would be to actually open file headers / metadata and decide whether 
    : a given file is needed. This is left to a different function.
    : 
    """
    #-- All the files in this directory
    all_files = sorted(glob.glob(input_dir+"/*.h5"))

    #----------------------------
    #-- Check further, if we're requesting a span of more than one day
    #-- e.g., if t_end_dir is onto the next day or multiple days
    #-- We'll check a potential range of days. Specify the start day and end day in case they're different
    t_start_dir = (t_start - timedelta(minutes=3)).strftime('/%Y_%m_%d/')
    t_start_day  = datetime.strptime(t_start_dir, '/%Y_%m_%d/')
    t_end_dir   = (t_end + timedelta(minutes=3)).strftime('/%Y_%m_%d/')
    t_end_day  = datetime.strptime(t_end_dir, '/%Y_%m_%d/')
    t_step_day = t_start_day
    while(t_step_day <= t_end_day):
        t_step_dir = t_step_day.strftime('/%Y_%m_%d/')
        #print(t_step_dir)
        if(os.path.exists(input_dir+t_step_dir)):
            all_files += sorted(glob.glob(input_dir+t_step_dir+"*.h5"))
        t_step_day += timedelta(days=1)

    #-- We'll repeat the above but with a different format of directory name (%Y%m%d vs %Y_%m_%d)
    t_start_dir = (t_start - timedelta(minutes=3)).strftime('/%Y%m%d/')
    t_start_day  = datetime.strptime(t_start_dir, '/%Y%m%d/')
    t_end_dir   = (t_end + timedelta(minutes=3)).strftime('/%Y%m%d/')
    t_end_day  = datetime.strptime(t_end_dir, '/%Y%m%d/')
    t_step_day = t_start_day
    while(t_step_day <= t_end_day):
        t_step_dir = t_step_day.strftime('/%Y%m%d/')
        # print(t_step_dir)
        if(os.path.exists(input_dir+t_step_dir)):
            all_files += sorted(glob.glob(input_dir+t_step_dir+"*.h5"))
        t_step_day += timedelta(days=1)

    #-- We'll repeat the above but with a different format of directory name (%Y-%m-%d vs %Y_%m_%d)
    t_start_dir = (t_start - timedelta(minutes=3)).strftime('/%Y-%m-%d/')
    t_start_day  = datetime.strptime(t_start_dir, '/%Y-%m-%d/')
    t_end_dir   = (t_end + timedelta(minutes=3)).strftime('/%Y-%m-%d/')
    t_end_day  = datetime.strptime(t_end_dir, '/%Y-%m-%d/')
    t_step_day = t_start_day
    while(t_step_day <= t_end_day):
        t_step_dir = t_step_day.strftime('/%Y-%m-%d/')
        #print(t_step_dir)
        if(os.path.exists(input_dir+t_step_dir)):
            all_files += sorted(glob.glob(input_dir+t_step_dir+"*.h5"))
        t_step_day += timedelta(days=1)

    
    #----------------------------
    #-- Check for subdirectories based on epochs, and see if we can find files within accordingly
    t_start_dir = (t_start - timedelta(minutes=3)).strftime('/%Y%m%d/')
    t_start_day  = datetime.strptime(t_start_dir, '/%Y%m%d/')
    t_end_dir   = (t_end + timedelta(minutes=3)).strftime('/%Y%m%d/')
    t_end_day  = datetime.strptime(t_end_dir, '/%Y%m%d/')

    epoch_dirs = sorted(glob.glob(input_dir+"/*epoch*"))

    t_step_day = t_start_day
    while(t_step_day <= t_end_day):
        t_step_dir = t_step_day.strftime('/%Y%m%d/')
        #print(t_step_dir)
    
        for this_epoch_dir in epoch_dirs:
            if(os.path.exists(this_epoch_dir+t_step_dir)):
                all_files += sorted(glob.glob(this_epoch_dir+t_step_dir+"*.h5"))
        t_step_day += timedelta(days=1)

    #-- We'll repeat the above but with a different format of directory name (%Y%m%d vs %Y_%m_%d)
    t_start_dir = (t_start - timedelta(minutes=3)).strftime('/%Y_%m_%d/')
    t_start_day  = datetime.strptime(t_start_dir, '/%Y_%m_%d/')
    t_end_dir   = (t_end + timedelta(minutes=3)).strftime('/%Y_%m_%d/')
    t_end_day  = datetime.strptime(t_end_dir, '/%Y_%m_%d/')
    t_step_day = t_start_day
    while(t_step_day <= t_end_day):
        t_step_dir = t_step_day.strftime('/%Y_%m_%d/')
        #print(t_step_dir)
        for this_epoch_dir in epoch_dirs:
            if(os.path.exists(this_epoch_dir+t_step_dir)):
                all_files += sorted(glob.glob(this_epoch_dir+t_step_dir+"*.h5"))
        t_step_day += timedelta(days=1)

    #-- We'll repeat the above but with a different format of directory name (%Y-%m-%d vs %Y_%m_%d)
    t_start_dir = (t_start - timedelta(minutes=3)).strftime('/%Y-%m-%d/')
    t_start_day  = datetime.strptime(t_start_dir, '/%Y-%m-%d/')
    t_end_dir   = (t_end + timedelta(minutes=3)).strftime('/%Y-%m-%d/')
    t_end_day  = datetime.strptime(t_end_dir, '/%Y-%m-%d/')
    t_step_day = t_start_day
    while(t_step_day <= t_end_day):
        t_step_dir = t_step_day.strftime('/%Y-%m-%d/')
        #print(t_step_dir)
        for this_epoch_dir in epoch_dirs:
            if(os.path.exists(this_epoch_dir+t_step_dir)):
                all_files += sorted(glob.glob(this_epoch_dir+t_step_dir+"*.h5"))
        t_step_day += timedelta(days=1)

    
    #----------------------------
    #-- Hopefully we've found all the files. Sort them before moving on.
    all_files = sorted(all_files)

    #----------------------------
    #-- To avoid loading every header of every file in the directory, do a quick search of candidates based on filename
    #-- We're only searching based broadly on the minutes listed, rather than trying to match filenames exactly
    consider_files = []
    t_step = t_start-timedelta(minutes=3)
    t_end_check = t_end+timedelta(minutes=3)
    if(verbose):
        print("Confirming a list of files, in loop over:")
        print(t_step)
        print(t_end_check)
    while(t_step<=t_end_check):
        ## glob seems slower than my for loop over the previously indexed/globbed array
        #consider_files += glob.glob("{0}/*{1}*.h5".format(input_dir,t_step.strftime('%Y%m%d_%H%M')))

        #-- Look for a filename that has the correct minute timestamp 
        consider_string = "{0}".format(t_step.strftime('%Y%m%d_%H%M'))
        for file in all_files:
            file_only = split('/',file)[-1]
            if(consider_string in file_only):
                consider_files += [file]
        t_step += timedelta(minutes=1)

    #-- If using longer (i.e., 10 min blocks), then we might miss a file this way
    #-- Kludgy fix? Repeat with hour resolution if needed
    if(len(consider_files)==0):
        if(verbose):
            print("Warning: no valid files found for consideration yet, trying to find anything within the day")
        t_step = t_start
        while(t_step<t_end):
            consider_string = "{0}".format(t_step.strftime('%Y%m%d'))
            # print(consider_string)
            for file in all_files:
                if(consider_string in file):
                    consider_files += [file]
            t_step += timedelta(hours=1)

    if(verbose):
        print("Considering the following files to look for our data:")
        print(consider_files)
    if(len(consider_files)==0):
        print("ERROR! No files found to load.")
        print(" This is very likely because the script only looks in subdirectories in a rigid way:")
        print(" /YYYYMMDD_epoch1/YYYYMMDD/*h5   or  /YYYY_MM_DD/*h5")
        print(" You may need to change the reader script to match your directory structure")
        return

    return consider_files

def load_das_custom(t_start, t_end, d_start=0, d_end=0, ichan=[], mapchan=[], convert=False, verbose=False, input_dir='./', return_axis=True, nth_channel=1):
    """
    data, heades, axis = load_das_custom(t_start, t_end, d_start=0, d_end=0, convert=False, verbose=False, input_dir='./')
    :Custom function to load files in a flexible way. 
    :User defines a start time and end time, and data will be returned accordingly.
    :This is NOT sophisticated...
    : -data filenames must contain the date and time in a certain format
    : -filenames must be sortable according to date
    : -gaps in the data will not be handled correctly
    : -it *should* be able to handle requests that span multiple days, but no guarantees
    : -etc.
    :
    :It is intended to be faster than loading full blocks. By knowing the indices of data needed,
    : one can avoid loading the entire data block in HDF5.
    :
    :INPUTS:
    :t_start -- datetime object the request should start at 
    :              (e.g. t_start = datetime.strptime('2021/10/12 09:24:30.0', '%Y/%m/%d %H:%M:%S.%f'))
    :t_end   -- datetime object end
    :d_start -- (optional) fibre distance at which to start returning data
    :d_end   -- (optional) fibre distance at which to stop
    :ichan   -- (optional) np array of specific indices to load (not compatible with d_start/d_end) 
                ichan refers to the simplest, absolute index within an HDF5 block.
    :mapchan -- (optional) np array of specific indices to load (not compatible with d_start/d_end) 
                mapchan considers d=0 to be index=0, thus accounting for potentially different negative 
                distances within the iDAS.

    :convert -- (optional) boolean to convert to strain rate if not already done
    :            WARNING: This requires knowing the sample rate of the raw data.
    :            If data had been downsampled but not converted, you will need to change "fs" in that conversion
    :verbose -- (optional) boolean to print more information about what is being loaded
    :input_dir -- (optional) string of directory in which to look for data
    :
    :OUTPUTS:
    :data    -- 2D numpy array [ num_samples, num_channels ]
    :headers -- dict of header information
    :axis    -- dict of constructed axis vectors: 
    :            - tt         -- timesteps in seconds
    :            - date_times -- absolute datetime objects
    :            - dd         -- channel distances
    """


    ##############################################
    ## STEP 1: Find possible files that need loading
    ##############################################
    consider_files = make_file_list(t_start, t_end, input_dir, verbose=verbose)

    
    ##############################################
    ## STEP 2: Skim headers from each file considered, 
    ##  if timewindow is valid then load the data as we go
    ##############################################
    if(verbose):
        print("----------------------------------------------")
        print("----- Requested: {0} to {1}".format(t_start.strftime('%Y-%m-%d %H:%M:%S'), t_end.strftime('%Y-%m-%d %H:%M:%S')))
        print("----- Checking through files, loading as we go")
        print("----------------------------------------------")
        

    found_data_yet = False
    for filename in consider_files:

        #-- Reset containers for headers      
        reset_attributes()
        
        #-- Open the headers of each file and look at the times
        headers = load_headers_only(filename, verbose=verbose)
        t0 = headers['t0']
        t1 = headers['t1']
        npts = headers['npts']
        nchan= headers['nchan']
        fs = headers['fs']
        dx = headers['dx']
        fm = headers['fm']
        d0 = headers['d0']
        d1 = headers['d1']


        #-- Check if either t0 or t1 (or both) lies within the desired bounds of t_start and t_end
        #print(t_start.strftime('%Y-%m-%d %H:%M:%S.%f'))
        #print(t_end.strftime('%Y-%m-%d %H:%M:%S.%f'))
        #print(t0.strftime('%Y-%m-%d %H:%M:%S.%f'))
        #print(t1.strftime('%Y-%m-%d %H:%M:%S.%f'))
        ## Do we use any of the data in this file? Consider the 4 cases for requests
        ##   t_0                                               t_1
        ##    |***************file******************************|
        ##
        ## |---case1---|
        ## ts         te
        ##                                                    |----case2---|
        ##                                                    ts          te
        ##                      |---case3---|
        ##                      ts          te
        ##
        ## |-------------------------------case4----------------------------|
        ## ts                                                               te
        ##
        ##      # case 1 & 4                    # case 2 & 4             # case 3
        if( (t_start<t0 and t0<t_end) or (t_start<t1 and t1<t_end)) or (t0<t_start and t_start<t1) or (t0==t_start) or (t1==t_end):
            if(verbose):
                print("Use it!")
            
            #-- Define the time index from which to pull
            tt = np.arange(0, npts/fs, 1.0/fs) 
            #-- Initial values: full range
            i_pull_start = 0
            i_pull_end   = npts-1
            
            if(t_start>t0):    # See if we should pull less on the front end
                t_rel = (t_start-t0).total_seconds()  
                i_pull_start = np.argmin(np.abs(tt-t_rel))
                if(verbose):
                    print("~~~ cut front ~~~~~~~~")
                    print(filename)
                    print("Requested start: {0}".format(t_start))
                    print("This file start: {0}".format(t0))
                    print("Starting at {0} seconds in".format(tt[i_pull_start]))
                
            if(t1>t_end):      # See if we should cut some off the end
                t_rel = (t_end-t0).total_seconds()  
                i_pull_end = np.argmin(np.abs(tt-t_rel))
                if(verbose):
                    print("~~~ cut end   ~~~~~~~~")
                    print(filename)
                    print("Requested end: {0}".format(t_end))
                    print("This file end: {0}".format(t1))
                    print("Cutting at {0} seconds in".format(tt[i_pull_end]))                
            
            #-- Set up axis of distances / channels
            #dd = np.arange(d0, d1+dx*fm, dx*fm) 
            #dd = np.arange(d0, d1, dx*fm) 
            #-- NOTE: The end channel location calculated this way can differ from "d1" ("StopDistance") by 
            #--  order 0.01m, especially when accumulated over a long (>30km) fibre. Rounding erorrs?
            #-- Possibly one would need to add/subtract one index to dd to get the dimensions correct.
            #-- Temporary solution? Add 1/2 a sample to the end target of np.arange, to make sure the final sample is reached
            dd = np.arange(d0, d1+dx*fm/2, dx*fm) 

            #-- Open the file again for actual reading of data
            with h5py.File(filename, "r") as f:
                #-- Did the user specify any cutting along distance axis?
                #-- TODO: Logic is a bit rigid, requiring d_end to be specified and -then- check for nth_channel downsample.
                #--        Surely a better way is possible...
                if(d_end>0):
                    id1 = np.argmin(np.abs(dd-d_start))
                    id2 = np.argmin(np.abs(dd-d_end))
                    if(nth_channel>1):
                        d_pull = np.arange(id1,id2+1)
                        d_pull = d_pull[::nth_channel]
                        if(verbose):
                            print("pulling every {0} traces".format(nth_channel))
                        data_tmp = f["Acquisition/Raw[0]/RawData"][i_pull_start:i_pull_end+1, d_pull]
                        dd = dd[d_pull]
                        dx = dx*nth_channel
                        if(verbose):
                            print("New dx = {0}".format(dx))
                    else:
                        data_tmp = f["Acquisition/Raw[0]/RawData"][i_pull_start:i_pull_end+1, id1:id2+1]
                        dd = dd[id1:id2+1]

                    if(verbose):
                        print("Returning channels over distances: {0}  --  {1}".format(dd[0],dd[-1]))
                        print("data: {0} x {1},  dd: {2}".format(np.shape(data_tmp)[0],np.shape(data_tmp)[1],np.shape(dd)))

                #-- Did the user specify an array of specific indices?
                #--   ichan would refer to the simplest, absolute index within an HDF5 block.
                #--   mapchan considers d=0 to be index=0, thus accounting for potentially different negative distances within the iDAS.
                elif(len(ichan)>0):
                    dd = dd[ichan]
                    data_tmp = f["Acquisition/Raw[0]/RawData"][i_pull_start:i_pull_end+1, ichan]

                elif(len(mapchan)>0):
                    zero_correct = -int(np.round(headers['d0'] / (headers['dx']*headers['fm'])))
                    dd = dd[mapchan+zero_correct].astype(int)
                    data_tmp = f["Acquisition/Raw[0]/RawData"][i_pull_start:i_pull_end+1, mapchan+zero_correct]


                #-- Otherwise just return all channels
                else:
                    data_tmp = f["Acquisition/Raw[0]/RawData"][i_pull_start:i_pull_end+1,:]
                    if(verbose):
                        print("Returning all channels")
                
            # See if we've defined data yet, otherwise concatenate with previous
            if(found_data_yet == False):
                data = data_tmp
                final_t0 = t0 + timedelta(seconds=tt[i_pull_start])
                final_t1 = t0 + timedelta(seconds=tt[i_pull_end])
                found_data_yet = True
            else:
                data = np.concatenate((data,data_tmp))
                final_t1 = t0 + timedelta(seconds=tt[i_pull_end])
                
            #print(np.shape(data))
            # NOTE, growing large arrays like this is slow.
            # Better would be to determine all the useful times, 
            #  init an emtpy array of the correct size,
            #  and then fill elements
                
                
    #print(final_t0)
    #print(final_t1)
    try:
        data
    except:
        print("ERROR! No data was loaded")



    ###################################
    ## STEP 3: Compute vectors for channel spacing and timing
    ## and update headers
    ##################################
    #-- Update headers to reflect the pulled data
    headers['d0'] = dd[0]
    headers['d1'] = dd[-1]
    headers['t0'] = final_t0
    headers['t1'] = final_t1
    headers['npts'] = np.shape(data)[0]
    headers['nchan'] = np.shape(data)[1]
    if(d_end>0 and nth_channel > 0):
        dx = dx*nth_channel
        if(verbose):
            print("New dx = {0}".format(dx))
    headers['dx'] = dx


    axis = dict()

    #-- dd was already created when looking up indices above
    #-- (No need to recalculate it at the end; this function assumes each file read has the same channels)
    if(len(dd)!=np.shape(data)[1]):
        print("WARNING. Data returned does not match channel spacing calculated from metadata (dd)")
        print("dd is {0} channels from {1} to {2}, spacing {3}".format(len(dd),dd[0],dd[-1], dd[1]-dd[0]))
        print("Reported metadata: d0: {0}, d1:{1}".format(d0,d1))
        print("data is {0} channels".format(np.shape(data)[1]))
    axis['dd'] = dd


    if(return_axis or verbose):
        #-- Convert time-axis into numpy-happy date time objects
        # (also, not using the obspy UTCdatetime object)
        tt = np.arange(0, np.shape(data)[0]/fs, 1.0/fs) 
        if(len(tt) != np.shape(data)[0]):
            tt = tt[0:np.shape(data)[0]]
        date_times = [None]*len(tt)
        for i,t in enumerate(tt):
            #date_times.append((final_t0+t)._get_datetime())
            date_times[i] = final_t0 + timedelta(seconds=t)
        axis['tt'] = tt
        axis['date_times'] = date_times


        
    if(verbose):
        print("----------------------------------------------")
        print("Requested time chunk: {0}  --  {1}".format(t_start.strftime('%Y/%m/%d %H:%M:%S.%f'),t_end.strftime('%Y/%m/%d %H:%M:%S.%f')))
        print("Returning time chunk: {0}  --  {1}".format(date_times[0].strftime('%Y/%m/%d %H:%M:%S.%f'),date_times[-1].strftime('%Y/%m/%d %H:%M:%S.%f')))
        print("Data[ {0} samples, {1} channels ]".format(np.shape(data)[0],np.shape(data)[1]))
        print(headers)


    #-- Convert everything
    if(convert==True):
        #-- DCB note: Silixa raw PRODML files report units as strain rate, even when 
        #--  the following conversion has not yet been applied. Up to the user to mark 
        #--  whether data has actually been converted yet or not...
        #-- We started writing a custom header defining any amplitude scaling that has been 
        #--  applied (i.e., 1.0 if no scaling, some number otherwise)
        scale = True
        if('amp_scaling' in headers):
            #-- Check if amplitude has been scaled yet or is still 1.0
            #--  (close to 1.0, possible rounding errors with read/write)
            if(np.abs(headers['amp_scaling']-1.0)>0.0001):
                 print("WARNING: flag \"convert\" is TRUE, but units are already scaled somehow")
                 print("   Doing nothing regarding conversion.")
                 scale = False
        if(scale==True):
            #-- DCB note: the sample rate (fs) used below is the ORIGINAL sample rate
            #--  at which data is recorded. If files have been downsampled, use the custom
            #--  header['fs_orig']
            if('fs_orig' in headers.keys()):
               fs = headers['fs_orig']

            data = 116. * data / 8192. * fs / 10. 
            headers['amp_scaling'] = 116. / 8192. * fs / 10.
            headers['unit'] = '(nm/m)/s'
            if(verbose):
                print("Converted to strain rate!")

    if(return_axis):
        return data, headers, axis
    else:   
        return data, headers


