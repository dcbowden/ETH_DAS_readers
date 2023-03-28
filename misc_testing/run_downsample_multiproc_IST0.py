import h5py
import pathlib
import glob
import numpy as np
from datetime import datetime, timedelta
import multiprocessing
import os
import glob

#-- To import a function on a relative path:
import sys
sys.path.append("../ETH_DAS_readers/")
from pydas_readers.readers import load_das_h5, write_das_h5
from pydas_readers.util import block_filters
from pydas_readers.mapping import channel_mapping

INPUT_DIR = "/mnt/Istanbul_data_NTFS/Istanbul/"
#input_dir = "/media/Neda/Users/danielb/Istanbul/Istanbul_data/"
OUTPUT_BASE_DIR = "/media/Neda/Users/danielb/Istanbul/Istanbul_data_downsampled/"

NPROC = 6
VERBOSE = False
DOWNSAMPLE_FACTOR = 2
OVERWRITE = False
DRYRUN = False 




#-- Get channel mapping details
#--  (here, "mapping" will be a dict with "lat"/"lon"/"dist"/etc. 
#--   and "i0" or "ii" which correspond to the good indices of the data.
#--   Distances less than zero are not mapped, and a few areas of urban dark fiber
#--   needed to be discarded. These are never loaded from the HDF5 file, and left out
#--   of the downsampled dataset.)
#--
#--  If channel mapping is not assigned, or not built into a dict like this,
#--   just omit the "mapchan = ..." argument of load_das_custom()
channel_mapping.update_filename("/media/Neda/Users/danielb/Istanbul/ETH_DAS_readers/pydas_readers/mapping/Channel_mapping_information_catalouge_v4.1.csv")
nth_channel=2
MAPPING = channel_mapping.get_mapping(data_type="mapped", d_start=0, d_end=8000, nth_channel=nth_channel) 
print("number of channels: {0}".format(len(MAPPING['dd'])))

def downsample_file(filename):

    print("Downsampling: {0}".format(filename))
    try:
        headers_target = load_das_h5.load_headers_only(filename, verbose=VERBOSE)

        savedir = OUTPUT_BASE_DIR + filename.split("/")[-3] + "/" + filename.split("/")[-2] + "/"
        new_filename = "{0}/istanbul_downsampled_{1}.h5".format(savedir, headers_target['t0'].strftime('%Y%m%d_%H%M%S.%f')[:-3])
        #print(new_filename)
        if(OVERWRITE==False):
            if(os.path.exists(new_filename)):
                print("{0} already exists! Skipping.".format(filename))
                return 0


        # Attempt to load the new entire block length
        # -1sec at the front and +1sec at the end
        # start and end times are meant to exactly match the "load_headers" output
        target_t0 = headers_target['t0']
        target_t1 = headers_target['t1']
        data, headers, axis = load_das_h5.load_das_custom(target_t0 - timedelta(seconds=29),
                                                                 target_t1 + timedelta(seconds=29),
                                                                 input_dir=INPUT_DIR,
                                                                 convert=False,
                                                                 verbose=VERBOSE,
                                                                 mapchan=MAPPING['i0'],
                                                                 return_axis=True)

        headers, axis = channel_mapping.fix_things(headers, axis, MAPPING)


        #print("d0: {0},  d0_absolute: {1}".format(headers['d0'], headers['d0_absolute']))
        #print("dx: {0}".format(headers['dx']))
    
        # Here:
        #  - consider convert=True, to already correct to nano-strainrate.
        #     (the value in headers['amp_scaling'] will change so you can keep track of this later)
        #
        #    I don't convert here, however, because it could be more precise to do it later.
        #    The HDF5 data are saved at low precision. The less you touch the data before
        #    converting to float64, the better.
        #
        #  - Want to cut the start and end points? Use d_start=___ and d_end=___
        #
        #  - Want to also downsample spatially? The load function has a flag so you can pull 
        #      every n'th channel with indexing like [::n]. "nth_channel"
        #
        #  - In our Istanbul scripts, these arguments like d_start=__ and nth_channel=__
        #      are built into the channel mapping function outside this loop. But one can
        #      call directly here also.
        #
        # We add 29 second before and after, so that any filtering & tapering edge effects don't show in the actual data.
        # TODO: is +/- 29second enough? For DAS month earthquakes, possibly longer periods will be needed, so the buffer
        #  should correspond to 1 minimum period?

        # We will want to chop off that +/- seconds later.
        #  but we don't necessarily KNOW that we could pull from the file before and after
        #  (i.e., for the first file in a directory, the previous file won't exist)
        # So instead we'll look at "date_times" to figure out the correct indices for output later
        # We know we want t0 to t1, because those were directly from the headers of the intended file.
        date_times = axis['date_times']
        i0_out = np.argmin(np.abs(np.array(date_times) - target_t0)) 
        i1_out = np.argmin(np.abs(np.array(date_times) - target_t1)) + 1
        # Added 1 to the index, because we want the final range [i0:i1] to be inclusive
        # and adding it here, specifically, so it's an even number for downsample division

        # samples of the raw data
        #  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        #..|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--|--...
        #           i0                            i1
        #           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        #                  data[i0:i1]
        #
        #
        # downsampled (example for x2)
        #     4     5     6     7     8     9    10    11    12
        #.....|-----|-----|-----|-----|-----|-----|-----|-----|...
        #           i0                            i1
        #           ^~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        #                data2[i0:i1]
        # Check i0: 10/2=5
        # Check i1: 20/2=10
        # Note: the last timestamp for downsampled data (here, "i=9")
        #        will be slightly different, because of the different sample rate


        # In case we didn't load any time after, that +1 in the index is problematic
        if(i1_out > np.shape(data)[0]):
            i1_out = np.shape(data)[0]
            if(VERBOSE):
                print("Warning: less data will return than was requested!")

        if(VERBOSE):
            print("Pre downsample target {0} to {1}".format(i0_out,i1_out))
            print(axis['date_times'][i0_out])
            print(axis['date_times'][i1_out])

        # Adjust indices for the downsampling
        i0_out = int(i0_out/DOWNSAMPLE_FACTOR)
        i1_out = int(i1_out/DOWNSAMPLE_FACTOR)
        if(VERBOSE):
            print("Postdownsample target {0} to {1}".format(i0_out,i1_out))

        # Downsample
        fs = headers['fs']
        data2 = block_filters.chebychev_lowpass_downsamp(data,fs,DOWNSAMPLE_FACTOR,verbose=VERBOSE)
        #
        # Note: the downsample function is based on how obspy does it.
        # We filter before decimating to avoid alisasing problems.
        #
        # The chebyshev type 2 filter ensures NO frequencies are left above the cutoff
        #  whereas the butterworth filter seismologists usually like only starts rolling off
        #  at that cutoff. This chebychev is much sharper, so if you're really interested in
        #  signals near the newly downsampled nyquist... change the filter or pick a higher nyquist.


        # Finally, cut off those +/- seconds if present
        data2 = data2[i0_out:i1_out]
        if(VERBOSE):
            print("**** i0_out: {0}".format(i0_out))
            print("**** i1_out: {0}".format(i1_out))
            print(np.shape(data2))


        # Update headers
        headers['fs'] = fs/DOWNSAMPLE_FACTOR
        headers['fs_orig'] = fs
        headers['npts'] = np.shape(data2)[0]
        headers['t0']  = headers_target['t0']
        # headers['t1']  = headers_target['t1']
        headers['t1'] = headers_target['t0'] +timedelta(seconds= (np.shape(data2)[0] - 1)/(fs/DOWNSAMPLE_FACTOR))
        #headers['dx']  = headers['dx']*nth_channel


        ## See if the directory for this day exists, else create it
        #day_dir = os.path.join(save_dir,headers_target['t0'].strftime('%Y_%m_%d'))
        #if not os.path.exists(day_dir):
        #    os.makedirs(day_dir)
        #new_filename = "{0}/downsampled_{1}.h5".format(day_dir, headers_target['t0'].strftime('%Y%m%d_%H%M%S.%f'))
        if(VERBOSE):
            print(headers)
        write_das_h5.write_block(data2,headers,new_filename)
    except:
        print("ERROR {0}".format(filename))
    

if __name__ == "__main__":
    epochs = sorted(glob.glob(INPUT_DIR+"*epoch*"))
    # epoch 1 has a different settings, deal with it later.
    for epoch_dir in epochs:
    #for epoch_dir in epochs[0:1]:

        #-- Set up that the output goes into a matching epoch dir
        output_epoch_dir = OUTPUT_BASE_DIR + epoch_dir.split('/')[-1] + '/'
        if(not os.path.exists(output_epoch_dir)):
            os.mkdir(output_epoch_dir)

        print(epoch_dir + "-->" + output_epoch_dir )    

        #-- Run downsample script on each file in a given day
        days = sorted(glob.glob(epoch_dir+"/*"))
        for day_dir in days:
        #for day_dir in days[0:1]:

            #-- Set up that the output goes into a matching day dir
            output_day_dir = output_epoch_dir + day_dir.split('/')[-1] + '/'
            if(not os.path.exists(output_day_dir)):
                os.mkdir(output_day_dir)
            print(day_dir + " --> " + output_day_dir )    

            files = sorted(glob.glob(day_dir+"/*h5"))
            if(DRYRUN):
                print("Num files: {0}".format(len(files)))
                continue
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #-- Do the downsampling!
            
            #-- Single processor
            #for this_file in files:
            #for this_file in files[0:2]:
            #    downsample_file(this_file)

            #-- Multiprocessor 
            pool = multiprocessing.Pool(processes=NPROC)
            pool.map(downsample_file,files)
            pool.close()

