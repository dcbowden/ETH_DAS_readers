"""
Basic filtering, assuming one has a contiguous
numpy block of data. Chunks of code taken directly from OBSPY.

For most applications a Butterworth filter is appropriate. The more 
severe Chebychev is included for downsampling applications, where 
aliasing needs to be avoided.

Daniel Bowden, ETH Zürich
daniel.bowden@erdw.ethz.ch
Last updated: Jan 2023
"""

from scipy.signal import butter, lfilter, filtfilt, firwin, resample, hilbert
import scipy.signal as signal
from scipy.signal import (cheb2ord, cheby2, convolve, get_window, iirfilter,
                          remez)
from scipy.signal import sosfilt, sosfiltfilt
from scipy.signal import zpk2sos
import numpy as np
from datetime import datetime, timedelta
#from obspy.core import UTCDateTime
import glob
#import h5py
import pathlib
import sys
import os

from pydas_readers.util import block_cleaning


def block_bandpass(data, freqmin, freqmax, df, corners=4, zerophase=False, taper=0, verbose=False):
    """
    Butterworth-Bandpass Filter. Taken directly from OBSPY
    

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter. 2D numpy array [ npts, nchan ]
                  OR a 1D numpy array [ npts, ]
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :param taper: Value between 0 and 0.5 (i.e., 0.01 means 1%)
        Linear taper the edges in time-domain
    :return: Filtered data.
    """
    if(verbose):
        print("Filtering {0}Hz to {1}Hz".format(freqmin,freqmax))
    if(data.dtype!="float64"):
        data = data.astype('float64')

    vector_input = False
    if(data.ndim==1):
        vector_input = True
        data = data[:,None]

    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band',
                        ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)


    # Taper
    if(taper>0):
        #npts = np.shape(data)[0]
        #lwind = int(npts*taper)
        #taper=np.linspace(0,1,lwind)[None].T
        #data[:lwind,:] *= taper
        #data[-lwind:,:] *= np.flipud(taper)
        data = block_cleaning.taper(data, taper_ratio=taper)


    if(zerophase):
        if(vector_input):
            firstpass = sosfilt(sos, data, axis=0)
            return np.squeeze(sosfilt(sos, firstpass[::-1],axis=0)[::-1])
        else:
            firstpass = sosfilt(sos, data, axis=0)
            return sosfilt(sos, firstpass[::-1], axis=0)[::-1]
    else:
        if(vector_input):
            return np.squeeze(sosfilt(sos, data, axis=0))
        else:
            return sosfilt(sos, data, axis=0)
    

def chebychev_lowpass_downsamp(data, fs, factor, zerophase=False, verbose=False):
    """
    Custom Chebychev type two lowpass filter useful for
    decimation filtering.

    This filter is stable up to a reduction in frequency with a factor of
    10. If more reduction is desired, simply decimate in steps.

    Partly based on a filter in ObsPy.

    :param trace: The trace to be filtered.
    :param freqmax: The desired lowpass frequency.
    """
    freqout = fs/factor
    freqmax = fs/factor/2
    if(verbose):
        print("Downsampling {0}Hz to {1}Hz".format(fs,freqout))
    
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 96, 1e99
    ws = freqmax / (fs * 0.5)  # stop band frequency
    wp = ws  # pass band frequency

    while True:
        if order <= 12:
            break
        wp *= 0.99
        order, wn = signal.cheb2ord(wp, ws, rp, rs, analog=0)

    sos = cheby2(order, rs, wn, btype='low', analog=0, output='sos')

    data2 = np.zeros([  int(np.ceil(np.shape(data)[0]/factor)), np.shape(data)[1]])
    
    if(zerophase):
        for i in range(np.shape(data)[1]):
            y = sosfiltfilt(sos, data[:,i])   
            data2[:,i] = y[::factor]
    else:
        for i in range(np.shape(data)[1]):
            y = sosfilt(sos, data[:,i])   
            data2[:,i] = y[::factor]
    if(verbose):
        print("   Downsampling completed.")
    return data2
