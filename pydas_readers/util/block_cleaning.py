"""
Basic signal processing / cleaning tools, 
assuming one has a contiguous numpy block of data.

Current functions include: taper, detrend, trim,
and a phase-weighted rolling average


Daniel Bowden, ETH Zürich
daniel.bowden@erdw.ethz.ch
Last updated: Jan 2023
"""

import scipy.signal as ss
import numpy as np
from datetime import datetime, timedelta
import glob
import pathlib
import sys
import os

def taper(data, taper_ratio=0.01):
    """
    Taper both edges of timeseries

    :param data: Data to taper. 2D numpy array [ npts, nchan ]
                  OR a 1D numpy array [ npts, ]
    :param taper_ratio: fraction of data to taper, between 0 and 0.5
                  i.e., 0.01 means 1%
    :return: tapered data
    """
    if(data.dtype!="float64"):
        data = data.astype('float64')

    npts = np.shape(data)[0]
    lwind = int(npts*taper_ratio)
    taper = np.linspace(0,1,lwind)#[None].T
    
    if(len(data.shape) == 2):
        data[:lwind,:] *= taper[None].T
        data[-lwind:,:] *= np.flipud(taper[None].T)

    elif(len(data.shape) == 1):
        data[:lwind] *= taper
        data[-lwind:] *= np.flipud(taper)
    return data

def detrend(data, type='linear'):
    """
    Detrend each individual trace separately

    :param data: Data to taper. 2D numpy array [ npts, nchan ]
                  OR a 1D numpy array [ npts, ]
    :param type: type of detrending, now only linear or simple
    :return: detrended data
    """
    vector_input = False
    if(data.ndim==1):
        vector_input = True
        data = data[:,None]
        
    if(data.dtype!="float64"):
        data = data.astype('float64')
        
    npts = np.shape(data)[0]
    
    ###################
    if type == 'linear':
        for i in range(np.shape(data)[1]):
            data[:,i] = ss.detrend(data[:,i])
        
    ###################
    if type == 'simple':
        for i in range(np.shape(data)[1]):
            x1, x2 = data[0,i], data[-1,i]
            data[:,i] -= x1 + np.arange(npts) * (x2 - x1) / float(npts-1)
            
    if(vector_input):
        return np.squeeze(data)
    else:
        return data
    
def trim(data, t_start, t_end, headers, axis=[]):
    """
    Detrend each individual trace separately.
    Performs deepcopies - more memory but safer.
    
    Simply rounds to the nearest sample. 
     (See obspy documentation for how this could be improved in the future: 
      https://docs.obspy.org/packages/autogen/obspy.core.trace.Trace.trim.html)

    :param data: Data to trim. 2D numpy array [ npts, nchan ]
    :param t_start: python datetime object for start
    :param t_end:   python datetime object for end
    :param headers: dict of headers / metadata
    :optional param axis: dict of pre-computed axis vectors
      (actually the function doesn't need "axis", but if you pass it,
       the function will update it for you)
    :return: data, headers [, axis]
    """
    

    t0 = headers['t0']
    fs = headers['fs']

    trim0 = round((t_start - t0).total_seconds() * fs)
    trim1 = round((t_end   - t0).total_seconds() * fs)

    # Update headers
    # This is imprecise: t0 is set to whatever was input, 
    #  ignoring rounding differences to actual samples
    # TODO: figure out exact rounding error and adjust:
    ## print(  ((t_start - t0).total_seconds() * fs) - trim0)    
    headers2 = headers.copy()
    headers2['t0'] = t_start
    headers2['npts'] = trim1-trim0
    
    if(len(axis)>0):
        axis2 = axis.copy()
        
        # Better estimate of t0 if "axis" was given
        headers2['t0'] = axis['date_times'][trim0]
        
        # Update the timing axis 
        axis2['tt'] = axis['tt'][trim0:trim1]
        axis2['date_times'] = axis['date_times'][trim0:trim1]
        
        return data[trim0:trim1, :].copy(), headers2, axis2
    else:
        return data[trim0:trim1, :].copy(), headers2

def pws_rolling_average(data,ns):
    """
    Smooth data and remove incoherent traces
    Returned N'th trace is a phase-weighted average of [-ns:ns] neighboring traces
    The first ns traces and last ns traces are returned as zeros
    
    :param data: Data to clean. 2D numpy array [ npts, nchan ]
    :param ns: number of traces to average over
    :return: data_pws
    """
    data_pws = np.zeros(np.shape(data))
    nchan = np.shape(data)[1]
    # Note: Andreas' version only took hilbert() of data as it went, rather than the whole block
    #  This version makes a full copy to do hilbert on. This is more memory intesive, but it is 
    #  a bit faster. Tradeoffs.
    dh = ss.hilbert(data,axis=0)
    for i in range(ns,nchan-ns-1):
        pw = np.mean(dh[:,i-ns:i+ns+1] / np.abs(dh[:,i-ns:i+ns+1]), axis=1)
        data_pws[:,i] = np.real(pw**2 * np.mean(data[:,i-ns:i+ns+1],axis=1))
    return data_pws



