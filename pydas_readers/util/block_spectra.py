from scipy.fftpack import fft, fftfreq, fftshift
import numpy as np
import matplotlib.pyplot as plt

def spectrum(data, headers, ampl1, ampl2, dB=False, log=False, stack=False):
    """
    A function to take the frequency spectrum of DAS data

    intput:
    data: numpy array, with time series as columns
    headers: dictionary of metadata (note: make sure that the sampling rate is correct!)
    freq1: lower frequency
    freq2: upper frequency
    ampl1: lower amplitude (find these values with trial and error)
    ampl2: upper amlitude

    dB: amplitude on the dB-scale
    log: frequencies on a log-scale
    stack: return the stack of all spectra
    
    output:
    density: 2d numpy array containing averaged spectra of all channels
    frequencies: the x-axis of the density array, the frequencies
    amplitudes: the y-axis of the density array, the amplitudes
    st: stack of all spectra
    freq: frequency axis of the stack (returned for plotting purposes)
    """

    # size of averaging grid in frequency (nf) and amplitude (na)
    nf = 200 
    na = nf + 1

    # amlitudes to look at
    amplitudes = np.linspace(ampl1, ampl2, na) 

    density = np.zeros((nf, na)) # averaging grid

    npts = headers['npts']
    scaling = 2/npts # to get correct amplitude out of the numpy fft

    freq = np.fft.rfftfreq(npts, 1./headers['fs']) # fft frequencies
    if log:
        freq1 = np.log10(freq[freq>0][0])
        freq2 = np.log10(freq[-1])
        frequencies = np.logspace(freq1, freq2, num=nf)
    else:
        frequencies = np.linspace(freq[0], freq[-1], nf) 


    if np.any(stack) != False:
        st = np.zeros(freq.shape)

    # loop over all channels to calculate spectrum, and store results in averaging grid
    for i in range(data.shape[1]):
        tr = data[:,i].copy()

        sp = np.abs(np.fft.rfft(tr)) #fft
        y = np.abs(sp) * scaling # scale amplitude to physical unit
        if dB:
            y = 10*np.log10(y) # scale amplitude to dB scale
        if stack:
            st += y
        
        # store spectrum in grid
        pre_density = np.zeros((nf, na))
        for ii in range(len(sp)):
            f = np.where(frequencies > freq[ii])[0]
            if len(f) == 0:
                break
            else:
                ix = f[0]
                
            a = np.where(amplitudes > y[ii])[0]
            if len(a) == 0:
                break
            else:
                iy = a[0]
                
            pre_density[ix,iy] += 1
            
            #break
        pre_density[pre_density>1] = 1
        
        density += pre_density
        
    if np.any(stack) != False:
        st /= data.shape[1]

    density /= data.shape[1] 
    density *= 100 # density is given as percentage of channels that have that amplitude at that frequency
    density = density.T

    if np.any(stack) != False:
        return density, frequencies, amplitudes, st, freq
    else:
        return density, frequencies, amplitudes



def plot_spectrum(density, frequencies, amplitudes, headers, dB=False, log=False, stack=False, freq=False, fname=False):
    """
    plotting function to plot frequency spectrum of DAS data.

    input:
    density: 2d numpy array containing averaged spectra of all channels
    frequencies: the x-axis of the density array, the frequencies
    amplitudes: the y-axis of the density array, the amplitudes
    headers: dictionary of metadata

    dB: plot amplitude on the dB-scale
    log: plot frequencies on a log-scale
    stack: plot the stack of all spectra
    freq: frequencies of the stack
    fname: path to store image

    """

    plt.figure(1, figsize=(12,8), dpi=300)
    
    plt.pcolor(frequencies, amplitudes, density, shading='nearest')
    
    if np.any(stack) != False:
        plt.plot(freq, stack, 'gray', linewidth=2, linestyle='--')


    if log:
        plt.xscale('log')
    
    plt.xlabel('Frequency (Hz)')

    if dB:
        plt.ylabel('Amplitude (dB)')
    else:
        if headers['unit'] == '(nm/m)/s * Hz/m':
            plt.ylabel('Amplitude (10$^{-9}$/s)')

    cbar = plt.colorbar()
    cbar.set_label('Percentage of channels (%)', rotation=270, labelpad=15)

    plt.tight_layout()

    if fname != False:
        plt.savefig(fname, dpi=300)
    plt.show()