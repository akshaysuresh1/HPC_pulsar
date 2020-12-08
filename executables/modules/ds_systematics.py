# A set of common functionalities for processing and plotting dynamic spectra.

# Import required packages.
import numpy as np
from astropy.modeling.models import Gaussian2D
############################################################################
# Remove additive temporal baseline noise from data
'''
Inputs:
ds = Input dynamic spectrum, 2D array, axes = [Frequency, Time]
'''
def remove_additive_time_noise(ds):
    add_noise = np.mean(ds,axis=0)
    print('Removing additive temporal noise from data.')
    cleaned_ds = ds - add_noise
    return cleaned_ds, add_noise
############################################################################
# Chop the band edges. Works for both chan_bw>0 and chan_bw<0.
'''
Inputs:
data = 2D data array, axes = [Frequency, Time]
freqs_GHz = Radio frequencies (usually GHz)
freq_low = Lowest acceptable frequency (GHz) for bandpass
freq_high = Highest acceptable frequency (GHz) for bandpass
chan_bw = Channel bandwidth (MHz) of data
'''
def chop_freq_axis(data,freqs_GHz,freq_low,freq_high,chan_bw):
    if (chan_bw>0):
        ind_low = np.where(freqs_GHz>=freq_low)[0][0]
        ind_high = np.where(freqs_GHz>freq_high)[0][0]
        freqs_GHz = freqs_GHz[ind_low:ind_high]
        data = data[ind_low:ind_high,:]
    else:
        ind_low = np.where(freqs_GHz<=freq_high)[0][0]
        ind_high = np.where(freqs_GHz<freq_low)[0][0]
        freqs_GHz = freqs_GHz[ind_low:ind_high]
        data = data[ind_low:ind_high,:]
    print('Bandpass edges clipped.')
    return data, freqs_GHz
############################################################################
# Flip the frequency axis if channel bandwidth is negative.
'''
Inputs:
data = 2D data array, axes = [Frequency, Time]
freqs_GHz = Radio frequencies (usually GHz)
chan_bw = Channel bandwidth (MHz) of data
'''
def flip_freq_axis(data,freqs_GHz,chan_bw):
    if (chan_bw<0):
        freqs_GHz = np.flip(freqs_GHz)
        data = np.flip(data,axis=0)
        chan_bw = -chan_bw
    print('Frequencies arranged in ascending order.')
    return data, freqs_GHz, chan_bw
############################################################################
# Obtain median noisy bandpass shape.
'''
Inputs:
raw_ds = Input dynamic spectrum, 2D array, axes = [Frequency, Time]
'''
def calc_median_bandpass(raw_ds):
    bandpass = np.zeros(len(raw_ds))
    for i in range(len(bandpass)):
        bandpass[i] = np.nanmedian(raw_ds[i])
    print('Median bandpass shape calculated.')
    return bandpass
############################################################################
# Correct data for bandpass shape.
'''
Inputs:
raw_ds = Input dynamic spectrum, 2D array, axes = [Frequency, Time]
bandpass = 1D array of bandpass fluxes
'''
def correct_bandpass(raw_ds,bandpass):
    ds_bp_corrected = raw_ds/bandpass[:,None] - 1
    print('Bandpass shape removed from data.')
    return ds_bp_corrected
############################################################################
