# Various 1D filters that can be applied to data.
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import fftconvolve
######################################################
# Savgol filter (a low-pass filter)
'''
Inputs:
curve = Input noisy 1D-curve to smooth
window_length = Window length (number of time bins) for Savgol filter. Must be an odd number.
polynomial_degree = Degree of polynomial to fit within each window
'''
def savgol_lowpass(curve,window_length,polynomial_degree):
    smoothed_curve = savgol_filter(curve,window_length,polynomial_degree)
    return smoothed_curve
######################################################
# 1D NumPy filter (np.hanning/np.hamming/np.blackman)
'''
Inputs:
data_1d = 1D data array
filter = NumPy filter function (np.hanning/np.hamming/np.blackman)
kernel_size = No. of samples constituting a filter window
'''
def pass_1dfilter(data_1d,filter,kernel_size):
    kernel = filter(kernel_size)
    output = fftconvolve(data_1d,kernel,mode='same')
    return output
######################################################
# Block average a 1D array by factor R
'''
Inputs:
data = 1D array
R = Window length for performing block averaging
'''
def blockavg1d(data,R):
    R = int(R)
    data_size = np.size(data)
    pad_size = int(np.ceil(float(data_size)/R)*R - float(data_size))
    # Pad the data with NaNs to a size divisible by R.
    pad_array1d = np.ones(pad_size)*np.NaN
    if np.ma.is_masked(data):
        padded_data = np.ma.append(data,pad_array1d)
    else:
        padded_data = np.append(data,pad_array1d)
    pad_axis_length = len(padded_data)
    blkavg_data = np.nanmean(padded_data.reshape((pad_axis_length//R,R)),axis=1)
    return blkavg_data
######################################################
def print_hello(N_times):
    print('Hello world')
