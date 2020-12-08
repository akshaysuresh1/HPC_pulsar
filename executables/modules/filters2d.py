# Various 2D filters that can be applied to data.
from .filters1d import blockavg1d
import numpy as np
from astropy.modeling.models import Gaussian2D
from scipy.signal import gaussian, fftconvolve
############################################################################
# Master function that assigns operations based on chosen smoothing mode.
def smooth_master(data,smoothing_method,convolution_method,kernel_size_freq_chans,kernel_size_time_samples,freqs,times):
    if (smoothing_method=='Gaussian2D'):
        smooth_data = smooth_2DGaussian_zeroCCF(data,convolution_method,kernel_size_freq_chans,kernel_size_time_samples)
        return smooth_data, freqs, times
    if (smoothing_method=='Blockavg2D'):
        smooth_data, blkavg_freqs, blkavg_times = blockavg_ds(data,kernel_size_freq_chans,kernel_size_time_samples,freqs,times)
        return smooth_data, blkavg_freqs, blkavg_times
    if (smoothing_method in ['blackman','hanning','hamming']):
        smooth_data = pass_2dfilter(data,convolution_method,smoothing_method,kernel_size_freq_chans,kernel_size_time_samples)
        return smooth_data, freqs, times
############################################################################
# Function to recast a 2D array to a larger size by padding with zeros.
# Specify the start indices along the two axes. For i>start_index, non-zero values are permitted.
'''
Inputs:
array2D = 2D array of non-zeros values that needs to be recast into a larger shape by padding with zeroes
output_shape = Final shape of 2D array to be output
start_index_axis0 = Start index along axis 0 for permitting non-zero values
start_index_axis1 = Start index along axis 1 for permitting non-zero values
'''
def recast_2Darray(array2D, output_shape, start_index_axis0, start_index_axis1):
    output_array = np.zeros(output_shape)
    nonzero_length_axis0, nonzero_length_axis1 = array2D.shape
    stop_index_axis0 = start_index_axis0 + nonzero_length_axis0
    stop_index_axis1 = start_index_axis1 + nonzero_length_axis1
    output_array[start_index_axis0:stop_index_axis0,start_index_axis1:stop_index_axis1] = array2D
    return output_array
############################################################################
# Calculate 2D convolution using FFT.
'''
Inputs:
arr1, arr2 = Two 2D arrays of the same shape to be convolved.
'''
def convolve_fft2D(arr1,arr2):
    conv_product = np.fft.ifft2(np.fft.fft2(arr1)*np.fft.fft2(arr2)).real
    conv_product = np.fft.fftshift(conv_product)
    return conv_product
############################################################################
# Convole dynamic spectrum with a 2D Gaussian kernel assuming zero cross-correlation coefficient (CCF).
'''
Inputs:
data = dynamic spectrum, 2D array = [Frequency, Time]
convolution_method = 'fftconvolve' or 'fft2'
chan_fwhm = FWHM of 2D Gaussian along frequency axis measured in units of number of channels
t_sample_fwhm = Temporal FWHM of 2D Gaussian measured in units of number of time samples
'''
def smooth_2DGaussian_zeroCCF(data,convolution_method,chan_fwhm,t_sample_fwhm):
    chan_sigma = np.round(chan_fwhm/np.sqrt(8*np.log(2))).astype(int)
    t_sample_sigma = np.round(t_sample_fwhm/np.sqrt(8*np.log(2))).astype(int)
    print('Smoothing data with a 2D Gaussian kernel assuming zero CCF...')
    # 1D Gaussian along frequency axis
    freq_extent = 20*chan_sigma
    freq_Gaussian = gaussian(freq_extent,chan_sigma)
    # 1D Gaussian along time axis
    time_extent = 20*t_sample_sigma
    time_Gaussian = gaussian(time_extent,t_sample_sigma)
    # Construct 2D Gaussian from two 1D Gaussians assuming zero CCF.
    Gaussian_2D = np.outer(freq_Gaussian,time_Gaussian)
    Gaussian_2D = Gaussian_2D/np.sum(Gaussian_2D) # Normalize the 2D Gaussian to unit area.
    # Convolve 2D Gaussian with data.
    if (convolution_method=='fftconvolve'):
        output = fftconvolve(data,Gaussian_2D,mode='same')
    elif (convolution_method=='fft2'):
        # Recast 2D Gaussian to same shape as the data array.
        output_shape = data.shape
        start_index_axis0 = (data.shape[0]-freq_extent)//2
        start_index_axis1 = (data.shape[1]-time_extent)//2
        Gaussian_2D = recast_2Darray(Gaussian_2D, output_shape, start_index_axis0, start_index_axis1)
        output = convolve_fft2D(data,Gaussian_2D)
    return output
############################################################################
# Generate a general 2D Gaussian with user-supplied std. dev. along x-axis and y-axis.
# Angle between major axis of 2D gaussian and x-axis can also be specified.
# Also, a full covariance matrix can be provided.
'''
Inputs:
shape = Dimensions of 2D array to output = (No. of y-axis values, No. of x-axis samples)
x_fwhm = FWHM of 2D Gaussian along x-axis
y_fwhm = FWHM of 2D Gaussian along y-axis
theta = Rotation angle in radians. The rotation angle increases counterclockwise. Must be None if a covariance matrix (cov_matrix) is provided. If no cov_matrix is given, None means the default value (0).
cov_matrix = A 2x2 covariance matrix. If specified, overrides the x_stddev, y_stddev, and theta defaults.
'''
def gen_general_2DGaussian(shape,x_fwhm,y_fwhm,theta=None,cov_matrix=None):
    # Convert FWHM to sigma.
    x_stddev = int(np.round(x_fwhm/np.sqrt(8*np.log(2))))
    y_stddev = int(np.round(y_fwhm/np.sqrt(8*np.log(2))))
    # Initialize grid of x and y values.
    y_values = np.arange(shape[0])
    x_values = np.arange(shape[1])
    y_values, x_values = np.meshgrid(y_values,x_values)
    # Coordinates of the center.
    y_center = shape[0]//2
    x_center = shape[1]//2
    # Initialize Gaussian2D object.
    gauss2d_object = Gaussian2D(amplitude=1.0,x_mean=x_center, y_mean=y_center,x_stddev = x_stddev, y_stddev=y_stddev,theta=theta,cov_matrix=cov_matrix)
    gauss2d_array = gauss2d_object(freq_chans,time_samples)
    return gauss2d_array
############################################################################
# Perform a general 2D Gaussian smoothing of a dynamic spectrum.
'''
Inputs:
data = dynamic spectrum, 2D array = [Frequency, Time]
chan_fwhm = FWHM of 2D Gaussian along frequency axis measured in units of number of channels
t_sample_fwhm = Temporal FWHM of 2D Gaussian measured in units of number of time samples
theta = Rotation angle in radians. The rotation angle increases counterclockwise. Must be None if a covariance matrix (cov_matrix) is provided. If no cov_matrix is given, None means the default value (0).
cov_matrix = A 2x2 covariance matrix. If specified, overrides the x_stddev, y_stddev, and theta defaults.
'''
def smooth_2DGaussian(data,chan_fwhm,t_sample_fwhm,theta=None,cov_matrix=None):
    chan_sigma = np.round(chan_fwhm/np.sqrt(8*np.log(2))).astype(int)
    t_sample_sigma = np.round(t_sample_fwhm/np.sqrt(8*np.log(2))).astype(int)
    shape = data.shape
    # Generate 2D Gaussian kernel to smooth data.
    print('Generating 2D Gaussian kernel.')
    gauss2d_kernel = gen_general_2DGaussian(shape,t_sample_sigma,chan_sigma,theta=theta,cov_matrix=cov_matrix)
    # Normalize 2D Gaussian kernel.
    gauss2d_kernel = gauss2d_kernel/np.sum(gauss2d_kernel)
    # Smooth the data by convolving it with a 2D Gaussian.
    print('Smoothing data with a 2D Gaussian kernel...')
    smooth_data = convolve_fft2D(data,gauss2d_kernel)
    print('2D Gaussian smoothing complete.')
    return smooth_data
############################################################################
# Block average along the second axis of a 2D array.
'''
Inputs:
data = 2D array
R = Window length for performing block averaging.
'''
def blockavg_axis2(data,R):
    R = int(R) # Sanity check
    # Shape of the original data
    shape = data.shape
    ax1_length = shape[0]
    ax2_length = shape[1]
    # If ax2_length is not divisible by the length  R, pad the array with NaNs to a size divisible by R.
    pad_size = int(np.ceil(float(ax2_length)/R)*R - float(ax2_length))
    pad_array = np.zeros((ax1_length,pad_size))*np.NaN
    # Pad the data with pad_array.
    if np.ma.is_masked(data):
        padded_data = np.ma.concatenate((data,pad_array),axis=1)
    else:
        padded_data = np.concatenate((data,pad_array),axis=1)
    pad_ax2_length = padded_data.shape[1]
    blkavg_ax2_data = np.nanmean(padded_data.reshape((ax1_length,pad_ax2_length//R,R)),axis=2)
    return blkavg_ax2_data
############################################################################
# Block average independently along the frequency and time axes of a dynamic spectrum
'''
Inputs:
data = dynamic spectrum, 2D array = [Frequency, Time]
blkavg_factor_time = Window length (no. of time samples) for block averaging along time
blkavg_factor_freq = Window length (no. of channels) for block averaging along frequency
times = 1D array of observation times
freqs_GHz = 1D array of radio frequencies (usually GHz)
'''
def blockavg_ds(data, blkavg_factor_freq, blkavg_factor_time, freqs_GHz, times):
    # Ensure block averaging factors are atleast one.
    blkavg_factor_time = np.max([blkavg_factor_time,1])
    blkavg_factor_freq = np.max([blkavg_factor_freq,1])
    # Ensure block averaging factors are integers. If not, round them to nearest integer.
    blkavg_factor_time = int(np.round(blkavg_factor_time))
    blkavg_factor_freq = int(np.round(blkavg_factor_freq))
    # Block average data along time.
    print('Smoothing data by block averaging...')
    print('Block averaging along time by factor %d'% (blkavg_factor_time))
    if (blkavg_factor_time>=2):
        data = blockavg_axis2(data,blkavg_factor_time)
        blkavg_times = blockavg1d(times,blkavg_factor_time)
    else:
        blkavg_times = times
    print('Block averaging along time complete.')
    # Block average data along frequency.
    print('Block averaging along frequency by factor %d'% (blkavg_factor_freq))
    if (blkavg_factor_freq>=2):
        data = blockavg_axis2(data.T,blkavg_factor_freq).T
        blkavg_freqs = blockavg1d(freqs_GHz,blkavg_factor_freq)
    else:
        blkavg_freqs = freqs_GHz
    print('Block averaging along frequency complete.')
    return data,blkavg_freqs,blkavg_times
############################################################################
# Apply a 2D filter which is product of two 1D NumPy filters (np.hanning/np.hamming/np.blackman)
'''
Inputs:
data_2d = 2D data array
convolution_method = 'fftconvolve' or 'fft2'
smoothing_method = 'blackman','hanning','hamming'
kernel_size_freq_chans = No. of channels constituting a spectral kernel size (Gaussian std. dev. / window length)
kernel_size_time_samples = No. of channels constituting a spectral kernel size (Gaussian std. dev. / window length)
'''
def pass_2dfilter(data_2d,convolution_method,smoothing_method,kernel_size_freq_chans,kernel_size_time_samples):
    print('Smoothing data with a 2D %s filter'% (smoothing_method))
    if (smoothing_method=='blackman'):
        filter = np.blackman
    if (smoothing_method=='hanning'):
        filter = np.hanning
    if (smoothing_method=='hamming'):
        filter = np.hamming
    filter_time = filter(kernel_size_time_samples)
    filter_freq = filter(kernel_size_freq_chans)
    filter_2d = np.outer(filter_freq,filter_time)
    filter_2d = filter_2d/np.sum(filter_2d)
    # Convolve 2D filter with data.
    if (convolution_method=='fftconvolve'):
        output = fftconvolve(data_2d,filter_2d,mode='same')
    elif (convolution_method=='fft2'):
        # Recast filter_2d to the same shape as the data by padding with zeroes.
        output_shape = data_2d.shape
        start_index_axis0 = (data_2d.shape[0]-kernel_size_freq_chans)//2
        start_index_axis1 = (data_2d.shape[1]-kernel_size_time_samples)//2
        filter_2d = recast_2Darray(filter_2d, output_shape, start_index_axis0, start_index_axis1)
        output = convolve_fft2D(data_2d,filter_2d)
    return output
############################################################################
