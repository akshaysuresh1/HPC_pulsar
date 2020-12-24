# Functions relevant to PRESTO's dedisp.py script.
from builtins import zip, range
from .general_utils import create_dir
from .read_config import read_config
import os
import numpy as np
#######################################################################
# Loop over the DDplan steps and obtain a list of calls to execute.
def gen_call_list(basename, OUTPUT_DIR, maskfile, rawfiles, dDMs, dsubDMs, downsamps, subcalls, startDMs, dmspercalls, nsub, other_flags):
    calls = []
    # Output file
    output_basename = OUTPUT_DIR + '/'+ basename
    # RFI mask
    if maskfile is not None:
        mask_string = '-mask %s'% (maskfile)
    else:
        mask_string = ''
    additional_flags = other_flags+' '+mask_string
    for dDM, dsubDM, dmspercall, downsamp, subcall, startDM in zip(dDMs, dsubDMs, dmspercalls, downsamps, subcalls, startDMs):
        # Loop over the number of calls
        for ii in range(subcall):
            subDM = startDM + (ii+0.5)*dsubDM
            loDM = startDM + ii*dsubDM
            subcall = 'prepsubband %s -nsub %d -lodm %.2f -dmstep %.2f -numdms %d -downsamp %d -o %s %s' % (additional_flags, nsub, loDM, dDM, dmspercall, downsamp, output_basename, rawfiles)
            calls.append(subcall)
    print('Total no. of calls = %d'% (len(calls)) )
    return calls
#######################################################################
# Read inputs from a .config file and output list of prepsubband calls.
def config_call(cfg_file):
    dict = read_config(cfg_file)
    # Set default values.
    if dict['maskfile']=='':
        dict['maskfile'] = None
    # Create output directory if non-existent.
    create_dir(dict['OUTPUT_DIR'])
    # Generate list of prepsubband calls
    calls = gen_call_list(dict['basename'], dict['OUTPUT_DIR'], dict['maskfile'], dict['rawfiles'], dict['dDMs'], dict['dsubDMs'], dict['downsamps'], dict['subcalls'], dict['startDMs'], dict['dmspercalls'], dict['nsub'], dict['other_flags'])
    return calls
#######################################################################
# Calculate dispersive delay relative to a reference frequency for a given DM.
'''
Inputs:
freqs_GHz = 1D array of radio frequencies in GHz
DM = Dispersion measure (pc/cc) at which dispersive delay must be calculated
ref_freq = Reference frequency (GHz) relative to which dispersive delay must be calculated
'''
def calc_tDM(freqs_GHz,DM,ref_freq):
    a_DM = 4.1488064239e-3 # Dispersion constant (Refer to Kulkarni (2020) on ArXiv.)
    freq_factor = freqs_GHz**-2. - ref_freq**-2.
    tDM = a_DM*DM*freq_factor
    return tDM
#######################################################################
# Brute-force dedisperse a dynamic spectrum at a given DM.
'''
Inputs:
ds = 2D array, dynamic spectrum
freqs_GHz = 1D array of frequencies (GHz) covered by dynamic spectrum
DM = DM at which dedispersion must be performed
ref_freq = Frequency (GHz) relative to which dispersive delay must be calculated
freq_low = Lowest frequency (GHz) for which we intend to calculate the dispersive delay
t_resol = Time resolution (s) of the data
start_time = Start time (s) of the data
'''
def dedisperse_ds(ds,freqs_GHz,DM,ref_freq,freq_low,t_resol,start_time):
    tDM = calc_tDM(freqs_GHz,DM,ref_freq)
    tDM_indices = np.round(tDM/t_resol).astype(int)

    # Determine no. of time slices to clip based on lowest frequency.
    max_tDM = calc_tDM(freq_low,DM,ref_freq)
    max_tDM_indices = np.round(max_tDM/t_resol).astype(int)

    n_chans,n_tsamples = ds.shape
    n_dedisp_tsamples = n_tsamples - max_tDM_indices
    dedisp_ds = np.zeros((n_chans,n_dedisp_tsamples))
    dedisp_times = start_time+np.arange(0,n_dedisp_tsamples)*t_resol
    for i in range(n_chans):
        start_t_index = tDM_indices[i]
        stop_t_index = n_dedisp_tsamples + tDM_indices[i]
        dedisp_ds[i] = ds[i,start_t_index:stop_t_index]
    return dedisp_ds,dedisp_times
#######################################################################
