# Functions to read and modify an "rfifind" mask generated using PRESTO
import numpy as np
############################################################################
# Open a rfifind mask and return an array of channels and integrations to mask.
'''
Inputs:
mask_file = Name of rfifind mask together with its .mask extension

Returns:
nint = No. of integrations
int_times = 1D array of times (s) post integration for computation of stats by rfifind
ptsperint = No. of time samples per integration
mask_zap_chans = 1D array of entire channels to mask
mask_zap_ints = 1D array of entire time integrations to mask
mask_zap_chans_per_int = Set of channels to mask for each time integration.
'''
def read_rfimask(mask_file):
    x = open(mask_file)
    time_sig, freq_sig, MJD, dtint, lofreq, df = np.fromfile(x, dtype=np.float64, count=6)
    nchan, nint, ptsperint = np.fromfile(x, dtype=np.int32, count=3)
    freqs = np.arange(nchan)*df + lofreq
    int_times = np.arange(nint)*dtint
    nzap = np.fromfile(x, dtype=np.int32, count=1)[0]
    if nzap:
        mask_zap_chans = np.fromfile(x, dtype=np.int32, count=nzap)
    else:
        mask_zap_chans = np.asarray([])
    mask_zap_chans = set(mask_zap_chans)
    if len(mask_zap_chans)==nchan:
        print("WARNING!:  All channels recommended for masking!")
    nzap = np.fromfile(x, dtype=np.int32, count=1)[0]
    if nzap:
        mask_zap_ints = np.fromfile(x, dtype=np.int32, count=nzap)
    else:
        mask_zap_ints = np.asarray([])
    if len(mask_zap_ints)==nint:
        print("WARNING!:  All intervals recommended for masking!")
    nzap_per_int = np.fromfile(x, dtype=np.int32, count=nint)
    mask_zap_chans_per_int = []
    for nzap in nzap_per_int:
        if nzap:
            tozap = np.fromfile(x, dtype=np.int32, count=nzap)
        else:
            tozap = np.asarray([])
        mask_zap_chans_per_int.append(tozap)
    x.close()
    mask_zap_chans = np.array(list(mask_zap_chans))
    mask_zap_ints = np.array(list(mask_zap_ints))
    return nint, int_times, ptsperint, mask_zap_chans, mask_zap_ints, mask_zap_chans_per_int
############################################################################
# Modify a rfifind mask specification following bandpass edge clipping.
'''
Inputs:
mask_zap_chans = 1D array of entire channels to mask
mask_zap_chans_per_int = Set of channels to mask for each time integration.
ind_band_low = Index of low frequency bandpass edge
ind_band_high = Index of high frequency bandpass edge, allowed bandpass range is ind_band_low:ind_band_high
'''
def modify_zapchans_bandpass(mask_zap_chans, mask_zap_chans_per_int, ind_band_low, ind_band_high):
    new_zap_chans = mask_zap_chans[np.where(np.logical_and(mask_zap_chans>=ind_band_low, mask_zap_chans<ind_band_high))[0]] - ind_band_low
    new_zap_chans_per_int = []
    for i in range(len(mask_zap_chans_per_int)):
        new_zap_chans_per_int.append(mask_zap_chans_per_int[i][np.where(np.logical_and(mask_zap_chans_per_int[i]>=ind_band_low, mask_zap_chans_per_int[i]<ind_band_high))[0]] - ind_band_low)
    return new_zap_chans, new_zap_chans_per_int
############################################################################
