#!/usr/bin/env python
'''
This scripts performs the following tasks.
1. Reads raw data from a filterbank file.
2. Computes (if required) and removes bandpass.
3. Reads in a rfifind mask and applies it on the raw data.
4. Removes DM = 0 pc/cc signal (if specified).
5. Downsamples (or smooths) data along frequency and time.

Run using following syntax.
nice -(nice value) mpiexec -n (nproc) python -m mpi4py mask_ds_fil.py [-h] -i INPUTS_CFG
'''
from __future__ import print_function
from __future__ import absolute_import
# Load custom modules.
from modules.read_config import read_config
from modules.read_header import Header
from modules.read_fil import load_fil_data
from modules.read_rfifindmask import read_rfimask, modify_zapchans_bandpass
from modules.ds_systematics import remove_additive_time_noise, calc_median_bandpass, correct_bandpass
from modules.module_dedisp import calc_tDM, dedisperse_ds
from modules.filters1d import blockavg1d
from modules.filters2d import smooth_master
from modules.general_utils import setup_logger_stdout, create_dir
# Load standard packages.
from blimpy import Waterfall
from blimpy.io.sigproc import generate_sigproc_header
from mpi4py import MPI
import numpy as np
import os, logging, time, sys, glob
import subprocess as sp
from argparse import ArgumentParser
import matplotlib.pyplot as plt
#########################################################################
# Execute call  for child processors.
def myexecute(hotpotato, times_chunk, tstart, tstop, freqs_GHz, npol, nchans, chan_bw, n_bytes, hdr_size, nint_chunk, int_times_chunk, mask_zap_chans_per_intchunk, mask_zap_chans, logger, rank):
    # Read in a chunk of filterbank data.
    logger.info('RANK %d: Reading in data between times %.2f - %.2f s'% (rank, times_chunk[0], times_chunk[-1]))
    f = open(hotpotato['DATA_DIR']+'/'+hotpotato['fil_file'],'rb')
    current_cursor_position = f.tell()
    data = load_fil_data(f, tstart, tstop, npol, nchans, n_bytes, hdr_size, hotpotato['pol'], current_cursor_position)
    # Flip frequency axis of DS if channel bandwidth is negative.
    if (chan_bw<0):
        logger.info('RANK %d: Flipping frequency axis of DS'% (rank))
        data = np.flip(data,axis=0)
    # Clip bandpass edges.
    data = data[hotpotato['ind_band_low']:hotpotato['ind_band_high']]
    # Compute bandpass if needed.
    if hotpotato['bandpass_method']=='compute':
        hotpotato['median_bp'] = calc_median_bandpass(data)
    logger.info('RANK %d: Correcting data for bandpass shape'% (rank))
    if 0 in hotpotato['median_bp']:
        indices_zero_bp = np.where(hotpotato['median_bp']==0)[0]
        replace_value = np.median(hotpotato['median_bp'][np.where(hotpotato['median_bp']!=0)[0]])
        hotpotato['median_bp'][indices_zero_bp] = replace_value
        data[indices_zero_bp] = replace_value
    data = correct_bandpass(data, hotpotato['median_bp'])
    # Remove zerodm signal.
    if hotpotato['remove_zerodm']:
        data = remove_additive_time_noise(data)[0]
        logger.info('RANK %d: Zerodm removal completed.'% (rank))
    # Apply rfifind mask on data.
    boolean_rfimask = np.zeros(data.shape,dtype=bool)
    for i in range(nint_chunk):
        if i==nint_chunk-1:
            tstop_int = len(times_chunk)
        else:
            tstop_int = np.min(np.where(times_chunk>=int_times_chunk[i+1])[0])
        tstart_int = np.min(np.where(times_chunk>=int_times_chunk[i])[0])
        boolean_rfimask[mask_zap_chans_per_intchunk[i],tstart_int:tstop_int] = True
    logger.info('RANK %d: Applying RFI mask on data'% (rank))
    data = np.ma.MaskedArray(data,mask=boolean_rfimask)
    # Replaced masked entries with mean value.
    logger.info('RANK %d: Replacing masked entries with mean values'% (rank))
    data = np.ma.filled(data, fill_value=np.nanmean(data))
    # Dedisperse the data, if DM is non-zero.
    if hotpotato['DM']!=0:
        logger.info('RANK %d: Dedispersing data using DM = %.1f pc/cc'% (rank, hotpotato['DM']))
        data, times_chunk = dedisperse_ds(data, freqs_GHz, hotpotato['DM'], freqs_GHz[-1], freqs_GHz[0], times_chunk[1]-times_chunk[0], times_chunk[0])
    # Smooth and downsample the data.
    data, freqs_GHz, times_chunk = smooth_master(data,hotpotato['smoothing_method'],hotpotato['convolution_method'],hotpotato['kernel_size_freq_chans'],hotpotato['kernel_size_time_samples'],freqs_GHz,times_chunk)
    if hotpotato['smoothing_method']!='Blockavg2D':
        data, freqs_GHz, times_chunk = smooth_master(data,'Blockavg2D',hotpotato['convolution_method'],hotpotato['kernel_size_freq_chans'],hotpotato['kernel_size_time_samples'],freqs_GHz,times_chunk)
    mask_zap_check = list(np.sort(mask_zap_chans)//hotpotato['kernel_size_freq_chans'])
    mask_chans = np.array([chan for chan in np.unique(mask_zap_check) if mask_zap_check.count(chan)==hotpotato['kernel_size_freq_chans']])
    logger.info('RANK %d: Data smoothing and/or downsampling completed.'% (rank))
    # Remove residual spectral trend.
    data = data - np.median(data,axis=1)[:,None]
    logger.info('RANK %d: Residual spectral trend subtracted.'% (rank))
    # Write a .npz file to disk.
    save_array = [data, freqs_GHz, times_chunk, mask_chans]
    save_keywords = ['DS', 'Radio frequency (GHz)', 'Time (s)', 'Channel mask']
    np.savez(hotpotato['OUTPUT_DIR']+'/'+hotpotato['basename']+'_dsamp_tstart%08.2f'% (times_chunk[0]), **{name:value for name, value in zip(save_keywords,save_array)})

# Collate data from multiple small .npz files into a single large .npz file.
def collate_npz_to_npz(hdr, hotpotato, logger, remove_npz=True):
    logger.info('Collating data from multiple temporary .npz files into a single .npz  file')
    npz_list = sorted(glob.glob(hotpotato['OUTPUT_DIR']+'/'+hotpotato['basename']+'_dsamp_tstart*.npz'))
    if len(npz_list)==0:
        logger.warning('No .npz files detected. Quitting...')
        sys.exit(1)
    contents = np.load(npz_list[0], allow_pickle=True)
    freqs_GHz = contents['Radio frequency (GHz)']
    data = contents['DS']
    time_array = contents['Time (s)']
    mask_chans = contents['Channel mask']
    if len(npz_list)>1:
        for i in range(1,len(npz_list)):
            npz_append_contents = np.load(npz_list[i], allow_pickle=True)
            data = np.concatenate((data, npz_append_contents['DS']), axis=1)
            time_array = np.concatenate((time_array, npz_append_contents['Time (s)']))
    save_array = [data, freqs_GHz, time_array, mask_chans]
    save_keywords = ['DS', 'Radio frequency (GHz)', 'Time (s)', 'Channel mask']
    out_npz_filename = hotpotato['OUTPUT_DIR']+'/'+hotpotato['basename']+'_dsamp'
    logger.info('Writing data into .npz file %s'% (out_npz_filename))
    np.savez(out_npz_filename, **{name:value for name, value in zip(save_keywords,save_array)})
    logger.info('Write complete.')
    # Remove temporary npz files.
    if remove_npz:
        for npz_file in npz_list:
            rm_cmd = 'rm %s'% (npz_file)
            status = sp.check_call(rm_cmd, shell=True)
            if status==0:
                logger.info('Deleted %s.'% (npz_file))
            else:
                logger.debug('Error deleting %s.'% (npz_file))

# Collate data from temporary .npz files into a filterbank data product.
def collate_npz_to_fil(hdr, hotpotato, logger, remove_npz=True):
    logger.info('Collating data from multiple temporary .npz files into a single filterbank file')
    npz_list = sorted(glob.glob(hotpotato['OUTPUT_DIR']+'/'+hotpotato['basename']+'_dsamp_tstart*.npz'))
    if len(npz_list)==0:
        logger.warning('No .npz files detected. Quitting...')
        sys.exit(1)
    contents = np.load(npz_list[0], allow_pickle=True)
    freqs_GHz = contents['Radio frequency (GHz)']
    data = contents['DS']
    time_array = contents['Time (s)']
    if len(npz_list)>1:
        for i in range(1,len(npz_list)):
            npz_append_contents = np.load(npz_list[i], allow_pickle=True)
            data = np.concatenate((data, npz_append_contents['DS']), axis=1)
            time_array = np.concatenate((time_array, npz_append_contents['Time (s)']))
    data = np.array(data, dtype=np.float32) # Force np.float32 data type to avoid data format issues.
    data = data.T.reshape((data.T.shape[0], 1, data.T.shape[1]))
    # Update header to reflect data properties.
    hdr.primary.pop('hdr_size', None)
    hdr.primary['fch1'] = freqs_GHz[0]*1e3
    hdr.primary['foff'] = (freqs_GHz[1]-freqs_GHz[0])*1e3
    hdr.primary['nchans'] = len(freqs_GHz)
    hdr.primary['nifs'] = 1
    hdr.primary['tsamp'] = time_array[1] - time_array[0]
    hdr.primary['nbits'] = 32
    # Construct a Waterfall object that will be written to disk as a filterbank file.
    filename_out = hotpotato['OUTPUT_DIR']+'/'+hotpotato['basename']+'_dsamp.fil'
    wat = Waterfall() # Empty Waterfall object
    wat.header = hdr.primary
    with open(filename_out, 'wb') as fh:
        logger.info('Writing filterbank file to %s'% (hotpotato['OUTPUT_DIR']))
        fh.write(generate_sigproc_header(wat)) # Trick Blimpy into writing a sigproc header.
        np.float32(data.ravel()).tofile(fh)
    # Remove temporary npz files.
    if remove_npz:
        for npz_file in npz_list:
            rm_cmd = 'rm %s'% (npz_file)
            status = sp.check_call(rm_cmd, shell=True)
            if status==0:
                logger.info('Deleted %s.'% (npz_file))
            else:
                logger.debug('Error deleting %s.'% (npz_file))

# Divide time axis into multiple chunks, one chunk per child processor. Each chunk is further subdivided into blocks.
def divide_timechunks(nproc, nblocks, times, nint, int_times, mask_zap_chans_per_int, t_samp, downsample_time_factor, freq_low, DM, ref_freq):
    DMcurv_samples = np.round(calc_tDM(freq_low,DM,ref_freq)/t_samp).astype(int) # No. of samples spanning a broadband DM curve
    print('No. of samples spanning a broadband DM curve from %.2f GHz to %.2f GHz = %d'% (ref_freq, freq_low, DMcurv_samples))
    if nproc>1:
        equal_divide_size = np.round((len(times)-DMcurv_samples)/(nblocks*nproc)).astype(int) # Divide the time axis into nblocks*nproc chunks.
    else:
        equal_divide_size =  np.round((len(times)-DMcurv_samples)/nblocks).astype(int) # Divide the time axis into nblocks chunks.
    # Increase chunk size, if required, to make it exactly divisible by the temporal downsampling factor.
    chunk_size = equal_divide_size + downsample_time_factor - (equal_divide_size % downsample_time_factor) # Chunks of the dedispersed time axis
    # Calculate time edges of each chunk.
    tstart = 0
    tstop = chunk_size + DMcurv_samples
    tstart_values = [tstart] # List of start sample numbers (included) for each child processor
    tstop_values = [tstop] # List of stop sample numbers (excluded) for each child processor
    while tstop <= len(times):
        tstart = tstop - DMcurv_samples
        tstop = tstart + chunk_size + DMcurv_samples
        tstart_values.append(tstart)
        tstop_values.append(tstop)
    # Drop samples from the final chunk to make its size (after dedispersion) divisible by the downsampling factor chosen.
    tstop_values[-1] = tstart + DMcurv_samples + int(np.floor((len(times) - DMcurv_samples - tstart)/downsample_time_factor)*downsample_time_factor)
    # Convert list to arrays.
    tstart_values = np.array(tstart_values, dtype=int)
    tstop_values = np.array(tstop_values, dtype=int)
    # Section rfifind mask into chunks.
    nint_values = [] # List of no. of integrations to be processed by each child processor
    int_times_values = [] # Integration time (s) boundaries for each child processor
    mask_zap_chans_per_int_values = [] # Set of channels to mask for every integration assigned to a child processor
    for i in range(len(tstart_values)):
        idx1 = np.where(int_times<=tstart_values[i]*t_samp)[0][-1]
        idx2 = np.where(int_times<tstop_values[i]*t_samp)[0][-1]
        nint_values.append(idx2-idx1+1)
        int_times_values.append(int_times[idx1:idx2+1])
        mask_zap_chans_per_int_values.append(mask_zap_chans_per_int[idx1:idx2+1])
    nint_values = np.array(nint_values)
    print('Sectioning of time axis into disjoint chunks is complete.')
    # Cast arrays to shape (nproc, nblocks).
    if nproc>1:
        tmp_tstart = []
        tmp_tstop = []
        tmp_nint = []
        tmp_int_times = []
        tmp_zapchans = []
        for i in range(nproc):
            tmp_tstart.append(tstart_values[i*nblocks:(i+1)*nblocks])
            tmp_tstop.append(tstop_values[i*nblocks:(i+1)*nblocks])
            tmp_nint.append(nint_values[i*nblocks:(i+1)*nblocks])
            tmp_int_times.append(int_times_values[i*nblocks:(i+1)*nblocks])
            tmp_zapchans.append(mask_zap_chans_per_int_values[i*nblocks:(i+1)*nblocks])
        # Return statement
        return tmp_tstart, tmp_tstop, tmp_nint, tmp_int_times, tmp_zapchans
    else:
        return tstart_values, tstop_values, nint_values, int_times_values, mask_zap_chans_per_int_values

# Set defaults.
def set_defaults(hotpotato):
    if hotpotato['OUTPUT_DIR']=='':
        hotpotato['OUTPUT_DIR'] = hotpotato['DATA_DIR']
    if hotpotato['write_format']=='':
        hotpotato['write_format'] = 'npz'
    if hotpotato['N_blocks']=='':
        hotpotato['N_blocks'] = 1
    if hotpotato['DM']=='':
        hotpotato['DM'] = 0.
    if hotpotato['pol']=='':
        hotpotato['pol'] = 0
    if hotpotato['RFIMASK_DIR']=='':
        hotpotato['RFIMASK_DIR'] = hotpotato['DATA_DIR']
    if hotpotato['BANDPASS_DIR']=='':
        hotpotato['BANDPASS_DIR'] = hotpotato['DATA_DIR']
    if hotpotato['remove_zerodm']=='':
        hotpotato['remove_zerodm'] = False
    if hotpotato['smoothing_method']=='':
        hotpotato['smoothing_method'] = 'Blockavg2D'
    if hotpotato['convolution_method']=='':
        hotpotato['convolution_method'] = 'fftconvolve'
    return hotpotato
#########################################################################
# MAIN MPI function
def __MPI_MAIN__(parser):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    stat = MPI.Status()
    nproc = comm.Get_size()
    # Parent processor
    if rank==0:
        print('STARTING RANK 0')
        # Profile code execution.
        prog_start_time = time.time()

        parse_args = parser.parse_args()
        # Initialize parameter values
        inputs_cfg = parse_args.inputs_cfg

        # Construct list of calls to run from shell.
        hotpotato = set_defaults(read_config(inputs_cfg))
        hotpotato['basename'] = hotpotato['basename']+'_DM%.1f'% (hotpotato['DM'])
        parent_logger = setup_logger_stdout() # Set logger output of parent processor to stdout().

        parent_logger.info('Reading header of file: %s'% (hotpotato['fil_file']))
        hdr = Header(hotpotato['DATA_DIR']+'/'+hotpotato['fil_file'],file_type='filterbank') # Returns a Header object
        tot_time_samples = hdr.ntsamples # Total no. of time samples in entire dynamic spectrum.
        t_samp  = hdr.t_samp   # Sampling time (s)
        chan_bw = hdr.chan_bw  # Channel bandwidth (MHz)
        nchans  = hdr.nchans   # No. of channels
        npol    = hdr.npol     # No. of polarizations
        n_bytes = hdr.primary['nbits']/8.0 # No. of bytes per data sample
        hdr_size = hdr.primary['hdr_size'] # Header size (bytes)
        times = np.arange(tot_time_samples)*t_samp # 1D array of times (s)
        # Set up frequency array. Frequencies in GHz.
        freqs_GHz = (hdr.fch1 + np.arange(nchans)*chan_bw)*1e-3
        # Flip frequency axis if chan_bw<0.
        if (chan_bw<0):
            parent_logger.info('Channel bandwidth is negative.')
            freqs_GHz = np.flip(freqs_GHz)
            parent_logger.info('Frequencies rearranged in ascending order.')

        # Chop bandpass edges.
        hotpotato['ind_band_low'] = np.where(freqs_GHz>=hotpotato['freq_band_low'])[0][0]
        hotpotato['ind_band_high'] = np.where(freqs_GHz<=hotpotato['freq_band_high'])[0][-1]+1
        freqs_GHz = freqs_GHz[hotpotato['ind_band_low']:hotpotato['ind_band_high']]

        # Load median bandpass, if pre-computed.
        if hotpotato['bandpass_method']=='file':
            parent_logger.info('Loading median bandpass from %s'% (hotpotato['bandpass_npz']))
            hotpotato['median_bp'] = np.load(hotpotato['BANDPASS_DIR']+'/'+hotpotato['bandpass_npz'],allow_pickle=True)['Median bandpass']
            hotpotato['median_bp'] = hotpotato['median_bp'][hotpotato['ind_band_low']:hotpotato['ind_band_high']]
            parent_logger.info('Median bandpass loaded.')
        elif hotpotato['bandpass_method'] not in ['file', 'compute']:
            logger.debug('Unrecognized bandpass computation method.')
            sys.exit(1)

        # Load rfifind mask.
        parent_logger.info('Reading rfifind mask: %s'% (hotpotato['rfimask']))
        nint, int_times, ptsperint, mask_zap_chans, mask_zap_ints, mask_zap_chans_per_int = read_rfimask(hotpotato['RFIMASK_DIR']+'/'+hotpotato['rfimask'])
        int_times = np.round(int_times/t_samp)*t_samp
        mask_zap_chans, mask_zap_chans_per_int = modify_zapchans_bandpass(mask_zap_chans, mask_zap_chans_per_int, hotpotato['ind_band_low'], hotpotato['ind_band_high'])

        # Section up time axis and rfifind mask into chunks.
        tstart_values, tstop_values, nint_values, int_times_values, mask_zap_chans_per_int_values = divide_timechunks(nproc, hotpotato['N_blocks'], times, nint, int_times, mask_zap_chans_per_int, t_samp, hotpotato['kernel_size_time_samples'], freqs_GHz[0], hotpotato['DM'], freqs_GHz[-1])

        create_dir(hotpotato['OUTPUT_DIR'])
        if nproc==1:
            for nb in range(len(tstart_values)):
                times_chunk = np.arange(tstart_values[nb], tstop_values[nb])*t_samp
                myexecute(hotpotato, times_chunk, tstart_values[nb], tstop_values[nb], freqs_GHz, npol, nchans, chan_bw, n_bytes, hdr_size, nint_values[nb], int_times_values[nb], mask_zap_chans_per_int_values[nb], mask_zap_chans, parent_logger, rank)
        else:
            # Send data to child processors.
            for indx in range(1,nproc):
                comm.send((hotpotato, t_samp, tstart_values[indx-1], tstop_values[indx-1], freqs_GHz, npol, nchans, chan_bw, n_bytes, hdr_size, nint_values[indx-1], int_times_values[indx-1], mask_zap_chans_per_int_values[indx-1], mask_zap_chans), dest=indx, tag=indx)
            for nb in range(len(tstart_values[-1])):
                times_chunk = np.arange(tstart_values[-1][nb], tstop_values[-1][nb])*t_samp
                myexecute(hotpotato, times_chunk, tstart_values[-1][nb], tstop_values[-1][nb], freqs_GHz, npol, nchans, chan_bw, n_bytes, hdr_size, nint_values[-1][nb], int_times_values[-1][nb], mask_zap_chans_per_int_values[-1][nb], mask_zap_chans, parent_logger, rank)
            comm.Barrier() # Wait for all child processors to complete respective calls.

        # Collate data from multiple temporary .npz files into a single .npz file or a filterbank file.
        if hotpotato['write_format']=='npz':
            collate_npz_to_npz(hdr, hotpotato, parent_logger, remove_npz=True)
        elif  hotpotato['write_format']=='fil' or hotpotato['write_format']=='filterbank':
            collate_npz_to_fil(hdr, hotpotato, parent_logger, remove_npz=True)

        # Calculate total run time for the code.
        prog_end_time = time.time()
        run_time = (prog_end_time - prog_start_time)/60.0
        parent_logger.info('Code run time = %.5f minutes'% (run_time))
        print('FINISHING RANK 0')
    else:
        # Recieve data from parent processor.
        hotpotato, t_samp, tstart_dist, tstop_dist, freqs_GHz, npol, nchans, chan_bw, n_bytes, hdr_size, nint_dist, int_times_dist, mask_zap_chans_per_int_dist, mask_zap_chans = comm.recv(source=0, tag=rank)
        print('STARTING RANK: ',rank)
        child_logger = setup_logger_stdout() # Set up separate logger for each child processor.
        for nb in range(len(tstart_dist)):
            times_chunk = np.arange(tstart_dist[nb], tstop_dist[nb])*t_samp
            myexecute(hotpotato, times_chunk, tstart_dist[nb], tstop_dist[nb], freqs_GHz, npol, nchans, chan_bw, n_bytes, hdr_size, nint_dist[nb], int_times_dist[nb], mask_zap_chans_per_int_dist[nb], mask_zap_chans, child_logger, rank)
        print('FINISHING RANK: ',rank)
        comm.Barrier() # Wait for all processors to complete their respective calls.
#########################################################################
def usage():
    return """
usage: nice -(nice value) mpiexec -n (nproc) python -m mpi4py mask_ds_fil.py [-h] -i INPUTS_CFG

Argmunents in parenthesis are required numbers for an MPI run.

This scripts performs the following tasks.
1. Reads raw data from a filterbank file.
2. Computes (if required) and removes bandpass.
3. Reads in a rfifind mask and applies it on the raw data.
4. Removes DM = 0 pc/cc signal (if specified).
5. Downsamples (or smooths) data along frequency and time.

required arguments:
-i INPUTS_CFG  Configuration script of inputs

optional arguments:
-h, --help     show this help message and exit
    """
##############################################################################
def main():
    """ Command line tool for filterbank dynamic spectra processing"""
    parser = ArgumentParser(description="Downsample voluminous data from a filterbank file to a more manageable size.",usage=usage(),add_help=False)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs")
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    # Run MPI-parallelized prepsubband.
    __MPI_MAIN__(parser)
##############################################################################
if __name__=='__main__':
    main()
##############################################################################
