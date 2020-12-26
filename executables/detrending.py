#!/usr/bin/env python
'''
Detrend multiple dedispersed timeseries using either the PRESTO rednoise program or a fast running median filter.

Run using following syntax.
nice -<nice value> mpiexec -n <numproc> python -m mpi4py detrending.py -i INPUTS_CFG
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom packages
from modules.read_config import read_config
from modules.general_utils import setup_logger_stdout, create_dir
from modules.running_median import fast_running_median
# Standard imports
from mpi4py import MPI
import os, logging, time, sys, glob
import numpy as np
import subprocess as sp
from argparse import ArgumentParser
##############################################################################
# Execute call.
def myexecute(dat_file, hotpotato, logger, rank):
    if hotpotato['detrend_method']=='rednoise':
        rednoise(dat_file, hotpotato['startwidth'], hotpotato['endwidth'], hotpotato['endfreq'], hotpotato['OUTPUT_DIR'], logger, rank)
    elif hotpotato['detrend_method']=='rmed':
        running_median_sub(dat_file, hotpotato['rmed_width'], hotpotato['rmed_minpts'], hotpotato['OUTPUT_DIR'], logger, rank)

# Remove baseline fluctuations using PRESTO rednoise program.
def rednoise(dat_file, startwidth, endwidth, endfreq, OUTPUT_DIR, logger, rank):
    dat_basename = dat_file.split('.dat')[0]
    # Take FFT of input timeseries.
    realfft_cmd = 'realfft %s'% (dat_file)
    status = sp.check_call(realfft_cmd, shell=True)
    if status==0:
        logger.info('RANK %d: Computed FFT of %s' % (rank, dat_file))
    else:
        logger.warning('RANK %d: FFT error for %s'% (rank, dat_file))
    # Run fourier-domain rednoise removal.
    rednoise_cmd = 'rednoise -startwidth %d -endwidth %d -endfreq %.2f %s.fft'% (startwidth, endwidth, endfreq, dat_basename)
    status = sp.check_call(rednoise_cmd, shell=True)
    if status==0:
        logger.info('RANK %d: Red-noise removed from %s.fft' % (rank, dat_basename))
    else:
        logger.warning('RANK %d: Red-noise removal error for %s.fft'% (rank, dat_basename))
    # Copy .inf file and move rednoise products to OUTPUT_DIR.
    DM_tag = dat_basename.split('_DM')[-1]
    if '/' in dat_basename:
        red_basename = dat_basename.split('/')[-1].split('_DM')[0]+'_red_DM'+ DM_tag # Excludes path
    else:
        red_basename = dat_basename.split('_DM')[0] + '_red_DM' + DM_tag
    mv_inf_cmd = 'mv %s.inf %s/%s.inf'% (dat_basename + '_red', OUTPUT_DIR, red_basename)
    status = sp.check_call(mv_inf_cmd, shell=True)
    mv_redfft_cmd = 'mv %s.fft %s/%s.fft'% (dat_basename + '_red', OUTPUT_DIR, red_basename)
    status = sp.check_call(mv_redfft_cmd, shell=True)
    # Take inverse FFT of dereddened power spectrum to obtain detrended timeseries.
    realinvfft_cmd = 'realfft %s/%s.fft'% (OUTPUT_DIR, red_basename)
    status = sp.check_call(realinvfft_cmd, shell=True)
    if status==0:
        logger.info('RANK %d: Inverse FFT of %s/%s.fft completed.' % (rank, OUTPUT_DIR, red_basename))
    else:
        logger.warning('RANK %d: Error computing inverse FFT of %s/%s.fft'% (rank, OUTPUT_DIR, red_basename))

# Compute and subtract running median from time-series.
def running_median_sub(dat_file, rmed_width, rmed_minpts, OUTPUT_DIR, logger, rank):
    if '/' in dat_file:
        dat_base  = dat_file.split('/')[-1].split('.dat')[0]
    else:
        dat_base = dat_file.split('.dat')[0]
    dat_base, DM_tag = dat_base.split('_DM')
    output_basename = dat_base + '_rmed%.2f_DM%s'% (rmed_width, DM_tag)
    old_inf_file = dat_file.split('.dat')[0]+'.inf'
    new_inf_file = OUTPUT_DIR + '/' + output_basename+'.inf'
    basename_suffix = '_rmed%.2f'% (rmed_width)
    tsamp = modify_inf(old_inf_file, new_inf_file, basename_suffix) # Sampling time (s)
    width_samples = int(round(rmed_width/tsamp)) # No. of samples across running median window
    dedisp_ts = np.fromfile(dat_file, dtype=np.float32)
    # Computing trend across timeseries
    trend = fast_running_median(dedisp_ts, width_samples, rmed_minpts)
    # Subtracting trend to obtain detrended timeseries
    detrended_ts = np.float32(dedisp_ts - trend)
    detrended_ts.tofile(OUTPUT_DIR + '/' + output_basename + '.dat')
    logger.info("RANK %d: Detrending completed for %s."% (rank, dat_file))

# Update basename in .inf file and return sampling time.
def modify_inf(old_inf_file, new_inf_file, basename_suffix):
    f = open(old_inf_file, 'r')
    text = f.read()
    f.close()
    # Obtain sampling time (s).
    tsamp = float(text.split('Width of each time series bin (sec) ')[1].split('\n')[0].split('=')[1])
    # Read and update basename in new file.
    line1 = text.split('\n')[0]
    remainder = '\n'.join(text.split('\n')[1:])
    prefix, basename = line1.split('=')
    new_basename = basename + basename_suffix
    new_text = prefix+'='+new_basename+'\n'+remainder
    f = open(new_inf_file,'w')
    f.write(new_text)
    f.close()
    return tsamp

# Set defaults.
def set_defaults(hotpotato):
    if hotpotato['startwidth']=='':
        hotpotato['startwidth'] = 6
    if hotpotato['endwidth']=='':
        hotpotato['endwidth'] = 100
    if hotpotato['endfreq']=='':
        hotpotato['endfreq'] = 6.0
    if hotpotato['rmed_width']=='':
        hotpotato['rmed_width'] = 1.0
    if hotpotato['rmed_minpts']=='':
        hotpotato['rmed_minpts'] = 101
    return hotpotato
##############################################################################
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
        if hotpotato['detrend_method'] not in ['rmed', 'rednoise']:
            print('Detrending method %s not recognized. Choose either rednoise or rmed.'% (hotpotato['detrend_method']))
            sys.exit(1)
        parent_logger = setup_logger_stdout() # Set logger output on parent processor to stdout().

        # Generate list of .dat files
        dat_list = sorted(glob.glob(hotpotato['DAT_DIR']+'/'+hotpotato['glob_dat']))
        create_dir(hotpotato['OUTPUT_DIR'])
        if nproc>1:
            # Distribute calls evenly among child processors.
            distributed_dat_list = np.array_split(np.array(dat_list),nproc)
            # Send data to child processors
            for indx in range(1,nproc):
                comm.send((distributed_dat_list[indx-1], hotpotato), dest=indx, tag=indx)
            for datfile in distributed_dat_list[-1]:
                myexecute(datfile, hotpotato, parent_logger, rank)
            comm.Barrier() # Wait for all child processors to complete call execution.
        else:
            for datfile in dat_list:
                myexecute(datfile, hotpotato, parent_logger, rank)

        # Calculate total run time for the code.
        prog_end_time = time.time()
        run_time = (prog_end_time - prog_start_time)/60.0
        parent_logger.info('Code run time = %.5f minutes'% (run_time))
        print('FINISHING RANK 0')
    else:
        # Recieve data from parent processor.
        call_list, hotpotato = comm.recv(source=0, tag=rank)
        child_logger = setup_logger_stdout()
        print('STARTING RANK: ',rank)
        for datfile in call_list:
            myexecute(datfile, hotpotato, child_logger, rank)
        print('FINISHING RANK: ',rank)
        comm.Barrier() # Send completed status back to parent processor.
##############################################################################
def usage():
    return """
usage: nice -(nice value) mpiexec -n (nproc) python -m mpi4py detrending.py [-h] -i INPUTS_CFG

Detrend multiple dedispersed timeseries using either the PRESTO rednoise program or a fast running median filter.

Argmunents in parenthesis are required numbers for an MPI run.

required arguments:
-i INPUTS_CFG  Configuration script of inputs

optional arguments:
-h, --help     show this help message and exit
    """
##############################################################################
def main():
    """ Command line tool for detrending dedispersed time-series"""
    parser = ArgumentParser(description="Detrend time-series using PRESTO rednoise program or a fast running median technique.",usage=usage(),add_help=False)
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
