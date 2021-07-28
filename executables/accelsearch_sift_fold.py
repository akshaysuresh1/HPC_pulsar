#!/usr/bin/env python
'''
Run acceleration searches on dedispersed timeseries. Also, sift through candidates and output prepfold plots.

Run using following syntax.
nice -<nice value> mpiexec -n <numproc> python -m mpi4py accelsearch_sift_fold.py -i INPUTS_CFG
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom packages
from modules.read_config import read_config
from modules.general_utils import setup_logger_stdout, create_dir
from modules.accelsift import accelsift, read_candsift
# Standard imports
from mpi4py import MPI
import os, logging, time, sys, glob
import numpy as np
import subprocess as sp
from argparse import ArgumentParser
##############################################################################
# Accelsearch call.
def accelsearch(fftfile, hotpotato, logger, rank):
    accelsearch_cmd = 'accelsearch -numharm %d -zmax %d -wmax %d %s %s'% (hotpotato['numharm'], hotpotato['zmax'], hotpotato['wmax'], hotpotato['accel_params'], fftfile)
    logger.info('RANK %d: '% (rank) + accelsearch_cmd)
    status = sp.check_call(accelsearch_cmd, shell=True)
    if status==0:
        logger.info('RANK %d: Accelsearch with zmax = %d and wmax = %d completed on %s.'% (rank, hotpotato['zmax'], hotpotato['wmax'], fftfile))
    else:
        logger.warning('RANK %d: Accelsearch failed to run for %s.'% (rank, fftfile))

# Time-series folding with prepfold
def timeseries_fold(candfile, candnum, hotpotato, logger, rank):
    datfile = hotpotato['FFT_DIR'] + '/' + candfile.split('_ACCEL')[0]+'.dat'
    DM_tag = '_DM'+candfile.split('_ACCEL')[0].split('_DM')[1]
    output_basename = hotpotato['fold_basename'] + DM_tag
    prepfold_cmd = 'prepfold -noxwin -o %s -accelfile %s/%s.cand -accelcand %d %s'% (output_basename, hotpotato['FFT_DIR'], candfile, candnum, datfile)
    logger.info('RANK %d: '% (rank) + prepfold_cmd)
    status = sp.check_call(prepfold_cmd, shell=True)
    if status==0:
        logger.info('RANK %d: Time-series folded for candidate %d in file %s.'% (rank, candnum, candfile))
    else:
        logger.warning('RANK %d: Folding error for candidate %d in file %s.'% (rank, candnum, candfile))

# Set defaults.
def set_defaults(hotpotato):
    if hotpotato['zmax']=='':
        hotpotato['zmax'] = 0
    if hotpotato['wmax']=='':
        hotpotato['wmax'] = 0
    if hotpotato['numharm']=='':
        hotpotato['numharm'] = 8
    if hotpotato['min_num_DMs']=='':
        hotpotato['min_num_DMs'] = 2
    if hotpotato['low_DM_cutoff']=='':
        hotpotato['low_DM_cutoff'] = 2.0
    if hotpotato['sigma_threshold']=='':
        hotpotato['sigma_threshold'] = 4.0
    if hotpotato['c_pow_threshold']=='':
        hotpotato['c_pow_threshold'] = 100.0
    if hotpotato['r_err']=='':
        hotpotato['r_err'] = 1.1
    if hotpotato['P_min']=='':
        hotpotato['P_min'] = 0.0005
    if hotpotato['P_max']=='':
        hotpotato['P_max'] = 30.0
    if hotpotato['harm_pow_cutoff']=='':
        hotpotato['harm_pow_cutoff'] = 8.0
    if hotpotato['known_birds_p']=='':
        hotpotato['known_birds_p'] = []
    if hotpotato['known_birds_f']=='':
        hotpotato['known_birds_f'] = []
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

        parent_logger = setup_logger_stdout() # Set logger output of parent processor to stdout().
        # Construct list of calls to run from shell.
        hotpotato = set_defaults(read_config(inputs_cfg))
        # PRESTO sets zmax to a multiple of 2.
        hotpotato['zmax'] = int(2*np.ceil(hotpotato['zmax']/2))
        # PRESTO requries wmax to be a multiple of 20.
        hotpotato['wmax'] = int(20*np.ceil(hotpotato['wmax']/20))
        # Update FOLD_DIR to reflect above zmax and wmax values.
        hotpotato['FOLD_DIR'] = hotpotato['FOLD_DIR']+'/zmax%d_wmax%d_numharm%d'% (hotpotato['zmax'], hotpotato['wmax'], hotpotato['numharm'])

        # Generate list of .fft files
        fft_list = sorted(glob.glob(hotpotato['FFT_DIR']+'/'+hotpotato['glob_fft']))
        if nproc>1:
            # Distribute calls evenly among child processors.
            distributed_fft_list = np.array_split(np.array(fft_list),nproc)
            # Send data to child processors
            for indx in range(1,nproc):
                comm.send((distributed_fft_list[indx-1], hotpotato), dest=indx, tag=indx)
            for fftfile in distributed_fft_list[-1]:
                accelsearch(fftfile, hotpotato, parent_logger, rank)
            comm.Barrier() # Await child processors.
        else:
            for fftfile in fft_list:
                accelsearch(fftfile, hotpotato, parent_logger, rank)

        # Sift through accelsearch candidates.
        parent_logger.info('Sifting through accelsearch candidates')
        basename = fft_list[0].split('/')[-1].split('_DM')[0]
        # Glob string to parse .inf files
        globinf = hotpotato['FFT_DIR'] + '/' + basename + '*.inf'
        # Glob string to parse ACCEL files
        if hotpotato['wmax']!=0:
            accel_suffix = 'ACCEL_%d_JERK_%d'% (hotpotato['zmax'], hotpotato['wmax'])
        else:
            accel_suffix = 'ACCEL_%d'% (hotpotato['zmax'])
        globaccel = hotpotato['FFT_DIR'] + '/' + basename + '*_'+ accel_suffix
        # Output candidate file
        output_candfile_sifted = hotpotato['FFT_DIR'] + '/' + basename + '_' + accel_suffix +'.txt'
        accelsift(globinf, globaccel, output_candfile_sifted, hotpotato['min_num_DMs'], hotpotato['low_DM_cutoff'], hotpotato['sigma_threshold'], hotpotato['c_pow_threshold'], hotpotato['known_birds_p'], hotpotato['known_birds_f'], hotpotato['r_err'], hotpotato['P_min'], hotpotato['P_max'], hotpotato['harm_pow_cutoff'])

        # Read .txt file of sifted candidates in as a pandas DataFrame.
        parent_logger.info('Reading contents of %s into a Pandas DataFrame'% (output_candfile_sifted))
        df = read_candsift(output_candfile_sifted, basename)
        cand_files = np.array(df['Cand file']) # Array of candidate file names without their .cand extension
        cand_numbers = np.array(df['Cand num']) # Candidate numbers within above files

        create_dir(hotpotato['FOLD_DIR'])

        if nproc>1:
            # Distribute candidate numbers and candidate file list evenly among child processors.
            parent_logger.info('Distributing candidate info to child processors for parallelized time-series folding')
            distributed_candfile_list = np.array_split(np.array(cand_files),nproc)
            distributed_cand_numbers = np.array_split(np.array(cand_numbers),nproc)
            # Send data to child processors.
            for indx in range(1,nproc):
                comm.send((distributed_candfile_list[indx-1], distributed_cand_numbers[indx-1], hotpotato), dest=indx, tag=indx)
            WORKING_DIR = os.getcwd()
            os.chdir(hotpotato['FOLD_DIR'])
            for i in range(len(distributed_candfile_list[-1])):
                timeseries_fold(distributed_candfile_list[-1][i], distributed_cand_numbers[-1][i], hotpotato, parent_logger, rank)
            os.chdir(WORKING_DIR)
            comm.Barrier() # Wait for all child processors to complete call execution.
        else:
            WORKING_DIR = os.getcwd()
            os.chdir(hotpotato['FOLD_DIR'])
            for i in range(len(cand_files)):
                timeseries_fold(cand_files[i], cand_numbers[i], hotpotato, parent_logger, rank)
            os.chdir(WORKING_DIR)

        # Calculate total run time for the code.
        prog_end_time = time.time()
        run_time = (prog_end_time - prog_start_time)/60.0
        parent_logger.info('Code run time = %.5f minutes'% (run_time))
        print('FINISHING RANK 0')
    else:
        # Recieve data from parent processor.
        call_list, hotpotato = comm.recv(source=0, tag=rank)
        print('STARTING RANK: ',rank)
        child_logger = setup_logger_stdout() # Set up a separate logger for each child processor.
        for fftfile in call_list:
            accelsearch(fftfile, hotpotato, child_logger, rank)
        print('FINISHING RANK: ',rank)
        comm.Barrier() # Send completed status back to parent processor.

        # Recieve data from parent processor.
        candfile_list, candnum_list, hotpotato = comm.recv(source=0, tag=rank)
        print('RESTARTING RANK: ',rank)
        WORKING_DIR = os.getcwd()
        os.chdir(hotpotato['FOLD_DIR'])
        for i in range(len(candfile_list)):
            timeseries_fold(candfile_list[i], candnum_list[i], hotpotato, child_logger, rank)
        os.chdir(WORKING_DIR)
        print('FINISHING RANK: ',rank)
        comm.Barrier()
##############################################################################
def usage():
    return """
usage: nice -(nice value) mpiexec -n (nproc) python -m mpi4py accelsearch_sift_fold.py [-h] -i INPUTS_CFG

Run acceleration searches on dedispersed timeseries. Also, sift through candidates and output prepfold plots.

Argmunents in parenthesis are required numbers for an MPI run.

required arguments:
-i INPUTS_CFG  Configuration script of inputs

optional arguments:
-h, --help     show this help message and exit
    """
##############################################################################
def main():
    """ Command line tool for detrending dedispersed time-series"""
    parser = ArgumentParser(description="Run acceleration searches on dedispersed timeseries, and sift through candidates.",usage=usage(),add_help=False)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs")
    parser._action_groups.append(optional)

    # Run MPI-parallelized prepsubband.
    __MPI_MAIN__(parser)
##############################################################################
if __name__=='__main__':
    main()
##############################################################################
