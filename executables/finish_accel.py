#!/usr/bin/env python
'''
Say that an accelsearch runs gets interrupted for some reason. Use this script to restart accelsearch, while avoiding redundant processing.

Run using following syntax.
nice -<nice value> mpiexec -n <numproc> python -m mpi4py finish_accel.py -i INPUTS_CFG
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

# Set defaults.
def set_defaults(hotpotato):
    if hotpotato['zmax']=='':
        hotpotato['zmax'] = 0
    if hotpotato['wmax']=='':
        hotpotato['wmax'] = 0
    if hotpotato['numharm']=='':
        hotpotato['numharm'] = 8
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
        # Glob string to parse ACCEL files
        if hotpotato['wmax']!=0:
            accel_suffix = 'ACCEL_%d_JERK_%d'% (hotpotato['zmax'], hotpotato['wmax'])
        else:
            accel_suffix = 'ACCEL_%d'% (hotpotato['zmax'])

        # Generate list of basenames from .fft files.
        basename_fft_list = [x.split('/')[-1].split('.fft')[0] for x in sorted(glob.glob(hotpotato['FFT_DIR']+'/'+hotpotato['glob_fft']))]
        # Generate list of basename from ACCEL files.
        basename_accel_list = [x.split('/')[-1].split('_'+accel_suffix)[0] for x in sorted(glob.glob(hotpotato['FFT_DIR']+'/*'+accel_suffix))]
        # Find basenames for which accelsearch was left unfinished.
        unfinished_accel_list = [x for x in basename_fft_list if x not in basename_accel_list]
        # Construct list of .fft files for accelsearch runs.
        fft_list = [hotpotato['FFT_DIR']+'/' + x + '.fft' for x in unfinished_accel_list]

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
##############################################################################
def usage():
    return """
usage: nice -(nice value) mpiexec -n (nproc) python -m mpi4py finish_accel.py [-h] -i INPUTS_CFG

Say that an accelsearch runs gets interrupted for some reason. Use this script to restart accelsearch, while avoiding redundant processing.

Argmunents in parenthesis are required numbers for an MPI run.

required arguments:
-i INPUTS_CFG  Configuration script of inputs

optional arguments:
-h, --help     show this help message and exit
    """
##############################################################################
def main():
    """ Command line tool for detrending dedispersed time-series"""
    parser = ArgumentParser(description="Say that an accelsearch runs gets interrupted for some reason. Use this script to restart accelsearch, while avoiding redundant processing.",usage=usage(),add_help=False)
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
