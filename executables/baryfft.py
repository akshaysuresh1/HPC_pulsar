#!/usr/bin/env python
'''
Barycenter timeseries, compute FFT and zap periodic interference.

Run using following syntax.
nice -<nice value> mpiexec -n <numproc> python -m mpi4py baryfft.py -i INPUTS_CFG
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom packages
from modules.read_config import read_config
from modules.general_utils import setup_logger_stdout, create_dir
# Standard imports
from mpi4py import MPI
import os, logging, time, sys, glob
import numpy as np
import subprocess as sp
from argparse import ArgumentParser
##############################################################################
# Execute call.
def myexecute(datfile, hotpotato, logger, DATA_DIR, rank):
    if '/' in datfile:
        basename = datfile.split('/')[-1].split('.dat')[0]
    else:
        basename = datfile.split('.dat')[0]
    # Build up zapping command part by part.
    zap_prefix = 'zapbirds -zap -zapfile %s'% (hotpotato['zapfile'])
    # Barycenter timeseries if required.
    if hotpotato['do_barycenter']:
        base_prefix, DMvalue = basename.split('_DM')
        basename = base_prefix+'_bary_DM'+DMvalue
        prepdata_cmd = 'prepdata -o %s %s'% (DATA_DIR+'/'+basename, datfile)
        logger.info('RANK %d: '% (rank) + prepdata_cmd)
        status = sp.check_call(prepdata_cmd, shell=True)
        if status==0:
            logger.info('RANK %d: Barycentered %s.'% (rank, basename))
        else:
            logger.warning('RANK %d: Barycentering failed for %s.'% (rank, basename))
        zap_prefix = zap_prefix + ' -baryv %s'% (hotpotato['baryv'])
    # Compute FFT of timeseries.
    realfft_cmd = 'realfft %s'% (DATA_DIR+'/'+basename+'.dat')
    logger.info('RANK %d: '% (rank) + realfft_cmd)
    status = sp.check_call(realfft_cmd, shell=True)
    if status==0:
        logger.info('RANK %d: FFT of %s computed.'% (rank, basename))
    else:
        logger.warning('RANK %d: FFT computation failed for %s.'% (rank, basename))
    # Run zapping of periodic interference in FFT.
    fft_file = DATA_DIR+'/'+basename+'.fft'
    zap_cmd = zap_prefix + ' %s'% (fft_file)
    logger.info('RANK %d: '% (rank) + zap_cmd)
    status = sp.check_call(zap_cmd, shell=True)
    if status==0:
        logger.info('RANK %d: Zapping complete for %s.'% (rank, fft_file))
    else:
        logger.warning('RANK %d: Zapping failed for %s.'% (rank, fft_file))

# Calculate barycentric velocity for use when barycentering zaplist frequencies at DM = 0 pc/cc.
def calc_baryv(datfile, hotpotato, logger):
    baryv_cmd = 'prepdata -dm 0.0 -numout 8 -o tmp %s | grep Average'% (datfile)
    baryv = float(sp.check_output(baryv_cmd,shell=True).decode('utf-8').split('=')[1])
    hotpotato['baryv'] = baryv
    sp.check_call('rm -rf tmp.*', shell=True)
    logger.info('Average barycentric velocity (c) = %s'% (baryv))
    return hotpotato

# Set defaults.
def set_defaults(hotpotato):
    if hotpotato['do_barycenter']=='':
        hotpotato['do_barycenter'] = True
    if hotpotato['BARY_DIR']=='':
        hotpotato['BARY_DIR'] = hotpotato['DAT_DIR']
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
        parent_logger = setup_logger_stdout() # Set logger output of parent processor to stdout().

        # Generate list of .dat files
        dat_list = sorted(glob.glob(hotpotato['DAT_DIR']+'/'+hotpotato['glob_dat']))

        if hotpotato['do_barycenter']:
            create_dir(hotpotato['BARY_DIR'])
            calc_baryv(dat_list[0], hotpotato, parent_logger)
            DATA_DIR = hotpotato['BARY_DIR']
        else:
            DATA_DIR = hotpotato['DAT_DIR']

        if nproc>1:
            # Distribute calls evenly among child processors.
            distributed_dat_list = np.array_split(np.array(dat_list),nproc-1)
            # Send data to child processors
            for indx in range(1,nproc):
                comm.send((distributed_dat_list[indx-1], hotpotato, DATA_DIR), dest=indx, tag=indx)
            for datfile in distributed_dat_list[-1]:
                myexecute(datfile, hotpotato, parent_logger, DATA_DIR, rank)
            comm.Barrier() # Wait for all child processors to receive sent call.
        else:
            for datfile in dat_list:
                myexecute(datfile, hotpotato, parent_logger, DATA_DIR, rank)

        # Calculate total run time for the code.
        prog_end_time = time.time()
        run_time = (prog_end_time - prog_start_time)/60.0
        parent_logger.info('Code run time = %.5f minutes'% (run_time))
        print('FINISHING RANK 0')
    else:
        # Recieve data from parent processor.
        call_list, hotpotato, DATA_DIR = comm.recv(source=0, tag=rank)
        print('STARTING RANK: ',rank)
        child_logger = setup_logger_stdout() # Set up separate logger for each child processor.
        for datfile in call_list:
            myexecute(datfile, hotpotato, child_logger, DATA_DIR, rank)
        print('FINISHING RANK: ',rank)
        comm.Barrier() # Send completed status back to parent processor.
##############################################################################
def usage():
    return """
usage: nice -(nice value) mpiexec -n (nproc) python -m mpi4py baryfft.py [-h] -i INPUTS_CFG

Barycenter timeseries, compute FFT and zap periodic interference.

Argmunents in parenthesis are required numbers for an MPI run.

required arguments:
-i INPUTS_CFG  Configuration script of inputs

optional arguments:
-h, --help     show this help message and exit
    """
##############################################################################
def main():
    """ Command line tool for detrending dedispersed time-series"""
    parser = ArgumentParser(description="Barycenter timeseries, compute FFT and zap periodic interference.",usage=usage(),add_help=False)
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
