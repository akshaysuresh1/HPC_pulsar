#!/usr/bin/env python
'''
# Fix path contained within basename in PRESTO .inf files to prepare for red noise removal.

Run using following syntax.
nice -<nice value> mpiexec -n <numproc> python -m mpi4py fix_path_inf.py -i fix_path_inf.cfg
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom packages
from modules.read_config import read_config
from modules.general_utils import setup_logger_stdout
# Standard imports
from mpi4py import MPI
from presto.infodata import infodata
import os, logging, time, sys, glob
import numpy as np
import subprocess as sp
from argparse import ArgumentParser
##############################################################################
# Execute call.
def myexecute(inf_file, hotpotato, logger, rank):
    metadata = infodata(inf_file)
    if '/' in metadata.basenm:
        base = metadata.basenm.split('/')[-1]
        metadata.basenm = hotpotato['new_path'] + '/' + base
        metadata.to_file(inf_file)
        logger.info('RANK %d: Path fixed for %s.'% (rank, inf_file))

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
        # Read .cfg file into a dictionary.
        hotpotato = read_config(inputs_cfg)
        parent_logger = setup_logger_stdout() # Set logger output of parent processor to stdout().

        parent_logger.info('Building list of .inf files')
        inf_file_list = sorted(glob.glob(hotpotato['INF_DIR']+'/'+hotpotato['glob_inf']))

        if nproc>=2:
            # Split .inf files equally across all processors.
            distributed_list = np.array_split(np.array(inf_file_list), nproc)
            # Send calls to child processors.
            for indx in range(1,nproc):
                comm.send((distributed_list[indx-1], hotpotato), dest=indx, tag=indx)
            # Run tasks assigned to parent processor.
            for inf_file in distributed_list[-1]:
                myexecute(inf_file, hotpotato, parent_logger, rank)
            comm.Barrier() # Wait for all processors to complete their respective calls.
        else:
            # In case nproc=1, parent (or lone) processor runs all tasks.
            for inf_file in inf_file_list:
                myexecute(inf_file, hotpotato, parent_logger, rank)

        # Calculate total run time for the code.
        prog_end_time = time.time()
        run_time = (prog_end_time - prog_start_time)/60.0
        parent_logger.info('Code run time = %.5f minutes'% (run_time))
        print('FINISHING RANK 0')
    else:
        # Recieve data from parent processor.
        inf_list_ch, hotpotato = comm.recv(source=0, tag=rank)
        print('STARTING RANK: ',rank)
        child_logger = setup_logger_stdout() # Set up separate logger for each child processor.
        for inf_file in inf_list_ch:
            myexecute(inf_file, hotpotato, child_logger, rank)
        print('FINISHING RANK: ',rank)
        comm.Barrier() # Await all processsors to complete respective tasks.
##############################################################################
def usage():
    return """
usage: nice -(nice value) mpiexec -n (nproc) python -m mpi4py fix_path_inf.py [-h] -i INPUTS_CFG

Fix path contained within basename in PRESTO .inf files.

Argmunents in parenthesis are required numbers for an MPI run.

required arguments:
-i INPUTS_CFG  Configuration script of inputs

optional arguments:
-h, --help     show this help message and exit
    """
#############################################################################
def main():
    """ Command line tool for running fix_path_inf.py on multiple processors"""
    parser = ArgumentParser(description="Fix path contained within basename in PRESTO .inf files.",usage=usage(),add_help=False)
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
