#!/usr/bin/env python
'''
Supply outputs from DDplan.py to run multiple prepsubband calls in parallel.

Run using following syntax.
nice -<nice value> mpiexec -n <numproc> python -m mpi4py dedispersion.py -i <Configuration script of inputs>
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom packages
from modules.read_config import read_config
from modules.general_utils import setup_logger_stdout
from modules.module_dedisp import config_call
# Standard imports
from mpi4py import MPI
import os, logging, time, sys
import numpy as np
import subprocess as sp
from argparse import ArgumentParser
###################################################################################
# Execute call.
def myexecute(call, logger, rank):
    print('RANK %d: %s'% (rank, call))
    status = sp.check_call(call,shell=True)
    if status==0:
        logger.info('RANK %d: Call execution complete.'% (rank))
    else:
        logger.warning('RANK %d: Prepsubband call failed.'% (rank))

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
        calls = config_call(inputs_cfg)
        parent_logger = setup_logger_stdout() # Set logger output of parent processor to stdout().

        if nproc>=2:
            # In case of multiple processors, the parent processor distributes calls evenly between the child processors and itself.
            distributed_calls = np.array_split(np.array(calls),nproc)
            # Send calls to child processors.
            for indx in range(1,nproc):
                comm.send(distributed_calls[indx-1], dest=indx, tag=indx)
            # Run tasks assigned to parent processor.
            for call in distributed_calls[-1]:
                myexecute(call, parent_logger, rank)
            comm.Barrier() # Wait for all processors to complete their respective calls.
        else:
            # In case nproc=1, parent (or lone) processor runs all tasks,
            for call in calls:
                myexecute(call, parent_logger, rank)

        # Calculate total run time for the code.
        prog_end_time = time.time()
        run_time = (prog_end_time - prog_start_time)/60.0
        parent_logger.info('Code run time = %.5f minutes'% (run_time))
        print('FINISHING RANK 0')
    else:
        # Recieve data from parent processor.
        call_list = comm.recv(source=0, tag=rank)
        print('STARTING RANK: ',rank)
        child_logger = setup_logger_stdout() # Set up separate logger for each child processor.
        for call in call_list:
            myexecute(call, child_logger, rank)
        print('FINISHING RANK: ',rank)
        comm.Barrier() # Await all processsors to complete respective tasks.
##############################################################################
def usage():
    return """
usage: nice -(nice value) mpiexec -n (nproc) python -m mpi4py dedispersion.py [-h] -i INPUTS_CFG

Run MPI-prepsubband on a data set.

Argmunents in parenthesis are required numbers for an MPI run.

required arguments:
-i INPUTS_CFG  Configuration script of inputs to prepsubband

optional arguments:
-h, --help     show this help message and exit
    """
##############################################################################
def main():
    """ Command line tool for running prepsubband on multiple processors"""
    parser = ArgumentParser(description="Run MPI-prepsubband on a data set.",usage=usage(),add_help=False)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs to prepsubband")
    parser._action_groups.append(optional)

    # Run MPI-parallelized prepsubband.
    __MPI_MAIN__(parser)
##############################################################################
if __name__=='__main__':
    main()
##############################################################################
