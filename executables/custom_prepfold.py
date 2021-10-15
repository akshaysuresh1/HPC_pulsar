#!/usr/bin/env python
'''
Fold raw data using prepfold at user-specified parameters.

Usage:
python custom_prepfold.py -i <Configuration script of inputs>
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom packages
from modules.read_config import read_config
from modules.general_utils import create_dir, setup_logger_stdout
# Standard imports
import os, logging, time, sys, glob
import numpy as np
import subprocess as sp
from argparse import ArgumentParser
##############################################################
# Fold candidates.
def user_prepfold(inputs_cfg):
    # Profile code execution.
    prog_start_time = time.time()

    # Read inputs from config file and set default parameter values, if applicable.
    hotpotato = read_config(inputs_cfg)
    hotpotato = set_defaults(hotpotato)
    logger = setup_logger_stdout() # Set logger output to stdout().

    WORKING_DIR = os.getcwd() # Retrieve path to current working directory.
    # Shift operations to output path.
    create_dir(hotpotato['OUTPUT_DIR'])
    os.chdir(hotpotato['OUTPUT_DIR'])

    # Prepfold call
    datafiles = hotpotato['DATA_DIR']+'/'+hotpotato['glob_data']
    prepfold_cmd = 'prepfold -noxwin -o %s -n %d -nsub %d -p %s -dm %s -mask %s %s %s'% (hotpotato['basename'], hotpotato['num_phasebins'], hotpotato['N_subbands'], hotpotato['P'], hotpotato['DM'], hotpotato['maskfile'], hotpotato['flags'], datafiles)
    logger.info(prepfold_cmd)
    status = sp.check_call(prepfold_cmd, shell=True)
    if status==0:
        logger.info('Raw data successfully folded.')
    else:
        logger.warning('Folding error.')
    os.chdir(WORKING_DIR)

    # Calculate total run time for the code.
    prog_end_time = time.time()
    run_time = (prog_end_time - prog_start_time)/60.0
    logger.info('Code run time = %.5f minutes'% (run_time))

# Set defaults.
def set_defaults(hotpotato):
    if hotpotato['OUTPUT_DIR']=='':
        hotpotato['OUTPUT_DIR'] = hotpotato['DATA_DIR']
    if hotpotato['maskfile']=='':
        hotpotato['maskfile'] = None
    if hotpotato['DM']=='':
        hotpotato['DM'] = 0.0
    if hotpotato['P']=='':
        hotpotato['P'] = 1.0
    if hotpotato['num_phasebins']=='':
        hotpotato['num_phasebins'] = 64
    if hotpotato['N_subbands']=='':
        hotpotato['N_subbands'] = 128
    return hotpotato
##############################################################
def main():
    """ Command line tool for folding raw data at user-specified parameters using prepfold. """
    parser = ArgumentParser(description="Fold raw data using prepfold.")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs")
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    parse_args = parser.parse_args()
    # Initialize parameter values
    inputs_cfg = parse_args.inputs_cfg

    user_prepfold(inputs_cfg)
##############################################################
if __name__=='__main__':
    main()
##############################################################
