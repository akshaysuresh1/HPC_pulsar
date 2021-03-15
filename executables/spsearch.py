#!/usr/bin/env python
'''
This script runs matched filtering-based single pulse searches on dedispersed time series.
'''
from __future__ import print_function
from __future__ import absolute_import
# Custom packages
from modules.read_config import read_config
from modules.general_utils import create_dir, setup_logger_stdout
# Standard imports
import os, logging, time, sys
import subprocess as sp
from argparse import ArgumentParser
##############################################################
# Single pulse search functionality
def run_spsearch(inputs_cfg):
    # Profile code execution.
    prog_start_time = time.time()

    # Read inputs from config file and set default parameter values, if applicable.
    hotpotato = read_config(inputs_cfg)
    # Set default paramter values, if applicable.
    if hotpotato['threshold']=='':
        hotpotato['threshold'] = 6.0

    WORKING_DIR = os.getcwd()
    os.chdir(hotpotato['DAT_DIR']) # Write data products directly to output directory.
    logger = setup_logger_stdout() # Set logger output to stdout().

    # Single pulse search command
    if hotpotato['other_params']=='':
        spsearch_cmd = 'single_pulse_search.py -t %.1f -g %s'% (hotpotato['threshold'], hotpotato['dat_files'])
    else:
        spsearch_cmd = 'single_pulse_search.py %s -t %.1f -g %s'% (hotpotato['other_params'], hotpotato['threshold'], hotpotato['dat_files'])
    # Execute single pulse search command.
    logger.info(spsearch_cmd)
    status = sp.check_call(spsearch_cmd, shell=True)
    if status==0:
        logger.info('Single pulse searching completed successfully.')
    else:
        logger.error('Single pulse search call terminated!')

    os.chdir(WORKING_DIR)
    # Calculate total run time for the code.
    prog_end_time = time.time()
    run_time = (prog_end_time - prog_start_time)/60.0
    logger.info('Code run time = %.5f minutes'% (run_time))
###################################################
def main():
    """ Command line tool for running single_pulse_search.py """
    parser = ArgumentParser(description="Run single_pulse_search.py on dedispersed time series.")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs for single pulse searching")
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    parse_args = parser.parse_args()
    # Initialize parameter values
    inputs_cfg = parse_args.inputs_cfg

    run_spsearch(inputs_cfg)
##############################################################
if __name__=='__main__':
    main()
##############################################################
