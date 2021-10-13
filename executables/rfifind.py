#!/usr/bin/env python
'''
This script runs rfifind on one or multiple data sets.

Usage:
python rfifind.py -i <Configuration script of inputs>
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
# Rfifind functionality
def run_rfifind(inputs_cfg):
    # Profile code execution.
    prog_start_time = time.time()

    # Read inputs from config file and set default parameter values, if applicable.
    hotpotato = read_config(inputs_cfg)
    # Set default paramter values, if applicable.
    if hotpotato['RFIMASK_DIR']=='':
        hotpotato['RFIMASK_DIR'] = hotpotato['DATA_DIR']
    if hotpotato['do_masktweak']=='':
        hotpotato['do_masktweak'] = False

    WORKING_DIR = os.getcwd()

    create_dir(hotpotato['RFIMASK_DIR'])
    os.chdir(hotpotato['RFIMASK_DIR']) # Write data products directly to output directory.
    logger = setup_logger_stdout() # Set logger output to stdout().

    rfimask_file = hotpotato['basename']+'_rfifind.mask' # Name of rfifind mask
    # Base rfifind shell command to be run if rfifind mask is not available.
    rfifind_cmd = 'rfifind -o %s %s %s'% (hotpotato['basename'], hotpotato['rfifind_params'],hotpotato['DATA_DIR']+'/'+hotpotato['data_files'])
    if not os.path.isfile(rfimask_file):
        logger.info('No pre-existing rfifind mask found.')
        # Basic rfifind call
        logger.info(rfifind_cmd)
        status = sp.check_call(rfifind_cmd, shell=True) # Overwrites existing mask.
        if status==0:
            logger.info('Rfifind call completed successfully.')
        else:
            logger.error('Rfifind call terminated!')
    else:
        logger.info('%s already exists. Delete existing file if you want RFI stats recomputed.'% (rfimask_file))
    # Tweak rfifind mask once generated.
    if hotpotato['do_masktweak']:
        rfifind_cmd = rfifind_cmd + ' -nocompute %s -mask %s'% (hotpotato['masktweak_params'], rfimask_file)
        logger.info(rfifind_cmd)
        status = sp.check_call(rfifind_cmd, shell=True)
        if status==0:
            logger.info('RFI mask tweaked according to supplied prescription.')
        else:
            logger.error('RFI mask tweak failed!')

    os.chdir(WORKING_DIR)
    # Calculate total run time for the code.
    prog_end_time = time.time()
    run_time = (prog_end_time - prog_start_time)/60.0
    logger.info('Code run time = %.5f minutes'% (run_time))
###################################################
def main():
    """ Command line tool for running rfifind. """
    parser = ArgumentParser(description="Run rfifind on one or more data sets.")
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs to rfifind")
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)

    parse_args = parser.parse_args()
    # Initialize parameter values
    inputs_cfg = parse_args.inputs_cfg

    run_rfifind(inputs_cfg)
##############################################################
if __name__=='__main__':
    main()
##############################################################
