#!/usr/bin/env python
'''
Write out useful accelsearch and FFT of DM = 0 pc/cc timeseries to facilitate manual creation of .birds file.
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
def run_birds(hotpotato):
    # Profile code execution.
    prog_start_time = time.time()

    logger = setup_logger_stdout() # Set logger output to stdout().
    create_dir(hotpotato['BIRDS_DIR'])

    if hotpotato['data_type']=='ds':
        ds_file = hotpotato['SRC_DIR'] + '/' + hotpotato['data_file']
        output_DM0_basename = hotpotato['BIRDS_DIR'] + '/' + hotpotato['basename']
        # Output DM = 0 pc/cc timeseries using prepdata.
        prepdata_cmd = 'prepdata -dm 0.0 -nobary -o %s/%s %s'% (hotpotato['BIRDS_DIR'], hotpotato['basename'], ds_file)
        logger.info(prepdata_cmd)
        status = sp.check_call(prepdata_cmd, shell=True)
        if status==0:
            logger.info('DM = 0 pc/cc topocentric timeseries generated.')
        else:
            logger.warning('Prepdata failed.')
    elif hotpotato['data_type']=='timeseries':
        # Copy DM = 0 pc/cc timeseries to BIRDS_DIR. Start by copying .dat file first.
        cp_dat_cmd = 'cp %s/%s %s/%s.dat'% (hotpotato['SRC_DIR'], hotpotato['data_file'], hotpotato['BIRDS_DIR'], hotpotato['basename'])
        logger.info(cp_dat_cmd)
        status = sp.check_call(cp_dat_cmd, shell=True)
        # Copy .inf file associated with above .dat file.
        inf_file = hotpotato['data_file'].split('.dat')[0] + '.inf'
        cp_inf_cmd = 'cp %s/%s %s/%s.inf'% (hotpotato['SRC_DIR'], inf_file, hotpotato['BIRDS_DIR'], hotpotato['basename'])
        logger.info(cp_inf_cmd)
        status = sp.check_call(cp_inf_cmd, shell=True)
        if status==0:
            logger.info('Copied DM = 0 pc/cc timeseries to specified output path.')
        else:
            logger.warning('Error in copy of DM = 0 pc/cc timeseries to output path.')
    else:
        logger.critical('Data format not recognized.')
        sys.exit(1)

    # Compute FFT of DM = 0 pc/cc timeseries.
    realfft_cmd = 'realfft %s/%s.dat'% (hotpotato['BIRDS_DIR'], hotpotato['basename'])
    logger.info(realfft_cmd)
    status = sp.check_call(realfft_cmd, shell=True)
    if status==0:
        logger.info('FFT of DM = 0 pc/cc timeseries computed.')
    else:
        logger.warning('Error in FFT computation.')

    # Trick accelsearch into finding periodic interference in DM = 0 pc/cc topocentric timeseries.
    accelsearch_cmd = 'accelsearch -numharm 4 -zmax 0 %s/%s.dat'% (hotpotato['BIRDS_DIR'], hotpotato['basename'])
    logger.info(accelsearch_cmd)
    status = sp.check_call(accelsearch_cmd, shell=True)
    if status==0:
        logger.info('Successfully ran accelsearch on DM = 0 pc/cc to identify periodic interference.')
    else:
        logger.warning('Accelsearch failed.')

    # Copy rfifind .inf file to output directory with appropriate basename.
    cp_cmd = 'cp %s %s/%s_birds.inf'% (hotpotato['rfimask_inf'], hotpotato['BIRDS_DIR'], hotpotato['basename'])
    logger.info(cp_cmd)
    status = sp.check_call(cp_cmd, shell=True)
    if status==0:
        logger.info('Copy completed.')
    else:
        logger.warning('Copy of .inf file failed.')

    # Calculate total run time for the code.
    prog_end_time = time.time()
    run_time = (prog_end_time - prog_start_time)/60.0
    logger.info('Code run time = %.5f minutes'% (run_time))
##############################################################
def main():
    """ Command line tool for running rfifind. """
    parser = ArgumentParser(description="Write out useful accelsearch and FFT of DM = 0 pc/cc timeseries to facilitate manual creation of .birds file.")
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

    hotpotato = read_config(inputs_cfg)
    run_birds(hotpotato)
##############################################################
if __name__=='__main__':
    main()
##############################################################
