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
from modules.accelsift import read_candsift
# Standard imports
import os, logging, time, sys, glob
import numpy as np
import subprocess as sp
from argparse import ArgumentParser
##############################################################
# Fold candidates.
def fold_rawcands(inputs_cfg):
    # Profile code execution.
    prog_start_time = time.time()

    # Read inputs from config file and set default parameter values, if applicable.
    hotpotato = read_config(inputs_cfg)
    logger = setup_logger_stdout() # Set logger output to stdout().
    WORKING_DIR = os.getcwd()

    # Read in selected candidates to fold on raw data.
    os.chdir(hotpotato['candfiles_path'])
    candfiles = sorted(glob.glob(hotpotato['cand_basename']+'*.png'))
    os.chdir(WORKING_DIR)

    datafiles = hotpotato['DATA_DIR']+'/'+hotpotato['glob_data']

    # Extract file basename from accelsift .txt file.
    accelsift_basename = hotpotato['accelsift_txt']
    if '/' in accelsift_basename:
        accelsift_basename = accelsift_basename.split('/')[-1]
    if '_ACCEL' in accelsift_basename:
        accelsift_basename = accelsift_basename.split('_ACCEL')[0]
    cands_df = read_candsift(hotpotato['accelsift_txt'], accelsift_basename)
    cands_DM = np.array(cands_df['DM']) # Candidate dispersion measure (pc/cc)
    cands_period = np.array(cands_df['P (ms)'])*1.e-3 # Candidate periods (s)
    cands_num = np.array(cands_df['Cand num'], dtype=int) # Candidate numbers in ACCEL files

    create_dir(hotpotato['FOLD_DIR'])
    os.chdir(hotpotato['FOLD_DIR'])
    # Folding candidates on raw data.
    N_cands = len(candfiles)
    logger.info('No. of candidates to fold on raw data = %d'% (N_cands))
    for i in range(N_cands):
        select_DM = float(candfiles[i].split('_DM')[1].split('_')[0])
        select_num = int(candfiles[i].split('Cand_')[1].split('.')[0])
        select_index = np.where(np.logical_and(cands_DM==select_DM, cands_num==select_num))[0][0]

        outfile_name = candfiles[i].split('.pfd.png')[0]+'_rawfold'
        prepfold_cmd = 'prepfold -noxwin -o %s -n %d -nsub %d -p %s -dm %s -mask %s %s %s'% (outfile_name,hotpotato['num_phasebins'], hotpotato['N_subbands'], cands_period[select_index], cands_DM[select_index], hotpotato['maskfile'], hotpotato['other_foldparams'], datafiles)
        logger.info(prepfold_cmd)

        status = sp.check_call(prepfold_cmd, shell=True)
        if status==0:
            logger.info('Raw data successfully folded for candidate %s.'% (candfiles[i]))
        else:
            logger.warning('Folding error for candidate %s.'% (candfiles[i]))

    os.chdir(WORKING_DIR)

    # Calculate total run time for the code.
    prog_end_time = time.time()
    run_time = (prog_end_time - prog_start_time)/60.0
    logger.info('Code run time = %.5f minutes'% (run_time))
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

    fold_rawcands(inputs_cfg)
##############################################################
if __name__=='__main__':
    main()
##############################################################
