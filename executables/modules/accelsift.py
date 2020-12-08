from __future__ import absolute_import
from builtins import map
import re
import glob
import presto.sifting as sifting
from operator import itemgetter, attrgetter
import pandas as pd
import numpy as np
##################################################################
# Sift through accelsearch candidates and output a .txt file.
def accelsift(globinf, globaccel, outfile, min_num_DMs, low_DM_cutoff, sigma_threshold, c_pow_threshold, known_birds_p, known_birds_f, r_err, P_min, P_max, harm_pow_cutoff):
    # Set sifting values
    sifting.sigma_threshold = sigma_threshold
    sifting.c_pow_threshold = c_pow_threshold
    sifting.known_birds_p = known_birds_p
    sifting.known_birds_f = known_birds_f
    sifting.r_err = r_err
    sifting.short_period = P_min
    sifting.long_period = P_max
    sifting.harm_pow_cutoff = harm_pow_cutoff

    inffiles = sorted(glob.glob(globinf))
    candfiles = sorted(glob.glob(globaccel))
    # Check to see if this is from a short search
    if len(re.findall("_[0-9][0-9][0-9]M_" , inffiles[0])):
        dmstrs = [x.split("DM")[-1].split("_")[0] for x in candfiles]
    else:
        dmstrs = [x.split("DM")[-1].split(".inf")[0] for x in inffiles]
    dms = list(map(float, dmstrs))
    dms.sort()
    dmstrs = ["%.2f"%x for x in dms]

    # Read in all the candidates
    cands = sifting.read_candidates(candfiles)

    # Remove candidates that are duplicated in other ACCEL files
    if len(cands):
        cands = sifting.remove_duplicate_candidates(cands)

    # Remove candidates with DM problems
    if len(cands):
        cands = sifting.remove_DM_problems(cands, min_num_DMs, dmstrs, low_DM_cutoff)

    # Remove candidates that are harmonically related to each other
    # Note:  this includes only a small set of harmonics
    if len(cands):
        cands = sifting.remove_harmonics(cands)

    # Write candidates to STDOUT
    if len(cands):
        cands.sort(key=attrgetter('sigma'), reverse=True)
        sifting.write_candlist(cands, outfile)
##################################################################
# Read candidates from a .txt file output using the above accelsift() module.
def read_candsift(candsift_txt, basename):
    f = open(candsift_txt, 'r')
    lines = f.read().split('\n')
    f.close()
    # Select lines containing the basename.
    select_lines = [line for line in lines if basename in line]
    N_cands = len(select_lines)
    cand_files = []
    cand_numbers = [] #np.zeros(N_cands)
    DM = []
    SNR = []
    sigma = []
    numharm = []
    ipow = []
    cpow = []
    period = [] # ms
    r = []
    z = []
    numhits = []
    # Line entries = Candfile:num, DM, SNR, sigma, numharm, ipow, cpow, P (ms), r, z, numhits
    columns = ['Cand file', 'Cand num', 'DM', 'SNR', 'sigma', 'numharm', 'ipow', 'cpow', 'P (ms)', 'r', 'z', 'numhits']
    for i in range(N_cands):
        candfilenum, DM_str, SNR_str, sigma_str, numharm_str, ipow_str, cpow_str, period_str, r_str, z_str, numhits_str = select_lines[i].split()
        candfile, candnum = candfilenum.split(':')
        cand_files.append(candfile)
        cand_numbers.append(int(candnum))
        DM.append(float(DM_str))
        SNR.append(float(SNR_str))
        sigma.append(float(sigma_str))
        numharm.append(int(numharm_str))
        ipow.append(float(ipow_str))
        cpow.append(float(cpow_str))
        period.append(float(period_str))
        r.append(float(r_str))
        z.append(float(z_str))
        numhits.append(int(numhits_str.split('(')[-1].split(')')[0]))
    content = [cand_files, cand_numbers, DM, SNR, sigma, numharm, ipow, cpow, period, r, z, numhits]
    dict = {}
    for i in range(len(columns)):
        dict[columns[i]] = content[i]
    df = pd.DataFrame(data=dict)
    return df
##################################################################
