# Functions relevant to PRESTO's dedisp.py script.
from builtins import zip, range
from .general_utils import create_dir
from .read_config import read_config
import os
#######################################################################
# Loop over the DDplan steps and obtain a list of calls to execute.
def gen_call_list(basename, OUTPUT_DIR, maskfile, rawfiles, dDMs, dsubDMs, downsamps, subcalls, startDMs, dmspercalls, nsub, other_flags):
    calls = []
    # Output file
    output_basename = OUTPUT_DIR + '/'+ basename
    # RFI mask
    if maskfile is not None:
        mask_string = '-mask %s'% (maskfile)
    else:
        mask_string = ''
    additional_flags = other_flags+' '+mask_string
    for dDM, dsubDM, dmspercall, downsamp, subcall, startDM in zip(dDMs, dsubDMs, dmspercalls, downsamps, subcalls, startDMs):
        # Loop over the number of calls
        for ii in range(subcall):
            subDM = startDM + (ii+0.5)*dsubDM
            loDM = startDM + ii*dsubDM
            subcall = 'prepsubband %s -nsub %d -lodm %.2f -dmstep %.2f -numdms %d -downsamp %d -o %s %s' % (additional_flags, nsub, loDM, dDM, dmspercall, downsamp, output_basename, rawfiles)
            calls.append(subcall)
    print('Total no. of calls = %d'% (len(calls)) )
    return calls
#######################################################################
# Read inputs from a .config file and output list of prepsubband calls.
def config_call(cfg_file):
    dict = read_config(cfg_file)
    # Set default values.
    if dict['maskfile']=='':
        dict['maskfile'] = None
    # Create output directory if non-existent.
    create_dir(dict['OUTPUT_DIR'])
    # Generate list of prepsubband calls
    calls = gen_call_list(dict['basename'], dict['OUTPUT_DIR'], dict['maskfile'], dict['rawfiles'], dict['dDMs'], dict['dsubDMs'], dict['downsamps'], dict['subcalls'], dict['startDMs'], dict['dmspercalls'], dict['nsub'], dict['other_flags'])
    return calls
#######################################################################
