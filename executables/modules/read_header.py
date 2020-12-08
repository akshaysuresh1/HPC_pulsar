# Define "Header" class for storing metadata information from filterbank/psrfits files in a common format.

import glob
from astropy.io import fits
from blimpy import Waterfall
from blimpy.io.sigproc import len_header
###################################################
class Header(object):
    " Basic Header object"

    def __init__(self,glob_file_string,file_type='filterbank'):
        file_list = sorted(glob.glob(glob_file_string))
        N_files = len(file_list)
        self.ntsamples = 0
        self.file_type = file_type  # File type

        if (file_type=='filterbank'):
            for i in range(N_files):
                wat = Waterfall(file_list[i],load_data=False)
                self.ntsamples += wat.n_ints_in_file
                if (i==0):
                    self.primary = wat.header
                    self.primary['hdr_size'] = len_header(file_list[i])
                    self.subint = {}

            self.t_samp = self.primary['tsamp']          # Time sampling (s)
            self.tot_time = self.t_samp * self.ntsamples # Total time (s)
            self.chan_bw = self.primary['foff']          # Channel bandwidth (MHz)
            self.fch1 = self.primary['fch1']             # Start frequency (MHz)
            self.nchans = self.primary['nchans']         # No. of channels
            self.npol = self.primary['nifs']             # No. of polarizations


        if (file_type=='psrfits'):
            for i in range(N_files):
                hdulist = fits.open(file_list[i])
                subint = hdulist['SUBINT'].header
                self.ntsamples += subint['NSBLK']*subint['NAXIS2'] # No. of time samples in a file
                if (i==0):
                    self.primary = hdulist['PRIMARY'].header
                    self.subint = subint
                    self.fch1 = hdulist['SUBINT'].data['DAT_FREQ'][0][0] # Start frequency (MHz)

            self.t_samp = self.subint['TBIN'] # Time sampling (s)
            self.tot_time = self.t_samp * self.ntsamples # Total time (s)
            self.chan_bw = self.subint['CHAN_BW'] # Channel bandwidth (MHz)
            self.nchans = self.primary['OBSNCHAN'] # No. of channels
            self.npol = self.subint['NPOL'] # No. of polarizations

    # Print attributes of a Header object.
    def __str__(self):
        lines = [
        "File type                 =  %s"% (self.file_type),
        "Sampling time (ms)        =  %.5g"% (self.t_samp*1e3),
        "No. of time samples       =  %d"% (self.ntsamples),
        "Total time (s)            =  %.5g"% (self.tot_time),
        "Channel bandwidth (MHz)   =  %.5g"% (self.chan_bw),
        "No. of spectral channels  =  %d"% (self.nchans),
        "Start frequency (MHz)     =  %.5g"% (self.fch1),
        "No. of polarizations      =  %d"% (self.npol),

        ]
        output = '\n'.join(lines)
        return output

    # Object representation
    def __repr__(self):
        return str(self)
###################################################
