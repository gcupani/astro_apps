# ------------------------------------------------------------------------------
# FORMAT_DAS
# Format spectra to be processed with the ESPRESSO DAS
# v2.0 - 2018-11-30
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------
# Sample run:
# > python format_das.py -h         // Help
# > python format_das.py -i='uves'  // Format UVES spectra
# ------------------------------------------------------------------------------

import argparse
from astropy.io import ascii, fits
from copy import deepcopy as dc
import glob
import numpy as np
import os

### Scroll down for editable parameters... ###

class Format():
    """ Class for handling spectrum formats """

    def __init__(self, path_i, # Path
                 instr,        # Instrument
                 rv=0.0,       # RV
                 zem=0.0,      # z
                 path_o=None,  # Path to reformatted frames
                 prefix=None   # Prefix for reformatted frames
                               # (Default: OBS.TARG.NAME_DATE-OBS_ARM)
    ):
        """ Initialize the format """
        
        self.path_i = path_i
        self.path_o = path_o
        self.prefix = prefix
        self.rv = rv
        self.zem = zem        

    def convert(self, hdr):
        """ Convert to DAS format and save the frames """

        # Adjust prefix
        if self.prefix == None:
            self.prefix = hdr['ESO OBS TARG NAME']+'_'+hdr['DATE-OBS']+self.arm
        if self.path_o != None:
            self.prefix = self.path_o+self.prefix

        # Tags for non-DRS input data
        tag = ['WAVE_MATRIX_FIBER', 'SPEC_FLUX_2D', 'SPEC_FLUXERR_2D']

        # Corresponding data
        data = [self.wave, self.flux, self.err]
        
        for t, d in zip(tag, data):

            # Add required ESPRESSO keywords to the header
            hdr['HIERARCH ESO QC CCF RV'] = self.rv
            hdr['HIERARCH ESO OCS OBJ Z EM'] = self.zem
            hdr['HIERARCH ESO DAS Z_EM'] = self.zem
            hdr['HIERARCH ESO PRO CATG'] = t

            # NB: Images are 2D, with just one pixel along y.
            #     This is required by current version of espda_coadd_spec.
            hdu1 = fits.PrimaryHDU([d], header=hdr)  
            hdul = fits.HDUList([hdu1])
            name = self.prefix+'_'+t+'.fits'
            hdul.writeto(name, overwrite=True, checksum=True)
            print "...saved %-17s" % t, "as", name+"."

    def create_wave(self, hdul):
        """ Create wavelength array from CRVAL1 and CDELT1 """
        
        start = hdul[0].header['CRVAL1']
        step = hdul[0].header['CDELT1']
        end = start+step*hdul[0].header['NAXIS1']
        wave = np.arange(start, end, step)
        try:
            wave = wave[:len(hdul[1].data)]
        except:
            wave = wave[:len(hdul[0].data)]
        return wave

    def xsh(self):
        """ Extract data in X-shooter MERGE1D format """
        
        hdul = fits.open(self.path_i)
        hdr = dc(hdul[0].header)
        self.wave = self.create_wave(hdul)
        self.flux = hdul[0].data
        self.err = hdul[1].data
        self.arm = self.path_i[-9:-5]  # UVB, VIS or NIR
        
        # Define ESPRESSO-like BINX and BINY
        try:  # UVB/VIS
            hdr['HIERARCH ESO DET BINX'] = hdr['HIERARCH ESO DET WIN1 BINX']
            hdr['HIERARCH ESO DET BINY'] = hdr['HIERARCH ESO DET WIN1 BINY']
        except:
            hdr['HIERARCH ESO DET BINX'] = 1
            hdr['HIERARCH ESO DET BINX'] = 2 

        self.convert(hdr)
            
    def uves(self):
        """ Extract data in UVES RED/ERRORBAR format """

        tag = ['RED', 'ERRORBAR']
        attr = ['flux', 'err']
        for t, a in zip(tag, attr):
            hdul = fits.open(self.path_i % t)
            hdr = dc(hdul[0].header)
            setattr(self, a, hdul[0].data)
            if a == 'flux':
                self.wave = self.create_wave(hdul)
                self.wave = self.wave*0.1                
        self.arm = self.path_i[-10:-5]  # BLUE, REDL or REDU
        self.convert(hdr)
        
def run(**kwargs):
    """ Convert spectra into DAS-suitable format """
    
    frames = np.array(ascii.read(kwargs['framelist'],
                                 format='no_header')['col1'])
    path_i = kwargs['root']+'from_pipe/'  # Path to the products of the pipeline
    path_o = kwargs['root']+'to_das/'     # Path to the reformatted frames

    # Create directory for products
    try:
        os.mkdir(path_o)
    except:
        print "Directory "+path_o+" already exists."

    # Loop over frames
    for f in frames:
        print "Processing", path_i+f+"..."

        # Initialize the format
        fmt = Format(path_i+f, kwargs['instr'], rv=kwargs['rv'],
                     zem=kwargs['zem'], path_o=path_o)  

        # Convert
        getattr(fmt, kwargs['instr'])()

def main():
    """ Read the CL parameters and run """ 

    p = argparse.ArgumentParser()
    p.add_argument('-i', '--instr', type=str, default='xsh',
                   help="Instrument ('xsh' or 'uves'; more will come).")
    p.add_argument('-d', '--root', type=str, default='./',
                   help="Root directory, where subdirectories input "
                   "(from_pipe/) and outputs (to_das/) are located.")
    p.add_argument('-l', '--framelist', type=str, default='frame_list.dat',
                   help="List of frames; must be an ascii file with a column "
                   "of entries. X-shooter entries should be filenames of "
                   "MERGE1D frames. UVES entries should be filenames of "
                   "RED_SCI_POINT frames, with 'RED' replaced by '%%s' (to "
                   "catch both RED and ERRORBAR frames).")
    p.add_argument('-r', '--rv', type=float, default=0.0,
                   help="Radial velocity of the target (km/s).")
    p.add_argument('-z', '--zem', type=float, default=0.0,
                   help="Redshift of the target.")
    args = vars(p.parse_args())
    run(**args)

if __name__ == '__main__':
    main()
