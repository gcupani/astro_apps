# FORMAT_DAS
# Format spectra to be processed with the ESPRESSO DAS
# v1.0 - 2018-11-29
# Guido Cupani - INAF-OATs

from astropy.io import fits
from copy import deepcopy as dc
import glob
import numpy as np
import os

### Scroll down for editable parameters... ###

class Format():
    """ Class for handling spectrum formats """

    def __init__(self, hdul,   # HDUList
                 instr,        # Instrument
                 rv=0.0,       # RV
                 zem=0.0,      # z
                 path=None,    # Path to reformatted frames
                 prefix=None   # Prefix for reformatted frames
                               # (Default: OBS.TARG.NAME_DATE)
    ):
        """ Initialize the format """
        
        self.hdul = fits.open(path_i+f)
        if prefix == None:
            prefix = self.hdul[0].header['ESO OBS TARG NAME']+'_'\
                     +self.hdul[0].header['DATE']
        if path != None:
            prefix = path+prefix
        self.prefix = prefix
        self.rv = rv
        self.zem = zem
        getattr(self, instr)()  # Load the format of the specific instrument

    def convert(self):
        """ Convert to DAS format and save the frames """

        # Tags for non-DRS input data
        tag = ['WAVE_MATRIX_FIBER', 'SPEC_FLUX_2D', 'SPEC_FLUXERR_2D']

        # Corresponding data
        data = [self.wave, self.flux, self.err]

        for t, d in zip(tag, data):
            hdr = dc(self.hdul[0].header)

            # Add required ESPRESSO keywords to the header
            hdr['HIERARCH ESO QC CCF RV'] = self.rv
            hdr['HIERARCH ESO OCS OBJ Z EM'] = self.zem
            hdr['HIERARCH ESO DAS Z_EM'] = self.zem
            hdr['HIERARCH ESO PRO CATG'] = t
            
            hdu1 = fits.PrimaryHDU([d], header=hdr)
            hdul = fits.HDUList([hdu1])
            name = self.prefix+'_'+t+'.fits'
            hdul.writeto(name, overwrite=True, checksum=True)
            print "...saved %-17s" % t, "as", name+"."

    def create_wave(self):
        """ Create wavelength array from CRVAL1 and CDELT1 """
        
        start = self.hdul[0].header['CRVAL1']
        step = self.hdul[0].header['CDELT1']
        end = start+step*self.hdul[0].header['NAXIS1']
        return np.arange(start, end, step)[:len(self.hdul[1].data)]

    def xsh(self):
        """ Extract data in X-shooter MERGE1D format """
        
        self.wave = self.create_wave()
        self.flux = hdul[0].data
        self.err = hdul[1].data

        # Define ESPRESSO-like BINX and BINY 
        self.hdul[0].header['HIERARCH ESO DET BINX'] = \
            self.hdul[0].header['HIERARCH ESO DET WIN1 BINX']
        self.hdul[0].header['HIERARCH ESO DET BINY'] = \
            self.hdul[0].header['HIERARCH ESO DET WIN1 BINY']

        
### Parameters to edit ###
instr = 'xsh'               # Instrument (xsh); others will be available soon 
root = '/data/espresso/cupani/analysis/A1689_XSH/'
path_i = root+'from_pipe/'  # Path to the products of X-shooter pipeline
path_o = root+'to_das/'     # Path to the reformatted frames 
frame = ['SCI_SLIT_FLUX_MERGE1D_UVB.fits']  # Can be a list
#path = ['SCI_SLIT_FLUX_ORDER1D_UVB.fits']  # Not working yet on order frames
rv = [0.0]                                  # Also a list, one RV per frame
zem = [0.0]                                 # Also a list, one z per frame
##########################

# Create directory for products
try:
    os.mkdir(path_o)
except:
    print "Directory "+path_o+" already exists."

# Loop over frames
for f, r, z in zip(frame, rv, zem):
    print "Processing", path_i+f+"..."
    hdul = fits.open(path_i+f)  # Open HDU list
    fmt = Format(hdul, instr, rv=r, zem=z, path=path_o)  # Initialize the format
    fmt.convert()  # Convert to DAS format
