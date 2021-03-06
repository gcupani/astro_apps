# ------------------------------------------------------------------------------
# FORMAT_DAS
# Format spectra to be processed with the ESPRESSO DAS
# v3.0 - 2019-04-30
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------
# Sample run:
# > python format_das.py -h         // Help
# > python format_das.py -i='uves'  // Format UVES spectra
# ------------------------------------------------------------------------------

import argparse
from astropy import units as u
from astropy.constants import c
from astropy.io import ascii, fits
from astropy.time import Time, TimeDelta
from copy import deepcopy as dc
import glob
import numpy as np
import os
from toolbox import air_to_vacuum, earth_to_bary

### Scroll down for editable parameters... ###

class Format():
    """ Class for handling spectrum formats """

    def __init__(self, path_i,  # Path
                 rv=0.0,        # RV
                 zem=0.0,       # z
                 path_o=None,   # Path to reformatted frames
                 prefix=None,   # Prefix for reformatted frames
                                # (Default: OBS.TARG.NAME_DATE-OBS_ARM)
                 baryvac_f=True # Flag for barycentric-vacuum correction
    ):
        """ Initialize the format """

        self.path_i = path_i
        self.path_o = path_o
        self.prefix = prefix
        self.rv = rv
        self.zem = zem
        self.baryvac_f = baryvac_f

    def convert(self, hdr):
        """ Convert to DAS format and save the frames """

        # Adjust prefix
        if self.prefix == None:
            self.prefix = hdr['ESO OBS TARG NAME']+'_'+hdr['DATE-OBS']+self.arm
        else:
            self.prefix = self.prefix+self.arm
        if self.path_o != None:
            self.prefix = self.path_o+self.prefix

        # Tags for non-DRS input data
        tag = ['WAVE_MATRIX_FIBER', 'SPEC_FLUX_2D', 'SPEC_FLUXERR_2D']

        # Corresponding data
        data = [self.wave, self.flux, self.err]

        # Old style: three separate frames
        """
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
            print("...saved %-17s" % t, "as", name+".")
        """

        # New style: single multi-extension frame
        hdr['HIERARCH ESO QC CCF RV'] = self.rv
        hdr['HIERARCH ESO OCS OBJ Z EM'] = self.zem
        hdr['HIERARCH ESO DAS Z_EM'] = self.zem
        hdr['HIERARCH ESO PRO TYPE'] = 'REDUCED'
        hdr['HIERARCH ESO PRO CATG'] = 'S2D_A'
        hdr['HIERARCH ESO PRO REC1 PIPE ID'] = 'espdr'
        try:
            # NIR frames from xshooter_reduce have this faulty keyword
            del hdr['HIERARCH ESO DET CHIP PXSPACE']
        except:
            pass

        #print(hdr)

        hdu0 = fits.PrimaryHDU([], header=hdr)
        hdu1 = fits.ImageHDU([self.flux], name='SCIDATA')
        hdu2 = fits.ImageHDU([self.err], name='ERRDATA')
        hdu3 = fits.ImageHDU([np.zeros(self.flux.shape)], name='QUALDATA')
        hdu4 = fits.ImageHDU([self.wave], name='WAVEDATA_VAC_BARY')
        hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4])
        name = self.prefix+'.fits'
        hdul.writeto(name, overwrite=True, checksum=True)
        print("...saved S2D multi-extension frame as", name+".")

    def create_wave(self, hdul, log=False, A_to_nm=False):
        """ Create wavelength array from CRVAL1 and CDELT1 """

        start = hdul[0].header['CRVAL1']
        step = hdul[0].header['CDELT1']
        end = start+step*hdul[0].header['NAXIS1']
        wave = np.arange(start, end, step)
        try:
            wave = wave[:len(hdul[1].data)]
        except:
            wave = wave[:len(hdul[0].data)]
        if log:
            wave = 10**wave
        if A_to_nm:
            wave = wave*0.1
        return wave

    def create_wave_2d(self, hdul):
        """ Create wavelength array from CRVAL1 and CDELT1 """

        wave = np.empty(np.shape(hdul[0].data))
        naxis1 = hdul[0].header['NAXIS1']
        naxis2 = hdul[0].header['NAXIS2']
        step = hdul[0].header['CDELT1']
        for i in range(naxis2):
            start = hdul[0].header['WSTART'+str(i+1)]
            end = start+naxis1*step
            wave[i,:] = np.arange(start,end,step)[:naxis1]
        return wave

    def xsh(self):
        """ Extract data in X-shooter pipeline MERGE1D format """

        hdul = fits.open(self.path_i)
        hdr = dc(hdul[0].header)
        self.wave = self.create_wave(hdul)
        self.flux = hdul[0].data
        self.err = hdul[1].data

        # 'find' gives -1 for all but one arm tags; the sum of the three calls
        # is always 2 positions left from the position where the arm tag
        # appears; +1 is meant to take also the underscore before the arm tag
        arm_start = self.path_i.find('UVB')+self.path_i.find('VIS')\
                    +self.path_i.find('NIR')+1
        self.arm = self.path_i[arm_start:arm_start+4]  # UVB, VIS or NIR

        # Define ESPRESSO-like BINX and BINY
        try:  # UVB/VIS
            hdr['HIERARCH ESO DET BINX'] = hdr['HIERARCH ESO DET WIN1 BINX']
            hdr['HIERARCH ESO DET BINY'] = hdr['HIERARCH ESO DET WIN1 BINY']
        except:
            hdr['HIERARCH ESO DET BINX'] = 1
            hdr['HIERARCH ESO DET BINX'] = 2

        if self.baryvac_f:
            air_to_vacuum(self)
            self.hdr = hdr
            self.expt = self.hdr['EXPTIME'] * u.s
            self.date = self.hdr['DATE-OBS']
            start = Time(self.date, format='isot', scale='utc')
            mid = TimeDelta(self.expt / 2.0, format='sec')
            self.midtime = start + mid
            #earth_to_bary(self)
            barycor_vel = self.hdr['ESO QC VRAD BARYCOR']*u.km/u.s
            self.wave = self.wave * (1 + barycor_vel/c).value
        self.convert(hdr)

    def xsh_reduce(self):
        """ Extract data in xshooter_reduce ARM/ARMe format """

        hdul = fits.open(self.path_i)
        hdul_e = fits.open(self.path_i[:-5]+'e.fits')
        hdr = dc(hdul[0].header)
        self.wave = self.create_wave(hdul, log=True)
        self.flux = hdul[0].data
        self.err = hdul_e[0].data

        # See 'xsh' above for explanation
        arm_start = self.path_i.find('UVB')+self.path_i.find('VIS')\
                    +self.path_i.find('NIR')+1
        self.arm = self.path_i[arm_start:arm_start+4]  # UVB, VIS or NIR

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
                #self.wave = self.wave*0.1
        self.arm = self.path_i[-10:-5]  # BLUE, REDL or REDU
        self.convert(hdr)

    def uves_2d(self):
        """ Extract data in UVES 2D WCALIB/ERRORBAR_WCALIB format """

        tag = ['WCALIB', 'ERRORBAR_WCALIB']
        attr = ['flux', 'err']
        for t, a in zip(tag, attr):
            hdul = fits.open(self.path_i % t)
            hdr = dc(hdul[0].header)
            setattr(self, a, hdul[0].data[:, 150:-150])
            if a == 'flux':
                self.wave = self.create_wave_2d(hdul)
                #self.wave = self.wave*0.1
                self.wave = self.wave[:,150:-150]
        try:
            wlen = hdr['ESO INS GRAT1 WLEN']
        except:
            wlen = hdr['ESO INS GRAT2 WLEN']
        self.arm = self.path_i[-10:-5]+'_'+str(wlen)  # BLUE, REDL or REDU
        if self.baryvac_f:
            air_to_vacuum(self)
            self.hdr = hdr
            self.expt = self.hdr['EXPTIME'] * u.s
            self.date = self.hdr['DATE-OBS']
            start = Time(self.date, format='isot', scale='utc')
            mid = TimeDelta(self.expt / 2.0, format='sec')
            self.midtime = start + mid
            #earth_to_bary(self)
            barycor_vel = self.hdr['ESO QC VRAD BARYCOR']*u.km/u.s
            self.wave = self.wave * (1 + barycor_vel/c).value
        self.convert(hdr)


def run(**kwargs):
    """ Convert spectra into DAS-suitable format """

    frames = np.array(ascii.read(kwargs['framelist'],
                                 format='no_header')['col1'])
    path_o = kwargs['outdir']
    prefix = kwargs['prefix']
    if prefix == 'None':
        prefix = None

    # Create directory for products
    try:
        os.mkdir(path_o)
    except:
        print("Directory "+path_o+" already exists.")

    # Loop over frames
    for path_i in frames:
        print("Processing", path_i+"...")

        # Initialize the format
        fmt = Format(path_i, rv=kwargs['rv'], zem=kwargs['zem'],
                     path_o=path_o, prefix=prefix, baryvac_f=kwargs['baryvac'])

        # Convert
        getattr(fmt, kwargs['instr'])()

def main():
    """ Read the CL parameters and run """

    p = argparse.ArgumentParser()
    p.add_argument('-i', '--instr', type=str, default='xsh',
                   help="Instrument ('xsh' or 'uves'; more will come).")
    p.add_argument('-l', '--framelist', type=str, default='frame_list.dat',
                   help="List of frames; must be an ascii file with a column "
                   "of entries. X-shooter entries should be filenames of "
                   "MERGE1D frames. UVES entries should be filenames of "
                   "RED_SCI_POINT frames, with 'RED' replaced by '%%s' (to "
                   "catch both RED and ERRORBAR frames). UVES 2D entries "
                   "should be filenames of WCALIB_FF_SCI_POINT frames, "
                   "with 'WCALIB' replaced by '%%s' (to catch both WCALIB and "
                   "WCALIB_ERRORBAR frames).")
    p.add_argument('-o', '--outdir', type=str, default='./',
                   help="Output directory.")
    p.add_argument('-p', '--prefix', type=str, default='None',
                   help="Prefix of the output file(s).")
    p.add_argument('-r', '--rv', type=float, default=0.0,
                   help="Radial velocity of the target (km/s).")
    p.add_argument('-z', '--zem', type=float, default=0.0,
                   help="Redshift of the target.")
    p.add_argument('-b', '--baryvac', type=int, default=1,
                   help="Flag for barycentric-vacuum correction.")
    args = vars(p.parse_args())
    run(**args)

if __name__ == '__main__':
    main()
