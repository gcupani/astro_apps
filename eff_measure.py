# ------------------------------------------------------------------------------
# EFF_MEASURE
# Measure the efficiency of ESPRESSO from reduced standard stars
# v1.0 - 2018-12-05
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------
# Sample run:
# > python eff_measure.py -h         // Help
# ------------------------------------------------------------------------------

import argparse
from astropy import units as u
from astropy.io import ascii, fits
from formats import ESOExt, ESOStd, EsprEff, EsprS1D
import matplotlib.pyplot as plt
from plots import Plot21
import numpy as np

def binspec(wave, flux, bins=np.arange(360.0, 800.0, 16.0)*u.nm,
            binsize=8.0*u.nm):
    bin_spec = []
    for b in bins:
        where = np.where(np.logical_and(wave > b-binsize/2, wave < b+binsize/2))
        bin_spec.append(np.median(flux[where])*len(flux[where]))
    return np.array(bin_spec)

def run(**kwargs):

    # Load parameters
    frames = np.array(ascii.read(kwargs['framelist'],
                                 format='no_header')['col1'])
    cal = kwargs['cal']
    show = bool(kwargs['show'])
    save = bool(kwargs['save'])
    
    for f in frames:
        print "Processing", f+"..."
    
        # Load observed spectrum
        spec = EsprS1D(f)

        # Load pipeline efficiency
        try:
            eff = EsprEff(f[:-10]+'_0006.fits')
            eff_airm = eff.interp/spec.airm
            pipe = True
        except:
            pipe = False
    
        # Load catalogue spectrum
        std = ESOStd(cal+'f'+spec.targ.lower()+'.dat', area=spec.area,
                     expt=spec.expt)
    
        # Load extinction
        ext = ESOExt(cal+'atmoexan.fits')

        # Correct catalogue spectrum for extinction
        la_silla_interp = np.interp(std.wave, ext.wave, ext.la_silla)
        std_ext = std.flux * pow(10, -0.4*la_silla_interp*(spec.airm-1))
    
        # Bin spectra and sum counts
        bins = np.arange(360.0, 800.0, 16.0) * u.nm
        bin_flux = binspec(spec.wave, spec.flux, bins)
        bin_std_ext = binspec(std.wave, std_ext, bins)

        # Plot results
        fig = plt.figure(figsize=(7,12))        
        fig.suptitle(f.split('/')[1][:-10]+', '+spec.targ+', '+spec.mode)
        ax = []
        ax.append(fig.add_subplot(211))
        ax.append(fig.add_subplot(212))
        ax[0].semilogy(bins, bin_flux, c='C0', label="ESPRESSO")
        ax[0].semilogy(spec.wave, spec.flux, c='black', label="ESPRESSO")        
        ax[0].semilogy(bins, bin_std_ext, c='C2', label="catalogue")
        ax[0].set_xlabel("Wavelength (nm)")
        ax[0].set_ylabel("Photons")
        ax[0].legend()
        ax[1].plot(bins, bin_flux/bin_std_ext, c='C2', label="measured")
        if pipe:
            ax[1].plot(eff.wave, eff.interp, c='black', linestyle='--',
                       label="DRS")
            ax[1].plot(eff.wave, eff_airm, c='black', linestyle=':',
                            label="DRS/airmass")
        ax[1].set_xlabel("Wavelength (nm)")
        ax[1].set_xlabel("Efficiency")    
        ax[1].legend()
        if show:
            plt.show()
            
        if save:
            #plt.savefig(filename=f[:-10]+'_eff.pdf', format='pdf')

            # Save results
            hdu0 = fits.PrimaryHDU(header=spec.hdul[0].header)
            col0 = fits.Column(name='wave', format='D', array=bins)
            col2 = fits.Column(name='ph_espr', format='D', array=bin_flux)
            col1 = fits.Column(name='ph_cat', format='D', array=bin_std_ext)
            col3 = fits.Column(name='eff', format='D',
                               array=bin_flux/bin_std_ext)
            hdu1 = fits.BinTableHDU.from_columns([col0, col1, col2, col3])
            hdul = fits.HDUList([hdu0, hdu1])
            hdul.writeto(f[:-10]+'_eff.fits', overwrite=True)
            print "...saved plot/efficiencies as", f[:-10]+'_eff.pdf/fits'+"."

        
def main():
    """ Read the CL parameters and run """

    p = argparse.ArgumentParser()
    p.add_argument('-l', '--framelist', type=str, default='frame_list.dat',
                   help="List of frames; must be an ascii file with a column "
                   "of entries. ")
    #p.add_argument('-r', '--red', type=str,
    #                    help="Reduced star spectrum.")
    p.add_argument('-c', '--cal', type=str,
                   default='/data/cupani/ESPRESSO/utils/',
                   help="Path to calibration directory, including catalogue "
                   "spectra and atmoexan.fits.")
    p.add_argument('-d', '--show', type=int, default=1,
                   help="Show plot with measured efficiencies.")
    p.add_argument('-s', '--save', type=int, default=1,
                   help="Save plot and table with measured efficiencies.")

    args = vars(p.parse_args())
    run(**args)
    
if __name__ == '__main__':
    main()
