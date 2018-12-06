# ------------------------------------------------------------------------------
# ESPR_EFF
# Measure the efficiency of ESPRESSO from reduced standard stars
# v1.1 - 2018-12-06
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------
# Sample run:
# > python espr_eff.py -h  // Help
# > python ~/Devel/astro_apps/espr_eff.py -l=espr_eff_list.dat -c=/data/cupani/ESPRESSO/utils/ -s=1 -d=0   // Measure efficiency from COM4UT standards
# ------------------------------------------------------------------------------

import argparse
from astropy import units as u
from astropy.io import ascii, fits
from formats import ESOExt, ESOStd, EsprEff, EsprS1D
import matplotlib.pyplot as plt
from plots import Plot21
import numpy as np

def binspec(wave, flux, bins=np.arange(360.0, 800.0, 1.6)*u.nm,
            binsize=1.6*u.nm):
    bin_spec = []
    for b in bins:
        binw = np.where(np.logical_and(wave > b-binsize/2, wave < b+binsize/2))
        fluxw = np.where(flux[binw] > 0)
        
        flux_bin = flux[binw]
        flux_nonzero = flux[binw][fluxw]
        bin_spec.append(np.sum(flux_bin)*len(flux_bin)/len(flux_nonzero))
        #bin_spec.append(np.median(flux[binw])*len(flux[binw]))
    return np.array(bin_spec)

def run(**kwargs):

    # Load parameters
    frames = np.array(ascii.read(kwargs['framelist'],
                                 format='no_header')['col1'])
    cal = kwargs['cal']
    plotf = kwargs['plot']
    save = bool(kwargs['save'])
    
    figg = plt.figure(figsize=(7,7))
    color = -1
    targ = ''
    for f in frames:
        print "Processing", f+"..."
    
        # Load observed spectrum
        spec = EsprS1D(f)

        # Load pipeline efficiency
        try:
            eff = EsprEff(f[:-10]+'_0005.fits')
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
        bins = np.arange(360.0, 800.0, 1.6) * u.nm
        bin_flux = binspec(spec.wave, spec.flux, bins)
        bin_std_ext = binspec(std.wave, std_ext, bins)

        # Individual plot
        figi = plt.figure(figsize=(7,12))        
        figi.suptitle(f.split('/')[1][:-10]+', '+spec.targ+', '+spec.mode)
        axi = []
        axi.append(figi.add_subplot(211))
        axi.append(figi.add_subplot(212))
        axg = figg.add_subplot(111)
        axi[0].semilogy(bins, bin_flux, c='C0', label="ESPRESSO")
        axi[0].semilogy(bins, bin_std_ext, c='C1', label="catalogue")
        axi[0].set_xlabel("Wavelength (nm)")
        axi[0].set_ylabel("Photons")
        axi[0].legend()
        if pipe:
            axi[1].plot(eff.wave, eff.eff, c='black', linestyle='--',
                       label="DRS")
        axi[1].plot(bins, bin_flux/bin_std_ext, c='C2', label="measured")
        axi[1].set_xlabel("Wavelength (nm)")
        axi[1].set_ylabel("Efficiency")    
        axi[1].legend()
                
        if plotf == 'all':
            plt.draw()
        if save:
            plt.savefig(filename=f[:-10]+'_eff.pdf', format='pdf')

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
        if plotf != 'all':
            plt.close()

        # Global plot
        if spec.targ != targ:
            targ = spec.targ
            color += 1
            axg.plot(bins, bin_flux/bin_std_ext, c='C'+str(color), label=targ)
        else:
            axg.plot(bins, bin_flux/bin_std_ext, c='C'+str(color))
        axg.set_xlabel("Wavelength (nm)")
        axg.set_ylabel("Efficiency")
        axg.legend()
        
    if plotf != 'no':
        plt.show()
    if save:
        plt.savefig(kwargs['framelist'][:-4]+'_eff.pdf', format='pdf')
    plt.close()
        
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
    p.add_argument('-p', '--plot', type=str, default='all',
                   help="Show plots (all: individual plots and global plot; "
                   "glob: only global plot; no: none).")
    p.add_argument('-s', '--save', type=int, default=1,
                   help="Save plot and table with measured efficiencies.")

    args = vars(p.parse_args())
    run(**args)
    
if __name__ == '__main__':
    main()
