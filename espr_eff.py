# ------------------------------------------------------------------------------
# ESPR_EFF
# Measure the efficiency of ESPRESSO from reduced standard stars
# v1.2 - 2018-12-06
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------
# Sample run:
# > python espr_eff.py -h  // Help
# > python ~/Devel/astro_apps/espr_eff.py -l=espr_eff_list.dat -c=/data/cupani/ESPRESSO/utils/ -s=1 -d=0   // Measure efficiency from COM4UT standards
# ------------------------------------------------------------------------------

import argparse
from astropy import units as u
from astropy.io import ascii, fits
from astropy.time import Time
from formats import ESOExt, ESOStd, EsprEff, EsprS1D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from plots import Plot21
import numpy as np
from scipy.signal import savgol_filter as sg

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
    
    figg = plt.figure(figsize=(7,9))
    gs = gridspec.GridSpec(3, 1)
    axg = []
    axg.append(figg.add_subplot(gs[0:2]))
    axg.append(figg.add_subplot(gs[2]))
    color = -1
    targ = ''
    for f in frames:
        print "Processing", f+"..."
    
        # Load observed spectrum
        spec = EsprS1D(f)

        # Convert to electons
        spec.adu_to_electron()
        
        # Load extinction
        ext = ESOExt(cal+'atmoexan.fits')

        # Correct observed spectrum for extinction
        la_silla_spec = np.interp(spec.wave, ext.wave, ext.la_silla)
        spec.flux = spec.flux * 10 ** (0.4*la_silla_spec*(spec.airm-1))

        # Load pipeline efficiency
        try:
            eff = EsprEff(f[:-10]+'_0005.fits')
            pipe = True
        except:
            pipe = False
    
        # Load catalogue spectrum
        std = ESOStd(cal+'f'+spec.targ.lower()+'.dat', area=spec.area,
                     expt=spec.expt)
    
        # Bin spectra and sum counts
        bins = np.arange(378.0, 788.0, 1.6) * u.nm
        bin_spec = binspec(spec.wave, spec.flux.value, bins)
        bin_std = binspec(std.wave, std.flux, bins)

        # Smooth the efficiency curve
        eff_raw = bin_spec/bin_std  
        eff_raw[np.isnan(eff_raw)] = np.median(eff_raw[~np.isnan(eff_raw)])
        eff_smooth = sg(eff_raw, 75, 3)
        
        # Individual plot
        figi = plt.figure(figsize=(7,12))        
        figi.suptitle(f.split('/')[1][:-10]+', '+spec.targ+', '+spec.mode)
        axi = []
        axi.append(figi.add_subplot(211))
        axi.append(figi.add_subplot(212))
        axi[0].semilogy(bins, bin_spec, c='C0', label="ESPRESSO")
        axi[0].semilogy(bins, bin_std, c='C1', label="catalogue")
        axi[0].set_xlabel("Wavelength (nm)")
        axi[0].set_ylabel("Photons")
        axi[0].legend()
        if pipe:
            axi[1].plot(eff.wave, eff.eff, c='black', linestyle='--',
                       label="DRS")
        axi[1].plot(bins, eff_smooth, c='C2', label="measured")
        axi[1].set_xlabel("Wavelength (nm)")
        axi[1].set_ylabel("Efficiency")    
        axi[1].legend()
                
        if plotf == 'all':
            plt.draw()
        if save:
            figi.savefig(filename=f[:-10]+'_eff.pdf', format='pdf')

            # Save results
            hdu0 = fits.PrimaryHDU(header=spec.hdul[0].header)
            col0 = fits.Column(name='wave', format='D', array=bins)
            col1 = fits.Column(name='eff', format='D', array=eff_smooth)
            hdu1 = fits.BinTableHDU.from_columns([col0, col1])
            hdul = fits.HDUList([hdu0, hdu1])
            hdul.writeto(f[:-10]+'_eff.fits', overwrite=True)
            print "...saved plot/efficiencies as", f[:-10]+'_eff.pdf/fits'+"."
        if plotf != 'all':
            plt.close()

        # Compute averages
        if spec.targ != targ:
            avecomp = f != frames[0]
            if avecomp:
                eff_save = eff_stack
                avecolor = color
                avetarg = targ
                avetime = str(Time(np.average(midtime_arr), format='mjd').isot)
                aveiq = "%3.2f" % np.average(iq_arr)
            eff_stack = eff_smooth
            midtime_arr = [spec.midtime.mjd]
            binx_arr = [spec.binx]
            iq_arr = [spec.dimm]
            targ = spec.targ
            color += 1
        else:
            eff_stack = np.vstack((eff_stack, eff_smooth))
            midtime_arr = np.append(midtime_arr, spec.midtime.mjd)
            binx_arr = np.append(binx_arr, spec.binx)
            iq_arr = np.append(iq_arr, spec.dimm)
            avecomp = f == frames[-1]
            if avecomp:
                eff_save = eff_stack
                avecolor = color
                avetarg = targ
                avetime = str(Time(np.average(midtime_arr), format='mjd').isot)
                aveiq = "%3.2f" % np.average(iq_arr)

        # Global plot
        if avecomp:
            label = avetarg+', '+spec.date[:10]+', '+avetime+', IQ: '+aveiq
            eff_ave = np.average(eff_save, axis=0)
            eff_std = np.std(eff_save, axis=0)
            if 'eff_ref' not in locals():
                eff_ref = eff_ave
            axg[0].plot(bins, eff_ave, c='C'+str(avecolor), label=label)
            axg[0].fill_between(bins.value, eff_ave-eff_std,
                                eff_ave+eff_std,
                                facecolor='C'+str(avecolor), alpha=0.3)
            axg[1].plot(bins, eff_ave/eff_ref, c='C'+str(avecolor), label=label)
            axg[1].fill_between(bins.value, (eff_ave-eff_std)/eff_ref,
                                (eff_ave+eff_std)/eff_ref,
                                facecolor='C'+str(avecolor), alpha=0.3)
        axg[0].plot(bins, eff_smooth, c='C'+str(color), linestyle=':')
        
    if len(np.unique(binx_arr)) == 1:
        figg.suptitle(str(spec.binx)+'x'+str(spec.biny))
    axg[0].set_ylabel("Efficiency")
    axg[0].legend()
    axg[1].set_xlabel("Wavelength (nm)")
    axg[1].set_ylabel("Normalized to reference")
        
    if plotf != 'no':
        plt.show()
    if save:
        figg.savefig(kwargs['framelist'][:-4]+'.pdf', format='pdf')
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
