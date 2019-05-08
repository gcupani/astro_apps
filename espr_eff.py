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
from formats import AAEff, ESOExt, ESOStd, EsprEff, EsprS1D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.signal import savgol_filter as sg
from toolbox import prof_ee

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

def seeing_v_wave(ratio, norm_wave):
    return ratio*norm_wave.value**-0.2

def extract(**kwargs):
    """ Extract the efficiency from a list of S1D_ frames """
    
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
        #spec.flux = spec.flux * 10 ** (0.4*la_silla_spec*(spec.airm-1))
        spec.flux = spec.flux * 10 ** (0.4*la_silla_spec*(spec.airm))

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
            label = avetarg+', '+avetime[:10]+', '+avetime+', IQ: '+aveiq
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

            # Save averages
            hdu0 = fits.PrimaryHDU(header=spec.hdul[0].header)
            hdu0.header['HIERARCH ESO AVE IQ'] = aveiq
            hdu0.header['HIERARCH ESO OBS TARG NAME'] = avetarg
            hdu0.header['HIERARCH ESO MIDTIME'] = avetime
            col0 = fits.Column(name='wave', format='D', array=bins)
            col1 = fits.Column(name='eff_ave', format='D', array=eff_ave)
            col2 = fits.Column(name='eff_std', format='D', array=eff_std)
            hdu1 = fits.BinTableHDU.from_columns([col0, col1, col2])
            hdul = fits.HDUList([hdu0, hdu1])
            hdul.writeto(kwargs['framelist'][:-4]+'_'+avetime[:10]+'.fits',
                         overwrite=True)
            print "...saved average efficiencies as", \
                kwargs['framelist'][:-4]+'_'+avetime[:10]+'_eff.fits'+"."
            
        #axg[0].plot(bins, eff_smooth, c='C'+str(color), linestyle=':')
        
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

def model(**kwargs):
    """ Model the efficiency from a set of previously produced 
    *_list_YYYY-MM-DD.fits frames """

    frames = np.array(ascii.read(kwargs['framelist'],
                                 format='no_header')['col1'])
    save = bool(kwargs['save'])
    func = kwargs['func']
    alpha = kwargs['alpha']
    
    fig = plt.figure(figsize=(18,10))
    gs = gridspec.GridSpec(3, 2)
    ax = []
    ax.append(fig.add_subplot(gs[0:3,0], projection='3d'))
    ax.append(fig.add_subplot(gs[0,1]))
    ax.append(fig.add_subplot(gs[1,1]))
    ax.append(fig.add_subplot(gs[2,1]))
    
    
    # Reference wavelength for DIMM seeing measurement
    dimm_wave = 500
    ax[3].axvline(dimm_wave,  c='black', linestyle=':', #linewidth=1,
                  label="Reference wavelength of DIMM seeing measurements")

    # Nominal radius of the fiber
    rad_nom = 0.5

    color = -1
    for f in frames:
        print "Processing", f+"..."
        color += 1

        # Load efficiency spectrum
        spec = AAEff(f)
        #spec.midtime = spec.hdr['ESO MIDTIME']

        # Compute the encircled energy profile of a Moffat function with
        # FWHM corresponding to the measured IQ
        fwhm = float(spec.hdr['ESO AVE IQ'])
        p = prof_ee(fwhm, func=func, alpha=alpha)
        #fl = 1-p.ee
        fl = p.ee        
        if 'fl_ref' not in locals():
            fl_ref = fl

        # Define reference efficiency
        if 'eff_ref' not in locals():
            eff_ref = spec.eff

        else:
            # Normalized efficiency at the DIMM reference wavelength
            ratio_est = np.interp(dimm_wave, spec.wave, spec.eff)\
                        /np.interp(dimm_wave, spec.wave, eff_ref)
            ratio_estu = np.interp(dimm_wave, spec.wave, spec.eff+spec.eff_std)\
                         /np.interp(dimm_wave, spec.wave, eff_ref)
            ratio_estd = np.interp(dimm_wave, spec.wave, spec.eff-spec.eff_std)\
                         /np.interp(dimm_wave, spec.wave, eff_ref)
            ests = np.argsort(fl/fl_ref)
            rad_est = np.interp(ratio_est, fl[ests]/fl_ref[ests], p.rad[ests])
            rad_estu = np.interp(ratio_estu, fl[ests]/fl_ref[ests], p.rad[ests])
            rad_estd = np.interp(ratio_estd, fl[ests]/fl_ref[ests], p.rad[ests])
            noms = np.argsort(p.rad)
            ratio_nom = np.interp(rad_nom, p.rad[noms], fl[noms]/fl_ref[noms])

        label = spec.targ+', IQ: '+str(fwhm)
        if func == 'gauss':
            ax[0].set_title("Gaussian profiles")
        if func == 'moffat':
            ax[0].set_title("Moffat profiles, alpha: %2.1f" % alpha)
        ax[0].plot_surface(p.x, p.y, p.z, color='C'+str(color))
        ax[0].set_xlabel('x (arcsec)')
        ax[0].set_ylabel('y (arcsec)')
        ax[0].set_zlabel('Normalized amplitude')
        ax[0].view_init(30, -60)
        ax[1].plot(p.rad, fl, label=label)
        ax[1].set_xlabel("Aperture radius (arcsec)")
        #ax[1].set_ylabel("Fiber losses")
        ax[1].set_ylabel("Encircled energy")
        ax[2].plot(p.rad, fl/fl_ref)
        ax[2].set_xlabel("Aperture radius (arcsec)")
        #ax[2].set_ylabel("Fiber losses normalized to reference")
        ax[2].set_ylabel("Encircled energy norm. to reference")
        ax[3].plot(spec.wave, spec.eff/eff_ref, c='C'+str(color))
        ax[3].fill_between(spec.wave.value, (spec.eff-spec.eff_std)/eff_ref,
                           (spec.eff+spec.eff_std)/eff_ref,
                           facecolor='C'+str(color), alpha=0.3)
        if 'ratio_est' in locals():
            ax[2].axvline(rad_nom, c='green', linestyle=':', #linewidth=1,
                          label="Nominal fiber radius")
            ax[2].axvline(rad_est, c='red', linestyle=':',
                          #linewidth=1,
                          label=r"Effective fiber radius: "
                          "$%4.3f^{+%4.3f}_{%4.3f}$ arcsec" \
                          % (rad_est, rad_estd-rad_est, rad_estu-rad_est))
            ax[2].axvspan(rad_estu, rad_estd, color='red', alpha=0.1)
            ax[3].plot(spec.wave, seeing_v_wave(ratio_nom, spec.wave/dimm_wave),
                       c='green', linestyle=':', #linewidth=1,
                       label="Predicted efficiency for a nominal fiber radius")
            ax[3].plot(spec.wave, seeing_v_wave(ratio_est, spec.wave/dimm_wave),
                       c='red', linestyle=':', #linewidth=1,
                       label="Predicted efficiency for the effective fiber "
                       "radius")
            ax[3].fill_between(spec.wave.value,
                               seeing_v_wave(ratio_estd, spec.wave/dimm_wave),
                               seeing_v_wave(ratio_estu, spec.wave/dimm_wave),
                               facecolor='red', alpha=0.1)
        ax[3].set_xlabel("Wavelength")
        ax[3].set_ylabel("Efficiency normalized to reference") 
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    fig.tight_layout()
    plt.show()
    if save:
        print func
        if func == 'gauss':
            fig.savefig(kwargs['framelist'][:-4]+'_'+func+'.pdf', format='pdf')
        if func == 'moffat':
            fig.savefig(kwargs['framelist'][:-4]+'_'+func+'_%2.1f.pdf' % alpha,
                        format='pdf')

        
def main():
    """ Read the CL parameters and run """

    p = argparse.ArgumentParser()
    p.add_argument('-m', '--mode', type=str, default='extract',
                   help="Mode: 'extract' or 'model'.")
    p.add_argument('-l', '--framelist', type=str, default='frame_list.dat',
                   help="List of frames; must be an ascii file with a column "
                   "of entries. If mode is 'extract', the frames must be S1D "
                   "frames from the DRS; if mode is 'model', they must be "
                   "*_list_YYYY-MM-DD.fits frames produced in a previous "
                   "execution of espr_eff.py.")
    #p.add_argument('-r', '--red', type=str,
    #                    help="Reduced star spectrum.")
    p.add_argument('-p', '--plot', type=str, default='all',
                   help="Show plots (all: individual plots and global plot; "
                   "glob: only global plot; no: none).")
    p.add_argument('-s', '--save', type=int, default=1,
                   help="Save plot and table with measured efficiencies.")
    p.add_argument('-c', '--cal', type=str,
                   default='/data/cupani/ESPRESSO/utils/',
                   help="Path to calibration directory, including catalogue "
                   "spectra and atmoexan.fits (only in 'extract' mode).")
    p.add_argument('-f', '--func', type=str, default='moffat',
                   help="Model function ('gauss', 'moffat'; only in 'model' "
                   "mode).")
    p.add_argument('-a', '--alpha', type=float, default=3.0,
                   help="Alpha parameter for Moffat profile (only in 'model' "
                   "mode, with 'moffat' function).")

    args = vars(p.parse_args())

    if args['mode'] == 'extract':
        extract(**args)
    elif args['mode'] == 'model':
        model(**args)
    
if __name__ == '__main__':
    main()
