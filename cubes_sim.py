# ------------------------------------------------------------------------------
# CUBES_SIM
# Simulate the spectral format of CUBES and estimate the SNR for an input spectrum
# v1.0 - 2019-09-17
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------
# Sample run:
# > python cubes_sim.py J1124-1705.fits 17 22.5 0.7 0.25 10.0 3600 cubes_sim.png
# ------------------------------------------------------------------------------

from cubes_sim_config import *
from astropy import units as au
from astropy.io import ascii, fits
from astropy.modeling.fitting import LevMarLSQFitter as lm
from astropy.modeling.functional_models import Gaussian1D, Gaussian2D, Moffat2D
import matplotlib
#matplotlib.use('agg')
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import CubicSpline as cspline #interp1d
from scipy.interpolate import UnivariateSpline as uspline
from scipy.ndimage import gaussian_filter, interpolation
from scipy.special import expit
import sys

class CCD(object):

    def __init__(self, psf, spec, xsize=ccd_xsize, ysize=ccd_ysize,
                 xbin=ccd_xbin, ybin=ccd_ybin,
                 pix_xsize=pix_xsize, pix_ysize=pix_ysize,
                 spat_scale=spat_scale, slice_n=slice_n, func=extr_func):
        self.psf = psf
        self.spec = spec
        self.xsize = xsize/xbin
        self.ysize = ysize/ybin
        self.xbin = xbin
        self.ybin = ybin
        self.npix = xbin*ybin
        self.pix_xsize = pix_xsize*xbin
        self.pix_ysize = pix_ysize*ybin
        self.spat_scale = spat_scale

        self.signal = np.zeros((int(self.ysize.value), int(self.xsize.value)))
        self.func = func

    def add_arms(self, n=3):
        s = int(self.xsize.value/(n*3-1))
        self.xcens = np.arange(s, self.xsize.value, 3*s)
        if n == 3:
            self.wmins = [305, 328, 355] * au.nm
            self.wmaxs = [335, 361, 390] * au.nm
            self.wmins_d = [300, 331.5, 358] * au.nm  # Dichroich transition wavelengths
            self.wmaxs_d = [331.5, 358, 400] * au.nm

        self.mod_init = []
        self.sl_cen = []
        for i, (x, m, M, m_d, M_d) in enumerate(zip(
            self.xcens, self.wmins, self.wmaxs, self.wmins_d, self.wmaxs_d)):
            self.arm_counter = i
            self.arm_range = np.logical_and(self.spec.wave.value>m.value,
                                            self.spec.wave.value<M.value)
            self.arm_wave = self.spec.wave[self.arm_range].value
            self.arm_targ = self.spec.targ_conv[self.arm_range].value
            xlength = int(slice_length/self.spat_scale/self.pix_xsize)
            self.sl_hlength = xlength // 2
            self.psf_xlength = int(np.ceil(self.psf.seeing/self.spat_scale
                                           /self.pix_xsize))
            xspan = xlength + int(slice_gap.value/self.xbin)
            xshift = (slice_n*xspan+xlength)//2
            self.add_slices(int(x), xshift, xspan, self.psf_xlength,
                            wmin=m.value, wmax=M.value, wmin_d=m_d.value,
                            wmax_d=M_d.value)
            line, = self.spec.ax.plot(
                self.arm_wave, self.arm_targ \
                * self.tot_eff(self.arm_wave, m_d.value, M_d.value), c='C3')
            if i == 0:
                line.set_label('On detector')

        self.shot = np.sqrt(self.signal)
        self.dark = np.sqrt((ccd_dark*ccd_gain*self.npix*self.spec.phot.texp)\
                            .to(au.photon).value)
        self.ron = (ccd_ron*ccd_gain).value
        self.noise = np.sqrt(self.shot**2 + self.dark**2 + self.ron**2)
        print("Median shot noise: %2.3e ph/pix" % np.nanmedian(self.shot[self.shot>0]))
        print("Dark noise: %2.3e ph/pix" % self.dark)
        print("Readout noise: %2.3e ph/pix" % self.ron)
        print("Median total noise: %2.3e ph/pix" % np.nanmedian(self.noise[self.shot>0]))

        self.noise_rand = np.random.normal(0., np.abs(self.noise), self.signal.shape)

        self.image = np.round(self.signal + self.noise_rand)

    def add_slice(self, trace, xcen, wmin, wmax, wmin_d, wmax_d):

        if wmin is not None and wmax is not None:
            wave = self.wave_grid(wmin, wmax)
            norm = self.spec.norm_conv[np.logical_and(
                self.spec.wave.value>wmin, self.spec.wave.value<wmax)]
        else:
            norm = self.spec.norm_conv

        sl_trace = self.rebin(trace.value, self.sl_hlength*2)
        sl_norm = self.rebin(norm.value, self.ysize.value)

        # Efficiency
        if wmin_d is not None and wmax_d is not None:
            sl_norm = sl_norm * self.tot_eff(wave, wmin_d, wmax_d)

        signal = np.round(np.multiply.outer(sl_norm, sl_trace))
        self.signal[:,xcen-self.sl_hlength:xcen+self.sl_hlength] = signal

        #return sl_hlength, sl_trace, sl_norm, np.mean(signal)
        return sl_trace, sl_norm, np.mean(signal)


    def add_slices(self, xcen, xshift, xspan, psf_xlength, wmin, wmax, wmin_d,
                   wmax_d):

        xcens = range(xcen-xshift, xcen+xshift, xspan)
        for c, t in zip(xcens[1:], self.psf.traces):
            _, _, sl_msignal = self.add_slice(t, c, wmin, wmax, wmin_d, wmax_d)

            self.mod_init.append(
                Gaussian1D(amplitude=sl_msignal, mean=c, stddev=psf_xlength))
            self.sl_cen.append(c)


    def draw(self, fig):
        if fig is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = fig.axes[3]
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        image = np.zeros(self.image.shape)
        thres = np.infty
        image[self.image > thres] = thres
        image[self.image < thres] = self.image[self.image < thres]
        im = self.ax.imshow(image, vmin=0)
        self.ax.set_title('CCD')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.text(0.025, 0.025, "Total: %1.3e %s"
                     % (np.sum(self.signal), au.photon),
                     ha='left', va='bottom', color='white',
                     transform=self.ax.transAxes)
        cax.xaxis.set_label_position('top')
        cax.set_xlabel('Photon')
        fig.colorbar(im, cax=cax, orientation='vertical')

    def extr_arms(self, n=3, slice_n=slice_n):

        wave_snr = np.arange(self.wmins[0].value, self.wmaxs[2].value, snr_sampl.value)
        for a in range(n):
            wave_extr = self.wave_grid(self.wmins[a], self.wmaxs[a])
            for s in range(slice_n):
                print("Extracting slice %i of arm %i..." % (s+1, a+1))
                i = a*slice_n+s
                x = range(self.sl_cen[i]-self.sl_hlength,
                          self.sl_cen[i]+self.sl_hlength)
                s_extr = np.empty(int(self.ysize.value))
                n_extr = np.empty(int(self.ysize.value))
                for p in range(int(self.ysize.value)):
                    y = self.image[p, self.sl_cen[i]-self.sl_hlength:
                                      self.sl_cen[i]+self.sl_hlength]
                    b1 = np.mean(self.image[p, self.sl_cen[i]-self.sl_hlength+1:
                                            self.sl_cen[i]-self.sl_hlength+6])
                    b2 = np.mean(self.image[p, self.sl_cen[i]+self.sl_hlength-6:
                                            self.sl_cen[i]+self.sl_hlength-1])
                    y = y - 0.5*(b1+b2)
                    dy = self.noise[p, self.sl_cen[i]-self.sl_hlength:
                                      self.sl_cen[i]+self.sl_hlength]
                    s_extr[p], n_extr[p] = getattr(self, 'extr_'+self.func)\
                        (y, dy=dy, mod=self.mod_init[i], x=x, p=p)
                if s == 0:
                    flux_extr = s_extr
                    err_extr = n_extr
                else:
                    flux_extr += s_extr
                    err_extr = np.sqrt(err_extr**2 + n_extr**2)
            dw = (wave_extr[2:]-wave_extr[:-2])*0.5
            dw = np.append(dw[:1], dw)
            dw = np.append(dw, dw[-1:])
            flux_extr = flux_extr / dw
            err_extr = err_extr / dw
            line = self.spec.ax.scatter(wave_extr, flux_extr, s=2, c='C0')
            #line, = self.spec.ax.step(wave_extr, flux_extr, linewidth=3, c='C0')

            print("Median error of extraction: %2.3e" % np.nanmedian(err_extr))
            flux_window = flux_extr[3000//ccd_xbin:3100//ccd_ybin]
            print("RMS of extraction: %2.3e"
                  % np.sqrt(np.nanmean(np.square(
                            flux_window-np.nanmean(flux_window)))))

            #wave_snr = wave_extr[::snr_sampl]
            snr_extr = flux_extr/err_extr
            snr_extr[np.where(np.isnan(snr_extr))] = 0
            snr_extr[np.where(np.isinf(snr_extr))] = 0
            snr_spl = cspline(wave_extr, snr_extr)(wave_snr)
            snr_spl[wave_snr<self.wmins[a].value] = 0.0
            snr_spl[wave_snr>self.wmaxs[a].value] = 0.0
            if a == 0:
                snr = snr_spl
                line.set_label('Extracted')
            else:
                snr = np.sqrt(snr**2+snr_spl**2)
            #snr = uspline(wave_extr, snr_extr)(wave_snr)

            """
            linet, = self.spec.ax_snr.plot(wave_snr, snr, c='black')

            if a == 0:
                line.set_label('Extracted')
                linet.set_label('SNR')
                self.spec.ax_snr.text(self.wmaxs[2].value, 0,
                                      "Median SNR: %2.1f" % np.median(snr),
                                      ha='right', va='bottom')
            self.spec.ax_snr.legend(loc=2)
            """
        linet, = self.spec.ax_snr.plot(wave_snr, snr, linestyle='--', c='black')
        linet.set_label('SNR')
        self.spec.ax_snr.text(0.99, 0.92,
                              "Median SNR: %2.1f" % np.median(snr),
                              ha='right', va='top',
                              transform=self.spec.ax_snr.transAxes)
        self.spec.ax_snr.legend(loc=2, fontsize=8)

    def extr_sum(self, y, dy, **kwargs):
        sel = np.s_[self.sl_hlength-self.psf_xlength
                    :self.sl_hlength+self.psf_xlength]
        ysel = y[sel]
        dysel = dy[sel]
        s = np.sum(ysel)
        n = np.sqrt(np.sum(dysel**2))
        if np.isnan(s) or np.isnan(n) or np.isinf(s) or np.isinf(n) \
            or np.abs(s) > 1e30 or np.abs(n) > 1e30:
            s = 0
            n = 1
        return s, n

    def extr_opt(self, y, dy, mod, x, p):
        mod_fit = lm()(mod, x, y)(x)
        mod_fit[mod_fit < 1e-3] = 0
        if np.sum(mod_fit*dy) > 0 and not np.isnan(mod_fit).any():
            mod_norm = mod_fit/np.sum(mod_fit)
            #print(mod_fit)
            w = dy>0
            s = np.sum(mod_norm[w]*y[w]/dy[w]**2)/np.sum(mod_norm[w]**2/dy[w]**2)
            n = np.sqrt(np.sum(mod_norm[w])/np.sum(mod_norm[w]**2/dy[w]**2))
        else:
            s = 0
            n = 1
        if np.isnan(s) or np.isnan(n) or np.isinf(s) or np.isinf(n) \
            or np.abs(s) > 1e30 or np.abs(n) > 1e30:
            s = 0
            n = 1
        return s, n

    def rebin(self, arr, length):
        # Adapted from http://www.bdnyc.org/2012/03/rebin-numpy-arrays-in-python/
        #pix_length = length/self.spat_scale/self.pix_xsize
        # Good for now, but need to find more sophisticated solution
        zoom_factor = length / arr.shape[0]
        new = interpolation.zoom(arr, zoom_factor)
        if np.sum(new) != 0:
            return new/np.sum(new)*np.sum(arr)
        else:
            return new


    def tot_eff(self, wave, wmin_d, wmax_d, fact=2):
        dch_shape = expit(fact*(wave-wmin_d))*expit(fact*(wmax_d-wave))
        i = self.arm_counter
        adc = eff_adc[i]
        slc = eff_slc[i]
        dch = dch_shape * eff_dch[i]
        spc = eff_spc[i]
        grt = eff_grt[i]
        ccd = eff_ccd[i]
        tel = eff_tel[i]
        return adc * slc * dch * spc * grt * ccd * tel

    def wave_grid(self, wmin, wmax):
        return np.linspace(wmin, wmax, self.ysize.value)


class Photons(object):

    def __init__(self, targ_mag=targ_mag, bckg_mag=bckg_mag, area=area,
                 texp=texp):
        self.targ_mag = targ_mag
        self.bckg_mag = bckg_mag
        self.area = area #(400*au.cm)**2 * np.pi
        self.texp = texp
        if mag_syst == 'Vega':
            self.wave_ref = wave_U
            self.flux_ref = flux_U
        if mag_syst == 'AB':
            self.wave_ref = wave_u
            self.flux_ref = flux_u

        f = self.flux_ref * self.area * texp  # Flux density @ 555.6 nm, V = 0
        self.targ = f * pow(10, -0.4*self.targ_mag)
        self.bckg = f * pow(10, -0.4*self.bckg_mag) / au.arcsec**2

        self.atmo()

    def atmo(self):
        data = fits.open('database/atmoexan.fits')[1].data
        self.atmo_wave = data['LAMBDA']*0.1 * au.nm
        self.atmo_ex = data['LA_SILLA']


class PSF(object):

    def __init__(self, spec, seeing=seeing, slice_width=slice_width,
                 xsize=slice_length, ysize=slice_length, sampl=psf_sampl,
                 func=psf_func):
        #self.phot = phot
        self.spec = spec
        self.seeing = seeing
        self.slice_width = slice_width
        self.xsize = xsize
        self.ysize = ysize
        self.area = xsize * ysize
        self.sampl = sampl
        x = np.linspace(-xsize.value/2, xsize.value/2, sampl.value)
        y = np.linspace(-ysize.value/2, ysize.value/2, sampl.value)
        self.x, self.y = np.meshgrid(x, y)

        getattr(self, func)()  # Apply the chosen function for the PSF

        self.bckg = np.ones(self.spec.wave.shape) \
                    * self.spec.phot.bckg * self.area
        self.bckg_int = np.sum(self.bckg)/len(self.bckg) \
                        * (self.spec.wmax-self.spec.wmin)
        self.z = self.z/np.sum(self.z)  # Normalize the counts
        self.z_targ = self.z * self.spec.targ_int  # Counts from the target
        self.z_bckg = np.full(self.z.shape, self.bckg_int.value) / self.z.size \
                      * self.bckg_int.unit  # Counts from the background
        self.z = self.z_targ + self.z_bckg  # Total counts

    def add_slice(self, cen=None):  # Central pixel of the slice

        width = self.slice_width
        length = self.ysize
        if cen == None:
            cen = (0, 0)
        hwidth = width.value / 2
        hlength = length.value / 2

        rect = patches.Rectangle((cen[0]-hwidth, cen[1]-hlength), width.value,
                                 length.value, edgecolor='r', facecolor='none')

        # In this way, fractions of pixels in the slices are counted
        ones = np.ones(self.x.shape)
        pix_xsize = self.xsize / self.sampl
        pix_ysize = self.ysize / self.sampl
        left_dist = (self.x-cen[0]+hwidth) / pix_xsize.value
        right_dist = (cen[0]+hwidth-self.x) / pix_xsize.value
        down_dist = (self.y-cen[1]+hlength) / pix_ysize.value
        up_dist = (cen[1]+hlength-self.y) / pix_ysize.value
        mask_left = np.maximum(-ones, np.minimum(ones, left_dist))*0.5+0.5
        mask_right = np.maximum(-ones, np.minimum(ones, right_dist))*0.5+0.5
        mask_down = np.maximum(-ones, np.minimum(ones, down_dist))*0.5+0.5
        mask_up = np.maximum(-ones, np.minimum(ones, up_dist))*0.5+0.5
        mask = mask_left*mask_right*mask_down*mask_up
        cut = np.asarray(mask_down*mask_up>0).nonzero()

        mask_z = self.z * mask
        cut_shape = (int(len(cut[0])/mask_z.shape[1]), mask_z.shape[1])
        cut_z = mask_z[cut].reshape(cut_shape)

        flux = np.sum(mask_z)
        trace = np.sum(cut_z, axis=1)
        self.ax.add_patch(rect)

        return flux, trace

    def add_slices(self, n=slice_n, cen=None):

        width = self.slice_width
        length = self.ysize
        if cen == None:
            cen = (0,0)
        hwidth = width.value / 2
        hlength = length.value / 2
        shift = ((n+1)*width.value)/2
        cens = np.arange(cen[0]-shift, cen[0]+shift, width.value)
        self.flux_slice = 0
        for i, c in enumerate(cens[1:]):
            flux, trace = self.add_slice((c, cen[1]))
            self.flux_slice += flux
            if i == 0:
                self.fluxes = np.array([flux.value])
                self.traces = [trace]
            else:
                self.fluxes = np.append(self.fluxes, flux.value)
                self.traces = np.vstack((self.traces, trace))

        self.slice_area = min(n * width * length, n * width * self.ysize)
        self.bckg_slice = self.bckg_int * self.slice_area/self.area
        self.targ_slice = self.flux_slice-self.bckg_slice
        losses = 1-(self.targ_slice.value)/np.sum(self.z_targ.value)


        self.ax.text(-self.xsize.value/2*0.95, -self.ysize.value/2*0.71,
                     "Total: %2.3e %s"
                     % (self.flux_slice.value, self.flux_slice.unit),
                     ha='left', va='bottom', color='white')
        self.ax.text(-self.xsize.value/2*0.95, -self.ysize.value/2*0.79,
                     "Target: %2.3e %s"
                     % (self.targ_slice.value, self.targ_slice.unit),
                     ha='left', va='bottom', color='white')
        self.ax.text(-self.xsize.value/2*0.95, -self.ysize.value/2*0.87,
                     "Sky: %2.3e %s"
                     % (np.sum(self.bckg_slice.value), self.bckg_slice.unit),
                     ha='left', va='bottom', color='white')
        self.ax.text(-self.xsize.value/2*0.95, -self.ysize.value/2*0.95,
                     "Losses: %2.1f%%" \
                     % (losses*100),
                     ha='left', va='bottom', color='white')

        # Update spectrum plot with flux into slices and background
        self.spec.targ_loss = self.spec.targ*(1-losses)
        self.spec.targ_conv = gaussian_filter(
            self.spec.targ_loss, self.sigma.value)*self.spec.targ_loss.unit
        self.spec.norm_conv = gaussian_filter(
            self.spec.norm, self.sigma.value)*self.spec.targ_loss.unit

        self.spec.ax.plot(self.spec.wave, self.spec.targ_conv,
                          label='Collected')


    def draw(self, fig=None):
        if fig is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = fig.axes[2]
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = self.ax.contourf(self.x, self.y, self.z, 100, vmin=0)
        self.ax.set_title('PSF')
        self.ax.set_xlabel('X (%s)' % self.xsize.unit)
        self.ax.set_ylabel('Y (%s)' % self.xsize.unit)
        self.ax.text(-self.xsize.value/2*0.95, -self.ysize.value/2*0.63,
                     "FWHM: %3.2f arcsec" % self.fwhm, ha='left', va='bottom',
                     color='white')
        cax.xaxis.set_label_position('top')
        cax.set_xlabel('Photon')
        fig.colorbar(im, cax=cax, orientation='vertical')


    def gaussian(self, cen=psf_cen):
        ampl = 1
        theta = 0
        sigma = self.seeing.value/2 / np.sqrt(2*np.log(2))
        m = Gaussian2D(ampl, cen[0], cen[1], sigma, sigma, theta)
        self.z = m.evaluate(self.x, self.y, ampl, cen[0], cen[1], sigma, sigma,
                            theta)
        self.sigma = m.x_stddev
        self.fwhm = m.x_fwhm


    def moffat(self, cen=psf_cen):
        ampl = 1
        alpha = 3
        gamma = self.seeing.value/2 / np.sqrt(2**(1/alpha)-1)
        m = Moffat2D(1, cen[0], cen[1], gamma, alpha)
        self.z = m.evaluate(self.x, self.y, ampl, cen[0], cen[1], gamma,
                            alpha)
        self.fwhm = m.fwhm


    def tophat(self, cen=(0,0)):
        self.z = np.array((self.x-cen[0])**2 + (self.y-cen[1])**2
                          < (self.seeing.value/2)**2, dtype=int)
        self.fwhm = self.seeing.value


class Spec(object):

    def __init__(self, phot, file=None, wmin=305*au.nm, wmax=390*au.nm,
                 dw=1e-3*au.nm, func=spec_func):
        self.phot = phot
        self.file = file
        self.wmin = wmin
        self.wmax = wmax
        self.wmean = 0.5*(wmin+wmax)
        self.wave = np.arange(wmin.value, wmax.value, dw.value)*wmin.unit

        # Extrapolate extinction
        spl = cspline(self.phot.atmo_wave, self.phot.atmo_ex)(self.wave)
        self.atmo_ex = pow(10, -0.4*spl*airmass)

        flux = getattr(self, func)()  # Apply the chosen function for the spectrum
        self.normalize(flux)

    def draw(self, fig=None):
        if fig is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = fig.axes[0]
            self.ax_snr = fig.axes[1]
        self.ax.set_title("Spectrum: texp = %2.2f %s, targ_mag = %2.2f, "
                          "bckg_mag = %2.2f, airmass=%2.2f" \
                          % (self.phot.texp.value, self.phot.texp.unit, self.phot.targ_mag,
                             self.phot.bckg_mag, airmass))
        self.ax.plot(self.wave, self.targ/self.atmo_ex, label='Original')
        self.ax.plot(self.wave, self.targ, label='Extincted')
        self.ax.set_xlabel('Wavelength (%s)' % self.wave.unit)
        self.ax.set_ylabel('Flux density\n(%s)' % self.targ.unit)
        self.ax.grid(linestyle=':')
        self.ax_snr.set_xlabel('Wavelength')
        self.ax_snr.set_ylabel('SNR per pixel')
        self.ax_snr.grid(linestyle=':')

    def flat(self):
        return np.ones(self.wave.shape) * self.atmo_ex

    def normalize(self, flux):
        self.norm = flux/np.sum(flux) / au.nm
        self.targ = flux * self.phot.targ
        self.targ_int = np.sum(self.targ)/len(self.targ.value) \
                        * (self.wmax-self.wmin)

    def PL(self, index=-1.5):
        return (self.wave.value/self.phot.wave_ref.value)**index * self.atmo_ex

    def qso(self):#, name=qso_file):
        if self.file is None:
            name = qso_file
        else:
            name = self.file
        try:
            data = fits.open(name)[1].data
            data = data[:-1]
            data = data[np.logical_and(data['wave']*0.1 > self.wmin.value,
                                       data['wave']*0.1 < self.wmax.value)]
            wavef = data['wave']*0.1 * au.nm
            fluxf = data['flux']
            spl = cspline(wavef, fluxf)(self.wave.value)
            sel = np.where(self.wave.value<313)
            spl[sel] = 1.0
        except:
            data = ascii.read(name, data_start=1, names=['col1', 'col2', 'col3', 'col4'])
            wavef = data['col1']*0.1 * au.nm * (1+qso_zem)
            fluxf = data['col2']
            spl = cspline(wavef, fluxf)(self.wave.value)

        flux = spl/np.mean(spl) #* au.photon/au.nm
        #if qso_lya_abs:
        #    flux = self.lya_abs(flux)
        return flux * self.atmo_ex

    def lya_abs(self, flux):
        logN_0 = 12
        logN_1 = 14
        logN_2 = 18
        logN_3 = 19
        index = -1.65

        f_0 = 1.0
        f_1 = 1e14/np.log(1e14)
        f_2 = np.log(1e18)/np.log(1e14)*1e5

        tau_norm = 0.0028
        tau_index = 3.45



        den = (10**(logN_1*(2+index))-10**(logN_0*(2+index))) / (2+index)
        den = den + f_1 * (10**(logN_2*(1+index)))/(1+index) * np.log(10**(logN_2)) \
              - 1/(1+index)
        den = den - f_1 * (10**(logN_1*(1+index)))/(1+index) * np.log(10**(logN_1)) \
              - 1/(1+index)
        #print(den)

        corr = np.ones(len(flux))
        z = self.wave.value/121.567 - 1
        corr[z<qso_zem] = np.exp(-1e6*tau_norm*(1+z[z<qso_zem])**tau_index*1/den)*f_0
        #print(np.exp(tau_norm*(1+z[z<qso_zem])**tau_index*1/den), corr)
        return flux * corr

    def star(self):#, name=star_file):
        if self.file is None:
            name = star_file
        else:
            name = self.file
        data = ascii.read(name, data_start=2, names=['col1', 'col2'])#name)
        wavef = data['col1']*0.1 * au.nm
        fluxf = data['col2']
        spl = cspline(wavef, fluxf)(self.wave.value)
        flux = spl/np.mean(spl) #* au.photon/au.nm
        return flux * self.atmo_ex

def main():

    # Set up window
    fig = plt.figure(figsize=(12,8))
    axs = [plt.subplot(5,1,1), plt.subplot(5,1,2),
           plt.subplot(2,2,3, aspect='equal'),
           plt.subplot(2,2,4, aspect='equal')]
    axs[0].get_shared_x_axes().join(axs[0], axs[1])

    phot = Photons(targ_mag=targ_mag, bckg_mag=bckg_mag, texp=texp)

    spec = Spec(phot, file=in_file, func=spec_func)
    spec.draw(fig)

    psf = PSF(spec, seeing=seeing, slice_width=slice_width, xsize=slice_length,
              ysize=slice_length)
    psf.draw(fig)
    psf.add_slices()

    ccd = CCD(psf, spec, xbin=ccd_xbin, ybin=ccd_ybin)
    ccd.add_arms()
    ccd.extr_arms()
    ccd.draw(fig)

    spec.ax.legend(loc=2, fontsize=8)
    # Tight layout shrinks images when binning higher than 1x1 is used
    #if ccd_xbin==1 and ccd_ybin==1:
    #    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if out_fig != None:
        plt.savefig(out_fig, format='png')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        in_file = 'database/'+sys.argv[1]
        targ_mag = float(sys.argv[2])
        bckg_mag = float(sys.argv[3])
        seeing = float(sys.argv[4])*au.arcsec
        airmass = float(sys.argv[5])
        slice_width = float(sys.argv[6])*au.arcsec
        slice_length = float(sys.argv[7])*au.arcsec
        ccd_xbin = int(sys.argv[8])
        ccd_ybin = int(sys.argv[9])
        texp = float(sys.argv[10])*au.s
        out_fig = 'database/'+sys.argv[11]
        if sys.argv[1] == 'Zheng+97.txt':
            spec_func = 'qso'
        else:
            spec_func = 'star'
    else:
        print('Loading command line parameters from cubes_sim_config.py...')
        if spec_func == 'qso':
            in_file = 'database/'+qso_file
        if spec_func == 'star':
            in_file = 'database/'+star_file
        out_fig = None
    main()
