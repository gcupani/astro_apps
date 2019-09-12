from astropy import units as au
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import  CubicSpline as cspline #interp1d
from scipy.ndimage import interpolation
from scipy.special import expit
from astropy.modeling.fitting import LevMarLSQFitter as lm
from astropy.modeling.functional_models import Gaussian1D, Gaussian2D, Moffat2D


eff_adc = [0.96, 0.96, 0.96]  # ADC efficiency
eff_slc = [0.98, 0.98, 0.98]  # Slicer efficiency
eff_dch = [0.96, 0.96, 0.96]  # Dichroics efficiency
eff_spc = [0.93, 0.94, 0.93]  # Spectrograph efficiency
eff_grt = [0.90, 0.90, 0.90]  # Grating efficiency
eff_ccd = [0.85, 0.85, 0.85]  # CCD QE
eff_tel = [0.72, 0.72, 0.72]  # Telescope efficiency

ccd_xsize = 4096*au.pixel  # X size of the CCD
ccd_ysize = 4096*au.pixel  # Y size of the CCD
pix_xsize = 15*au.micron  # X size of the pixels
pix_ysize = 15*au.micron  # Y size of the pixels
spat_scale = 0.25*au.arcsec/(30*au.micron)  # Spatial scale
slice_n = 6  # Number of slices
slice_length = 10*au.arcsec  # Length of the slice
slice_width = 0.25*au.arcsec  # Width of the slice
slice_gap = 40*au.pixel  # Length of the slice
ccd_bias = 100*au.adu
ccd_ron = 2*au.adu
ccd_gain = 1*au.photon/au.adu

seeing = 1.0*au.arcsec  # Seeing
psf_func = 'gaussian'  # Function to represent the PSF
psf_sampl = 1000*au.pixel  # Size of the PSF image
psf_cen = (0,0)  # Center of the PSF

texp = 3600*au.s  # Exposure time
targ_mag = 16  # Magnitude of the target
bckg_mag = 22.5  # Magnitude of the background
airmass = 1.0  # Airmass

spec_func = 'PL'  # Function for the template spectrum or filename
spec_file = '/Users/guido/GitHub/astrocook/test_data/J1124-1705.fits'
extr_func = 'sum'  # Function for extracting the spectrum (sum or opt)

class CCD(object):

    def __init__(self, psf, spec, xsize=ccd_xsize, ysize=ccd_ysize,
                 pix_xsize=pix_xsize, pix_ysize=pix_ysize,
                 spat_scale=spat_scale, slice_n=slice_n, func=extr_func):
        self.psf = psf
        self.spec = spec
        self.xsize = xsize
        self.ysize = ysize
        self.pix_xsize = pix_xsize
        self.pix_ysize = pix_ysize
        self.spat_scale = spat_scale

        self.signal = np.zeros((int(xsize.value), int(ysize.value)))
        self.func = func

    def add_arms(self, n=3):
        s = int(self.xsize.value/(n*3-1))
        self.xcens = np.arange(s, self.xsize.value, 3*s)
        if n == 3:
            self.wmins = [305, 328, 355] * au.nm
            self.wmaxs = [335, 361, 390] * au.nm
            self.wmins_d = [300, 331.5, 358] * au.nm  # Dichroich transition wavelengths
            self.wmaxs_d = [331.5, 358, 400] * au.nm
            #wmins = [305, 330, 360] * au.nm
            #wmaxs = [330, 360, 390] * au.nm

        self.mod_init = []
        self.sl_cen = []
        #self.sl_hlength = []
        for i, (x, m, M, m_d, M_d) in enumerate(zip(
            self.xcens, self.wmins, self.wmaxs, self.wmins_d, self.wmaxs_d)):
            self.arm_counter = i
            self.arm_range = np.logical_and(self.spec.wave.value>m.value,
                                            self.spec.wave.value<M.value)
            self.arm_wave = self.spec.wave[self.arm_range].value
            self.arm_targ = self.spec.targ_loss[self.arm_range].value
            xlength = int(slice_length/self.spat_scale/self.pix_xsize)
            self.sl_hlength = xlength // 2
            psf_xlength = int(seeing/self.spat_scale/self.pix_xsize)
            xspan = xlength + int(slice_gap.value)
            xshift = (slice_n*xspan+xlength)//2
            self.add_slices(int(x), xshift, xspan, psf_xlength,
                            wmin=m.value, wmax=M.value, wmin_d=m_d.value,
                            wmax_d=M_d.value)
            line, = self.spec.ax.plot(
                self.arm_wave, self.arm_targ \
                * self.tot_eff(self.arm_wave, m_d.value, M_d.value), c='C3')
            if i == 0:
                line.set_label('On detector')


        self.shot = np.sqrt(self.signal)
        self.noise = np.sqrt(self.shot**2+(ccd_ron*ccd_gain).value**2)
        self.noise_rand = np.random.normal(0, self.noise, self.signal.shape)

        self.image = np.round(self.signal + self.noise_rand) #\
                              #+ (ccd_bias*ccd_gain).value)

        """
        for i, (c, hl) in enumerate(zip(self.sl_cen, self.sl_hlength)):
            if i == 0:
                snr = self.signal[:, c-hl:c+hl]/self.noise[:, c-hl:c+hl]
            else:
                snr = np.append(snr, self.signal[:, c-hl:c+hl]\
                                     /self.noise[:, c-hl:c+hl])
        self.median_snr = np.median(snr)
        """

    def add_slice(self, trace, xcen, wmin, wmax, wmin_d, wmax_d):

        if wmin is not None and wmax is not None:
            wave = self.wave_grid(wmin, wmax)
            norm = self.spec.norm[np.logical_and(self.spec.wave.value > wmin,
                                                 self.spec.wave.value < wmax)]
            #norm = norm * self.dich(wave, wmin_d, wmax_d)
        else:
            norm = self.spec.norm

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
            #self.sl_hlength.append(sl_hlength)


    def draw(self, fig):
        if fig is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = fig.axes[2]
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = self.ax.imshow(self.image, vmin=0)
        self.ax.set_title('CCD')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.text(100, 4000, "Total: %1.3e %s"
                     % (np.sum(self.signal), au.photon),
                     ha='left', va='bottom', color='white')
        """
        self.ax.text(100, 4000, "Median SNR: %2.3f"
                     % self.median_snr,
                     ha='left', va='bottom', color='white')
        """
        fig.colorbar(im, cax=cax, orientation='vertical')

    def extr_arms(self, n=3, slice_n=slice_n):


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
                    dy = self.shot[p, self.sl_cen[i]-self.sl_hlength:
                                      self.sl_cen[i]+self.sl_hlength]
                    s_extr[p], n_extr[p] = getattr(self, 'extr_'+self.func)\
                        (y, dy=dy, mod=self.mod_init[i], x=x, p=p)
                    #mod_fit = lm()(self.mod_init[i], x, y)
                if s == 0:
                    flux_extr = s_extr
                    err_extr = n_extr
                else:
                    flux_extr += s_extr
                    err_extr = np.sqrt(err_extr**2 + n_extr**2)

            dw = (wave_extr[2:]-wave_extr[:-2])*0.5
            dw = np.append(dw[:1], dw)
            dw = np.append(dw, dw[-1:])
            flux_extr = flux_extr / dw#np.append(1, wave_extr[1:]-wave_extr[:-1])
            err_extr = err_extr / dw#np.append(1, wave_extr[1:]-wave_extr[:-1])
            line = self.spec.ax.scatter(wave_extr, flux_extr, s=2, c='C0')

            if a == 0:
                axt = self.spec.ax.twinx()
                axt.set_ylabel('SNR per pixel')

            #print(err_extr)
            #print(np.median(err_extr))
            #print(np.sqrt(np.mean(np.square(flux_extr[3000:3100]-np.mean(flux_extr[3000:3100])))))

            no_nan = np.logical_and(True,
                                    True)
            wave_snr = wave_extr[::200]
            snr = cspline(wave_extr, flux_extr/err_extr)(wave_snr)
            linet = axt.scatter(wave_snr, snr, s=4, c='black')

            if a == 0:
                line.set_label('Extracted')
                linet.set_label('SNR')
                axt.text(self.wmaxs[2].value, 0,
                         "Median SNR: %2.1f" % np.median(snr),
                         ha='right', va='bottom')
            axt.legend(loc=1)


    def extr_sum(self, y, dy, **kwargs):
        s = np.sum(y)
        n = np.sqrt(np.sum(dy**2))
        if s != 0 and False:
            fact = np.sum(y)/s
        else:
            fact = 1.0
        s = s * fact
        n = n * fact
        return s, n

    def extr_wgh(self, y, dy, **kwargs):
        w = np.ones(dy[dy>0].shape)
        w = 1/dy[dy>0]**2
        #w = w/np.mean(w)
        if len(w) > 0:
            s = np.average(y[dy>0], weights=w)
            n = np.sqrt(1/np.sum(w)**2)
        else:
            s = 0.0
            n = 1.0

        if np.sum(s) > 0:
            fact = np.sum(y)/np.sum(s)
        else:
            fact = 1.0
        s = s * fact
        n = n * fact
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
            print(p, mod_fit[w], mod_norm[w], dy[w], y[w], s, n)
            s = 0
            n = 1
        #print(s/n)
        if p > 2000:
            print(s,n, np.sum(y),np.sqrt(np.sum(dy**2)))
            print(s/n, np.sum(y)/np.sqrt(np.sum(dy**2)))
            fig, ax = plt.subplots()
            ax.plot(x, y)
            ax.plot(x, mod_fit)
            plt.show()
        return s, n

    def rebin(self, arr, length):
        # Adapted from http://www.bdnyc.org/2012/03/rebin-numpy-arrays-in-python/
        #pix_length = length/self.spat_scale/self.pix_xsize
        # Good for now, but find more sophisticated solution
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
        #return dch_shape

    def wave_grid(self, wmin, wmax):
        return np.linspace(wmin, wmax, self.ysize.value)


class Photons(object):

    def __init__(self, texp=texp):
        self.texp = texp
        self.area = (400*au.cm)**2 * np.pi
        f600 = 9480 * au.photon / au.cm**2 / au.s / au.nm \
               * self.area * texp  # Flux density @ 600 nm
        self.targ = f600 * pow(10, -0.4*targ_mag)
        self.bckg = f600 * pow(10, -0.4*bckg_mag) / au.arcsec**2

        self.atmo()

    def atmo(self):
        data = fits.open('atmoexan.fits')[1].data
        self.atmo_wave = data['LAMBDA']*0.1 * au.nm
        self.atmo_ex = data['LA_SILLA']


class PSF(object):

    def __init__(self, spec, xsize=slice_length, ysize=slice_length,
                 sampl=psf_sampl, func=psf_func):
        #self.phot = phot
        self.spec = spec
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

    def add_slice(self, width=slice_width, length=slice_length,
                  cen=None):  # Central pixel of the slice

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
        #mask_left = self.x > cen[0]-hwidth
        #mask_right = self.x < cen[0]+hwidth
        mask = mask_left*mask_right*mask_down*mask_up
        cut = np.asarray(mask_down*mask_up>0).nonzero()

        mask_z = self.z * mask
        cut_shape = (int(len(cut[0])/mask_z.shape[1]), mask_z.shape[1])
        cut_z = mask_z[cut].reshape(cut_shape)

        flux = np.sum(mask_z)
        trace = np.sum(cut_z, axis=1)
        self.ax.add_patch(rect)

        #self.flux = flux
        #self.trace = trace
        return flux, trace

    def add_slices(self, n=slice_n, width=slice_width, length=slice_length,
                   cen=None):

        if cen == None:
            cen = (0,0)
        hwidth = width.value / 2
        hlength = length.value / 2
        shift = ((n+1)*width.value)/2
        cens = np.arange(cen[0]-shift, cen[0]+shift, width.value)
        self.flux_slice = 0
        for i, c in enumerate(cens[1:]):
            flux, trace = self.add_slice(width, length, (c, cen[1]))
            self.flux_slice += flux
            if i == 0:
                self.fluxes = np.array([flux.value])
                self.traces = trace
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
        self.spec.ax.plot(self.spec.wave, self.spec.targ_loss,
                          label='Collected')
        #self.spec.ax.plot(self.spec.wave, self.bckg, label='Sky flux')

    def draw(self, fig=None):
        if fig is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = fig.axes[1]
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = self.ax.contourf(self.x, self.y, self.z, 100, vmin=0)
        self.ax.set_title('PSF')
        self.ax.set_xlabel('X (%s)' % self.xsize.unit)
        self.ax.set_ylabel('Y (%s)' % self.xsize.unit)
        self.ax.text(-self.xsize.value/2*0.95, -self.ysize.value/2*0.63,
                     "FWHM: %3.2f arcsec" % self.fwhm, ha='left', va='bottom',
                     color='white')
        fig.colorbar(im, cax=cax, orientation='vertical')


    def gaussian(self, seeing=seeing, cen=psf_cen):
        ampl = 1
        theta = 0
        sigma = seeing.value/2 / np.sqrt(2*np.log(2))
        m = Gaussian2D(ampl, cen[0], cen[1], sigma, sigma, theta)
        self.z = m.evaluate(self.x, self.y, ampl, cen[0], cen[1], sigma, sigma,
                            theta)
        self.fwhm = m.x_fwhm


    def moffat(self, seeing=seeing, cen=psf_cen):
        ampl = 1
        alpha = 3
        gamma = seeing.value/2 / np.sqrt(2**(1/alpha)-1)
        m = Moffat2D(1, cen[0], cen[1], gamma, alpha)
        self.z = m.evaluate(self.x, self.y, ampl, cen[0], cen[1], gamma,
                            alpha)
        self.fwhm = m.fwhm


    def tophat(self, seeing=seeing, cen=(0,0)):
        self.z = np.array((self.x-cen[0])**2 + (self.y-cen[1])**2
                          < (seeing.value/2)**2, dtype=int)
        self.fwhm = seeing.value


class Spec(object):

    def __init__(self, phot, wmin=305*au.nm, wmax=390*au.nm, dw=1e-3*au.nm,
                 func=spec_func):
        self.phot = phot

        self.wmin = wmin
        self.wmax = wmax
        self.wmean = 0.5*(wmin+wmax)
        self.wave = np.arange(wmin.value, wmax.value, dw.value)*wmin.unit

        # Extrapolate extinction
        spl = cspline(self.phot.atmo_wave, self.phot.atmo_ex)(self.wave)
        #self.atmo_ex = pow(10, -0.4*spl*(airmass-1))
        self.atmo_ex = pow(10, -0.4*spl*airmass)

        flux = getattr(self, func)()  # Apply the chosen function for the spectrum
        self.normalize(flux)

    def draw(self, fig=None):
        if fig is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = fig.axes[0]
        self.ax.set_title('Spectrum')
        self.ax.plot(self.wave, self.targ/self.atmo_ex, label='Original')
        self.ax.plot(self.wave, self.targ, label='Extincted')
        self.ax.set_xlabel('Wavelength (%s)' % self.wave.unit)
        self.ax.set_ylabel('Flux density (%s)' % self.targ.unit)
        #self.ax.set_yscale('log')

    def file(self, name=spec_file):
        data = fits.open(name)[1].data
        data = data[np.logical_and(data['wave']*0.1 > self.wmin.value,
                                   data['wave']*0.1 < self.wmax.value)]
        self.wave = data['wave']*0.1 * au.nm
        self.norm = data['flux'] * au.photon/au.nm

    def flat(self):
        return np.ones(self.wave.shape) * self.atmo_ex

    def normalize(self, flux):
        self.norm = flux/np.sum(flux) / au.nm
        self.targ = flux * self.phot.targ
        self.targ_int = np.sum(self.targ)/len(self.targ.value) \
                        * (self.wmax-self.wmin)

    def PL(self, index=-2):
        return (self.wave.value/600)**index * self.atmo_ex # Normalized @ 600 nm
        """
        self.norm = flux/np.sum(flux) / au.nm
        #self.targ = self.norm * self.psf.targ_tot * au.photon
        #self.targ = self.norm * self.phot.targ * au.nm
        self.targ = flux * self.phot.targ
        self.targ_int = np.sum(self.targ)/len(self.targ.value) \
                        * (self.wmax-self.wmin)
        """


def main():

    #fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    fig = plt.figure(figsize=(12,9))
    axs = [plt.subplot(211), plt.subplot(223, aspect='equal'), plt.subplot(224)]

    phot = Photons()

    spec = Spec(phot)
    spec.draw(fig)

    psf = PSF(spec)
    psf.draw(fig)
    psf.add_slices()

    ccd = CCD(psf, spec)
    ccd.add_arms()
    ccd.extr_arms()
    ccd.draw(fig)

    spec.ax.legend(loc=2)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
