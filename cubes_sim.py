from astropy import units as au
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.ndimage import interpolation
from astropy.modeling.functional_models import Gaussian2D, Moffat2D

texp = 3600*au.s  # Exposure time

ccd_xsize = 4096*au.pixel  # X size of the CCD
ccd_ysize = 4096*au.pixel  # Y size of the CCD
pix_xsize = 15*au.micron  # X size of the pixels
pix_ysize = 15*au.micron  # Y size of the pixels
spat_scale = 0.25*au.arcsec/(30*au.micron)  # Spatial scale
slice_n = 6  # Number of slices
slice_length = 10*au.arcsec  # Length of the slice
slice_width = 0.25*au.arcsec  # Width of the slice
slice_gap = 40*au.pixel  # Length of the slice
arm_n = 3  # Number of arms


seeing = 1*au.arcsec  # Seeing
psf_func = 'gaussian'  # Function to represent the PSF
psf_sampl = 10*au.pixel  # Size of the PSF image
psf_cen = (0,0)

spec_func = 'PL'  # Function for the template spectrum or filename
spec_file = '/Users/guido/GitHub/astrocook/test_data/J1124-1705.fits'

class CCD(object):

    def __init__(self, psf, spec, xsize=ccd_xsize, ysize=ccd_ysize,
                 pix_xsize=pix_xsize, pix_ysize=pix_ysize,
                 spat_scale=spat_scale, slice_n=slice_n):
        self.psf = psf
        self.spec = spec
        self.xsize = xsize
        self.ysize = ysize
        self.pix_xsize = pix_xsize
        self.pix_ysize = pix_ysize
        self.spat_scale = spat_scale

        self.image = np.zeros((int(xsize.value), int(ysize.value)))

    def add_arms(self, n=arm_n):
        s = int(self.xsize.value/(n*3-1))
        xcens = np.arange(s, self.xsize.value, 3*s)
        if n == 3:
            #wmins = [305, 328, 355] * au.nm
            #wmaxs = [335, 361, 390] * au.nm
            wmins = [305, 330, 360] * au.nm
            wmaxs = [330, 360, 390] * au.nm

        for x, m, M in zip(xcens, wmins, wmaxs):
            self.add_slices(xcens=int(x), wmin=m.value,
                            wmax=M.value)


    def add_slice(self, trace, length=slice_length,
                  xcen=None, wmin=None, wmax=None):
        if xcen == None:
            xcen = int(self.ysize.value//2)
        pix_hlength = int(length/self.spat_scale/self.pix_xsize // 2)

        if wmin is not None and wmax is not None:
            flux = self.spec.norm[np.logical_and(self.spec.wave.value > wmin,
                                                 self.spec.wave.value < wmax)]
            wave = np.linspace(wmin, wmax, self.ysize.value)
        else:
            flux = self.spec.norm

        new_trace = self.rebin(trace.value, pix_hlength*2)
        new_flux = self.rebin(flux.value, self.ysize.value)

        #spec.ax.plot(wave, new_flux)
        #print(self.image[:,xcen-pix_hlength:xcen+pix_hlength].shape)
        #print(np.multiply.outer(new_flux, new_trace).shape)
        self.image[:,xcen-pix_hlength:xcen+pix_hlength] = \
            np.multiply.outer(new_flux, new_trace)

    def add_slices(self, n=slice_n, length=slice_length,
                   gap=slice_gap, xcens=None, wmin=None, wmax=None):
        if xcens == None:
            xcens = int(self.ysize.value//2)
        pix_length = int(length/self.spat_scale/self.pix_xsize)
        pix_span = pix_length + int(gap.value)
        pix_shift = (n*pix_span+pix_length)//2
        pix_xcens = range(xcens-pix_shift, xcens+pix_shift, pix_span)
        for c, t in zip(pix_xcens[1:], self.psf.traces):
            self.add_slice(t, length, c, wmin, wmax)

    def draw(self, fig):
        if fig is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = fig.axes[2]
        divider = make_axes_locatable(self.ax)
        #cax = divider.append_axes('right', size='5%', pad=0.1)
        im = self.ax.imshow(self.image)
        self.ax.set_title('CCD')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.text(50, 4000, "Flux on detector: %i %s"
                     % (np.sum(self.image), au.photon),
                     ha='left', va='bottom', color='white')
        #fig.colorbar(im, cax=cax, orientation='vertical')

    def rebin(self, arr, length):
        # Adapted from http://www.bdnyc.org/2012/03/rebin-numpy-arrays-in-python/
        #pix_length = length/self.spat_scale/self.pix_xsize
        zoom_factor = length / arr.shape[0]
        new = interpolation.zoom(arr, zoom_factor)
        if np.sum(new) != 0:
            return new/np.sum(new)*np.sum(arr)
        else:
            return new

class Photons(object):

    def __init__(self, texp=texp):
        self.texp = texp
        self.area = (400*au.cm)**2 * np.pi
        self.target = 9480 * au.photon / au.cm**2 / au.s / au.nm \
                      * self.area * texp
        print(self.target)

class PSF(object):

    def __init__(self, phot, xsize=seeing*3, ysize=seeing*3,
                 sampl=psf_sampl, func=psf_func):
        self.phot = phot
        self.xsize = xsize
        self.ysize = ysize
        self.sampl = sampl
        x = np.linspace(-xsize.value/2, xsize.value/2, sampl.value)
        y = np.linspace(-ysize.value/2, ysize.value/2, sampl.value)
        self.x, self.y = np.meshgrid(x, y)

        getattr(self, func)()  # Apply the chosen function for the PSF
        print(self.z)
        self.z = self.z/np.sum(self.z) * self.phot.target   # Adjust the counts
        print(self.z)

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

        self.flux = flux
        self.trace = trace
        return flux, trace

    def add_slices(self, n=slice_n, width=slice_width, length=slice_length,
                   cen=None):
        if cen == None:
            cen = (0,0)
        hwidth = width.value / 2
        hlength = length.value / 2
        shift = ((n+1)*width.value)/2
        cens = np.arange(cen[0]-shift, cen[0]+shift, width.value)
        self.flux_tot = 0
        for i, c in enumerate(cens[1:]):
            flux, trace = self.add_slice(width, length, (c, cen[1]))
            self.flux_tot += flux
            if i == 0:
                self.fluxes = np.array([flux.value])
                self.traces = trace
            else:
                self.fluxes = np.append(self.fluxes, flux.value)
                self.traces = np.vstack((self.traces, trace))
        self.ax.text(-self.xsize.value/2*0.95, -self.ysize.value/2*0.79,
                     "Total flux: %i %s" % (np.sum(self.z.value), au.photon),
                     ha='left', va='bottom', color='white')
        self.ax.text(-self.xsize.value/2*0.95, -self.ysize.value/2*0.87,
                     "Flux into slices: %i %s"
                     % (self.flux_tot.value, au.photon),
                     ha='left', va='bottom', color='white')
        self.ax.text(-self.xsize.value/2*0.95, -self.ysize.value/2*0.95,
                     "Losses: %2.1f%%" \
                     % ((1-self.flux_tot.value/np.sum(self.z.value))*100),
                     ha='left', va='bottom', color='white')

    def draw(self, fig=None):
        if fig is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = fig.axes[1]
        divider = make_axes_locatable(self.ax)
        #cax = divider.append_axes('right', size='5%', pad=0.1)
        im = self.ax.contourf(self.x, self.y, self.z, 100)
        self.ax.set_title('PSF')
        self.ax.set_xlabel('X (%s)' % self.xsize.unit)
        self.ax.set_ylabel('Y (%s)' % self.xsize.unit)
        self.ax.text(-self.xsize.value/2*0.95, -self.ysize.value/2*0.71,
                     "FWHM: %3.2f arcsec" % self.fwhm, ha='left', va='bottom',
                     color='white')
        #fig.colorbar(im, cax=cax, orientation='vertical')

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

        getattr(self, func)()  # Apply the chosen function for the spectrum

    def draw(self, fig=None):
        if fig is None:
            fig, self.ax = plt.subplots()
        else:
            self.ax = fig.axes[0]
        self.ax.set_title('Spectrum')
        self.ax.plot(self.wave, self.flux)
        self.ax.set_xlabel('Wavelength (%s)' % self.wave.unit)
        self.ax.set_ylabel('Flux density (%s)' % self.flux.unit)

    def PL(self, index=-2):
        flux = self.wave.value**index
        self.norm = flux/np.sum(flux) / au.nm
        #self.flux = self.norm * self.psf.flux_tot * au.photon
        self.flux = self.norm * self.phot.target * au.nm
        self.flux = flux / au.nm
        

    def file(self, name=spec_file):
        data = fits.open(name)[1].data
        data = data[np.logical_and(data['wave']*0.1 > self.wmin.value,
                                   data['wave']*0.1 < self.wmax.value)]
        self.wave = data['wave']*0.1 * au.nm
        self.norm = data['flux'] * au.photon/au.nm

def main():

    fig, axs = plt.subplots(2, 2, figsize=(9, 9))

    phot = Photons()

    spec = Spec(phot)
    spec.draw(fig)

    psf = PSF(phot)
    psf.draw(fig)
    psf.add_slices()


    ccd = CCD(psf, spec)
    ccd.add_arms()
    ccd.draw(fig)

    plt.show()

if __name__ == '__main__':
    main()
