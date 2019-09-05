from astropy import units as au
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.ndimage import interpolation
from astropy.modeling.functional_models import Gaussian2D, Moffat2D


ccd_xsize = 4096*au.pixel  # X size of the CCD
ccd_ysize = 4096*au.pixel  # Y size of the CCD
pix_xsize = 15*au.micron  # X size of the pixels
pix_ysize = 15*au.micron  # Y size of the pixels
spat_scale = 0.25*au.arcsec/(30*au.micron)  # Spatial scale
slice_n = 6  # Number of slices
slice_length = 10*au.arcsec  # Length of the slice
slice_width = 0.25*au.arcsec  # Width of the slice
slice_gap = 40*au.pixel  # Length of the slice
arm_cens = [512*au.pixel, 2048*au.pixel, 3584*au.pixel]  # Center of add_arms

seeing = 1.0*au.arcsec  # Seeing
psf_func = 'gaussian'  # Function to represent the PSF
psf_sampl = 1000*au.pixel

class CCD(object):

    def __init__(self, xsize=ccd_xsize, ysize=ccd_ysize, pix_xsize=pix_xsize,
                 pix_ysize=pix_ysize, spat_scale=spat_scale, slice_n=slice_n):
        self.xsize = xsize
        self.ysize = ysize
        self.pix_xsize = pix_xsize
        self.pix_ysize = pix_ysize
        self.spat_scale = spat_scale

        self.image = np.zeros((int(xsize.value), int(ysize.value)))

    def add_arms(self, traces, cens=arm_cens):
        for c in arm_cens:
            self.add_slices(traces, pix_cen=int(c.value))


    def add_slice(self, trace, length=slice_length,
                  pix_cen=None):  # Central pixel of the slice
        if pix_cen == None:
            pix_cen = int(self.ysize.value//2)
        pix_hlength = int(length/self.spat_scale/self.pix_xsize // 2)

        new_trace = self.rebin(trace)
        self.image[:,pix_cen-pix_hlength:pix_cen+pix_hlength] = new_trace

    def add_slices(self, traces, n=slice_n, length=slice_length, gap=slice_gap,
                   pix_cen=None):
        if pix_cen == None:
            pix_cen = int(self.ysize.value//2)
        pix_length = int(length/self.spat_scale/self.pix_xsize)
        pix_span = pix_length + int(gap.value)
        pix_shift = (n*pix_span+pix_length)//2
        pix_cens = range(pix_cen-pix_shift, pix_cen+pix_shift, pix_span)
        for c, t in zip(pix_cens[1:], traces):
            self.add_slice(t, length, c)

    def draw(self):
        fig, self.ax = plt.subplots()
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = self.ax.imshow(self.image)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        fig.colorbar(im, cax=cax, orientation='vertical')

    def rebin(self, trace, length=slice_length):
        # Adapted from http://www.bdnyc.org/2012/03/rebin-numpy-arrays-in-python/
        pix_length = length/self.spat_scale/self.pix_xsize
        zoom_factor = pix_length.value / trace.shape[0]
        return interpolation.zoom(trace, zoom_factor)

class PSF(object):

    def __init__(self, xsize=seeing*3, ysize=seeing*3,
                 sampl=psf_sampl, func=psf_func):
        self.xsize = xsize
        self.ysize = ysize
        self.sampl = sampl
        x = np.linspace(-xsize.value/2, xsize.value/2, sampl.value)
        y = np.linspace(-ysize.value/2, ysize.value/2, sampl.value)
        self.x, self.y = np.meshgrid(x, y)

        getattr(self, func)()  # Apply the chosen function for the PSF


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
        zeros = np.zeros(self.x.shape)
        pix_xsize = self.xsize / self.sampl
        pix_ysize = self.ysize / self.sampl
        left_dist = (self.x-cen[0]+hwidth) / pix_xsize.value
        right_dist = (cen[0]+hwidth-self.x) / pix_xsize.value
        down_dist = (self.x-cen[1]+hlength) / pix_ysize.value
        up_dist = (cen[1]+hlength-self.x) / pix_ysize.value
        mask_left = np.maximum(zeros, np.minimum(ones, left_dist+1))
        mask_right = np.maximum(zeros, np.minimum(ones, right_dist+1))
        mask_down = np.maximum(zeros, np.minimum(ones, down_dist+1))
        mask_up = np.maximum(zeros, np.minimum(ones, up_dist+1))
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
        for i, c in enumerate(cens[1:]):
            flux, trace = self.add_slice(width, length, (c, cen[1]))
            if i == 0:
                self.fluxes = np.array([flux])
                self.traces = trace
            else:
                self.fluxes = np.append(self.fluxes, flux)
                self.traces = np.vstack((self.traces, trace))

    def draw(self):
        fig, self.ax = plt.subplots()
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = self.ax.contourf(self.x, self.y, self.z, 100)
        self.ax.set_xlabel('X (%s)' % self.xsize.unit)
        self.ax.set_ylabel('Y (%s)' % self.xsize.unit)
        self.ax.text(-self.xsize.value/2.1, -self.ysize.value/2.1,
                     "FWHM = %3.2f arcsec" % self.fwhm, ha='left', va='bottom',
                     color='white')
        fig.colorbar(im, cax=cax, orientation='vertical')

    def gaussian(self, seeing=seeing, cen=(0,0)):
        ampl = 1
        theta = 0
        sigma = seeing.value/2 / np.sqrt(2*np.log(2))
        m = Gaussian2D(ampl, cen[0], cen[1], sigma, sigma, theta)
        self.z = m.evaluate(self.x, self.y, ampl, cen[0], cen[1], sigma, sigma,
                            theta)
        self.fwhm = m.x_fwhm


    def moffat(self, seeing=seeing, cen=(0,0)):
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



def main():

    psf = PSF()
    psf.draw()
    psf.add_slices()

    ccd = CCD()
    ccd.add_arms(psf.traces)
    ccd.draw()

    plt.show()

if __name__ == '__main__':
    main()
