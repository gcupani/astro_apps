from astropy import units as au
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


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

psf_func = 'tophat'  # Function to represent the PSF
seeing = 1.5*au.arcsec  # Seeing
sampl = 80*au.pixel

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

        self.image[:,pix_cen-pix_hlength:pix_cen+pix_hlength] = trace

    def add_slices(self, traces, n=slice_n, length=slice_length, gap=slice_gap,
                   pix_cen=None):
        if pix_cen == None:
            pix_cen = int(self.ysize.value//2)
        pix_length = int(length/self.spat_scale/self.pix_xsize)
        pix_span = pix_length + int(gap.value)
        pix_shift = (n*pix_span+pix_length)//2
        pix_cens = range(pix_cen-pix_shift, pix_cen+pix_shift, pix_span)
        for c, t in zip(pix_cens[1:], traces):
            print(t)
            self.add_slice(t, length, c)

    def draw(self):
        fig, self.ax = plt.subplots()
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = self.ax.imshow(self.image)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        fig.colorbar(im, cax=cax, orientation='vertical')

class PSF(object):

    def __init__(self, xsize=5*au.arcsec, ysize=5*au.arcsec,
                 sampl=sampl, func=psf_func):
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

        mask = np.logical_and(self.x > cen[0]-hwidth, self.x < cen[0]+hwidth)

        flux = np.sum(self.z * mask)
        trace = np.sum(self.z * mask, axis=1)
        self.ax.add_patch(rect)

        self.flux = flux
        self.trace = trace
        return flux, trace
        #self.ax.contourf(self.x, self.y, self.z * mask)

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
        print(self.fluxes)


    def draw(self):
        fig, self.ax = plt.subplots()
        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        im = self.ax.contourf(self.x, self.y, self.z)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        fig.colorbar(im, cax=cax, orientation='vertical')


    def tophat(self, seeing=seeing, pix_cen=None):
        self.z = np.array(self.x**2 + self.y**2 < (seeing.value/2)**2, dtype=int)


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
