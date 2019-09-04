from astropy import units as au
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


ccd_xsize = 4096*au.pixel  # X size of the CCD
ccd_ysize = 4096*au.pixel  # Y size of the CCD
pix_xsize = 15*au.micron  # X size of the pixels
pix_ysize = 15*au.micron  # Y size of the pixels
spat_scale = 0.25*au.arcsec/(30*au.micron)  # Spatial scale
slice_n = 6  # Number of slices
slice_length = 10*au.arcsec  # Length of the slice
slice_gap = 40*au.pixel  # Length of the slice
arm_cens = [512*au.pixel, 2048*au.pixel, 3584*au.pixel]  # Center of add_arms

seeing = 1.*au.arcsec  # Seeing

def show(image):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im = ax.imshow(image)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(im, cax=cax, orientation='vertical')

class CCD(object):

    def __init__(self, xsize=ccd_xsize, ysize=ccd_ysize, pix_xsize=pix_xsize,
                 pix_ysize=pix_ysize, spat_scale=spat_scale, slice_n=slice_n):
        self.xsize = xsize
        self.ysize = ysize
        self.pix_xsize = pix_xsize
        self.pix_ysize = pix_ysize
        self.spat_scale = spat_scale

        self.image = np.zeros((int(xsize.value), int(ysize.value)))


    def add_arms(self, cens=arm_cens):
        for c in arm_cens:
            self.add_slices(pix_cen=int(c.value))


    def add_slice(self, length=slice_length,
                  pix_cen=None):  # Central pixel of the slice
        if pix_cen == None:
            pix_cen = int(self.ysize.value//2)
        pix_hlength = int(length/self.spat_scale/self.pix_xsize // 2)

        self.image[:,pix_cen-pix_hlength:pix_cen+pix_hlength] = 1

    def add_slices(self, n=slice_n, length=slice_length, gap=slice_gap,
                   pix_cen=None):
        if pix_cen == None:
            pix_cen = int(self.ysize.value//2)
        pix_length = int(length/self.spat_scale/self.pix_xsize)
        pix_span = pix_length + int(gap.value)
        pix_shift = (n*pix_span+pix_length)//2
        pix_cens = range(pix_cen-pix_shift, pix_cen+pix_shift, pix_span)
        for c in pix_cens[1:]:
            self.add_slice(length, c)

    def show(self):
        show(self.image)


class PSF(object):

    def __init__(self, xsize=1000, ysize=1000):
        self.xsize = xsize
        self.ysize = ysize
        self.image = np.zeros((xsize, ysize))


    def show(self):
        show(self.image)

    def tophat(self, seeing=seeing):


def main():

    psf = PSF()
    psf.show()

    ccd = CCD()
    ccd.add_arms()
    ccd.show()

    plt.show()

if __name__ == '__main__':
    main()
