from astropy.io import fits
import astroscrappy as ascr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

sci = 'ESPRE.2019-05-03T23:16:23.402.fits'
bias = 'ESPRESSO_master_bias.fits'

def cut(data):
    down = data[:4617,:]
    up = data[4681:,:]
    cut_data = np.vstack((down, up))
    return cut_data

def detect_cosmics(data):
    mask, clean = ascr.detect_cosmics(data)
    return mask, clean

def frac(n, bins):
    return np.sum(n[np.where(bins[:-1]>14)])/np.sum(n)

def hist(data, fig, ax, title='Histogram', **kwargs):
    n, bins, patches = ax.hist(np.ravel(data), bins=20000,
                               range=[-10000,10000], histtype='step', **kwargs)

    ax.set_xlim(np.median(data)-20, np.median(data)+70)
    #ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Pixel value')
    ax.set_xlabel('Number of pixels')
    return n, bins

def imshow(data, fig, ax, title='Image'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    im = ax.imshow(np.log(data), vmin=0)
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(im, cax=cax, orientation='vertical', label='Log counts')


def read(file):
    frame = fits.open(file)
    blue = frame[1].data
    red = frame[2].data
    return blue, red


def main():

    fig = plt.figure(figsize=(15,5))
    #axs = [plt.subplot(231), plt.subplot(232), plt.subplot(233),
    #       plt.subplot(234), plt.subplot(235), plt.subplot(236)]
    axs = [plt.subplot(121), plt.subplot(122)]

    #frame = fits.open('ESPRE.2019-05-03T23:16:23.402.fits')
    #blue = frame[1].data
    raw_b, raw_r = read(sci)
    bias_b, bias_r = read(bias)

    sci_b = cut(raw_b)-bias_b
    sci_r = cut(raw_r)-bias_r

    #sci_b = raw_b
    #sci_r = raw_r

    #imshow(b, fig, fig.axes[0], 'blue')
    #imshow(r, fig, fig.axes[1], 'red')
    n_b, bins_b = hist(sci_b, fig, fig.axes[0], color='deepskyblue')
    n_r, bins_r = hist(sci_r, fig, fig.axes[1], color='lightsalmon')
    print(frac(n_b, bins_b))
    print(frac(n_r, bins_r))

    #sci_b_m, sci_b_c = detect_cosmics(sci_b)
    try:
        sci_b_c = np.load('sci_b_c.npy')
        sci_r_c = np.load('sci_r_c.npy')

    except:
        sci_b_m, sci_b_c = detect_cosmics(sci_b)
        sci_r_m, sci_r_c = detect_cosmics(sci_r)
        np.save('sci_b_c.npy', sci_b_c)
        np.save('sci_r_c.npy', sci_r_c)


    #imshow(b_c, fig, fig.axes[3], 'blue')
    #imshow(r_c, fig, fig.axes[4], 'red')
    n_b_c, bins_b_c = hist(sci_b_c, fig, fig.axes[0], color='blue')
    n_r_c, bins_r_c = hist(sci_r_c, fig, fig.axes[1], color='red')
    print(frac(n_b_c, bins_b_c))
    print(frac(n_r_c, bins_r_c))

    fig.axes[0].axvline(15, linestyle=':', color='blue')
    fig.axes[1].axvline(14, linestyle=':', color='red')

    plt.show()

if __name__ == '__main__':
    main()
