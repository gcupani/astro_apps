from astropy.io import fits
import astroscrappy as ascr
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

sci = 'ESPRE.2019-05-03T23:16:23.402.fits'
#dark = 'ESPRE.2019-04-27T19:16:13.836.fits'
mbias = 'ESPRESSO_master_bias.fits'
dark = 'ESPRE.2019-04-27T18:15:17.053.fits'
flat = 'ESPRE.2019-05-04T13:48:28.947.fits'
hp = 'ESPRESSO_hot_pixels.fits'
bp = 'ESPRESSO_bad_pixels.fits'

targ = 'dark'  # Use 'sci' or 'dark'
xscale = 'log'  # X scale of the plot
thr_1 = 15  # First count threshold for plotting/statistics
thr_2 = 40  # Second count threshold for plotting/statistics
trace = True  # True to restrict to the object trace

def cut(data):
    down = data[:4617,:]
    up = data[4681:,:]
    cut_data = np.vstack((down, up))
    return cut_data

def detect_cosmics(data, inmask=None):
    mask, clean = ascr.detect_cosmics(data, inmask=inmask,
                                      sigclip=2, sigfrac=0.2)
    return mask, clean

def frac(n, bins, thr):
    return np.sum(n[np.where(bins[:-1]>thr)])/np.sum(n)

def hist(data, fig, ax, title=None, **kwargs):
    n, bins, patches = ax.hist(np.ravel(data), bins=20000,
                               range=[-10000,10000], histtype='step', **kwargs)

    if xscale == 'linear':
        ax.set_xlim(-20, 70)
    ax.set_xscale(xscale)
    ax.set_yscale('log')
    ax.set_xlabel('Pixel counts')
    ax.set_ylabel('Number of pixels')
    if title != None:
        ax.set_title(title)
    ax.grid(zorder=0, linestyle=':')
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

    mbias_b, mbias_r = read(mbias)
    if targ == 'sci':
        targ_b, targ_r = read(sci)
    if targ == 'dark':
        targ_b, targ_r = read(dark)
    flat_b, flat_r = read(flat)

    targ_l = [targ_b, targ_r]
    flat_l = [flat_b, flat_r]
    mbias_l = [mbias_b, mbias_r]
    title_l = ['Blue', 'Red']
    lcolor_l = ['deepskyblue', 'lightsalmon']
    dcolor_l = ['blue', 'red']
    suffix_l = ['b', 'r']

    iterator = zip(targ_l, flat_l, mbias_l, fig.axes, title_l, lcolor_l,
                   dcolor_l, suffix_l)
    #iterator = zip([targ_l[0]], [flat_l[0]], [mbias_l[0]], [fig.axes[0]],
    #               [title_l[0]], [lcolor_l[0]], [dcolor_l[0]], [suffix_l[0]])
    for t, f, mb, ax, title, lc, dc, s in iterator:

        data = cut(t)-mb
        if trace:
            mask = cut(f)-mb>2500  # Choose only regions where flat is high
            #plt.imshow(np.log(np.abs(data*(1-mask))))
            #plt.show()
            data = data*mask

        n, bins = hist(data, fig, ax, title=title, color=lc)
        try:
            ciao
            data_m = np.load(targ+'_'+s+'_m.npy')
            data_c = np.load(targ+'_'+s+'_c.npy')
        except:
            data_m, data_c = detect_cosmics(data)
            np.save(targ+'_'+s+'_m.npy', data_m)
            np.save(targ+'_'+s+'_c.npy', data_c)

        #plt.imshow(data_c)
        #plt.show()
        n_c, bins_c = hist(data[~data_m], fig, ax, color=dc)
        n_m, bins_m = hist(data[data_m], fig, ax, color='silver')
        ax.axvline(thr_1, linestyle='--', color=dc)
        ax.axvline(thr_2, linestyle=':', color=dc)
        ax.text(5e2, 4.12e5, "f(>%i) = %2.3f" % (thr_1,
                1-frac(n_c, bins_c, thr_1)/frac(n, bins, thr_1)))
        ax.text(5e2, 1.3e5, "f(>%i) = %2.3f" % (thr_2,
                1-frac(n_c, bins_c, thr_2)/frac(n, bins, thr_2)))

    plt.show()

if __name__ == '__main__':
    main()
