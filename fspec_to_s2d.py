# ------------------------------------------------------------------------------
# FSPEC to S2D
# Convert from ESPRESSO DAS FSPEC format to ESPRESSO DRS S2D format
# v1.0 - 2019-04-29
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------

from astropy.io import fits
from astropy.table import Table
import numpy as np
from matplotlib import pyplot as plt

def pad(array, length):
    return np.array([np.pad(a, (0, length-len(a)), 'constant',
                            constant_values=(-9999.0,-9999.0)) \
                     for a in array])

input = []  # Put filenames here (full paths, comma-separated)
output = []  # Put name tags here, e.g. QNNNN+NNNN_setup, one per input filename
flux_col = 'FLUX'
fluxerr_col = 'FLUXERR'
for i, o in zip(input, output):
    print("Converting "+i+" into S2D_"+o+"_n...")

    frame = fits.open(i)
    data = frame[1].data
    specid = np.unique(data['SPECID'])
    for s in specid:
        sel = Table(data[np.where(data['SPECID']==s)])
        sel.sort(('ORDID', 'WAVEL'))
        diff = np.diff(sel['ORDID'])
        diff_sel = np.where(diff>0)[0]
        flux_split = np.split(np.array(sel[flux_col]), diff_sel)
        fluxerr_split = np.split(np.array(sel[fluxerr_col]), diff_sel)
        wavel_split = np.split(np.array(sel['WAVEL'])*10.0, diff_sel)
        diff_max = np.max([len(spl) for spl in flux_split])
        flux = pad(flux_split, diff_max)
        fluxerr = pad(fluxerr_split, diff_max)
        wavel = pad(wavel_split, diff_max)

        hdu0 = fits.PrimaryHDU([], header=frame[0].header)
        hdu0.header['ESO PRO CATG'] = 'S2D_A'
        hdu0.header['ESO PRO TYPE'] = 'REDUCED'
        hdu0.header['ESO PRO REC1 PIPE ID'] = 'espdr'
        hdu1 = fits.ImageHDU(flux)
        hdu2 = fits.ImageHDU(fluxerr)
        hdu3 = fits.ImageHDU(np.empty(np.shape(flux)))
        hdu4 = fits.ImageHDU(wavel)
        hdu5 = fits.ImageHDU(wavel)
        hdul = fits.HDUList([hdu0, hdu1, hdu2, hdu3, hdu4, hdu5])
        name = 'S2D_'+o+'_'+str(s)+'.fits'
        hdul.writeto(name, overwrite=True)
        print(" Created frame "+name+".")
