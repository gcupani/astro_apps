# ------------------------------------------------------------------------------
# UVES to ESPRESSO
# Convert from UVES 2D frames to ESPRESSO 2D frames (suitable for DAS)
# v2.0 - 2019-04-29
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------

from astropy.io import fits
import numpy as np

names = []  # Put filenames here (full paths, comma-separated)
z_em = 0.0
for n in names:
    for c in ['BLUE', 'REDL', 'REDU']:

        print("Reformatting "+n+", "+c+"...")

        flux_frame = fits.open(n+'_WCALIB_FF_SCI_POINT_'+c+'.fits')
        err_frame = fits.open(n+'_ERRORBAR_WCALIB_FF_SCI_POINT_'+c+'.fits')
        flux = flux_frame[0].data
        err = err_frame[0].data
        wave = np.empty(np.shape(flux))
        obj = flux_frame[0].header['OBJECT']
        date = flux_frame[0].header['DATE-OBS']
        try:
            wlen = flux_frame[0].header['ESO INS GRAT1 WLEN']
        except:
            wlen = flux_frame[0].header['ESO INS GRAT2 WLEN']
        naxis1 = flux_frame[0].header['NAXIS1']
        naxis2 = flux_frame[0].header['NAXIS2']
        step = flux_frame[0].header['CDELT1']*0.1
        for i in range(naxis2):
            start = flux_frame[0].header['WSTART'+str(i+1)]*0.1
            end = start+naxis1*step
            wave[i,:] = np.arange(start,end,step)[:naxis1]

        flux_out = fits.HDUList(
            [fits.PrimaryHDU(flux, header=flux_frame[0].header)])
        err_out  = fits.HDUList(
            [fits.PrimaryHDU(err, header=err_frame[0].header)])
        wave_out = fits.HDUList(
            [fits.PrimaryHDU(wave, header=flux_frame[0].header)])

        flux_out[0].header['ESO PRO CATG'] = "SPEC_FLUX_2D"
        err_out[0].header['ESO PRO CATG'] = "SPEC_FLUXERR_2D"
        wave_out[0].header['ESO PRO CATG'] = "WAVE_MATRIX_FIBER"
        for f in [flux_out, err_out, wave_out]:
            f[0].header['ESO PRO TYPE'] = "REDUCED"
            f[0].header.insert('HIERARCH ESO PRO TYPE',
                ('HIERARCH ESO OCS OBJ Z_EM', z_em, 'QSO emission redshift'),
                after = True)
        name_out = obj+'_'+date+'_'+wlen
        flux_out.writeto(name_out+'_flux_reformat.fits', overwrite=True)
        err_out.writeto(name_out+'_err_reformat.fits',  overwrite=True)
        wave_out.writeto(name_out+'_wave_reformat.fits', overwrite=True)
