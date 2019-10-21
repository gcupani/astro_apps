from astropy import units as au
import numpy as np

# See http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
wave_U = 360 * au.nm  # Effective wavelength, U band
wave_u = 356 * au.nm  # Effective wavelength, u band
flux_U = 7561 * au.photon / au.cm**2 / au.s / au.nm  # Flux density @ 360.0 nm, mag_U = 0 (Vega)
flux_u = 15393 * au.photon / au.cm**2 / au.s / au.nm  # Flux density @ 356.0 nm, mag_u = 0 (AB)

eff_adc = [0.96, 0.96, 0.96]  # ADC efficiency
eff_slc = [0.98, 0.98, 0.98]  # Slicer efficiency
eff_dch = [0.96, 0.96, 0.96]  # Dichroics efficiency
eff_spc = [0.93, 0.94, 0.93]  # Spectrograph efficiency
eff_grt = [0.90, 0.90, 0.90]  # Grating efficiency
eff_ccd = [0.85, 0.85, 0.85]  # CCD QE
eff_tel = [0.72, 0.72, 0.72]  # Telescope efficiency
resol = [2.0e4, 2.1e4, 2.2e4]  # Instrument resolution

ccd_xsize = 4096*au.pixel  # X size of the CCD
ccd_ysize = 4096*au.pixel  # Y size of the CCD
ccd_xbin = 8  # X binning of the CCD
ccd_ybin = 16  # Y binning of the CCD
pix_xsize = 15*au.micron  # X size of the pixels
pix_ysize = 15*au.micron  # Y size of the pixels
spat_scale = 0.25*au.arcsec/(30*au.micron)  # Spatial scale
slice_n = 6  # Number of slices
slice_length = 10*au.arcsec  # Length of the slice
slice_width = 0.25*au.arcsec  # Width of the slice
slice_gap = 40*au.pixel  # Length of the slice
ccd_bias = 100*au.adu
ccd_ron = 2*au.adu
ccd_gain = 1.1*au.photon/au.adu
ccd_dark = 0.5*au.adu/au.h

seeing = 1.0*au.arcsec  # Seeing
psf_func = 'gaussian'  # Function to represent the PSF ('tophat', 'gaussian', 'moffat')
psf_sampl = 1000*au.pixel  # Size of the PSF image
psf_cen = (0,0)  # Center of the PSF
area = (400*au.cm)**2 * np.pi  # Telescope area
texp = 600*au.s  # Exposure time
mag_syst = 'AB'  # Magnitude system
targ_mag = 11.15  # Magnitude of the target @ 350 nm
bckg_mag = 22.5  # Magnitude of the background @ 350 nm
airmass = 1.0  # Airmass

spec_func = 'qso'  # Function for the template spectrum ('flat', 'PL', 'qso', 'star')
qso_file = 'J1124-1705.fits'
star_file = 'Castelliap000T5250g45.dat'
extr_func = 'sum'  # Function for extracting the spectrum ('sum', 'opt' [very slow])
snr_sampl = 50  # Data points per SNR point
