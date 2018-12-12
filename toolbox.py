from astropy import units
from astropy.constants import c
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.modeling.functional_models import Moffat2D
#from astropy.time import Time, TimeDelta
import barycen
import numpy as np

def air_to_vacuum(spec):
    """ Air to vacuum correction 
    http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion
    """
    
    s = 1e3/spec.wave
    n = 1 + 0.00008336624212083 \
        + 0.02408926869968 / (130.1065924522 - s**2) \
        + 0.0001599740894897 / (38.92568793293 - s**2)
    spec.wave = spec.wave*n                
    
def earth_to_bary(spec, site_name='paranal'):
    """ Earth to barycentric frame correction
    https://github.com/janerigby/jrr/blob/master/barycen.py
    """

    site = EarthLocation.of_site(site_name)
    target = SkyCoord(spec.hdr['RA'], spec.hdr['DEC'],
                      unit=(units.hourangle, units.deg), frame='icrs')
    #start_time = Time(spec.hdr['DATE-OBS'], format='isot', scale='utc')
    #midpt = TimeDelta(spec.hdr['EXPTIME'] / 2.0, format='sec')
    #time = start_time + midpt  # time at middle of observation
    barycor_vel = barycen.compute_barycentric_correction(spec.midtime, target,
                                                         location=site)
    spec.wave = spec.wave * (1 + barycor_vel/c).value

def moffat_ee(fwhm=1.0, alpha=3.0, max_aperture=3.0):
    """ Compute the encircled energy for a Moffat profile, given the FWHM """

    # Find the value of Moffat gamma that gives the required FWHM 
    gamma_arr = np.arange(fwhm*0.5, fwhm*2, fwhm*1e-2)
    m_arr = Moffat2D(1.0, 0.0, 0.0, gamma_arr, alpha)
    gamma = np.interp(fwhm, m_arr.fwhm, gamma_arr)

    # Create a Moffat profile with the required FWHM
    m = Moffat2D(1.0, 0.0, 0.0, gamma, alpha)
    grid = np.arange(0, max_aperture, 1e-2)
    grid_ext = np.arange(0, max_aperture*10, 1e-2)
    x, y = np.meshgrid(grid, grid)
    x_ext, y_ext = np.meshgrid(grid_ext, grid_ext)
    m.x = x
    m.y = y
    m.z = m(x, y)
    zsum = np.sum(m(x_ext, y_ext))

    # Compute the encircled energy
    m.ee = []
    m.rad = np.arange(0, max_aperture, 1e-2)
    for r in m.rad:
        where = x**2 + y**2 < r**2
        m.zin = np.zeros((len(x), len(y)))
        m.zin[where] = m.z[where]
    
        # Compute the encircled energy
        m.ee.append(np.sum(m.zin)/zsum)
    return m

    
    
    
