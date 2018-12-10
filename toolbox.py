from astropy import units
from astropy.constants import c
from astropy.coordinates import SkyCoord, EarthLocation
#from astropy.time import Time, TimeDelta
import barycen

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
