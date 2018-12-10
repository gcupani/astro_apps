from astropy import units as u
from astropy.constants import c, h
from astropy.io import ascii, fits
from astropy.time import Time, TimeDelta
import numpy as np

ut_area = (4*u.m)**2 * np.pi

def save(name, hdr, wave, flux, err, dim='1D'):
    if dim == '1D':
        wavec = fits.Column(name='wave', array=wave, format='F')
        fluxc = fits.Column(name='flux', array=flux, format='E')
        errc = fits.Column(name='err', array=err, format='E')
        hdu = fits.BinTableHDU.from_columns([wavec, fluxc, errc])
        hdu1 = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu1, hdu])
    else:
        hdu1 = fits.PrimaryHDU(wave, header=hdr)
        hdu2 = fits.ImageHDU(flux)
        hdu3 = fits.ImageHDU(err)
        hdul = fits.HDUList([hdu1, hdu2, hdu3])
    hdul.writeto(name, overwrite=True)

class AppSpec(object):
    """ Class for generic App spectra """

    def __init__(self, name):
        pass

class ESOExt(object):
    """ Class for ESO extinction spectra """

    def __init__(self, name='atmoexan.fits', area=ut_area, expt=3600*u.s):
        hdul = fits.open(name)
        self.wave = hdul[1].data['LAMBDA']*0.1 * u.nm
        self.la_silla = hdul[1].data['LA_SILLA']
        
class ESOSpec(object):
    """ Class for ESO science spectra """
    
    def __init__(self, name): 
        self.hdul = fits.open(name)
        self.hdr = self.hdul[0].header
        self.expt = self.hdr['EXPTIME'] * u.s
        self.targ = self.hdr['ESO OBS TARG NAME']
        self.ut = self.hdr['TELESCOP'][-1]
        self.date = self.hdr['DATE-OBS']
        start = Time(self.date, format='isot', scale='utc')
        mid = TimeDelta(self.expt / 2.0, format='sec')
        self.midtime = start + mid

    def save(self, name, dim='1D'):
        save(name, self.hdr, self.wave, self.flux, self.err, dim)
        
class ESOStd(object):
    """ Class for ESO standard star catalogue spectra """

    def __init__(self, name, area=ut_area, expt=3600*u.s):
        """ Load a spectrum """
    
        data = ascii.read(name)
        self.wave = data['col1'] * 0.1 * u.nm
        self.dwave = data['col4'] * 0.1 * u.nm
        self.irr = data['col2'] * 1e-16 * u.erg/u.cm**2/u.s/u.angstrom
        self.flux = self.irr / (h*c/self.wave) * area * expt * self.dwave

class EsprSpec(ESOSpec):
    """ Class for generic ESPRESSO spectra """

    def __init__(self, name):
        super(EsprSpec, self).__init__(name) 

        self.binx = self.hdr['ESO DET BINX']
        self.biny = self.hdr['ESO DET BINY']

        ins_mode = self.hdr['ESO INS MODE']
        if ins_mode == 'SINGLEUHR':
            self.mode = ins_mode
        else:
            self.mode = ins_mode+str(self.binx)+str(self.biny)

        airm_start = self.hdr['ESO TEL'+str(self.ut)+' AIRM START']
        airm_end = self.hdr['ESO TEL'+str(self.ut)+' AIRM END']
        self.airm = 0.5 * (airm_start+airm_end)

        nut = self.hdr['ESO OCS TEL NO']
        self.area = nut * ut_area

        self.ia_fwhmlin = self.hdr['ESO TEL'+str(self.ut)+' IA FWHMLIN']
        dimm_start = self.hdr['ESO TEL'+str(self.ut)+' AMBI FWHM START']
        dimm_end = self.hdr['ESO TEL'+str(self.ut)+' AMBI FWHM END']
        self.dimm = 0.5 * (dimm_start+dimm_end)
        
class EsprEff(EsprSpec):
    """ Class for ESPRESSO ABS_EFF_RAW_A spectra """
        
    def __init__(self, name):
        super(EsprEff, self).__init__(name) 

        self.wave = self.hdul[1].data['wavelength']*0.1 * u.nm
        self.eff = self.hdul[1].data['efficiency']
        """
        try:
            self.eff = self.hdul[1].data['raw_efficiency']
        except:
            try:
                self.eff = self.hdul[1].data['efficiency']
            except:
                try:
                    self.eff = self.hdul[1].data['efficiency_interpolated']
                except:
                    pass
        """
    
class EsprS1D(EsprSpec):
    """ Class for ESPRESSO S1D spectra """

    def __init__(self, name):
        super(EsprS1D, self).__init__(name) 

        self.wave = self.hdul[1].data['wavelength']*0.1 * u.nm
        self.flux = self.hdul[1].data['flux'] * u.adu
        self.err = self.hdul[1].data['error']

    def adu_to_electron(self):
        conad = {'blue': 1.09, 'red': 1.12}
        blue = np.where(self.wave.value < 521.5)
        red = np.where(self.wave.value > 521.5)
        self.flux[blue] = self.flux[blue] * conad['blue']
        self.flux[red] = self.flux[red] * conad['red']


class XshSpec(ESOSpec):
    """ Class for generic X-shooter spectra """
    
    def __init__(self, name):
        super(XshSpec, self).__init__(name) 

        self.crval1 = self.hdr['CRVAL1']
        self.cdelt1 = self.hdr['CDELT1']
        self.naxis1 = self.hdr['NAXIS1']
    
class XshMerge(XshSpec):
    """ Class for X-shooter MERGE spectra """
    
    def __init__(self, name):
        super(XshMerge, self).__init__(name) 
    
        self.wave = np.arange(self.crval1, self.crval1+self.naxis1*self.cdelt1,
                              self.cdelt1)
        self.flux = self.hdul[0].data
        self.err = self.hdul[1].data
        #if len(self.wave) > len(self.flux):
        self.wave = self.wave[:len(self.flux)]
        
