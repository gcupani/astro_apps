from astropy import units as u
from astropy.constants import c, h
from astropy.io import ascii, fits
import numpy as np

ut_area = (4*u.m)**2 * np.pi

class ESOExt(object):
    """ Class for ESO extinction spectra """

    def __init__(self, name='atmoexan.fits', area=ut_area, expt=3600*u.s):
        hdul = fits.open(name)
        self.wave = hdul[1].data['LAMBDA']*0.1 * u.nm
        self.la_silla = hdul[1].data['LA_SILLA']
        

class ESOStd(object):
    """ Class for ESO standard star catalogue spectra """

    def __init__(self, name, area=ut_area, expt=3600*u.s):
        """ Load a spectrum """
    
        data = ascii.read(name)
        self.wave = data['col1'] * 0.1 * u.nm
        self.dwave = data['col4'] * 0.1 * u.nm
        self.irr = data['col2'] * 1e-16 * u.erg/u.cm**2/u.s/u.angstrom
        self.flux = self.irr / (h*c/self.wave) * area * expt * self.dwave

        
class EsprSpec(object):
    """ Class for generic ESPRESSO spectra """

    def __init__(self, name):
        self.hdul = fits.open(name)
        hdr = self.hdul[0].header
        self.expt = hdr['EXPTIME'] * u.s
        self.targ = hdr['ESO OBS TARG NAME']
        self.binx = hdr['ESO DET BINX']
        self.biny = hdr['ESO DET BINY']
        self.ut = hdr['TELESCOP'][-1]

        ins_mode = hdr['ESO INS MODE']
        if ins_mode == 'SINGLEUHR':
            self.mode = ins_mode
        else:
            self.mode = ins_mode+str(self.binx)+str(self.biny)

        airm_start = hdr['ESO TEL'+str(self.ut)+' AIRM START']
        airm_end = hdr['ESO TEL'+str(self.ut)+' AIRM END']
        self.airm = 0.5 * (airm_start+airm_end)

        nut = hdr['ESO OCS TEL NO']
        self.area = nut * ut_area

        
class EsprEff(EsprSpec):
    """ Class for ESPRESSO efficiency spectra """
        
    def __init__(self, name):
        super(EsprEff, self).__init__(name) 

        self.wave = self.hdul[1].data['wavelength']*0.1 * u.nm
        self.interp = self.hdul[1].data['efficiency_interpolated']

    
class EsprS1D(EsprSpec):
    """ Class for ESPRESSO S1D spectra """

    def __init__(self, name):
        super(EsprS1D, self).__init__(name) 

        self.wave = self.hdul[1].data['wavelength']*0.1 * u.nm
        self.flux = self.hdul[1].data['flux']
        self.err = self.hdul[1].data['error']

