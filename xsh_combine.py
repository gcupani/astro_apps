# ------------------------------------------------------------------------------
# XSH_COMBINE
# Combine X-shooter spectra
# v2.0 - 2018-12-06
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------
# Sample run:
# > python xsh_combine.py -h         // Help
# > python $APP_PATH/xsh_combine.py -l=xsh_combine_list.dat
# ------------------------------------------------------------------------------

import argparse
from astropy.io import ascii
from formats import XshMerge, save
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from toolbox import air_to_vacuum, earth_to_bary

class ArmStack(object):
    pass
    

def ave(data, wgh):
    if len(np.shape(data)) == 1:
        return data
    if len(np.shape(data)) == 2:
        return np.average(data, axis=0, weights=wgh)

def run(**kwargs):
    
    # Load parameters
    frames = np.array(ascii.read(kwargs['framelist'],
                                 format='no_header')['col1'])
    name = kwargs['name']

    first = True
    fig = Figure()
    ax = plt.gca()

    wave_stack = ArmStack()
    flux_stack = ArmStack()
    err_stack = ArmStack()
    
    for f in frames:
        print("Processing", f+"...")
        dim = f[-11:-9]
        arm = f[-8:-5]
        
        # Load spectrum
        spec = XshMerge(f)

        # Correct wavelengths from air to vacuum
        air_to_vacuum(spec)
        spec.save(f[:-5]+'_VAC.fits', dim)

        # Correct wavelengths from Earth to barycentric frame
        earth_to_bary(spec)
        spec.save(f[:-5]+'_VAC_BARY.fits', dim)

        if dim == '1D':
            #spec.err[spec.err == 0] = -9999.*1e-15
            if hasattr(flux_stack, arm):
                setattr(flux_stack, arm,
                        np.vstack((getattr(flux_stack, arm), spec.flux)))
                setattr(err_stack, arm,
                        np.vstack((getattr(err_stack, arm), spec.err)))
            else:
                setattr(wave_stack, arm, spec.wave)
                setattr(flux_stack, arm, spec.flux)
                setattr(err_stack, arm, spec.err)


    flux_ave = ArmStack()
    err_ave = ArmStack()
    spec_app = None
    for arm in ['UVB', 'VIS', 'NIR']:
        if hasattr(flux_stack, arm):
            wave = getattr(wave_stack, arm)
            flux = getattr(flux_stack, arm)
            err = getattr(err_stack, arm)
            setattr(flux_ave, arm, ave(flux, wgh=1/err**2))
            setattr(err_ave, arm,
                    np.sqrt(ave(err**2, wgh=1/err**2)/np.shape(err)[0]))
            spec_arr = np.array([wave, getattr(flux_ave, arm),
                             getattr(err_ave, arm)])
            save(name+'_%s_VAC_BARY_COMB.fits' % arm, spec.hdr, wave,
                 getattr(flux_ave, arm), getattr(err_ave, arm))
            if spec_app is not None:
                spec_app = np.append(spec_app, spec_arr, axis=1)
            else:
                spec_app = spec_arr

                
    ax.plot(spec_app[0], spec_app[1], c='black', label='Weighted average')
    ax.plot(wave_stack.UVB, flux_ave.UVB+err_ave.UVB, c='black', linewidth=0.5)
    ax.plot(wave_stack.UVB, flux_ave.UVB-err_ave.UVB, c='black', linewidth=0.5)
    ax.legend()
    plt.show()


def main():
    """ Read the CL parameters and run """

    p = argparse.ArgumentParser()
    p.add_argument('-l', '--framelist', type=str,
                   default='xsh_combine_list.dat',
                   help="List of frames; must be an ascii file with a column "
                   "of entries. It's mandatory that frames are grouped in "
                   "UVB/VIS/NIR sets (in this order) and that at least 2 "
                   "groups are present.")
    p.add_argument('-n', '--name', type=str,
                   default='noname', help="Name prefix for the output frames")

    args = vars(p.parse_args())
    run(**args)

if __name__ == '__main__':
    main()
