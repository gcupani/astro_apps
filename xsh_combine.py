# ------------------------------------------------------------------------------
# XSH_COMBINE
# Combine X-shooter spectra
# v1.0 - 2018-12-06
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

def run(**kwargs):
    
    # Load parameters
    frames = np.array(ascii.read(kwargs['framelist'],
                                 format='no_header')['col1'])

    first = True
    fig = Figure()
    ax = plt.gca()
    for f in frames:
        print "Processing", f+"..."
        dim = f[-11:-9]
        arm = f[-8:-5]
        name = f.split('/')[2][:6]
        
        # Load spectrum
        spec = XshMerge(f)

        # Correct wavelengths from air to vacuum
        air_to_vacuum(spec)
        spec.save(f[:-5]+'_VAC.fits', dim)

        # Correct wavelengths from Earth to barycentric frame
        earth_to_bary(spec)
        spec.save(f[:-5]+'_VAC_BARY.fits', dim)

        if dim == '1D':
            spec_arr = np.array([spec.wave, spec.flux, spec.err])
            if arm == 'UVB': 
                naxis1_app = np.append(0, spec.naxis1)
                spec_arr_app = spec_arr
            else:
                naxis1_app = np.append(naxis1_app, naxis1_app[-1]+spec.naxis1)
                spec_arr_app = np.append(spec_arr_app, spec_arr, axis=1)

            if arm == 'NIR':
                if first:
                    flux_stack = spec_arr_app[1]
                    err_stack = spec_arr_app[2]
                    first = False
                else:
                    flux_stack = np.vstack((flux_stack, spec_arr_app[1]))
                    err_stack = np.vstack((err_stack, spec_arr_app[2]))
                line, = ax.plot(spec_arr_app[0],
                                spec_arr_app[1]-int(name[5])*1e-17,
                                linewidth=1.0, label=f[:6])

    flux_ave = np.average(flux_stack, axis=0, weights=1/err_stack**2)
    err_ave = np.sqrt(
        np.average(err_stack**2, axis=0, weights=1/err_stack**2)\
        / np.shape(err_stack)[0])
    arm = ['UVB', 'VIS', 'NIR']
    for start, end, a in zip(naxis1_app[:-1], naxis1_app[1:], arm):
        save('ION2_ALL_FLUX_MERGE1D_%s_VAC.fits' % a, spec.hdr,
             spec_arr_app[0][start:end], flux_ave[start:end],
             err_ave[start:end])
        
    ax.plot(spec_arr_app[0], flux_ave, c='black', label='Weighted average')
    ax.plot(spec_arr_app[0], flux_ave+err_ave, c='black', linewidth=0.5)
    ax.plot(spec_arr_app[0], flux_ave-err_ave, c='black', linewidth=0.5)
    ax.legend()
    plt.show()
                    
def main():
    """ Read the CL parameters and run """

    p = argparse.ArgumentParser()
    p.add_argument('-l', '--framelist', type=str, default='frame_list.dat',
                   help="List of frames; must be an ascii file with a column "
                   "of entries. It's mandatory that frames are grouped in "
                   "UVB/VIS/NIR sets (in this order) and that at least 2 "
                   "groups are present.")

    args = vars(p.parse_args())
    run(**args)

if __name__ == '__main__':
    main()
