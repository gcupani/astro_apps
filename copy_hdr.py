# ------------------------------------------------------------------------------
# COPY_HDR
# Copy headers into ESPRESSO frames in case the saving failed at the telescope
# v1.0 - 2019-01-21
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------
# Sample run:
# > python copy_hdr.py -f=/data/espresso/dimarcantonio/QSO_lens/exp1/ESPRESSO_MULTIMR_OBS335_0039.fits -d=/data/espresso/dimarcantonio/QSO_lens/exp1/ESPRESSO_MULTIMR_OBS335_0039_hdr.fits
# ------------------------------------------------------------------------------

import argparse
from astropy.io import ascii, fits

def run(**kwargs):
    """ Copy headers into ESPRESSO frames """

    path = kwargs['frame']
    frame = fits.open(path)
    hdr = open(kwargs['header']).read()
    for i in range(int(len(hdr)/80)):
        line = hdr[i*80: (i+1)*80]
        split = [s.split("/") for s in line.split("=")]
        flat = [i.replace("'", "").strip() for sub in split for i in sub]
        if len(flat) == 3:
            try:
                if "." in flat[1]:
                    frame[0].header[flat[0]] = float(flat[1])
                else:
                    frame[0].header[flat[0]] = int(flat[1])
            except:
                frame[0].header[flat[0]] = flat[1]
    frame.writeto(path[:-5]+'_new_hdr.fits', overwrite=True)
    print "Header copied into "+path[:-5]+"_new_hdr.fits."

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--frame', type=str,
                   help="Path to frame.")
    p.add_argument('-d', '--header', type=str,
                   help="Path to header.")
    args = vars(p.parse_args())
    run(**args)
    
if __name__ == '__main__':
    main()
