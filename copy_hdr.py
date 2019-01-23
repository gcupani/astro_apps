# ------------------------------------------------------------------------------
# COPY_HDR
# Copy headers into ESPRESSO frames in case the saving failed at the telescope
# v1.0 - 2019-01-21
# Guido Cupani - INAF-OATs
# ------------------------------------------------------------------------------
# Sample run:
# > python copy_hdr.py -f=/data/espresso/dimarcantonio/QSO_lens/exp1/ESPRESSO_MULTIMR_OBS335_0039.fits -d=/data/espresso/dimarcantonio/QSO_lens/exp1/ESPRESSO_MULTIMR_OBS335_0039_hdr.fits -c=/data/espresso/dimarcantonio/QSO_lens/exp1/TCSHDR_UTn_ESDET_130.tcs
# ------------------------------------------------------------------------------

import argparse
from astropy.io import ascii, fits

extra = {'HIERARCH ESO OCS EM OBJ1 TMMEAN': 0.5}

def run(**kwargs):
    """ Copy headers into ESPRESSO frames """

    path = kwargs['frame']
    tel = kwargs['telescop']
    frame = fits.open(path)
    hdr = open(kwargs['header']).read()
    frame[0].header['TELESCOP'] = tel
    frame[0].header['ARCFILE'] = "ESPRE."+frame[0].header['DATE-OBS']+".fits"
    
    for k in extra.keys():
        frame[0].header[k] = extra[k]

    for i in range(int(len(hdr)/80)):
        line = hdr[i*80: (i+1)*80]
        split = [s.split("/") for s in line.split("=")]
        flat = [i.replace("'", "").strip() for sub in split for i in sub]
        if len(flat) > 1:
            try:
                if "." in flat[1]:
                    frame[0].header[flat[0]] = float(flat[1])
                else:
                    frame[0].header[flat[0]] = int(flat[1])
            except:
                frame[0].header[flat[0]] = flat[1]

    tel_range = list(tel.split('U')[-1])
    for t in tel_range:
        tcs_tel = kwargs['tcs'].replace("UTn", "UT"+t)
        hdr = open(tcs_tel).read()
        for i in range(int(len(hdr)/80)):
            line = hdr[i*80: (i+1)*80]
            split = [s.split("/") for s in line.split("=")]
            flat = [i.replace("'", "").strip() for sub in split for i in sub]
            if len(flat) > 1:
                split_tel = flat[0].replace("TEL", "TEL"+t)
                if "ALPHA" in split_tel \
                   or "DELTA" in split_tel\
                   or "TARG" in split_tel or "HIERARCH" in split_tel:
                    try:
                        if "." in flat[1]:
                            frame[0].header[split_tel] = float(flat[1])
                        else:
                            frame[0].header[split_tel] = int(flat[1])
                    except:
                        frame[0].header[split_tel] = flat[1]

                        
                
    frame.writeto(path[:-5]+'_new_hdr.fits', overwrite=True)    
    print "Header copied into "+path[:-5]+"_new_hdr.fits."

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-f', '--frame', type=str,
                   help="Path to frame.")
    p.add_argument('-d', '--header', type=str,
                   help="Path to header.")
    p.add_argument('-c', '--tcs', type=str,
                   help="Path to the TCSHDR_UTn_ESDET_nnn.tcs files.")
    p.add_argument('-t', '--telescop', type=str, default="ESO-VLT-U1234",
                   help="Value for the TELESCOP keyword.")
    args = vars(p.parse_args())
    run(**args)
    
if __name__ == '__main__':
    main()
