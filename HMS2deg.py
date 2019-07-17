import argparse

def HMS2deg(ra='', dec=''):
  RA, DEC, rs, ds = '', '', 1, 1
  if dec:
    D, M, S = [float(i) for i in dec.split()]
    if str(D)[0] == '-':
      ds, D = -1, abs(D)
    deg = D + (M/60) + (S/3600)
    DEC = '{0}'.format(deg*ds)

  if ra:
    H, M, S = [float(i) for i in ra.split()]
    if str(H)[0] == '-':
      rs, H = -1, abs(H)
    deg = (H*15) + (M/4) + (S/240)
    RA = '{0}'.format(deg*rs)

  if ra and dec:
    return (RA, DEC)
  else:
    return RA or DEC

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-r', '--ra', type=str, help="Right ascension in hours.")
    p.add_argument('-d', '--dec', type=str, help="Declination in degrees.")
    args = vars(p.parse_args())
    ra, dec = HMS2deg(**args)
    print(ra, dec)

if __name__ == '__main__':
    main()
