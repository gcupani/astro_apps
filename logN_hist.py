from astropy.io import ascii, fits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import interp2d


def hist2d(h, bins, title=None, name=None):
    fig, ax = plt.subplots()
    """
    h = ax.hist2d(x, y, bins=bins)
    plt.colorbar(h[3], ax=ax)
    """
    x, y = np.meshgrid(bins[0], bins[1])
    im = ax.pcolormesh(x, y, h.T)
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\log N$')
    if name is not None:
        plt.savefig(name)
    plt.draw()

list = ascii.read('CIV_colden/CIV-qso_data.csv')

z_step = 0.05
logN_step = 0.2

z_r = np.arange(1.00-z_step/2, 4.00+z_step/2, z_step)
z_interp = z_r[:-1]+z_step/2
logN_r = np.arange(10-logN_step/2, 15.5+logN_step/2, logN_step)
logN_interp = logN_r[:-1]+logN_step/2

succ = 0
for i, l in enumerate(list):
    name = l['name'].strip()+'_2019-07-18'
    zmin = l['zmin']
    zmax = l['zmax']
    try:
        print("Adding %s (zmin=%f, zmax=%f)..." % (name, zmin, zmax), end=" ")
        merge = ascii.read(name + '_merge.dat')
        compl = ascii.read(name + '_compl.dat')
        corr = ascii.read(name + '_corr.dat')

        z = np.array(merge['col1'])
        logN = np.array(merge['col2'])

        # dn
        dn_i, _, _ = np.histogram2d(z, logN, bins=[z_r, logN_r])

        # dz
        dz_i = np.zeros(np.shape(dn_i))
        z_sel = np.where(np.logical_and(z_interp>zmin+z_step,
                                        z_interp<zmax-z_step))[0]
        z_sel = np.tile(z_sel, (len(logN_r), 1))
        dz_i[z_sel] = 1
        z_sel = np.where(np.logical_and(z_interp>zmin,
                                        z_interp<zmin+z_step))[0]
        frac = (z_interp[z_sel]-zmin)/z_step
        z_sel = np.tile(z_sel, (len(logN_r), 1))
        dz_i[z_sel] = frac
        z_sel = np.where(np.logical_and(z_interp>zmax-z_step,
                                        z_interp<zmax))[0]
        frac = (zmax-z_interp[z_sel])/z_step
        z_sel = np.tile(z_sel, (len(logN_r), 1))
        dz_i[z_sel] = frac

        # Completeness
        z_compl = np.array(compl['col1'])
        logN_compl = np.array(compl['col2'])
        r_compl = np.array(compl['col4'])

        ### Some points in the table have strange values of z or completeness
        compl_sel = np.where(np.logical_and(
            np.logical_and(z_compl>1, z_compl<6), r_compl<=1))
        z_compl = z_compl[compl_sel]
        logN_compl = logN_compl[compl_sel]
        r_compl = r_compl[compl_sel]
        z_digit = z_interp[np.digitize(z_compl, z_r)-1]

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(z_compl, logN_compl, r_compl)
        ax.scatter(z_digit, logN_compl, r_compl)
        plt.draw()
        """

        ### Interpolate
        z_digit = z_interp[np.digitize(z_compl, z_r)-1]
        z_u = np.unique(z_digit)
        logN_u = np.unique(logN_compl)
        r_digit = np.zeros((len(z_u), len(logN_u)))
        c_digit = np.zeros((len(z_u), len(logN_u)))
        for zi in range(len(z_u)):
            for logNi in range(len(logN_u)):
                sel = np.where(np.logical_and(z_digit == z_u[zi],
                                              logN_compl == logN_u[logNi]))[0]
                if len(sel) > 0:
                    r_digit[zi, logNi] = np.mean(r_compl[sel])
                    c_digit[zi, logNi] += 1
        c_digit[c_digit == 0] = 1
        r_digit = r_digit/c_digit
        f = interp2d(z_u, logN_u, r_digit.T, kind='linear')#, fill_value=0)
        r_compli = f(z_interp, logN_interp).T

        ### Cut
        r_compli[r_compli < 0.5] = 0

        ### Plot
        #hist2d(dz_i * r_compli, [z_r, logN_r],
        #       title=r'$dz$'+'\nnormalized by completeness')

        # Correctness
        logN_corr = np.sort(corr['col1'])
        r_corr = np.array([max(1-c4/c3, 0.0)
                           for (c4, c3) in zip(corr['col4'], corr['col3'])])
        r_corr[np.isnan(r_corr)] = 1
        r_corr = r_corr[np.argsort(logN_corr)]

        ### Interpolate
        r_corri = np.interp(logN_interp, logN_corr, r_corr)
        r_corri[logN_interp>np.max(logN_corr)] = 1
        r_corri = np.tile(r_corri, (len(z_interp), 1))

        ### Cut
        r_corri[r_corri < 0.5] = 0

        ### Plot
        #hist2d(r_corri, [z_r, logN_r], title='Correctness')

        if i == 0:
            dn = dn_i
            dn_corr = dn_i * r_corri
            dz = dz_i
            dz_compl = dz_i * r_compli
        else:
            dn += dn_i
            dn_corr += dn_i * r_corri
            dz += dz_i
            dz_compl += dz_i * r_compli

        succ += 1
        print("success!")
    except:
        print("not found!")

print("%i objects added." % succ)
hist2d(dn, [z_r, logN_r], title=r'$dn$', name='dn.png')
hist2d(dn_corr, [z_r, logN_r], title=r'$dn$'+'\nnormalized by correctness',
       name='dn_corr.png')
hist2d(dz, [z_r, logN_r], title=r'$dz$', name='dz.png')
hist2d(dz_compl, [z_r, logN_r], title=r'$dz$'+'\nnormalized by completeness',
       name='dz_compl.png')
hist2d(dn/dz, [z_r, logN_r], title=r'$dn/dz$', name='dn_dz.png')
hist2d(dn_corr/dz_compl, [z_r, logN_r],
       title=r'$dn/dz$'+'\nnormalized by correctness and completeness',
       name='dn_dz_corr_compl.png')

plt.show()
