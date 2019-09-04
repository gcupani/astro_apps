from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.ndimage import gaussian_filter1d as gauss1d
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors.nearest_centroid import NearestCentroid as NC


frame = fits.open('/Users/guido/GitHub/astrocook/test_data/J0003-2323.fits')
data = frame[1].data
wave = data['WAVE']
orflux = data['FLUX']
orerr = data['sigma']
r = range(100, len(orflux)-100)
r = range(10000, 12000)

flux = gauss1d(orflux, 0)
err = gauss1d(orerr, 0)

hw = 10
rms = np.array([np.sqrt(np.mean((flux[i-hw:i+hw]-np.mean(flux[i-hw:i+hw]))**2))
                for i in range(hw, len(flux)-hw)])
rms = np.concatenate((np.full(hw, np.nan), rms, np.full(hw, np.nan)))

dflux = np.gradient(flux)#np.ediff1d(flux)
d2flux = np.gradient(dflux)
d3flux = np.gradient(d2flux)

ddflux = dflux[5:]-dflux[:-5]

s = np.where(flux[r]<np.max(flux[r])+1*np.std(flux[r]))[0]


#"""
#x = np.stack((dflux[r][s], d2flux[r][s], d3flux[r][s]), axis=1)
x = np.stack((wave[r][s], flux[r][s]), axis=1)
#x = np.stack((flux[r][s], ddflux[r][s]), axis=1)
x = np.stack((flux[r][s], err[r][s], rms[r][s]), axis=1)
xw = np.stack((wave[r][s], flux[r][s], flux[r][s]), axis=1)

#db = DBSCAN(eps=0.2, min_samples=10).fit(x)
db = DBSCAN(eps=0.02, min_samples=10).fit(x)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

w = np.where(np.logical_and(np.abs(dflux[r])<5e-3, d2flux[r]>0))

if n_clusters_ > 0:

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    #ax10 = fig1.add_subplot(221)
    #ax11 = fig1.add_subplot(222)
    #ax12 = fig1.add_subplot(223)
    """
    ax10.scatter(dflux[r], d2flux[r])
    ax10.scatter(dflux[r][w], d2flux[r][w])
    ax11.scatter(d3flux[r], d2flux[r])
    ax11.scatter(d3flux[r][w], d2flux[r][w])
    ax12.scatter(dflux[r], d3flux[r])
    ax12.scatter(dflux[r][w], d3flux[r][w])
    """
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(wave[r], orflux[r])
    ax2.plot(wave[r][s], flux[r][s])
    #ax2.scatter(wave[r][w], orflux[r][w])
    #ax2.scatter(wave[r][w], flux[r][w])
    #ax2.plot(wave[r][s], dflux[r][s])
    #ax2.plot(wave[r][s], d2flux[r][s])
    #ax2.plot(wave[r][s], d3flux[r][s])
    for i, (k, col) in enumerate(zip(unique_labels, colors)):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
            size = 2
        else:
            size = 5
        class_member_mask = (labels == k)
        s = x[class_member_mask] #& core_samples_mask]
        sw = xw[class_member_mask] #& core_samples_mask]
        ax1.plot(s[:,0], s[:,1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor=tuple(col), markersize=size)
        """
        ax10.plot(s[:,0], s[:,1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)
        ax11.plot(s[:,2], s[:,1], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)
        ax12.plot(s[:,0], s[:,2], 'o', markerfacecolor=tuple(col),
                markeredgecolor='k', markersize=14)
        """
        if k == -1 or True:
            ax2.plot(sw[:,0], sw[:,1], 'o', markerfacecolor=tuple(col),
                    markeredgecolor=tuple(col), markersize=size)

plt.show()
