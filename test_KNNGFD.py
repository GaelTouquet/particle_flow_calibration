#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does KNNGFD works.
"""

import matplotlib.pyplot as plt
import numpy as np
from tools import importPickle, savefig
from tools import gaussian_fit, gaussian_param, optimized_binwidth
from sklearn import neighbors
import math


font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
fontgreen = {'family': 'serif',
        'color':  'green',
        'weight': 'normal',
        'size': 16,
        }


directory = "pictures/testKNNGFD/"


filename = 'charged_hadrons_100k.energydata'
data1 = importPickle(filename)
filename = 'prod2_200_400k.energydata'
#on fusionne les 2 jeux de données
data1 = data1.mergeWith(importPickle(filename))
#on sépare data en 2
data1,data2 = data1.splitInTwo()

# paramètres de calibration
lim = 150
n_neighbors = 250

def getMeans(energy_x,y):
    ind  = np.invert(np.isnan(y))
    y = y[ind]
    energy_x = energy_x[ind]

    neighborhood = neighbors.NearestNeighbors(n_neighbors=500)
    neighborhood.fit(np.transpose(np.matrix(energy_x)))
    step = 0.5
    ener = np.arange(min(energy_x),max(energy_x),step)
    sigma_gaussianfit = []
    mean_gaussianfit = []
    means = []
    energy = []
    reducedChi2 = []
    for e in ener:
        dist, ind = neighborhood.kneighbors(X = e)
        y_ind = y[ind][np.invert(np.isnan(y[ind]))]
        params,reduced = gaussian_fit(y_ind,binwidth = 0.1, giveChi2 = True)
        if not(math.isnan(params[1])):
            means.append(np.mean(y_ind))
            sigma_gaussianfit.append(params[0])
            mean_gaussianfit.append(params[1])
            energy.append(e)
            reducedChi2.append(reduced)
    return energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2




KNNGFD = data1.kNNGaussianFitDirect(n_neighbors=n_neighbors,lim=lim)

#courbe de calibration pour ecal = 0
hcal_train = KNNGFD.hcal_train_ecal_eq_0
true_train = KNNGFD.true_train_ecal_eq_0
h = np.arange(min(hcal_train),lim,0.1)
e = np.zeros(len(h))
t = KNNGFD.predict(e,h)

fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(hcal_train,true_train,'.',markersize=1)
plt.plot(h,t,lw=2)
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.ylabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.axis([0,max(hcal_train),0,max(true_train)])

plt.subplot(1,2,2)
plt.plot(hcal_train,true_train,'.',markersize=1)
plt.plot(h,t,lw=2)
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.ylabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.axis([0,15,0,15])
plt.show()
savefig(fig,directory,"calibration.png")

#ecalib/etrue pour ecal = 0
h = data2.hcal[np.logical_and(data2.ecal == 0,data2.ecal+data2.hcal < lim)]
t = data2.true[np.logical_and(data2.ecal == 0,data2.ecal+data2.hcal < lim)]
e = np.zeros(len(h))
c = KNNGFD.predict(e,h)
r = c/t

energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2 = getMeans(t,r)
fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(t,r,'.',markersize=1)
plt.axis([0,200,0,2])
plt.plot(energy,mean_gaussianfit,lw=3)
plt.xlabel(r"$e_{true}$",fontsize=15)
plt.ylabel(r"$e_{calib}/e_{true}$",fontsize=15)
plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=15)

h2 = data2.hcal[np.logical_and(data2.ecal != 0,data2.ecal+data2.hcal < lim)]
t2 = data2.true[np.logical_and(data2.ecal != 0,data2.ecal+data2.hcal < lim)]
e2 = data2.ecal[np.logical_and(data2.ecal != 0,data2.ecal+data2.hcal < lim)]
c2 = KNNGFD.predict(e2,h2)
r2 = c2/t2

energy2, means2, mean_gaussianfit2, sigma_gaussianfit2, reducedChi22 = getMeans(t2,r2)
plt.subplot(1,2,2)
plt.plot(t2,r2,'.',markersize=1)
plt.axis([0,200,0,2])
plt.plot(energy2,mean_gaussianfit2,lw=3)
plt.xlabel(r"$e_{true}$",fontsize=15)
plt.ylabel(r"$e_{calib}/e_{true}$",fontsize=15)
plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.show()
savefig(fig,directory,"ecalib_over_etrue.png")

#histogram of ecalib and etrue
fig = plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
c = c[np.invert(np.isnan(c))]
c = np.array(c)
bw = optimized_binwidth(c)
bw = np.arange(min(c), max(c) + bw, bw)
plt.hist(c,bw)
plt.xlabel(r"$e_{calib}$",fontsize=15)
plt.title(r"$e_{calib}$ for $e_{cal} = 0$",fontsize=15)
plt.subplot(2,2,2)
c2 = c2[np.invert(np.isnan(c2))]
c2 = np.array(c2)
bw = optimized_binwidth(c2)
bw = np.arange(min(c2), max(c2) + bw, bw)
plt.hist(c2,bw)
plt.xlabel(r"$e_{calib}$",fontsize=15)
plt.title(r"$e_{calib}$ for $e_{cal} \neq 0$",fontsize=15)
plt.subplot(2,2,3)
t = t[np.invert(np.isnan(t))]
t = np.array(t)
bw = optimized_binwidth(t)
bw = np.arange(min(t), max(t) + bw, bw)
plt.hist(t,bw)
plt.xlabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.subplot(2,2,4)
t2 = t2[np.invert(np.isnan(t2))]
t2 = np.array(t2)
bw = optimized_binwidth(t2)
bw = np.arange(min(t2), max(t2) + bw, bw)
plt.hist(t2,bw)
plt.xlabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} \neq 0$",fontsize=15)
plt.show()
savefig(fig,directory,"histograms_ecalib_etrue.png")

fig = plt.figure(figsize=(10,12))
#mean
ax = plt.subplot(3,1,1)
plt.plot(energy,mean_gaussianfit,lw=3, label = "r$e_{calib}/e_{true}$")
plt.xlabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.ylabel(r"$<e_{calib}/e_{true}>$",fontsize=15)
plt.legend(loc='upper right')
major_ticks = np.arange(0, 200, 50)
minor_ticks = np.arange(0, 200, 10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
# and a corresponding grid
ax.grid(which='both')
# or if you want differnet settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=1)

#sigma
ax = plt.subplot(3,1,2)
plt.plot(energy,sigma_gaussianfit,lw=3, label = "r$e_{calib}/e_{true}$")
plt.xlabel(r"$e_{true}$",fontsize=15)
plt.ylabel(r"$\sigma (e_{calib}/e_{true})$",fontsize=15)
plt.legend(loc='upper right')
major_ticks = np.arange(0, 200, 50)
minor_ticks = np.arange(0, 200, 10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
# and a corresponding grid
ax.grid(which='both')
# or if you want differnet settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=1)

#chi2
ax = plt.subplot(3,2,5)
plt.plot(energy,reducedChi2,lw=3, label = r"$\chi^2/df$")
plt.xlabel(r"$e_{true}$",fontsize=15)
plt.ylabel(r"$\chi^2/df$",fontsize=15)
plt.legend(loc='upper right')
major_ticks = np.arange(0, 200, 50)
minor_ticks = np.arange(0, 200, 10)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
# and a corresponding grid
ax.grid(which='both')
# or if you want differnet settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=1)

ax = plt.subplot(3,2,6)
x = np.array(reducedChi2)
x = x[np.invert(np.isnan(x))]
bw = optimized_binwidth(x)
bins = np.arange(min(x), max(x) + bw, bw)
plt.hist(x,bins)
plt.xlabel(r"$\chi^2/df$",fontsize=15)

plt.show()

savefig(fig,directory,"ecalib_over_etrue_curve.png")
