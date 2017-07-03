#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does KNNGF works and why there is a structure in ecalib/etrue .
"""

import matplotlib.pyplot as plt
import numpy as np
from pfcalibration.tools import importPickle, savefig
from pfcalibration.tools import gaussian_fit, binwidth_array
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


directory = "pictures/testKNNGF_structure/"


filename = 'charged_hadrons_100k.energydata'
data1 = importPickle(filename)
filename = 'prod2_200_400k.energydata'
#on fusionne les 2 jeux de données
data1 = data1.mergeWith(importPickle(filename))
#on sépare data en 2
data1,data2 = data1.splitInTwo()
data2,data2 = data2.splitInTwo()

# paramètres de calibration
lim = 150
n_neighbors = 250
energystep = 1

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
        params,reduced = gaussian_fit(y_ind,giveChi2 = True)
        if not(math.isnan(params[1])):
            means.append(np.mean(y_ind))
            sigma_gaussianfit.append(params[0])
            mean_gaussianfit.append(params[1])
            energy.append(e)
            reducedChi2.append(reduced)
    return energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2





KNNGF = data1.kNNGaussianFit(n_neighbors=n_neighbors,lim=lim,energystep=energystep,kind='cubic')

#ecalib/etrue pour ecal = 0
h = data2.hcal[np.logical_and(data2.ecal == 0,data2.ecal+data2.hcal < lim)]
t = data2.true[np.logical_and(data2.ecal == 0,data2.ecal+data2.hcal < lim)]
e = np.zeros(len(h))
c = KNNGF.predict(e,h)
r = c/t

energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2 = getMeans(t,r)
fig = plt.figure(figsize=(10,5))
plt.plot(t,r,'.',markersize=1)
plt.axis([0,200,0,2])
plt.plot(energy,mean_gaussianfit,lw=3)
plt.xlabel(r"$e_{true}$",fontsize=15)
plt.ylabel(r"$e_{calib}/e_{true}$",fontsize=15)
plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=15)
savefig(fig,directory,"ecalib_over_etrue.png")

#histogram of hcal
fig = plt.figure(figsize=(10,5))
bw = binwidth_array(h)
entries, bins, other = plt.hist(h,bw)
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.title(r"$h_{cal}$ for $e_{cal} = 0$",fontsize=15)
plt.show()
savefig(fig,directory,"histograms_hcal.png")

#histogram of ecalib
fig = plt.figure(figsize=(10,5))
#bw = binwidth_array(c)
bw = binwidth_array(c,0.01)
entries, bins, other = plt.hist(c,bw)
plt.xlabel(r"$e_{calib}$",fontsize=15)
plt.title(r"$e_{calib}$ for $e_{cal} = 0$",fontsize=15)
plt.show()
savefig(fig,directory,"histograms_ecalib.png")

imax = np.argmax(entries)
borne_inf = bins[imax]
borne_sup = bins[imax+1]
index = np.logical_and(c >= borne_inf,c < borne_sup)
ecalibs = c[index]
fig = plt.figure(figsize=(10,5))
plt.hist(ecalibs,binwidth_array(ecalibs))
plt.xlabel(r"$e_{calib}$",fontsize=15)
plt.title(r"$e_{calib}$ for the highest bar",fontsize=15)
savefig(fig,directory,"histograms_ecalib_etrue_highest.png")

#courbe de calibration pour ecal = 0
hcal_train = KNNGF.hcal_train[KNNGF.ecal_train==0]
true_train = KNNGF.true_train[KNNGF.ecal_train==0]
hcal_calib = np.arange(min(hcal_train),lim,0.1)
ecal_calib = np.zeros(len(hcal_calib))
calib = KNNGF.predict(ecal_calib,hcal_calib)

fig = plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(hcal_train,true_train,'.',markersize=1,label=r"training data")
plt.plot(h[index],t[index],'.',label=r"$e_{calib} \in ["+str(np.around(borne_inf,2))+","+str(np.around(borne_sup,2))+"[$")
plt.plot(hcal_calib,calib,lw=3,label = "calibration")
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.ylabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.axis([0,max(hcal_train),0,max(true_train)])
plt.legend(loc = "upper right")

plt.subplot(2,2,3)
plt.plot(hcal_train,true_train,'.',markersize=1,label=r"training data")
plt.plot(h[index],t[index],'.',label=r"$e_{calib} \in ["+str(np.around(borne_inf,2))+","+str(np.around(borne_sup,2))+"[$")
plt.plot(hcal_calib,calib,lw=3,label = "calibration")
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.ylabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.axis([min(h[index]),max(h[index]),0,2*borne_sup])


savefig(fig,directory,"calibration.png")