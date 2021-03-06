#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does KNNGFD works.
"""
import matplotlib.pyplot as plt
from pfcalibration.tools import gaussian_fit, binwidth_array
from sklearn import neighbors
import math
import numpy as np
from pfcalibration.tools import importData,importCalib # to import binary data
from pfcalibration.tools import savefig                # to save a figure

# file to save the pictures
directory = "pictures/testKNNGFD_structure/"

#importation of simulated particles
filename = 'charged_hadrons_100k.energydata'
data1 = importData(filename)
filename = 'prod2_200_400k.energydata'
data2 = importData(filename)
# we merge the 2 sets of data
data1 = data1.mergeWith(data2)
# we split the data in 2 sets
data1,data2 = data1.splitInTwo()
#data 1 -> training data
#data 2 -> data to predict

# parameters of the calibration
lim = 150                   # if ecal + hcal > lim, ecalib = math.nan
n_neighbors_ecal_eq_0=2000  # number of neighbors for ecal = 0
n_neighbors_ecal_neq_0=250  # number of neighbors for ecal ≠ 0
energystep_ecal_eq_0 = 1
energystep_ecal_neq_0 = 5
    
try:
    # We import the calibration
    filename = "calibrations/KNNGaussianFitDirect_162Kpart_ecal_train_ecal_neq_0_min_0.292026728392_hcal_train_ecal_eq_0_min_1.00043606758_hcal_train_ecal_neq_0_min_1.00002634525_lim_150_n_neighbors_ecal_eq_0_2000_n_neighbors_ecal_neq_0_250.calibration"
    KNNGFD = importCalib(filename)
except FileNotFoundError:
    # We create the calibration
    KNNGFD = data1.KNNGaussianFitDirect(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                                        n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                                        lim=lim)
    KNNGFD.saveCalib()

def getMeans(energy_x,y,n_neighbors=n_neighbors_ecal_eq_0):
    ind  = np.invert(np.isnan(y))
    y = y[ind]
    energy_x = energy_x[ind]

    neighborhood = neighbors.NearestNeighbors(n_neighbors=1000)
    neighborhood.fit(np.transpose(np.matrix(energy_x)))
    step = 0.1
    ener = np.arange(min(energy_x),max(energy_x),step)
    sigma_gaussianfit = []
    mean_gaussianfit = []
    means = []
    energy = []
    reducedChi2 = []
    for e in ener:
        dist, ind = neighborhood.kneighbors(X = e)
        y_ind = y[ind][np.invert(np.isnan(y[ind]))]
        params,reduced = gaussian_fit(y_ind,binwidth = 0.1,giveReducedChi2 = True,reducedChi2Max = 10)
        if not(math.isnan(params[1])):
            means.append(np.mean(y_ind))
            sigma_gaussianfit.append(params[0])
            mean_gaussianfit.append(params[1])
            energy.append(e)
            reducedChi2.append(reduced)
    return energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2

#ecalib/etrue pour ecal = 0
h = data2.hcal[np.logical_and(data2.ecal == 0,data2.ecal+data2.hcal < lim)]
t = data2.true[np.logical_and(data2.ecal == 0,data2.ecal+data2.hcal < lim)]
e = np.zeros(len(h))
c = KNNGFD.predict(e,h)
r = c/t

energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2 = getMeans(t,r,n_neighbors = 1000)
fig = plt.figure(figsize=(10,5))
plt.plot(t,r,'.',markersize=1)
plt.axis([0,200,0,2])
plt.plot(energy,mean_gaussianfit,lw=3)
plt.xlabel(r"$e_{true}$",fontsize=15)
plt.ylabel(r"$e_{calib}/e_{true}$",fontsize=15)
plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=15)
savefig(fig,directory,"ecalib_over_etrue.png")

#courbe de calibration pour ecal = 0
hcal_train = KNNGFD.hcal_train[KNNGFD.ecal_train==0]
true_train = KNNGFD.true_train[KNNGFD.ecal_train==0]
hcal_calib = np.arange(min(hcal_train),lim,0.1)
ecal_calib = np.zeros(len(hcal_calib))
calib = KNNGFD.predict(ecal_calib,hcal_calib)
fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(hcal_train,true_train,'.',markersize=1,label=r"training data")
plt.plot(hcal_calib,calib,lw=3,label = "calibration")
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.ylabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.axis([0,lim,0,max(calib)])
plt.legend(loc = "lower right")
plt.subplot(1,2,2)
plt.plot(hcal_train,true_train,'.',markersize=1,label=r"training data")
plt.plot(hcal_calib,calib,lw=3,label = "calibration")
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.ylabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.axis([0,15,0,15])
plt.legend(loc = "lower right")
savefig(fig,directory,"calibration.png")




#histogram of hcal
fig = plt.figure(figsize=(10,5))
bw = binwidth_array(h)
entries, bins, other = plt.hist(h,bw)
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.title(r"$h_{cal}$ for $e_{cal} = 0$",fontsize=15)
plt.show()
savefig(fig,directory,"histograms_hcal.png")

#histogram of etrue
fig = plt.figure(figsize=(10,5))
bw = binwidth_array(t)
entries, bins, other = plt.hist(h,bw)
plt.xlabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.show()
savefig(fig,directory,"histograms_etrue.png")

#histogram of ecalib
fig = plt.figure(figsize=(10,5))
bw = binwidth_array(c)
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
hcals = h[index]
fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(ecalibs)
plt.xlabel(r"$e_{calib}$",fontsize=15)
plt.title(r"$e_{calib}$ for the highest bar",fontsize=15)
plt.subplot(1,2,2)
m = np.mean(hcals)
plt.hist(hcals-m)
plt.xlabel(r"$h_{cal} - "+str(np.round(m,2))+"$",fontsize=15)
plt.title(r"$h_{cal} - \mu(h_{cal})$ for the highest bar",fontsize=15)
savefig(fig,directory,"histograms_ecalib_etrue_highest.png")

#courbe de calibration pour ecal = 0
hcal_train = KNNGFD.hcal_train[KNNGFD.ecal_train==0]
true_train = KNNGFD.true_train[KNNGFD.ecal_train==0]
hcal_calib = np.arange(min(hcal_train),lim,0.1)
ecal_calib = np.zeros(len(hcal_calib))
calib = KNNGFD.predict(ecal_calib,hcal_calib)

fig = plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(hcal_train,true_train,'.',markersize=1,label=r"training data")
plt.plot(h[index],t[index],'.',label=r"$e_{calib} \in ["+str(np.around(borne_inf,2))+","+str(np.around(borne_sup,2))+"[$")
plt.plot(hcal_calib,calib,lw=3,label = "calibration")
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.ylabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.axis([0,lim,0,max(calib)])
plt.legend(loc = "lower right")

plt.subplot(2,2,3)
plt.plot(hcal_train,true_train,'.',markersize=1,label=r"training data")
plt.plot(h[index],t[index],'.',label=r"$e_{calib} \in ["+str(np.around(borne_inf,2))+","+str(np.around(borne_sup,2))+"[$")
plt.plot(hcal_calib,calib,lw=3,label = "calibration")
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.ylabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.axis([min(h[index]),max(h[index]),0,2*borne_sup])

plt.subplot(2,2,4)
neighborhood = KNNGFD.neighborhood(e[index],h[index])
plt.plot(hcal_train,true_train,'.',markersize=1,label=r"training data")
plt.plot(h[index],t[index],'.',label=r"$e_{calib} \in ["+str(np.around(borne_inf,2))+","+str(np.around(borne_sup,2))+"[$")
plt.plot(hcal_calib,calib,lw=3,label = "calibration")
for neigh in neighborhood:
    plt.plot(neigh[1],neigh[2],'.')
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.ylabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.axis([min(h[index]),max(h[index]),0,2*borne_sup])
savefig(fig,directory,"calibration_neigh.png")
    