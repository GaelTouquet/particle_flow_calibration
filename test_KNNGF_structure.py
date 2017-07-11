#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does KNNGF works.
"""

import matplotlib.pyplot as plt
import numpy as np
from pfcalibration.tools import importPickle, savefig
from pfcalibration.tools import gaussian_fit, binwidth_array
from sklearn import neighbors
import math


# file to save the pictures
directory = "pictures/testKNNGF/"
#importation of simulated particles
filename = 'charged_hadrons_100k.energydata'
data1 = importPickle(filename)
filename = 'prod2_200_400k.energydata'
data2 = importPickle(filename)
# we merge the 2 sets of data
data1 = data1.mergeWith(importPickle(filename))
# we split the data in 2 sets
data1,data2 = data1.splitInTwo()
#data 1 -> training data
#data 2 -> data to predict

# parameters of the calibration
lim = 150
n_neighbors_ecal_eq_0=2000
n_neighbors_ecal_neq_0=200

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

def getMeans2D(energy_x,energy_y,z):
    ind  = np.invert(np.isnan(z))
    z = z[ind]
    energy_x = energy_x[ind]
    energy_y = energy_y[ind]
    neighborhood = neighbors.NearestNeighbors(n_neighbors=n_neighbors_ecal_neq_0)
    neighborhood.fit(np.transpose(np.matrix([energy_x,energy_y])))
    step = 0.5
    ener_x = np.arange(0,lim+step,step)
    ener_y = np.arange(0,lim+step,step)
    sigma_gaussianfit = []
    mean_gaussianfit = []
    
    for y in ener_y:
        line_mean = []
        line_sigma = []
        for x in ener_x:
            if x+y < lim:
                dist, ind = neighborhood.kneighbors(X = [[x,y]])
                z_ind = z[ind]
                params = gaussian_fit(z_ind,binwidth=0.1,reducedChi2Max = 10)
                if math.isnan(params[0]):
                    line_mean.append(np.mean(z_ind))
                    line_sigma.append(np.sqrt(np.std(z_ind)))
                else:
                    line_mean.append(params[1])
                    line_sigma.append(params[0])
            else:
                line_mean.append(math.nan)
                line_sigma.append(math.nan)
        mean_gaussianfit.append(line_mean)
        sigma_gaussianfit.append(line_sigma)
    return mean_gaussianfit, sigma_gaussianfit


KNNGF = data1.KNNGaussianFit(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                             n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                             lim=lim,kind='cubic')

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