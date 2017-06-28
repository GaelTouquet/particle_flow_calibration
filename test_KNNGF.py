#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

import matplotlib.pyplot as plt
import numpy as np
from tools import importPickle, savefig
from tools import gaussian_fit, gaussian_param, mean, optimized_binwidth
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


directory = "pictures/testKNNGF/"


filename = 'charged_hadrons_100k.energydata'
data1 = importPickle(filename)
filename = 'prod2_200_400k.energydata'
#on fusionne les 2 jeux de données
data1 = data1.mergeWith(importPickle(filename))
#on sépare data en 2
data1,data2 = data1.splitInTwo()

# paramètres de calibration
lim = 150
lim_min = 20
lim_max = 150
n_neighbors = 300
energystep = 2

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
            means.append(mean(y_ind))
            sigma_gaussianfit.append(params[0])
            mean_gaussianfit.append(params[1])
            energy.append(e)
            reducedChi2.append(reduced)
    return energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2





KNNGF = data1.kNNGaussianFit(n_neighbors=n_neighbors,lim=lim,energystep=energystep,kind='linear')

#chi2 for ecal == 0
fig = plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(KNNGF.evaluatedPoint_hcal_ecal_eq_0,KNNGF.evaluatedPoint_reducedchi2_ecal_eq_0)
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.ylabel(r"$\chi^2/df$",fontsize=15)
plt.title(r"$\chi^2/df$ for $e_{cal} = 0$",fontsize=15)
plt.subplot(1,2,2)
x = np.array(KNNGF.evaluatedPoint_reducedchi2_ecal_eq_0)
bw = 0.1
bins = np.arange(min(x), max(x) + bw, bw)
plt.hist(x,bins)
plt.xlabel(r"$\chi^2/df$",fontsize=15)
plt.title(r"$\chi^2/df$ for $e_{cal} = 0$",fontsize=15)
plt.show()
savefig(fig,directory,"calibration_chi2_for_ecal_eq_0.png")


#chi2 for ecal neq 0
fig = plt.figure(figsize=(12,5))
plt.subplot(1,2,2)
x = np.array(KNNGF.evaluatedPoint_reducedchi2)
x = x[np.invert(np.isnan(x))]
bw = optimized_binwidth(x)
bins = np.arange(min(x), max(x) + bw, bw)
plt.hist(x,bins)
plt.xlabel(r"$\chi^2/df$",fontsize=15)
plt.title(r"$\chi^2/df$ for $e_{cal} \neq 0$",fontsize=15)
plt.subplot(1,2,1)
x = KNNGF.interpolation_ecal_neq_0.x
y = KNNGF.interpolation_ecal_neq_0.y
l = len(x)
z = KNNGF.evaluatedPoint_reducedchi2[1:]
z = np.resize(z,(l,l))
xx,yy = np.meshgrid(x,y)
mask = xx + yy > lim
z[mask] = math.nan
xmin = min(x)
xmax = max(x)
ymin = min(y)
ymax = max(y)
vmin = 0.5
vmax = 1.5
im = plt.imshow(z,cmap=plt.cm.seismic, extent=(xmin,xmax,ymin,ymax), origin='lower',vmin=vmin,vmax=vmax,interpolation='bilinear')
plt.colorbar(im)
plt.show()
savefig(fig,directory,"calibration_chi2_for_ecal_neq_0.png")

#quelques histogrammes for ecal == 0
fig = plt.figure(figsize=(10,10))
j = 0
l = len(KNNGF.evaluatedPoint_hcal_ecal_eq_0)
for i in [0,int(l/4),int(l*3/4),l-1]:
    j += 1
    plt.subplot(2,2,j)
    e = 0
    h = KNNGF.evaluatedPoint_hcal_ecal_eq_0[i]
    entries = KNNGF.evaluatedPoint_entries_ecal_eq_0[i]
    bin_middles = KNNGF.evaluatedPoint_bin_middles_ecal_eq_0[i]
    error = np.sqrt(entries)
    params = KNNGF.evaluatedPoint_parameters_ecal_eq_0[i]
    plt.errorbar(bin_middles, entries, yerr=error, fmt='o')
    xplot = np.arange(min(bin_middles),max(bin_middles),1)
    plt.plot(xplot,gaussian_param(xplot,*params),lw=3)
    plt.xlabel(r"$e_{true}$",fontsize=15)
    plt.title(r"histogram of $e_{true}$ for $(e_{cal}="+str(np.around(e,2))+",h_{cal}="+str(np.around(h,2))+")$",fontsize=12)
plt.show()
savefig(fig,directory,"calibration_hist_for_ecal_eq_0.png")

#quelques histogrammes for ecal ≠ 0
fig = plt.figure(figsize=(10,10))
j = 0
l = len(KNNGF.evaluatedPoint_hcal)
for i in [0,int(l/4),int(l*3/4),l-1]:
    j += 1
    plt.subplot(2,2,j)
    e = KNNGF.evaluatedPoint_ecal[i]
    h = KNNGF.evaluatedPoint_hcal[i]
    entries = KNNGF.evaluatedPoint_entries[i]
    bin_middles = KNNGF.evaluatedPoint_bin_middles[i]
    error = np.sqrt(entries)
    params = KNNGF.evaluatedPoint_parameters[i]
    plt.errorbar(bin_middles, entries, yerr=error, fmt='o')
    xplot = np.arange(min(bin_middles),max(bin_middles),1)
    plt.plot(xplot,gaussian_param(xplot,*params),lw=3)
    plt.xlabel(r"$e_{true}$",fontsize=15)
    plt.title(r"histogram of $e_{true}$ for $(e_{cal}="+str(np.around(e,2))+",h_{cal}="+str(np.around(h,2))+")$",fontsize=12)
plt.show()
savefig(fig,directory,"calibration_hist_for_ecal_neq_0.png")

#courbe de calibration pour ecal = 0
hcal_train = KNNGF.hcal_train[KNNGF.ecal_train==0]
true_train = KNNGF.true_train[KNNGF.ecal_train==0]
h = np.arange(0,lim,0.1)
e = np.zeros(len(h))
t = KNNGF.predict(e,h)
fig = plt.figure(figsize=(8,5))
plt.plot(hcal_train,true_train,'.',markersize=1)
plt.plot(h,t,lw=3)
plt.xlabel(r"$h_{cal}$",fontsize=15)
plt.ylabel(r"$e_{true}$",fontsize=15)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.show()
savefig(fig,directory,"calibration.png")

#ecalib/etrue pour ecal = 0
h = data2.hcal[data2.ecal == 0]
t = data2.true[data2.ecal == 0]
e = np.zeros(len(h))
c = KNNGF.predict(e,h)
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

h2 = data2.hcal[data2.ecal != 0]
t2 = data2.true[data2.ecal != 0]
e2 = data2.ecal[data2.ecal != 0]
c2 = KNNGF.predict(e2,h2)
r2 = c2/t2

energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2 = getMeans(t2,r2)
plt.subplot(1,2,2)
plt.plot(t2,r2,'.',markersize=1)
plt.axis([0,200,0,2])
plt.plot(energy,mean_gaussianfit,lw=3)
plt.xlabel(r"$e_{true}$",fontsize=15)
plt.ylabel(r"$e_{calib}/e_{true}$",fontsize=15)
plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=15)
plt.show()
savefig(fig,directory,"ecalib_over_etrue.png")

#ecalib
fig = plt.figure(figsize=(5,5))
c = c[np.invert(np.isnan(c))]
c = np.array(c)
bw = optimized_binwidth(c)
bw = np.arange(min(c), max(c) + bw, bw)
plt.bar(KNNGF.interpolation_ecal_eq_0.y,1000*np.ones(len(KNNGF.interpolation_ecal_eq_0.y)),color='yellow')
plt.hist(c,bw)
plt.xlabel(r"$e_{calib}$",fontsize=15)
plt.title(r"$e_{calib}$ for $e_{cal} = 0$",fontsize=15)
plt.show()

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
