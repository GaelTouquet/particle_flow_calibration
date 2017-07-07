#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does KNNGF works.
"""

import matplotlib.pyplot as plt
import pfcalibration.usualplots as usplt
import numpy as np
from pfcalibration.tools import importPickle, savefig
from pfcalibration.tools import gaussian_param, binwidth_array
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
energystep = 1



KNNGF = data1.KNNGaussianFit(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                             n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                             lim=lim,energystep=energystep,kind='cubic')
classname = type(KNNGF).__name__

#courbe de calibration pour ecal = 0
fig = plt.figure(figsize=(10,4))
usplt.plotCalibrationCurve(KNNGF)
#plt.show()
savefig(fig,directory,classname+"_calibration.png")

#ecalib/true in function of etrue
fig = plt.figure(figsize=(10,4))
usplt.plot_ecalib_over_etrue_functionof_etrue(KNNGF,data2)
#plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue.png")

#histogram of ecalib and etrue
fig = plt.figure(figsize=(10,8))
usplt.hist_ecalib(KNNGF,data2)
savefig(fig,directory,classname+"_histograms_ecalib_etrue.png")

#ecalib/etrue in function of ecal,hcal
fig = plt.figure(figsize=(10,5))
usplt.plot_ecalib_over_etrue_functionof_ecal_hcal(KNNGF,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_functionof_ecal_hcal.png")

#ecalib/etrue gaussian fit curve
fig = plt.figure(figsize=(10,12))
usplt.plot_gaussianfitcurve_ecalib_over_etrue_functionof_ecal_hcal(KNNGF,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_curve.png")



#chi2 for each point of the calibration
hcal_ecal_eq_0 = KNNGF.evaluatedPoint_hcal_ecal_eq_0
rchi2_ecal_eq_0 = KNNGF.evaluatedPoint_reducedchi2_ecal_eq_0
ecal_ecal_neq_0 = KNNGF.interpolation_ecal_neq_0.x
hcal_ecal_neq_0 = KNNGF.interpolation_ecal_neq_0.y
eecal_ecal_neq_0,hhcal_ecal_neq_0 = np.meshgrid(ecal_ecal_neq_0,hcal_ecal_neq_0)
rchi2_ecal_neq_0 = KNNGF.evaluatedPoint_reducedchi2[1:]
rrchi2_ecal_neq_0 = np.resize(rchi2_ecal_neq_0,(len(hcal_ecal_neq_0),len(ecal_ecal_neq_0)))
rrchi2_ecal_neq_0[eecal_ecal_neq_0 + hhcal_ecal_neq_0 > lim] = math.nan
x = np.array(KNNGF.evaluatedPoint_reducedchi2)
x = x[np.array(KNNGF.evaluatedPoint_hcal)+np.array(KNNGF.evaluatedPoint_ecal) < lim]

fig = plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.plot(hcal_ecal_eq_0,KNNGF.evaluatedPoint_reducedchi2_ecal_eq_0)
plt.xlabel(r"$h_{cal}$",fontsize=12)
plt.ylabel(r"$\chi^2/df$",fontsize=12)
plt.title(r"$\chi^2/df$ for $e_{cal} = 0$",fontsize=12)
plt.subplot(2,2,2)
plt.hist(rchi2_ecal_eq_0,binwidth_array(rchi2_ecal_eq_0,0.25))
plt.xlabel(r"$\chi^2/df$",fontsize=12)
plt.title(r"$\chi^2/df$ for $e_{cal} = 0$",fontsize=12)
plt.subplot(2,2,3)
xmin = min(ecal_ecal_neq_0)
xmax = max(ecal_ecal_neq_0)
ymin = min(hcal_ecal_neq_0)
ymax = max(hcal_ecal_neq_0)
vmin = 0.5
vmax = 1.5
im = plt.imshow(rrchi2_ecal_neq_0,cmap=plt.cm.seismic, extent=(xmin,xmax,ymin,ymax), origin='lower',vmin=vmin,vmax=vmax,interpolation='bilinear')
plt.colorbar(im)
plt.title(r"$\chi^2/df$ for $e_{cal} \neq 0$",fontsize=12)
plt.ylabel(r"$h_{cal}$",fontsize=12)
plt.xlabel(r"$e_{cal}$",fontsize=12)
plt.subplot(2,2,4)
plt.hist(x,binwidth_array(x))
plt.xlabel(r"$\chi^2/df$",fontsize=12)
plt.title(r"$\chi^2/df$ for $e_{cal} \neq 0$",fontsize=12)
plt.tight_layout()
plt.show()
savefig(fig,directory,classname+"_chi2_calib.png")

# some histograms
i1 = int(len(KNNGF.evaluatedPoint_hcal_ecal_eq_0)/2)
i2 = int(len(KNNGF.evaluatedPoint_hcal)/5)
e1 = 0
h1 = KNNGF.evaluatedPoint_hcal_ecal_eq_0[i1]
entries1 = KNNGF.evaluatedPoint_entries_ecal_eq_0[i1]
bin_middles1 = KNNGF.evaluatedPoint_bin_middles_ecal_eq_0[i1]
error1 = np.sqrt(entries1)
params1 = KNNGF.evaluatedPoint_parameters_ecal_eq_0[i1]
e2 = KNNGF.evaluatedPoint_ecal[i2]
h2 = KNNGF.evaluatedPoint_hcal[i2]
entries2 = KNNGF.evaluatedPoint_entries[i2]
bin_middles2 = KNNGF.evaluatedPoint_bin_middles[i2]
error2 = np.sqrt(entries2)
params2 = KNNGF.evaluatedPoint_parameters[i2]

fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.errorbar(bin_middles1, entries1, yerr=error1, fmt='o')
xplot = np.arange(min(bin_middles1),max(bin_middles1),1)
plt.plot(xplot,gaussian_param(xplot,*params1),lw=3)
plt.xlabel(r"$e_{true}$",fontsize=12)
plt.title(r"histogram of $e_{true}$ for $(e_{cal}="+str(np.around(e1,2))+",h_{cal}="+str(np.around(h1,2))+")$",fontsize=12)
plt.subplot(1,2,2)
plt.errorbar(bin_middles2, entries2, yerr=error2, fmt='o')
plt.plot(xplot,gaussian_param(xplot,*params2),lw=3)
xplot = np.arange(min(bin_middles2),max(bin_middles2),1)
plt.xlabel(r"$e_{true}$",fontsize=12)
plt.title(r"histogram of $e_{true}$ for $(e_{cal}="+str(np.around(e2,2))+",h_{cal}="+str(np.around(h2,2))+")$",fontsize=12)
plt.tight_layout()
plt.show()
savefig(fig,directory,classname+"_hist_calib.png")


#NEIGHBORS
fig = plt.figure(figsize=(10,4))
#neigh for ecal == 0
KNNGF = data1.kNNGaussianFit(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                             n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                             lim=lim,energystep=30,kind='cubic')
neigh_hcal_ecal_eq_0 = KNNGF.evaluatedPoint_neighbours_hcal_ecal_eq_0
neigh_true_ecal_eq_0 = KNNGF.evaluatedPoint_neighbours_true_ecal_eq_0
hcal_train = KNNGF.hcal_train[KNNGF.ecal_train == 0]
true_train = KNNGF.true_train[KNNGF.ecal_train == 0]
plt.subplot(1,2,1)
plt.plot(hcal_train,true_train,'.',markersize=1)
plt.xlabel(r"$h_{cal}$",fontsize=12)
plt.ylabel(r"$e_{true}$",fontsize=12)
for i in np.arange(len(neigh_hcal_ecal_eq_0)):
    plt.plot(neigh_hcal_ecal_eq_0[i],neigh_true_ecal_eq_0[i],'.',color='red',markersize=1)
plt.title(r"neighbors for $e_{cal} = 0$",fontsize=12)
plt.axis([0,lim,0,lim])


#neigh for ecal != 0
KNNGF = data1.kNNGaussianFit(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                             n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                             lim=lim,energystep=10,kind='cubic')
neigh_ecal_ecal_neq_0 = np.array(KNNGF.evaluatedPoint_neighbours_ecal)
neigh_ecal_ecal_neq_0 = neigh_ecal_ecal_neq_0[np.array(KNNGF.evaluatedPoint_ecal)+np.array(KNNGF.evaluatedPoint_hcal)<lim]
neigh_hcal_ecal_neq_0 = np.array(KNNGF.evaluatedPoint_neighbours_hcal)
neigh_hcal_ecal_neq_0 = neigh_hcal_ecal_neq_0[np.array(KNNGF.evaluatedPoint_ecal)+np.array(KNNGF.evaluatedPoint_hcal)<lim]
plt.subplot(1,2,2)
plt.xlabel(r"$e_{cal}$",fontsize=12)
plt.ylabel(r"$h_{cal}$",fontsize=12)
for i in np.arange(len(neigh_hcal_ecal_neq_0)):
    plt.plot(neigh_ecal_ecal_neq_0[i],neigh_hcal_ecal_neq_0[i],'.',markersize=1)
plt.title(r"neighbors for $e_{cal} \neq 0$",fontsize=12)
plt.axis([0,lim,0,lim])
plt.tight_layout()
plt.show()
savefig(fig,directory,classname+"_neighborhood.png")
