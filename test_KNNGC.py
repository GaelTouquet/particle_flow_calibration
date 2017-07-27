#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does KNNGC works.
"""

import matplotlib.pyplot as plt
import pfcalibration.usualplots as usplt               # usual plots function 
from pfcalibration.tools import importData,importCalib # to import binary data
from pfcalibration.tools import savefig                # to save a figure
import numpy as np
from pfcalibration.tools import gaussian_param, binwidth_array
import math

# file to save the pictures
directory = "pictures/testKNNGC/"
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
n_neighbors_ecal_eq_0 = 2000
n_neighbors_ecal_neq_0 = 250
weights = 'gaussian'
algorithm = 'auto'
sigma = 5
lim = 150
energystep = 1
kind = 'cubic'
cut = 2

try:
    # We import the calibration
    filename = "calibrations/KNNGaussianCleaning_162Kpart_cut_2_lim_150_n_neighbors_ecal_eq_0_2000_n_neighbors_ecal_neq_0_250_sigma_5.calibration"
    KNNGC = importCalib(filename)
except FileNotFoundError:
    # We create the calibration
    KNNGC = data1.KNNGaussianCleaning(n_neighbors_ecal_eq_0,n_neighbors_ecal_neq_0,
                             weights,algorithm,sigma,lim,energystep,kind,cut)
    KNNGC.saveCalib()
    print(KNNGC)
    
classname = KNNGC.classname
#plot 3D Training points
fig = plt.figure(1,figsize=(7, 7))
usplt.plot3D_training(data1)
plt.show()
savefig(fig,directory,classname+"_plot3D_training.png")

#plot 3D surface calibration
fig = plt.figure(1,figsize=(7, 7))
usplt.plot3D_surf(KNNGC)
plt.show()
savefig(fig,directory,classname+"_plot3D_surf.png")
savefig(fig,directory,classname+"_plot3D_surf.eps")
plt.close()

#courbe de calibration pour ecal = 0
fig = plt.figure(figsize=(12,4))
usplt.plotCalibrationCurve(KNNGC)
plt.show()
savefig(fig,directory,classname+"_calibration.png")
plt.close()

#ecalib/true in function of etrue
fig = plt.figure(figsize=(12,4))
usplt.plot_ecalib_over_etrue_functionof_etrue(KNNGC,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue.png")
plt.close()

#histogram of ecalib and etrue
fig = plt.figure(figsize=(12,5))
usplt.hist_ecalib(KNNGC,data2)
plt.show()
savefig(fig,directory,classname+"_histograms_ecalib_etrue.png")
savefig(fig,directory,classname+"_histograms_ecalib_etrue.eps")
plt.close()

#ecalib/etrue in function of ecal,hcal
fig = plt.figure(figsize=(12,4))
usplt.plot_ecalib_over_etrue_functionof_ecal_hcal(KNNGC,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_functionof_ecal_hcal.png")
plt.close()

#ecalib/etrue gaussian fit curve
fig = plt.figure(figsize=(12,10))
usplt.plot_gaussianfitcurve_ecalib_over_etrue_functionof_ecal_hcal(KNNGC,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_curve.png")
savefig(fig,directory,classname+"_ecalib_over_etrue_curve.eps")
plt.close()

#chi2 for each point of the calibration
hcal_ecal_eq_0 = KNNGC.evaluatedPoint_hcal_ecal_eq_0
rchi2_ecal_eq_0 = KNNGC.evaluatedPoint_reducedchi2_ecal_eq_0
ecal_ecal_neq_0 = KNNGC.interpolation_ecal_neq_0.x
hcal_ecal_neq_0 = KNNGC.interpolation_ecal_neq_0.y
eecal_ecal_neq_0,hhcal_ecal_neq_0 = np.meshgrid(ecal_ecal_neq_0,hcal_ecal_neq_0)
rchi2_ecal_neq_0 = KNNGC.evaluatedPoint_reducedchi2[1:]
rrchi2_ecal_neq_0 = np.resize(rchi2_ecal_neq_0,(len(hcal_ecal_neq_0),len(ecal_ecal_neq_0)))
rrchi2_ecal_neq_0[eecal_ecal_neq_0 + hhcal_ecal_neq_0 > lim] = math.nan
x = np.array(KNNGC.evaluatedPoint_reducedchi2)
x = x[np.array(KNNGC.evaluatedPoint_hcal)+np.array(KNNGC.evaluatedPoint_ecal) < lim]

fig = plt.figure(figsize=(7,6))
plt.subplot(2,2,3)
plt.plot(hcal_ecal_eq_0,KNNGC.evaluatedPoint_reducedchi2_ecal_eq_0)
plt.xlabel(r"$E_{\rm hcal} \rm{(GeV)}$",fontsize=18)
plt.ylabel(r"$\chi^2/df$",fontsize=18)
plt.title(r"$\chi^2/df$ for $E_{\rm ecal} = 0$",fontsize=18)
plt.subplot(2,2,4)
plt.hist(rchi2_ecal_eq_0,binwidth_array(rchi2_ecal_eq_0,0.1))
plt.xlabel(r"$\chi^2/df$",fontsize=18)
plt.title(r"$\chi^2/df$ for $E_{\rm ecal} = 0$",fontsize=18)
plt.subplot(2,2,1)
xmin = min(ecal_ecal_neq_0)
xmax = max(ecal_ecal_neq_0)
ymin = min(hcal_ecal_neq_0)
ymax = max(hcal_ecal_neq_0)
vmin = 0.5
vmax = 1.5
im = plt.imshow(rrchi2_ecal_neq_0,cmap=plt.cm.seismic, extent=(xmin,xmax,ymin,ymax), origin='lower',vmin=vmin,vmax=vmax,interpolation='bilinear')
plt.colorbar(im)
plt.title(r"$\chi^2/df$ for $E_{\rm ecal} \neq 0$",fontsize=18)
plt.ylabel(r"$E_{\rm hcal} \rm{(GeV)}$",fontsize=18)
plt.xlabel(r"$E_{\rm ecal} \rm{(GeV)}$",fontsize=18)
plt.subplot(2,2,2)
plt.hist(x,binwidth_array(x))
plt.xlabel(r"$\chi^2/df$",fontsize=18)
plt.title(r"$\chi^2/df$ for $E_{\rm ecal} \neq 0$",fontsize=18)
plt.tight_layout()
plt.show()
savefig(fig,directory,classname+"_chi2_calib.png")
savefig(fig,directory,classname+"_chi2_calib.eps")

# some histograms
i1 = int(len(KNNGC.evaluatedPoint_hcal_ecal_eq_0)/2)
i2 = int(len(KNNGC.evaluatedPoint_hcal)/5)
e1 = 0
h1 = KNNGC.evaluatedPoint_hcal_ecal_eq_0[i1]
t1 = KNNGC.evaluatedPoint_neighbours_true_ecal_eq_0[i1]
m1 = np.mean(KNNGC.evaluatedPoint_neighbours_true_ecal_eq_0[i1])
entries1 = KNNGC.evaluatedPoint_entries_ecal_eq_0[i1]
bin_middles1 = KNNGC.evaluatedPoint_bin_middles_ecal_eq_0[i1]
error1 = np.sqrt(entries1)
params1 = KNNGC.evaluatedPoint_parameters_ecal_eq_0[i1]
e2 = KNNGC.evaluatedPoint_ecal[i2]
h2 = KNNGC.evaluatedPoint_hcal[i2]
entries2 = KNNGC.evaluatedPoint_entries[i2]
bin_middles2 = KNNGC.evaluatedPoint_bin_middles[i2]
error2 = np.sqrt(entries2)
params2 = KNNGC.evaluatedPoint_parameters[i2]

fig = plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.errorbar(bin_middles1, entries1, yerr=error1, fmt='o')
xplot = np.arange(min(bin_middles1),max(bin_middles1),1)
plt.plot(xplot,gaussian_param(xplot,*params1),lw=3)
plt.xlabel(r"$E_{\rm true} \rm{(GeV)}$",fontsize=18)
plt.title(r"histogram of $E_{true}$ for $(E_{ecal}="+str(np.around(e1,2))+",E_{hcal}="+str(np.around(h1,2))+")$",fontsize=18)
plt.subplot(1,2,2)
plt.errorbar(bin_middles2, entries2, yerr=error2, fmt='o')
plt.plot(xplot,gaussian_param(xplot,*params2),lw=3)
xplot = np.arange(min(bin_middles2),max(bin_middles2),1)
plt.xlabel(r"$E_{\rm true} \rm{(GeV)}$",fontsize=18)
plt.title(r"histogram of $E_{true}$ for $(E_{ecal}="+str(np.around(e2,2))+",E_{hcal}="+str(np.around(h2,2))+")$",fontsize=18)
plt.tight_layout()
plt.show()
savefig(fig,directory,classname+"_hist_calib.png")
savefig(fig,directory,classname+"_hist_calib.eps")
