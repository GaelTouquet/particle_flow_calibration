#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does KNNGFD works.
"""

import matplotlib.pyplot as plt
import numpy as np
import pfcalibration.usualplots as usplt               # usual plots function 
from pfcalibration.tools import importData,importCalib # to import binary data
from pfcalibration.tools import savefig                # to save a figure


# file to save the pictures
directory = "pictures/testKNNGFD/"
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

classname = KNNGFD.classname
#plot 3D Training points
fig = plt.figure(1,figsize=(5, 5))
usplt.plot3D_training(data1)
#plt.show()
savefig(fig,directory,classname+"_plot3D_training.png")
plt.close()

#plot 3D surface calibration
fig = plt.figure(1,figsize=(5, 5))
usplt.plot3D_surf(KNNGFD)
#plt.show()
savefig(fig,directory,classname+"_plot3D_surf.png")
plt.close()

#courbe de calibration pour ecal = 0
fig = plt.figure(figsize=(10,4))
usplt.plotCalibrationCurve(KNNGFD)
#plt.show()
savefig(fig,directory,classname+"_calibration.png")
plt.close()

#ecalib/true in function of etrue
fig = plt.figure(figsize=(10,4))
usplt.plot_ecalib_over_etrue_functionof_etrue(KNNGFD,data2)
#plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue.png")
plt.close()

#histogram of ecalib and etrue
fig = plt.figure(figsize=(10,6))
usplt.hist_ecalib(KNNGFD,data2)
#plt.show()
savefig(fig,directory,classname+"_histograms_ecalib_etrue.png")
plt.close()

#ecalib/etrue in function of ecal,hcal
fig = plt.figure(figsize=(10,5))
usplt.plot_ecalib_over_etrue_functionof_ecal_hcal(KNNGFD,data2)
#plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_functionof_ecal_hcal.png")
plt.close()

#ecalib/etrue gaussian fit curve
fig = plt.figure(figsize=(10,12))
usplt.plot_gaussianfitcurve_ecalib_over_etrue_functionof_ecal_hcal(KNNGFD,data2)
#plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_curve.png")
plt.close()

#courbe de calibration pour ecal = 0
hcal_train = KNNGFD.hcal_train_ecal_eq_0
true_train = KNNGFD.true_train_ecal_eq_0
h = np.arange(min(hcal_train),lim,0.1)
e = np.zeros(len(h))
t = KNNGFD.predict(e,h)

#neigh for ecal == 0
h_neigh = np.arange(10,lim,30)
e_neigh = np.zeros(len(h_neigh))
neigh = KNNGFD.neighborhood(e_neigh,h_neigh)
fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(hcal_train,true_train,'.',markersize=1)
plt.plot(h,t,lw=2)
plt.xlabel(r"$h_{cal}$",fontsize=12)
plt.ylabel(r"$e_{true}$",fontsize=12)
for i in np.arange(len(neigh)):
    plt.plot(neigh[i][1],neigh[i][2],'.',color='red',markersize=1)
plt.axis([0,max(h),0,max(t)])
plt.title(r"neighbors for $e_{cal} = 0$",fontsize=12)
#neigh for ecal != 0
h_neigh = np.arange(1,lim,10)
e_neigh = np.arange(1,lim,10)
h_neigh,e_neigh = np.meshgrid(h_neigh,e_neigh)
e_neigh = np.concatenate(e_neigh)
h_neigh =np.concatenate(h_neigh)
neigh = KNNGFD.neighborhood(e_neigh,h_neigh)
plt.subplot(1,2,2)
plt.xlabel(r"$e_{cal}$",fontsize=12)
plt.ylabel(r"$h_{cal}$",fontsize=12)
for i in np.arange(len(neigh)):
    if len(neigh[i][0]) != 0 :
        plt.plot(neigh[i][0],neigh[i][1],'.',markersize=1)
plt.axis([0,lim,0,lim])
plt.title(r"neighbors for $e_{cal} \neq 0$",fontsize=12)
plt.tight_layout()
#plt.show()
savefig(fig,directory,classname+"_neighborhood.png")
plt.close()
