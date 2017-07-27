#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does KNN works.
"""

import matplotlib.pyplot as plt
import pfcalibration.usualplots as usplt               # usual plots function 
from pfcalibration.tools import importData,importCalib # to import binary data
from pfcalibration.tools import savefig                # to save a figure
import numpy as np

# file to save the pictures
directory = "pictures/testKNN/"
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
    filename = "calibrations/KNN_162Kpart_lim_150_n_neighbors_ecal_eq_0_2000_n_neighbors_ecal_neq_0_250_recalibrated_False_sigma_5.calibration"
    KNN = importCalib(filename)
except FileNotFoundError:    
    # We create the calibration
    KNN = data1.KNN(n_neighbors_ecal_eq_0,n_neighbors_ecal_neq_0,
                                 weights,algorithm,sigma,lim)
    KNN.saveCalib()
    print(KNN)
classname = KNN.classname

#plot 3D Training points
fig = plt.figure(1,figsize=(7, 7))
usplt.plot3D_training(data1)
plt.show()
savefig(fig,directory,classname+"_plot3D_training.png")
plt.close()

#plot 3D surface calibration
fig = plt.figure(1,figsize=(7, 7))
usplt.plot3D_surf(KNN)
plt.show()
savefig(fig,directory,classname+"_plot3D_surf.png")
savefig(fig,directory,classname+"_plot3D_surf.eps")
plt.close()

#courbe de calibration pour ecal = 0
fig = plt.figure(figsize=(12,4))
usplt.plotCalibrationCurve(KNN)
plt.show()
savefig(fig,directory,classname+"_calibration.png")
plt.close()

#ecalib/true in function of etrue
fig = plt.figure(figsize=(12,4))
usplt.plot_ecalib_over_etrue_functionof_etrue(KNN,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue.png")
plt.close()

#histogram of ecalib and etrue
fig = plt.figure(figsize=(12,5))
usplt.hist_ecalib(KNN,data2)
plt.show()
savefig(fig,directory,classname+"_histograms_ecalib_etrue.png")
savefig(fig,directory,classname+"_histograms_ecalib_etrue.eps")
plt.close()

#ecalib/etrue in function of ecal,hcal
fig = plt.figure(figsize=(12,4))
usplt.plot_ecalib_over_etrue_functionof_ecal_hcal(KNN,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_functionof_ecal_hcal.png")
plt.close()

#ecalib/etrue gaussian fit curve
fig = plt.figure(figsize=(12,10))
usplt.plot_gaussianfitcurve_ecalib_over_etrue_functionof_ecal_hcal(KNN,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_curve.png")
savefig(fig,directory,classname+"_ecalib_over_etrue_curve.eps")
plt.close()


#NEIGHBORS
fig = plt.figure(figsize=(12,6))
hcal_train = KNN.hcal_train[KNN.ecal_train == 0]
true_train = KNN.true_train[KNN.ecal_train == 0]
plt.subplot(1,2,1)
plt.plot(hcal_train,true_train,'.',markersize=1)
plt.xlabel(r"$E_{\rm hcal} \rm{(GeV)}$",fontsize=20)
plt.ylabel(r"$E_{\rm true} \rm{(GeV)}$",fontsize=20)
for i in np.arange(0,KNN.lim,30):
    ind = KNN.neigh_ecal_eq_0.kneighbors([[i]])[1][0]
    hcal_neigh = hcal_train[ind]
    true_neigh = true_train[ind]
    plt.plot(hcal_neigh,true_neigh,'.',color='red',markersize=1)
plt.title(r"neighbors for $E_{\rm ecal} = 0$",fontsize=18)
plt.axis([0,lim,0,lim])


#neigh for ecal != 0
plt.subplot(1,2,2)
ecal_train = KNN.ecal_train[KNN.ecal_train != 0]
hcal_train = KNN.hcal_train[KNN.ecal_train != 0]
plt.xlabel(r"$E_{\rm ecal} \rm{(GeV)}$",fontsize=20)
plt.ylabel(r"$E_{\rm hcal} \rm{(GeV)}$",fontsize=20)
for i in np.arange(0,KNN.lim,14):
    for j in np.arange(0,KNN.lim,14):
        if i+j < KNN.lim:
            ind = KNN.neigh_ecal_neq_0.kneighbors([[i,j]])[1][0]
            ecal_neigh = ecal_train[ind]
            hcal_neigh = hcal_train[ind]
            plt.plot(ecal_neigh,hcal_neigh,'.',markersize=1)
plt.title(r"neighbors for $E_{\rm ecal} \neq 0$",fontsize=18)
plt.axis([0,lim,0,lim])
plt.tight_layout()
plt.show()
savefig(fig,directory,classname+"_neighborhood.png")
