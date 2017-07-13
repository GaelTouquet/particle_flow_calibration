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
    
classname = KNNGC.classname
#plot 3D Training points
fig = plt.figure(1,figsize=(5, 5))
usplt.plot3D_training(data1)
plt.show()
savefig(fig,directory,classname+"_plot3D_training.png")

#plot 3D surface calibration
fig = plt.figure(1,figsize=(5, 5))
usplt.plot3D_surf(KNNGC)
plt.show()
savefig(fig,directory,classname+"_plot3D_surf.png")
plt.close()

#courbe de calibration pour ecal = 0
fig = plt.figure(figsize=(10,4))
usplt.plotCalibrationCurve(KNNGC)
plt.show()
savefig(fig,directory,classname+"_calibration.png")
plt.close()

#ecalib/true in function of etrue
fig = plt.figure(figsize=(10,4))
usplt.plot_ecalib_over_etrue_functionof_etrue(KNNGC,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue.png")
plt.close()

#histogram of ecalib and etrue
fig = plt.figure(figsize=(10,6))
usplt.hist_ecalib(KNNGC,data2)
plt.show()
savefig(fig,directory,classname+"_histograms_ecalib_etrue.png")
plt.close()

#ecalib/etrue in function of ecal,hcal
fig = plt.figure(figsize=(10,5))
usplt.plot_ecalib_over_etrue_functionof_ecal_hcal(KNNGC,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_functionof_ecal_hcal.png")
plt.close()

#ecalib/etrue gaussian fit curve
fig = plt.figure(figsize=(10,12))
usplt.plot_gaussianfitcurve_ecalib_over_etrue_functionof_ecal_hcal(KNNGC,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_curve.png")
plt.close()
