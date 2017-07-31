#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to compare all calibrations.
"""

from pfcalibration.tools import importData, importCalib # to import binary data
import pfcalibration.usualplots as usplt
from pfcalibration.tools import savefig
import matplotlib.pyplot as plt



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

directory = "pictures/comparisons/"

# We import LR
try:
    # We import the calibration
    filename = "calibrations/LinearRegression_162Kpart_lim_150_lim_max_150_lim_min_10.calibration"
    LR = importCalib(filename)
except FileNotFoundError:
    lim_min = 10
    lim_max = 150
    lim = 150
    # We create the calibration
    LR = data1.LinearRegression(lim_min = lim_min, lim_max=lim_max, lim=lim)
    # We save the calibration
    LR.saveCalib()
    print(LR)

# We import KNNGFD
try:
    # We import the calibration
    filename = "calibrations/KNNGaussianFitDirect_162Kpart_ecal_train_ecal_neq_0_min_0.292026728392_hcal_train_ecal_eq_0_min_1.00043606758_hcal_train_ecal_neq_0_min_1.00002634525_lim_150_n_neighbors_ecal_eq_0_2000_n_neighbors_ecal_neq_0_250.calibration"
    KNNGFD = importCalib(filename)
except FileNotFoundError:
    # We create the calibration
    lim = 150                   # if ecal + hcal > lim, ecalib = math.nan
    n_neighbors_ecal_eq_0=2000  # number of neighbors for ecal = 0
    n_neighbors_ecal_neq_0=250  # number of neighbors for ecal ≠ 0
    energystep_ecal_eq_0 = 1
    energystep_ecal_neq_0 = 5
    KNNGFD = data1.KNNGaussianFitDirect(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                                        n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                                        lim=lim)
    KNNGFD.saveCalib()

# We import KNNGF  
try:
    # We import the calibration
    filename = "calibrations/KNNGaussianFit_162Kpart_hcal_train_ecal_eq_0_min_1.00043606758_lim_150_n_neighbors_ecal_eq_0_2000_n_neighbors_ecal_neq_0_250.calibration"
    KNNGF = importCalib(filename)
except FileNotFoundError:
    # We create the calibration
    lim = 150                   # if ecal + hcal > lim, ecalib = math.nan
    n_neighbors_ecal_eq_0=2000  # number of neighbors for ecal = 0
    n_neighbors_ecal_neq_0=250  # number of neighbors for ecal ≠ 0
    energystep_ecal_eq_0 = 1
    energystep_ecal_neq_0 = 5
    kind = 'cubic'
    KNNGF = data1.KNNGaussianFit(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                                 n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                                 lim=lim,energystep_ecal_eq_0=energystep_ecal_eq_0,energystep_ecal_neq_0=energystep_ecal_neq_0,kind=kind)
    KNNGF.saveCalib()
    print(KNNGF)

# We import KNNGC
try:
    # We import the calibration
    filename = "calibrations/KNNGaussianCleaning_162Kpart_cut_2_lim_150_n_neighbors_ecal_eq_0_2000_n_neighbors_ecal_neq_0_250_sigma_5.calibration"
    KNNGC = importCalib(filename)
except FileNotFoundError:
    # We create the calibration
    n_neighbors_ecal_eq_0 = 2000
    n_neighbors_ecal_neq_0 = 250
    weights = 'gaussian'
    algorithm = 'auto'
    sigma = 5
    lim = 150
    energystep = 1
    kind = 'cubic'
    cut = 2
    KNNGC = data1.KNNGaussianCleaning(n_neighbors_ecal_eq_0,n_neighbors_ecal_neq_0,
                             weights,algorithm,sigma,lim,energystep,kind,cut)
    KNNGC.saveCalib()
    print(KNNGC)

# We import KNN    
try:
    filename = "calibrations/KNN_162Kpart_lim_150_n_neighbors_ecal_eq_0_2000_n_neighbors_ecal_neq_0_250_recalibrated_False_sigma_5.calibration"
    KNN = importCalib(filename)
except FileNotFoundError:    
    # We create the calibration
    n_neighbors_ecal_eq_0 = 2000
    n_neighbors_ecal_neq_0 = 250
    weights = 'gaussian'
    algorithm = 'auto'
    sigma = 5
    lim = 150
    energystep = 1
    kind = 'cubic'
    KNN = data1.KNN(n_neighbors_ecal_eq_0,n_neighbors_ecal_neq_0,
                                 weights,algorithm,sigma,lim)
    KNN.saveCalib()
    print(KNN)

# We import CL  
try:
    filename = "calibrations/CalibrationLego_162Kpart_delta_2.66856677123_ecal_max_262.853826966_hcal_max_262.853826966_lim_150_nbLego_100.calibration"
    CL = importCalib(filename)
except FileNotFoundError:    
    # We create the calibration
    nbLego = 100
    CL = data1.CalibrationLego(nbLego=nbLego)
    CL.saveCalib()
    print(CL)

calibs = [LR,CL,KNN,KNNGC,KNNGF]
fig = plt.figure(figsize=(12,5))
usplt.comparison_ecaliboveretrue_ecal_eq_0(calibs,data2)
plt.show()
savefig(fig,directory,"comparison1.png")
savefig(fig,directory,"comparison1.eps")
plt.close()

calibs = [LR,CL,KNN,KNNGC,KNNGF]
fig = plt.figure(figsize=(12,10))
usplt.comparison(calibs,data2)
plt.show()
savefig(fig,directory,"comparison2.png")
savefig(fig,directory,"comparison2.eps")
plt.close()

for calib in calibs :
    fig = plt.figure(figsize=(6,4))
    usplt.plot_ecalib_over_etrue_functionof_ecal_hcal_ecal_neq_0(calib,data2)
    plt.show()
    savefig(fig,directory,"ecaliboveretrue_"+calib.classname+".png")
    savefig(fig,directory,"ecaliboveretrue_"+calib.classname+".eps")
    plt.close()
    