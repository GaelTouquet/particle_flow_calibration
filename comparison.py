#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to compare all calibrations.
"""

from pfcalibration.tools import importData, importCalib # to import binary data


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

# We import LR
filename = "calibrations/LinearRegression_162Kpart_lim_150_lim_max_80_lim_min_20.calibration"
LinearRegression = importCalib(filename)

# We import KNNGFD
filename = "calibrations/KNNGaussianFitDirect_162Kpart_ecal_train_ecal_neq_0_min_0.292026728392_hcal_train_ecal_eq_0_min_1.00043606758_hcal_train_ecal_neq_0_min_1.00002634525_lim_150_n_neighbors_ecal_eq_0_2000_n_neighbors_ecal_neq_0_250.calibration"
KNNGFD = importCalib(filename)

# We import KNNGF
filename = "calibrations/KNNGaussianFit_162Kpart_hcal_train_ecal_eq_0_min_1.00043606758_lim_150_n_neighbors_ecal_eq_0_2000_n_neighbors_ecal_neq_0_250.calibration"
KNNGF = importCalib(filename)

# We import KNNGC
filename = "calibrations/KNNGaussianCleaning_162Kpart_cut_2_lim_150_n_neighbors_ecal_eq_0_2000_n_neighbors_ecal_neq_0_250_sigma_5.calibration"
KNNGC = importCalib(filename)

# We import KNN
filename = "calibrations/KNN_162Kpart_lim_150_n_neighbors_ecal_eq_0_2000_n_neighbors_ecal_neq_0_250_recalibrated_False_sigma_5.calibration"
KNN = importCalib(filename)