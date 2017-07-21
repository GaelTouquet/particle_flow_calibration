#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does LinearRegression works.
"""

import matplotlib.pyplot as plt
import pfcalibration.usualplots as usplt
from pfcalibration.tools import savefig
from pfcalibration.tools import importData,importCalib


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
lim_min = 20
lim_max=80
lim=150

# file to save the pictures
directory = "pictures/testLinearRegression/"
try:
    # We import the calibration
    filename = "calibrations/LinearRegression_162Kpart_lim_150_lim_max_80_lim_min_20.calibration"
    LinearRegression = importCalib(filename)
except FileNotFoundError:
    # We create the calibration
    LinearRegression = data1.LinearRegression(lim_min = 20, lim_max=80, lim=150)
    # We save the calibration
    LinearRegression.saveCalib()
    print(LinearRegression)




classname = LinearRegression.classname
#plot 3D Training points
fig = plt.figure(1,figsize=(6, 4))
usplt.plot3D_training(data1)
plt.show()
savefig(fig,directory,classname+"_plot3D_training.png")
plt.close()

#plot 3D surface calibration
fig = plt.figure(1,figsize=(6, 4))
usplt.plot3D_surf(LinearRegression)
plt.show()
savefig(fig,directory,classname+"_plot3D_surf.png")
savefig(fig,directory,classname+"_plot3D_surf.eps")
plt.close()

#courbe de calibration pour ecal = 0
fig = plt.figure(figsize=(12,4))
usplt.plotCalibrationCurve(LinearRegression)
plt.show()
savefig(fig,directory,classname+"_calibration.png")
plt.close()

#ecalib/true in function of etrue
fig = plt.figure(figsize=(12,4))
usplt.plot_ecalib_over_etrue_functionof_etrue(LinearRegression,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue.png")
plt.close()

#histogram of ecalib and etrue
fig = plt.figure(figsize=(12,5))
usplt.hist_ecalib(LinearRegression,data2)
plt.show()
savefig(fig,directory,classname+"_histograms_ecalib_etrue.png")
savefig(fig,directory,classname+"_histograms_ecalib_etrue.eps")
plt.close()

#ecalib/etrue in function of ecal,hcal
fig = plt.figure(figsize=(12,4))
usplt.plot_ecalib_over_etrue_functionof_ecal_hcal(LinearRegression,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_functionof_ecal_hcal.png")
plt.close()

#ecalib/etrue gaussian fit curve
fig = plt.figure(figsize=(12,10))
usplt.plot_gaussianfitcurve_ecalib_over_etrue_functionof_ecal_hcal(LinearRegression,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_curve.png")
savefig(fig,directory,classname+"_ecalib_over_etrue_curve.eps")
plt.close()
