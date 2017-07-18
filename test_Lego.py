#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does CalibrationLego works.
"""

import matplotlib.pyplot as plt
import pfcalibration.usualplots as usplt               # usual plots function 
from pfcalibration.tools import importData,importCalib # to import binary data
from pfcalibration.tools import savefig                # to save a figure
import numpy as np


# file to save the pictures
directory = "pictures/testCalibrationLego/"
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
nbLego = 60
    
try:
    filename = "calibrations/CalibrationLego_162Kpart_delta_2.66856677123_ecal_max_262.853826966_hcal_max_262.853826966_lim_150_nbLego_100.calibration"
    CalibrationLego = importCalib(filename)
except FileNotFoundError:    
    # We create the calibration
    CalibrationLego = data1.CalibrationLego(nbLego=nbLego)
    CalibrationLego.saveCalib()
    
classname = CalibrationLego.classname
#plot 3D Training points
fig = plt.figure(1,figsize=(6, 4))
usplt.plot3D_training(data1)
plt.show()
savefig(fig,directory,classname+"_plot3D_training.png")
plt.close()

#plot 3D surface calibration
fig = plt.figure(1,figsize=(6, 4))
usplt.plot3D_surf(CalibrationLego)
plt.show()
savefig(fig,directory,classname+"_plot3D_surf.png")
savefig(fig,directory,classname+"_plot3D_surf.eps")
plt.close()

#courbe de calibration pour ecal = 0
fig = plt.figure(figsize=(12,4))
usplt.plotCalibrationCurve(CalibrationLego)
plt.show()
savefig(fig,directory,classname+"_calibration.png")
savefig(fig,directory,classname+"_calibration.eps")
plt.close()

#ecalib/true in function of etrue
fig = plt.figure(figsize=(12,4))
usplt.plot_ecalib_over_etrue_functionof_etrue(CalibrationLego,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue.png")
savefig(fig,directory,classname+"_ecalib_over_etrue.eps")
plt.close()

#histogram of ecalib and etrue
fig = plt.figure(figsize=(12,5))
usplt.hist_ecalib(CalibrationLego,data2)
plt.show()
savefig(fig,directory,classname+"_histograms_ecalib_etrue.png")
savefig(fig,directory,classname+"_histograms_ecalib_etrue.eps")
plt.close()

#ecalib/etrue in function of ecal,hcal
fig = plt.figure(figsize=(12,4))
usplt.plot_ecalib_over_etrue_functionof_ecal_hcal(CalibrationLego,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_functionof_ecal_hcal.png")
plt.close()

#ecalib/etrue gaussian fit curve
fig = plt.figure(figsize=(12,10))
usplt.plot_gaussianfitcurve_ecalib_over_etrue_functionof_ecal_hcal(CalibrationLego,data2)
plt.show()
savefig(fig,directory,classname+"_ecalib_over_etrue_curve.png")
savefig(fig,directory,classname+"_ecalib_over_etrue_curve.eps")
plt.close()

# Plot of the Legos
ecal = CalibrationLego.ecal
hcal = CalibrationLego.hcal
true = CalibrationLego.true
true[np.isnan(true)] = 0
ind = true != 0

fig = plt.figure(1,figsize=(6, 4))
ax = plt.axes(projection='3d')
x = ecal[ind]
y = hcal[ind]
z = np.zeros(len(x))
dx = CalibrationLego.delta*np.ones(len(x))
dy = CalibrationLego.delta*np.ones(len(y))
dz =  true[ind]
ax.bar3d(x,y,z,dx,dy,dz,color='yellow')
ax.view_init(20,-110)
ax.set_xlim([0,max(x)])
ax.set_ylim([0,max(y)])
ax.set_xlabel(r"$e_{cal}$",fontsize=20)
ax.set_ylabel(r"$h_{cal}$",fontsize=20)
ax.set_zlabel(r"$e_{true}$",fontsize=20)
plt.show()
savefig(fig,directory,classname+"_plot3D_legos.png")
savefig(fig,directory,classname+"_plot3D_legos.eps")
plt.close()
