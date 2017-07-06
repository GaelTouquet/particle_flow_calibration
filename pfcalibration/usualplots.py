#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

import numpy as np
import math
from matplotlib import gridspec
import matplotlib.pyplot as plt
from sklearn import neighbors
from pfcalibration.tools import gaussian_fit, gaussian_param, binwidth_array


def plotCalibrationCurve(calib):
    """
    Calibration Curve for ecal = 0
    """
    # Training data
    hcal_train = calib.hcal_train[calib.ecal_train == 0]
    true_train = calib.true_train[calib.ecal_train == 0]
    # the curve
    h = np.arange(min(hcal_train),calib.lim,0.1)
    e = np.zeros(len(h))
    t = calib.predict(e,h)
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1]) 
    plt.subplot(gs[0])
    plt.plot(hcal_train,true_train,'.',markersize=1)
    plt.plot(h,t,lw=2)
    plt.xlabel(r"$h_{cal}$",fontsize=12)
    plt.ylabel(r"$e_{true}$",fontsize=12)
    plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=12)
    plt.axis([0,max(h),0,max(t)])
    plt.subplot(gs[1])
    plt.plot(hcal_train,true_train,'.',markersize=1)
    plt.plot(h,t,lw=2)
    plt.xlabel(r"$h_{cal}$",fontsize=12)
    plt.ylabel(r"$e_{true}$",fontsize=12)
    plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=12)
    plt.axis([0,12,0,calib.predict(0,12)])
    plt.tight_layout()

def getMeans(energy_x,y):
    ind  = np.invert(np.isnan(y))
    y = y[ind]
    energy_x = energy_x[ind]

    neighborhood = neighbors.NearestNeighbors(n_neighbors=1000)
    neighborhood.fit(np.transpose(np.matrix(energy_x)))
    step = 0.1
    ener = np.arange(min(energy_x),max(energy_x),step)
    sigma_gaussianfit = []
    mean_gaussianfit = []
    means = []
    energy = []
    reducedChi2 = []
    for e in ener:
        dist, ind = neighborhood.kneighbors(X = e)
        y_ind = y[ind][np.invert(np.isnan(y[ind]))]
        params,reduced = gaussian_fit(y_ind,binwidth = 0.1,giveReducedChi2 = True,reducedChi2Max = 10)
        if not(math.isnan(params[1])):
            means.append(np.mean(y_ind))
            sigma_gaussianfit.append(params[0])
            mean_gaussianfit.append(params[1])
            energy.append(e)
            reducedChi2.append(reduced)
    return energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2

def ecalib_over_etrue_functionof_etrue(calib,dataToPredict):
    """
    plot ecalib/etrue = f(etrue)
    """
    h = dataToPredict.hcal[np.logical_and(dataToPredict.ecal == 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    t = dataToPredict.true[np.logical_and(dataToPredict.ecal == 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    e = np.zeros(len(h))
    h2 = dataToPredict.hcal[np.logical_and(dataToPredict.ecal != 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    t2 = dataToPredict.true[np.logical_and(dataToPredict.ecal != 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    e2 = dataToPredict.ecal[np.logical_and(dataToPredict.ecal != 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    
    c = calib.predict(e,h) 
    r = c/t
    c2 = calib.predict(e2,h2)
    r2 = c2/t2
    
    energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2 = getMeans(t,r)
    energy2, means2, mean_gaussianfit2, sigma_gaussianfit2, reducedChi22 = getMeans(t2,r2)
    
    plt.subplot(1,2,1)
    plt.plot(t,r,'.',markersize=1)
    plt.axis([0,200,0,2])
    plt.plot(energy,mean_gaussianfit,lw=3)
    plt.xlabel(r"$e_{true}$",fontsize=12)
    plt.ylabel(r"$e_{calib}/e_{true}$",fontsize=12)
    plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=12)
    plt.subplot(1,2,2)
    plt.plot(t2,r2,'.',markersize=1)
    plt.axis([0,200,0,2])
    plt.plot(energy2,mean_gaussianfit2,lw=3)
    plt.xlabel(r"$e_{true}$",fontsize=12)
    plt.ylabel(r"$e_{calib}/e_{true}$",fontsize=12)
    plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=12)
    plt.tight_layout()
    
def hist_ecalib(calib,dataToPredict):
    """
    Histogram of ecalib and etrue
    """
    h = dataToPredict.hcal[np.logical_and(dataToPredict.ecal == 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    t = dataToPredict.true[np.logical_and(dataToPredict.ecal == 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    e = np.zeros(len(h))
    h2 = dataToPredict.hcal[np.logical_and(dataToPredict.ecal != 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    t2 = dataToPredict.true[np.logical_and(dataToPredict.ecal != 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    e2 = dataToPredict.ecal[np.logical_and(dataToPredict.ecal != 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    
    c = calib.predict(e,h) 
    c2 = calib.predict(e2,h2)

    plt.subplot(2,2,1)
    plt.hist(c,binwidth_array(c))
    plt.xlabel(r"$e_{calib}$",fontsize=12)
    plt.title(r"$e_{calib}$ for $e_{cal} = 0$",fontsize=12)
    plt.subplot(2,2,2)
    c2 = c2[np.invert(np.isnan(c2))]
    plt.hist(c2,binwidth_array(c2))
    plt.xlabel(r"$e_{calib}$",fontsize=12)
    plt.title(r"$e_{calib}$ for $e_{cal} \neq 0$",fontsize=12)
    plt.subplot(2,2,3)
    t = t[np.invert(np.isnan(t))]
    plt.hist(t,binwidth_array(t))
    plt.xlabel(r"$e_{true}$",fontsize=12)
    plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=12)
    plt.subplot(2,2,4)
    t2 = t2[np.invert(np.isnan(t2))]
    plt.hist(t2,binwidth_array(t2))
    plt.xlabel(r"$e_{true}$",fontsize=12)
    plt.title(r"$e_{true}$ for $e_{cal} \neq 0$",fontsize=12)
    plt.tight_layout()



