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
from pfcalibration.tools import gaussian_fit, binwidth_array
from mpl_toolkits.mplot3d import Axes3D



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
    plt.xlabel(r"$h_{cal}$",fontsize=20)
    plt.ylabel(r"$e_{true}$",fontsize=20)
    plt.title(r"$e_{cal} = 0$",fontsize=20)
    plt.axis([0,max(h),0,max(t)])
    plt.subplot(gs[1])
    plt.plot(hcal_train,true_train,'.',markersize=1)
    plt.plot(h,t,lw=2)
    plt.xlabel(r"$h_{cal}$",fontsize=20)
    plt.ylabel(r"$e_{true}$",fontsize=20)
    plt.title(r"$e_{cal} = 0$",fontsize=20)
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

def getMeans2D(energy_x,energy_y,z,lim):
    ind  = np.invert(np.isnan(z))
    z = z[ind]
    energy_x = energy_x[ind]
    energy_y = energy_y[ind]
    neighborhood = neighbors.NearestNeighbors(n_neighbors=250)
    neighborhood.fit(np.transpose(np.matrix([energy_x,energy_y])))
    step = 0.5
    ener_x = np.arange(0,lim+step,step)
    ener_y = np.arange(0,lim+step,step)
    sigma_gaussianfit = []
    mean_gaussianfit = []
    
    for y in ener_y:
        line_mean = []
        line_sigma = []
        for x in ener_x:
            if x+y < lim:
                dist, ind = neighborhood.kneighbors(X = [[x,y]])
                z_ind = z[ind]
                params = gaussian_fit(z_ind,binwidth=0.1,reducedChi2Max = 10)
                if math.isnan(params[0]):
                    line_mean.append(np.mean(z_ind))
                    line_sigma.append(np.sqrt(np.std(z_ind)))
                else:
                    line_mean.append(params[1])
                    line_sigma.append(params[0])
            else:
                line_mean.append(math.nan)
                line_sigma.append(math.nan)
        mean_gaussianfit.append(line_mean)
        sigma_gaussianfit.append(line_sigma)
    return mean_gaussianfit, sigma_gaussianfit

def plot_ecalib_over_etrue_functionof_etrue(calib,dataToPredict):
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
    plt.axis([0,200,0.5,1.5])
    plt.plot(energy,mean_gaussianfit,lw=3)
    plt.plot([0,200],[1,1],'--',lw = 2,color='yellow')
    plt.xlabel(r"$e_{true}$",fontsize=20)
    plt.ylabel(r"$e_{calib}/e_{true}$",fontsize=20)
    plt.title(r"$e_{cal} = 0$",fontsize=20)
    plt.subplot(1,2,2)
    plt.plot(t2,r2,'.',markersize=1)
    plt.axis([0,200,0.5,1.5])
    plt.plot(energy2,mean_gaussianfit2,lw=3)
    plt.plot([0,200],[1,1],'--',lw = 2,color='yellow')
    plt.xlabel(r"$e_{true}$",fontsize=20)
    plt.ylabel(r"$e_{calib}/e_{true}$",fontsize=20)
    plt.title(r"$e_{cal} \neq 0$",fontsize=20)
    plt.tight_layout()
    
def plot_ecalib_over_etrue_functionof_ecal_hcal(calib,dataToPredict):
    """
    plot ecalib/etrue = f(ecal,hcal)
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
    plt.subplot(1,2,1)
    plt.title(r" $e_{cal} = 0$",fontsize = 15)
    plt.plot(h,r,'.',markersize=1,label=r"$e_{calib}/e_{true}$")
    plt.plot([0,200],[1,1],'--',lw = 3, color = "yellow")
    plt.ylabel(r"$e_{calib}/e_{true}$",fontsize = 15)
    plt.xlabel(r"$h_{cal}$",fontsize = 15)
    plt.axis([0,calib.lim,0,2])
    energy_ecal_eq_0, means_ecal_eq_0, mean_gaussianfit_ecal_eq_0, sigma_gaussianfit_ecal_eq_0, reducedChi2_ecal_eq_0 = getMeans(h,r)
    plt.plot(energy_ecal_eq_0,mean_gaussianfit_ecal_eq_0,lw=3,label="mean (gaussian fit)")
    plt.plot(energy_ecal_eq_0,means_ecal_eq_0,lw=3,label="mean")
    plt.legend(loc='upper right')
    plt.subplot(1,2,2)
    Z_mean, Z_sigma = getMeans2D(e2,h2,r2,calib.lim)
    im = plt.imshow(Z_mean, cmap=plt.cm.seismic, extent=(0,calib.lim,0,calib.lim), origin='lower',vmin=0.9,vmax=1.1)
    plt.colorbar(im)
    plt.title(r"$e_{cal} \neq 0$",fontsize = 15)
    plt.xlabel(r"$e_{cal}$",fontsize = 15)
    plt.ylabel(r"$h_{cal}$",fontsize = 15)
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
    plt.hist(c,binwidth_array(c,2))
    plt.xlabel(r"$e_{calib}$",fontsize=20)
    plt.title(r"$e_{cal} = 0$",fontsize=20)
    plt.subplot(2,2,2)
    c2 = c2[np.invert(np.isnan(c2))]
    plt.hist(c2,binwidth_array(c2,2))
    plt.xlabel(r"$e_{calib}$",fontsize=20)
    plt.title(r"$e_{cal} \neq 0$",fontsize=20)
    plt.subplot(2,2,3)
    t = t[np.invert(np.isnan(t))]
    plt.hist(t,binwidth_array(t,2))
    plt.xlabel(r"$e_{true}$",fontsize=20)
    plt.title(r"$e_{cal} = 0$",fontsize=20)
    plt.subplot(2,2,4)
    t2 = t2[np.invert(np.isnan(t2))]
    plt.hist(t2,binwidth_array(t2,2))
    plt.xlabel(r"$e_{true}$",fontsize=20)
    plt.title(r"$e_{cal} \neq 0$",fontsize=20)
    plt.tight_layout()

def plot_gaussianfitcurve_ecalib_over_etrue_functionof_ecal_hcal(calib,dataToPredict):
    """
    plot the gaussian fit curve of ecalib/etrue = f(etrue)
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
    #mean
    ax = plt.subplot(4,1,1)
    plt.plot(energy,mean_gaussianfit,lw=3, label = r"$e_{cal} = 0$")
    plt.plot(energy2,mean_gaussianfit2,lw=3, label = r"$e_{cal} \neq 0$")
    plt.xlabel(r"$e_{true}$",fontsize=20)
    plt.title(r"$e_{calib}/e_{true}$",fontsize=20)
    plt.ylabel(r"$<e_{calib}/e_{true}>$",fontsize=20)
    plt.legend(loc='upper right')
    major_ticks = np.arange(0, 200, 50)
    minor_ticks = np.arange(0, 200, 10)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    # and a corresponding grid
    ax.grid(which='both')
    # or if you want differnet settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1)
    #sigma
    ax = plt.subplot(4,1,2)
    plt.plot(energy,sigma_gaussianfit,lw=3, label = r"$e_{cal} = 0$")
    plt.plot(energy2,sigma_gaussianfit2,lw=3, label = r"$e_{cal} \neq 0$")
    plt.xlabel(r"$e_{true}$",fontsize=20)
    plt.ylabel(r"$\sigma (e_{calib}/e_{true})$",fontsize=20)
    plt.legend(loc='upper right')
    major_ticks = np.arange(0, 200, 50)
    minor_ticks = np.arange(0, 200, 10)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    # and a corresponding grid
    ax.grid(which='both')
    # or if you want differnet settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1)
    #chi2
    ax = plt.subplot(4,1,3)
    plt.plot(energy,reducedChi2,lw=3, label = r"$e_{cal} = 0$")
    plt.plot(energy2,reducedChi22,lw=3, label = r"$e_{cal} \neq 0$")
    plt.xlabel(r"$e_{true}$",fontsize=20)
    plt.ylabel(r"$\chi^2/df$",fontsize=20)
    plt.legend(loc='upper right')
    major_ticks = np.arange(0, 200, 50)
    minor_ticks = np.arange(0, 200, 10)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    # and a corresponding grid
    ax.grid(which='both')
    # or if you want differnet settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1)
    #hist chi2/df for ecal == 0
    ax = plt.subplot(4,2,7)
    x = np.array(reducedChi2)
    x = x[np.invert(np.isnan(x))]
    bins = binwidth_array(x,binwidth = 0.5)
    plt.hist(x,bins,label = r"$e_{cal} = 0$")
    plt.xlabel(r"$\chi^2/df$",fontsize=20)
    plt.legend(loc='upper right')
    #hist chi2/df for ecal != 0
    ax = plt.subplot(4,2,8)
    x = np.array(reducedChi22)
    x = x[np.invert(np.isnan(x))]
    bins = binwidth_array(x,binwidth = 0.5)
    plt.hist(x,bins,label = r"$e_{cal} \neq 0$")
    plt.xlabel(r"$\chi^2/df$",fontsize=20)
    plt.legend(loc='upper right')
    plt.tight_layout()

def plot3D_training(data1):
    ax = plt.axes(projection='3d')
    ax.scatter(data1.ecal, data1.hcal, data1.true,s=1)
    ax.view_init(10,280)
    ax.set_xlim([0,data1.ecal_max])
    ax.set_ylim([0,data1.hcal_max])
    ax.set_zlim([0,data1.true_max])
    ax.set_title("Training points")
    ax.set_xlabel(r'$e_{cal}$',fontsize=20)
    ax.set_ylabel(r'$h_{cal}$',fontsize=20)
    ax.set_zlabel(r'$e_{true}$',fontsize=20)

    
def plot3D_surf(calib):
    """
    Plot the surface of the calibration
    Parameters
    ----------
    calib : the calibration
    """
    ecal = np.arange(0,calib.lim,3)
    hcal = np.arange(0,calib.lim,3)
    ecal,hcal = np.meshgrid(ecal,hcal)
    ecalib = calib.predict(ecal,hcal)
    ax = plt.axes(projection='3d')
    ax.plot_wireframe(ecal,hcal,ecalib,color='red')
    ax.view_init(10,280)
    ax.set_xlim([0,max(calib.ecal_train)])
    ax.set_ylim([0,max(calib.hcal_train)])
    ax.set_zlim([0,max(calib.true_train)])
    ax.set_title("Calibration surface")
    ax.set_xlabel(r'$e_{cal}$',fontsize=20)
    ax.set_ylabel(r'$h_{cal}$',fontsize=20)
    ax.set_zlabel(r'$e_{true}$',fontsize=20)
    
def comparison(calibs,dataToPredict):
    """
    To compare the comparisons
    
    Parameters
    ----------
    calibs : array of calibration
    """
    
    calib = calibs[0]
    h = dataToPredict.hcal[np.logical_and(dataToPredict.ecal == 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    t = dataToPredict.true[np.logical_and(dataToPredict.ecal == 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    e = np.zeros(len(h))
    h2 = dataToPredict.hcal[np.logical_and(dataToPredict.ecal != 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    t2 = dataToPredict.true[np.logical_and(dataToPredict.ecal != 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    e2 = dataToPredict.ecal[np.logical_and(dataToPredict.ecal != 0,dataToPredict.ecal+dataToPredict.hcal < calib.lim)]
    
    r = []
    r2 = []
    
    for calib in calibs:
        c = calib.predict(e,h)
        r.append(c/t)
        c2 = calib.predict(e2,h2)
        r2.append(c2/t2)
    
    ax = plt.subplot(2,2,1)
    plt.plot([0,200],[1,1],'--',lw = 2,color='black')
    for i in np.arange(len(calibs)):
        energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2 = getMeans(t,r[i])
        plt.plot(energy,mean_gaussianfit,label = calibs[i].classname,lw=2)
    plt.xlabel(r"$e_{true}$",fontsize=20)
    plt.ylabel(r"$<e_{calib}/e_{true}>$",fontsize=20)
    plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=20)
    plt.axis([0,160,0.8,1.2])
    plt.legend(loc="upper right")
    major_ticks = np.arange(0, 200, 20)
    minor_ticks = np.arange(0, 200, 2)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    # and a corresponding grid
    ax.grid(which='both')
    # or if you want differnet settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1)
    
    ax = plt.subplot(2,2,2)
    plt.plot([0,200],[1,1],'--',lw = 2,color='black')
    for i in np.arange(len(calibs)):
        energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2 = getMeans(t2,r2[i])
        plt.plot(energy,mean_gaussianfit,label = calibs[i].classname,lw=2)
    plt.xlabel(r"$e_{true}$",fontsize=20)
    plt.ylabel(r"$<e_{calib}/e_{true}>$",fontsize=20)
    plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} \neq 0$",fontsize=20)
    plt.axis([0,160,0.8,1.2])
    plt.legend(loc="upper right")
    major_ticks = np.arange(0, 200, 20)
    minor_ticks = np.arange(0, 200, 2)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    # and a corresponding grid
    ax.grid(which='both')
    # or if you want differnet settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1)
    
    ax = plt.subplot(2,2,3)
    plt.plot([0,200],[1,1],'--',lw = 2,color='black')
    for i in np.arange(len(calibs)):
        energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2 = getMeans(t,r[i])
        plt.plot(energy,sigma_gaussianfit,label = calibs[i].classname,lw=2)
    plt.xlabel(r"$e_{true}$",fontsize=20)
    plt.ylabel(r"$\sigma(e_{calib}/e_{true})$",fontsize=20)
    plt.title(r"$\sigma(e_{calib}/e_{true})$ for $e_{cal} = 0$",fontsize=20)
    plt.axis([0,160,0,0.6])
    plt.legend(loc="upper right")
    major_ticks = np.arange(0, 200, 20)
    minor_ticks = np.arange(0, 200, 2)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    # and a corresponding grid
    ax.grid(which='both')
    # or if you want differnet settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1)
    
    ax = plt.subplot(2,2,4)
    plt.plot([0,200],[1,1],'--',lw = 2,color='black')
    for i in np.arange(len(calibs)):
        energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2 = getMeans(t2,r2[i])
        plt.plot(energy,sigma_gaussianfit,label = calibs[i].classname,lw=2)
    plt.xlabel(r"$e_{true}$",fontsize=20)
    plt.ylabel(r"$\sigma(e_{calib}/e_{true})$",fontsize=20)
    plt.title(r"$\sigma(e_{calib}/e_{true})$ for $e_{cal} \neq 0$",fontsize=20)
    plt.axis([0,160,0,0.6])
    plt.legend(loc="upper right")
    major_ticks = np.arange(0, 200, 20)
    minor_ticks = np.arange(0, 200, 2)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    # and a corresponding grid
    ax.grid(which='both')
    # or if you want differnet settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=1)
    
    plt.tight_layout()
    
