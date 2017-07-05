#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does KNNGFD works.
"""

import matplotlib.pyplot as plt
import numpy as np
from pfcalibration.tools import gaussian_fit, binwidth_array, importPickle, savefig
from sklearn import neighbors
import math
from matplotlib import gridspec



font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
fontgreen = {'family': 'serif',
        'color':  'green',
        'weight': 'normal',
        'size': 16,
        }


directory = "pictures/testKNNGFD/"


filename = 'charged_hadrons_100k.energydata'
data1 = importPickle(filename)
filename = 'prod2_200_400k.energydata'
#on fusionne les 2 jeux de données
data1 = data1.mergeWith(importPickle(filename))
#on sépare data en 2
data1,data2 = data1.splitInTwo()
#data 1 -> training data
#data 2 -> data to predict

# paramètres de calibration
lim = 150
n_neighbors_ecal_eq_0=2000
n_neighbors_ecal_neq_0=200


def getMeans(energy_x,y,n_neighbors=n_neighbors_ecal_eq_0):
    ind  = np.invert(np.isnan(y))
    y = y[ind]
    energy_x = energy_x[ind]

    neighborhood = neighbors.NearestNeighbors(n_neighbors=1000)
    neighborhood.fit(np.transpose(np.matrix(energy_x)))
    step = 0.5
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

def getMeans2D(energy_x,energy_y,z):
    ind  = np.invert(np.isnan(z))
    z = z[ind]
    energy_x = energy_x[ind]
    energy_y = energy_y[ind]

    neighborhood = neighbors.NearestNeighbors(n_neighbors=n_neighbors_ecal_neq_0)
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


KNNGFD = data1.kNNGaussianFitDirect(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,lim=lim)
classname = type(KNNGFD).__name__


#courbe de calibration pour ecal = 0
hcal_train = KNNGFD.hcal_train_ecal_eq_0
true_train = KNNGFD.true_train_ecal_eq_0
h = np.arange(min(hcal_train),lim,0.1)
e = np.zeros(len(h))
t = KNNGFD.predict(e,h)

fig = plt.figure(figsize=(10,4))
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
plt.axis([0,12,0,15])
plt.show()
savefig(fig,directory,classname+"calibration.png")

#neigh for ecal == 0
h_neigh = np.arange(10,lim,50)
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
plt.show()
savefig(fig,directory,classname+"neighborhood.png")

#ecalib/etrue pour ecal = 0
h = data2.hcal[np.logical_and(data2.ecal == 0,data2.ecal+data2.hcal < lim)]
t = data2.true[np.logical_and(data2.ecal == 0,data2.ecal+data2.hcal < lim)]
e = np.zeros(len(h))

c = KNNGFD.predict(e,h)
r = c/t

energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2 = getMeans(t,r)
fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(t,r,'.',markersize=1)
plt.axis([0,200,0,2])
plt.plot(energy,mean_gaussianfit,lw=3)
plt.xlabel(r"$e_{true}$",fontsize=12)
plt.ylabel(r"$e_{calib}/e_{true}$",fontsize=12)
plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=12)

h2 = data2.hcal[np.logical_and(data2.ecal != 0,data2.ecal+data2.hcal < lim)]
t2 = data2.true[np.logical_and(data2.ecal != 0,data2.ecal+data2.hcal < lim)]
e2 = data2.ecal[np.logical_and(data2.ecal != 0,data2.ecal+data2.hcal < lim)]
c2 = KNNGFD.predict(e2,h2)
r2 = c2/t2

energy2, means2, mean_gaussianfit2, sigma_gaussianfit2, reducedChi22 = getMeans(t2,r2)
plt.subplot(1,2,2)
plt.plot(t2,r2,'.',markersize=1)
plt.axis([0,200,0,2])
plt.plot(energy2,mean_gaussianfit2,lw=3)
plt.xlabel(r"$e_{true}$",fontsize=12)
plt.ylabel(r"$e_{calib}/e_{true}$",fontsize=12)
plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=12)
plt.show()
savefig(fig,directory,classname+"ecalib_over_etrue_functionof_etrue.png")


#ecalib/etrue in function of ecal,hcal
fig = plt.figure(figsize=(12,5))
gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3]) 
plt.subplot(gs[0])
plt.title(classname+r" $e_{cal} = 0$",fontsize = 15)
plt.plot(h,r,'.',markersize=1,label=r"$e_{calib}/e_{true}$")
plt.plot([0,200],[1,1],'--',lw = 3, color = "yellow")
plt.ylabel(r"$e_{calib}/e_{true}$",fontsize = 15)
plt.xlabel(r"$h_{cal}$",fontsize = 15)
plt.axis([0,lim,0,2])
energy_ecal_eq_0, means_ecal_eq_0, mean_gaussianfit_ecal_eq_0, sigma_gaussianfit_ecal_eq_0, reducedChi2_ecal_eq_0 = getMeans(h,r)
plt.plot(energy_ecal_eq_0,mean_gaussianfit_ecal_eq_0,lw=3,label="mean (gaussian fit)")
plt.plot(energy_ecal_eq_0,means_ecal_eq_0,lw=3,label="mean")
plt.legend(loc='upper right')
plt.subplot(gs[1])
Z_mean, Z_sigma = getMeans2D(e2,h2,r2)
im = plt.imshow(Z_mean, cmap=plt.cm.seismic, extent=(0,lim,0,lim), origin='lower',vmin=0.9,vmax=1.1)
plt.colorbar(im)
plt.title(r" $e_{calib}/e_{true}$ for $e_{cal} \neq 0$",fontsize = 15)
plt.xlabel(r"$e_{cal}$",fontsize = 15)
plt.ylabel(r"$h_{cal}$",fontsize = 15)
plt.show()
savefig(fig,directory,classname+"ecalib_over_etrue_functionof_ecal_hcal.png")


#histogram of ecalib and etrue
fig = plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
c = c[np.invert(np.isnan(c))]
c = np.array(c)
bw = binwidth_array(c,1)
plt.hist(c,bw)
plt.xlabel(r"$e_{calib}$",fontsize=12)
plt.title(r"$e_{calib}$ for $e_{cal} = 0$",fontsize=12)
plt.subplot(2,2,2)
c2 = c2[np.invert(np.isnan(c2))]
c2 = np.array(c2)
bw = binwidth_array(c2,1)
plt.hist(c2,bw)
plt.xlabel(r"$e_{calib}$",fontsize=12)
plt.title(r"$e_{calib}$ for $e_{cal} \neq 0$",fontsize=12)
plt.subplot(2,2,3)
t = t[np.invert(np.isnan(t))]
t = np.array(t)
bw = binwidth_array(t,1)
plt.hist(t,bw)
plt.xlabel(r"$e_{true}$",fontsize=12)
plt.title(r"$e_{true}$ for $e_{cal} = 0$",fontsize=12)
plt.subplot(2,2,4)
t2 = t2[np.invert(np.isnan(t2))]
t2 = np.array(t2)
bw = binwidth_array(t2,1)
plt.hist(t2,bw)
plt.xlabel(r"$e_{true}$",fontsize=12)
plt.title(r"$e_{true}$ for $e_{cal} \neq 0$",fontsize=12)
plt.show()
savefig(fig,directory,classname+"histograms_ecalib_etrue.png")

fig = plt.figure(figsize=(10,12))
#mean
ax = plt.subplot(4,1,1)
plt.plot(energy,mean_gaussianfit,lw=3, label = r"$e_{cal} = 0$")
plt.plot(energy2,mean_gaussianfit2,lw=3, label = r"$e_{cal} \neq 0$")
plt.xlabel(r"$e_{true}$",fontsize=12)
plt.title(r"$e_{calib}/e_{true}$ for $e_{cal} = 0$",fontsize=12)
plt.ylabel(r"$<e_{calib}/e_{true}>$",fontsize=12)
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
plt.xlabel(r"$e_{true}$",fontsize=12)
plt.ylabel(r"$\sigma (e_{calib}/e_{true})$",fontsize=12)
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
plt.xlabel(r"$e_{true}$",fontsize=12)
plt.ylabel(r"$\chi^2/df$",fontsize=12)
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
plt.xlabel(r"$\chi^2/df$",fontsize=12)
plt.legend(loc='upper right')
#hist chi2/df for ecal != 0
ax = plt.subplot(4,2,8)
x = np.array(reducedChi22)
x = x[np.invert(np.isnan(x))]
bins = binwidth_array(x,binwidth = 0.5)
plt.hist(x,bins,label = r"$e_{cal} \neq 0$")
plt.xlabel(r"$\chi^2/df$",fontsize=12)
plt.legend(loc='upper right')
plt.show()
savefig(fig,directory,classname+"ecalib_over_etrue_curve.png")
