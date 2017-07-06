#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

Script to understand how does KNNGF works.
"""

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from pfcalibration.tools import importPickle, savefig
from pfcalibration.tools import gaussian_fit, gaussian_param, binwidth_array
from sklearn import neighbors
import math


# file to save the pictures
directory = "pictures/testKNNGF/"
#importation of simulated particles
filename = 'charged_hadrons_100k.energydata'
data1 = importPickle(filename)
filename = 'prod2_200_400k.energydata'
data2 = importPickle(filename)
# we merge the 2 sets of data
data1 = data1.mergeWith(importPickle(filename))
# we split the data in 2 sets
data1,data2 = data1.splitInTwo()
#data 1 -> training data
#data 2 -> data to predict

# parameters of the calibration
lim = 150
n_neighbors_ecal_eq_0=2000
n_neighbors_ecal_neq_0=200
energystep = 1

def getMeans(energy_x,y,n_neighbors=n_neighbors_ecal_eq_0):
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


KNNGF = data1.kNNGaussianFit(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                             n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                             lim=lim,energystep=energystep,kind='cubic')
classname = type(KNNGF).__name__

#chi2 for each point of the calibration
hcal_ecal_eq_0 = KNNGF.evaluatedPoint_hcal_ecal_eq_0
rchi2_ecal_eq_0 = KNNGF.evaluatedPoint_reducedchi2_ecal_eq_0
ecal_ecal_neq_0 = KNNGF.interpolation_ecal_neq_0.x
hcal_ecal_neq_0 = KNNGF.interpolation_ecal_neq_0.y
eecal_ecal_neq_0,hhcal_ecal_neq_0 = np.meshgrid(ecal_ecal_neq_0,hcal_ecal_neq_0)
rchi2_ecal_neq_0 = KNNGF.evaluatedPoint_reducedchi2[1:]
rrchi2_ecal_neq_0 = np.resize(rchi2_ecal_neq_0,(len(hcal_ecal_neq_0),len(ecal_ecal_neq_0)))
rrchi2_ecal_neq_0[eecal_ecal_neq_0 + hhcal_ecal_neq_0 > lim] = math.nan
x = np.array(KNNGF.evaluatedPoint_reducedchi2)
x = x[np.array(KNNGF.evaluatedPoint_hcal)+np.array(KNNGF.evaluatedPoint_ecal) < lim]

fig = plt.figure(figsize=(10,8))
plt.subplot(2,2,1)
plt.plot(hcal_ecal_eq_0,KNNGF.evaluatedPoint_reducedchi2_ecal_eq_0)
plt.xlabel(r"$h_{cal}$",fontsize=12)
plt.ylabel(r"$\chi^2/df$",fontsize=12)
plt.title(r"$\chi^2/df$ for $e_{cal} = 0$",fontsize=12)
plt.subplot(2,2,2)
plt.hist(rchi2_ecal_eq_0,binwidth_array(rchi2_ecal_eq_0,0.25))
plt.xlabel(r"$\chi^2/df$",fontsize=12)
plt.title(r"$\chi^2/df$ for $e_{cal} = 0$",fontsize=12)
plt.subplot(2,2,3)
xmin = min(ecal_ecal_neq_0)
xmax = max(ecal_ecal_neq_0)
ymin = min(hcal_ecal_neq_0)
ymax = max(hcal_ecal_neq_0)
vmin = 0.5
vmax = 1.5
im = plt.imshow(rrchi2_ecal_neq_0,cmap=plt.cm.seismic, extent=(xmin,xmax,ymin,ymax), origin='lower',vmin=vmin,vmax=vmax,interpolation='bilinear')
plt.colorbar(im)
plt.title(r"$\chi^2/df$ for $e_{cal} \neq 0$",fontsize=12)
plt.ylabel(r"$h_{cal}$",fontsize=12)
plt.xlabel(r"$e_{cal}$",fontsize=12)
plt.subplot(2,2,4)
plt.hist(x,binwidth_array(x))
plt.xlabel(r"$\chi^2/df$",fontsize=12)
plt.title(r"$\chi^2/df$ for $e_{cal} \neq 0$",fontsize=12)
plt.tight_layout()
plt.show()
savefig(fig,directory,classname+"chi2_calib.png")

#quelques histogrammes
i1 = int(len(KNNGF.evaluatedPoint_hcal_ecal_eq_0)/2)
i2 = int(len(KNNGF.evaluatedPoint_hcal)/5)
e1 = 0
h1 = KNNGF.evaluatedPoint_hcal_ecal_eq_0[i1]
entries1 = KNNGF.evaluatedPoint_entries_ecal_eq_0[i1]
bin_middles1 = KNNGF.evaluatedPoint_bin_middles_ecal_eq_0[i1]
error1 = np.sqrt(entries1)
params1 = KNNGF.evaluatedPoint_parameters_ecal_eq_0[i1]
e2 = KNNGF.evaluatedPoint_ecal[i2]
h2 = KNNGF.evaluatedPoint_hcal[i2]
entries2 = KNNGF.evaluatedPoint_entries[i2]
bin_middles2 = KNNGF.evaluatedPoint_bin_middles[i2]
error2 = np.sqrt(entries2)
params2 = KNNGF.evaluatedPoint_parameters[i2]

fig = plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.errorbar(bin_middles1, entries1, yerr=error1, fmt='o')
xplot = np.arange(min(bin_middles1),max(bin_middles1),1)
plt.plot(xplot,gaussian_param(xplot,*params1),lw=3)
plt.xlabel(r"$e_{true}$",fontsize=12)
plt.title(r"histogram of $e_{true}$ for $(e_{cal}="+str(np.around(e1,2))+",h_{cal}="+str(np.around(h1,2))+")$",fontsize=12)
plt.subplot(1,2,2)
plt.errorbar(bin_middles2, entries2, yerr=error2, fmt='o')
plt.plot(xplot,gaussian_param(xplot,*params2),lw=3)
xplot = np.arange(min(bin_middles2),max(bin_middles2),1)
plt.xlabel(r"$e_{true}$",fontsize=12)
plt.title(r"histogram of $e_{true}$ for $(e_{cal}="+str(np.around(e2,2))+",h_{cal}="+str(np.around(h2,2))+")$",fontsize=12)
plt.tight_layout()
plt.show()
savefig(fig,directory,classname+"hist_calib.png")

#courbe de calibration pour ecal = 0
hcal_train = KNNGF.hcal_train[KNNGF.ecal_train == 0]
true_train = KNNGF.true_train[KNNGF.ecal_train == 0]
h = np.arange(min(hcal_train),lim,0.1)
e = np.zeros(len(h))
t = KNNGF.predict(e,h)
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
plt.tight_layout()
plt.show()
savefig(fig,directory,classname+"calibration.png")

#ecalib/etrue 
h = data2.hcal[np.logical_and(data2.ecal == 0,data2.ecal+data2.hcal < lim)]
t = data2.true[np.logical_and(data2.ecal == 0,data2.ecal+data2.hcal < lim)]
e = np.zeros(len(h))
h2 = data2.hcal[np.logical_and(data2.ecal != 0,data2.ecal+data2.hcal < lim)]
t2 = data2.true[np.logical_and(data2.ecal != 0,data2.ecal+data2.hcal < lim)]
e2 = data2.ecal[np.logical_and(data2.ecal != 0,data2.ecal+data2.hcal < lim)]

c = KNNGF.predict(e,h) 
r = c/t
c2 = KNNGF.predict(e2,h2)
r2 = c2/t2

energy, means, mean_gaussianfit, sigma_gaussianfit, reducedChi2 = getMeans(t,r)
energy2, means2, mean_gaussianfit2, sigma_gaussianfit2, reducedChi22 = getMeans(t2,r2)

fig = plt.figure(figsize=(10,4))
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
plt.show()
savefig(fig,directory,classname+"ecalib_over_etrue.png")


#histogram of ecalib and etrue
fig = plt.figure(figsize=(10,8))
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
plt.tight_layout()
plt.show()
savefig(fig,directory,classname+"ecalib_over_etrue_curve.png")


#NEIGHBORS
fig = plt.figure(figsize=(10,4))
#neigh for ecal == 0
KNNGF = data1.kNNGaussianFit(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                             n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                             lim=lim,energystep=30,kind='cubic')
neigh_hcal_ecal_eq_0 = KNNGF.evaluatedPoint_neighbours_hcal_ecal_eq_0
neigh_true_ecal_eq_0 = KNNGF.evaluatedPoint_neighbours_true_ecal_eq_0
plt.subplot(1,2,1)
plt.plot(hcal_train,true_train,'.',markersize=1)
plt.xlabel(r"$h_{cal}$",fontsize=12)
plt.ylabel(r"$e_{true}$",fontsize=12)
for i in np.arange(len(neigh_hcal_ecal_eq_0)):
    plt.plot(neigh_hcal_ecal_eq_0[i],neigh_true_ecal_eq_0[i],'.',color='red',markersize=1)
plt.title(r"neighbors for $e_{cal} = 0$",fontsize=12)


#neigh for ecal != 0
KNNGF = data1.kNNGaussianFit(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                             n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                             lim=lim,energystep=10,kind='cubic')
neigh_ecal_ecal_neq_0 = np.array(KNNGF.evaluatedPoint_neighbours_ecal)
neigh_ecal_ecal_neq_0 = neigh_ecal_ecal_neq_0[np.array(KNNGF.evaluatedPoint_ecal)+np.array(KNNGF.evaluatedPoint_hcal)<lim]
neigh_hcal_ecal_neq_0 = np.array(KNNGF.evaluatedPoint_neighbours_hcal)
neigh_hcal_ecal_neq_0 = neigh_hcal_ecal_neq_0[np.array(KNNGF.evaluatedPoint_ecal)+np.array(KNNGF.evaluatedPoint_hcal)<lim]
plt.subplot(1,2,2)
plt.xlabel(r"$e_{cal}$",fontsize=12)
plt.ylabel(r"$h_{cal}$",fontsize=12)
for i in np.arange(len(neigh_hcal_ecal_neq_0)):
    plt.plot(neigh_ecal_ecal_neq_0[i],neigh_hcal_ecal_neq_0[i],'.',markersize=1)
plt.title(r"neighbors for $e_{cal} \neq 0$",fontsize=12)
plt.axis([0,lim,0,lim])
plt.tight_layout()
plt.show()
savefig(fig,directory,classname+"neighborhood.png")
