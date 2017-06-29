#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

import numpy as np
import pickle
from scipy.optimize import curve_fit
import math
import warnings
from scipy.optimize import OptimizeWarning
import matplotlib.pyplot as plt
from os import mkdir

def selectionCone(x,y,z,theta,delta):
    """
    To select the points whose x and y coordinates are
    in the cone (theta,theta+delta)

    Parameters
    ----------
    x : the x coordinates (an array-like)
    y : the y coordinates (an array-like)
    z : the z coordinates (an array-like)
    theta : the angle of rotation
    delta :the angle of the cone

    Returns
    -------
    a numpy array of the x,y,z coordinates of the points in the cone
    """
    cos = x/np.sqrt(x*x + y*y)
    ind = np.logical_and(cos < np.cos(theta),cos > np.cos(theta+delta))
    return np.array([x[ind],y[ind],z[ind]])

def projectionPlan(x,y,z,theta):
    """
    Projette les coordonnées x,y,z sur le plan
    tourné de theta
    Parameters
    ----------
    x :
    y :
    z :
    theta :

    Returns
    -------
    """
    return np.array([x*np.cos(theta) + y*np.sin(theta),z])

def projectionPlanInverse(x,z,theta):
    """
    Transforme les coord du plan tourné de theta
    en coord x, y, z
    Parameters
    ----------
    x :
    z :
    theta :
    Returns
    -------
    """
    return np.array([x*np.cos(theta), x*np.sin(theta),z])

def projectionCone(x,y,z,theta,delta):
    """
    Combine les deux fonctions precedentes
    Parameters
    ----------
    x : the x coordinates (an array-like)
    y : the y coordinates (an array-like)
    z : the z coordinates (an array-like)
    theta : the angle of rotation
    delta :the angle of the cone
    Returns
    -------
    """
    res = selectionCone(x,y,z,theta,delta)
    return projectionPlan(res[0],res[1],res[2],theta)

def importPickle(filename):
    """
    Parameters
    ----------
    Returns
    -------
    """
    datafile = open(filename, "rb")
    mon_depickler = pickle.Unpickler(datafile)
    data = mon_depickler.load()
    datafile.close()
    print("the datas include",len(data.ecal),"particles")
    return data

def exportPickle(filename,objectToSave):
    """
    Parameters
    ----------
    Returns
    -------
    """
    dataFile = open(filename, "wb")
    pickler = pickle.Pickler(dataFile)
    pickler.dump(objectToSave)
    dataFile.close()

def gaussian_param(x,sigma=1,mu=0,k=1):
    return k*np.exp(-(x-mu)**2/(2*sigma**2))

def gaussian_param_normalized(x,sigma=1,mu=0):
    k = 1/(sigma*np.sqrt(2*np.pi))
    return gaussian_param(x,sigma,mu,k)

def optimized_binwidth(x_input):
    """
    Optimized binwidth with the Shimazaki and Shinomoto methods
    """
    x = x_input[np.invert(np.isnan(x_input))]
    x_max = max(x)
    x_min = min(x)
    N_MIN = 4   #Minimum number of bins (integer)
                #N_MIN must be more than 1 (N_MIN > 1).
    N_MAX = 200  #Maximum number of bins (integer)
    N = range(N_MIN,N_MAX) # #of Bins
    N = np.array(N)
    D = (x_max-x_min)/N    #Bin size vector
    C = np.zeros(shape=(np.size(D),1))
    #Computation of the cost function
    for i in np.arange(np.size(N)):
        edges = np.linspace(x_min,x_max,N[i]+1) # Bin edges
        ki, bin_edges = np.histogram(x,edges) # Count # of events in bins
        k = np.mean(ki) #Mean of event count
        v = sum((ki-k)**2)/N[i] #Variance of event count
        C[i] = (2*k-v)/((D[i])**2) #The cost Function
    #Optimal Bin Size Selection
    cmin = min(C)
    idx  = np.where(C==cmin)
    idx = int(idx[0][0])
    optD = D[idx]
    return optD




def gaussian_fit(x_input,binwidth = 'optimized',info=False,giveChi2 = False):

    with warnings.catch_warnings():
        try:
            #we create the histogram
            warnings.simplefilter("error", OptimizeWarning)
            x = x_input[np.invert(np.isnan(x_input))]
            if len(x) == 0:
                return [math.nan,math.nan,math.nan]

            if binwidth == 'optimized':
                binwidth = optimized_binwidth(x_input)
            bins = np.arange(min(x), max(x) + binwidth, binwidth)
            entries, bin_edges = np.histogram(x,bins=bins)

            bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
            bin_middles = bin_middles[entries != 0]
            entries = entries[entries != 0]
            # we fit the histogram
            p0 = np.sqrt(np.std(entries)),bin_middles[np.argmax(entries)],max(entries)
            error = np.sqrt(entries)
            parameters, cov_matrix = curve_fit(gaussian_param, bin_middles, entries,sigma=error,absolute_sigma=True,p0=p0)
            # we look if the fit is good
            crit = np.sqrt(np.diag(cov_matrix))
            #we compute the Chi2
            chi2 = np.sum(((gaussian_param(bin_middles,*parameters)-entries)/error)**2)
            reduced = chi2/(len(bin_middles)-len(p0))

            if info:
                print("parameters :",parameters)
                print("diag of cov matrix :",crit)
                print("reduced chi2:",reduced)
            abort = False

            if reduced > 5:
                abort = True
            if abort:
                if giveChi2:
                    return[math.nan,math.nan,math.nan], math.nan
                else:
                    return [math.nan,math.nan,math.nan]
            #sigma has to be > 0
            parameters[0] = np.abs(parameters[0])
            if giveChi2:
                return parameters, reduced
            else:
                return parameters
        except:
            if giveChi2:
                return[math.nan,math.nan,math.nan], math.nan
            else:
                return [math.nan,math.nan,math.nan]

def gaussian_fit_plot_issues(x_input,filename,binwidth = 0.1,info=False,giveChi2 = False):
    with warnings.catch_warnings():
        try:
            #we create the histogram
            warnings.simplefilter("error", OptimizeWarning)
            x = x_input[np.invert(np.isnan(x_input))]
            if len(x) == 0:
                return [math.nan,math.nan,math.nan]

            if binwidth == 'optimized':
                binwidth = optimized_binwidth(x_input)
            bins = np.arange(min(x), max(x) + binwidth, binwidth)
            entries, bin_edges = np.histogram(x,bins=bins)

            bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
            bin_middles = bin_middles[entries != 0]
            entries = entries[entries != 0]
            # we fit the histogram
            p0 = np.sqrt(np.std(entries)),bin_middles[np.argmax(entries)],max(entries)
            error = np.sqrt(entries)
            parameters, cov_matrix = curve_fit(gaussian_param, bin_middles, entries,sigma=error,p0=p0)
            # we look if the fit is good
            crit = np.sqrt(np.diag(cov_matrix))
            #we compute the Chi2

            chi2 = np.sum(((gaussian_param(bin_middles,*parameters)-entries)/error)**2)
            reduced = chi2/(len(bin_middles)-len(parameters))

            if info:
                print("parameters :",parameters)
                print("diag of cov matrix :",crit)
            abort = False
            if reduced > 5:
                abort = True
            if abort:
                fig = plt.figure(figsize=(10,5))
                plt.subplot(1,2,1)
                plt.errorbar(bin_middles, entries, yerr=error, fmt='o')
                xplot = np.linspace(0,max(bin_edges),200)
                plt.plot(xplot,gaussian_param(xplot,*parameters),lw=3)
                plt.subplot(1,2,2)
                plt.hist(x,bins=bins)
                plt.plot(xplot,gaussian_param(xplot,*parameters),lw=3)
                plt.show()
                fig.savefig(filename,bbox_inches='tight')
                if giveChi2:
                    return[math.nan,math.nan,math.nan], math.nan
                else:
                    return [math.nan,math.nan,math.nan]
            #sigma has to be > 0
            parameters[0] = np.abs(parameters[0])
            if giveChi2:
                return parameters, reduced
            else:
                return parameters
        except:
            if giveChi2:
                return[math.nan,math.nan,math.nan], math.nan
            else:
                return [math.nan,math.nan,math.nan]


def savefig(fig,directory,filename):
    """
    To save a figure
    Parameters
    ----------
    fig : pyplot.figure
    the figure to save
    directory : string
    the path and the name of the directory
    filename : string
    the name of the image saved.
    Do not forget the extension (.png, .eps ...)
    """
    splitted = directory.split('/')
    director = ""
    for s in splitted:
        if len(s) > 0:
            director += s+'/'
            try:
                mkdir(director)
            except FileExistsError:
                pass
    fig.savefig(directory+filename,bbox_inches='tight')
