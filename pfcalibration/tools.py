#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

import numpy as np
import dill as pickle
from scipy.optimize import curve_fit
import math
import warnings
from scipy.optimize import OptimizeWarning
import matplotlib.pyplot as plt
from os import mkdir
import shelve


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
    Projects the points (x,y,z) on the plan turned of the theta angle
    tourné de theta
    Parameters
    ----------
    x : numpy array
    the x coordinates
    y : numpy array
    the y coordinates
    z : numpy array
    the z coordinates
    theta : float
    the theta angle

    Returns
    -------
    the coordinates x_theta,z
    """
    return np.array([x*np.cos(theta) + y*np.sin(theta),z])

def projectionPlanInverse(x,z,theta):
    """
    Transforms the coordinates of the plan turned of the theta angle into x,y,z coordiates
    Parameters
    ----------
    x : x_theta coordinates
    z : z coordinates
    theta : the theta angle
    Returns
    -------
    the x,y,z coordinates
    """
    return np.array([x*np.cos(theta), x*np.sin(theta),z])

def projectionCone(x,y,z,theta,delta):
    """
    Combine selectioCone and projectionPlan
    To project the points in the cone on the plan
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



#def exportPickle(filename,objectToSave):
    #"""
    #Te export an object into a binary file
    
    #Parameters
    #----------
    #filename : str
    #path+file name
    #objectToSave : object
    #"""
    #dataFile = open(filename, "wb")
    #pickler = pickle.Pickler(dataFile)
    #pickler.dump(objectToSave)
    #dataFile.close()
    
def exportPickle(filename,objectToSave):
    """
    Te export an object into a binary file
    
    Parameters
    ----------
    filename : str
    path+file name
    objectToSave : object
    """
    np.save(filename,objectToSave)
    ##d = shelve.open(filename)
    ##d['key'] = objectToSave
    ##d.close()
    #dataFile = open(filename, "w")
    #dataFile.write(str(objectToSave))
    ## pickler = pickle.Pickler(dataFile)
    ## pickler.dump(objectToSave)
    ##dataFile.close()



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
    x = np.array(x)
    x_max = max(x)
    x_min = min(x)
    N_MIN = 1   #Minimum number of bins (integer)
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

def binwidth_array(x_input,binwidth = 'optimized'):
    """
    Creates an array on bins adapted to an array (aim : histogram)
    
    Parameters
    ----------
    x_input: array
    binwidth : if 'optimized', it will be evaluated thanks to the Shimazaki and Shinomoto methods
   
    Returns
    -------
    the bin array
    """
    x = np.array(x_input)
    x = x[np.invert(np.isnan(x))]
    if binwidth == 'optimized':
        binwidth = optimized_binwidth(x)
    return np.arange(min(x), max(x) + binwidth, binwidth)




def gaussian_fit(x_input,binwidth = 'optimized',giveReducedChi2 = False, reducedChi2Max = 5,info=False):
    """
    Gaussian fit of an histogram of a set of data
    
    Parameters
    ---------
    x_input: array
    binwidth : if 'optimized', it will be evaluated thanks to the Shimazaki and Shinomoto methods
    giveReducedChi2 : Boolean
    default value, False
    If true, will return the reduced Chi2 
    reducedChi2Max : float
    if reducedChi2 > reducedChi2Max the fit is rejected
    
    Returns
    -------
    parameters : array
    parameters[0], sigma
    parameters[1], mean
    parameters[2], coefficient
    if giveReducedChi2 is True returns will be (parameters,reducedChi2)
    """
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

            if reduced > reducedChi2Max:
                if giveReducedChi2:
                    return[math.nan,math.nan,math.nan], math.nan
                else:
                    return [math.nan,math.nan,math.nan]
            #sigma has to be > 0
            parameters[0] = np.abs(parameters[0])
            if giveReducedChi2:
                return parameters, reduced
            else:
                return parameters
        except:
            if giveReducedChi2:
                return[math.nan,math.nan,math.nan], math.nan
            else:
                return [math.nan,math.nan,math.nan]

def gaussian_fit_plot_issues(x_input,filename,binwidth = 0.1,info=False,giveReducedChi2 = False):
    """
    Gaussian fit of an histogram of a set of data
    If the calibration is rejected, it will plot the histogram
    Parameters
    ---------
    x_input: array
    binwidth : if 'optimized', it will be evaluated thanks to the Shimazaki and Shinomoto methods
    giveReducedChi2 : Boolean
    default value, False
    If true, will return the reduced Chi2 
    
    Returns
    -------
    parameters : array
    parameters[0], sigma
    parameters[1], mean
    parameters[2], coefficient
    if giveReducedChi2 is True returns will be (parameters,reducedChi2)
    """
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
                if giveReducedChi2:
                    return[math.nan,math.nan,math.nan], math.nan
                else:
                    return [math.nan,math.nan,math.nan]
            #sigma has to be > 0
            parameters[0] = np.abs(parameters[0])
            if giveReducedChi2:
                return parameters, reduced
            else:
                return parameters
        except:
            if giveReducedChi2:
                return[math.nan,math.nan,math.nan], math.nan
            else:
                return [math.nan,math.nan,math.nan]


def savefig(fig,directory="img/",filename="img.png"):
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
    
    
from pfcalibration.EnergyData import EnergyData

    
def importPickle(filename):
    """
    To import an object in a binary file
    
    Parameters
    ----------
    filename: string
    the path and name of the file
    Returns
    -------
    obj : 
    the imported object 
    """
    print('opening :', filename)
    obj = np.load(filename+'.npy')
    #d = shelve.open(filename)
    #obj = d['key']
    #return obj
    #datafile = open(filename, "rb")
    #mon_depickler = pickle.Unpickler(datafile)
    #obj = mon_depickler.load()
    obj = EnergyData(obj[:,0],obj[:,1],obj[:,2],obj[:,3],obj[:,4],obj[:,5])
    #datafile.close()
    return obj

    
def importData(filename):
    """
    Import the EnergyData
    
    Parameters
    ----------
    filename : str
    path+filename
    
    Returns
    -------
    the EnergyData
    """
    data = importPickle(filename)
    print("Data imported and includes",len(data.ecal),"particles")
    return data

def importCalib(filename):
    """
    Import the calibration
    
    Parameters
    ----------
    filename : str
    path+filename
    
    Returns
    -------
    the calibration
    """
    calib = importPickle(filename)
    print(calib.classname+" imported")
    print(calib)
    return calib