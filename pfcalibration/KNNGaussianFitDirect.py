#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

from sklearn import neighbors
import numpy as np
import math
from pfcalibration.tools import gaussian_param
from pfcalibration.Calibration import Calibration
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning


class KNNGaussianFitDirect(Calibration):
    """
    Inherit from Calibration.
    
    Class to calibrate the true energy of a particle thanks to training datas.
    We use the a k neareast neighbours method, we fit the histogramm of the
    true energy of the neighbours by a gaussian and consider the mean of the
    gaussian distribution is the approximation of the true energy.
    We do an iterpolation to determine the other values.

    Attributs
    ---------
    ecal_train : array
    ecal value to train the calibration

    hcal_train : array
    ecal value to train the calibration

    true_train : array
    ecal value to train the calibration

    lim : float
    to reject calibration points with ecal + hcal > lim
    if lim = - 1, there is no limit
    
    n_neighbors: int
    Number of neighbors to use by default for k_neighbors queries.

    algortihm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
    Algorithm used to compute the nearest neighbors:
    'ball_tree' will use BallTree
    'kd_tree' will use KDtree
    'brute' will use a brute-force search.
    'auto' will attempt to decide the most appropriate algorithm based
    on the values passed to fit method.

    ecal_train_ecal_eq_0 : array
    ecal value to train the calibration
    for ecal == 0
    
    hcal_train_ecal_eq_0 : array
    ecal value to train the calibration
    for ecal == 0

    true_train_ecal_eq_0 : array
    ecal value to train the calibration
    for ecal == 0

    ecal_train_ecal_neq_0 : array
    ecal value to train the calibration
    for ecal != 0

    hcal_train_ecal_neq_0 : array
    ecal value to train the calibration
    for ecal != 0

    true_train_ecal_neq_0 : array
    ecal value to train the calibration
    for ecal != 0

    neigh_ecal_neq_0 : sklearn.neighbors.NearestNeighbors
    the sklearn.neighbors.NearestNeighbors for ecal != 0

    neigh_ecal_eq_0 : sklearn.neighbors.NearestNeighbors
    the sklearn.neighbors.NearestNeighbors for ecal == 0

    """

    def __init__(self,ecal_train=[],hcal_train=[],true_train=[],n_neighbors_ecal_eq_0=2000,n_neighbors_ecal_neq_0=250,algorithm='auto',lim=-1):
        """
        Parameters
        ----------
        ecal_train : array-like
        ecal value to train the calibration

        hcal_train : array-like
        hcal value to train the calibration

        true_train : array-like
        true value to train the calibration

        n_neighbors_ecal_eq_0: int
        Number of neighbors to use by default for k_neighbors queries.
        for ecal == 0
        
        n_neighbors_ecal_neq_0: int
        Number of neighbors to use by default for k_neighbors queries.
        for ecal != 0

        algortihm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
        Algorithm used to compute the nearest neighbors:
        'ball_tree' will use BallTree
        'kd_tree' will use KDtree
        'brute' will use a brute-force search.
        'auto' will attempt to decide the most appropriate algorithm based on
        the values passed to fit method.

        lim : float
        to reject calibration points with ecal + hcal > lim
        if lim = - 1, there is no limit
        """

        Calibration.__init__(self,ecal_train,hcal_train,true_train,lim)
        
        self.n_neighbors_ecal_eq_0 = n_neighbors_ecal_eq_0
        self.n_neighbors_ecal_neq_0 = n_neighbors_ecal_neq_0
        self.algorithm = algorithm

        #Case ecal == 0
        self.neigh_ecal_eq_0 = neighbors.NearestNeighbors(n_neighbors=self.n_neighbors_ecal_eq_0, algorithm=algorithm)
        self.hcal_train_ecal_eq_0 = self.hcal_train[self.ecal_train == 0]
        self.hcal_train_ecal_eq_0_min = min(self.hcal_train_ecal_eq_0)
        self.true_train_ecal_eq_0 = self.true_train[self.ecal_train == 0]
        self.neigh_ecal_eq_0.fit(np.transpose(np.matrix(self.hcal_train_ecal_eq_0)))
        

        # Case ecal != 0
        self.neigh_ecal_neq_0 = neighbors.NearestNeighbors(n_neighbors=self.n_neighbors_ecal_neq_0, algorithm=algorithm)
        self.ecal_train_ecal_neq_0 = self.ecal_train[self.ecal_train != 0]
        self.hcal_train_ecal_neq_0 = self.hcal_train[self.ecal_train != 0]
        self.true_train_ecal_neq_0 = self.true_train[self.ecal_train != 0]
        self.hcal_train_ecal_neq_0_min = min(self.hcal_train_ecal_neq_0)
        self.ecal_train_ecal_neq_0_min = min(self.ecal_train_ecal_neq_0)
        self.neigh_ecal_neq_0.fit(np.transpose(np.matrix([self.ecal_train_ecal_neq_0,self.hcal_train_ecal_neq_0])))
        
        
    def predict(self,e,h):
        """
        To predict the true energies thanks to couples of ecal, hcal

        Parameters
        ----------
        e : a numpy array of ecal energies
        h : a numpy array of hcal energies

        Returns
        -------
        true : a numpy array of predicted true energies
        the value is NaN if the asked value is off-limit
        """
        def predictSingleValue(ecal,hcal):
            if ecal+hcal > self.lim:
                return math.nan
            reduced = math.nan
            if ecal == 0:
                dist, ind = self.neigh_ecal_eq_0.kneighbors(X = hcal)
                dist = dist[0]
                ind = ind[0]
                
                dlim = hcal-self.hcal_train_ecal_eq_0_min + 0.1
                if hcal < self.hcal_train_ecal_eq_0_min:
                    dlim = self.hcal_train_ecal_eq_0_min + 0.1
                ind = ind[dist <= dlim]

                true_neigh = self.true_train_ecal_eq_0[ind]
                binwidth = 1

                nbins = np.arange(min(true_neigh), max(true_neigh) + binwidth, binwidth)
                with warnings.catch_warnings():
                    try:
                        #we create the histogram
                        warnings.simplefilter("error", OptimizeWarning)
                        entries, bin_edges = np.histogram(true_neigh,bins=nbins)
                        bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
                        bin_middles = bin_middles[entries != 0]
                        entries = entries[entries != 0]

                        # we fit the histogram
                        p0 = np.sqrt(np.std(entries)),bin_middles[np.argmax(entries)],max(entries)
                        error = np.sqrt(entries)
                        parameters, cov_matrix = curve_fit(gaussian_param, bin_middles, entries,sigma=error,p0=p0)
                        res = parameters[1]

                        chi2 = np.sum(((gaussian_param(bin_middles,*parameters)-entries)/error)**2)
                        reduced = chi2/(len(bin_middles)-len(parameters))

                        if reduced > 10:
                            raise OptimizeWarning

                    except (OptimizeWarning, RuntimeError):
                        parameters = p0
                        res = parameters[1]
                        print("calibration issue for ecal = 0, hcal = ",hcal,"reduced chi2 = ",reduced)
                    finally:
                        return res
            else:
                dist, ind = self.neigh_ecal_neq_0.kneighbors(X = [[ecal,hcal]])
                dist = dist[0]
                ind = ind[0]

                true_neigh = self.true_train_ecal_neq_0[ind]
                binwidth = 1
                nbins = np.arange(min(true_neigh), max(true_neigh) + binwidth, binwidth)
                
                try:
                    #we create the histogram
                    warnings.simplefilter("error", OptimizeWarning)
                    entries, bin_edges = np.histogram(true_neigh,bins=nbins)
                    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
                    bin_middles = bin_middles[entries != 0]
                    entries = entries[entries != 0]

                    # we fit the histogram
                    p0 = np.sqrt(np.std(entries)),bin_middles[np.argmax(entries)],max(entries)
                    error = np.sqrt(entries)
                    parameters, cov_matrix = curve_fit(gaussian_param, bin_middles, entries,sigma=error,p0=p0)
                    res = parameters[1]
                    
                    chi2 = np.sum(((gaussian_param(bin_middles,*parameters)-entries)/error)**2)
                    reduced = chi2/(len(bin_middles)-len(parameters))

                    if reduced > 10:
                        raise OptimizeWarning
                except (OptimizeWarning, RuntimeError):                            
                    parameters = p0
                    res = parameters[1]
                    print("calibration issue for ecal = ",ecal,", hcal = ",hcal,"reduced chi2 = ",reduced,"remained neighbours = ",len(true_neigh))
                finally:
                    return res

        vect = np.vectorize(predictSingleValue)
        
        return vect(e,h)
    
    def neighborhoodSingleValue(self,ecal,hcal):
        if ecal+hcal > self.lim:
            return [[],[],[]]

        if ecal == 0:
            dist, ind = self.neigh_ecal_eq_0.kneighbors(X = hcal)
            dist = dist[0]
            ind = ind[0]
            dlim = hcal-self.hcal_train_ecal_eq_0_min + 0.1
            ind = ind[dist <= dlim]
            true_neigh = self.true_train_ecal_eq_0[ind]
            hcal_neigh = self.hcal_train_ecal_eq_0[ind]
            ecal_neigh = np.zeros(len(hcal_neigh))
        else:
            dist, ind = self.neigh_ecal_neq_0.kneighbors(X = [[ecal,hcal]])
            true_neigh = self.true_train_ecal_neq_0[ind][0]
            ecal_neigh = self.ecal_train_ecal_neq_0[ind][0]
            hcal_neigh = self.hcal_train_ecal_neq_0[ind][0]
        return [ecal_neigh,hcal_neigh,true_neigh]
        
    def neighborhood(self,e,h):
        """
        neingbourhood of a point ecal, hcal

        Parameters
        ----------
        e : a numpy array of ecal energies
        h : a numpy array of hcal energies

        Returns
        -------
        true : a numpy array of predicted true energies
        the value is NaN if the asked value is off-limit
        """
        
        res = []
        for i in np.arange(len(e)):
            res.append(self.neighborhoodSingleValue(e[i],h[i]))
        return res
