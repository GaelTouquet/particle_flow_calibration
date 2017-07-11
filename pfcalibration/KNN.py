#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

from sklearn import neighbors
import time
import numpy as np
from math import nan
from pfcalibration.Calibration import Calibration

class KNN(Calibration):
    """
    Inherit from Calibration.
    
    Class to calibrate the true energy of a particle thanks to training datas.
    We use the a k neareast neighbours method.
    We do the pondered mean of the true energies of k neareast neighbours.

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
    
    n_neighbors_ecal_eq_0: int
    Number of neighbors to use by default for k_neighbors queries.
    for ecal == 0
    
    n_neighbors_ecal_neq_0: int
    Number of neighbors to use by default for k_neighbors queries.
    for ecal != 0

    weight : str or callable
    weight function used in prediction. Possible values:
    'uniform' : uniform weights. All points in each neighborhood are
    weighted equally.
    'distance' : weight points by the inverse of their distance. in
    this case, closer neighbors of a query point will have a greater
    influence than neighbors which are further away.
    [callable] : a user-defined function which accepts an array of
    distances, and returns an array of the same shape containing the
    weights.
    'gaussian'
    Gaussian weights are used by default.
    
    algortihm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
    Algorithm used to compute the nearest neighbors:
    'ball_tree' will use BallTree
    'kd_tree' will use KDtree
    'brute' will use a brute-force search.
    'auto' will attempt to decide the most appropriate algorithm based on
    the values passed to fit method.

    sigma : float
    sigma for the gaussian if weight == 'gaussian'

    neigh_ecal_neq_0 : sklearn.neighbors.NearestNeighbors
    the sklearn.neighbors.NearestNeighbors for ecal != 0

    neigh_ecal_eq_0 : sklearn.neighbors.NearestNeighbors
    the sklearn.neighbors.NearestNeighbors for ecal == 0
    """
    
    def __init__(self,ecal_train=[],hcal_train=[],true_train=[],
                 n_neighbors_ecal_eq_0=2000,n_neighbors_ecal_neq_0=250,
                 weights='gaussian',algorithm='auto',sigma=5,lim=-1):
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

        weight : str or callable
        weight function used in prediction. Possible values:
        'uniform' : uniform weights. All points in each neighborhood are
        weighted equally.
        'distance' : weight points by the inverse of their distance. in this
        case, closer neighbors of a query point will have a greater influence
        than neighbors which are further away.
        [callable] : a user-defined function which accepts an array of
        distances, and returns an array of the same shape containing the weights.
        'gaussian'
        Gaussian weights are used by default.

        algortihm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
        Algorithm used to compute the nearest neighbors:
        'ball_tree' will use BallTree
        'kd_tree' will use KDtree
        'brute' will use a brute-force search.
        'auto' will attempt to decide the most appropriate algorithm based on
        the values passed to fit method.

        sigma : float
        sigma for the gaussian if weight == 'gaussian'

        lim : float
        to reject calibration points with ecal + hcal > lim
        if lim = - 1, there is no limit
        """

        Calibration.__init__(self,ecal_train,hcal_train,true_train,lim)
        
        if weights == 'gaussian':
            def gaussian(x):
                return np.exp(-(x**2) / (sigma**2) / 2 )
            self.weights = gaussian
        else:
            self.weights = weights

        self.n_neighbors_ecal_eq_0 = n_neighbors_ecal_eq_0
        self.n_neighbors_ecal_neq_0 = n_neighbors_ecal_neq_0
        self.algorithm = algorithm
        self.sigma = sigma

        self.recalibrated = False

        # Case ecal != 0
        X_train = [ecal_train[ecal_train!=0],hcal_train[ecal_train!=0]]
        self.X_train1 = X_train
        X_train = np.transpose(np.matrix(X_train))
        Y_train = true_train[ecal_train!=0]
        self.Y_train1 = Y_train
        Y_train = np.transpose(np.matrix(Y_train))
        neigh_ecal_neq_0 = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors_ecal_neq_0, weights=self.weights, algorithm=self.algorithm)
        neigh_ecal_neq_0.fit(X_train,Y_train)

        #case ecal == 0
        X_train = hcal_train[ecal_train==0]
        self.X_train2 = X_train
        X_train = np.transpose(np.matrix(X_train))
        Y_train = true_train[ecal_train==0]
        self.Y_train2 = Y_train
        Y_train = np.transpose(np.matrix(Y_train))
        neigh_ecal_eq_0 = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors_ecal_eq_0, weights=self.weights, algorithm=algorithm)
        neigh_ecal_eq_0.fit(X_train,Y_train)

        self.neigh_ecal_neq_0 = neigh_ecal_neq_0
        self.neigh_ecal_eq_0 = neigh_ecal_eq_0

    def predictSingleValue(self,e,h):
        """
        To predict the true energie from a couple of ecal, hcal

        Parameters
        ----------
        e : the ecal energy
        h : the hcal energy

        Returns
        -------
        true : the predicted true energy
        the value is NaN if the asked value is off-limit
        """
        if e+h < self.lim:
            if e != 0:
                x =[[e,h]]
                return self.neigh_ecal_neq_0.predict(x)
            else:
                return self.neigh_ecal_eq_0.predict(h)
        else:
            return nan

    def predict(self,e,h,timeInfo=False):
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
        begin = time.time()
        vect = np.vectorize(self.predictSingleValue)
        ecalib = vect(e,h)
        end = time.time()
        if timeInfo:
            print("Calibration made in",end-begin,"s")
        return ecalib

    def refresh(self):
        # Case ecal != 0
        X_train = [self.ecal_train[self.ecal_train!=0],self.hcal_train[self.ecal_train!=0]]
        self.X_train1 = X_train
        X_train = np.transpose(np.matrix(X_train))
        Y_train = self.true_train[self.ecal_train!=0]
        self.Y_train1 = Y_train
        Y_train = np.transpose(np.matrix(Y_train))
        neigh_ecal_neq_0 = neighbors.KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm)
        neigh_ecal_neq_0.fit(X_train,Y_train)

        #case ecal == 0
        X_train = self.hcal_train[self.ecal_train==0]
        self.X_train2 = X_train
        X_train = np.transpose(np.matrix(X_train))
        Y_train = self.true_train[self.ecal_train==0]
        self.Y_train2 = Y_train
        Y_train = np.transpose(np.matrix(Y_train))
        neigh_ecal_eq_0 = neighbors.KNeighborsRegressor(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm)
        neigh_ecal_eq_0.fit(X_train,Y_train)

        self.neigh_ecal_neq_0 = neigh_ecal_neq_0
        self.neigh_ecal_eq_0 = neigh_ecal_eq_0

    def deleteTrainPoints(self,e,h,t,refresh=True):

        if len(e)!=0 and len(h)!=0 and len(t) != 0:

            def deleleteOnePoint(e0,h0,t0):
                bool_e = self.ecal_train == e0
                bool_h = self.hcal_train == h0
                bool_t = self.true_train == t0
                index = np.logical_and(bool_e,bool_h)
                index = np.logical_and(index,bool_t)
                index = np.argwhere(index == True)
                self.ecal_train = np.delete(self.ecal_train,index)
                self.hcal_train = np.delete(self.hcal_train,index)
                self.true_train = np.delete(self.true_train,index)

            vect = np.vectorize(deleleteOnePoint)
            vect(e,h,t)

            if refresh:
                self.refresh()
