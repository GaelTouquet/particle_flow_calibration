#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

import numpy as np
import time
import math
from sklearn import linear_model

from pfcalibration.Calibration import Calibration

class LinearRegression(Calibration):
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
    if ecal + hcal > lim, the calibrated energy ecalib = math.nan
    if lim = - 1, there is no limit
    
    lim_min : float
    linear regression is done with points with ecal + hcal > lim_min
    if lim_min = - 1, there is no limit
    
    lim_max : float
    linear regression is done with points with ecal + hcal < lim_max
    if lim_max = - 1, there is no limit
    
    linRegr_ecal_eq_0 : sklearn.linear_model
    linear regression for ecal = 0
    
    linRegr_ecal_eq_0 : sklearn.linear_model
    linear regression for ecal ≠ 0

    """
    def __init__(self,ecal_train=[],hcal_train=[],true_train=[],lim_min=-1,lim_max=-1,lim=-1):
        """
        Constructor of the class
        
        Parameters
        ---------
        ecal_train : array
        ecal value to train the calibration
    
        hcal_train : array
        ecal value to train the calibration
    
        true_train : array
        ecal value to train the calibration
        
        lim_min : float
        linear regression is done with points with ecal + hcal > lim_min
        if lim_min = - 1, there is no limit
        
        lim_max : float
        linear regression is done with points with ecal + hcal < lim_max
        if lim_max = - 1, there is no limit
        
        lim : float
        if ecal + hcal > lim, the calibrated energy ecalib = math.nan
        if lim = - 1, there is no limit
        """
        
        # We use the constructor of the mother class 
        Calibration.__init__(self,ecal_train,hcal_train,true_train,lim)
        
        self.lim_min = lim_min
        self.lim_max = lim_max
        
        if lim_min == -1:
            ind_min = np.ones(len(self.ecal_train),dtype=bool)
        else:
            ind_min = self.ecal_train + self.hcal_train > lim_min
        if lim_max == -1:
            ind_max = np.ones(len(self.ecal_train),dtype=bool)
        else:
            ind_max = self.ecal_train+ self.hcal_train < lim_max

        #CASE : ecal != 0
        ind_0 = self.ecal_train != 0
        ind = np.logical_and(ind_min,ind_max)
        ind = np.logical_and(ind,ind_0)
        X_train = [self.ecal_train[ind],self.hcal_train[ind]]
        X_train = np.transpose(np.matrix(X_train))
        Y_train = self.true_train[ind]
        Y_train = np.transpose(np.matrix(Y_train))
        linRegr_ecal_neq_0 = linear_model.LinearRegression()
        linRegr_ecal_neq_0.fit(X_train,Y_train)

        #CASE : ecal == 0
        ind_0 = self.ecal_train == 0
        ind = np.logical_and(ind_min,ind_max)
        ind = np.logical_and(ind,ind_0)
        X_train = self.hcal_train[ind]
        X_train = np.transpose(np.matrix(X_train))
        Y_train = self.true_train[ind]
        Y_train = np.transpose(np.matrix(Y_train))
        linRegr_ecal_eq_0 = linear_model.LinearRegression()
        linRegr_ecal_eq_0.fit(X_train,Y_train)
        
        self.linRegr_ecal_neq_0 = linRegr_ecal_neq_0
        self.linRegr_ecal_eq_0 = linRegr_ecal_eq_0


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
        """
        if e+h <= self.lim:
            if e != 0:
                true = self.linRegr_ecal_neq_0.coef_[0][0]*e + self.linRegr_ecal_neq_0.coef_[0][1]*h + self.linRegr_ecal_neq_0.intercept_[0]
            else:
                true = self.linRegr_ecal_eq_0.coef_[0][0]*h + self.linRegr_ecal_eq_0.intercept_[0]
            return true
        else:
            return math.nan

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
        """
        begin = time.time()
        vect = np.vectorize(self.predictSingleValue)
        ecalib = vect(e,h)
        end = time.time()
        if timeInfo:
            print("Prediction made in",end-begin,"s")
        return ecalib
            
        
    def __str__(self):
        """
        To present the calibration as a string
        """
        res = Calibration.__str__(self)
        res += "\nfor ecal == 0 : "
        res += "\n\tecalib = "+str(self.linRegr_ecal_eq_0.coef_[0][0])+" hcal + "+str(self.linRegr_ecal_eq_0.intercept_[0])
        res += "\nfor ecal != 0 : "
        res += "\n\tecalib = "+str(self.linRegr_ecal_neq_0.coef_[0][0])+" ecal + "+str(self.linRegr_ecal_neq_0.coef_[0][1])+" ecal + "+str(self.linRegr_ecal_neq_0.intercept_[0])
        return res
    
    def __repr__(self):
        """
        To present the calibration as a string
        """
        res = Calibration.__str__(self)
        res += "\nfor ecal == 0 : "
        res += "\n\tecalib = "+str(self.linRegr_ecal_eq_0.coef_[0][0])+" hcal + "+str(self.linRegr_ecal_eq_0.intercept_[0])
        res += "\nfor ecal != 0 : "
        res += "\n\tecalib = "+str(self.linRegr_ecal_neq_0.coef_[0][0])+" hcal + "+str(self.linRegr_ecal_neq_0.coef_[0][1])+" ecal + "+str(self.linRegr_ecal_neq_0.intercept_[0])
        return res