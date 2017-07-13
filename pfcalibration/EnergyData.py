#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""
import numpy as np
from pfcalibration.CalibrationLego import CalibrationLego
from pfcalibration.LinearRegression import LinearRegression
from pfcalibration.KNN import KNN
from pfcalibration.KNNGaussianCleaning import KNNGaussianCleaning
from pfcalibration.KNNGaussianFit import KNNGaussianFit
from pfcalibration.KNNGaussianFitDirect import KNNGaussianFitDirect
import time

class EnergyData:
    """
    Stores all the datas of the simulated hadrons
    """
    def __init__(self,true = np.array([]),p = np.array([]),ecal = np.array([]),hcal = np.array([]),eta = np.array([])):
        """
        Constructeur de la classe

        Parameters
        ----------
        true : the true energie of the hadrons, a numpy array
        p : impulsion, a numpy array
        ecal : the energie detected by the electromagnetic calorimeter, a numpy array
        hcal : the energie detected by the hadronical calorimeter, a numpy array
        eta : pseudorapidity, a numpy array
        """
        self.true = true
        self.p  = p
        self.ecal = ecal
        self.hcal = hcal
        self.eta = eta
        self.ecal_max = np.max(ecal)
        self.hcal_max = np.max(hcal)
        self.ecal_min = np.min(ecal)
        self.hcal_min = np.min(hcal)
        self.ener_max = max(self.ecal_max,self.hcal_max)
        self.true_max = np.max(true)

    def LinearRegression(self,lim_min = 20, lim_max=80, lim=150):
        """
        To create a LinearRegression Calibration with this EnergyData as training values.
        
        Parameters
        ----------
        
        Returns
        -------
        calib : pfcalibration.LinearRegression
        The calibration
        """
        
        begin = time.time()
        calib = LinearRegression(self.ecal,self.hcal,self.true,lim_min, lim_max, lim)
        end = time.time()
        print("LinearRegression - Calibration made in",end-begin,"s")
        return calib

    def calibrationLego(self,nbLego):
        """
        To create a CalibrationLego with this EnergyData as training values.
        
        Parameters
        ----------
        
        Returns
        -------
        calib : pfcalibration.CalibrationLego
        The calibration
        """
        
        begin = time.time()
        calib = CalibrationLego(self,nbLego)
        end = time.time()
        print("LinearRegression - Calibration made in",end-begin,"s")
        return calib

    def KNN(self,n_neighbors_ecal_eq_0=2000,n_neighbors_ecal_neq_0=250,weights='gaussian',algorithm='auto',sigma=5,lim=-1):
        """
        To create a KNN Calibration with this EnergyData as training values.
        
        Parameters
        ----------
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
        if ecal + hcal > lim, the calibrated energy ecalib = math.nan
        if lim = - 1, there is no limit
        
        Returns
        -------
        calib : pfcalibration.KNN
        The calibration
        """
        begin = time.time()
        calib = KNN(self.ecal,self.hcal,self.true,n_neighbors_ecal_eq_0,n_neighbors_ecal_neq_0,weights,algorithm,sigma,lim)
        end = time.time()
        print("KNN - Calibration made in",end-begin,"s")
        return calib

    def KNNGaussianCleaning(self,n_neighbors_ecal_eq_0=2000,n_neighbors_ecal_neq_0=250,weights='gaussian',algorithm='auto',sigma=5,lim=-1,energystep=1,kind='cubic',cut=2):
        """
        To create a KNNGaussianCleaning Calibration with this EnergyData as training values.
        
        Parameters
        ----------
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
        if ecal + hcal > lim, the calibrated energy ecalib = math.nan
        if lim = - 1, there is no limit

        energystep : float
        step between two points of evaluation

        kind : str or int, optional
        Specifies the kind of interpolation as a string (‘linear’, ‘nearest’,
        ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’ where ‘zero’, ‘slinear’,
        ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth,
        first, second or third order) or as an integer specifying the order of
        the spline interpolator to use. Default is ‘linear’

        cut : float
        cut to reject points
        we only consider the points with true energy between
        mu - cut * sigma and mu - cut * sigma (mu and sigma the mean and std
        of the gaussian fit)
        
        Returns
        -------
        calib : pfcalibration.KNNGaussianCleaning
        The calibration
        """
        begin = time.time()
        calib = KNNGaussianCleaning(self.ecal,self.hcal,self.true,n_neighbors_ecal_eq_0,n_neighbors_ecal_neq_0,weights,algorithm,sigma,lim,energystep,kind,cut)
        end = time.time()
        print("KNNGaussianCleaning - Calibration made in",end-begin,"s")
        return calib

    def KNNGaussianFit(self,n_neighbors_ecal_eq_0=2000,n_neighbors_ecal_neq_0=250,algorithm='auto',lim=-1,energystep_ecal_eq_0=1,energystep_ecal_neq_0 = 5,kind='cubic'):
        """
        To create a KNNGaussianFit Calibration with this EnergyData as training values.
        
        Parameters
        ----------
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
        if ecal + hcal > lim, the calibrated energy ecalib = math.nan
        if lim = - 1, there is no limit

        energystep_ecal_eq_0 : float
        step between two points of evaluation
        for ecal == 0
        
        energystep_ecal_neq_0 : float
        step between two points of evaluation
        for ecal != 0

        kind : str or int, optional
        Specifies the kind of interpolation as a string (‘linear’, ‘nearest’,
        ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’ where ‘zero’, ‘slinear’,
        ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth,
        first, second or third order) or as an integer specifying the order of
        the spline interpolator to use. Default is ‘linear’
        
        Returns
        -------
        calib : pfcalibration.KNNGaussianFit
        The calibration
        """
        begin = time.time()
        calib = KNNGaussianFit(self.ecal,self.hcal,self.true,n_neighbors_ecal_eq_0,n_neighbors_ecal_neq_0,algorithm,lim,energystep_ecal_eq_0,energystep_ecal_neq_0,kind)
        end = time.time()
        print("KNNGaussianFit - Calibration made in",end-begin,"s")
        return calib

    def KNNGaussianFitDirect(self,n_neighbors_ecal_eq_0=2000,n_neighbors_ecal_neq_0=250,algorithm='auto',lim=-1):
        """
        To create a KNNGaussianFitDirect Calibration with this EnergyData as training values.
        
        Parameters
        ----------
        
        Returns
        -------
        calib : pfcalibration.KNNGaussianFitDirect
        The calibration
        """
        begin = time.time()
        calib = KNNGaussianFitDirect(self.ecal,self.hcal,self.true,n_neighbors_ecal_eq_0,n_neighbors_ecal_neq_0,algorithm,lim)
        end = time.time()
        print("KNNGaussianFitDirect - Calibration made in",end-begin,"s")
        return calib

    def splitInTwo(self):
        """
        To split in two sets of datas

        Returns
        -------
        data1,data2 : pfcalibration.EnergyData, pfcalibration.EnergyData
        the both pfcalibration.EnergyData

        """
        true1 = []
        p1 = []
        ecal1 = []
        hcal1 = []
        eta1 = []
        true2 = []
        p2 = []
        ecal2 = []
        hcal2 = []
        eta2 = []
        for i in np.arange(len(self.ecal)):
            if i%2 == 0:
                true1.append(self.true[i])
                p1.append(self.p[i])
                ecal1.append(self.ecal[i])
                hcal1.append(self.hcal[i])
                eta1.append(self.eta[i])
            else:
                true2.append(self.true[i])
                p2.append(self.p[i])
                ecal2.append(self.ecal[i])
                hcal2.append(self.hcal[i])
                eta2.append(self.eta[i])
        data1 = EnergyData(np.array(true1),np.array(p1),np.array(ecal1),np.array(hcal1),np.array(eta1))
        data2 = EnergyData(np.array(true2),np.array(p2),np.array(ecal2),np.array(hcal2),np.array(eta2))
        return data1, data2

    def oneOverTen(self):
        """
        To keep only one over ten particules
        
        Returns
        -------
        data1 : pfcalibration.EnergyData
        """
        true1 = []
        p1 = []
        ecal1 = []
        hcal1 = []
        eta1 = []
        for i in np.arange(len(self.ecal)):
            if i%10 == 0:
                true1.append(self.true[i])
                p1.append(self.p[i])
                ecal1.append(self.ecal[i])
                hcal1.append(self.hcal[i])
                eta1.append(self.eta[i])
        data1 = EnergyData(np.array(true1),np.array(p1),np.array(ecal1),np.array(hcal1),np.array(eta1))
        return data1
    
    def mergeWith(self,another):
        """
        To meger the data of 'self' with those of 'another'

        Parameters
        ----------
        another : pfcalibration.EnergyData

        Returns
        -------
        merged : pfcalibration.EnergyData

        """
        true = np.concatenate([self.true,another.true])
        p = np.concatenate([self.p,another.p])
        ecal = np.concatenate([self.ecal,another.ecal])
        hcal = np.concatenate([self.hcal,another.hcal])
        eta = np.concatenate([self.eta,another.eta])
        merged = EnergyData(true, p, ecal, hcal, eta)
        return merged
