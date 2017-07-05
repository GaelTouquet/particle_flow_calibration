#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""
import numpy as np
from sklearn import linear_model
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
        print("the datas include",len(self.ecal),"particles")

    def linearRegression(self,lim_min = -1, lim_max=-1, lim=-1):
        """
        Linear regression of true = f(ecal,hcal)

        Parameters
        ----------

        Returns
        -------
        regr : sklearn.linear_model.LinearRegression()

        """
        begin = time.time()
        if lim == -1:
            lim = self.ener_max
        if lim_min == -1:
            ind_min = np.ones(len(self.ecal),dtype=bool)
        else:
            ind_min = self.ecal + self.hcal > lim_min
        if lim_max == -1:
            ind_max = np.ones(len(self.ecal),dtype=bool)
        else:
            ind_max = self.ecal + self.hcal < lim_max

        #CASE : ecal != 0
        ind_0 = self.ecal != 0
        ind = np.logical_and(ind_min,ind_max)
        ind = np.logical_and(ind,ind_0)
        X_train = [self.ecal[ind],self.hcal[ind]]
        X_train = np.transpose(np.matrix(X_train))
        Y_train = self.true[ind]
        Y_train = np.transpose(np.matrix(Y_train))
        regr1 = linear_model.LinearRegression()
        regr1.fit(X_train,Y_train)

        #CASE : ecal == 0
        ind_0 = self.ecal == 0
        ind = np.logical_and(ind_min,ind_max)
        ind = np.logical_and(ind,ind_0)
        X_train = self.hcal[ind]
        X_train = np.transpose(np.matrix(X_train))
        Y_train = self.true[ind]
        Y_train = np.transpose(np.matrix(Y_train))
        regr2 = linear_model.LinearRegression()
        regr2.fit(X_train,Y_train)
        regr = LinearRegression(regr1,regr2,lim_min, lim_max, lim)

        end = time.time()
        print("linearRegression - Calibration made in",end-begin,"s")
        return regr

    def calibrationLego(self,nbLego,timeInfo = True):
        """
        Effectue une calibration lego

        Returns
        -------
        cal : CalibrationLego()

        """
        return CalibrationLego(self,nbLego,timeInfo)

    def kNN(self,n_neighbors=1,weights='gaussian',algorithm='auto',sigma=1,lim=-1):
        begin = time.time()
        calib = KNN(self.ecal,self.hcal,self.true,n_neighbors,weights,algorithm,sigma,lim)
        end = time.time()
        print("KNN - Calibration made in",end-begin,"s")
        return calib

    def kNNGaussianCleaning(self,n_neighbors=2000,weights='gaussian',algorithm='auto',sigma=1,lim=-1,energystep = 5,kind='cubic',cut=2):
        begin = time.time()
        calib = KNNGaussianCleaning(self.ecal,self.hcal,self.true,n_neighbors,weights,algorithm,sigma,lim,energystep,kind,cut)
        end = time.time()
        print("KNNGaussianCleaning - Calibration made in",end-begin,"s")
        return calib

    def kNNGaussianFit(self,n_neighbors=2000,algorithm='auto',lim=-1,energystep = 3,kind='cubic'):
        begin = time.time()
        calib = KNNGaussianFit(self.ecal,self.hcal,self.true,n_neighbors,algorithm,lim,energystep,kind)
        end = time.time()
        print("KNNGaussianFit - Calibration made in",end-begin,"s")
        return calib

    def kNNGaussianFitDirect(self,n_neighbors_ecal_eq_0=2000,n_neighbors_ecal_neq_0=250,algorithm='auto',lim=-1):
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
        Fusionne les données de self et de another

        Parameters
        ----------
        another : EnergyData

        Returns
        -------
        merged : EnergyData

        """
        true = np.concatenate([self.true,another.true])
        p = np.concatenate([self.p,another.p])
        ecal = np.concatenate([self.ecal,another.ecal])
        hcal = np.concatenate([self.hcal,another.hcal])
        eta = np.concatenate([self.eta,another.eta])
        merged = EnergyData(true, p, ecal, hcal, eta)
        return merged
