#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Samuel Niang
For IPNL (Nuclear Physics Institue of Lyon)
"""

import numpy as np
import time
import math

class CalibrationLego:
    """
    Class to calibrate data 
    The aim :
        given one ecal, hcal, what is the true energie?
    The way :
        The plane (ecal,hcal) is divided in nbLego*nbLego squares an for each 
        square, we estimate the mean value of true energy
    """
    def __init__(self,data,nbLego,timeInfo = False):
        begin = time.time()
        nbLego = int(nbLego)
        self.nbLego = nbLego

        ecal = np.linspace(0,data.ener_max,nbLego)
        hcal = np.linspace(0,data.ener_max,nbLego)
        self.delta = hcal[1]-hcal[0]
        self.nbLego = nbLego
        ecal_calib = []
        hcal_calib = []
        true_calib = []
        # when ecal == 0
        true_calib_lim = []
        precision = []
        
        bins_ecal = np.arange(nbLego-1)
        bins_hcal = np.arange(nbLego-1)
        for i in bins_ecal:
                ind_ecal = np.logical_and(data.ecal>= ecal[i],data.ecal<ecal[i+1])
                ind_ecal = np.logical_and(data.ecal!=0,ind_ecal)
                for j in bins_hcal:
                    ind_hcal = np.logical_and(data.hcal>= hcal[j],data.hcal<hcal[j+1])
                    ind_true = np.logical_and(ind_ecal,ind_hcal)
                    true = data.true[ind_true]
                    if true.size != 0:
                        true_calib.append(np.mean(true))
                        precision.append(np.std(true)/np.sqrt(len(true))/np.mean(true))
                    else :
                        true_calib.append(0)
                        precision.append(None)
                    ecal_calib.append(np.mean([ecal[i],ecal[i+1]]))
                    hcal_calib.append(np.mean([hcal[j],hcal[j+1]]))
        # when ecal == 0 
        for i in bins_hcal:
            ind_hcal = np.logical_and(data.hcal>= hcal[i],data.hcal<hcal[i+1])
            ind_true = np.logical_and(data.ecal == 0,ind_hcal)
            true = data.true[ind_true]
            if true.size != 0:
                true_calib_lim.append(np.mean(true))
            else :
                true_calib_lim.append(0)
                    
        self.ecal = np.array(ecal_calib)
        self.ecal_max = max(self.ecal)
        self.hcal = np.array(hcal_calib)
        self.hcal_max = max(self.hcal)
        self.true = np.array(true_calib)
        self.true_lim = np.array(true_calib_lim)
        self.precision = np.array(precision)
        end = time.time()
        if timeInfo:
            print("Lego profile made in",end-begin,"s")
        
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
        
        if (e >  self.ecal_max or h > self.hcal_max):
            true = math.nan
        else:
            if e != 0:
                i_ecal = int(e/self.delta)
                i_hcal = int(h/self.delta)
                i = i_hcal + i_ecal*(self.nbLego-1)   
                true = self.true[i] 
            else:
                i_hcal = int(h/self.delta)
                true = self.true_lim[i_hcal]
        return true

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
            print("Calibration made in",end-begin,"s")
        return ecalib
        
    def getPrecisionSingleValue(self,e,h):
        """
        To get the relative precision on the true energy from a couple of ecal, hcal
        
        Parameters
        ----------
        e : the ecal energy
        h : the hcal energy
        
        Returns
        -------
        precision : the relative precision on the true energy
        """
        if (e >  self.ecal_max or h > self.hcal_max):
            precision = 0
        else:
            i_ecal = int(e/self.delta)
            i_hcal = int(h/self.delta)
            i = i_hcal + i_ecal*(self.nbLego-1)   
            precision = self.precision[i] 
        return precision
    
    def getPrecision(self,e,h):
        """
        To predict the true energies from couples of ecal, hcal
        
        Parameters
        ----------
        e : a numpy array of ecal energies
        h : a numpy array of hcal energies
        
        Returns
        -------
        precision : a numpy array of relative precisions on the true energies
        """
        vect = np.vectorize(self.getPrecisionSingleValue)
        precision = vect(e,h)
        return precision
        
