#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

import numpy as np
import time
import math

class LinearRegression:

    def __init__(self,regr1,regr2,lim_min,lim_max,lim):
        self.regr1 = regr1
        self.regr2 = regr2
        self.lim = lim
        self.lim_min = lim_min
        self.lim_max = lim_max

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
                true = self.regr1.coef_[0][0]*e + self.regr1.coef_[0][1]*h + self.regr1.intercept_[0]
            else:
                true = self.regr2.coef_[0][0]*h + self.regr2.intercept_[0]
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
    def display(self):
        print("for ecal != 0 : ")
        print("\t coef : ",self.regr1.coef_)
        print("\t intercept : ",self.regr1.intercept_)
        print("for ecal == 0 : ")
        print("\t coef : ",self.regr2.coef_)
        print("\t intercept : ",self.regr2.intercept_)
