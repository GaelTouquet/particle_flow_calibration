#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""
from os import mkdir
from pfcalibration.tools import exportPickle
import collections

class Calibration:
    """
    Mother Class to calibrate the true energy of a particle thanks to training datas.
    All claibrations have to inherit from this class.
    
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
    
    numberPart : int
    number of particles
    
    numberPart_str : str
    number of particles
    """
    
    def __init__(self,ecal_train=[],hcal_train=[],true_train=[],lim=-1):
        
        self.ecal_train = ecal_train
        self.hcal_train = hcal_train
        self.true_train = true_train
        
        # we reject calibration points with ecal + hcal > lim
        if lim == -1:
            lim = min(max(ecal_train),max(hcal_train))
        self.lim = lim
        
        self.numberPart = len(self.ecal_train)
        if  len(self.hcal_train) != self.numberPart or len(self.true_train) != self.numberPart or len(self.hcal_train) != len(self.true_train):
            raise ValueError("ecal_train, hcal_train and true_train do not have the same length")
            
        self.numberPart_str = str(int(self.numberPart/1000))+"K"
        
        self.classname = type(self).__name__
    
    def saveCalib(self,directory = "calibrations"):
        
        splitted = directory.split('/')
        director = ""
        
        for s in splitted:
            if len(s) > 0:
                director += s+'/'
                try:
                    mkdir(director)
                except FileExistsError:
                    pass
        filename  = director      
        filename += self.classname+"_"
        filename += str(self.numberPart_str)+"part"
        od = collections.OrderedDict(sorted((self.__dict__).items()))
        for elem, value in od.items():
            if isinstance(value,(int,float)) and elem != "numberPart_str" and elem != "numberPart" :
                filename +="_"+elem+"_"+str(value)
        filename += '.calibration'
        exportPickle(filename,self)
    
    def __str__(self):
        res = self.classname+":"
        od = collections.OrderedDict(sorted((self.__dict__).items()))
        for elem, value in od.items():
            if isinstance(value,(int,float)) and elem != "numberPart_str":
                res +="\n"+elem+" -> "+str(value)
        return res
    
    def __repr__(self):
        return self.__str__(self)
    
