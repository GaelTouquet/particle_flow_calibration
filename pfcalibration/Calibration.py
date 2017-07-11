#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

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
    """
    
    def __init__(self,ecal_train=[],hcal_train=[],true_train=[],lim=-1):
        
        self.ecal_train = ecal_train
        self.hcal_train = hcal_train
        self.true_train = true_train
        
        # we reject calibration points with ecal + hcal > lim
        if lim == -1:
            lim = min(max(ecal_train),max(hcal_train))
        self.lim = lim