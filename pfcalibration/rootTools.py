#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

import numpy as np
from ROOT import TFile
from pfcalibration.EnergyData import EnergyData
from pfcalibration.tools import exportPickle

"""
Fuctions using ROOT
"""

def importRootData(filename):
    """
    Importation of the simulated datas in ROOT to a numpy array
    Parameters
    ----------
    filename : the name of the root file, string
    Returns
    -------
    tmpdata : a numpy array with
        the true energie of the hadrons
        the impulsion
        the energie detected by the electromagnetic calorimeter
        the energie detected by the hadronical calorimeter
        the pseudorapidity
    """
    # we import the data
    root_file = TFile(filename)
    tree = root_file.Get('s')
    tmpdata = []
    # we place the datas in a numpy array
    for event in tree:
        if len(event.pfcs) == 1 and event.pfcs[0] == 1 and event.hcal != 0:
            tmpdata.append((event.true, event.p, event.ecal, event.hcal, event.eta))
    return np.array(tmpdata)

def rootToPython(filename):
    """
    Turns a root file into a binary file containing a EnergyData object usable by the programs
    """
    splited = filename.split('.')
    if splited[len(splited)-1] == "root":
        data = importRootData(filename)
        filename = ''.join(splited[0:len(splited)-1])
        energydata = EnergyData(data[:,0],data[:,1],data[:,2],data[:,3],data[:,4])
        exportPickle(filename+'.energydata',energydata)

    else:
        print("It is not a root file")
