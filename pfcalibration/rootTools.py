#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

import numpy as np
from ROOT import TFile
#import shelve
#from pfcalibration.EnergyData import EnergyData
#from pfcalibration.tools import exportPickle
# import pickle

"""
Fuctions using ROOT
"""

def exportPickle(filename,objectToSave):
    """
    Te export an object into a binary file
    
    Parameters
    ----------
    filename : str
    path+file name
    objectToSave : object
    """
    np.save(filename,objectToSave)
    ##d = shelve.open(filename)
    ##d['key'] = objectToSave
    ##d.close()
    #dataFile = open(filename, "w")
    #dataFile.write(str(objectToSave))
    ## pickler = pickle.Pickler(dataFile)
    ## pickler.dump(objectToSave)
    ##dataFile.close()

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
    ntot = tree.GetEntries()
    n= 0
    for event in tree:
        n+=1
        if n == 15120000:
            break
        if n%10000==0:
            print 'event :', n, '/', ntot
        if len(event.pfcs) == 1 and event.pfcs[0] == 1 and event.hcal != 0 and abs(event.eta)<1.5:
            tmpdata.append((event.true, event.p, event.ecal, event.hcal, event.eta, event.Ecalib))
    return np.array(tmpdata)

def rootToPython(filename):
    """
    Turns a root file into a binary file containing a EnergyData object usable by the programs
    """
    splited = filename.split('.')
    if splited[len(splited)-1] == "root":
        data = importRootData(filename)
        filename = ''.join(splited[0:len(splited)-1])
        energydata = data#EnergyData(data[:,0],data[:,1],data[:,2],data[:,3],data[:,4])
        exportPickle(filename+'_Data',energydata)

    else:
        print("It is not a root file")
