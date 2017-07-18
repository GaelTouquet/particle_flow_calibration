#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)

"""

from pfcalibration.tools import importData, importCalib # to import binary data
import pfcalibration.usualplots as usplt
from pfcalibration.tools import savefig, projectionCone, selectionCone
import matplotlib.pyplot as plt
import numpy as np


#importation of simulated particles
filename = 'charged_hadrons_100k.energydata'
data1 = importData(filename)
filename = 'prod2_200_400k.energydata'
data2 = importData(filename)
# we merge the 2 sets of data
data1 = data1.mergeWith(data2)
# we split the data in 2 sets
data1,data2 = data1.splitInTwo()
#data 1 -> training data
#data 2 -> data to predict

directory = "pictures/explain/"
lim = 150
theta = np.pi/4
delta = np.pi/40
rlim = lim/(np.cos(theta)+np.sin(theta))

ecal = data1.ecal
hcal = data1.hcal
true = data1.true

proj = projectionCone(ecal,hcal,true,theta,delta)
select = selectionCone(ecal,hcal,true,theta,delta)

hcal_ecal_eq_0 = data1.hcal[ecal == 0]
true_ecal_eq_0 = data1.true[ecal == 0]

fig = plt.figure(figsize=(12,6))
plt.subplot(2,2,2)
plt.title(r"Points with $e_{cal} = 0$",fontsize = 18)
plt.xlabel(r"$h_{cal}$",fontsize = 18)
plt.ylabel(r"$e_{true}$",fontsize = 18)

plt.plot(hcal_ecal_eq_0,true_ecal_eq_0,'.',markersize=1)
ind = hcal_ecal_eq_0 > lim
plt.plot(hcal_ecal_eq_0[ind],true_ecal_eq_0[ind],'.',markersize=1)
plt.plot([lim,lim],[0,max(true_ecal_eq_0)],'--',lw=2)

plt.subplot(2,2,4)
plt.title(r"Points with $\theta \in [\pi/4;\pi/4 + \pi/40]$",fontsize = 18)
plt.xlabel(r"$e_{\theta}$",fontsize = 18)
plt.ylabel(r"$e_{true}$",fontsize = 18)

plt.plot(proj[0],proj[1],'.',markersize=1)
ind = proj[0] > rlim
plt.plot(proj[0][ind],proj[1][ind],'.',markersize=1)
plt.plot([rlim,rlim],[0,max(proj[1])],'--',lw=2)

plt.subplot(1,2,1)
plt.plot(ecal,hcal,'.',markersize=1)
ind = ecal+hcal > lim
plt.plot(ecal[ind],hcal[ind],'.',markersize=1, label = r"rejected points")
plt.plot(select[0],select[1],'.',markersize=1, label = r"$\theta \in [\pi/4;\pi/4 + \pi/40]$")
plt.xlabel(r"$e_{cal}$",fontsize = 18)
plt.ylabel(r"$h_{cal}$",fontsize = 18)
plt.legend(loc="upper right",fontsize = 16 )

plt.tight_layout()
plt.show()
savefig(fig,directory,"limit.png")
plt.close()