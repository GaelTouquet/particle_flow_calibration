#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developed by Samuel Niang
For IPNL (Nuclear Physics Institute of Lyon)
"""

from sklearn import neighbors
import numpy as np
import math
from scipy.interpolate import interp2d,interp1d
from tools import gaussian_param, optimized_binwidth
from scipy.optimize import curve_fit
import warnings
from scipy.optimize import OptimizeWarning


class KNNGaussianFit:
    """
    Class to calibrate the true energy of a particle thanks to training datas.
    We use the a k neareast neighbours method, we fit the histogramm of the
    true energy of the neighbours by a gaussian and consider the mean of the
    gaussian distribution is the approximation of the true energy.
    We do an iterpolation to determine the other values.

    Attributs
    ---------
    n_neighbors: int
    Number of neighbors to use by default for k_neighbors queries.

    algortihm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
    Algorithm used to compute the nearest neighbors:
    'ball_tree' will use BallTree
    'kd_tree' will use KDtree
    'brute' will use a brute-force search.
    'auto' will attempt to decide the most appropriate algorithm based
    on the values passed to fit method.

    lim : float
    to reject calibration points with ecal + hcal > lim
    if lim = - 1, there is no limit

    kind : str or int, optional
    Specifies the kind of interpolation as a string (‘linear’, 'cubic')
    Default is 'cubic'

    ecal_train : array
    ecal value to train the calibration

    hcal_train : array
    ecal value to train the calibration

    true_train : array
    ecal value to train the calibration

    neigh_ecal_neq_0 : sklearn.neighbors.NearestNeighbors
    the sklearn.neighbors.NearestNeighbors for ecal != 0

    neigh_ecal_eq_0 : sklearn.neighbors.NearestNeighbors
    the sklearn.neighbors.NearestNeighbors for ecal == 0

    interpolation_ecal_neq_0 : scipy.interpolate.interp2d
    a 2D interpolation for the ecal != 0

    interpolation_ecal_eq_0 : scipy.interpolate.interp1d
    a 1D interpolation for the ecal == 0

    evaluatedPoint_hcal_ecal_eq_0 : array
    the hcal coordinates where we evaluate a true energy values for
    ecal == 0

    evaluatedPoint_true_ecal_eq_0 : array
    evaluated energy values for ecal == 0

    evaluatedPoint_neighbours_hcal_ecal_eq_0 : array of arrays
    hcal energy values of neighbours of the points where we evaluate a
    true energy value, for ecal == 0

    evaluatedPoint_neighbours_true_ecal_eq_0 : array of arrays
    true energy values of neighbours of the points where we evaluate a
    true energy value, for ecal == 0

    evaluatedPoint_parameters_ecal_eq_0 : array of arrays
    array[0] sigma
    array[1] mu
    array[2] k
    g(x) = k * exp(-1/2 [(x-mu)/sigma]**2)
    the parameters of the gaussian which fit each distributions of
    neighbours, for ecal == 0

    evaluatedPoint_entries_ecal_eq_0 : array of arrays
    the histograms entries of each distributions of neighbours, for
    ecal == 0

    evaluatedPoint_bin_middles_ecal_eq_0 : array of arrays
    the histograms bin middles of each distributions of neighbours,
    for ecal == 0

    evaluatedPoint_true_min_ecal_eq_0 : array
    true energy corresponding to mu - cut * sigma, for ecal == 0

    evaluatedPoint_true_max_ecal_eq_0 : array
    true energy corresponding to mu + cut * sigma, for ecal == 0

    evaluatedPoint_ecal : array
    the ecal coordinates where we evaluate a true energy values for
    ecal != 0

    evaluatedPoint_hcal : array
    the hcal coordinates where we evaluate a true energy values for
    ecal != 0

    evaluatedPoint_true : array
    evaluated energy values for ecal != 0

    evaluatedPoint_neighbours_ecal : array of arrays
    ecal energy values of neighbours of the points where we evaluate a
    true energy value, for ecal != 0

    evaluatedPoint_neighbours_hcal : array of arrays
    hcal energy values of neighbours of the points where we evaluate a
    true energy value, for ecal != 0

    evaluatedPoint_neighbours_true : array of arrays
    true energy values of neighbours of the points where we evaluate a
    true energy value, for ecal != 0

    evaluatedPoint_parameters : array of arrays
    array[0] sigma
    array[1] mu
    array[2] k
    g(x) = k * exp(-1/2 [(x-mu)/sigma]**2)
    the parameters of the gaussian which fit each distributions of
    neighbours, for ecal != 0

    evaluatedPoint_entries : array of arrays
    the histograms entries of each distributions of neighbours, for
    ecal != 0

    evaluatedPoint_bin_middles : array of arrays
    the histograms bin middles of each distributions of neighbours,
    for ecal != 0

    """
    def __init__(self,ecal_train=[],hcal_train=[],true_train=[],n_neighbors=1,algorithm='auto',lim=-1,energystep = 3,kind='cubic'):
        """
        Parameters
        ----------
        ecal_train : array-like
        ecal value to train the calibration

        hcal_train : array-like
        hcal value to train the calibration

        true_train : array-like
        true value to train the calibration

        n_neighbors: int
        Number of neighbors to use by default for k_neighbors queries.

        algortihm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
        Algorithm used to compute the nearest neighbors:
        'ball_tree' will use BallTree
        'kd_tree' will use KDtree
        'brute' will use a brute-force search.
        'auto' will attempt to decide the most appropriate algorithm based on
        the values passed to fit method.

        lim : float
        to reject calibration points with ecal + hcal > lim
        if lim = - 1, there is no limit

        energystep : float
        step between two points of evaluation

        kind : str or int, optional
        Specifies the kind of interpolation as a string (‘linear’, ‘nearest’,
        ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’ where ‘zero’, ‘slinear’,
        ‘quadratic’ and ‘cubic’ refer to a spline interpolation of zeroth,
        first, second or third order) or as an integer specifying the order of
        the spline interpolator to use. Default is ‘linear’

        """

        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.kind=kind
        self.evaluatedPoint_hcal_ecal_eq_0 = []
        self.evaluatedPoint_true_ecal_eq_0 = []
        self.evaluatedPoint_neighbours_hcal_ecal_eq_0 = []
        self.evaluatedPoint_neighbours_true_ecal_eq_0 = []
        self.evaluatedPoint_parameters_ecal_eq_0 = []
        self.evaluatedPoint_entries_ecal_eq_0 = []
        self.evaluatedPoint_bin_middles_ecal_eq_0 = []
        self.evaluatedPoint_ecal = []
        self.evaluatedPoint_hcal = []
        self.evaluatedPoint_true = []
        self.evaluatedPoint_neighbours_ecal = []
        self.evaluatedPoint_neighbours_hcal = []
        self.evaluatedPoint_neighbours_true = []
        self.evaluatedPoint_parameters = []
        self.evaluatedPoint_entries = []
        self.evaluatedPoint_bin_middles = []
        self.evaluatedPoint_reducedchi2 = []
        self.evaluatedPoint_reducedchi2_ecal_eq_0 = []

        # we reject calibration points with ecal + hcal > lim
        if lim == -1:
            lim = max(max(ecal_train),max(hcal_train))
        self.lim = lim
        self.ecal_train = ecal_train[ecal_train+hcal_train<=lim]
        self.hcal_train = hcal_train[ecal_train+hcal_train<=lim]
        self.true_train = true_train[ecal_train+hcal_train<=lim]


        #Case ecal == 0
        self.neigh_ecal_eq_0 = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm)
        y = self.hcal_train[self.ecal_train == 0]
        z = self.true_train[self.ecal_train == 0]
        self.neigh_ecal_eq_0.fit(np.transpose(np.matrix(y)))

        def forOnePoint_ecal_eq_0(h):
            # the neighbours of the point (ecal,hcal) = (0,h)
            dist, ind = self.neigh_ecal_eq_0.kneighbors(X = h)
            true = z[ind][0]
            hcal = y[ind][0]
            binwidth = 1
            nbins = np.arange(min(true), max(true) + binwidth, binwidth)
            with warnings.catch_warnings():
                try:
                    #we create the histogram
                    warnings.simplefilter("error", OptimizeWarning)
                    entries, bin_edges = np.histogram(true,bins=nbins)
                    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
                    bin_middles = bin_middles[entries != 0]
                    entries = entries[entries != 0]
                    reduced = math.nan

                    # we fit the histogram
                    p0 = np.sqrt(np.std(entries)),bin_middles[np.argmax(entries)],max(entries)
                    error = np.sqrt(entries)
                    parameters, cov_matrix = curve_fit(gaussian_param, bin_middles, entries,sigma=error,p0=p0)
                    res = parameters[1]

                    chi2 = np.sum(((gaussian_param(bin_middles,*parameters)-entries)/error)**2)
                    reduced = chi2/(len(bin_middles)-len(parameters))

                    if reduced > 5:
                        raise OptimizeWarning

                except (OptimizeWarning, RuntimeError):
                    parameters = p0
                    res = parameters[1]
                    print("calibration issue for ecal = 0, hcal = ",h,"reduced chi2 = ",reduced)
                finally:
                    # we save the values in the attributs
                    self.evaluatedPoint_hcal_ecal_eq_0.append(h)
                    self.evaluatedPoint_true_ecal_eq_0.append(res)
                    self.evaluatedPoint_parameters_ecal_eq_0.append(parameters)
                    self.evaluatedPoint_neighbours_hcal_ecal_eq_0.append(hcal)
                    self.evaluatedPoint_neighbours_true_ecal_eq_0.append(true)
                    self.evaluatedPoint_entries_ecal_eq_0.append(entries)
                    self.evaluatedPoint_bin_middles_ecal_eq_0.append(bin_middles)
                    self.evaluatedPoint_reducedchi2_ecal_eq_0.append(reduced)

                    return res

        #we define the first point of evaluation
        dist, ind = self.neigh_ecal_eq_0.kneighbors(X = 0)
        hcal = y[ind][0]
        hcal_min = (max(hcal)+min(hcal))/2
        # we evaluate the true energies
        hcal = np.arange(hcal_min,self.lim+energystep,energystep)
        vect = np.vectorize(forOnePoint_ecal_eq_0)
        true = vect(hcal)
        hcal = hcal[np.invert(np.isnan(true))]
        true = true[np.invert(np.isnan(true))]
        # we create the interpolation
        self.interpolation_ecal_eq_0 = interp1d(hcal,true,kind=kind,fill_value='extrapolate')


        # Case ecal != 0
        self.neigh_ecal_neq_0 = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm)
        x = self.ecal_train[self.ecal_train != 0]
        y = self.hcal_train[self.ecal_train != 0]
        z = self.true_train[self.ecal_train != 0]
        self.neigh_ecal_neq_0.fit(np.transpose(np.matrix([x,y])))

        def forOnePoint_ecal_neq_0(e,h):
            # the neighbours of the point (ecal,hcal) = (e,h)
            dist, ind = self.neigh_ecal_neq_0.kneighbors([[e,h]])
            true = z[ind][0]
            hcal = y[ind][0]
            ecal = x[ind][0]
            binwidth = 1
            nbins = np.arange(min(true), max(true) + binwidth, binwidth)
            with warnings.catch_warnings():
                try:
                    #we create the histogram
                    warnings.simplefilter("error", OptimizeWarning)
                    entries, bin_edges = np.histogram(true,bins=nbins)
                    bin_middles = 0.5*(bin_edges[1:] + bin_edges[:-1])
                    bin_middles = bin_middles[entries != 0]
                    entries = entries[entries != 0]
                    reduced = math.nan

                    # we fit the histogram
                    p0 = np.sqrt(np.std(entries)),bin_middles[np.argmax(entries)],max(entries)
                    error = np.sqrt(entries)
                    parameters, cov_matrix = curve_fit(gaussian_param, bin_middles, entries,sigma=error,p0=p0)
                    res = parameters[1]

                    chi2 = np.sum(((gaussian_param(bin_middles,*parameters)-entries)/error)**2)
                    reduced = chi2/(len(bin_middles)-len(parameters))

                    if reduced > 5:
                        raise OptimizeWarning

                except (OptimizeWarning,RuntimeError):
                    parameters = p0
                    res = parameters[1]
                    if e + h < self.lim:
                        print("calibration issue for ecal = ",e," hcal = ",h)
                finally:
                    # we save the values in the attributs
                    self.evaluatedPoint_neighbours_ecal.append(ecal)
                    self.evaluatedPoint_neighbours_hcal.append(hcal)
                    self.evaluatedPoint_neighbours_true.append(true)
                    self.evaluatedPoint_entries.append(entries)
                    self.evaluatedPoint_bin_middles.append(bin_middles)
                    self.evaluatedPoint_ecal.append(e)
                    self.evaluatedPoint_hcal.append(h)
                    self.evaluatedPoint_true.append(res)
                    self.evaluatedPoint_parameters.append(parameters)
                    self.evaluatedPoint_reducedchi2.append(reduced)
                    return res

        #we define the first point of evaluation
        dist, ind = self.neigh_ecal_neq_0.kneighbors(X = [[0,0]])
        hcal = y[ind][0]
        ecal = x[ind][0]
        hcal_min = (max(hcal)+min(hcal))/2
        ecal_min = (max(ecal)+min(ecal))/2
        # we evaluate the true energies
        hcal = np.linspace(hcal_min,self.lim,(self.lim-hcal_min)/energystep)
        ecal = np.linspace(ecal_min,self.lim,(self.lim-ecal_min)/energystep)
        eecal, hhcal = np.meshgrid(ecal,hcal)
        vect = np.vectorize(forOnePoint_ecal_neq_0)
        true = vect(eecal,hhcal)
        # we create the interpolation
        self.interpolation_ecal_neq_0 = interp2d(ecal,hcal,true,kind=kind)

    def predict(self,e,h):
        """
        To predict the true energies thanks to couples of ecal, hcal

        Parameters
        ----------
        e : a numpy array of ecal energies
        h : a numpy array of hcal energies

        Returns
        -------
        true : a numpy array of predicted true energies
        the value is NaN if the asked value is off-limit
        """
        def predictSingleValue(ecal,hcal):
            if ecal+hcal < self.lim:
                if ecal == 0:
                    res = self.interpolation_ecal_eq_0(hcal)
                else:
                    res = self.interpolation_ecal_neq_0(ecal,hcal)
                if res < 0:
                    res = 0
            else:
                res = math.nan
            return res

        vect = np.vectorize(predictSingleValue)
        return vect(e,h)
