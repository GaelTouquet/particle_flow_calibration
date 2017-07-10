# How does it works
For the particles flow we need to know the energies of the particles thanks to the hadronic calorimeter and electromagnetic calorimeter.
We use simulated particles to creates models to obtains a calibrate energy thanks to simulated particles.

## Root file to python
First of all you have to install ROOT.

After, if you have a root file with simulated particles, you have to convert them in a python file understandable by the program.

For this you have to use the program '`convertRootFile.py`'

Step 1 : make the file as executable

`chmod +x convertRootFile.py`

Step 2 : launch the program

example with `prod2_200_400k.root` and `charged_hadrons_100k.root`

`./convertRootFile.py prod2_200_400k.root charged_hadrons_100k.root`

Step 3 : you can use the new files in '`.energydata`' in the other programs

## To create a calibration
### Importation of data
```python
# file to save the pictures
directory = "pictures/testKNNGF/"
#importation of simulated particles
filename = 'charged_hadrons_100k.energydata'
data1 = importPickle(filename)
filename = 'prod2_200_400k.energydata'
data2 = importPickle(filename)
# we merge the 2 sets of data
data1 = data1.mergeWith(importPickle(filename))
# we split the data in 2 sets
data1,data2 = data1.splitInTwo()
#data 1 -> training data
#data 2 -> data to predict
```

### Example with KNNGF method
```python
# parameters of the calibration
lim = 150                   # if ecal + hcal > lim, ecalib = math.nan
n_neighbors_ecal_eq_0=2000  # number of neighbors for ecal = 0
n_neighbors_ecal_neq_0=250  # number of neighbors for ecal ≠ 0
energystep = 1
# We create the calibration
calibration = data1.KNNGaussianFit(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                             n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                             lim=lim,energystep=energystep,kind='cubic')
```

## To use a calibration
### Example with KNNGF method
```python
# with one point
ecal = 60
hcal = 60
ecalib = calibration.predict(ecal,hcal)

# with a numpy array
ecal = np.arange(0,lim,1)
hcal = np.arange(0,lim,1)
ecalib = calibration.predict(ecal,hcal)

# with a meshgrid
ecal = np.arange(0,lim,1)
hcal = np.arange(0,lim,1)
ecal,hcal = np.meshgrid(ecal,hcal)
ecalib = calibration.predict(ecal,hcal)
```