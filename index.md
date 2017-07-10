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
lim = 150
n_neighbors_ecal_eq_0=2000
n_neighbors_ecal_neq_0=200
energystep = 1
# We create the calibration
KNNGF = data1.KNNGaussianFit(n_neighbors_ecal_eq_0=n_neighbors_ecal_eq_0,
                             n_neighbors_ecal_neq_0=n_neighbors_ecal_neq_0,
                             lim=lim,energystep=energystep,kind='cubic')
```
