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
### Example with KNNGF method
```python
# we import the data files
filename = 'charged_hadrons_100k.energydata'
data1 = importPickle(filename)
filename = 'prod2_200_400k.energydata'
data2 = importPickle(filename)

# we merge
data1 = data1.mergeWith(data2)

# We split
data1,data2 = data1.splitInTwo()

# we create the calibration
lim = 150 # reject the points with ecal + hcal > lim
n_neighbors = 250 # number of neighbors to do the average
energystep = 1 # step of the grid of evaluation
kind = 'cubic' # kind of interpolation

KNNGF = data1.kNNGaussianFit(n_neighbors=n_neighbors,lim=lim,energystep=energystep,kind=kind)
```
