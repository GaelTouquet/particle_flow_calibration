### How does it works
For the particles flow we need to know the energies of the particles thanks to the hadronic calorimeter and electromagnetic calorimeter.
We use simulated particles to creates models to obtains a calibrate energy thanks to simulated particles.

### Root file to python
First of all you have to install ROOT.

After, if you have a root file with simulated particles, you have to convert them in a python file understandable by the program.

For this you have to use the program '`convertRootFile.py`'

Step 1 : make the file as executable
```markdown
`chmod +x convertRootFile.py`
```
Step 2 : launch the program

example with `prod2_200_400k.root` and `charged_hadrons_100k.root`
```markdown
`./convertRootFile.py prod2_200_400k.root charged_hadrons_100k.root`
```
Step 3 : you can use the new files in '`.energydata`' in the other programs

### To create a calibration
