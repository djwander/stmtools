# stmtools

This package contains python code for easy and quick data analysis of STM measurements acquired with the Nanonis SPM Controller (tested for Version 4).
## Basic structure
The package follows and object oriented approach, so each measurement is represented by an object.
**NanonisMeasurement.py** containts the main code which loads the raw data and provides a user friendly interface to it.

Sample specific standard data treatment procedures are implemented in separate classes.<br>
**SuperconductingSample.py** contains such code for superconducting samples, which I studied during my PhD.<br>
**SuperconductingTipDeconvolution.py** hosts code for the deconvolution of DOS measurements using a superconducting tip.<br>
**STMZSpectroscopy.py** provides basic analysis of STM IZ measurements such as the extraction of the current decay constant and the corresponding sample workfunction.


The above listed sample specific classes use physical laws and constants. These are grouped in **Physics.py**. 
Note that Physics.py only contains theory or literature values and laws but no code related to experimental data and how to treat it.


Finally, the file **SionludiMeasurement.py** implements code to read the state (here: the temperature) of our Sionludi dilution refrigerator.
This is a very specific example for how to integrate the temperature measurement into the NanonisMeasurement classes. If you use a diferent refrigerator, you might want to write your own class implementing a similar behavior.

## Getting started
For representing 1D and 2D data, the package uses the simscidapy package. 
I therefore reccomend to start by taking a look at simscidapy and the examples it provides to understand how to use it for easy and quick data analysis.

Then, look at the examples in **NanonisMeasurement-Examples.ipynb** to get an overview of how to load and plot all kinds of Nanonis Measurements.

To see an example for how I decided to implement common data treatment procedures in a clean and reusable way, take a look at **STMZSpectroscopy.py** and its examples in **STMZSpectroscopy-Examples.ipynb**.

Finally, if you are working with superconducting samples or tips, check out **SuperconductingSample-Examples.ipynb** and **SuperconductingTipDeconvolution-Examples.ipynb**.

