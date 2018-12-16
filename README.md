# covariance_emulator

The `covariance_emulator` is a tool to predict covariance matrices using Gaussian process regression given a set of training matrices at a specified set of domain locations (i.e. "x values"). This is very helpful in cosmology, since the covariance matrix depends directly on the cosmological parameters being estimated in the likelihood we write down.

This tool is general, in that it can be used to construct emulators for any set of covariance matrix (in fact, any set of matrices that can be LDL decomposed).

## Installation

For now, the easiest way to install is to clone the repository and install with the `setup.py` script.
```
git clone https://github.com/tmcclintock/covariance_emulator
cd covariance_emulator
python setup.py install
```
You should then run the unit tests with
```
python setup.py test
```
If any fail, please copy/paste their output into an issue.