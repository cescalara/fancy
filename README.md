# fancy: **F**itting **AN**alysis toolbox for ultra-high-energy **C**osmic ra**Y**s

[![CI](https://github.com/cescalara/fancy/actions/workflows/tests.yml/badge.svg?branch=ta_updates)](https://github.com/cescalara/fancy/actions/workflows/tests.yml)

`fancy` is a toolbox used for source-UHECR association analyses using Bayesian hierarchical modelling.

## Dependencies

For computation / fitting:
- `numpy` 
- `scipy`
- `h5py`
- `astropy`
- `tqdm`
- `cmdstanpy==1.0.1` (Currently only tested with 1.0.1)

For plotting:
- `matplotlib`
- `seaborn`
- `pandas`

For CD/CI:
- `versioneer`
- `requests`

## Optional Dependencies

These dependencies are optional, but are required if one wants to fully utilise the functionalities of the framework:
- [`CRPropa3`](https://github.com/CRPropa/CRPropa3) : UHECR Propagation Framework based on a Monte-Carlo approach. Used for GMF backpropagation & lensing.
- [`PriNCe`](https://github.com/joheinze/PriNCe/tree/master) : Propagation Framework for UHECR transport on cosmological scales. Used for generating composition matrices.
- [`cartopy`](https://scitools.org.uk/cartopy/docs/latest/) : Plotting package to generate skymaps. Used to generate association skymaps.
- [`arviz`](https://python.arviz.org/en/v0.18.0/index.html) : visualisation tool for Bayesian statistics (stan outputs). Useful for general debugging.

Details of the installation of each package is given in the documentation.

## Tested configurations

- Ubuntu 20.04 + Python 3.8.19

## Installation
Installation is done via `pip`:

```
pip install git+https://github.com/cescalara/fancy.git
```

There is one further step one must take to get set up:
* Run `install_cmdstan` to set up cmdstanpy (see [these instructions](https://cmdstanpy.readthedocs.io/en/v1.0.1/installation.html#function-install-cmdstan) for more information)

When using GMF lensing, one needs to run the following command to install the GMF lens:
* `install_gmflens`: Installs a GMF lensing map based on the model of [Jansson & Farrar 2012](http://ui.adsabs.harvard.edu/abs/2012ApJ...757...14J/abstract) including turbulent effects

Additionally, for using GMF lensing specifically, the following **hard** dependencies must be followed, due to an error in SWIG for re-sampling random particles (`crpropa.getRandomParticles`):
- `numpy==1.21.5`
- `python=3.8`

## Examples

* [Joint fit of energies and arrival directions assuming proton composition](https://github.com/cescalara/uhecr_model) 
* [WIP] Including the GMF and heavier compositions coming soon!

## License

This code is under the BSD-3 license. See [LICENSE](LICENSE) for more details.
