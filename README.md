# fancy

**Forked over from [cescalara/fancy](https://github.com/cescalara/fancy)**

`fancy` is a toolbox used for UHECR modelling, fitting, and plotting. The majority of the tools used in [uhecr-project/uhecr_model](https://github.com/uhecr-project/uhecr_model) (forked over from [cescalara/uhecr_model](https://github.com/cescalara/uhecr_model)) are contained in this package. 

## Dependencies

With this updated version, this code uses the following (main) dependencies:
- Python <=3.8
- NumPy <=1.19  (to allow calculations with MKL)
- seaborn >=0.11  (for plots; due to overhauls with kdeplot)
- h5py >=3.0  (for I/O, due to deprecations with <v3.0)
- Basemap   (for skymaps)
- astropy  (for celestial parameters)
- mpi4py  (for energy / propagation calculations)
- matplotlib  (for plots)

Additionally, this code depends on [uhecr-project/stan_utility](https://github.com/uhecr-project/stan_utility) (forked from [cescalara/stan_utility](https://github.com/cescalara/stan_utility)) for core pystan calculations. This package depends on `pystan<=2.19`. 

## Installation
Installation is done via `pip`:
```
pip install fancy
```
Additionally, in order to utilize the plotting styles in `uhecr-project/uhecr_model`, run `init_config.sh`. This also
allows one to download the galactic field model utilized with the GMF model implemented.

## License

This code is under the BSD-3 license. See [LICENSE](LICENSE) for more details.