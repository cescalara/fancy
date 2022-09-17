# fancy

[![CI](https://github.com/cescalara/fancy/actions/workflows/tests.yml/badge.svg?branch=ta_updates)](https://github.com/cescalara/fancy/actions/workflows/tests.yml)

`fancy` is a toolbox used for source-UHECR association analyses. 

The package is tested with Python 3.8/3.9 on the latest MacOS and Ubuntu linux. Installation via pip should take care of the dependencies (see below for more info). [`CRPropa3`](https://github.com/CRPropa/CRPropa3) is treated as an optional dependency.  

## Installation
Installation is done via `pip`:

```
pip install git+https://github.com/cescalara/fancy.git
```

There are two further steps one must take to get set up:
* Run `install_cmdstan` to set up cmdstanpy (see [these instructions](https://cmdstanpy.readthedocs.io/en/v1.0.1/installation.html#function-install-cmdstan) for more information)
* Run `init_config.sh` to set up plotting styles, nuclear tables and GMF model

## License

This code is under the BSD-3 license. See [LICENSE](LICENSE) for more details.
