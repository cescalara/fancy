#!/bin/bash 
#
# Initial configurations required to use this package
#
# set up mplstyles 
python ./fancy/plotting/config_mplstyles.py
#
# set up GMF model from CRPropa resources
# add some flag if we want to import GMF model later on
wget https://www.desy.de/~crpropa/data/magnetic_lenses/JF12full.tgz -P ./fancy/analysis
tar -xvf ./fancy/analysis/JF12full.tgz
