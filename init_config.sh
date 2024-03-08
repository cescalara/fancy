#!/bin/bash 

# Initial configurations required to use this package

# directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# set up mplstyles 
printf "Configuring mplstyles...\n"
python ./fancy/plotting/config_mplstyles.py

# make nuclear tables
printf "making nuclear tables...\n"
python ./fancy/interfaces/make_nuclear_table.py

# set up GMF model from CRPropa resources
# add some flag if we want to import GMF model later on
# TODO: also make some conditional to ignore fetching of JF12Full.tgz since this takes a while
printf "Fetching GMF model from CRPropa...\n"
wget https://www.desy.de/~crpropa/data/magnetic_lenses/JF12full.tgz -P $SCRIPT_DIR/fancy/analysis
tar -xvf $SCRIPT_DIR/fancy/analysis/JF12full.tgz -C $SCRIPT_DIR/fancy/analysis
rm $SCRIPT_DIR/fancy/analysis/JF12full.tgz
printf "Done."
