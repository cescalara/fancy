import os, shutil
import matplotlib as mpl

from .allskymap import AllSkyMap
from .allskymap_cartopy import AllSkyMapCartopy
from .corner import *

def config_mplstyle():
    '''
    Main function that moves the custom mplstyle files in ./mplstyles to mpl_configdir/stylelib (where mpl_configdir is provided by mpl.get_configdir()) so that it is available for use with plt.style.use(). 

    Source: https://matplotlib.org/stable/tutorials/introductory/customizing.html 
    '''
    # __file__ is the path of the python script (i.e. path to this file)
    mplstyles_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "mplstyles")
    stylelib_dir = os.path.join(os.path.abspath(mpl.get_configdir()), "stylelib")

    # first check if the path to the stylelib and mpl_config directory exists, if not then create that directory
    if not os.path.isdir(stylelib_dir):
        os.mkdir(stylelib_dir)

    # finally copy all the respective files into the stylelibs directory
    for fname in os.listdir(mplstyles_dir):
        print(f"Adding mplstyle {fname} in {stylelib_dir}")
        shutil.copyfile(os.path.join(mplstyles_dir, fname),
                        os.path.join(stylelib_dir, fname))
