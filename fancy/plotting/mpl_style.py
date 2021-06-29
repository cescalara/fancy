''' Move the mplstyle files located in ./mplstyles to the appropriate mpl_configdir in the local file system so that the styles "minimalist" and "blues" can be used
'''
import os
import shutil
import os.path as osp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def config_mplstyle(clear=False):
    '''
    Main function that moves the custom mplstyle files in ./mplstyles to mpl_configdir/stylelib (where mpl_configdir is provided by mpl.get_configdir()) so that it is available for use with plt.style.use(). 

    Source: https://matplotlib.org/stable/tutorials/introductory/customizing.html 
    '''
    mplstyles_dir = osp.join(os.getcwd(), "mplstyles")

    stylelib_dir = osp.join(osp.abspath(mpl.get_configdir()), "stylelib")

    # first check if the path to the stylelib and mpl_config directory exists, if not then create that directory
    if not os.path.isdir(stylelib_dir):
        os.mkdir(stylelib_dir)

    # if we want to reset (clear=true), erase all mplstyles inside the mpl_config directory
    if clear:
        for fname in os.listdir(stylelib_dir):
            os.remove(osp.join(stylelib_dir, fname))

    # finally copy all the respective files into the stylelibs directory
    for fname in os.listdir(mplstyles_dir):
        shutil.copyfile(osp.join(mplstyles_dir, fname),
                        osp.join(stylelib_dir, fname))

    # '''not really sure why the above doesnt want to work, temporary workaround by using the full path to the location where mplstyles is located below'''
    # mplstyles_dir = osp.join(os.getcwd(), "mplstyles")

    # # a list of paths going to the mplstyles directory
    # mplstyle_filelist = [osp.join(mplstyles_dir, fname) for fname in os.listdir(mplstyles_dir)]

    # return mplstyle_filelist


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # run the main function
    config_mplstyle(clear=True)

    # check if we have our customized styles
    print(plt.style.available)

    plt.style.use(["minimalist", "blues"])

    # plot a random gaussian distribution to see if we have
    # our desired style
    gaussian = np.random.normal(0., 0.1, 1000)

    count, bins, _ = plt.hist(gaussian, 30, density=True)
    plt.show()
