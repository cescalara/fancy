import numpy as np
from matplotlib import pyplot as plt

from ..utils import PlotStyle
from ..allskymap import *

__all__ = ['Source']

class Source():
    """
    Stores the data and parameters for sources
    """

    
    def __init__(self, data):
        """
        Stores the data and parameters for sources.
        
        :param data: data passed as an instance of Data
        """

        self.N = data.get_len()
        
        self.coord = data.get_coordinates()

        self.distance = data.get_by_name('D') # in Mpc

        self.name = data.get_by_name('name')

        self.label = 'Source'

        
    def plot(self, fig = None, skymap = None, save = False):
        """
        Plot the sources on a map of the sky. 

        :param fig: pass a figure to plot on an existing figure
        :param skymap: pass an AllSkyMap instance to plot on an existing skymap
        :param save: if True, save the figure after plotting to `../plots/`

        :return: the figure and AllSkyMap instance for further plotting.
        """
        
        # plot style
        style = PlotStyle()

        # figure
        if fig == None:
            fig = plt.figure(figsize = (12, 6));
        #ax = plt.gca()

        # skymap
        if skymap == None:
            skymap = AllSkyMap(projection = 'hammer', lon_0 = 0, lat_0 = 0);

        # plot the candidate locations
        label = 0
        for lon, lat in np.nditer([self.coord.galactic.l.deg, self.coord.galactic.b.deg]):
            if label == 0:
                skymap.tissot(lon, lat, 5., 30, 
                              facecolor = style.textcolor, 
                              alpha = style.alpha_level, label = 'Source')
                label = 1
            else:
                skymap.tissot(lon, lat, 5., 30, 
                              facecolor = style.textcolor, alpha = style.alpha_level)
            
        if save == True:
            plt.savefig('plots/source_map.pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0.5)
            
        return fig, skymap
