import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from ..utils import PlotStyle
from ..allskymap import *

__all__ = ['Uhecr']

class Uhecr():
    """
    Stores the data and parameters for UHECRs
    """

    
    def __init__(self, data):
        """
        Stores the data and parameters for UHECRs.
        
        :param data: data passed as an instance of Data
        """

        self.N = data.get_len()
        
        self.coord = data.get_coordinates()

        self.year = data.get_by_name('year')

        self.day = data.get_by_name('day')

        self.incidence_angle = data.get_by_name('incidence angle')

        self.energy = data.get_by_name('energy')

        self.coord_uncertainty = 4.0 # uncertainty in degrees

        self.label = 'UHECR'

        
    def plot(self, fig = None, skymap = None, save = False):
        """
        Plot the UHECRs on a map of the sky. 

        :param fig: pass a figure to plot on an existing figure
        :param skymap: pass an AllSkyMap instance to plot on an existing skymap

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

        # plot the UHECR locations
        # use colormap for energy
        norm_E = matplotlib.colors.Normalize(self.energy.min(), self.energy.max())
        cmap = style.cmap

        lon = self.coord.galactic.l.deg
        lat = self.coord.galactic.b.deg
        
        label = 0
        for E, lon, lat in np.nditer([self.energy, lon, lat]):

            # shift up to top 4 colors in palette, using first for background
            color = cmap(norm_E(E) + 0.2) 

            # just label once
            if label == 0:
                skymap.tissot(lon, lat, self.coord_uncertainty, 30, facecolor = color, 
                            alpha = style.alpha_level, label = self.label)
                label = 1
            else:
                skymap.tissot(lon, lat, self.coord_uncertainty, 30, facecolor = color, alpha = style.alpha_level)

        # map background, parallels, meridians and labels
        skymap.drawmapboundary(fill_color = cmap(0))
        skymap.drawparallels(np.arange(-75, 76, 15), linewidth = 1, dashes = [1,2],
                             labels=[1, 0, 0, 0], textcolor = 'white', fontsize = 14, alpha = 0.7);
        # workaround for saving properly
        plt.gcf().subplots_adjust(left = 0.3)
        skymap.drawmeridians(np.arange(-150, 151, 30), linewidth = 1, dashes = [1,2]);
        lons = np.arange(-150, 151, 30)
        skymap.label_meridians(lons, fontsize = 14, vnudge = 1,
                             halign = 'left', hnudge = -1,
                             alpha = 0.7)  

        # colorbar
        cb_ax = plt.axes([0.35, 0, .5, .03], frameon = False)  
        vals = np.linspace(self.energy.min(), self.energy.max(), 100)
        bar = matplotlib.colorbar.ColorbarBase(cb_ax, values = vals, norm = norm_E, cmap = cmap, 
                                               orientation = 'horizontal', drawedges = False, alpha = 1)
        children = bar.ax.get_children()
        obj = children[1]
        obj.set_linewidth(0)
        bar.set_label('UHECR Energy [EeV]', alpha = 0.7)

        if save == True:
            plt.savefig('plots/uhecr_map.pdf', dpi = 1000, bbox_inches = 'tight', pad_inches = 0.5)
            
        return fig, skymap
