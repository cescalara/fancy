import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord

__all__ = ['Data', 'Source', 'Uhecr']

class Data():
    """
    A container for storage of data.

    Parses information for known data files in txt format. 
    """

    def __init__(self):
        """
        A container for storage of data.
        
        Parses information for known data files in txt format. 
        """

        self._filename = None
        self._data = None

        # uhecr and source objects are stored in a
        # dictionary with keys equal to their labels
        self.uhecr = {}
        self.source = {}        

        
    def _define_type(self):
        """
        Determine the file type for parsing.

        :return: array of names of cols in file
        """
        if 'UHECR' in self._filename:
            self._filetype = 'uhecr'
            self._filelayout = ['year', 'day', 'incidence angle',
                      'energy', 'ra', 'dec',
                      'glon', 'glat']

        elif 'agn' in self._filename:
            self._filetype = 'agn'
            self._filelayout = ['name', 'glon', 'glat', 'D']
        
        else:
            print ('File layout not recognised')
            self._filetype = None 
            self._filelayout = None


    
    def _parse(self):
        """
        Parse the data form the object's file.
        
        :return: arrays for each column in the data file
        """

        output = pd.read_csv(self._filename, comment = '#',
                             delim_whitespace = True,
                             names = self._filelayout)

        output_dict = output.to_dict()
        
        return output_dict


    def get_len(self):
        """
        Get the length of the data set.

        :return: the length of the data set
        """

        if self._filetype == 'uhecr':
            n = len(self.data['year'])

        elif self._filetype == 'agn':
            n = len(self.data['name'])

        return n

    
    def get_coordinates(self):
        """
        Get the galactic coordinates from self.data
        and return them as astropy SkyCoord
        
        :return: astropy.coordinates.SkyCoord
        """

        glon = np.array( list(self.data['glon'].values()) )
        glat = np.array( list(self.data['glat'].values()) )

        return SkyCoord(l = glon * u.degree, b = glat * u.degree, frame = 'galactic')

    
    
    def get_by_name(self, name):
        """
        Get data entries by name.

        :param name: name of the data as in self._filelayout
        :return: an array of data entries
        """

        try:
            selected_data = np.array( list(self.data[name].values()) )

        except ValueError:
            print ('No data of type', name)
            selected_data = []

        return selected_data


    def add_source(self, filename, label = None):
        """
        Add a source object to the data cotainer

        :param filename: name of the file containing the object's data
        :param label: reference label for the source object
        """
        self._filename = filename
        self._define_type()
        self._data = self._parse()

        new_source = Source(self._data)

        # generate numbered labels by default
        if label == None:
            label = 'source#' + str(len(self.source))
            
        # append source object to dictonary with it's label as a key
        self.source[label] = new_source


    def add_uhecr(self, filename, label = None):
        """
        Add a uhecr object to the data cotainer

        :param filename: name of the file containing the object's data
        :param label: reference label for the source object
        """
        self._filename = filename
        self._define_type()
        self._data = self._parse()

        new_uhecr = Uhecr(self._data)

        # generate numbered labels by default
        if label == None:
            label = 'uhecr#' + str(len(self.uhecr))
            
        # append source object to dictonary with it's label as a key
        self.uhecr[label] = new_uhecr


    def add_detector(self, name = None):
        """
        Add a detector object to complement the data.

        :param name: the name of the detector
        """
        return 0


    def _add_skymap_labels(self):
        """
        Add the standard labels for parallels and meridians to the skymap
        """

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


    def _add_uhecr_colorbar(self):
        """
        Add a colorbar normalised over all the Uhecr energies.
        
        TODO: check all energies, instead of using the first
        """

        # fix this
        norm_E = matplotlib.colors.Normalize(self.uhecr[0].energy.min(), self.uhecr[0].max())
        cmap = style.cmap

        # colorbar
        cb_ax = plt.axes([0.35, 0, .5, .03], frameon = False)  
        vals = np.linspace(self.energy.min(), self.energy.max(), 100)
        bar = matplotlib.colorbar.ColorbarBase(cb_ax, values = vals, norm = norm_E, cmap = cmap, 
                                               orientation = 'horizontal', drawedges = False, alpha = 1)
        children = bar.ax.get_children()
        obj = children[1]
        obj.set_linewidth(0)
        bar.set_label('UHECR Energy [EeV]', alpha = 0.7)


        
    def show(self):
        """
        Plot the data on a map of the sky. 

        :return: the figure and AllSkyMap instance for further editing.
        """

        # plot style
        style = PlotStyle()

        # figure
        if fig == None:
            fig = plt.figure(figsize = (12, 6));
        ax = plt.gca()

        # skymap
        if skymap == None:
            skymap = AllSkyMap(projection = 'hammer', lon_0 = 0, lat_0 = 0);

        # iterate over the uhecr objects
        if self.uhecr != {}:
            for label, uhecr in d.items():
                uhecr.plot(style, label)

        # add a colorbar
        #self._add_uhecr_colorbar()
        
        # iterate over the source objects
        if self.source != {}:
            for label, source in d.items():
                source.plot(style, label)

        # iterate over the detector objects
        # add this
        
        # standard labels and background
        self._add_skymap_labels()
        
        # legend
        ax = fig.gca()
        plt.legend(bbox_to_anchor=(0.85, 0.85))
        leg = ax.get_legend()
        leg.legendHandles[0].set_color(style.cmap(0.4))
        for text in leg.get_texts():
            plt.setp(text, color = style.textcolor, alpha = 0.7)
        
        
        
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

        
    def plot(self, style, label):
        """
        Plot the sources on a map of the sky. 

        Called by Data.show()

        :param style: the PlotStyle instance
        :param label: the object's label
        """
    
        # plot the source locations
        write_label = True
        for lon, lat in np.nditer([self.coord.galactic.l.deg, self.coord.galactic.b.deg]):
            if write_label:
                skymap.tissot(lon, lat, 5., 30, 
                              facecolor = style.textcolor, 
                              alpha = style.alpha_level, label = label)
                write_label = False
            else:
                skymap.tissot(lon, lat, 5., 30, 
                              facecolor = style.textcolor, alpha = style.alpha_level)
            
   
        
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


    def plot(self, style, label):
        """
        Plot the Uhecr instance on a skymap.

        Called by Data.show()
      
        :param style: the PlotStyle instance
        :param label: the object's label
        """

        # plot the UHECR locations
        # use colormap for energy
        norm_E = matplotlib.colors.Normalize(self.energy.min(), self.energy.max())
        cmap = style.cmap

        lon = self.coord.galactic.l.deg
        lat = self.coord.galactic.b.deg
        
        write_label = True
        for E, lon, lat in np.nditer([self.energy, lon, lat]):

            # shift up to top 4 colors in palette, using first for background
            color = cmap(norm_E(E) + 0.2) 

            # just label once
            if write_label:
                skymap.tissot(lon, lat, self.coord_uncertainty, 30, facecolor = color, 
                            alpha = style.alpha_level, label = label)
                write_label = False
            else:
                skymap.tissot(lon, lat, self.coord_uncertainty, 30, facecolor = color, alpha = style.alpha_level)


