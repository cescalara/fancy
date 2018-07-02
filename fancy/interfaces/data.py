import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from datetime import date, timedelta

from .stan import coord_to_uv
from ..detector.detector import Detector
from ..plotting import AllSkyMap
from ..utils import PlotStyle, Solarized
from ..detector.auger import *

__all__ = ['Data', 'Source', 'Uhecr']


class Data():
    """
    A container for high level storage of data.
    """

    def __init__(self):
        """
        A container for high level storage of data.      
        """

        self._filename = None
        self._data = None

        # uhecr, source and detector objects are stored in a
        # dictionary with keys equal to their labels
        self.uhecr = {}
        self.source = {}        
        self.detector = {}

        
    def add_source(self, filename, label = None):
        """
        Add a source object to the data cotainer

        :param filename: name of the file containing the object's data
        :param label: reference label for the source object
        """

        if label == None:
            label = 'source'
        
        raw_data = RawData(filename)
        new_source = Source(raw_data, label)
    
        # define source object
        self.source = new_source


    def add_uhecr(self, filename, label = None):
        """
        Add a uhecr object to the data cotainer

        :param filename: name of the file containing the object's data
        :param label: reference label for the source object
        """

        if label == None:
            label = 'uhecr'  
    
        raw_data = RawData(filename)
        new_uhecr = Uhecr(raw_data, label)

        # define uhecr object
        self.uhecr = new_uhecr


    def add_detector(self, location, threshold_zenith_angle, area, total_exposure, label = None):
        """
        Add a detector object to complement the data.

        :param name: the name of the detector
        """

        if label == None:
            label = 'detector#' + str(len(self.detector))
            
        new_detector = Detector(location, threshold_zenith_angle, area, total_exposure, label)
        
        # define detector
        self.detector = new_detector


    def _uhecr_colorbar(self, style):
        """
        Add a colorbar normalised over all the Uhecr energies.

        :param style: an instance of PlotStyle
        """

        max_energies = []
        min_energies = []
        # find the min and max uhecr energies
        max_energies.append(max(self.uhecr.energy))
        min_energies.append(min(self.uhecr.energy))
            
        max_energy = max(max_energies)
        min_energy = min(min_energies)

        norm_E = matplotlib.colors.Normalize(min_energy, max_energy)
        cmap = style.cmap

        # colorbar
        cb_ax = plt.axes([0.25, 0, .5, .03], frameon = False)  
        vals = np.linspace(min_energy, max_energy, 100)
        bar = matplotlib.colorbar.ColorbarBase(cb_ax, values = vals, norm = norm_E, cmap = cmap, 
                                               orientation = 'horizontal', drawedges = False, alpha = 1)
        bar.ax.get_children()[1].set_linewidth(0)
        bar.set_label('UHECR Energy [EeV]', color = style.textcolor)

        
    def show(self, save = False, savename = None, cmap = None):
        """
        Plot the data on a map of the sky. 
        
        :param save: boolean input, saves figure if True
        :param savename: location to save to, required if 
                         save == True
        """

        # plot style
        if cmap == None:
            style = PlotStyle()
        else:
            style = PlotStyle(cmap_name = cmap)
            
        # figure
        fig = plt.figure(figsize = (12, 6));
        ax = plt.gca()

        # skymap
        skymap = AllSkyMap(projection = 'hammer', lon_0 = 0, lat_0 = 0);

        # uhecr objects
        if self.uhecr != {}:
                self.uhecr.plot(style, skymap)

        # source objects
        if self.source != {}:
            self.source.plot(style, skymap)

        # detector objects
        if self.detector != {}:
            self.detector.draw_exposure_lim(skymap)
                
        # standard labels and background
        skymap.draw_standard_labels(style.cmap, style.textcolor)
    
        # legend
        plt.legend(bbox_to_anchor=(0.85, 0.85))
        leg = ax.get_legend()
        frame = leg.get_frame()
        frame.set_linewidth(0)
        frame.set_facecolor('None')
        for text in leg.get_texts():
            plt.setp(text, color = style.textcolor)
        
        # add a colorbar if uhecr objects plotted
        if self.uhecr != {}:
            self._uhecr_colorbar(style)

        if save:
            plt.savefig(savename, dpi = 1000,
                    bbox_extra_artists = [leg],
                    bbox_inches = 'tight', pad_inches = 0.5)
        
        return fig, skymap

    
            
class RawData():
    """
    Parses information for known data files in txt format. 
    """

    def __init__(self, filename):
        """
        Parses information for known data files in txt format. 
        """

        self._filename = filename
        self._define_type()
        self._data = self._parse()
        

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
            n = len(self._data['year'])

        elif self._filetype == 'agn':
            n = len(self._data['name'])

        return n

    
    def get_coordinates(self):
        """
        Get the galactic coordinates from self.data
        and return them as astropy SkyCoord
        
        Add distance if possible (allows conversion to cartesian coords)
        
        :return: astropy.coordinates.SkyCoord
        """

        glon = np.array( list(self._data['glon'].values()) )
        glat = np.array( list(self._data['glat'].values()) )

        try:
            dist = np.array( list(self._data['D'].values()) )
        except:
            return SkyCoord(l = glon * u.degree, b = glat * u.degree, frame = 'galactic')
        else:
            return SkyCoord(l = glon * u.degree, b = glat * u.degree,
                            frame = 'galactic', distance = dist * u.mpc)

    
    
    def get_by_name(self, name):
        """
        Get data entries by name.

        :param name: name of the data as in self._filelayout
        :return: an array of data entries
        """

        try:
            selected_data = np.array( list(self._data[name].values()) )

        except ValueError:
            print ('No data of type', name)
            selected_data = []

        return selected_data
            
        
class Source():
    """
    Stores the data and parameters for sources.
    """

    
    def __init__(self, raw_data, label):
        """
        Stores the data and parameters for sources.
        
        :param raw_data: data passed as an instance of RawData
        :param label: identifier
        """

        self.label = label
        
        self.N = raw_data.get_len()
        
        self.coord = raw_data.get_coordinates()

        self.distance = raw_data.get_by_name('D') # in Mpc

        self.name = raw_data.get_by_name('name')

        self.label = 'Source'

        self.unit_vector = coord_to_uv(self.coord)

        
    def plot(self, style, skymap):
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
                              facecolor = Solarized().base1, 
                              alpha = style.alpha_level, label = self.label)
                write_label = False
            else:
                skymap.tissot(lon, lat, 5., 30, 
                              facecolor = Solarized().base1, alpha = style.alpha_level)
            

        
class Uhecr():
    """
    Stores the data and parameters for UHECRs
    """

    
    def __init__(self, raw_data, label):
        """
        Stores the data and parameters for UHECRs.
        
        :param data: data passed as an instance of Data
        :param label: identifier
        """

        self.label = label
        
        self.N = raw_data.get_len()
        
        self.coord = raw_data.get_coordinates()

        self.year = raw_data.get_by_name('year')

        self.day = raw_data.get_by_name('day')

        self.incidence_angle = raw_data.get_by_name('incidence angle')

        self.energy = raw_data.get_by_name('energy')

        self.coord_uncertainty = 4.0 # uncertainty in degrees

        self.label = 'UHECR'

        self.unit_vector = coord_to_uv(self.coord)

        self.period = self._find_period()

        self.A = self._find_area()
        

        
    def plot(self, style, skymap):
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
                            alpha = style.alpha_level, label = self.label)
                write_label = False
            else:
                skymap.tissot(lon, lat, self.coord_uncertainty, 30, facecolor = color,
                              alpha = style.alpha_level)

                
    def _find_area(self):
        """
        Find the effective area of the observatory at 
        the time of detection.

        Possible areas are calculated from the exposure reported
        in Abreu et al. (2010).

        :param period: list of periods defining the time of detection
        """
    
        possible_areas = [A1, A2, A3]
        
        area = [possible_areas[i-1] for i in self.period]

        return area

                
    def _find_period(self):
        """
        For a given year or day, find UHECR period based on dates
        in table 1 in Abreu et al. (2010).
        
        :param year: a list of years 
        :param day: a list of julian days
        """

        # check dates
        period = []
        for y, d in np.nditer([self.year, self.day]):
            d = int(d)
            test_date = date(y, 1, 1) + timedelta(d)

            if period_1_start <= test_date <= period_1_end:
                period.append(1)
            elif period_2_start <= test_date <= period_2_end:
                period.append(2)
            elif period_3_start <= test_date <= period_3_end:
                period.append(3)
            else:
                print('Error: cannot determine period for year', year, 'and day', day)
        
        return period
