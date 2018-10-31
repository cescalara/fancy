import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from datetime import date, timedelta
import h5py

from .stan import coord_to_uv
from ..detector.detector import Detector
from ..plotting import AllSkyMap
from ..utils import PlotStyle, Solarized

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
            label = 'AGN_VCV'
        
        new_source = Source(filename, label)
    
        # define source object
        self.source = new_source


    def add_uhecr(self, filename, label = None):
        """
        Add a uhecr object to the data cotainer

        :param filename: name of the file containing the object's data
        :param label: reference label for the uhecr dataset
        """

        if label == None:
            label = 'auger2010'  
    
        new_uhecr = Uhecr(filename, label)

        # define uhecr object
        self.uhecr = new_uhecr


    def add_detector(self, location, threshold_zenith_angle, area, total_exposure, kappa_c, label = None):
        """
        Add a detector object to complement the data.

        :param name: the name of the detector
        """

        if label == None:
            label = 'detector#' + str(len(self.detector))
            
        new_detector = Detector(location, threshold_zenith_angle, area, total_exposure, kappa_c, label)
        
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
        #if self.detector != {}:
        #    self.detector.draw_exposure_lim(skymap)
                
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
        if self.uhecr != {} and self.uhecr.N != 1:
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

    def __init__(self, filename, filelayout):
        """
        Parses information for known data files in txt format. 
        :filename: name of the file to parse
        :filelayout: list of column names in file
        """

        self._filename = filename
        self._filelayout = filelayout
        self._data = self._parse()

    
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

    
    def __init__(self, filename, label):
        """
        Stores the data and parameters for sources.
        
        :param filename: file ocntaining source data
        :param label: identifier
        """

        self.label = label
        
        with h5py.File(filename, 'r') as f:
            data = f[self.label]
            self.distance = data['D'].value
            self.N = len(self.distance)
            glon = data['glon'].value
            glat = data['glat'].value
            self.coord = get_coordinates(glon, glat)

            # get fluxes
            if self.label == 'SBG_23' or self.label == 'SBG_63':
                self.flux = data['L'].value
            elif self.label == '2FHL_250Mpc' or self.label == '3FHL_250Mpc_FA':
                self.flux = data['flux'].value
            elif self.label == 'swift_BAT_213':
                self.flux = data['F'].value
            else:
                self.flux = None

            if self.label != 'cosmo_150':
                # get names
                self.name = []
                for i in range(self.N):
                    self.name.append(data['name'][i])
            
        self.unit_vector = coord_to_uv(self.coord)
        try:
            self.flux_weight = [fl / max(self.flux) for fl in self.flux] 
        except:
            print('No flux weights calculated for sources.')
            
            
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
            
    def select_sources(self, selection):
        """
        Select sources by providing certain indices from a list.
        """

        # store selection
        self.selection = selection
        
        # make selection
        self.unit_vector = [self.unit_vector[i] for i in selection]
        self.distance = [self.distance[i] for i in selection]

        self.N = len(self.distance)

        self.coord = self.coord[selection]
        try:
            self.flux = self.flux[selection]
            self.flux_weight = self.flux_weight[selection]
        except:
            print('No fluxes to select on.')
            
    def select_distance(self, Dth):
        """
        Select sources with distance <= Dth. 
        Dth should be eneterd in [Mpc].
        """

        selection = [i for i, d in enumerate(self.distance) if d <= Dth]
        self.selection = selection
        
        self.unit_vector = [self.unit_vector[i] for i in selection]
        self.distance = [self.distance[i] for i in selection]

        self.N = len(self.distance)

        self.coord = self.coord[selection]
        try:
            self.flux = self.flux[selection]
            self.flux_weight = self.flux_weight[selection]
        except:
            print('No fluxes to select on.')
        
        
class Uhecr():
    """
    Stores the data and parameters for UHECRs
    """

    
    def __init__(self, filename, label):
        """
        Stores the data and parameters for UHECRs.
        
        :param filename: name of UHECR data file
        :param label: identifier
        """

        self.label = label

        with h5py.File(filename, 'r') as f:
            data = f[self.label]

            self.year = data['year'].value
            self.day = data['day'].value
            self.incidence_angle = data['theta'].value
            self.energy = data['energy'].value
            self.N = len(self.energy)
            glon = data['glon'].value
            glat = data['glat'].value
            self.coord = get_coordinates(glon, glat)
            
        self.coord_uncertainty = 4.0 # uncertainty in degrees
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

        # plot the UHECR location
        if self.N != 1:
            # use colormap for energy
            norm_E = matplotlib.colors.Normalize(min(self.energy), max(self.energy))
        cmap = style.cmap

        lon = self.coord.galactic.l.deg
        lat = self.coord.galactic.b.deg
        
        write_label = True
        for E, lon, lat in np.nditer([self.energy, lon, lat]):

            if self.N != 1:
                # shift up to top 4 colors in palette, using first for background
                color = cmap(norm_E(E) + 0.2)
            else:
                color = cmap(0.5)

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
        in Abreu et al. (2010) or Collaboration et al. 2014.
        """

        if self.label == 'auger2010':
            from ..detector.auger2010 import A1, A2, A3
            possible_areas = [A1, A2, A3]
            area = [possible_areas[i-1] for i in self.period]

        if self.label == 'auger2014':
            from ..detector.auger2014 import A1, A2, A3, A4, A1_incl, A2_incl, A3_incl, A4_incl
            possible_areas_vert = [A1, A2, A3, A4]
            possible_areas_incl = [A1_incl, A2_incl, A3_incl, A4_incl]

            # find area depending on period and incl
            area = []
            for i, p in enumerate(self.period):
                if self.incidence_angle[i] <= 60:
                    area.append(possible_areas_vert[p - 1])
                if self.incidence_angle[i] > 60:
                    area.append(possible_areas_incl[p - 1])

        return area

                
    def _find_period(self):
        """
        For a given year or day, find UHECR period based on dates
        in table 1 in Abreu et al. (2010) or in Collaboration et al. 2014.
        """

        from ..detector.auger2014 import (period_1_start, period_1_end, period_2_start, period_2_end,
                                          period_3_start, period_3_end, period_4_start, period_4_end)

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
            elif test_date >= period_3_end:
                period.append(4)
            else:
                print('Error: cannot determine period for year', y, 'and day', d)
        
        return period

    
    def select_period(self, period):
        """
        Select certain periods for analysis, other periods will be discarded. 
        """

        # find selected periods
        if len(period) == 1:
            selection = np.where(np.asarray(self.period) == period[0])
        if len(period) == 2:
            selection = np.concatenate([np.where(np.asarray(self.period) == period[0]),
                                        np.where(np.asarray(self.period) == period[1])], axis = 1)

        # keep things as lists
        selection = selection[0].tolist()
            
        # make selection
        self.A = [self.A[i] for i in selection]
        self.period = [self.period[i] for i in selection]
        self.energy = [self.energy[i] for i in selection]
        self.incidence_angle = [self.incidence_angle[i] for i in selection]
        self.unit_vector = [self.unit_vector[i] for i in selection] 

        self.N = len(self.period)

        self.day = [self.day[i] for i in selection]
        self.year = [self.year[i] for i in selection]

        self.coord = self.coord[selection]

    def select_energy(self, Eth):
        """
        Select out only UHECRs above a certain energy.
        """

        selection = np.where(np.asarray(self.energy) >= Eth)
        selection = selection[0].tolist()

        # make selection
        self.A = [self.A[i] for i in selection]
        self.period = [self.period[i] for i in selection]
        self.energy = [self.energy[i] for i in selection]
        self.incidence_angle = [self.incidence_angle[i] for i in selection]
        self.unit_vector = [self.unit_vector[i] for i in selection] 

        self.N = len(self.period)

        self.day = [self.day[i] for i in selection]
        self.year = [self.year[i] for i in selection]

        self.coord = self.coord[selection]


# convenience functions
    
def get_coordinates(glon, glat, D = None):
    """
    Convert glon and glat to astropy SkyCoord
    Add distance if possible (allows conversion to cartesian coords)
        
    :return: astropy.coordinates.SkyCoord
    """

    if D:
        return SkyCoord(l = glon * u.degree, b = glat * u.degree,
                        frame = 'galactic', distance = D * u.mpc)
    else:
        return SkyCoord(l = glon * u.degree, b = glat * u.degree, frame = 'galactic')
    
