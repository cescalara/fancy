import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from datetime import date, timedelta
import h5py

from .stan import coord_to_uv, uv_to_coord
from ..detector.detector import Detector
from ..plotting import AllSkyMap

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

        # uhecr, source and detector objects
        self.uhecr = None
        self.source = None       
        self.detector = None

        
    def add_source(self, filename, label = None):
        """
        Add a source object to the data cotainer from file.

        :param filename: name of the file containing the object's data
        :param label: reference label for the source object
        """

        if label == None:
            label = 'VCV_AGN'
        
        new_source = Source()
        new_source.from_data_file(filename, label)
    
        # define source object
        self.source = new_source


    def add_uhecr(self, filename, label = None):
        """
        Add a uhecr object to the data container from file.

        :param filename: name of the file containing the object's data
        :param label: reference label for the uhecr dataset
        """

        new_uhecr = Uhecr()
        new_uhecr.from_data_file(filename, label)

        # define uhecr object
        self.uhecr = new_uhecr


    def add_detector(self, detector_properties):
        """
        Add a detector object to complement the data.

        :param detector_properties: dict of properties.
        """
            
        new_detector = Detector(detector_properties)
        
        # define detector
        self.detector = new_detector


    def _uhecr_colorbar(self, cmap):
        """
        Add a colorbar normalised over all the Uhecr energies.

        :param cmap: matplotlib colorbar object
        """

        max_energies = []
        min_energies = []
        # find the min and max uhecr energies
        max_energies.append(max(self.uhecr.energy))
        min_energies.append(min(self.uhecr.energy))
            
        max_energy = max(max_energies)
        min_energy = min(min_energies)
            
        norm_E = matplotlib.colors.Normalize(min_energy, max_energy)

        # colorbar
        cb_ax = plt.axes([0.25, 0, .5, .03], frameon = False)  
        vals = np.linspace(min_energy, max_energy, 100)
        bar = matplotlib.colorbar.ColorbarBase(cb_ax, values = vals, norm = norm_E, cmap = cmap, 
                                               orientation = 'horizontal', drawedges = False, alpha = 1)
        #bar.ax.get_children()[1].set_linewidth(0)
        bar.set_label('UHECR Energy [EeV]')

        
    def show(self, save = False, savename = None, cmap = None):
        """
        Plot the data on a map of the sky. 
        
        :param save: boolean input, saves figure if True
        :param savename: location to save to, required if 
                         save == True
        """

        # plot style
        if cmap == None:
            cmap = plt.cm.get_cmap('viridis')
            
        # figure
        fig, ax = plt.subplots();
        fig.set_size_inches((12, 6))

        # skymap
        skymap = AllSkyMap(projection = 'hammer', lon_0 = 0, lat_0 = 0);

        # uhecr object
        if self.uhecr:
            self.uhecr.plot(skymap)

        # source object
        if self.source:
            self.source.plot(skymap)

        # detector object
        #if self.detector:
        #    self.detector.draw_exposure_lim(skymap)
                
        # standard labels and background
        skymap.draw_standard_labels()
    
        # legend
        ax.legend(frameon = False, bbox_to_anchor=(0.85, 0.85))
        
        # add a colorbar if uhecr objects plotted
        if self.uhecr and self.uhecr.N != 1:
            self._uhecr_colorbar(cmap)

        if save:
            plt.savefig(savename, dpi = 1000,
                    bbox_extra_artists = [leg],
                    bbox_inches = 'tight', pad_inches = 0.5)
        
        return fig, skymap


    def from_file(self, filename):
        """
        Load data from an Analysis output file.
        :param filename: file name
        """

        # Read out information on data and detector
        uhecr_properties = {}
        source_properties = {}
        detector_properties = {}
        with h5py.File(filename, 'r') as f:

            uhecr = f['uhecr']

            for key in uhecr:
                uhecr_properties[key] = uhecr[key].value

            source = f['source']

            for key in source:
                source_properties[key] = source[key].value

            detector = f['detector']

            for key in detector:
                detector_properties[key] = detector[key].value

        uhecr = Uhecr()
        uhecr.from_properties(uhecr_properties)

        source = Source()
        source.from_properties(source_properties)

        detector = Detector(detector_properties)
        
        # Add to data object
        self.uhecr = uhecr
        self.source = source
        self.detector = detector
        
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

    
    def __init__(self):
        """
        Initialise empty container.
        """

        
    def from_data_file(self, filename, label):
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
            
            if self.label != 'cosmo_150' and self.label != 'VCV_AGN':
                # Get names
                self.name = []
                for i in range(self.N):
                    self.name.append(data['name'][i])
                    
        self.unit_vector = coord_to_uv(self.coord)

    def _get_properties(self):
        """
        Convenience function to pack object into dict.
        """

        self.properties = {}
        self.properties['label'] = self.label
        self.properties['N'] = self.N
        self.properties['unit_vector'] = self.unit_vector
        self.properties['distance'] = self.distance
        
        
    def from_properties(self, source_properties):
        """
        Define sources from properties dict.
            
        :param source_properties: dict containing source properties.
        :param label: identifier
        """

        self.label = source_properties['label']

        self.N = source_properties['N']
        self.unit_vector = source_properties['unit_vector']
        self.distance = source_properties['distance']

        self.coord = uv_to_coord(self.unit_vector)
    
        
    def plot(self, skymap, size = 2.0, color = 'k'):
        """
        Plot the sources on a map of the sky. 

        Called by Data.show()

        :param skymap: the AllSkyMap
        :param size: radius of tissots
        :param color: colour of tissots
        """
        
        alpha_level = 0.9
        
        # plot the source locations
        write_label = True
        for lon, lat in np.nditer([self.coord.galactic.l.deg, self.coord.galactic.b.deg]):
            if write_label:
                skymap.tissot(lon, lat, size, 30, 
                              facecolor = 'k', 
                              alpha = alpha_level, label = self.label)
                write_label = False
            else:
                skymap.tissot(lon, lat, size, 30, 
                              facecolor = 'k', alpha = alpha_level)

    def save(self, file_handle):
        """
        Save to the passed H5py file handle,
        i.e. something that cna be used with 
        file_handle.create_dataset()
        
        :param file_handle: file handle
        """
        
        self._get_properties()

        for key, value in self.properties.items():
                file_handle.create_dataset(key, data = value)
        
                
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

    
    def __init__(self):
        """
        Initialise empty container.
        """

        self.properties = None
        self.source_labels = None

    def from_data_file(self, filename, label):
        """
        Define UHECR from data file of original information.
        
        Handles calculation of observation periods and 
        effective areas assuming the UHECR are detected 
        by the Pierre Auger Observatory.
        
        :param filename: name of the data file
        :param label: reference label for the UHECR data set 
        """

        self.label = label
        
        with h5py.File(filename, 'r') as f:

            data = f[self.label]
            
            self.year = data['year'].value
            self.day = data['day'].value
            self.zenith_angle = data['theta'].value
            self.energy = data['energy'].value
            self.N = len(self.energy)
            glon = data['glon'].value
            glat = data['glat'].value
            self.coord = get_coordinates(glon, glat)
            
            self.unit_vector = coord_to_uv(self.coord)
            self.period = self._find_period()
            self.A = self._find_area()

    def _get_properties(self):
        """
        Convenience function to pack object into dict.
        """
        
        self.properties = {}
        self.properties['label'] = self.label
        self.properties['N'] = self.N
        self.properties['unit_vector'] = self.unit_vector
        self.properties['energy'] = self.energy
        self.properties['A'] = self.A
        self.properties['zenith_angle'] = self.zenith_angle 

        # Only if simulated UHECRs
        if isinstance(self.source_labels, (list, np.ndarray)):
            self.properties['source_labels'] = self.source_labels
    
    def from_properties(self, uhecr_properties):
        """
        Define UHECR from properties dict.
            
        :param uhecr_properties: dict containing UHECR properties.
        :param label: identifier
        """

        self.label = uhecr_properties['label']

        # Read from input dict
        self.N = uhecr_properties['N']
        self.unit_vector = uhecr_properties['unit_vector']
        self.energy = uhecr_properties['energy']
        self.zenith_angle = uhecr_properties['zenith_angle']
        self.A = uhecr_properties['A']

        # Only if simulated UHECRs
        try:
            self.source_labels = uhecr_properties['source_labels']
        except:
            pass
        
        # Get SkyCoord from unit_vector
        self.coord = uv_to_coord(self.unit_vector)

        
    def plot(self, skymap, size = 2):
        """
        Plot the Uhecr instance on a skymap.

        Called by Data.show()
      
        :param skymap: the AllSkyMap
        :param size: tissot radius
        :param source_labels: source labels (int)
        """

        lons = self.coord.galactic.l.deg
        lats = self.coord.galactic.b.deg

        alpha_level = 0.7
        
        # If source labels are provided, plot with colour
        # indicating the source label.
        if isinstance(self.source_labels, (list, np.ndarray)):

            Nc = max(self.source_labels)

            # Use a continuous cmap
            cmap = plt.cm.get_cmap('plasma', Nc) 

            write_label = True
            
            for lon, lat, lab in np.nditer([lons, lats, self.source_labels]):
                color = cmap(lab)
                if write_label:
                    skymap.tissot(lon, lat, size, npts = 30, facecolor = color,
                                  alpha = 0.5, label = self.label)
                    write_label = False
                else:
                    skymap.tissot(lon, lat, size, npts = 30, facecolor = color, alpha = 0.5)

        # Otherwise, use the cmap to show the UHECR energy. 
        else:
            
            # use colormap for energy
            norm_E = matplotlib.colors.Normalize(min(self.energy), max(self.energy))
            cmap = plt.cm.get_cmap('viridis', len(self.energy))
        
            write_label = True
            for E, lon, lat in np.nditer([self.energy, lons, lats]):

                color = cmap(norm_E(E))
                    
                if write_label:
                    skymap.tissot(lon, lat, size, 30, facecolor = color, 
                                  alpha = alpha_level, label = self.label)
                    write_label = False
                else:
                    skymap.tissot(lon, lat, size, 30, facecolor = color,
                                  alpha = alpha_level)

            

                
    def save(self, file_handle):
        """
        Save to the passed H5py file handle,
        i.e. something that cna be used with 
        file_handle.create_dataset()
        
        :param file_handle: file handle
        """
        
        self._get_properties()

        for key, value in self.properties.items(): 
                file_handle.create_dataset(key, data = value)
            
                
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
                if self.zenith_angle[i] <= 60:
                    area.append(possible_areas_vert[p - 1])
                if self.zenith_angle[i] > 60:
                    area.append(possible_areas_incl[p - 1])

        else:
            print('Error: effective areas and periods not defined')
            
        return area

                
    def _find_period(self, year, day):
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
    
