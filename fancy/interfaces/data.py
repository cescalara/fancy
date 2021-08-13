import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from datetime import date, timedelta
import h5py

from .uhecr import Uhecr
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

    def add_source(self, filename, label=None):
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

    def add_uhecr(self, filename, label=None):
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
        cb_ax = plt.axes([0.25, 0, .5, .03], frameon=False)
        vals = np.linspace(min_energy, max_energy, 100)
        bar = matplotlib.colorbar.ColorbarBase(cb_ax,
                                               values=vals,
                                               norm=norm_E,
                                               cmap=cmap,
                                               orientation='horizontal',
                                               drawedges=False,
                                               alpha=1)
        #bar.ax.get_children()[1].set_linewidth(0)
        bar.set_label('UHECR Energy [EeV]')

    def show(self, save=False, savename=None, cmap=None):
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
        fig, ax = plt.subplots()
        fig.set_size_inches((12, 6))

        # skymap
        skymap = AllSkyMap(projection='hammer', lon_0=0, lat_0=0)

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
        ax.legend(frameon=False, bbox_to_anchor=(0.85, 0.85))

        # add a colorbar if uhecr objects plotted
        if self.uhecr and self.uhecr.N != 1:
            self._uhecr_colorbar(cmap)

        if save:
            plt.savefig(savename,
                        dpi=1000,
                        bbox_extra_artists=[leg],
                        bbox_inches='tight',
                        pad_inches=0.5)

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
                uhecr_properties[key] = uhecr[key][()]

            source = f['source']

            for key in source:
                source_properties[key] = source[key][()]

            detector = f['detector']

            for key in detector:
                detector_properties[key] = detector[key][()]

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

        output = pd.read_csv(self._filename,
                             comment='#',
                             delim_whitespace=True,
                             names=self._filelayout)

        output_dict = output.to_dict()

        return output_dict

    def get_by_name(self, name):
        """
        Get data entries by name.

        :param name: name of the data as in self._filelayout
        :return: an array of data entries
        """

        try:
            selected_data = np.array(list(self._data[name].values()))

        except ValueError:
            print('No data of type', name)
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
            self.distance = data['D'][()]
            self.N = len(self.distance)
            glon = data['glon'][()]
            glat = data['glat'][()]
            self.coord = self.get_coordinates(glon, glat)

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

    def plot(self, skymap, size=2.0, color='k'):
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
        for lon, lat in np.nditer(
            [self.coord.galactic.l.deg, self.coord.galactic.b.deg]):
            if write_label:
                skymap.tissot(lon,
                              lat,
                              size,
                              30,
                              facecolor='k',
                              alpha=alpha_level,
                              label=self.label)
                write_label = False
            else:
                skymap.tissot(lon,
                              lat,
                              size,
                              30,
                              facecolor='k',
                              alpha=alpha_level)

    def save(self, file_handle):
        """
        Save to the passed H5py file handle,
        i.e. something that cna be used with 
        file_handle.create_dataset()
        
        :param file_handle: file handle
        """

        self._get_properties()

        for key, value in self.properties.items():
            file_handle.create_dataset(key, data=value)

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
            pass

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


# convenience functions


    def get_coordinates(self, glon, glat, D=None):
        """
        Convert glon and glat to astropy SkyCoord
        Add distance if possible (allows conversion to cartesian coords)
            
        :return: astropy.coordinates.SkyCoord
        """

        if D:
            return SkyCoord(l=glon * u.degree,
                            b=glat * u.degree,
                            frame='galactic',
                            distance=D * u.mpc)
        else:
            return SkyCoord(l=glon * u.degree, b=glat * u.degree, frame='galactic')
