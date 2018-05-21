
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord

from ..plotting import AllSkyMap
from ..utils import PlotStyle, Solarized


__all__ = ['Data', 'Source', 'Uhecr', 'Detector']


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
        raw_data = RawData(filename)
        new_source = Source(raw_data)

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
        raw_data = RawData(filename)
        new_uhecr = Uhecr(raw_data)

        # generate numbered labels by default
        if label == None:
            label = 'uhecr#' + str(len(self.uhecr))
            
        # append source object to dictonary with it's label as a key
        self.uhecr[label] = new_uhecr


    def add_detector(self, coords, threshold_zenith_angle, label = None):
        """
        Add a detector object to complement the data.

        :param name: the name of the detector
        """

        new_detector = Detector(coords, threshold_zenith_angle)

        # generate numbered labels by default
        if label == None:
            label = 'detector#' + str(len(self.detector))
            
        # append source object to dictonary with it's label as a key
        self.detector[label] = new_detector


    def _uhecr_colorbar(self, style):
        """
        Add a colorbar normalised over all the Uhecr energies.

        :param style: an instance of PlotStyle
        """

        max_energies = []
        min_energies = []
        # find the min and max uhecr energies
        for label, uhecr in self.uhecr.items():
            max_energies.append(max(uhecr.energy))
            min_energies.append(min(uhecr.energy))
            
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

        # iterate over the uhecr objects
        if self.uhecr != {}:
            for label, uhecr in self.uhecr.items():
                uhecr.plot(style, label, skymap)

        # iterate over the source objects
        if self.source != {}:
            for label, source in self.source.items():
                source.plot(style, label, skymap)

        # iterate over the detector objects
        if self.detector != {}:
            for label, detector in self.detector.items():
                detector.draw_exposure_lim(skymap, label = label)
                
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
    Stores the data and parameters for sources
    """

    
    def __init__(self, raw_data):
        """
        Stores the data and parameters for sources.
        
        :param data: data passed as an instance of Data
        """

        self.N = raw_data.get_len()
        
        self.coord = raw_data.get_coordinates()

        self.distance = raw_data.get_by_name('D') # in Mpc

        self.name = raw_data.get_by_name('name')

        self.label = 'Source'

        
    def plot(self, style, label, skymap):
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
                              alpha = style.alpha_level, label = label)
                write_label = False
            else:
                skymap.tissot(lon, lat, 5., 30, 
                              facecolor = Solarized().base1, alpha = style.alpha_level)
            
   
        
class Uhecr():
    """
    Stores the data and parameters for UHECRs
    """

    
    def __init__(self, raw_data):
        """
        Stores the data and parameters for UHECRs.
        
        :param data: data passed as an instance of Data
        """

        self.N = raw_data.get_len()
        
        self.coord = raw_data.get_coordinates()

        self.year = raw_data.get_by_name('year')

        self.day = raw_data.get_by_name('day')

        self.incidence_angle = raw_data.get_by_name('incidence angle')

        self.energy = raw_data.get_by_name('energy')

        self.coord_uncertainty = 4.0 # uncertainty in degrees

        self.label = 'UHECR'


    def plot(self, style, label, skymap):
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
                skymap.tissot(lon, lat, self.coord_uncertainty, 30, facecolor = color,
                              alpha = style.alpha_level)


class Angle():
    """
    Store angles as degree or radian for convenience.
    """

    def __init__(self, angle, type = None):
        """
        Store angles as degree or radian for convenience.
       
        :param angle: a single angle or 
        """

        self._defined_types = ['deg', 'rad']

        # default: pass arguments in degrees
        if type == None:
            type = self._defined_types[0]

        if type == self._defined_types[0]:
            self.deg = angle
            if isinstance(angle, int) or isinstance(angle, float):
                self.rad = np.deg2rad(angle)
            else:
                self.rad = [np.deg2rad(a) for a in angle]
        elif type == self.defined_types[1]:
            if isinstance(angle, int) or isinstance(angle, float):
                self.deg = np.rad2deg(angle)
            else:
                self.deg = [np.rad2deg(a) for a in angle]
            self.rad = angle


            
class Detector():
    """
    UHECR observatory information and instrument response. 
    """

    def __init__(self, coords, threshold_zenith_angle):
        """
        UHECR observatory information and instrument response. 
        
        :param coords: latitiude and longitude of detector
                       in deg.
        :param threshold_zenith_angle: maximum detectable
                                       zenith angle in deg.
        """

        # stroed in radians
        self.coords = self._convert_coordinates(coords)
        self.threshold_zenith_angle = Angle(threshold_zenith_angle)

        self._view_options = ['map', 'decplot']
        self.exposure()

        
    def _convert_coordinates(self, coords):
        """
        Convert input coordinates into astropy coord
        objects for convenience.

        It is assumed that the input coordinates are 
        standard latitude and longitude of a position
        on the Earth's surface.
        
        :param coords: latitude and longitude of detector
        :return: astropy coord object
        """

        #return np.asarray([np.deg2rad(c) for c in coords])
        return SkyCoord(l = coords[1] * u.degree,
                        b = coords[0] * u.degree, frame = 'galactic')

        
    def exposure(self):
        """
        Calculate and plot the exposure for a given detector 
        locaiton.
        """

        # define a range of declination to evaluate the
        # exposure at
        num_points = 500
        declination = np.linspace(-90, 90, num_points)
        dec_rad = np.deg2rad(declination) # convert to radians

        proj_fac = np.asarray([self._alpha_m(dec) for dec in dec_rad])

        # normalise
        self._projection_factor = np.nan_to_num(proj_fac / max(proj_fac))

        # find the point at which the pjection factor is 0
        self.limiting_dec = Angle((declination[proj_fac == 0])[0], 'deg')
        

    def _avg_proj_factor(self, dec):
        """
        Define the average projection factor as a function
        of declination.

        :param dec: the values of declination for which the 
                    projection factor is evaluated.
        """

        a_0 = self.coords.galactic.b.rad
        return ( np.cos(a_0) * np.cos(dec) *
                np.sin(self._alpha_m(dec)) ) 

    
    def _alpha_m(self, dec):
        """
        Calculate a factor needed for the exposure calculation
        
        :param dec: an array of declination values for which
                    this factor is to be evaluated.
        """

        xi_val = self._xi(dec)
        if (xi_val > 1):
            res = 0
        elif (xi_val < -1):
            res = np.pi
        else:
            res = np.arccos(xi_val)
        return res
        

    def _xi(self, dec):
        """
        Calculate a factor needed for the exposure calculation
        
        :param dec: an array of declination values for which
                    this factor is to be evaluated.
        """

        theta_m = self.threshold_zenith_angle.rad
        a_0 = self.coords.galactic.b.rad
        
        numerator = np.cos(theta_m) - np.sin(a_0) * np.sin(dec)
        denominator = np.cos(a_0) * np.cos(dec)
        return numerator/denominator


    def show(self, view = None, save = False, savename = None, cmap = None):
        """
        Make a plot of the detector's exposure
        
        :param view: a keyword describing how to show the plot
                     options are described by self._view_options
        :param save: boolean input, if True, the figure is saved
        :param savename: location to save to, required if save is 
                         True
        """

        # define the style
        if cmap == None:
            style = PlotStyle(cmap_name = 'macplus')
        else:
            style = PlotStyle(cmap_name = cmap)
            
        # default is skymap
        if view == None:
            view = self._view_options[0]
        else:
            if view not in self._view_options:
                print ('ERROR:', 'view option', view, 'is not defined')
                return

        # sky map
        if view == self._view_options[0]:

            # figure
            fig = plt.figure(figsize = (12, 6))
            ax = plt.gca()
            
            # skymap
            skymap = AllSkyMap(projection = 'hammer', lon_0 = 0, lat_0 = 0)

            num_points = 500
            
            # define RA and DEC over all coordinates
            rightascensions = np.linspace(-180, 180, num_points)
            declinations = np.linspace(-90, 90, num_points)

            cmap = style.cmap
            norm_proj = matplotlib.colors.Normalize(self._projection_factor.min(),
                                                    self._projection_factor.max())

            # plot the exposure map
            # NB: use scatter as plot and pcolormesh have bugs in shiftdata methods
            for dec, proj in np.nditer([declinations, self._projection_factor]):
                decs = np.tile(dec, num_points)
                c = SkyCoord(ra = rightascensions * u.degree, 
                             dec = decs * u.degree, frame = 'icrs')
                lon = c.galactic.l.deg
                lat = c.galactic.b.deg
                skymap.scatter(lon, lat, latlon = True, linewidth = 3, 
                             color = cmap(norm_proj(proj)), alpha = 0.7)

            # plot exposure boundary
            self.draw_exposure_lim(skymap)
            
            # add labels
            skymap.draw_standard_labels(style.cmap, style.textcolor)

            # add colorbar
            self._exposure_colorbar(style)

        # decplot
        elif view == self._view_options[1]:

            # plot for all decs
            num_points = 500
            declination = np.linspace(-90, 90, num_points)
 
            plt.figure()
            plt.plot(declination, self._projection_factor, linewidth = 5, alpha = 0.7)
            plt.xlabel('$\delta$');
            plt.ylabel('projection factor');


        if save:
            plt.savefig(savename, dpi = 1000,
                    bbox_inches = 'tight', pad_inches = 0.5)

            
    def _exposure_colorbar(self, style):
        """
        Plot a colorbar for the exposure map
        """
            
        cb_ax = plt.axes([0.25, 0, .5, .03], frameon = False)  
        vals = np.linspace(self._projection_factor.min(),
                           self._projection_factor.max(), 100)
        
        norm_proj = matplotlib.colors.Normalize(self._projection_factor.min(),
                                                self._projection_factor.max())
    
        bar = matplotlib.colorbar.ColorbarBase(cb_ax, values = vals, norm = norm_proj, cmap = style.cmap, 
                                               orientation = 'horizontal', drawedges = False, alpha = 1)

        bar.ax.get_children()[1].set_linewidth(0)
        bar.set_label('Relative exposure', color = style.textcolor)
        

    def draw_exposure_lim(self, skymap, label = None):
        """
        Draw a line marking the edge of the detector's exposure.
        
        :param skymap: an AllSkyMap instance.
        :param label: a label for the limit.
        """

        if label == None:
            label = 'detector'

        num_points = 500
        rightascensions = np.linspace(-180, 180, num_points)  
        limiting_dec = self.limiting_dec.deg
        boundary_decs = np.tile(limiting_dec, num_points)
        c = SkyCoord(ra = rightascensions * u.degree,
                     dec = boundary_decs * u.degree, frame = 'icrs')
        lon = c.galactic.l.deg
        lat = c.galactic.b.deg

        skymap.scatter(lon, lat, latlon = True, linewidth = 3, 
                       color = 'k', alpha = 0.5,
                       label = 'limit of ' + label + '\'s exposure')
