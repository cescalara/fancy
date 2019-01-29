import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from scipy import integrate

from .exposure import *
from ..plotting import AllSkyMap
from ..utils import PlotStyle, Solarized

__all__ = ['Detector', 'Angle']

class Detector():
    """
    UHECR observatory information and instrument response. 
    """

    def __init__(self, detector_properties):
        """
        UHECR observatory information and instrument response. 
        
        :param detector_properties: dict of properties.
        """

        self.properties = detector_properties
        
        self.label = detector_properties['label']

        lat = detector_properties['lat'] # radians
        lon = detector_properties['lon'] # radians
        height = detector_properties['height'] # metres
        
        self.location = EarthLocation(lat = lat * u.rad, lon = lon * u.rad,
                               height = height * u.m)
        
        self.threshold_zenith_angle = Angle(detector_properties['theta_m'], 'rad') # radians

        self._view_options = ['map', 'decplot']

        # See Equation 9 in Capel & Mortlock (2019)
        self.kappa_c = detector_properties['kappa_c']
        self.coord_uncertainty = np.sqrt(7552.0 / self.kappa_c)

        self.energy_uncertainty = detector_properties['f_E']
        
        self.num_points = 500

        self.params = [np.cos(self.location.lat.rad),
                       np.sin(self.location.lat.rad),
                       np.cos(self.threshold_zenith_angle.rad)]

        self.exposure()

        self.area = detector_properties['A'] # km^2

        self.alpha_T = detector_properties['alpha_T'] # km^2 sr yr
        
        self.M, err = integrate.quad(m_integrand, 0, np.pi, args = self.params)
        
        self.params.append(self.alpha_T)
        self.params.append(self.M)
        
        
    def exposure(self):
        """
        Calculate and plot the exposure for a given detector 
        location.
        """

        # define a range of declination to evaluate the
        # exposure at
        self.declination = np.linspace(-np.pi/2, np.pi/2, self.num_points)

        m = np.asarray([m_dec(d, self.params) for d in self.declination])
        
        # normalise to a maximum at 1
        self.exposure_factor = (m / m_dec(-np.pi/2, self.params))

        # find the point at which the exposure factor is 0
        self.limiting_dec = Angle((self.declination[m == 0])[0], 'rad')
        

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
            cmap = plt.cm.get_cmap('viridis')
            
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
            fig, ax = plt.subplots()
            fig.set_size_inches((12, 6))
    
            # skymap
            skymap = AllSkyMap(projection = 'hammer', lon_0 = 0, lat_0 = 0)

            
            # define RA and DEC over all coordinates
            rightascensions = np.linspace(-np.pi, np.pi, self.num_points)
            declinations = self.declination
            
            norm_proj = matplotlib.colors.Normalize(self.exposure_factor.min(),
                                                    self.exposure_factor.max())

            # plot the exposure map
            # NB: use scatter as plot and pcolormesh have bugs in shiftdata methods
            for dec, proj in np.nditer([declinations, self.exposure_factor]):
                decs = np.tile(dec, self.num_points)
                c = SkyCoord(ra = rightascensions * u.rad, 
                             dec = decs * u.rad, frame = 'icrs')
                lon = c.galactic.l.deg
                lat = c.galactic.b.deg
                skymap.scatter(lon, lat, latlon = True, linewidth = 3, 
                             color = cmap(norm_proj(proj)), alpha = 0.7)

            # plot exposure boundary
            self.draw_exposure_lim(skymap)
            
            # add labels
            skymap.draw_standard_labels()

            # add colorbar
            self._exposure_colorbar(style)

        # decplot
        elif view == self._view_options[1]:

            # plot for all decs
 
            fig, ax = plt.subplots()
            ax.plot(self.declination, self.exposure_factor, linewidth = 5, alpha = 0.7)
            ax.set_xlabel('$\delta$');
            ax.set_ylabel('m($\delta$)');


        if save:
            fig.savefig(savename, dpi = 1000,
                    bbox_inches = 'tight', pad_inches = 0.5)
      
            
    def save(self, file_handle):
        """
        Save to the passed H5py file handle,
        i.e. something that cna be used with 
        file_handle.create_dataset()
        
        :param file_handle: file handle
        """
        
        for key, value in self.properties.items():
            file_handle.create_dataset(key, data = value)
      
            
    def _exposure_colorbar(self, style):
        """
        Plot a colorbar for the exposure map
        """
            
        cb_ax = plt.axes([0.25, 0, .5, .03], frameon = False)  
        vals = np.linspace(self.exposure_factor.min(),
                           self.exposure_factor.max(), 100)
        
        norm_proj = matplotlib.colors.Normalize(self.exposure_factor.min(),
                                                self.exposure_factor.max())
    
        bar = matplotlib.colorbar.ColorbarBase(cb_ax, values = vals, norm = norm_proj, cmap = style.cmap, 
                                               orientation = 'horizontal', drawedges = False, alpha = 1)

        bar.ax.get_children()[1].set_linewidth(0)
        bar.set_label('Relative exposure', color = style.textcolor)
        

    def draw_exposure_lim(self, skymap):
        """
        Draw a line marking the edge of the detector's exposure.
        
        :param skymap: an AllSkyMap instance.
        :param label: a label for the limit.
        """

        rightascensions = np.linspace(-180, 180, self.num_points)  
        limiting_dec = self.limiting_dec.deg
        boundary_decs = np.tile(limiting_dec, self.num_points)
        c = SkyCoord(ra = rightascensions * u.degree,
                     dec = boundary_decs * u.degree, frame = 'icrs')
        lon = c.galactic.l.deg
        lat = c.galactic.b.deg

        skymap.scatter(lon, lat, latlon = True, s = 8, 
                       color = 'grey', alpha = 1,
                       label = 'Limit of ' + self.label + '\'s exposure', zorder = 1)

            
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
        elif type == self._defined_types[1]:
            if isinstance(angle, int) or isinstance(angle, float):
                self.deg = np.rad2deg(angle)
            else:
                self.deg = [np.rad2deg(a) for a in angle]
            self.rad = angle


