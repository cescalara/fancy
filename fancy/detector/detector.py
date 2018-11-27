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

    def __init__(self, location, threshold_zenith_angle, area, total_exposure, kappa_c, label):
        """
        UHECR observatory information and instrument response. 
        
        :param location: EarthLocation object
        :param threshold_zenith_angle: maximum detectable
                                       zenith angle in rad.
        :param area: effective area in [km^2]
        :param total_exposure: in [km^2 sr year]
        :param label: identifier
        """

        self.label = label
        
        self.location = location
        
        self.threshold_zenith_angle = Angle(threshold_zenith_angle, 'rad')

        self._view_options = ['map', 'decplot']

        self.kappa_c = kappa_c
        
        self.num_points = 500

        self.params = [np.cos(self.location.lat.rad),
                       np.sin(self.location.lat.rad),
                       np.cos(self.threshold_zenith_angle.rad)]

        self.exposure()

        self.area = area

        self.alpha_T = total_exposure
        
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

            
            # define RA and DEC over all coordinates
            rightascensions = np.linspace(-np.pi, np.pi, self.num_points)
            declinations = self.declination
            
            cmap = style.cmap
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
            skymap.draw_standard_labels(style.cmap, style.textcolor)

            # add colorbar
            self._exposure_colorbar(style)

        # decplot
        elif view == self._view_options[1]:

            # plot for all decs
 
            plt.figure()
            plt.plot(self.declination, self.exposure_factor, linewidth = 5, alpha = 0.7)
            plt.xlabel('$\delta$');
            plt.ylabel('m($\delta$)');


        if save:
            plt.savefig(savename, dpi = 1000,
                    bbox_inches = 'tight', pad_inches = 0.5)

            
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

        skymap.scatter(lon, lat, latlon = True, s = 10, 
                       color = 'k', alpha = 0.1,
                       label = 'limit of ' + self.label + '\'s exposure')

            
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


