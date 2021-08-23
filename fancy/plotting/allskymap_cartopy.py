import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import colors
from fancy.plotting.colours import lightgrey, grey, white
from astropy.coordinates import SkyCoord
import astropy.units as u

from ..detector.exposure import m_dec


import cartopy.crs as ccrs

# To allow the geodesics to be smoother, the threshold must be more lower.
# So we construct a simple class and change the internal property of ccrs.Mollweide
# workaround from https://stackoverflow.com/questions/40270990/cartopy-higher-resolution-for-great-circle-distance-line
class LowerThresholdMollweide(ccrs.Mollweide):
    @property
    def threshold(self):
        return 1e4

__all__ = ['AllSkyMapCartopy']

class AllSkyMapCartopy():
    '''
    The cartopy equivalent of AllSkyMap constructed using the Basemap package. 

    The following functions in AllSkyMap are replaced as such:
    - AllSkyMap.geodesic -> matplotlib.pyplot.plot
    - AllSkyMap.tissot -> cartopy.mpl.geoaxes.GeoAxes.tissot
    - AllSkyMap.scatter -> matplotlib.pyplot.scatter

    All such functions are able to perform limb crossing automatically.

    Some other differences:
    - Since cartopy only supports Mollweide projections, this is used instead.
    - Since we work with lon / lat coordinates, ccrs.PlateCarree coordinates will be used for transformations
    '''
    
    def __init__(self, projection="moll", lon_0 = 0., figsize=(12,7), fig_params={}, ax_params = {}):
        '''
        Initializes using plt.axes. Returns matplotlib.axes.Axes object.

        :params: projection: Currently only supports mollweide
        :params: lon_0: the central longitude in which the figure is defined.
        :params: figsize: figure size
        :params: fig_params: tuple that contains other parameters to set with figures
        :params: ax_params: keyword arguments to be passed to plt.axes
        '''

        if projection == "moll":
            proj = LowerThresholdMollweide(central_longitude=lon_0)
        else:
            raise ValueError("Only moll projections are allowed.")

        if plt.gca():
            self.fig = plt.figure(figsize=figsize, **fig_params)
            self.ax = plt.gca(projection=proj, **ax_params)

        else:
            self.fig = plt.figure(figsize=figsize, **fig_params)
            self.ax = plt.axes(projection=proj, **ax_params)

        self.lon_0 = lon_0   # for shifting with labels later on

        # initial configurations
        self.set_extent()


    def add_grid_labels(self, dx, dy, xlims, ylims,
                        lft=False, rgt=False, top=False, bot=False, mid=False, spherical=False):
        """Add grid line labels manually for projections that aren't supported. 
        Obtained from https://github.com/SciTools/cartopy/issues/881

        Args:
            xlocs (np.array): array of labels for longitude
            ylocs (np.array): array of labels for latitude
            lft (bool): whether to label the left side
            rgt (bool): whether to label the right side
            top (bool): whether to label the top side
            bot (bool): whether to label the bottom side
            spherical (bool): pad the labels better if a side of ax is spherical
        """
        x0, x1, dx = xlims[0], xlims[1], dx
        y0, y1, dy = ylims[0], ylims[1] + dy, dy

        if dx <= 10:
            dtype = float
        else:
            dtype = int

        if dy <= 10:
            dtype = float
        else:
            dtype = int

        for lon in np.arange(x0 + dx, x1, dx, dtype=dtype):
            if top:
                text = self.ax.text(lon, y1, r'{0}$^\circ$\n\n'.format(lon),
                            va='center', ha='center',
                            transform=ccrs.PlateCarree())
            if bot:
                text = self.ax.text(lon, y0, r'\n\n{0}$^\circ$'.format(lon),
                            va='center', ha='center',
                            transform=ccrs.PlateCarree())

            if mid:
                ymid = (y0 + y1) / 2. - 20
                text = self.ax.text(lon, ymid, r'${0}^\circ$'.format(lon),
                    va='center', ha='center',
                    transform=ccrs.PlateCarree())

        for lat in np.arange(y0 + dy, y1, dy, dtype=dtype):
            if spherical:
                if lat == 0:
                    va = 'center'
                elif lat > 0:
                    va = 'bottom'
                elif lat < 0:
                    va = 'top'
            else:
                va = 'center'
            if lft:
                text = self.ax.text(x0, lat, r'${0}^\circ$       '.format(lat), va=va, ha='left',
                            transform=ccrs.PlateCarree())
            if rgt:
                text = self.ax.text(x1, lat, r'       {0}$^\circ$'.format(lat), va=va, ha='left',
                            transform=ccrs.PlateCarree())


    def set_gridlines(self, dx=60, dy=30, xlims=(-180, 180), ylims=(-90, 90)):
        '''
        Sets the gridlines going through the skymap and draw the labels.

        :param: dx, dy : spacing between points (default 60, 30)
        :param xlims, ylims: tuple of limits of longitude and latitude (default +-180, +-90)

        Note: if lon_0 < 0, ylocs shift to the right instead (bug that needs to be fixed).
        '''
        xlocs = np.arange(xlims[0], xlims[1], dx)
        ylocs = np.arange(ylims[0] - dy, ylims[1] + dy, dy)
        ytick_fmt = ticker.StrMethodFormatter(r"${x}^\circ$")
        gl = self.ax.gridlines(crs=ccrs.PlateCarree(), xlocs=xlocs, ylocs=ylocs, yformatter=ytick_fmt)
        gl.ylabels_left = True
        self.add_grid_labels(dx, dy, xlims, ylims, lft=False, mid=True)
        # self.ax.gridlines(xlocs=xlocs, ylocs=ylocs)
        # self.add_grid_labels(self.lon_0 + xlocs, ylocs, lft=True, mid=True)

    def set_extent(self, glob=True, extents=None):
        '''
        Set the extent of the Axes.

        If glob is True, then the Axes extent is set to the limits of the projection (default).
        Otherwise, the extent is set to extents = (x0, x1, y0, y1)
        '''
        if glob:
            self.ax.set_global()
        else:
            self.ax.set_extent(extents=extents, crs=ccrs.PlateCarree())

    def legend(self, **kwargs):
        '''
        Create mpl instance of legend. 
        '''
        return self.ax.legend(**kwargs)

    def geodesic(self, lon1, lat1, lon2, lat2, **kwargs):
        '''
        Plot geodesics onto the skymap. lons / lats are assumed to be from
        SkyCoord objects.

        lon1, lat1: longitude and latitude of first point
        lon2, lat2: same but for second point

        lons \in [0, 360], lats \in [-90, 90]
        '''
        # shift lons by 180 due to cartopy / mpl plotting differences
        lon1, lon2 = 180. - lon1, 180. - lon2
        self.ax.plot([lon1, lon2], [lat1, lat2], transform=ccrs.Geodetic(), **kwargs)

    def tissot(self, lon, lat, rad, npts=100, **kwargs):
        '''
        Plot tissot onto skymap. lon / lat are assumed to be from SkyCoord objects.

        :param lon: longitudes of each point ([0, 360])
        :param lat: latitudes of each point ([-90, 90])
        :param rad: size of each point (basemap 10.0 \approx 20)
        :param npts: number of points at each point
        '''
        # shift longitudes by 180.
        lon = 180. - lon

        # multiply radius by 55 (base unit)
        rad *= 55. * 2.  # *2 to account for double radius size

        self.ax.tissot(rad, lon, lat, npts, **kwargs)

    def scatter(self, lons, lats, color, **kwargs):
        '''
        Plot scatter plot onto skymap (array-wise). Assumes we have taken
        longitudes and latitudes from SkyCoord, so that lons / lats are defined
        in that sense.

        :param lons: longitudes of each point ([0, 360])
        :param lats: latitudes of each point ([-90, 90])
        :param color: either color of all points or some array for color mapping
        '''
        self.ax.scatter(180. - lons, lats, c=color, transform=ccrs.PlateCarree(), **kwargs)

    def exposure_limit(self, limiting_dec, num_points=10000, **kwargs):
        '''
        Plot limit of exposure of given observatory.

        :param limiting_dec: declination of limit of exposure in degrees
        :param num_points: number of points to plot with
        '''
        rightascensions = np.linspace(-180, 180, num_points)
        boundary_decs = np.tile(limiting_dec, num_points)
        c = SkyCoord(ra = rightascensions * u.degree,
                    dec = boundary_decs * u.degree, frame = 'icrs')
        exp_lon = c.galactic.l.deg
        exp_lat = c.galactic.b.deg

        # shift by 180. since SkyCoord defines from [0, 360]
        self.ax.scatter(180. - exp_lon, exp_lat, transform=ccrs.PlateCarree(), **kwargs)

    def exposure_map(self, detector_params, num_points=220, **kwargs):
        '''
        Plot exposure of given observatory as contour map

        :param detector_params: parameters used to evaluate exposure function
        :param num_points: number of points to plot with
        '''
        rightascensions = np.linspace(-180, 180, num_points)
        declinations = np.linspace(-np.pi/2, np.pi/2, num_points)

        # exposure function in full declination width
        m_full = np.asarray([m_dec(d, detector_params) for d in declinations])
        exposure_factor = m_full / np.max(m_full)

        exp_cmap = colors.LinearSegmentedColormap.from_list('custom', 
                                                        [lightgrey, grey], N = 6)
        norm_proj = colors.Normalize(exposure_factor.min(), exposure_factor.max())

        for dec, proj in np.nditer([declinations, exposure_factor]):
            decs = np.tile(dec, num_points)
            c = SkyCoord(ra = rightascensions * u.rad,
                        dec = decs * u.rad, frame = 'icrs')
            lon = 180. - c.galactic.l.deg
            lat = c.galactic.b.deg
            if (proj == 0):
                self.ax.scatter(lon, lat, transform=ccrs.PlateCarree(), linewidth = 3,
                            color = white, alpha = 1)
            else:
                self.ax.scatter(lon, lat, transform=ccrs.PlateCarree(), linewidth = 3,
                            color = exp_cmap(norm_proj(proj)), alpha = 1)


    def save(self, filename):
        '''
        Saves the skymap to a given filename.
        '''
        self.fig.savefig(filename, bbox_inches='tight')

