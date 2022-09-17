import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib import colors
from fancy.plotting.colours import lightgrey, grey, white
from astropy.coordinates import SkyCoord
import astropy.units as u

from ..detector.exposure import m_dec

try:

    import cartopy.crs as ccrs

except ImportError:

    ccrs = None

__all__ = ["AllSkyMapCartopy"]


class AllSkyMapCartopy:
    """
    The cartopy equivalent of AllSkyMap.

    The following functions in AllSkyMap are replaced as such:
    - AllSkyMap.geodesic -> matplotlib.pyplot.plot
    - AllSkyMap.tissot -> cartopy.mpl.geoaxes.GeoAxes.tissot
    - AllSkyMap.scatter -> matplotlib.pyplot.scatter

    All such functions are able to perform limb crossing automatically.

    Some other differences:
    - Since cartopy only supports Mollweide projections, this is used instead.
    - Since we work with lon / lat coordinates, ccrs.PlateCarree coordinates will be used for transformations
    """
    def __init__(
            self,
            projection="moll",
            lon_0=0.0,
            figsize=(12, 7),
            ax=None,
            **kwargs,
    ):
        """
        Initializes using plt.axes. Returns matplotlib.axes.Axes object.

        :params: projection: Currently only supports mollweide
        :params: lon_0: the central longitude in which the figure is defined.
        :params: figsize: figure size
        :params: kwargs: keyword args passed to plt.subplots()
        :params: ax:
        """

        if not ccrs:

            raise ImportError(
                "Cartopy must be installed to use this functionality")

        if projection == "moll":

            proj = ccrs.Mollweide(central_longitude=lon_0)
            proj._threshold = proj._threshold / 10.0
            transform = ccrs.PlateCarree()

        else:

            raise ValueError("Only moll projections are allowed.")

        self.lon_0 = lon_0
        self.transform = transform

        if ax is not None:

            self.fig = ax.get_figure()
            self.ax = ax

        else:

            fig, ax = plt.subplots(subplot_kw={"projection": proj}, **kwargs)
            fig.set_size_inches(figsize)

            self.fig = fig
            self.ax = ax

            # initial configurations
            self.set_extent()

            # Stick to galactic definition of
            # West<-360, 0->East
            self.ax.invert_xaxis()

    def set_gridlines(
        self,
        dx=60,
        dy=30,
        xlims=[0, 360],
        ylims=[-90, 90],
        fontsize=12,
        draw_ylabels=True,
        **kwargs,
    ):
        """
        Sets the gridlines going through the skymap and draw the labels from cartopy.mpl.geoaxes.gridlines().

        draw_ylabels sets whether to draw the ylabels or not.
        Currently there is an issue with duplicate ylabels when lon_0 = 0. Set draw_ylabels = False for lon_0 = 0. 
        """

        # xlocs needs to be fixed for gridlines reasons
        xlocs = np.arange(-180, 180, dx)
        ylocs = np.arange(ylims[0], ylims[1], dy)

        # labels from gridlines only constructs ylabels
        gl = self.ax.gridlines(
            draw_labels=draw_ylabels,
            crs=ccrs.PlateCarree(),
            xlocs=ticker.FixedLocator(xlocs),
            ylocs=ticker.FixedLocator(ylocs),
            x_inline=False,
            y_inline=False,
            formatter_kwargs={"direction_label": False},
            **kwargs,
        )
        # need this to remove xlabels and left ylabels
        gl.top_labels = gl.bottom_labels = gl.left_labels = False
        gl.right_labels = draw_ylabels  # otherwise will draw ylabels even when draw_labels = False
        gl.ylabel_style = {"fontsize": fontsize}

        # manually construct xgrid labels
        xtick_labels_dtype = float if dx <= 10 else int
        xtick_labels_init = np.arange(xlims[0],
                                      xlims[1],
                                      dx,
                                      dtype=xtick_labels_dtype)

        # ignore labels at boundary
        # subtract 180 since boundary at 180 when xtick = 0
        xtick_labels_nonzero_idx = np.nonzero(
            np.abs(xtick_labels_init - self.lon_0 - 180) % 360)
        xtick_labels = xtick_labels_init[xtick_labels_nonzero_idx]

        xpadding = -5.0  # padding from central latitude

        for xtick_label in xtick_labels:
            self.ax.text(
                xtick_label,
                xpadding,
                f"{xtick_label}\u00B0",
                va="center",
                ha="center",
                transform=self.transform,
                fontsize=fontsize,
            )

    def set_extent(self, glob=True, extents=None):
        """
        Set the extent of the Axes.

        If glob is True, then the Axes extent is set to the limits of the projection (default).
        Otherwise, the extent is set to extents = (x0, x1, y0, y1)
        """
        if glob:
            self.ax.set_global()
        else:
            self.ax.set_extent(extents=extents, crs=self.transform)

    def legend(self, **kwargs):
        """
        Create mpl instance of legend.
        """
        return self.ax.legend(**kwargs)

    def title(self, title, **kwargs):
        """wrapper for ax.set_title()"""
        self.ax.set_title(title, **kwargs)

    def geodesic(self, lon1, lat1, lon2, lat2, **kwargs):
        """
        Plot geodesics onto the skymap. lons / lats are assumed to be from
        SkyCoord objects.

        lon1, lat1: longitude and latitude of first point
        lon2, lat2: same but for second point

        lons \in [0, 360], lats \in [-90, 90]
        """

        self.ax.plot([lon1, lon2], [lat1, lat2],
                     transform=ccrs.Geodetic(),
                     **kwargs)

    def tissot(self, lon, lat, rad, npts=100, **kwargs):
        """
        Plot tissot onto skymap. lon / lat are assumed to be from SkyCoord objects.

        :param lon: longitudes of each point ([-180, 180])
        :param lat: latitudes of each point ([-90, 90])
        :param rad: size of each point (basemap 10.0 \approx 20 \approx 1 degree)
        :param npts: number of points at each point
        """

        # multiply radius by 55 (base unit)
        rad *= 55.0 * 2.0  # *2 to account for double radius size

        return self.ax.tissot(rad, lon, lat, npts, **kwargs)

    def scatter(self, lons, lats, **kwargs):
        """
        Plot scatter plot onto skymap (array-wise). Assumes we have taken
        longitudes and latitudes from SkyCoord, so that lons / lats are defined
        in that sense.

        :param lons: longitudes of each point ([-180, 180])
        :param lats: latitudes of each point ([-90, 90])
        """
        return self.ax.scatter(lons, lats, transform=self.transform, **kwargs)

    def contourf(self, lons, lats, vals, **kwargs):
        """Plot filled contour in skymap"""
        return self.ax.contourf(lons,
                                lats,
                                vals,
                                transform=self.transform,
                                **kwargs)

    def exposure_limit(self,
                       limiting_dec,
                       coord="G",
                       num_points=10000,
                       **kwargs):
        """
        Plot limit of exposure of given observatory.

        :param limiting_dec: declination of limit of exposure in degrees
        :param coord: the coordinate system to plot it in ("G" = galactic, "E"= equatorial).
            - galactic coords in lon, lat \in [-180, 180], [-90, 90]
            - equatorial coords in ra, dec \in [360, 0], [-90, 90] (like in TA paper)
        :param num_points: number of points to plot with
        """
        rightascensions = np.linspace(-180, 180, num_points)
        boundary_decs = np.tile(limiting_dec, num_points)
        c = SkyCoord(ra=rightascensions * u.degree,
                     dec=boundary_decs * u.degree,
                     frame="icrs")

        if coord == "G":
            x = c.galactic.l.deg
            y = c.galactic.b.deg
        elif coord == "E":
            x = c.icrs.ra.deg
            y = c.icrs.dec.deg

        return self.ax.scatter(x, y, transform=self.transform, **kwargs)

    def exposure_map(self,
                     detector_params,
                     coord="G",
                     num_points=220,
                     **kwargs):
        """
        Plot exposure of given observatory as contour map

        :param detector_params: parameters used to evaluate exposure function
        :param coord: the coordinate system to plot it in ("G" = galactic, "E"= equatorial).
            - galactic coords in lon, lat \in [-180, 180], [-90, 90]
            - equatorial coords in ra, dec \in [360, 0], [-90, 90] (like in TA paper)
        :param num_points: number of points to plot with
        """
        rightascensions = np.linspace(-180, 180, num_points)
        declinations = np.linspace(-np.pi / 2, np.pi / 2, num_points)

        # exposure function in full declination width
        m_full = np.asarray([m_dec(d, detector_params) for d in declinations])
        exposure_factor = m_full / np.max(m_full)

        exp_cmap = colors.LinearSegmentedColormap.from_list("custom",
                                                            [lightgrey, grey],
                                                            N=6)
        norm_proj = colors.Normalize(exposure_factor.min(),
                                     exposure_factor.max())

        for dec, proj in np.nditer([declinations, exposure_factor]):
            decs = np.tile(dec, num_points)
            c = SkyCoord(ra=rightascensions * u.rad,
                         dec=decs * u.rad,
                         frame="icrs")

            if coord == "G":
                x = c.galactic.l.deg
                y = c.galactic.b.deg
            elif coord == "E":
                x = c.icrs.ra.deg
                y = c.icrs.dec.deg

            if proj == 0:
                self.ax.scatter(
                    x,
                    y,
                    transform=self.transform,
                    linewidth=3,
                    color=white,
                    alpha=1,
                )
            else:
                self.ax.scatter(
                    x,
                    y,
                    transform=self.transform,
                    linewidth=3,
                    color=exp_cmap(norm_proj(proj)),
                    alpha=1,
                )

    def save(self, filename, dpi=300, **kwargs):
        """
        Saves the skymap to a given filename.
        """
        self.fig.savefig(filename, bbox_inches="tight", dpi=dpi, **kwargs)
