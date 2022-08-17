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

            raise ImportError("Cartopy must be installed to use this functionality")

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

    def add_grid_labels(
        self,
        dx,
        dy,
        xlims,
        ylims,
        lft=False,
        rgt=False,
        top=False,
        bot=False,
        mid=False,
        ynudge=5,
        spherical=False,
        reverse=False,
    ):
        """
        Add grid line labels manually for projections that aren't supported.
        Obtained from https://github.com/SciTools/cartopy/issues/881

        Args:
            xlocs (np.array): array of labels for longitude
            ylocs (np.array): array of labels for latitude
            lft (bool): whether to label the left side
            rgt (bool): whether to label the right side
            top (bool): whether to label the top side
            bot (bool): whether to label the bottom side
            spherical (bool): pad the labels better if a side of ax is spherical
            reverse (bool) : run the longitudes in reverse
        """

        x0, x1, dx = xlims[0], xlims[1], dx
        y0, y1, dy = ylims[0], ylims[1], dy

        if dx <= 10:
            dtype = float
        else:
            dtype = int

        if dy <= 10:
            dtype = float
        else:
            dtype = int

        for lon in np.arange(x0 + dx, x1, dx, dtype=dtype):
            if reverse:
                if lon <= 0:
                    label = r"{0}$^\circ$".format(np.abs(lon))
                if lon > 0:
                    label = r"-{0}$^\circ$".format(np.abs(lon))
            else:
                label = "{0}$^\circ$".format(lon)

            if top:
                self.ax.text(
                    lon,
                    y1,
                    r"{0}$^\circ$\n\n".format(lon),
                    va="center",
                    ha="center",
                    transform=self.transform,
                )
            if bot:
                self.ax.text(
                    lon,
                    y0,
                    r"\n\n{0}$^\circ$".format(lon),
                    va="center",
                    ha="center",
                    transform=self.transform,
                )

            if mid:
                ymid = (y0 + y1) / 2.0 - ynudge
                self.ax.text(
                    lon,
                    ymid,
                    label,
                    va="center",
                    ha="center",
                    transform=self.transform,
                    fontsize=10,
                )

        for lat in np.arange(y0 + dy, y1, dy, dtype=dtype):
            if spherical:
                if lat == 0:
                    va = "center"
                elif lat > 0:
                    va = "bottom"
                elif lat < 0:
                    va = "top"
            else:
                va = "center"
            if lft:
                self.ax.text(
                    x0,
                    lat,
                    "${0}^\circ$     ".format(lat),
                    va=va,
                    ha="right",
                    transform=self.transform,
                    fontsize=12,
                )
            if rgt:
                self.ax.text(
                    x1,
                    lat,
                    r"       {0}$^\circ$".format(lat),
                    va=va,
                    ha="left",
                    transform=self.transform,
                )

    def set_gridlines(
        self,
        label_fmt="default",
        dx=60,
        dy=30,
        xlims=[-180, 180],
        ylims=[-90, 90],
        ynudge=5,
        reverse=False,
        **kwargs,
    ):
        """
        Sets the gridlines going through the skymap and draw the labels.

        :param label_fmt: the format type to draw labels with. "mpl" is the default option,
                    along with 'TA', which defines lon in [360, 0].
        :param: dx, dy : spacing between points (default 60, 30)
        :param xlims, ylims: tuple of limits of longitude and latitude (default +-180, +-90)

        Note: if lon_0 < 0, ylocs shift to the right instead (bug that needs to be fixed).
        """

        # needs to be fixed for gridlines reasons
        xlocs = np.arange(-180, 180, dx)

        # use preset xlims / ylims
        if label_fmt == "default":
            xlims = np.array([0, 360])
            ylims = np.array([-90, 90])
            reverse = False
        if label_fmt == "mpl":
            xlims = np.array([-180, 180])
            ylims = np.array([-90, 90])
            reverse = False
        elif label_fmt == "TA":
            xlims = np.array([-180, 180])
            ylims = np.array([-90, 90])
            reverse = True
        elif label_fmt == "custom":
            xlims = np.array(xlims)
            ylims = np.array(ylims)
            reverse = reverse

        ylocs = np.arange(ylims[0], ylims[1], dy)

        ytick_fmt = ticker.StrMethodFormatter(r"${x}^\circ$")
        self.ax.gridlines(
            draw_labels=False,
            crs=self.transform,
            xlocs=xlocs,
            ylocs=ylocs,
            yformatter=ytick_fmt,
            xformatter=ytick_fmt,
            y_inline=False,
            **kwargs,
        )
        self.add_grid_labels(
            dx,
            dy,
            xlims,
            ylims,
            lft=True,
            rgt=False,
            mid=True,
            ynudge=ynudge,
            reverse=reverse,
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

        self.ax.plot([lon1, lon2], [lat1, lat2], transform=ccrs.Geodetic(), **kwargs)

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
        return self.ax.contourf(lons, lats, vals, transform=self.transform, **kwargs)

    def exposure_limit(self, limiting_dec, coord="G", num_points=10000, **kwargs):
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
        c = SkyCoord(
            ra=rightascensions * u.degree, dec=boundary_decs * u.degree, frame="icrs"
        )

        if coord == "G":
            x = c.galactic.l.deg
            y = c.galactic.b.deg
        elif coord == "E":
            x = c.icrs.ra.deg
            y = c.icrs.dec.deg

        return self.ax.scatter(x, y, transform=self.transform, **kwargs)

    def exposure_map(self, detector_params, coord="G", num_points=220, **kwargs):
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

        exp_cmap = colors.LinearSegmentedColormap.from_list(
            "custom", [lightgrey, grey], N=6
        )
        norm_proj = colors.Normalize(exposure_factor.min(), exposure_factor.max())

        for dec, proj in np.nditer([declinations, exposure_factor]):
            decs = np.tile(dec, num_points)
            c = SkyCoord(ra=rightascensions * u.rad, dec=decs * u.rad, frame="icrs")

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
