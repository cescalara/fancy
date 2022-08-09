import numpy as np
from matplotlib import pyplot as plt
import ligo.skymap.plot
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path

from astropy.visualization.wcsaxes.patches import _rotate_polygon
import astropy.units as u
from astropy.wcs import WCS
from astropy.io.fits import Header

from pyproj import Geod


__all__ = ["AllSkyMap"]


class SphericalCircle(PathPatch):
    # created from the astropy.visualization.wcsaxes.patches.SphericalCircle class
    # changed to path from polygon to create disjointed parts
    # code from https://github.com/grburgess/pyipn
    """
    Create a patch representing a spherical circle - that is, a circle that is
    formed of all the points that are within a certain angle of the central
    coordinates on a sphere. Here we assume that latitude goes from -90 to +90
    This class is needed in cases where the user wants to add a circular patch
    to a celestial image, since otherwise the circle will be distorted, because
    a fixed interval in longitude corresponds to a different angle on the sky
    depending on the latitude.

    Edited to suit Galactic coords [0, 360] deg with boundary at lon_0+180 deg.

    Parameters
    ----------
    center : tuple or `~astropy.units.Quantity`
        This can be either a tuple of two `~astropy.units.Quantity` objects, or
        a single `~astropy.units.Quantity` array with two elements.
    radius : `~astropy.units.Quantity`
        The radius of the circle
    resolution : int, optional
        The number of points that make up the circle - increase this to get a
        smoother circle.
    vertex_unit : `~astropy.units.Unit`
        The units in which the resulting polygon should be defined - this
        should match the unit that the transformation (e.g. the WCS
        transformation) expects as input.

    Notes
    -----
    Additional keyword arguments are passed to `~matplotlib.patches.Polygon`
    """

    def __init__(
        self, center, radius, resolution=100, lon_0=0.0, vertex_unit=u.degree, **kwargs
    ):

        boundary = (lon_0 + 180) % 360

        # Extract longitude/latitude, either from a tuple of two quantities, or
        # a single 2-element Quantity.
        longitude, latitude = center

        # Start off by generating the circle around the North pole
        lon = np.linspace(0.0, 2 * np.pi, resolution + 1)[:-1] * u.radian
        lat = np.repeat(0.5 * np.pi - radius.to_value(u.radian), resolution) * u.radian

        lon, lat = _rotate_polygon(lon, lat, longitude, latitude)

        # Extract new longitude/latitude in the requested units
        lon = lon.to_value(vertex_unit)
        lat = lat.to_value(vertex_unit)

        # Create polygon vertices
        vertices_in = np.array([lon, lat]).transpose()
        vertices_out = vertices_in[:]

        # split path into two sections if circle crosses boundary
        codes = []
        last = 0
        first = True
        n_insert = 1
        for i, v in enumerate(vertices_in):

            if first:

                codes.append(Path.MOVETO)
                first = False

            elif self._cross_boundary(last, v[0], boundary) and (
                np.absolute(latitude.to_value(vertex_unit))
                + radius.to_value(vertex_unit)
                < 90
            ):

                codes.append(Path.MOVETO)

            else:

                codes.append(Path.LINETO)

            last = v[0]

        circle_path = Path(vertices_in, codes)

        super().__init__(circle_path, **kwargs)

    def _cross_boundary(self, point_one: float, point_two: float, boundary: float):
        """
        Check if points cross a boundary in the range [0, 360].
        """

        # scale to boundary at zero
        p1 = (point_one - boundary) % 360
        p2 = (point_two - boundary) % 360

        return self._cross_zero(p1, p2)

    def _cross_zero(self, point_one: float, point_two: float):
        """
        Check if points cross the 0/360 boundary.
        """

        if (point_one >= 0 and point_one < 90) and (
            point_two < 360 and point_two > 270
        ):
            return True

        if (point_one < 360 and point_one > 270) and (
            point_two >= 0 and point_two < 90
        ):
            return True

        return False


class AllSkyMap(object):
    """
    AllSkyMap is a plotting interface for celestial data. It tries to use
    matplotlib as much as possible, with projections implemented from
    ligo.skymap and pyproj for geodesic calculations.

    This version is an upgrade to more lightweight dependencies, without
    basemap/cartopy. The implemented methods are designed to be backwards
    compatible with the old AllSkyMap where possible.
    """

    def __init__(
        self,
        projection: str = "galactic degrees mollweide",
        transform: str = "galactic",
        lon_0: float = 0.0,
        ax=None,
    ):

        self.projection = projection
        self.transform = transform
        self.lon_0 = lon_0
        self.boundary = (lon_0 + 180) % 360
        self._east_lon = (self.boundary + 1e-20) % 360
        self._west_lon = (self.boundary - 1e-20) % 360

        if not ax:

            fig, ax = plt.subplots(subplot_kw={"projection": self.projection})

            # Change centre of horizontal axis
            h = Header(ax.header, copy=True)
            h["CRVAL1"] = self.lon_0
            ax.reset_wcs(WCS(h))

            # Put zero to the left
            if self.lon_0 % 360 <= 180:
                ax.invert_xaxis()

            self.fig = fig
            self.ax = ax

        else:

            self.fig = ax.get_figure()
            self.ax = ax

    def _east_hem(self, lon):
        """
        Return True if lon is in the eastern hemisphere of the map wrt lon_0.
        """

        if (lon - 0.0) % 360.0 <= 180.0:

            return True

        else:

            return False

    def _cross_zero(self, lon1, lon2):
        """
        Return True if a line joint lon1 and lon2 crosses over the 0/360 boundary.
        """

        # from east -> west
        if (lon1 > 0 and lon1 < 90) and (lon2 > -90 and lon2 < 0):
            return True

        # from west->east
        if (lon1 > -90 and lon1 < 0) and (lon2 > 0 and lon2 < 90):
            return True

        return False

    def geodesic(self, lon1, lat1, lon2, lat2, del_s=1000, clip=True, **kwargs):
        """
        Plot a geodesic curve from (lon1, lat1) to (lon2, lat2), with
        points separated by arc length del_s (in m).

        If the geodesic does not cross the map limb, there will be only a single curve;
        if it crosses the limb, there will be two curves.

        :return: a list of Line2D instances for the curves comprising the geodesic
        """

        # Find lons and lats along geodesic
        gc = Geod(ellps="WGS84")

        az12, az21, dist = gc.inv(lon1, lat1, lon2, lat2)

        npoints = int((dist + 0.5 ** del_s) / del_s)

        lonlats = gc.npts(
            lon1,
            lat1,
            lon2,
            lat2,
            npoints,
            initial_idx=0,
            terminus_idx=0,
        )

        lons = []
        lats = []
        for lon, lat in lonlats:
            lons.append(lon)
            lats.append(lat)

        # Break the arc into segments as needed, when there is a longitudinal
        # hemisphere crossing.
        segs = []
        seg_lons, seg_lats = [lon1], [lat1]
        cur_hem = self._east_hem(lon1)

        crossed_zero = False

        for i in range(len(lons))[1:]:

            if self._east_hem(lons[i]) == cur_hem:

                seg_lons.append(lons[i])
                seg_lats.append(lats[i])

            else:

                # We should interpolate a new pt at the boundary, but in
                # the meantime just rely on the step size being small.

                # if crossing zero, don't need new seg
                if self._cross_zero(lons[i - 1], lons[i]) or crossed_zero:

                    crossed_zero = True
                    seg_lons.append(lons[i])
                    seg_lats.append(lats[i])

                else:

                    segs.append((seg_lons, seg_lats))
                    seg_lons, seg_lats = [lons[i]], [lats[i]]
                    cur_hem = not cur_hem

        segs.append((seg_lons, seg_lats))

        # Plot each segment; return a list of the mpl lines.
        lines = []
        for lons, lats in segs:

            line = self.ax.plot(
                lons, lats, transform=self.ax.get_transform(self.transform), **kwargs
            )[0]

            lines.append(line)

        # If there are multiple segments and no color args, reconcile the
        # colors, which mpl will have autoset to different values.
        # *** Does this screw up mpl's color set sequence for later lines?
        if "c" not in kwargs or "color" in kwargs:

            if len(lines) > 1:

                c1 = lines[0].get_color()

                for line in lines[1:]:

                    line.set_color(c1)

        return lines

    def draw_standard_labels(self, minimal=False, fontsize=14):
        """
        Add the standard labels for parallels and meridians to the map.

        :param minimal: If True, draw less dense label grid.
        """

        self.ax.grid()
        self.ax.coords["glon"].set_ticks([0, 60, 120, 180, 240, 300] * u.deg)
        self.ax.coords["glon"].set_major_formatter("dd")

    def tissot(self, lon, lat, radius, npts=100, ax=None, **kwargs):
        """
        Draw a polygon centered at ``lon, lat``.  The polygon
        approximates a circle on the surface of the map with radius
        ``radius`` degrees latitude along longitude ``lon``,
        made up of ``npts`` vertices.

        Uses the SphericalCircle class

        :return: A matplotlib PathPatch object
        """

        if not ax:

            ax = self.ax

        circle = SphericalCircle(
            (lon * u.deg, lat * u.deg),
            radius * u.deg,
            lon_0=self.lon_0,
            resolution=npts,
            transform=ax.get_transform(self.transform),
            **kwargs,
        )

        ax.add_patch(circle)

        return circle

    def _alt_tissot(self, lon, lat, radius, npts=100, ax=None, **kwargs):
        """
        Alternative tissot style [WIP].
        """

        if not ax:

            ax = self.ax

        g = Geod(ellps="WGS84")

        az12, az21, dist = g.inv(lon, lat, lon, lat + radius)

        start_hem = self._east_hem(lon)

        segs1 = [lon, lat + radius]
        over, segs2 = [], []
        delaz = 36 / npts
        az = az12
        last_lon = lon

        # handling of the poles
        if np.absolute(lat) + radius >= 90:

            # use half of the pts for the shape
            # and other half on the border
            N1 = int(npts / 2)
            lats = np.zeros(N1)
            lons = np.zeros(N1)

            for i in range(N1):

                az += delaz * 2
                lon_i, lat_i, az21 = g.fwd(lon, lat, az, dist)
                lons[i] = lon_i
                lats[i] = lat_i

            a = list(np.argsort(lons))
            lons = lons[a]
            lats = lats[a]

            N2 = int(npts / 4)
            segs = []
            dL = (90 - np.absolute(lats[0])) / (N2 - 1)
            r = range(N2)

            # for the south pole, reverse the ordering
            # in order to plot correct polygon
            if lat < 0:

                r = list(reversed(r))
                segs.extend(zip(lons, lats))

            # first half of map border
            lon_1 = (self.boundary + 1e-20) * np.sign(lat) * np.ones(N2)
            lat_1 = np.sign(lat) * (90 - np.array(r) * dL)
            segs.extend(zip(lon_1, lat_1))

            if lat > 0:
                segs.extend(zip(lons, lats))

            # second half of the map border
            r = list(reversed(r))
            lon_1 = (self.boundary - 1e-20) * np.sign(lat) * np.ones(N2)
            lat_1 = np.sign(lat) * (90 - np.array(r) * dL)
            segs.extend(zip(lon_1, lat_1))

            poly = Polygon(
                segs,
                transform=ax.get_transform(self.transform),
                **kwargs,
            )
            ax.add_patch(poly)
            return [poly]

        # handle boundary cross away from pole
        if start_hem:
            adj_lon = self._east_lon
            opp_lon = self._west_lon
        else:
            adj_lon = self._west_lon
            opp_lon = self._east_lon

        for i in range(npts):

            az = az + delaz
            lon_i, lat_i, az21 = g.fwd(lon, lat, az, dist)

            if self._east_hem(lon_i) == start_hem:

                segs1.append((lon_i, lat_i))
                last_lon = lon

            else:

                segs1.append((adj_lon, lat_i))
                segs2.append((opp_lon, lat_i))
                over.append((lon_i, lat_i))
                last_lon = lon

        poly1 = Polygon(
            segs1,
            transform=ax.get_transform(self.transform),
            **kwargs,
        )
        ax.add_patch(poly1)

        if segs2:
            over.reverse()
            segs2.extend(over)
            poly2 = Polygon(
                segs2,
                transform=ax.get_transform(self.transform),
                **kwargs,
            )
            ax.add_patch(poly2)
            return [poly1, poly2]
        else:
            return [poly]

    def scatter(self, x, y, ax=None, **kwargs):
        """
        Pass to matplotlib scatter.
        """

        if not ax:

            ax = self.ax

        ax.scatter(
            x,
            y,
            transform=ax.get_transform(self.transform),
            **kwargs,
        )
