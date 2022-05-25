import numpy as np
from matplotlib import pyplot as plt
import ligo.skymap.plot
from astropy.visualization.wcsaxes.patches import _rotate_polygon
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import astropy.units as u
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

    Edited to suit Galactic coords [0, 360] deg with boundary at 180 deg.

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

    def __init__(self, center, radius, resolution=100, vertex_unit=u.degree, **kwargs):

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
        vertices = np.array([lon, lat]).transpose()

        # split path into two sections if circle crosses -180, 180 bounds
        codes = []
        last = 0
        first = True
        for v in vertices:
            if first:
                codes.append(Path.MOVETO)
                first = False
            elif (
                ((last <= 180 and v[0] > 180) or (last > 180 and v[0] <= 180))
                and np.absolute(v[0] - last) < 300
                and ((v[0] + radius < 90) or v[0] - (radius < -90))
            ):
                codes.append(Path.MOVETO)

            else:
                codes.append(Path.LINETO)
            last = v[0]

        circle_path = Path(vertices, codes)

        super().__init__(circle_path, **kwargs)


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
        projection="galactic degrees mollweide",
        transform="galactic",
        ax=None,
    ):

        self.projection = projection
        self.transform = transform

        if not ax:

            fig, ax = plt.subplots(subplot_kw={"projection": self.projection})
            self.fig = fig
            self.ax = ax

        else:

            self.fig = ax.get_figure()
            self.ax = ax

    def _east_hem(self, lon):
        """
        Return True if lon is in the eastern hemisphere of the map wrt lon_0.
        """

        if (lon - self._lon_0) % 360.0 <= self.east_lon:

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

    def geodesic(self, lon1, lat1, lon2, lat2, del_s=0.01, clip=True, **kwargs):
        """
        Plot a geodesic curve from (lon1, lat1) to (lon2, lat2), with
        points separated by arc length del_s.

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
        cur_hem = self.east_hem(lon1)

        crossed_zero = False

        for i in range(len(lons))[1:]:

            if self.east_hem(lons[i]) == cur_hem:

                seg_lons.append(lons[i])
                seg_lats.append(lats[i])

            else:

                # We should interpolate a new pt at the boundary, but in
                # the meantime just rely on the step size being small.

                # if crossing zero, don't need new seg
                if self.cross_zero(lons[i - 1], lons[i]) or crossed_zero:

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

            x, y = self(lons, lats)

            line = self.ax.plot(x, y, **kwargs)[0]

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

        raise NotImplementedError("draw_standard_labels() coming soon")

    def tissot(self, lon_0, lat_0, radius_deg, npts, ax=None, **kwargs):
        """
        Draw a polygon centered at ``lon_0, lat_0``.  The polygon
        approximates a circle on the surface of the map with radius
        ``radius_deg`` degrees latitude along longitude ``lon_0``,
        made up of ``npts`` vertices.

        Uses the SphericalCircle class

        :return: A matplotlib PathPatch object
        """

        if not ax:

            ax = self.ax

        circle = SphericalCircle(
            (lon_0 * u.deg, lat_0 * u.deg),
            radius_deg * u.deg,
            transform=ax.get_transform(self.transform),
            **kwargs,
        )

        ax.add_patch(circle)

        return circle
