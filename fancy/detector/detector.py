import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from scipy import integrate

from fancy.detector.exposure import *
from fancy.plotting import AllSkyMap

__all__ = ["Detector", "Angle"]


class Detector:
    """
    UHECR observatory information and instrument response.
    """

    def __init__(self, detector_properties):
        """
        UHECR observatory information and instrument response.

        :param detector_properties: dict of properties.
        """

        self.properties = detector_properties

        self.label = detector_properties["label"]

        # if read from h5 file, convert bytestr to str
        if isinstance(self.label, bytes):
            self.label = self.label.decode("UTF-8")

        lat = detector_properties["lat"]  # radians
        lon = detector_properties["lon"]  # radians
        height = detector_properties["height"]  # metres

        self.location = EarthLocation(
            lat=lat * u.rad, lon=lon * u.rad, height=height * u.m
        )

        self.threshold_zenith_angle = Angle(
            detector_properties["theta_m"], "rad"
        )  # radians

        self._view_options = ["map", "decplot"]

        # See Equation 9 in Capel & Mortlock (2019)
        self.kappa_d = detector_properties["kappa_d"]
        self.coord_uncertainty = np.sqrt(7552.0 / self.kappa_d)

        self.energy_uncertainty = detector_properties["f_E"]

        self.num_points = 500

        self.params = [
            np.cos(self.location.lat.rad),
            np.sin(self.location.lat.rad),
            np.cos(self.threshold_zenith_angle.rad),
        ]

        self.exposure()

        self.area = detector_properties["A"]  # km^2

        self.alpha_T = detector_properties["alpha_T"]  # km^2 sr yr

        self.M, err = integrate.quad(m_integrand, 0, np.pi, args=self.params)

        self.params.append(self.alpha_T)
        self.params.append(self.M)

        self.start_year = detector_properties["start_year"]

        self.Eth = detector_properties["Eth"]

    def exposure(self):
        """
        Calculate and plot the exposure for a given detector
        location.
        """

        # define a range of declination to evaluate the
        # exposure at
        self.declination = np.linspace(-np.pi / 2, np.pi / 2, self.num_points)

        m = np.asarray([m_dec(d, self.params) for d in self.declination])

        self.exposure_max = np.max(m)

        # normalise to a maximum at 1
        # max value of exposure factor is normalization constant
        self.exposure_factor = m / self.exposure_max

        # find the point at which the exposure factor is 0
        # indexing value depends on TA or PAO
        # since TA only sees from dec ~ -10deg,
        # PAO only sees until dec ~ +45 deg
        declim_index = -1 if self.label.find("TA") != -1 else 0
        self.limiting_dec = Angle((self.declination[m == 0])[declim_index], "rad")

    def show(
        self,
        view=None,
        coord: str = "gal",
        save: bool = False,
        savename=None,
        cmap=None,
    ):
        """
        Make a plot of the detector's exposure

        :param view: a keyword describing how to show the plot
                     options are described by self._view_options
        :param save: boolean input, if True, the figure is saved
        :param savename: location to save to, required if save is
                         True
        """

        # define the style
        if cmap is None:
            cmap = plt.cm.get_cmap("viridis")

        # default is skymap
        if view is None:
            view = self._view_options[0]
        else:
            if view not in self._view_options:
                print("ERROR:", "view option", view, "is not defined")
                return

        # sky map
        if view == self._view_options[0]:

            # skymap
            skymap = AllSkyMap()
            skymap.fig.set_size_inches(12, 6)

            # define RA and DEC over all coordinates
            rightascensions = np.linspace(-np.pi, np.pi, self.num_points)
            declinations = self.declination

            norm_proj = matplotlib.colors.Normalize(
                self.exposure_factor.min(), self.exposure_factor.max()
            )

            # plot the exposure map
            # NB: use scatter as plot and pcolormesh have bugs in shiftdata methods
            for dec, proj in np.nditer([declinations, self.exposure_factor]):
                decs = np.tile(dec, self.num_points)
                c = SkyCoord(ra=rightascensions * u.rad, dec=decs * u.rad, frame="icrs")

                if coord == "gal":
                    lon = c.galactic.l.deg
                    lat = c.galactic.b.deg
                elif coord == "eq":
                    lon = c.ra.degree
                    lat = c.dec.degree
                else:
                    raise Exception("Coordinate {0} is not defined.".format(coord))

                skymap.scatter(
                    lon,
                    lat,
                    linewidth=3,
                    color=cmap(norm_proj(proj)),
                    alpha=0.7,
                )

            # plot exposure boundary
            self.draw_exposure_lim(skymap, coord=coord)

            # add labels
            skymap.draw_standard_labels()

            # add colorbar
            self._exposure_colorbar(cmap)

        # decplot
        elif view == self._view_options[1]:

            # plot for all decs

            fig, ax = plt.subplots()
            ax.plot(self.declination, self.exposure_factor, linewidth=5, alpha=0.7)
            ax.set_xlabel("$\delta$")
            ax.set_ylabel("m($\delta$)")

        if save:
            fig.savefig(savename, dpi=1000, bbox_inches="tight", pad_inches=0.5)

    def save(self, file_handle):
        """
        Save to the passed H5py file handle,
        i.e. something that cna be used with
        file_handle.create_dataset()

        :param file_handle: file handle
        """

        for key, value in self.properties.items():
            file_handle.create_dataset(key, data=value)

    def _exposure_colorbar(self, cmap):
        """
        Plot a colorbar for the exposure map

        :param cmap: matplotlib cmap object
        """

        cb_ax = plt.axes([0.25, 0, 0.5, 0.03], frameon=False)
        vals = np.linspace(self.exposure_factor.min(), self.exposure_factor.max(), 100)

        norm_proj = matplotlib.colors.Normalize(
            self.exposure_factor.min(), self.exposure_factor.max()
        )

        bar = matplotlib.colorbar.ColorbarBase(
            cb_ax,
            values=vals,
            norm=norm_proj,
            cmap=cmap,
            orientation="horizontal",
            drawedges=False,
            alpha=1,
        )

        bar.ax.get_children()[1].set_linewidth(0)
        bar.set_label("Relative exposure")

    def draw_exposure_lim(self, skymap: AllSkyMap, coord: str = "gal"):
        """
        Draw a line marking the edge of the detector's exposure.

        :param skymap: an AllSkyMap instance.
        :param label: a label for the limit.
        """

        rightascensions = np.linspace(-180, 180, self.num_points)
        limiting_dec = self.limiting_dec.deg
        boundary_decs = np.tile(limiting_dec, self.num_points)
        c = SkyCoord(
            ra=rightascensions * u.degree, dec=boundary_decs * u.degree, frame="icrs"
        )
        if coord == "gal":
            lon = c.galactic.l.deg
            lat = c.galactic.b.deg
        elif coord == "eq":
            lon = c.ra.degree
            lat = c.dec.degree
        else:
            raise Exception("Coordinate {0} is not defined.".format(coord))

        skymap.scatter(
            lon,
            lat,
            s=8,
            color="grey",
            alpha=1,
            label="Limit of " + self.label[:-4] + "'s exposure",
            zorder=1,
        )


class Angle:
    """
    Store angles as degree or radian for convenience.
    """

    def __init__(self, angle, type=None):
        """
        Store angles as degree or radian for convenience.

        :param angle: a single angle or array of angles
        """

        self._defined_types = ["deg", "rad"]

        # default: pass arguments in degrees
        if type == None:
            type = self._defined_types[0]

        if type == self._defined_types[0]:
            self.deg = angle
            if np.isscalar(angle):
                self.rad = np.deg2rad(angle)
            else:
                self.rad = [np.deg2rad(a) for a in angle]
        elif type == self._defined_types[1]:
            if np.isscalar(angle):
                self.deg = np.rad2deg(angle)
            else:
                self.deg = [np.rad2deg(a) for a in angle]
            self.rad = angle


if __name__ == "__main__":
    # import auger2014 data
    from fancy.detector.auger2014 import detector_properties

    # create Detector object
    detector = Detector(detector_properties)

    # show the exposure skymap
    detector.show(view="map", coord="gal")
