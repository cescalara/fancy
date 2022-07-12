import numpy as np
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import h5py

from fancy.interfaces.stan import coord_to_uv, uv_to_coord

from fancy.plotting import AllSkyMap

__all__ = ["Source"]


class Source:
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

        with h5py.File(filename, "r") as f:
            data = f[self.label]
            self.distance = data["D"][()]
            self.N = len(self.distance)
            glon = data["glon"][()]
            glat = data["glat"][()]
            self.coord = self.get_coordinates(glon, glat)

            if self.label != "cosmo_150" and self.label != "VCV_AGN":
                # Get names
                self.name = []
                for i in range(self.N):
                    self.name.append(data["name"][i])

        self.unit_vector = coord_to_uv(self.coord)

    def _get_properties(self):
        """
        Convenience function to pack object into dict.
        """

        self.properties = {}
        self.properties["label"] = self.label
        self.properties["N"] = self.N
        self.properties["unit_vector"] = self.unit_vector
        self.properties["distance"] = self.distance

    def from_properties(self, source_properties):
        """
        Define sources from properties dict.

        :param source_properties: dict containing source properties.
        :param label: identifier
        """

        self.label = source_properties["label"]

        self.N = source_properties["N"]
        self.unit_vector = source_properties["unit_vector"]
        self.distance = source_properties["distance"]

        self.coord = uv_to_coord(self.unit_vector)

    def plot(self, skymap: AllSkyMap, size=2.0, color="k"):
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
            [self.coord.galactic.l.deg, self.coord.galactic.b.deg]
        ):
            if write_label:
                skymap.tissot(
                    lon,
                    lat,
                    size,
                    npts=30,
                    color="k",
                    alpha=alpha_level,
                    label=self.label,
                )
                write_label = False
            else:
                skymap.tissot(lon, lat, size, npts=30, color="k", alpha=alpha_level)

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
            print("No fluxes to select on.")

    # convenience functions

    def get_coordinates(self, glon, glat, D=None):
        """
        Convert glon and glat to astropy SkyCoord
        Add distance if possible (allows conversion to cartesian coords)

        :return: astropy.coordinates.SkyCoord
        """

        if D:
            return SkyCoord(
                l=glon * u.degree,
                b=glat * u.degree,
                frame="galactic",
                distance=D * u.mpc,
            )
        else:
            return SkyCoord(l=glon * u.degree, b=glat * u.degree, frame="galactic")
