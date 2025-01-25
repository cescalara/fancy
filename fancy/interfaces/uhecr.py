import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
from datetime import date, timedelta
import h5py

# from tqdm import tqdm as progress_bar
from multiprocessing import Pool, cpu_count

from fancy.interfaces.model import coord_to_uv, uv_to_coord

from fancy.plotting import AllSkyMap

try:

    import crpropa

except ImportError:

    crpropa = None

__all__ = ["Uhecr"]


class Uhecr:
    """
    Stores the data and parameters for UHECRs
    """

    def __init__(self):
        """
        Initialise empty container.
        """

        self.properties = None
        self.source_labels = None

        self.nthreads = int(0.75 * cpu_count())

        # stubs for empty data
        self.rigidity = []
        self.kappa_ds = []

    def _get_angerr(self):
        """Get angular reconstruction uncertainty from label"""

        if self.label == "TA2015":
            from fancy.detector.TA2015 import sig_omega
        elif self.label == "auger2014":
            from fancy.detector.auger2014 import sig_omega
        elif self.label == "auger2010":
            from fancy.detector.auger2010 import sig_omega
        else:
            raise Exception("Undefined detector type!")

        return np.deg2rad(sig_omega)

    def from_data_file(
        self, filename, label, mass_group=1, gmf_model="JF12", exp_factor=1.0,
    ):
        """
        Define UHECR from data file of original information.

        Handles calculation of observation periods and
        effective areas assuming the UHECR are detected
        by the Pierre Auger Observatory or TA.

        :param filename: name of the data file
        :param label: reference label for the UHECR data set
        """

        self.label = label

        with h5py.File(filename, "r") as f:

            data = f[self.label]

            self.year = data["year"][()]
            self.day = data["day"][()]
            self.zenith_angle = np.deg2rad(data["theta"][()])
            self.energy = data["energy"][()]
            if "rigidity" in data:
                self.rigidity = data["rigidity"][()]
            self.N = len(self.energy)
            glon = data["glon"][()]
            glat = data["glat"][()]
            self.coord = self.get_coordinates(glon, glat)

            # check if we can extract exposure of UHECR (auger2022 dataset)
            if "exposure" in data:
                self.exposure = data["exposure"][()]
            else:
                self.exposure = np.ones(self.N)

            self.unit_vector = self.coord.cartesian.xyz.value.T
            self.period = self._find_period()
            self.A = self._find_area(exp_factor)

            self.mass_group = mass_group
            # first check if 
            if "gmf" in data and gmf_model != "None": 
                
                # only read if data exists for both GMF model key and MG key
                config_key = f"{gmf_model}_mg{mass_group}"
                if config_key not in list(data['gmf'].keys()):
                    raise KeyError(f"GMF data for configuration {gmf_model}, MG{mass_group} is not found.")

                glons_gb = data["gmf"][config_key]["glons_gb"][()]
                glats_gb = data["gmf"][config_key]["glats_gb"][()]
                self.coords_gb = self.get_coordinates(glons_gb, glats_gb)
                self.unit_vector_gb =  self.coords_gb.cartesian.xyz.value.T
                self.kappa_gmfs = data["gmf"][config_key]["kappa_gmf"][()]  # deflection parameter

    def _get_properties(self, analysis_type):
        """
        Convenience function to pack object into dict.
        """

        self.properties = {}
        self.properties["label"] = self.label
        self.properties["N"] = self.N
        self.properties["unit_vector"] = self.unit_vector
        self.properties["energy"] = self.energy
        self.properties["A"] = self.A
        self.properties["zenith_angle"] = self.zenith_angle

        if analysis_type == "joint_gmf_composition":
            self.properties["mass_group"] = self.mass_group
            self.properties["kappa_gmf"] = self.kappa_gmfs
            self.properties["unit_vector_gb"] = self.unit_vector_gb

        # Only if simulated UHECRs
        # if isinstance(self.source_labels, (list, np.ndarray)):
        #     self.properties['source_labels'] = self.source_labels

    def from_properties(self, uhecr_properties):
        """
        Define UHECR from properties dict.

        :param uhecr_properties: dict containing UHECR properties.
        :param label: identifier
        """

        self.label = uhecr_properties["label"]

        # Read from input dict
        self.N = uhecr_properties["N"]
        self.unit_vector = uhecr_properties["unit_vector"]
        self.energy = uhecr_properties["energy"]
        self.zenith_angle = uhecr_properties["zenith_angle"]
        self.A = uhecr_properties["A"]
        self.kappa_gmf = uhecr_properties["kappa_gmf"]

        # # decode byte string if uhecr_properties is read from h5 file
        ptype_from_file = uhecr_properties["ptype"]
        self.ptype = (
            ptype_from_file.decode("UTF-8")
            if isinstance(ptype_from_file, bytes)
            else ptype_from_file
        )

        # Only if simulated UHECRs
        # try:
        #     self.source_labels = uhecr_properties['source_labels']
        # except:
        #     pass

        # Get SkyCoord from unit_vector
        self.coord = uv_to_coord(self.unit_vector)

    def from_simulation(self, uhecr_properties):
        """
        Define UHECR from properties dict, evaluated from simulating
        dataset.

        Only real difference to from_properties() is in kappa_gmf,
        since evaluation of it depends on the parameters initialized
        for Uhecr().

        :param uhecr_properties: dict containing UHECR properties.
        :param label: identifier
        """

        self.label = uhecr_properties["label"]

        # Read from input dict
        self.N = uhecr_properties["N"]
        self.unit_vector = uhecr_properties["unit_vector"]
        self.energy = uhecr_properties["energy"]
        self.zenith_angle = uhecr_properties["zenith_angle"]
        self.A = uhecr_properties["A"]

        # decode byte string if uhecr_properties is read from h5 file
        ptype_from_file = uhecr_properties["ptype"]
        self.ptype = (
            ptype_from_file.decode("UTF-8")
            if isinstance(ptype_from_file, bytes)
            else ptype_from_file
        )

        # Only if simulated UHECRs
        # try:
        #     self.source_labels = uhecr_properties['source_labels']
        # except:
        #     pass

        # Get SkyCoord from unit_vector
        self.coord = uv_to_coord(self.unit_vector)
        # kappa_gmf set to zero array by default, if joint+gmf then
        # evaluated in analysis.simulate
        self.kappa_gmf = np.zeros(self.N)

    def plot(self, skymap: AllSkyMap, size=2):
        """
        Plot the Uhecr instance on a skymap.

        Called by Data.show()

        :param skymap: the AllSkyMap
        :param size: tissot radius
        :param source_labels: source labels (int)
        """

        lons = self.coord.galactic.l.deg
        lats = self.coord.galactic.b.deg

        alpha_level = 0.7

        # If source labels are provided, plot with colour
        # indicating the source label.
        if isinstance(self.source_labels, (list, np.ndarray)):

            Nc = max(self.source_labels)

            # Use a continuous cmap
            cmap = plt.cm.get_cmap("plasma", Nc)

            write_label = True

            for lon, lat, lab in np.nditer([lons, lats, self.source_labels]):
                color = cmap(lab)
                if write_label:
                    skymap.tissot(
                        lon,
                        lat,
                        size,
                        npts=30,
                        color=color,
                        lw=0,
                        alpha=0.5,
                        label=self.label,
                    )
                    write_label = False
                else:
                    skymap.tissot(
                        lon, lat, size, npts=30, color=color, lw=0, alpha=0.5
                    ),

        # Otherwise, use the cmap to show the UHECR energy.
        else:

            # use colormap for energy
            norm_E = matplotlib.colors.Normalize(min(self.energy), max(self.energy))
            cmap = plt.cm.get_cmap("viridis", len(self.energy))

            write_label = True
            for E, lon, lat in np.nditer([self.energy, lons, lats]):

                color = cmap(norm_E(E))

                if write_label:
                    skymap.tissot(
                        lon,
                        lat,
                        size,
                        npts=30,
                        color=color,
                        lw=0,
                        alpha=alpha_level,
                        label=self.label,
                    )
                    write_label = False
                else:
                    skymap.tissot(
                        lon,
                        lat,
                        size,
                        npts=30,
                        color=color,
                        lw=0,
                        alpha=alpha_level,
                    )

    def save(self, file_handle, analysis_type):
        """
        Save to the passed H5py file handle,
        i.e. something that cna be used with
        file_handle.create_dataset()

        :param file_handle: file handle
        """

        self._get_properties(analysis_type)

        for key, value in self.properties.items():
            file_handle.create_dataset(key, data=value)

    def _find_area(self, exp_factor):
        """
        Find the effective area of the observatory at
        the time of detection.

        Possible areas are calculated from the exposure reported
        in Abreu et al. (2010) or Collaboration et al. 2014.
        """

        if self.label == "auger2010":
            from ..detector.auger2010 import A1, A2, A3

            possible_areas = [A1, A2, A3]
            area = [possible_areas[i - 1] * exp_factor for i in self.period]

        elif self.label == "auger2014":
            from ..detector.auger2014 import (
                A1,
                A2,
                A3,
                A4,
                A1_incl,
                A2_incl,
                A3_incl,
                A4_incl,
            )

            possible_areas_vert = [A1, A2, A3, A4]
            possible_areas_incl = [A1_incl, A2_incl, A3_incl, A4_incl]

            # find area depending on period and incl
            area = []
            for i, p in enumerate(self.period):
                if self.zenith_angle[i] <= 60:
                    area.append(possible_areas_vert[p - 1] * exp_factor)
                if self.zenith_angle[i] > 60:
                    area.append(possible_areas_incl[p - 1] * exp_factor)

        elif "auger2022" in self.label:
            from ..detector.auger2022 import M, period_start, A
            
            # get period for each event - in years, taking into account days
            start_julianyear = period_start.year + period_start.day / 365.25
            deltats = (self.year + self.day / 365.25) - start_julianyear
            
            # very hacky, but only exists currently for backwards compatibility anyways
            # if len(self.exposure) > 0:
            #     area = self.exposure / (M * deltats)
            # else:
            #     area = np.tile(A, self.N)
            area = np.tile(A, self.N)

        elif "TA2015" in self.label:
            from ..detector.TA2015 import A1, A2

            possible_areas = [A1, A2]
            area = [possible_areas[i - 1] * exp_factor for i in self.period]

        else:
            print("Error: effective areas and periods not defined")

        return area

    def _find_period(self):
        """
        For a given year or day, find UHECR period based on dates
        in table 1 in Abreu et al. (2010) or in Collaboration et al. 2014.
        """

        period = []
        if self.label == "auger2014":
            from ..detector.auger2014 import (
                period_1_start,
                period_1_end,
                period_2_start,
                period_2_end,
                period_3_start,
                period_3_end,
                period_4_start,
                period_4_end,
            )

            # check dates
            for y, d in np.nditer([self.year, self.day]):
                d = int(d)
                test_date = date(y, 1, 1) + timedelta(d)

                if period_1_start <= test_date <= period_1_end:
                    period.append(1)
                elif period_2_start <= test_date <= period_2_end:
                    period.append(2)
                elif period_3_start <= test_date <= period_3_end:
                    period.append(3)
                elif test_date >= period_3_end:
                    period.append(4)
                else:
                    print("Error: cannot determine period for year", y, "and day", d)

        elif self.label == "TA2015":
            from ..detector.TA2015 import (
                period_1_start,
                period_1_end,
                period_2_start,
                period_2_end,
            )

            for y, d in np.nditer([self.year, self.day]):
                d = int(d)
                test_date = date(y, period_1_start.month, period_1_start.day) + timedelta(d)

                if period_1_start <= test_date <= period_1_end:
                    period.append(1)
                elif period_2_start <= test_date <= period_2_end:
                    period.append(2)
                elif test_date >= period_2_end:
                    period.append(2)
                else:
                    print("Error: cannot determine period for year", y, "and day", d)

        return period

    def select_period(self, period):
        """
        Select certain periods for analysis, other periods will be discarded.
        """

        # find selected periods
        if len(period) == 1:
            selection = np.where(np.asarray(self.period) == period[0])
        if len(period) == 2:
            selection = np.concatenate(
                [
                    np.where(np.asarray(self.period) == period[0]),
                    np.where(np.asarray(self.period) == period[1]),
                ],
                axis=1,
            )

        # keep things as lists
        selection = selection[0].tolist()

        # make selection
        self.A = [self.A[i] for i in selection]
        self.period = [self.period[i] for i in selection]
        self.energy = [self.energy[i] for i in selection]
        self.incidence_angle = [self.incidence_angle[i] for i in selection]
        self.unit_vector = [self.unit_vector[i] for i in selection]

        self.N = len(self.period)

        self.day = [self.day[i] for i in selection]
        self.year = [self.year[i] for i in selection]

        self.coord = self.coord[selection]

    def select_energy(self, Eth):
        """
        Select out only UHECRs above a certain energy.
        """

        selection = np.where(np.asarray(self.energy) >= Eth)
        selection = selection[0].tolist()

        # make selection
        self.A = [self.A[i] for i in selection]
        self.period = [self.period[i] for i in selection]
        self.energy = [self.energy[i] for i in selection]
        self.incidence_angle = [self.incidence_angle[i] for i in selection]
        self.unit_vector = [self.unit_vector[i] for i in selection]

        self.N = len(self.period)

        self.day = [self.day[i] for i in selection]
        self.year = [self.year[i] for i in selection]

        self.coord = self.coord[selection]

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