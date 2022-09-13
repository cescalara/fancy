import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import h5py

from .uhecr import Uhecr
from .source import Source
from ..detector.detector import Detector
from fancy.plotting import AllSkyMap


class Data:
    """
    A container for high level storage of data.
    """

    def __init__(self):
        """
        A container for high level storage of data.
        """

        self._filename = None
        self._data = None

        # uhecr, source and detector objects
        self.uhecr = None
        self.source = None
        self.detector = None

    def add_source(self, filename, label=None):
        """
        Add a source object to the data cotainer from file.

        :param filename: name of the file containing the object's data
        :param label: reference label for the source object
        """

        if label == None:
            label = "VCV_AGN"

        new_source = Source()
        new_source.from_data_file(filename, label)

        # define source object
        self.source = new_source

    def add_uhecr(self, filename, label=None, ptype="p", gmf_model="JF12"):
        """
        Add a uhecr object to the data container from file.

        :param filename: name of the file containing the object's data
        :param label: reference label for the uhecr dataset
        :param ptype: composition type, this should be contained in the dataset itself
        """

        new_uhecr = Uhecr()
        new_uhecr.from_data_file(filename, label, ptype, gmf_model)

        # define uhecr object
        self.uhecr = new_uhecr

    def add_detector(self, detector_properties):
        """
        Add a detector object to complement the data.

        :param detector_properties: dict of properties.
        """

        new_detector = Detector(detector_properties)

        # define detector
        self.detector = new_detector

    def _uhecr_colorbar(self, cmap):
        """
        Add a colorbar normalised over all the Uhecr energies.

        :param cmap: matplotlib colorbar object
        """

        max_energies = []
        min_energies = []
        # find the min and max uhecr energies
        max_energies.append(max(self.uhecr.energy))
        min_energies.append(min(self.uhecr.energy))

        max_energy = max(max_energies)
        min_energy = min(min_energies)

        norm_E = matplotlib.colors.Normalize(min_energy, max_energy)

        # colorbar
        cb_ax = plt.axes([0.25, 0, 0.5, 0.03], frameon=False)
        vals = np.linspace(min_energy, max_energy, 100)
        bar = matplotlib.colorbar.ColorbarBase(
            cb_ax,
            values=vals,
            norm=norm_E,
            cmap=cmap,
            orientation="horizontal",
            drawedges=False,
            alpha=1,
        )
        # bar.ax.get_children()[1].set_linewidth(0)
        bar.set_label("UHECR Energy [EeV]")

    def show(self, save=False, savename=None, cmap=None):
        """
        Plot the data on a map of the sky.

        :param save: boolean input, saves figure if True
        :param savename: location to save to, required if
                         save == True
        """

        # plot style
        if cmap == None:
            cmap = plt.cm.get_cmap("viridis")

        # skymap
        skymap = AllSkyMap()
        skymap.fig.set_size_inches(12, 6)

        # uhecr object
        if self.uhecr:
            self.uhecr.plot(skymap)

        # source object
        if self.source:
            self.source.plot(skymap)

        # detector object
        # if self.detector:
        #    self.detector.draw_exposure_lim(skymap)

        # standard labels and background
        skymap.draw_standard_labels()

        # legend
        leg = skymap.ax.legend(frameon=False, bbox_to_anchor=(0.85, 0.85))

        # add a colorbar if uhecr objects plotted
        if self.uhecr and self.uhecr.N != 1:
            self._uhecr_colorbar(cmap)

        if save:
            skymap.fig.savefig(
                savename,
                dpi=500,
                bbox_extra_artists=[leg],
                bbox_inches="tight",
                pad_inches=0.5,
            )

        return skymap

    def from_file(self, filename):
        """
        Load data from an Analysis output file.
        :param filename: file name
        """

        # Read out information on data and detector
        uhecr_properties = {}
        source_properties = {}
        detector_properties = {}
        with h5py.File(filename, "r") as f:

            uhecr = f["uhecr"]

            for key in uhecr:
                uhecr_properties[key] = uhecr[key][()]

            source = f["source"]

            for key in source:
                source_properties[key] = source[key][()]

            detector = f["detector"]

            for key in detector:
                detector_properties[key] = detector[key][()]

        uhecr = Uhecr()
        uhecr.from_properties(uhecr_properties)

        source = Source()
        source.from_properties(source_properties)

        detector = Detector(detector_properties)

        # Add to data object
        self.uhecr = uhecr
        self.source = source
        self.detector = detector


class RawData:
    """
    Parses information for known data files in txt format.
    """

    def __init__(self, filename, filelayout):
        """
        Parses information for known data files in txt format.
        :filename: name of the file to parse
        :filelayout: list of column names in file
        """

        self._filename = filename
        self._filelayout = filelayout
        self._data = self._parse()

    def _parse(self):
        """
        Parse the data form the object's file.

        :return: arrays for each column in the data file
        """

        output = pd.read_csv(
            self._filename, comment="#", delim_whitespace=True, names=self._filelayout
        )

        output_dict = output.to_dict()

        return output_dict

    def get_by_name(self, name):
        """
        Get data entries by name.

        :param name: name of the data as in self._filelayout
        :return: an array of data entries
        """

        try:
            selected_data = np.array(list(self._data[name].values()))

        except ValueError:
            print("No data of type", name)
            selected_data = []

        return selected_data
