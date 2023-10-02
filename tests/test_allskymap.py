import numpy as np
from pathlib import Path

# import sys
# pyversion_info = sys.version_info

# # deal with versions since ligo.skymap does not work for Python3.9
# if pyversion_info.minor >= 9:
#     from fancy.plotting import AllSkyMap
# else:
#     from fancy.plotting import AllSkyMapCartopy as AllSkyMap

# defect to cartopy for now, to incorporate python3.8 as well
from fancy.plotting import AllSkyMapCartopy as AllSkyMap

from fancy.detector.auger2014 import detector_properties
from fancy.detector.detector import Detector


def test_basic_plotting(random_seed):

    np.random.seed(random_seed)

    N = 20

    lons = np.random.uniform(0, 360, N)
    lats = np.random.uniform(-90, 90, N)
    rads = np.random.uniform(5, 15, N)

    skymap = AllSkyMap()

    skymap.fig.set_size_inches(12, 6)

    for lon, lat, r in zip(lons, lats, rads):
        skymap.tissot(lon, lat, r, npts=30, alpha=0.5, lw=0)
        skymap.geodesic(lon, lat, 0, 0, color="k", alpha=0.5)
        skymap.scatter(lon, lat, color="k", linewidth=5, alpha=0.5)


def test_save(output_directory):

    file_name = Path(output_directory, "test_allskymap_plot.png")

    skymap = AllSkyMap()

    skymap.fig.set_size_inches(12, 6)

    skymap.tissot(0, 0, 10, npts=30)
    skymap.geodesic(0, 0, 20, 30, color="k")

    skymap.fig.savefig(
        file_name,
        dpi=1000,
        bbox_inches="tight",
        pad_inches=0.5,
    )


def test_detector_plotting():

    detector = Detector(detector_properties)

    for view, coord in zip(["map", "decplot"], ["gal", "eq"]):

        detector.show(view, coord)
