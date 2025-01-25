from abc import ABC, abstractmethod
from typing import List, Tuple
from scipy import stats, optimize

import os, h5py
import numpy as np
import astropy.units as u


from fancy import Data


class EnergyLoss(ABC):
    """
    Abstract base class for energy loss calculations.
    """

    def __init__(self, data: Data, verbose=False):
        """
        Abstract base class for energy loss calculations.
        """
        self.mass_group = data.detector.mass_group
        self.mass_group_idx = int(self.mass_group) - 1
        self.detector_type = data.detector.label
        self.Eth = data.detector.Eth
        self.Rth = data.detector.Rth
        self.Rth_max = data.detector.Rth_max
        self.verbose = verbose

        # minimum and maximum rigidity threshold
        self.Rmin = data.detector.Rth * u.EV
        self.Rmax = data.detector.Rth_max * u.EV

        print(
            f"Minimum rigidity for {self.detector_type}, mg{self.mass_group}: {self.Rmin:.2f}"
        )
        print(
            f"Maximum rigidity for {self.detector_type}, mg{self.mass_group}: {self.Rmax:.2f}\n"
        )

    @abstractmethod
    def initialise_grid(
        self,
        matrix_dir: str = "./resources/composition_weights_PSB.h5",
        alpha_min=-3,
        alpha_max=10,
        Nalphas=50,
    ):
        """
        Initalise our grid using composition weights

        :param matrix_dir: directory to composition weights
        :param alpha_min, alpha_max, Nalphas: the min / max and density of spectral index grid
        """
        # set grid for spectral index
        self.alpha_grid = np.linspace(alpha_min, alpha_max, Nalphas)
        self.Nalphas = Nalphas
        print(
            f"Shape of alpha grid: [{np.min(self.alpha_grid):.1f} : {np.max(self.alpha_grid):.1f} : {self.Nalphas}]"
        )

        if not os.path.exists(matrix_dir):
            raise FileNotFoundError(
                f"Composition weights file is missing. Re-run composition weight calculation."
            )

        with h5py.File(matrix_dir, "r") as f:
            self.distances = f["distances"][()] * u.Mpc
            rigidities_grid = (f["rigidities"][()] * u.GV).to(u.EV)
            dRs_grid = (f["rigidities_widths"][()] * u.GV).to(u.EV)

        Rmin_idx = np.digitize(self.Rmin, rigidities_grid, right=True)
        Rmax_idx = np.digitize(self.Rmax, rigidities_grid, right=True)

        # reset the Rmin and Rmax to those from this grid
        self.Rmin = rigidities_grid[Rmin_idx]
        self.Rmax = rigidities_grid[Rmax_idx]

        # truncate the rigidity grid & dR grid to Rmin and Rmax
        self.rigidities_grid = rigidities_grid[Rmin_idx:Rmax_idx+1]
        self.dRs_grid = dRs_grid[Rmin_idx:Rmax_idx+1]

        self.Ndistances = len(self.distances)
        self.NRs = len(self.rigidities_grid)

        ## setting up the rigidity grid
        # dlR = 0.05
        # lRbins = np.arange(
        #     np.log10(self.Rmin.to_value(u.EV)) - 0.5 * dlR,
        #     np.log10(self.Rmax.to_value(u.EV)) + 1.5 * dlR,
        #     dlR,
        # )  # 1.5 to include endpoint
        # self.dRs_grid = (10 ** lRbins[1:] - 10 ** lRbins[:-1]) * u.EV
        # self.rigidities_grid = 10 ** (0.5 * (lRbins[1:] + lRbins[:-1])) * u.EV
        # self.NRs = len(self.rigidities_grid)

        self.Eexs = np.zeros((self.Ndistances, self.Nalphas)) * u.EeV

    @abstractmethod
    def compute_Eexs(self):
        """Compute expected energies for all distance and energies"""
        pass

    @abstractmethod
    def p_gt_Rth(self, delta):
        """
        Probability that rigidity is anove threshold. For MG1, this is the arrival energy

        :param delta: Uncertainty in energy reconstruction (%)
        """
        pass
