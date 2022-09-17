from abc import ABC, abstractmethod
from typing import List, Tuple
from scipy import stats, optimize


class EnergyLoss(ABC):
    """
    Abstract base class for energy loss calculations.
    """

    def __init__(self):
        """
        Abstract base class for energy loss calculations.
        """

        pass

    @abstractmethod
    def get_Eth_src(self, Eth: float, D: List[float]):
        """
        Get correscponding Eth_src for Eth defined at Earth.

        :param Eth: Threshold energy in EeV
        :param D: Source distances in Mpc
        """

        pass

    @abstractmethod
    def get_arrival_energy(self, Esrc: float, D: float):
        """
        Get the arrival energy for a given initial energy.
        Takes into account all propagation affects.

        :param Esrc: Source energy in EeV
        :param D: Source distance in Mpc
        """

        pass

    @abstractmethod
    def get_arrival_energy_vec(self, args: Tuple):
        """
        Get the arrival energy for a given initial energy.
        Takes into account all propagation affects.
        Version for parallelisation.

        :param args: Tuple containing source energies in EeV
            and source distances in Mpc
        """

        pass

    def get_Eex(self, Eth_src: List[float], alpha: float):
        """
        Find the expected energy from a simple power
        law spectrum as the median.

        :param Eth_src: Source threshold energy in EeV
        :param alpha: Spectral index of the power law
        """

        Eex = []
        for e in Eth_src:
            Eex.append(2 ** (1 / (alpha - 1)) * e)

        return Eex

    def get_kappa_ex(
        self,
        energy: List[float],
        b_field: float,
        distance: List[float],
        charge: int = 1,
    ):
        """
        Find kappa_ex for a given B field.
        Based on the deflection approximation used in
        Achterberg et al. 1999.

        :param energy: Energy at the source in EeV
        :param b_field: Magnetic field strength in nG
        :param distance: Distance of source in Mpc
        """

        l = 1  # Mpc

        kappa_ex = []
        for i, e in enumerate(energy):
            theta_p = (
                0.0401
                * charge
                * (e / 50) ** (-1)
                * b_field
                * (distance[i] / 10) ** 0.5
                * l**0.5
            )
            kappa_ex.append(2.3 / theta_p**2)

        return kappa_ex

    def p_gt_Eth(self, Earr: float, Eerr: float, Eth: float):
        """
        Probability that arrival energy is anove threshold.

        :param Earr: Arrival energy in EeV
        :param Eerr: Uncertainty in energy reconstruction (%)
        :param Eth: Threshold energy in EeV
        """

        return 1 - stats.norm.cdf(Eth, Earr, Eerr * Earr)

    def get_Eth_sim(self, Eerr: float, Eth: float):
        """
        Get a sensible threshold energy for the simulation
        given the threshold energy and uncertainty in the
        energy reconstruction.

        :param Eerr: Uncertainty in energy reconstruction (%)
        :param Eth: Threshold energy in EeV
        """

        E = optimize.fsolve(self.p_gt_Eth, Eth, args=(Eerr, Eth))

        return round(E[0])
