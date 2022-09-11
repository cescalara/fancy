from typing import List, Tuple
from astropy.constants import c, m_p
from astropy import units as u
from scipy import integrate

from fancy.propagation.energy_loss import EnergyLoss
from fancy.propagation.cosmology import H0, Om, DH

try:

    import crpropa as cr

except ImportError:

    cr = None


class CRPropaApproxEnergyLoss(EnergyLoss):
    """
    Energy loss calculation using CRPropa3.
    """

    def __init__(self, ptype="p", method: str = "loss_length"):
        """
        Energy loss calculation using CRPropa3

        :param ptype: Particle type ["p", "N", "Fe"]
        :param method: Approximation method used
        """

        if not cr:

            raise ImportError("CRPropa3 must be installed to use this functionality")

        # Cosmology
        h = H0.to_value(u.km / (u.s * u.Mpc)) / 100
        cr.setCosmologyParameters(h, Om)

        # Available ptypes
        self._ptype_dict = {}
        self._ptype_dict["p"] = [1, 1]
        self._ptype_dict["N"] = [14, 7]
        self._ptype_dict["Fe"] = [56, 26]

        # Available methods
        self._methods = ["loss_length"]

        if ptype in self._ptype_dict:

            self._ptype = ptype

        else:

            raise ValueError(f"Particle type {ptype} is not recognised")

        if method in self._methods:

            self._method = method

        else:

            raise ValueError(f"Method {method} is not recognised")

        if self._method == "loss_length":

            self._loss_length_init()

    def get_Eth_src(self, Eth: float, D: List[float]):

        Eth_src = []
        for d in D:
            Eth_src.append(self._get_source_threshold_energy(Eth, d)[0])

        return Eth_src

    def get_arrival_energy(self, Esrc: float, D: List[float]):

        Esrc = Esrc * 1.0e18
        integrator = integrate.ode(self._dEdr).set_integrator("lsoda", method="bdf")
        integrator.set_initial_value(Esrc, 0)
        r1 = D
        dr = min(D / 10, 10)

        while integrator.successful() and integrator.t < r1:
            integrator.integrate(integrator.t + dr)

        Earr = integrator.y / 1.0e18
        return Earr

    def get_arrival_energy_vec(self, args: Tuple):

        Evec, D = args

        Earr_vec = []
        for E in Evec:
            E = E * 1.0e18
            integrator = integrate.ode(self._dEdr).set_integrator("lsoda", method="bdf")
            integrator.set_initial_value(E, 0)
            r1 = D
            dr = min(D / 10, 10)

            while integrator.successful() and integrator.t < r1:
                integrator.integrate(integrator.t + dr)

            Earr = integrator.y / 1.0e18

            Earr_vec.append(Earr[0])

        return Earr_vec

    def _loss_length_init(self):
        """
        Initialise loss length method by loading
        relevant modules and fields from CRPropa.
        """

        modules = [
            cr.PhotoPionProduction,
            cr.PhotoDisintegration,
            cr.ElectronPairProduction,
        ]
        module_labels = ["photo_pion", "photo_dis", "pair_prod"]

        fields = [cr.CMB, cr.IRB_Kneiske04]
        field_labels = ["cmb", "irb"]

        self._interaction_dict = {}
        for interaction, mlabel in zip(modules, module_labels):

            for field, flabel in zip(fields, field_labels):

                key = mlabel + "_" + flabel
                self._interaction_dict[key] = interaction(field())

        # For energy -> gamma conversion
        A = self._ptype_dict[self._ptype][0]
        self._mass = A * m_p

        self._nucleus_id = cr.nucleusId(*self._ptype_dict[self._ptype])

    def _total_loss_length(self, z: float, E: float):
        """
        Total loss length.

        :param E: Energy in eV
        :param z: Redshift
        """

        gamma = E * u.eV / (self._mass * c**2)
        gamma = gamma.to_value(u.dimensionless_unscaled)[0]

        inv_total_loss_length = 0.0

        for _, value in self._interaction_dict.items():

            inv_total_loss_length += 1.0 / value.lossLength(
                int(self._nucleus_id), gamma, z
            )

        # Also adiabatic
        inv_total_loss_length += 1.0 / DH.to_value(u.m)

        total_loss_length = (1.0 / inv_total_loss_length) * u.m

        return total_loss_length.to_value(u.Mpc)

    def _dEdr(self, r: float, E: float):
        """
        The ODE to solve for propagation energy losses.

        :param r: Distance in Mpc
        :param E: Energy in eV
        """

        z = r / DH.to_value(u.Mpc)

        return -E / self._total_loss_length(z, E)

    def _dEdr_rev(self, r: float, E: float, D: float):
        """
        The reverse ODE for source energies.

        :param r: Reverse distance in Mpc
        :param E: Energy in eV
        :param D: Starting distance in Mpc
        """

        r_rev = D - r

        z = r_rev / DH.to_value(u.Mpc)

        return E / self._total_loss_length(z, E)

    def _get_source_threshold_energy(self, Eth: float, D: float):
        """
        Get the equivalent source energy for a given arrival energy.
        Takes into account all propagation affects.
        Solves the ODE dE/dr = E / Lloss.
        NB: input Eth and output E in EeV!
        """
        Eth = Eth * 1.0e18
        integrator = integrate.ode(self._dEdr_rev).set_integrator("lsoda", method="bdf")
        integrator.set_initial_value(Eth, 0).set_f_params(D)
        r1 = D
        dr = 1

        while integrator.successful() and integrator.t < r1:
            integrator.integrate(integrator.t + dr)

        Eth_src = integrator.y / 1.0e18
        return Eth_src
