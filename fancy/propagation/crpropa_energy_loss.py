import numpy as np
import os
from typing import List, Tuple
from astropy.constants import c, m_p
from astropy import units as u
from scipy import integrate, interpolate
import h5py

from fancy.propagation.energy_loss import EnergyLoss
from fancy.propagation.cosmology import H0, Om, DH
from fancy.utils.package_data import get_path_to_energy_approx_tables

try:
    import crpropa as cr

except ImportError:
    cr = None


class CRPropaApproxEnergyLoss(EnergyLoss):
    """
    Energy loss calculation using CRPropa3.
    """

    def __init__(self, ptype="p", method: str = "mean_sim_energy"):
        """
        Energy loss calculation using CRPropa3

        :param ptype: Particle type ["p", "N", "Si", "Fe"]
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
        self._ptype_dict["He"] = [4, 2]
        self._ptype_dict["Li"] = [6, 3]
        self._ptype_dict["Be"] = [8, 4]
        self._ptype_dict["B"] = [10, 5]
        self._ptype_dict["C"] = [12, 6]
        self._ptype_dict["N"] = [14, 7]
        self._ptype_dict["O"] = [16, 8]
        self._ptype_dict["Ne"] = [20, 10]
        self._ptype_dict["Na"] = [22, 11]
        self._ptype_dict["Mg"] = [24, 12]
        self._ptype_dict["Al"] = [27, 13]
        self._ptype_dict["Si"] = [28, 14]
        self._ptype_dict["P"] = [31, 15]
        self._ptype_dict["S"] = [32, 16]
        self._ptype_dict["Cl"] = [35, 17]
        self._ptype_dict["Ar"] = [40, 18]
        self._ptype_dict["K"] = [40, 19]
        self._ptype_dict["Ca"] = [40, 20]
        self._ptype_dict["Sc"] = [45, 21]
        self._ptype_dict["Ti"] = [48, 22]
        self._ptype_dict["V"] = [51, 23]
        self._ptype_dict["Cr"] = [52, 24]
        self._ptype_dict["Mn"] = [55, 25]
        self._ptype_dict["Fe"] = [56, 26]

        # Available methods
        self._methods = ["loss_length", "mean_sim_energy"]

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

        elif self._method == "mean_sim_energy":
            self._mean_sim_energy_init()

    def get_Eth_src(self, Eth: float, D: List[float]):
        Eth_src = []
        for d in D:
            Eth_src.append(self._get_source_threshold_energy(Eth, d))

        return Eth_src

    def get_arrival_energy(self, Esrc: float, D: float):
        if self._method == "loss_length":
            Esrc = Esrc * 1.0e18
            integrator = integrate.ode(self._dEdr).set_integrator("lsoda", method="bdf")
            integrator.set_initial_value(Esrc, 0)
            r1 = D
            dr = min(D / 10, 10)

            while integrator.successful() and integrator.t < r1:
                integrator.integrate(integrator.t + dr)

            Earr = integrator.y[0] / 1.0e18

        elif self._method == "mean_sim_energy":
            Earr = (
                10 ** self._interp_log10_arrival_energy(np.log10(Esrc), np.log10(D))[0]
            )

        return Earr

    def get_arrival_energy_vec(self, args: Tuple):
        Evec, D = args

        if self._method == "loss_length":
            Earr_vec = []
            for E in Evec:
                E = E * 1.0e18
                integrator = integrate.ode(self._dEdr).set_integrator(
                    "lsoda", method="bdf"
                )
                integrator.set_initial_value(E, 0)
                r1 = D
                dr = min(D / 10, 10)

                while integrator.successful() and integrator.t < r1:
                    integrator.integrate(integrator.t + dr)

                Earr = integrator.y / 1.0e18

                Earr_vec.append(Earr[0])

        elif self._method == "mean_sim_energy":
            Earr_vec = []
            for E in Evec:
                Earr = 10 ** self._interp_log10_arrival_energy(np.log10(E), np.log10(D))

                Earr_vec.append(Earr)

        return Earr_vec

    def _get_source_threshold_energy(self, Eth: float, D: float):
        """
        Get the equivalent source energy for a given arrival energy.
        Takes into account all propagation affects.

        NB: input Eth and output E in EeV!
        """

        if self._method == "loss_length":
            Eth = Eth * 1.0e18
            integrator = integrate.ode(self._dEdr_rev).set_integrator(
                "lsoda", method="bdf"
            )
            integrator.set_initial_value(Eth, 0).set_f_params(D)
            r1 = D
            dr = 1

            while integrator.successful() and integrator.t < r1:
                integrator.integrate(integrator.t + dr)

            Eth_src = integrator.y[0] / 1.0e18

        elif self._method == "mean_sim_energy":
            Esrc_range = 10 ** np.linspace(np.log10(Eth), np.log10(Eth * 1e3), 20)

            Earr_range = []
            for Esrc in Esrc_range:
                Earr = (
                    10
                    ** self._interp_log10_arrival_energy(np.log10(Esrc), np.log10(D))[0]
                )

                Earr_range.append(Earr)

            Eth_src = 10 ** np.interp(
                np.log10(Eth), np.log10(Earr_range), np.log10(Esrc_range)
            )

        return Eth_src

    def _mean_sim_energy_init(self):
        """
        Initialise mean sim energy method
        by setting up the tables for interpolation.
        """

        self._nucleus_id = cr.nucleusId(*self._ptype_dict[self._ptype])

        if self._ptype == "N":
            self._table_file = get_path_to_energy_approx_tables(
                "crpropa_mean_energy_N.h5"
            )

        elif self._ptype == "Si":
            self._table_file = get_path_to_energy_approx_tables(
                "crpropa_mean_energy_Si.h5"
            )

        else:
            raise NotImplementedError()

        with h5py.File(self._table_file, "r") as f:
            self._Esrc_grid = f["Esrc_grid"][()]  # EeV
            self._D_grid = f["D_grid"][()]  # Mpc
            mean_mass_number = f["mean_mass_number"][()]
            mean_energy_per_nucleon = f["mean_energy_per_nucleon"][()]  # EeV

        self._mean_energy = mean_mass_number * mean_energy_per_nucleon

        self._interp_log10_arrival_energy = interpolate.interp2d(
            np.log10(self._Esrc_grid),
            np.log10(self._D_grid),
            np.log10(self._mean_energy).T,
        )

    @classmethod
    def run_crpropa_sim(Esrc: float, D: float, nucleus_id, N_sim=1_000):
        """
        Run a CRPRopa sim of a monoenergetic source at
        distance D and store useful outputs.

        Used to calculate tables used.

        :param Esrc: Source energy in EeV
        :param D: Source distance in Mpc
        """

        output_file = ".crpropa_tmp"

        sim = cr.ModuleList()
        sim.add(cr.SimplePropagation(1 * cr.kpc, min([D / 10, 10]) * cr.Mpc))
        sim.add(cr.Redshift())
        sim.add(cr.PhotoPionProduction(cr.CMB()))
        sim.add(cr.PhotoPionProduction(cr.IRB_Kneiske04()))
        sim.add(cr.PhotoDisintegration(cr.CMB()))
        sim.add(cr.PhotoDisintegration(cr.IRB_Kneiske04()))
        sim.add(cr.NuclearDecay())
        sim.add(cr.ElectronPairProduction(cr.CMB()))
        sim.add(cr.ElectronPairProduction(cr.IRB_Kneiske04()))
        sim.add(cr.MinimumEnergy(1 * cr.EeV))

        obs = cr.Observer()
        obs.add(cr.ObserverPoint())
        output = cr.TextOutput(output_file, cr.Output.Event1D)
        obs.onDetection(output)
        sim.add(obs)

        source = cr.Source()
        source.add(cr.SourcePosition(D * cr.Mpc))
        source.add(cr.SourceRedshift1D())
        source.add(cr.SourceEnergy(Esrc * cr.EeV))
        source.add(cr.SourceParticleType(nucleus_id))

        sim.setShowProgress(False)
        sim.run(source, N_sim, True)
        output.close()

        sim_data = np.genfromtxt(output_file, names=True)
        A = np.array([cr.massNumber(int(id)) for id in sim_data["ID"].astype(int)])
        E = 10 ** (np.log10(sim_data["E"]) + 18)

        mean_mass_number = np.mean(A)

        mean_energy_per_nucleon = np.mean(E / A)  # eV

        mean_energy = mean_mass_number * mean_energy_per_nucleon  # eV

        os.remove(output_file)

        return mean_energy

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
