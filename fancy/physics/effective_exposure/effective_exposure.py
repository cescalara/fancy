"""Class that calculates effective exposure."""

import os
import pickle

import astropy.units as u
import h5py
import healpy
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.interpolate import CubicSpline
from scipy.stats import norm
from tqdm import tqdm
from typing_extensions import Self

from fancy import Data
from fancy.detector.exposure import m_dec
from fancy.physics.gmf import GMFLensing
from fancy.utils.package_data import get_path_to_kappa_theta


class EffectiveExposure:
    """Class to manage calculation of the effective exposure from given source(s), and constructs tables that will be passed to stan for interpolation."""

    def __init__(self : Self, data: Data, gmf_model: str = "None", verbose : bool=False) -> None:
        """
        Class to manage calculation of the effective exposure from given source(s).

        Also constructs tables that will be passed to stan for interpolation.

        data : fancy.interfaces.Data
            Container object that tracks the source, UHECR, and detector information
        gmf_model : str, default="None"
            the GMF model to consider. Default is None, which ignores GMF effects
        verbose : bool, default=False
            to print out additional statements for debugging or not.
        """
        self.data = data
        self.gmf_model = gmf_model
        self.verbose = verbose

        # detector properties
        self.mass_group = data.detector.mass_group
        self.detector_type = data.detector.label

        self.source_type = data.source.label
        self.Dsrcs = data.source.distance * u.Mpc
        self.coords_src = data.source.coord  # this returns galactic coordinates
        src_names = data.source.name
        self.Nsrcs = len(self.Dsrcs)

        print(
            f"Configuration: {self.source_type}, {self.detector_type}, {self.mass_group}"
        )

        if verbose:
            print(f"Sources: {src_names}")
            print(f"Distances: {self.Dsrcs}")
            print(f"Coordinates: {self.coords_src}")

    def initialise_grids(
        self: Self,
        energy_loss_table_file: str,
        Bigmf_min: float = 0.001,
        Bigmf_max: float = 10,
        NBigmfs: int = 50,
        Npixels: int = 49152,
        kappa_theta_filename: str = "kappa_theta_map.pkl",
    ) -> None:
        """
        Initialise grids used for effective exposure calculation.

        The grids used are based on the energy loss tables that are generated from running
        the ProtonEnergyLoss or NucleiEnergyLoss model. Please ensure that the energy loss
        tables are generated for the particular configuration considered.

        Parameter:
        ----------
        energy_loss_table_file : str
            path to the table for energy losses.
        Bigmf_min : float, default=0.001
            the minimum value of the IGMF strength used when generating the grid (in log space)
        Bigmf_max : float, default=10
            same as Bigmf_min, but maximum value instead
        NBigmfs : int, default=50
            the number of points in the Bigmf grid
        Npixels : int, default=49152
            the number of pixels used to describe the healpy grid.
            Default is 49152, which is the default value used when generating the
            lens in CRPropa.
            DO NOT CHANGE UNLESS CRPROPA DOES SO!
        kappa_theta_filename : str, default=kappa_theta_map.pkl
            the path to the mapping from kappa <-> theta in the vMF distribution.
            The default name uses the file that exists in `fancy/utils/resources`
        """
        # generate logarithmically spaced magnetic field grid
        self.Bigmf_grid = (
            np.logspace(np.log10(Bigmf_min), np.log10(Bigmf_max), NBigmfs) * u.nG
        )
        self.NBigmfs = NBigmfs

        # initialise healpy grid for exposure calculation
        Npixels = 49152  # set from CRPropa, pixelisation of order 6
        Nside = healpy.npix2nside(Npixels)
        self.delta_ang = (4 * np.pi) / Npixels  # uniform grid spacing
        if self.verbose:
            print(f"Nside: {Nside}, angular spacing: {self.delta_ang:.3e} sr")

        # converting from pixels -> skycoord
        pix_arr = np.arange(0, Npixels, 1, dtype=int)
        uvs_healpy = np.array(healpy.pix2vec(Nside, pix_arr)).T
        self.coords_healpy = SkyCoord(
            uvs_healpy, frame="galactic", representation_type="cartesian"
        )

        # create grid of rigidities to compute effective exposure
        self.rigidities_grid = (
            np.logspace(
                np.log10(self.data.detector.Rth),
                np.log10(self.data.detector.Rth_max),
                50,
            )
            * u.EV
        )
        self.NRs = len(self.rigidities_grid)

        # read out relevant parameters from energy loss tables
        if not os.path.exists(energy_loss_table_file):
            raise FileNotFoundError(
                "Energy loss tables have not been generated. Construct them first!"
            )

        with h5py.File(energy_loss_table_file, "r") as f:
            # find the relevant group
            config_label = f"{self.detector_type}_mg{self.mass_group}"
            self.alpha_grid = f[config_label]["alpha_grid"][()]
            self.distances_grid = f[config_label]["distances_grid"][()] * u.Mpc

            self.Nalphas = len(self.alpha_grid)
            self.Ndistances = len(self.distances_grid)

            if self.mass_group != 1:
                log10_arrspect_grid = f[config_label]["log10_arrspect_grid"][()]
                log10_Rgrid = f[config_label]["log10_rigidities"][()]

                # get the arrival spectrum values for the values of the rigidity
                # grid defined using look-up tables
                # interpolation doesnt work for faraway sources due to sharp gradients
                Rs_idces = [
                    np.digitize(r.value, 10**log10_Rgrid, right=True)
                    for r in self.rigidities_grid
                ]

                self.arrspects_grid = np.zeros(
                    (self.Ndistances, self.NRs, self.Nalphas)
                ) * (1 / u.EV)
                for ir, r_idx in enumerate(Rs_idces):
                    self.arrspects_grid[:, ir, :] = (
                        10 ** log10_arrspect_grid[:, r_idx, :]
                    ) * (1 / u.EV)
            else:
                # parametrize expected energies as rigidities
                self.Eexs_grid = 10 ** f[config_label]["log10_Eexs_grid"][()] * u.EV
                self.Rth_srcs = f[config_label]["Rth_src_grid"][()] * u.EV

        # read in the theta <-> kappa interpolated file
        kappa_theta_file = str(get_path_to_kappa_theta(kappa_theta_filename))
        (_, _, self.f_log10_kappa) = pickle.load(open(kappa_theta_file, "rb"))

    def _compute_exposure(self: Self) -> None:
        """Compute the exposure as a function of declination in healpy."""
        # first transform coordianates to declination
        self.coords_healpy.representation_type = "unitspherical"
        self.coords_healpy.transform_to("icrs")
        decs_healpy_grid = self.coords_healpy.icrs.dec.rad

        # compute exposure, which is function of declination only
        p = self.data.detector.params
        self.exposures = p[3] / p[4] * m_dec(decs_healpy_grid, p) * (u.km**2 * u.yr)

        # transform the coordinates back to galactic
        self.coords_healpy.transform_to("galactic")

    def compute_effective_exposure(
        self: Self,
        gmflens: GMFLensing,
        kappa_max: float = 1e6,
        exposure_min: float = 1e-30,
    ) -> None:
        """
        Compute the effective exposure.

        Parameter:
        ----------
        gmflens: fancy.physics.gmf.gmflens.GMFLensing
            Container for the GMF lens that will map the lens back to Earth
        kappa_max : float, default=1e6
            maximum threshold value for kappa computation
        exposure_min: float, default=1e-30
            minimum threshold value for exposure in km^2 yr
        """
        # calculate effective exposure
        self.eff_exposure_grid = np.zeros((self.Nsrcs + 1, self.NRs, self.NBigmfs)) * (
            u.km**2 * u.yr
        )

        # first compute exposure
        self._compute_exposure()

        for id in range(self.Nsrcs + 1):
            if id < self.Nsrcs:  # sources
                Dsrc = self.Dsrcs[id]
                src_uv = self.coords_src[id].cartesian.xyz.value
                print(f"Dsrc = {Dsrc:.2f}")
            elif id == self.Nsrcs:  # background
                src_uv = np.array([1, 0, 0])  # some random unit vector

            for ir in tqdm(
                range(self.NRs),
                desc="Computing effective exposure grid over rigidities: ",
                total=self.NRs,
            ):
                R = self.rigidities_grid[ir]

                # if source model, then iterate for each magnetic field and compute individual kappas
                if id < self.Nsrcs:   
                    for ib, Bigmf in enumerate(self.Bigmf_grid):
                        # kigmf = 10 ** self.f_log10_kappa(
                        #     theta_igmf(R, Bigmf, Dsrc).to_value(u.deg)
                        # )
                        kigmf = 7552 * (theta_igmf(R, Bigmf, Dsrc) / (1 * u.deg)).value**-2
                        kigmf = min(kigmf, kappa_max)

                        self.coords_healpy.representation_type = "cartesian"
                        weighted_map = (
                            self.vMF(
                                self.coords_healpy.cartesian.xyz.value, src_uv, kigmf
                            )
                            * self.delta_ang
                        )
                        weighted_map /= np.sum(
                            weighted_map
                        )  # some numerical error in normalisation, so we force normalisation here

                        # lens the map only if we want to include GMF
                        if self.gmf_model != "None":
                            lensed_map = gmflens.apply_lens_to_map(
                                weighted_map, R.to_value(u.EV)
                            )
                            # compute effective exposure
                            eff_exp = np.dot(self.exposures, lensed_map)
                        else:
                            eff_exp = np.dot(self.exposures, weighted_map)

                        # set some limit incase the effective exposure is so small
                        eff_exp = max(eff_exp, exposure_min * u.km**2 * u.yr)
                        self.eff_exposure_grid[id, ir, ib] = eff_exp

                # background model, we do the same but without Bigmf since we dont have a kappa
                # i.e. we fix kappa = 0
                else:
                    self.coords_healpy.representation_type = "cartesian"
                    weighted_map = (
                        self.vMF(self.coords_healpy.cartesian.xyz.value, src_uv, 0.0)
                        * self.delta_ang
                    )
                    weighted_map /= np.sum(
                        weighted_map
                    )  # some numerical error in normalisation, so we force normalisation here

                    # lens the map only if we want to include GMF
                    if self.gmf_model != "None":
                        lensed_map = gmflens.apply_lens_to_map(
                            weighted_map, R.to_value(u.EV)
                        )
                        # compute effective exposure
                        eff_exp = np.dot(self.exposures, lensed_map)
                    else:
                        eff_exp = np.dot(self.exposures, weighted_map)

                    # compute effective exposure
                    self.eff_exposure_grid[id, ir, :] = eff_exp

    def get_weighted_exposure(self: Self) -> None:
        """Apply weights to the effective exposure to get the weighted exposure."""
        self.wexp_src_grid = np.zeros((self.Nsrcs, self.Nalphas, self.NBigmfs)) * (
            u.km**2 * u.yr
        )
        self.wexp_bg_grid = np.zeros(self.Nalphas) * (u.km**2 * u.yr)

        if self.mass_group != 1:
            # compute the detection threshold CCDF to take into account downscattering of events
            p_rdet = 1 - np.array(
                [
                    norm.cdf(
                        self.data.detector.Rth,
                        loc=R.value,
                        scale=self.data.detector.rigidity_uncertainty * R.value,
                    )
                    for R in self.rigidities_grid
                ]
            )

        for ia in range(self.Nalphas):
            alpha = self.alpha_grid[ia]

            # source case
            for id in range(self.Nsrcs):
                d_idx = np.digitize(
                    self.Dsrcs[id], self.distances_grid, right=True
                )  # get index from distance grid in prince calculation

                for ib in range(self.NBigmfs):
                    if self.mass_group != 1:
                        # integrate over all rigidities including detection effects
                        # TODO: update this function to np.trapezoid for numpy >=2.0
                        self.wexp_src_grid[id, ia, ib] = np.trapz(
                            y=self.arrspects_grid[d_idx, :, ia]
                            * self.eff_exposure_grid[id, :, ib]
                            * p_rdet,
                            x=self.rigidities_grid,
                        )
                    else:
                        # compute expected energy and find index in rigidity grid
                        # corresponding to it (we parametrize energy as rigidity for MG1)
                        Rex = self.Eexs_grid[d_idx, ia]
                        Rex_idx = min(np.digitize(
                            Rex.value, self.rigidities_grid.value, right=False
                        ), self.NRs-1)

                        # weighting factor calculated by analytical integral of source
                        # & arrival distribution, see CM19 for details
                        # TODO: update this for bounded energy spectrum
                        w_factor = (
                            self.Rth_srcs[d_idx].value / self.data.detector.Rth
                        ) ** (1.0 - alpha)

                        self.wexp_src_grid[id, ia, ib] = (
                            self.eff_exposure_grid[id, Rex_idx, ib] * w_factor
                        )

            # background case
            if self.mass_group != 1:
                # integrate over background spectrum w/ detection effects, which is jsut a power law
                bg_spectrum = bounded_power_law(
                    self.rigidities_grid.value,
                    alpha,
                    self.data.detector.Rth,
                    self.data.detector.Rth_max,
                ) * (1 / u.EV)
                self.wexp_bg_grid[ia] = np.trapz(
                    y=bg_spectrum * self.eff_exposure_grid[-1, :, 0] * p_rdet,
                    x=self.rigidities_grid,
                    axis=0,
                )

            else:  # for MG1 we just use the default exposure since rigidity / energy doesnt play a role here
                self.wexp_bg_grid[ia] = (
                    self.data.detector.alpha_T / (4 * np.pi) * (u.km**2 * u.yr)
                )

    def save(self: Self, outfile: str):
        """
        Save tabulated results to h5py File.

        Parameter:
        ----------
        outfile : str
            the path to the output file. must be in .h5 format.
        """
        assert (
            outfile.find(".h5") > 0
        ), f"Output file {outfile} needs to have a .h5 extension."
        with h5py.File(outfile, "a") as f:
            config_label = f"{self.source_type}_{self.detector_type}_mg{self.mass_group}_{self.gmf_model}"
            if config_label in f.keys():
                del f[config_label]
            config_gr = f.create_group(config_label)

            config_gr.create_dataset("Dsrcs", data=self.Dsrcs)
            config_gr.create_dataset("alpha_grid", data=self.alpha_grid)
            config_gr.create_dataset("rigidities_grid", data=self.rigidities_grid)
            config_gr.create_dataset(
                "log10_Bigmf_grid", data=np.log10(self.Bigmf_grid.to_value(u.nG))
            )
            config_gr.create_dataset("distances_grid", data=self.distances_grid)
            config_gr.create_dataset("effective_exposure", data=self.eff_exposure_grid)
            config_gr.create_dataset(
                "log10_wexp_src_grid",
                data=np.log10(self.wexp_src_grid.to_value(u.km**2 * u.yr)),
            )
            config_gr.create_dataset(
                "log10_wexp_bg_grid",
                data=np.log10(self.wexp_bg_grid.to_value(u.km**2 * u.yr)),
            )

    def vMF(self: Self, x: np.array, mu: np.array, kappa: float):
        """
        vMF distribution given mean direction mu, spread parameter kappa
        NB: shape of x must be (N, 3)
        """
        if kappa > 100:
            return np.exp(
                kappa * np.dot(x.T, mu) + np.log(kappa) - np.log(4 * np.pi / 2) - kappa
            )
        elif kappa < 1e-5:  # L'Hopital's rule
            return (
                (1 + kappa * np.dot(x.T, mu))
                / (4 * np.pi * np.cosh(kappa))
                * np.exp(kappa * np.dot(x.T, mu))
            )
        else:
            return (
                kappa / (4 * np.pi * np.sinh(kappa)) * np.exp(kappa * np.dot(x.T, mu))
            )


def theta_igmf(R, Bigmf, D, lc=1):
    """
    Deflection angle for IGMF in degrees

    :param R: rigidity in EV
    :param Bigmf: IGMF magnetic field strength in nG
    :param D: distance of the source in Mpc
    :param lc: coherence length in Mpc (default 1 Mpc)
    """
    return (
        2.3
        * (50 * u.EV / R)
        * (Bigmf / (1 * u.nG))
        * np.sqrt(D / (10 * u.Mpc))
        * np.sqrt(lc)
    ) * u.deg


def bounded_power_law(R, alpha_b, Rmin, Rmax):
    """Background spectrum"""
    if alpha_b != 1.0:
        norm = (1.0 - alpha_b) / (Rmax ** (1.0 - alpha_b) - Rmin ** (1.0 - alpha_b))
    else:
        norm = 1.0 / (np.log(Rmax) - np.log(Rmin))

    return norm * R ** (-alpha_b)
