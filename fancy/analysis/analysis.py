"""Container to manage the inputs and outputs of the fits."""

import pickle
from typing import Union

import cmdstanpy
import h5py
import numpy as np
from typing_extensions import Self  # change to typing for py>3.11

from fancy.interfaces.data import Data
from fancy.interfaces.model import Model
from fancy.utils.package_data import get_path_to_kappa_theta


class Analysis:
    """Container to manage the inputs and outputs of the fits."""

    # pre-defined analysis types
    arr_dir_type = "arrival_direction"
    joint_type = "joint"
    gmf_type = "joint_gmf"
    composition_type = "joint_composition"
    gmf_composition_type = "joint_gmf_composition"

    def __init__(
        self: Self,
        data: Data,
        model: Model,
        analysis_type: str = "joint_gmf_composition",
    ) -> None:
        """
        Container to manage the inputs and outputs of the fits.

        data: fancy.interfaces.data.Data
            Container that handles the source, uhecr, and detector information.
            All such information should already be initialised (see relevant class for
            more information.)
        model: fancy.interfaces.model.Model
            Container that handles the stan Model
        analysis_type: str, default=joint_gmf_composition
            The analysis type to consider.
        """
        self.data = data
        self.model = model
        self.analysis_type = analysis_type

        self.fit_input = None
        self.fit = None

    def use_tables(
        self: Self,
        exposure_table_file: str,
        energy_table_file: str,
        gmf_model: str = "None",
        kappa_theta_filename: str = "kappa_theta_map.pkl",
    ) -> None:
        """
        Pass in effective exposure & energy loss tables that have been pre-generated.

        Parameter:
        ---------
        exposure_table_file : str
            The table containing the information about the effective exposure
            of each configuration (source, detector, MG).
        energy_table_file : str
            The table containing information of the energy loss processes for
            each configuration (detector, MG)
        gmf_model : str, default=None
            The GMF model considered in the analysis.
            Used to read in the effective exposure information.
            Default is None, which considers no deflections from GMF
        kappa_theta_filename : str, default='kappa_theta_map.pkl'
            the file name for the map that converts the vMF parameter `kappa`
            to the RMS angular scale.
            Default is the default name saved in `fancy.utils.resources`
        """
        if self.analysis_type in set(
            [self.composition_type, self.gmf_composition_type]
        ):
            """Read from energy loss tables"""
            with h5py.File(energy_table_file, "r") as file:
                config_label = (
                    f"{self.data.detector.label}_mg{self.data.detector.mass_group}"
                )

                self.distances_grid = file[config_label]["distances_grid"][()]  # Mpc
                self.alpha_grid = file[config_label]["alpha_grid"][()]
                self.log10_Rgrid = file[config_label]["log10_rigidities"][()]
                self.log10_Eexs_grid = file[config_label]["log10_Eexs_grid"][
                    ()
                ]  # log10(EeV)

                if self.data.detector.mass_group != 1:
                    self.log10_arr_spect_grid = file[config_label][
                        "log10_arrspect_grid"
                    ][()]  # log10(1/EV)
                else:
                    self.Rarr_grid = file[config_label]["Rarr_grid"][()]  # log10(1/EV)

            """Read from exposure table"""
            with h5py.File(exposure_table_file, "r") as file:
                config_label = f"{self.data.source.label}_{self.data.detector.label}_mg{self.data.detector.mass_group}_{gmf_model}"
                self.log10_Bigmf_grid = file[config_label]["log10_Bigmf_grid"][
                    ()
                ]  # log10(nG)
                self.log10_wexp_src_grid = file[config_label]["log10_wexp_src_grid"][
                    ()
                ]  # km^2 yr
                self.log10_wexp_bg_grid = file[config_label]["log10_wexp_bg_grid"][
                    ()
                ]  # km^2 yr

            """Read from kappa_theta map"""
            kappa_theta_file = str(get_path_to_kappa_theta(kappa_theta_filename))
            (self.thetas_interp_arr, self.log10_kappas_interp_arr, _) = pickle.load(
                open(kappa_theta_file, "rb")
            )
        else:
            raise DeprecationWarning(
                f"Handles for analysis type {self.analysis_type} is deprecated."
            )

    def _prepare_fit_inputs(self: Self) -> None:
        """Gather inputs from Model, Data and IntegrationTables."""

        # prepare fit inputs
        self.fit_input = {
            "Ns": self.data.source.N,
            "varpi": self.data.source.coord.cartesian.xyz.value.T,
            "D": self.data.source.distance,
            "N": self.data.uhecr.N,
            "A": self.data.uhecr.A,
            "zenith_angle": self.data.uhecr.zenith_angle,
            "alpha_T": self.data.detector.alpha_T,
        }

        if self.analysis_type in set(
            [self.composition_type, self.gmf_composition_type]
        ):
            # arrival direction parameters
            if self.analysis_type == self.gmf_composition_type:  # coordinates at GB
                self.fit_input["arrival_direction"] = self.data.uhecr.unit_vector_gb
                self.fit_input["kappa_ds"] = self.data.uhecr.kappa_gmfs
            elif self.analysis_type == self.composition_type:  # coordinates at Earth
                self.fit_input["arrival_direction"] = self.data.uhecr.unit_vector
                self.fit_input["kappa_ds"] = np.full(
                    self.data.uhecr.N, self.data.detector.kappa_d
                )

            # direction parameters
            self.fit_input["Nkappas"] = len(self.log10_kappas_interp_arr)
            self.fit_input["log10_kappas_grid"] = self.log10_kappas_interp_arr
            self.fit_input["Nthetas"] = len(self.thetas_interp_arr)
            self.fit_input["thetas_grid"] = self.thetas_interp_arr

            # UHECR parameters
            # if we find rigidity in dataset, then use that, otherwise divide by meanZ of mass group
            if len(self.data.uhecr.rigidity) > 0:
                print("Using available rigidity data for analysis.")
                self.fit_input["Rdet"] = self.data.uhecr.rigidity
            else:
                print(f"Dividing energy data by mean charge {self.data.detector.meanZ} for rigidity.")
                self.fit_input["Rdet"] = (
                    self.data.uhecr.energy / self.data.detector.meanZ
                )
            self.fit_input["exp_factors"] = self.data.uhecr.exposure

            # detector parameters
            self.fit_input["Rth"] = self.data.detector.Rth
            self.fit_input["Rth_max"] = self.data.detector.Rth_max
            self.fit_input["Rerr"] = self.data.detector.rigidity_uncertainty

            # arrival spectrum parameters
            self.fit_input["Nds"] = len(self.distances_grid)
            self.fit_input["NRs"] = len(self.log10_Rgrid)
            self.fit_input["Nalphas"] = len(self.alpha_grid)
            self.fit_input["distances_grid"] = self.distances_grid
            self.fit_input["log10_Rgrid"] = self.log10_Rgrid
            self.fit_input["alpha_grid"] = self.alpha_grid

            if self.data.detector.mass_group != 1:
                self.fit_input["log_arr_spectrum_grid"] = np.log(
                    10.0**self.log10_arr_spect_grid
                )
            else:
                self.fit_input["Rarr_grid"] = self.Rarr_grid

            # Nex / flux parameters
            self.fit_input["log10_Eexs_grid"] = self.log10_Eexs_grid
            self.fit_input["NBigmfs"] = len(self.log10_Bigmf_grid)
            self.fit_input["log10_Bigmf_grid"] = self.log10_Bigmf_grid
            self.fit_input["log10_wexp_src_grid"] = self.log10_wexp_src_grid
            self.fit_input["log10_wexp_bg_grid"] = self.log10_wexp_bg_grid

        else:
            raise DeprecationWarning(
                f"Handles for analysis type {self.analysis_type} is deprecated."
            )

    def fit_model(
        self: Self,
        iterations: int = 1000,
        chains: int = 4,
        seed: Union[int, None] = None,
        warmup: Union[int, None] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> cmdstanpy.stanfit.mcmc.CmdStanMCMC:
        """
        Fit a model.

        Parameter:
        ----------
        iterations: int, default=1000
            number of iterations
        chains: int, default=4
            number of chains
        seed: int, default=None
            seed for RNG
        output_dir: str, default=None
            output directory for raw stan outputs
        warmup : int, default=None
            number of iterations used for warmup
        show_progress: bool, default=True
            to show the progress of the fits in a tqdm progress
            bar or not.

        Return:
        -------
        fit : cmdstanpy.stanfit.mcmc.CmdStanMCMC
            The fit output from stan that contains the samples
            as well as other diagnostic information.

            See https://cmdstanpy.readthedocs.io/en/v1.2.0/api.html#cmdstanpy.CmdStanMCMC
            for more information on what can be accessed

        Additional arguments that match the keyword arguments
        in cmdstanpy.model.model.sample can also be passed.
        See https://cmdstanpy.readthedocs.io/en/v1.2.0/api.html#cmdstanpy.CmdStanModel.sample
        for more details.
        """
        # Prepare fit inputs
        self._prepare_fit_inputs()

        # fit
        print("Performing fitting...")
        self.fit = self.model.model.sample(
            data=self.fit_input,
            iter_sampling=iterations,
            chains=chains,
            seed=seed,
            show_progress=show_progress,
            iter_warmup=warmup,
            **kwargs,
        )

        # Diagnositics
        print("Checking all diagnostics...")
        print(self.fit.diagnose())

        self.chain = self.fit.stan_variables()
        print("Done!")
        return self.fit

    def save(self: Self, outfile: str) -> None:
        """
        Write the analysis output to an output file.

        Parameter:
        ----------
        outfile : str
            the path to the output file where the
            analysis outputs are to be stored.
            Must be a .h5 format.
        """
        # ensure that the file is a h5 format
        assert outfile.find(".h5"), f"Output file {outfile} must have a .h5 extension!"

        with h5py.File(outfile, "w") as f:
            source_handle = f.create_group("source")
            if self.data.source:
                self.data.source.save(source_handle)

            uhecr_handle = f.create_group("uhecr")
            if self.data.uhecr:
                self.data.uhecr.save(uhecr_handle, self.analysis_type)

            detector_handle = f.create_group("detector")
            if self.data.detector:
                self.data.detector.save(detector_handle)

            model_handle = f.create_group("model")
            if self.model:
                self.model.save(model_handle)

            if self.fit is None:
                raise ValueError("Run `fit_model` first!")
            fit_handle = f.create_group("fit")
            # fit inputs
            fit_input_handle = fit_handle.create_group("input")
            for key, value in self.fit_input.items():
                fit_input_handle.create_dataset(key, data=value)

            # samples
            samples = fit_handle.create_group("samples")
            for key, value in self.chain.items():
                samples.create_dataset(key, data=value)

            # log posterior
            samples.create_dataset("log_post", data=self.fit.method_variables()["lp__"])


    # KW: 12.06.24: I have shifted the table calculation to EffectiveExposure & EnergyLoss modules since I want to make the Analysis object only used for performing fits and not as a general container.
    # Similar for the simulation, however I do not port this to the Simulation module since this requires the use of stan when there are more simpler ways to forwards simulate the events.
    # I comment out additional routines that are not used at the moment as well.
    #
    # def build_tables(
    #     self, num_points=100, sim_only=False, fit_only=False, parallel=True, nthreads=int(cpu_count() * 0.75), composition_file=None,
    # ):
    #     """
    #     Build the necessary integral tables.
    #     """

    #     if sim_only:

    #         # kappa_true table for simulation
    #         if (
    #             self.analysis_type == self.arr_dir_type
    #             or self.analysis_type == self.E_loss_type
    #         ):
    #             kappa_true = self.model.kappa
    #             D_src = self.data.source.distance

    #         if self.analysis_type == self.joint_type:
    #             D_src = self.data.source.distance
    #             self.Eex = self.energy_loss.get_Eex(self.Eth_src, self.model.alpha)
    #             self.kappa_ex = self.energy_loss.get_kappa_ex(
    #                 self.Eex, self.model.B, D_src
    #             )
    #             kappa_true = self.kappa_ex

    #         if self.analysis_type == self.gmf_type:

    #             A, Z = self.nuc_table[self.model.ptype]

    #             # shift by 0.02 to get kappa_ex at g.b.
    #             D_src = self.data.source.distance - 0.02
    #             self.Eex = self.energy_loss.get_Eex(self.Eth_src, self.model.alpha)

    #             self.kappa_ex = self.energy_loss.get_kappa_ex(
    #                 self.Eex, self.model.B, D_src
    #             )

    #             # Find kappa_gmf for each source via lensing
    #             varpi = self.data.source.unit_vector
    #             kappa_gmf = []
    #             for v, k, e in zip(varpi, self.kappa_ex, self.Eex):
    #                 k_gmf = self.gmf_deflections.get_kappa_gmf_per_source(v, k, e, A, Z)
    #                 kappa_gmf.append(k_gmf)

    #             kappa_true = np.array(kappa_gmf)

    #         if parallel:
    #             self.tables.build_for_sim_parallel(
    #                 kappa_true, self.model.alpha, self.model.B, D_src, nthreads
    #             )
    #         else:
    #             self.tables.build_for_sim(
    #                 kappa_true, self.model.alpha, self.model.B, D_src
    #             )

    #     if fit_only:

    #         # logarithmically spcaed array with 60% of points between KAPPA_MIN and 100
    #         # kappa_first = np.logspace(
    #         #     np.log(1), np.log(10), int(num_points * 0.7), base=np.e
    #         # )
    #         # kappa_second = np.logspace(
    #         #     np.log(10), np.log(100), int(num_points * 0.2) + 1, base=np.e
    #         # )
    #         # kappa_third = np.logspace(
    #         #     np.log(100), np.log(1000), int(num_points * 0.1) + 1, base=np.e
    #         # )
    #         # kappa = np.concatenate(
    #         #     (kappa_first, kappa_second[1:], kappa_third[1:]), axis=0
    #         # )
    #         kappa = np.logspace(0, 5, num_points)

    #         # full table for fit
    #         if self.analysis_type == self.gmf_type:

    #             kappa_gmf = []
    #             alpha_approx = 2.5

    #             # in the future, read Rex from the files
    #             Eex = 2 ** (1 / (alpha_approx - 1)) * self.model.Eth

    #             A, Z = self.nuc_table[self.model.ptype]

    #             if parallel:
    #                 self.tables.build_for_fit_parallel_gmf(
    #                     kappa, Eex, A, Z, self.gmf_deflections, nthreads
    #                 )
    #             else:
    #                 self.tables.build_for_fit_gmf(
    #                     kappa, Eex, A, Z, self.gmf_deflections
    #                 )

    #         elif self.analysis_type == self.composition_type:
    #             kappa_gmf = []

    #             pass

    #         else:
    #             if parallel:
    #                 self.tables.build_for_fit_parallel(kappa, nthreads)
    #             else:
    #                 self.tables.build_for_fit(kappa)

    # def build_energy_table(
    #     self,
    #     num_points=50,
    #     table_file=None,
    #     parallel=True,
    #     nthreads=int(cpu_count() * 0.75)
    # ):
    #     """
    #     Build the energy interpolation tables.
    #     """

    #     self.E_grid = np.logspace(
    #         np.log(self.model.Eth), np.log(1.0e4), num_points, base=np.e
    #     )
    #     self.Earr_grid = []

    #     if parallel and not isinstance(self.energy_loss, CRPropaApproxEnergyLoss) and not np.isscalar(self.data.source.distance):

    #         args_list = [(self.E_grid, d) for d in self.data.source.distance]
    #         # parallelize for each source distance
    #         with Pool(nthreads) as mpool:
    #             results = list(
    #                 progress_bar(
    #                     mpool.imap(self.energy_loss.get_arrival_energy_vec, args_list),
    #                     total=len(args_list),
    #                     desc="Precomputing energy grids",
    #                 )
    #             )

    #             self.Earr_grid = results

    #     else:
    #         for i in progress_bar(
    #             range(len(self.data.source.distance)), desc="Precomputing energy grids"
    #         ):
    #             d = self.data.source.distance[i]
    #             self.Earr_grid.append(
    #                 [self.energy_loss.get_arrival_energy(e, d) for e in self.E_grid]
    #             )

    #     if table_file:
    #         with h5py.File(table_file, "a") as f:
    #             E_group = f.create_group("energy")
    #             E_group.create_dataset("E_grid", data=self.E_grid)
    #             E_group.create_dataset("Earr_grid", data=self.Earr_grid)

    # def plot(self, type=None, cmap=None):
    #     """
    #     Plot the data associated with the analysis object.

    #     type == 'arrival direction':
    #     Plot the arrival directions on a skymap,
    #     with a colour scale describing which source
    #     the UHECR is from.

    #     type == 'energy'
    #     Plot the simulated energy spectrum from the
    #     source, to after propagation (arrival) and
    #     detection
    #     """

    #     # plot style
    #     if cmap == None:
    #         cmap = plt.cm.get_cmap("viridis")

    #     # plot arrival directions by default
    #     if type == None:
    #         type == "arrival_direction"

    #     if type == "arrival_direction":

    #         # figure
    #         fig, ax = plt.subplots()
    #         fig.set_size_inches((12, 6))

    #         # skymap
    #         skymap = AllSkyMap()

    #         self.data.source.plot(skymap)
    #         self.data.detector.draw_exposure_lim(skymap)
    #         self.data.uhecr.plot(skymap)

    #         # standard labels and background
    #         skymap.draw_standard_labels()

    #         # legend
    #         ax.legend(frameon=False, bbox_to_anchor=(0.85, 0.85))

    #     if type == "energy":

    #         bins = np.logspace(np.log(self.model.Eth), np.log(1e4), base=np.e)

    #         fig, ax = plt.subplots()

    #         if isinstance(self.E, (list, np.ndarray)):
    #             ax.hist(
    #                 self.E, bins=bins, alpha=0.7, label=r"$\tilde{E}$", color=cmap(0.0)
    #             )
    #         if isinstance(self.Earr, (list, np.ndarray)):
    #             ax.hist(self.Earr, bins=bins, alpha=0.7, label=r"$E$", color=cmap(0.5))

    #         ax.hist(
    #             self.data.uhecr.energy,
    #             bins=bins,
    #             alpha=0.7,
    #             label=r"$\hat{E}$",
    #             color=cmap(1.0),
    #         )

    #         ax.set_xscale("log")
    #         ax.set_yscale("log")
    #         ax.legend(frameon=False)

    # def use_crpropa_data(self, energy, unit_vector):
    #     """
    #     Build fit inputs from the UHECR dataset.
    #     """

    #     self.N = len(energy)
    #     self.arrival_direction = Direction(unit_vector)

    #     # simulate the zenith angles
    #     print("Simulating zenith angles...")
    #     self.zenith_angles = self._simulate_zenith_angles()
    #     print("Done!")

    #     # Make Uhecr object
    #     uhecr_properties = {}
    #     uhecr_properties["label"] = "sim_uhecr"
    #     uhecr_properties["N"] = self.N
    #     uhecr_properties["unit_vector"] = self.arrival_direction.unit_vector
    #     uhecr_properties["energy"] = energy
    #     uhecr_properties["zenith_angle"] = self.zenith_angles
    #     uhecr_properties["A"] = np.tile(self.data.detector.area, self.N)

    #     new_uhecr = Uhecr()
    #     new_uhecr.from_properties(uhecr_properties)

    #     self.data.uhecr = new_uhecr

    # def _get_zenith_angle(self, c_icrs, loc, time):
    #     """
    #     Calculate the zenith angle of a known point
    #     in ICRS (equatorial coords) for a given
    #     location and time.
    #     """
    #     c_altaz = c_icrs.transform_to(AltAz(obstime=time, location=loc))
    #     return np.pi / 2 - c_altaz.alt.rad

    # def _simulate_zenith_angles(self, start_year=2004):
    #     """
    #     Simulate zenith angles for a set of arrival_directions.

    #     :params: start_year: year in which measurements started.
    #     """

    #     if len(self.arrival_direction.d.icrs) == 1:
    #         c_icrs = self.arrival_direction.d.icrs[0]
    #     else:
    #         c_icrs = self.arrival_direction.d.icrs

    #     time = []
    #     zenith_angles = []
    #     stuck = []

    #     j = 0
    #     first = True
    #     for d in c_icrs:
    #         za = 99
    #         i = 0
    #         while za > self.data.detector.threshold_zenith_angle.rad:
    #             dt = np.random.exponential(1 / self.N)
    #             if first:
    #                 t = start_year + dt
    #             else:
    #                 t = time[-1] + dt
    #             tdy = Time(t, format="decimalyear")
    #             za = self._get_zenith_angle(d, self.data.detector.location, tdy)

    #             i += 1
    #             if i > 100:
    #                 za = self.data.detector.threshold_zenith_angle.rad
    #                 stuck.append(1)
    #         time.append(t)
    #         first = False
    #         zenith_angles.append(za)
    #         j += 1
    #         # print(j , za)

    #     if len(stuck) > 1:
    #         print(
    #             "Warning: % of zenith angles stuck is",
    #             len(stuck) / len(zenith_angles) * 100,
    #         )

    #     return zenith_angles

    # def simulate(self, seed=None, Eth_sim=None):
    #     """
    #     Run a simulation.

    #     :param seed: seed for RNG
    #     :param Eth_sim: the minimun energy simulated
    #     :param gmf: enable galactic magnetic field deflections
    #     """

    #     eps = self.tables.sim_table

    #     # handle selected sources
    #     if self.data.source.N < len(eps):
    #         eps = [eps[i] for i in self.data.source.selection]

    #     # convert scale for sampling
    #     D = self.data.source.distance
    #     alpha_T = self.data.detector.alpha_T
    #     Q = self.model.Q
    #     F0 = self.model.F0
    #     D, alpha_T, eps, F0, Q = convert_scale(D, alpha_T, eps, F0, Q)

    #     if (
    #         self.analysis_type == self.joint_type
    #         or self.analysis_type == self.E_loss_type
    #         or self.analysis_type == self.gmf_type
    #     ):
    #         # find lower energy threshold for the simulation, given Eth and Eerr
    #         if Eth_sim:
    #             self.model.Eth_sim = Eth_sim

    #     # compile inputs from Model and Data
    #     self.simulation_input = {
    #         "kappa_d": self.data.detector.kappa_d,
    #         "Ns": len(self.data.source.distance),
    #         "varpi": self.data.source.unit_vector,
    #         "D": D,
    #         "A": self.data.detector.area,
    #         "a0": self.data.detector.location.lat.rad,
    #         "lon": self.data.detector.location.lon.rad,
    #         "theta_m": self.data.detector.threshold_zenith_angle.rad,
    #         "alpha_T": alpha_T,
    #         "eps": eps,
    #     }

    #     self.simulation_input["Q"] = Q
    #     self.simulation_input["F0"] = F0
    #     self.simulation_input["distance"] = self.data.source.distance

    #     if (
    #         self.analysis_type == self.arr_dir_type
    #         or self.analysis_type == self.E_loss_type
    #     ):

    #         self.simulation_input["kappa"] = self.model.kappa

    #     if self.analysis_type == self.E_loss_type:

    #         self.simulation_input["alpha"] = self.model.alpha
    #         self.simulation_input["Eth"] = self.model.Eth_sim
    #         self.simulation_input["Eerr"] = self.data.detector.energy_uncertainty

    #     if self.analysis_type == self.joint_type or self.analysis_type == self.gmf_type:

    #         self.simulation_input["B"] = self.model.B
    #         self.simulation_input["alpha"] = self.model.alpha
    #         self.simulation_input["Eth"] = self.model.Eth_sim
    #         self.simulation_input["Eerr"] = self.data.detector.energy_uncertainty

    #         # get particle type we intialize simulation with
    #         _, Z = self.nuc_table[self.model.ptype]
    #         self.simulation_input["Z"] = Z

    #     try:
    #         if self.data.source.flux:
    #             self.simulation_input["flux"] = self.data.source.flux
    #         else:
    #             self.simulation_input["flux"] = np.zeros(self.data.source.N)
    #     except:
    #         self.simulation_input["flux"] = np.zeros(self.data.source.N)

    #     # run simulation
    #     print("Running Stan simulation...")
    #     self.simulation = self.model.simulation.sample(
    #         data=self.simulation_input,
    #         iter_sampling=1,
    #         chains=1,
    #         fixed_param=True,
    #         seed=seed,
    #     )

    #     # extract output
    #     print("Extracting output...")

    #     self.Nex_sim = self.simulation.stan_variable("Nex_sim")[0]
    #     # source_labels: to which source label each UHECR is associated with
    #     self.source_labels = (self.simulation.stan_variable("lambda")[0]).astype(int)

    #     if self.analysis_type == self.arr_dir_type:
    #         arrival_direction = self.simulation.stan_variable("arrival_direction")[0]

    #     elif (
    #         self.analysis_type == self.joint_type
    #         or self.analysis_type == self.E_loss_type
    #         or self.analysis_type == self.gmf_type
    #     ):

    #         self.Earr = self.simulation.stan_variable("Earr")[0]  # arrival energy
    #         self.E = self.simulation.stan_variable("E")[0]  # sampled from spectrum

    #         # simulate with deflections with GMF
    #         if self.analysis_type == self.gmf_type:
    #             kappas = self.simulation.stan_variable("kappa")[0]
    #             print("Simulating deflections...")
    #             arrival_direction, self.Edet = self._simulate_deflections(kappas)

    #         else:
    #             arrival_direction = self.simulation.stan_variable("arrival_direction")[
    #                 0
    #             ]

    #             self.Edet = self.simulation.stan_variable("Edet")[0]

    #             # make cut on Eth
    #             inds = np.where(self.Edet >= self.model.Eth)
    #             self.Edet = self.Edet[inds]
    #             arrival_direction = arrival_direction[inds]
    #         # self.source_labels = self.source_labels[inds]

    #     # convert to Direction object
    #     self.arrival_direction = Direction(arrival_direction)
    #     self.N = len(self.arrival_direction.unit_vector)

    #     # simulate the zenith angles
    #     print("Simulating zenith angles...")
    #     self.zenith_angles = self._simulate_zenith_angles(self.data.detector.start_year)

    #     # Make uhecr object
    #     uhecr_properties = {}
    #     uhecr_properties["label"] = self.data.detector.label
    #     uhecr_properties["N"] = self.N
    #     uhecr_properties["unit_vector"] = self.arrival_direction.unit_vector
    #     uhecr_properties["energy"] = self.Edet
    #     uhecr_properties["zenith_angle"] = self.zenith_angles
    #     uhecr_properties["A"] = np.tile(self.data.detector.area, self.N)
    #     # uhecr_properties['source_labels'] = self.source_labels

    #     uhecr_properties["ptype"] = (
    #         self.model.ptype if self.analysis_type == self.gmf_type else "p"
    #     )

    #     new_uhecr = Uhecr()
    #     new_uhecr.from_simulation(uhecr_properties)

    #     # evaluate kappa_gmf manually here
    #     if self.analysis_type == self.gmf_type:
    #         print("Computing kappa_gmf...")
    #         (
    #             new_uhecr.kappa_gmf,
    #             omega_defl_kappa_gmf,
    #             omega_rand_kappa_gmf,
    #             _,
    #         ) = new_uhecr.eval_kappa_gmf(
    #             particle_type=self.model.ptype, Nrand=100, gmf="JF12", plot=False
    #         )

    #         self.defl_plotvars.update(
    #             {
    #                 "omega_rand_kappa_gmf": omega_rand_kappa_gmf,
    #                 "omega_defl_kappa_gmf": omega_defl_kappa_gmf,
    #                 "kappa_gmf": new_uhecr.kappa_gmf,
    #             }
    #         )

    #     self.data.uhecr = new_uhecr

    #     print("Done!")
