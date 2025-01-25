"""Container for composition weights"""

import os
import pickle as pickle

import astropy.units as u
import h5py
import numpy as np
from astropy.cosmology import WMAP9, Planck18, z_at_value
from joblib import Parallel, delayed
from scipy.interpolate import UnivariateSpline
from scipy.optimize import Bounds, minimize
from tqdm import tqdm

try:
    import prince_cr as pcr
    import prince_cr.config
    from prince_cr import core, cross_sections, photonfields
    from prince_cr import util as pru
    from prince_cr.cr_sources import CosmicRaySource
    from prince_cr.solvers import UHECRPropagationSolverBDF
except ImportError:
    pcr = None


class CompositionMatrixContainer:
    """
    Container to handle all composition loss computation performed by Prince. In principle this only needs to be accessed if one wants to re-compute the composition weights.
    """

    prince_cr.config.x_cut = 1e-4
    prince_cr.config.x_cut_proton = 1e-2
    prince_cr.config.tau_dec_threshold = np.inf
    prince_cr.config.linear_algebra_backend = "MKL"
    prince_cr.config.secondaries = False
    prince_cr.config.ignore_particles = [
        0,
        11,
        12,
        13,
        14,
        15,
        16,
        20,
        21,
    ]
    prince_cr.config.cosmic_ray_grid = (8, 12, 40)

    # photon fields, combined CMB & EBL from Gilmore
    pf_gilmore = photonfields.CombinedPhotonField(
        [photonfields.CMBPhotonSpectrum, photonfields.CIBGilmore2D]
    )

    # ID of nuclei used for injection, use all available mass ids possible
    massids = [
        101,
        201,
        301,
        302,
        402,
        603,
        703,
        704,
        904,
        1004,
        1005,
        1105,
        1206,
        1306,
        1406,
        1407,
        1507,
        1608,
        1708,
        1808,
        1909,
        2010,
        2110,
        2210,
        2211,
        2311,
        2412,
        2512,
        2612,
        2613,
        2713,
        2814,
        2914,
        3014,
        3115,
        3216,
        3316,
        3416,
        3516,
        3617,
        3718,
        3818,
        3919,
        4020,
        4120,
        4220,
        4320,
        4422,
        4521,
        4622,
        4722,
        4822,
        4923,
        5024,
        5124,
        5224,
        5325,
        5426,
        5526,
        5626,
    ]

    def __init__(
        self,
        css="PSB",
        dmin=0.8,
        dmax=110,
        Nds=100,
        resources_path: str = os.path.dirname(os.path.realpath(__file__)),
        nthreads=8,
    ):
        """
        Container to handle all composition loss computation performed by Prince. In principle this only needs to be accessed if one wants to re-compute the composition weights.

        :param css: cross section model used for prince computation. Valid models are ["TALYS", "PSB"]
        :param dmin,dmax,Nds: minimum / maximum distance and density for distance grid in Mpc
        :param resources_path: path where resources (kernel, redshift_distance interpolator) is stored
        :param nthreads: number of threads used for solver
        """
        self.css = css
        self.dmax = dmax
        self.distances = np.linspace(dmin, dmax, Nds)
        self.resources_path = resources_path

        if pcr == None:
            raise ImportError("Prince-CR needs to be installed for using this module!")

        # create directory if it doesnt exist yet
        if not os.path.exists(self.resources_path):
            os.mkdir(self.resources_path)

        # set mkl threads
        prince_cr.config.set_mkl_threads(nthreads)

        # get the kernel
        self.prince_run_datapath = os.path.join(
            self.resources_path, f"prince_run_{css}_mkl.ppo"
        )

        if not os.path.exists(self.prince_run_datapath):
            print("Pre-computing kernel")
            self.__create_kernel()

        self.prince_run = pickle.load(open(self.prince_run_datapath, "rb"))

        # similarly get the converstion from z <-> d
        self.redshift_distance_datapath = os.path.join(
            self.resources_path, "redshift_tables_Planck_WMAP.pkl"
        )

        if not os.path.exists(self.redshift_distance_datapath):
            print("Pre-computing redshift <-> distance table")
            self.__create_distance_tables()

        (_, self.spl_d_to_z_plk, _, _) = pickle.load(
            open(self.redshift_distance_datapath, "rb")
        )

        # initialise solver once to get utility functions to compute A and Z from massIDs
        solv = UHECRPropagationSolverBDF(
            initial_z=1.0, final_z=0.0, prince_run=self.prince_run
        )
        self.fA = lambda x: solv.spec_man.ncoid2sref[x].A
        self.fZ = lambda x: solv.spec_man.ncoid2sref[x].Z

        # A and Z
        self.As = np.array([self.fA(massid) for massid in self.massids])
        self.Zs = np.array([self.fZ(massid) for massid in self.massids])

    def run_injection_solver(self, reset=False):
        """Run injection solver. It will try to find the `sol_injection_solver.pkl` and load from it. Otherwise it will compute it.

        :param reset: to reset the pre-computation or not
        """
        solver_res_path = os.path.join(self.resources_path, "injection_solver")
        if not os.path.exists(solver_res_path):
            os.mkdir(solver_res_path)

        solver_res_files = [
            os.path.join(solver_res_path, f"sol_injection_solver_D{dinit:.2f}.pkl")
            for dinit in self.distances
        ]

        if reset:
            print("Computing the injection solver")

            # create linearly spaced grid for distances
            redshifts = self.spl_d_to_z_plk(self.distances)

            for i, dinit in enumerate(self.distances):
                # there is still some memory issue that needs to be solved over here...
                # very temporary and bad hack below for now
                # if dinit < 86:
                #     continue

                # initialise theinjection solver helper
                solver = InjectionSolverRunContainer(
                    self.distances[i], redshifts[i], self.massids, self.prince_run
                )
                solver_res_single = solver.run()
                # write to pickle file
                pickle.dump(
                    solver_res_single, open(solver_res_files[i], "wb"), protocol=-1
                )

                # delete the solver helper to avoid any memory issues
                del solver

            # currently parallelising over distances is not working too well, maybe because its contained within a class...
            # but using many threads can be as fast as parallelising over this so in principle its not necessary
            # solver_res = Parallel(n_jobs=njobs)(delayed(self.run_single_injection_solver)(arg) for arg in run_args)

        else:
            print(
                "Using pre-computed solver results. Set reset == True to re-run the injection solver."
            )
            assert all(
                [os.path.exists(f) for f in solver_res_files]
            ), "Files dont exist. Please run injection solver with reset=True."

        solver_res = [pickle.load(open(f, "rb")) for f in solver_res_files]
        # solver_res = []

        return solver_res

    def determine_mass_groups(self):
        """Determine the masses within each mass group"""

        # first define the masses within each mass group
        mass_groups = [1, 2, 3, 4]
        mass_group_ids = []
        self.mass_group_idxlims = []

        for lnA_lower, lnA_upper in [(0, 1), (1, 2), (2, 3), (3, 4)]:
            id_per_mg = []

            for im, massid in enumerate(self.massids):
                # compute A
                lnA = np.log(self.fA(massid))

                # geeq and lessthan sign as with Dembinski+2017
                if lnA >= lnA_lower and lnA < lnA_upper:
                    id_per_mg.append(massid)

            mass_group_ids.append(id_per_mg)

            # also calcualte the lower and upper limit in massid array
            # for computation convenience later
            mg_lower_idx = np.digitize(id_per_mg[0], self.massids, right=True)
            mg_upper_idx = np.digitize(id_per_mg[-1], self.massids, right=True)

            self.mass_group_idxlims.append([mg_lower_idx, mg_upper_idx])

            print(f"Masses (A) contained in mass group {lnA_upper}: ")
            print([self.fA(mid) for mid in id_per_mg])

    def compute_weights(self, solver_res):
        """
        Compute composition weights, both the full version and also summed over each mass group

        :param solver_res: results from the solver
        """

        # first determine the mass groups
        self.determine_mass_groups()

        # define rigidity grid number here
        NRs = len(self.prince_run.cr_grid.grid)

        # propagation matrix resulting from propagation
        # this is defined for each arrival mass as well
        # shape is DIS x ASRC x AEARTH x RIGIDITY
        self.propa_matrix = np.zeros(
            (
                len(self.distances),
                len(self.massids),
                len(self.massids),
                NRs,
            )
        )

        # efficiency matrix for injection
        # shape is DIS x ASRC x MG x RIGIDITY
        self.inj_eff_matrix = np.zeros(
            (
                len(self.distances),
                len(self.massids),
                4,
                NRs,
            )
        )

        # iterate for each distance & source mass
        for id in tqdm(range(len(self.distances)), desc="Iterating over all distances: ", total=len(self.distances)):
            solver_res_per_d = solver_res[id]

             # temporary arrays to store results per distance bin
            earth_spects_mg = np.zeros(
                (len(self.massids), NRs, 4)
            )
            src_spects = np.zeros(
                (len(self.massids), NRs)
            )

            for img in range(4):

                mg_lidx, mg_uidx = self.mass_group_idxlims[img]

                # computing the source and earth spectrum from prince 
                for ims in range(len(self.massids)):
                    # ignore all source compositions below the minimum
                    # from the mass group
                    if ims < mg_lidx:
                        continue

                    res, src_spect = solver_res_per_d[ims]

                    # get the earth spectrum.
                    # we still store it for all arrival masses to
                    # verify if our production efficiency produces 
                    # MG3 particles well later
                    earth_spect_tmp = np.zeros(NRs)
                    for ime, mid in enumerate(self.massids):
                        self.propa_matrix[id, ims, ime, :] = res.get_solution(mid)[1] / src_spect

                        # evaluate the earth spectrum only within the mass group
                        if ime >= mg_lidx and ime <= mg_uidx:
                            earth_spect_tmp += res.get_solution(mid)[1]

                    # shifted index since the spectra are stored with [mg_uidx:]
                    earth_spects_mg[ims, :, img] = earth_spect_tmp
                    src_spects[ims, :] = src_spect

            for img in range(4):

                mg_lidx, mg_uidx = self.mass_group_idxlims[img]

                # yield the injection efficiency factors per rigidity bin per distance bin
                # store into full injection efficiency matrix, which is defined for 
                # all source masses (==0 where unphysical)
                self.inj_eff_matrix[id, mg_lidx:, img, :] = np.array([
                    minimize(
                        cost_function,
                        x0=np.ones(len(self.massids[mg_lidx:])),
                        args=(earth_spects_mg[mg_lidx:, iR,:], src_spects[mg_lidx:, iR], img),
                        bounds=Bounds(0,1),
                    ).x
                    for iR in range(NRs)
                ]).T

    def save(self, outfile : str):
        """Save data into h5py format"""

        # compute rigidity grid, assuming constant mass-to-charge ratio
        # NB: rigidities are in GV!
        A_per_Z = 2  # assume constant A/Z ratio
        rigidities = self.prince_run.cr_grid.grid * A_per_Z
        rigidities_widths = np.diff(self.prince_run.cr_grid.bins) * A_per_Z

        with h5py.File(outfile, "w") as f:
            f.create_dataset("distances", data=self.distances)
            f.create_dataset("massids", data=self.massids)
            f.create_dataset("As", data=self.As)
            f.create_dataset("Zs", data=self.Zs)
            f.create_dataset("en_per_nucs", data=self.prince_run.cr_grid.grid)
            f.create_dataset("rigidities", data=rigidities)
            f.create_dataset("rigidities_widths", data=rigidities_widths)

            f.create_dataset("mass_groups", data=[1, 2, 3, 4])
            f.create_dataset(
                "mass_group_idxlims", data=np.array(self.mass_group_idxlims, dtype=int)
            )
            f.create_dataset("inj_eff_matrix", data=self.inj_eff_matrix)
            f.create_dataset("propa_matrix", data=self.propa_matrix)

    def __create_kernel(self):
        """Create kernel and save the results if not yet done so"""

        if os.path.exists(self.prince_run_datapath):
            print("File already exists, no need for re-computation")
            return

        # cross section class, either use TALYS or PSB
        cs = cross_sections.CompositeCrossSection(
            [
                (0.0, cross_sections.TabulatedCrossSection, (self.css,)),
                (0.14, cross_sections.SophiaSuperposition, ()),
            ]
        )

        # generate kernel
        prince_run = core.PriNCeRun(
            max_mass=56, photon_field=self.pf_gilmore, cross_sections=cs
        )

        # pickle dump the results
        pickle.dump(prince_run, open(self.prince_run_datapath, "wb"), protocol=-1)

    def __create_distance_tables(self):
        """Create conversion table from redshift to Mpc if not yet done so"""
        if os.path.exists(self.redshift_distance_datapath):
            print("File already exists, no need for re-computation")
            return

        distance_grid_mpc = np.logspace(-1, np.log10(5000), 1000)

        redshift_grid_plk = [
            z_at_value(Planck18.comoving_distance, d * u.Mpc) for d in distance_grid_mpc
        ]
        redshift_grid_wmap = [
            z_at_value(WMAP9.comoving_distance, d * u.Mpc) for d in distance_grid_mpc
        ]

        # Fix the zeros
        redshift_grid_plk.insert(0, 0.0)
        redshift_grid_wmap.insert(0, 0.0)
        distance_grid_mpc = np.hstack([[0], distance_grid_mpc])

        # computing both z-> d and d -> z
        spl_z_to_d_plk = UnivariateSpline(
            redshift_grid_plk, distance_grid_mpc, s=0, k=2
        )
        spl_d_to_z_plk = UnivariateSpline(
            distance_grid_mpc, redshift_grid_plk, s=0, k=2
        )
        spl_z_to_d_wmap = UnivariateSpline(
            redshift_grid_wmap, distance_grid_mpc, s=0, k=2
        )
        spl_d_to_z_wmap = UnivariateSpline(
            distance_grid_mpc, redshift_grid_wmap, s=0, k=2
        )

        # dump
        pickle.dump(
            (spl_z_to_d_plk, spl_d_to_z_plk, spl_z_to_d_wmap, spl_d_to_z_wmap),
            open(self.redshift_distance_datapath, "wb"),
        )


class InjectionSolverRunContainer:
    """Helper class to instantiate and generate a single run for each distance"""

    def __init__(
        self, dinit: float, zinit: float, massids: list, prince_run: core.PriNCeRun
    ) -> None:
        """Initialise the run container.

        Parameters
        ----------
        dinit : float
            the distance of the source in Mpc
        """
        self.dinit = dinit
        self.zinit = zinit
        self.massids = massids
        self.prince_run = prince_run

    def run(self):
        """Wrapper for paralleilisation of injection solver"""
        print(f"solving for dinit={self.dinit:.3f} Mpc")
        results_per_dinit = []

        for massid in tqdm(self.massids):
            # initialise solver
            solver = UniformInjectionSolver(
                initial_z=self.zinit,
                final_z=0.0,
                prince_run=self.prince_run,
                enable_pairprod_losses=True,
                enable_adiabatic_losses=True,
                enable_injection_jacobian=False,
                enable_partial_diff_jacobian=True,
            )

            solver.add_source_class(
                NoInjection(self.prince_run, params={})
            )  # no source model
            solver.set_initial_state(
                massid, self.zinit
            )  # set uniform injection as initial state
            solver.solve(
                dz=min(self.zinit / 200, 3e-4), verbose=False, progressbar=False
            )  # solve
            # append for each mass
            results_per_dinit.append(
                (
                    solver.res,
                    solver.initial_state[solver.spec_man.ncoid2sref[massid].sl],
                )
            )

        return results_per_dinit


class UniformInjectionSolver(UHECRPropagationSolverBDF):
    def _init_solver(self, dz):
        # print("_init_")
        # initial_state = np.zeros(self.dim_states)

        self._update_jacobian(self.initial_z)
        self.current_z_rates = self.initial_z

        # find the maximum injection and reduce the system by this
        self.red_idx = self.initial_state.max()

        # Convert csr_matrix from GPU to scipy
        try:
            sparsity = self.had_int_rates.get_hadr_jacobian(self.initial_z, 1.0).get()
        except AttributeError:
            sparsity = self.had_int_rates.get_hadr_jacobian(self.initial_z, 1.0)

        from prince_cr.util import PrinceBDF

        self.r = PrinceBDF(
            self.eqn_derivative,
            self.initial_z,
            self.initial_state,
            self.final_z,
            max_step=np.abs(dz),
            atol=self.atol,
            rtol=self.rtol,
            #  jac = self.eqn_jac,
            jac_sparsity=sparsity,
            vectorized=True,
        )

    def get_available_ncoids(self):
        """Get all available massids in this spectrum"""
        return list(self.spec_man.ncoid2sref.keys())

    def set_initial_state(self, nco_id, initial_z):
        import warnings

        ewidths = np.diff(self.ebins)
        self.initial_state = np.zeros(self.dim_states)
        inj_spec = self.spec_man.ncoid2sref[nco_id]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.initial_state[inj_spec.sl] = 1.0

        pru.info(
            2,
            "Normalization: {0:5.3e}".format(
                np.sum(self.initial_state[inj_spec.sl] * ewidths)
            ),
        )
        pru.info(
            2,
            f"Redshift: {initial_z:5.3e}",
        )

        # self._update_jacobian(initial_z)
        self.initial_z = initial_z
        self.current_z_rates = self.initial_z


class NoInjection(CosmicRaySource):
    """Zero injection class to solve autonomous equation."""

    def injection_spectrum(self, pid, energy, params):
        return np.zeros_like(energy)



def cost_function(wAs : np.ndarray, *args):
    """Cost function to minimise weights per distance per rigidity for each mass group"""
    earth_spect_mg, src_spect, img = args

    Lsrc = (src_spect * wAs)
    Lmg = earth_spect_mg[...,img]
    Lmg_ex = np.sum(np.delete(earth_spect_mg, img, axis=-1), axis=-1)

    return np.log10(np.sum((Lsrc * Lmg_ex) / Lmg**2))