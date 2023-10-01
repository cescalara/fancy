from re import T
from scipy import integrate, interpolate
from scipy.integrate import trapezoid
import h5py
from tqdm import tqdm as progress_bar
import time 

from ..detector.exposure import *

from multiprocessing import Pool, cpu_count

__all__ = ["ExposureIntegralTable"]


class ExposureIntegralTable:
    """
    Handles the building and storage of exposure integral tables
    that are passed to Stan to be interpolated over.
    """

    def __init__(self, varpi=None, params=None, input_filename=None):
        """
        Handles the building and storage of integral tables
        that are passed to Stan to be used in both simulation
        and sampling.

        :param kappa: an array of kappa values to evaluate the integral for
        :param varpi: an array of 3D unit vectors to pass to the integrand
        :param params: an array of other parameters to pass to the integrand
        :param input_filename: the filename to use to initialise the object
        """

        self.varpi = varpi
        self.params = params

        self.table = []
        self.sim_table = []
        self.alphas = []

        if input_filename != None:
            self.init_from_file(input_filename)

    def build_for_sim(self, kappa, alpha, B, D):
        """
        Build the tabulated integrals to be used for simulations and posterior predictive checks.
        Save with the filename given.

        Expects self.kappa to be either a fixed value or a list of values of length equal to the
        total number of sources. The integral is evaluated once for each source.
        """

        self.sim_kappa = kappa
        self.sim_alpha = alpha
        self.sim_B = B
        self.sim_D = D

        # single fixed kappa
        if isinstance(self.sim_kappa, int) or isinstance(self.sim_kappa, float):
            k = self.sim_kappa
            results = []

            for i in progress_bar(
                range(len(self.varpi)), desc="Precomputing exposure integral"
            ):
                v = self.varpi[i]
                result, err = integrate.dblquad(
                    integrand,
                    0,
                    np.pi,
                    lambda phi: 0,
                    lambda phi: 2 * np.pi,
                    args=(v, k, self.params),
                )

                results.append(result)
            self.sim_table = results
            print()

        # different kappa for each source
        else:
            results = []
            for i in progress_bar(
                range(len(self.varpi)), desc="Precomputing exposure integral"
            ):
                v = self.varpi[i]
                result, err = integrate.dblquad(
                    integrand,
                    0,
                    np.pi,
                    lambda phi: 0,
                    lambda phi: 2 * np.pi,
                    args=(v, self.sim_kappa[i], self.params),
                )

                results.append(result)

            self.sim_table = results
            print()

    def build_for_sim_parallel(self, kappa, alpha, B, D, nthreads):
        """
        Build the tabulated integrals to be used for simulations and posterior predictive checks.
        Save with the filename given.

        This parallelizes the exposure integral evaluation used to simulate events.

        Expects self.kappa to be either a fixed value or a list of values of length equal to the
        total number of sources. The integral is evaluated once for each source.
        """

        self.sim_kappa = kappa
        self.sim_alpha = alpha
        self.sim_B = B
        self.sim_D = D

        # single fixed kappa
        if isinstance(self.sim_kappa, int) or isinstance(self.sim_kappa, float):
            k = self.sim_kappa

            args = [(v, k, self.params) for v in self.varpi]

            with Pool(nthreads) as mpool:
                results = list(
                    progress_bar(
                        mpool.imap(self.eps_per_source_sim, args),
                        total=len(self.varpi),
                        desc="Precomputing exposure integral",
                    )
                )

            self.sim_table = results
            print()

        # different kappa for each source
        else:
            args = [(v, k, self.params) for v, k in zip(self.varpi, self.sim_kappa)]

            with Pool(nthreads) as mpool:
                results = list(
                    progress_bar(
                        mpool.imap(self.eps_per_source_sim, args),
                        total=len(self.varpi),
                        desc="Precomputing exposure integral",
                    )
                )

            self.sim_table = results
            print()

    def eps_per_source_sim(self, args):
        """
        The exposure integral using the source direction and simulated magnetic deflections
        for each source.

        :param: args : tuple containing source unit vector, simulated kappa, and self.params
        """
        result, err = integrate.dblquad(
            integrand, 0, np.pi, lambda phi: 0, lambda phi: 2 * np.pi, args=args
        )
        return result

    def build_for_fit(self, kappa):
        """
        Build the tabulated integrals to be interpolated over in the fit.
        Save with filename given.

        Expects self.kappa to be a list of kappa values to evaluate the integral for,
        for each source individually.
        """

        self.kappa = kappa
        for i in progress_bar(
            range(len(self.varpi)), desc="Precomputing exposure integral"
        ):
            v = self.varpi[i]

            results = []
            for k in self.kappa:

                result, err = integrate.dblquad(
                    integrand,
                    0,
                    np.pi,
                    lambda phi: 0,
                    lambda phi: 2 * np.pi,
                    args=(v, k, self.params),
                )

                results.append(result)
            self.table.append(np.asarray(results))
            print()
        self.table = np.asarray(self.table)

    def eps_per_source(self, v):
        """
        Evaluate exposure integral per source. This corresponds to the inner for loop
        that contains the double integral evaluation for each kappa.

        :param: v : source unit vector
        """

        results = []
        for k in self.kappa:
            result, err = integrate.dblquad(
                integrand,
                0,
                np.pi,
                lambda phi: 0,
                lambda phi: 2 * np.pi,
                args=(v, k, self.params),
            )

            results.append(result)

        return results

    def build_for_fit_parallel(self, kappa, nthreads):
        """
        Build the tabulated integrals to be interpolated over in the fit.
        Save with filename given.

        This is the parallelized version, using multiprocessing over each source
        in the provided source catalogue.

        For SBG, runtime decreases from 30 min -> 1.5 min with ~28 cores

        Expects self.kappa to be a list of kappa values to evaluate the integral for,
        for each source individually.
        """

        self.kappa = kappa

        with Pool(nthreads) as mpool:
            results = list(
                progress_bar(
                    mpool.imap(self.eps_per_source, self.varpi),
                    total=len(self.varpi),
                    desc="Precomputing exposure integral",
                )
            )
            self.table.append(np.asarray(results))
            print()
        self.table = np.asarray(self.table)

    def build_for_fit_gmf(self, kappa, Eex, A, Z, deflections):
        """
        Build the tabulated integrals to be interpolated over in the fit.

        Updated calculation to handle GMF deflections.
        """

        self.kappa = kappa
        for i in progress_bar(
            range(len(self.varpi)), desc="Precomputing exposure integral"
        ):
            v = self.varpi[i]

            results = []
            for k in kappa:

                k_gmf = deflections.get_kappa_gmf_per_source(v, k, Eex, A, Z)

                result, err = integrate.dblquad(
                    integrand,
                    0,
                    np.pi,
                    lambda phi: 0,
                    lambda phi: 2 * np.pi,
                    args=(v, k_gmf, self.params),
                )

                results.append(result)
            self.table.append(np.asarray(results))
            print()
        self.table = np.asarray(self.table)

    def eps_per_source_gmf(self, args):
        """
        Evaluate exposure integral per source. This corresponds to the inner for loop
        that contains the double integral evaluation for each kappa.

        :param: v : source unit vector
        """

        v, kappa_gmf = args

        results = []
        for k in kappa_gmf:
            result, err = integrate.dblquad(
                integrand,
                0,
                np.pi,
                lambda phi: 0,
                lambda phi: 2 * np.pi,
                args=(v, k, self.params),
            )

            results.append(result)

        return results

    def eps_per_source_composition(self, args):
        """
        Evaluate exposure integral per source. This corresponds to the inner for loop
        that contains the double integral evaluation for each kappa.

        :param: v : source unit vector
        """

        indices, v, kappa, REs, J_REs, dRearths, deflections = args
        print(indices)
        results = np.zeros(len(kappa))

        for k, kap in enumerate(kappa):
            
            # compute kappa_gmf
            kappa_gmf = deflections.get_kappa_gmf_per_source_composition(v, kap, REs, dRearths, J_REs)
            # print("Computed kappa_gmf")

            result, err = integrate.dblquad(
                integrand,
                0,
                np.pi,
                lambda phi: 0,
                lambda phi: 2 * np.pi,
                args=(v, kappa_gmf, self.params),
            )
            # print("Computed exposure")

            results[k] = result

        return indices, results

    def build_for_fit_parallel_gmf(self, kappa, Eex, A, Z, deflections, nthreads):
        """
        Build the tabulated integrals to be interpolated over in the fit.
        Save with filename given.

        Update calculation to handle GMF deflections.

        This is the parallelized version, using multiprocessing over each source
        in the provided source catalogue.

        For SBG, runtime decreases from 30 min -> 1.5 min with ~28 cores

        Expects self.kappa to be a list of kappa values to evaluate the integral for,
        for each source individually.
        """

        self.kappa = kappa

        # Cannot parallelise w/ imap due to magnetic lens pickling...
        args = []
        for i, v in progress_bar(enumerate(self.varpi), total=len(self.varpi), desc="Precomputing kappa_gmf for each source"):

            kappa_gmf = []
            for k in self.kappa:

                k_gmf = deflections.get_kappa_gmf_per_source(v, k, Eex, A, Z)
                kappa_gmf.append(k_gmf)

            args.append((v, kappa_gmf))

        with Pool(nthreads) as mpool:
            results = list(
                progress_bar(
                    mpool.imap(self.eps_per_source_gmf, args),
                    total=len(self.varpi),
                    desc="Precomputing exposure integral",
                )
            )
            self.table.append(np.asarray(results))
            print()
        self.table = np.asarray(self.table)

    def build_for_fit_parallel_composition(self, kappa, alpha_grid, Rearths, dRearths, arr_spectrums, deflections, nthreads):
        """
        Build the tabulated integrals to be interpolated over in the fit.
        Save with filename given.

        Update calculation to handle GMF deflections.

        This is the parallelized version, using multiprocessing over each source
        in the provided source catalogue.

        For SBG, runtime decreases from 30 min -> 1.5 min with ~28 cores

        Expects self.kappa to be a list of kappa values to evaluate the integral for,
        for each source individually.
        """

        self.kappa = kappa
        self.alphas = alpha_grid

        # truncate alphas to smaller values
        # self.alphas = np.linspace(np.min(alpha_grid), np.max(alpha_grid), int(len(alpha_grid)//2))
        self.table = np.zeros((len(self.alphas), len(self.varpi), len(kappa)))

        args = []
        # create argument list for 2d parallelization based on sources & alphas
        print("Constructing arguments for precomputation")
        for indices in np.ndindex(len(self.alphas), len(self.varpi)):
            # unpack
            a, i = indices

            # get values
            v = self.varpi[i]
            alpha = self.alphas[a]

            # evaluate index corresponding to original alpha grid
            aa = np.digitize(alpha, alpha_grid, right=True)
            J_REs = arr_spectrums[i,aa,:] 

            # normalize
            J_REs /= np.sum(J_REs * dRearths)

            # dont perform exposure calculation if arrival spectrum is nan
            if np.all(np.isnan(J_REs)):
                continue

            args.append(((a, i), v, self.kappa, Rearths, dRearths, J_REs, deflections))

        #  measure time it takes to run this part
        t0 = time.perf_counter()

        # perform 2d paralleilization
        with Pool(nthreads) as mpool:
            # returns indices & effective exposure for each kappa
            results = tuple(
                progress_bar(
                    mpool.imap(self.eps_per_source_composition, args),
                    desc="Precomputing Exposure Integral & kappa_GMF",
                    total=len(args)
                )
            )

            # get indices & values within results
            for indices, vals in results:
                a, i = indices
                self.table[a,i,:] = vals

        t1 = time.perf_counter()

        print(f"Total time elapsed: {(t1 - t0) / 3600 :.5f} hrs")

    def init_from_file(self, input_filename):
        """
        Initialise the object from the given file.
        """

        with h5py.File(input_filename, "r") as f:

            self.varpi = f["varpi"][()]
            self.params = f["params"][()]
            self.kappa = f["main"]["kappa"][()]
            self.table = f["main"]["table"][()]
            
            if f["main"]["alphas"][()] is not h5py.Empty("f"):
                self.alphas = f["main"]["alphas"][()]

            if f["simulation"]["kappa"][()] is not h5py.Empty("f"):
                try:
                    self.sim_kappa = f["simulation"]["kappa"][()]
                    self.sim_table = f["simulation"]["table"][()]
                    self.sim_alpha = f["simulation"]["alpha"][()]
                    self.sim_B = f["simulation"]["B"][()]
                    self.sim_D = f["simulation"]["D"][()]
                except:
                    print("skipped simulation values")

    def save(self, output_filename):
        """
        Save the computed integral table(s) to a HDF5 file
        for later use as inputs.

        If no table is found, create an empty dataset.

        :param output_filename: the name of the file to write to
        """

        with h5py.File(output_filename, "w") as f:

            # common params
            f.create_dataset("varpi", data=self.varpi)
            f.create_dataset("params", data=self.params)

            # main interpolation table
            main = f.create_group("main")
            if self.table != []:
                main.create_dataset("kappa", data=self.kappa)
                main.create_dataset("table", data=self.table)
            else:
                main.create_dataset("kappa", data=h5py.Empty("f"))
                main.create_dataset("table", data=h5py.Empty("f"))

            if self.alphas != []:
                main.create_dataset("alphas", data=self.alphas)
            else:
                main.create_dataset("alphas", data=h5py.Empty("f"))

            # simulation table
            sim = f.create_group("simulation")
            if self.sim_table != []:
                sim.create_dataset("kappa", data=self.sim_kappa)
                sim.create_dataset("table", data=self.sim_table)
                sim.create_dataset("alpha", data=self.sim_alpha)
                sim.create_dataset("B", data=self.sim_B)
                sim.create_dataset("D", data=self.sim_D)
            else:
                sim.create_dataset("kappa", data=h5py.Empty("f"))
                sim.create_dataset("table", data=h5py.Empty("f"))
                sim.create_dataset("alpha", data=h5py.Empty("f"))
                sim.create_dataset("B", data=h5py.Empty("f"))
                sim.create_dataset("D", data=h5py.Empty("f"))
