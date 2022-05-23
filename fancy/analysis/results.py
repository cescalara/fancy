import h5py
from math import ceil
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from tqdm import tqdm as progress_bar
from cmdstanpy import CmdStanModel

from ..interfaces.stan import Direction, Mpc_to_km, convert_scale
from ..detector.exposure import m_integrand
from ..interfaces.integration import ExposureIntegralTable
from ..propagation.energy_loss import get_Eth_src, get_Eex, get_kappa_ex

# from ..plotting import AllSkyMap
from ..plotting import AllSkyMapCartgopy as AllSkyMap

__all__ = ["Results", "PPC"]


class Results:
    """
    Manage the output of Analysis object.
    """

    def __init__(self, filename):
        """
        Manage the output of Analysis object.
        Reads in a HDF5 file containting fit/simulation
        results for further plotting, analysis and PPC.
        """

        self.filename = filename

    def get_chain(self, list_of_keys):
        """
        Returns chain of desired parameters specified by list_of_keys.
        """

        chain = {}
        with h5py.File(self.filename, "r") as f:
            samples = f["fit/samples"]
            for key in list_of_keys:
                chain[key] = samples[key][()]

        return chain

    def get_truths(self, list_of_keys):
        """
        For the case where the analysis was based on simulated
        data, return input values or 'truths' for desired
        parameters specified by list_of_keys.
        """

        truths = {}
        with h5py.File(self.filename, "r") as f:

            try:
                model = f["model"]
                for key in list_of_keys:

                    if key == "f":

                        # reconstruct from other info
                        F0 = model["F0"][()]
                        L = model["L"][()]
                        D = f["source/distance"][()]
                        D = [d * Mpc_to_km for d in D]

                        Fs = sum([(l / (4 * np.pi * d ** 2)) for l, d in zip(L, D)])
                        f = Fs / (Fs + F0)
                        truths["f"] = f

                    else:

                        truths[key] = model[key][()]

            except:
                print("Error: file does not contain simulation inputs.")
        return truths

    def get_fit_parameters(self):
        """
        Return mean values of all main fit parameters.
        """

        list_of_keys = ["B", "alpha", "L", "F0", "lambda"]
        chain = self.get_chain(list_of_keys)

        fit_parameters = {}
        fit_parameters["B"] = np.mean(chain["B"])
        fit_parameters["alpha"] = np.mean(chain["alpha"])
        fit_parameters["F0"] = np.mean(chain["F0"])
        fit_parameters["L"] = np.mean(chain["L"])
        try:
            fit_parameters["lambda"] = np.mean(np.transpose(chain["lambda"]), axis=1)
        except:
            print("Found no lambda parameters.")

        return fit_parameters

    def get_input_data(self):
        """
        Return fit input data.
        """

        fit_input = {}
        detector = {}
        source = {}
        with h5py.File(self.filename, "r") as f:
            fit_input_handle = f["fit/input"]
            for key in fit_input_handle:
                fit_input[key] = fit_input_handle[key][()]

            detector_handle = f["detector"]
            for key in detector_handle:
                detector[key] = detector_handle[key][()]

            source_handle = f["source"]
            for key in source_handle:
                source[key] = source_handle[key][()]

        return fit_input, detector, source

    def run_ppc(self, stan_sim_file, include_paths, N=3, seed=None):
        """
        Run N posterior predictive simulations.
        """

        keys = ["L", "F0", "alpha", "B"]
        fit_chain = self.get_chain(keys)
        input_data, detector, source = self.get_input_data()

        self.ppc = PPC(stan_sim_file, include_paths)

        self.ppc.simulate(fit_chain, input_data, detector, source, N=N, seed=seed)


class PPC:
    """
    Handles posterior predictive checks.
    """

    def __init__(self, stan_sim_file, include_paths):
        """
        Handles posterior predictive checks.
        :param stan_sim_file: the stan file to use to run the simulation
        """

        stanc_options = {"include-paths": include_paths}

        # compile the stan model
        self.simulation = CmdStanModel(
            stan_file=stan_sim_file,
            model_name="ppc_sim",
            stanc_options=stanc_options,
        )

        self.arrival_direction_preds = []
        self.Edet_preds = []
        self.Nex_preds = []
        self.labels_preds = []

    def simulate(self, fit_chain, input_data, detector, source, seed=None, N=3):
        """
        Simulate from the posterior predictive distribution.
        """

        self.alpha = fit_chain["alpha"]
        self.B = fit_chain["B"]
        self.F0 = fit_chain["F0"]
        self.L = fit_chain["L"]

        self.arrival_direction = Direction(input_data["arrival_direction"])
        self.Edet = input_data["Edet"]
        self.Eth = input_data["Eth"]
        self.Eth_src = get_Eth_src(self.Eth, source["distance"])
        self.varpi = input_data["varpi"]

        # Get params from detector for exposure integral calculation
        self.params = []
        self.params.append(np.cos(detector["lat"]))
        self.params.append(np.sin(detector["lat"]))
        self.params.append(np.cos(detector["theta_m"]))
        self.params.append(detector["alpha_T"])
        M, Merr = integrate.quad(m_integrand, 0, np.pi, args=self.params)
        self.params.append(M)

        for i in progress_bar(range(N), desc="Posterior predictive simulation(s)"):

            # sample parameters from chain
            alpha = np.random.choice(self.alpha)
            B = np.random.choice(self.B)
            F0 = np.random.choice(self.F0)
            L = np.random.choice(self.L)

            # calculate eps integral
            Eex = get_Eex(self.Eth_src, alpha)
            kappa_ex = get_kappa_ex(Eex, np.mean(self.B), source["distance"])
            self.ppc_table = ExposureIntegralTable(varpi=self.varpi, params=self.params)
            self.ppc_table.build_for_sim(kappa_ex, alpha, B, source["distance"])

            eps = self.ppc_table.sim_table

            # rescale to Stan units
            D, alpha_T, eps = convert_scale(
                source["distance"], detector["alpha_T"], eps
            )

            # compile inputs
            self.ppc_input = {
                "kappa_d": input_data["kappa_d"],
                "Ns": input_data["Ns"],
                "varpi": input_data["varpi"],
                "D": D,
                "A": input_data["A"][0],
                "a0": detector["lat"],
                "theta_m": detector["theta_m"],
                "alpha_T": alpha_T,
                "eps": eps,
            }
            self.ppc_input["B"] = B
            self.ppc_input["L"] = np.tile(L, input_data["Ns"])
            self.ppc_input["F0"] = F0
            self.ppc_input["alpha"] = alpha
            self.ppc_input["Eerr"] = input_data["Eerr"]
            self.ppc_input["Eth"] = self.Eth

            # run simulation
            self.posterior_predictive = self.simulation.sampling(
                data=self.ppc_input,
                iter=1,
                chains=1,
                algorithm="Fixed_param",
                seed=seed,
            )

            # extract output
            self.Nex_preds.append(self.posterior_predictive.stan_variable("Nex_sim")[0])
            labels_pred = self.posterior_predictive.stan_variable("lambda")[0]
            arrival_direction = self.posterior_predictive.stan_variable(
                "arrival_direction"
            )[0]
            Edet_pred = self.posterior_predictive.stan_variable("Edet")[0]
            arr_dir_pred = Direction(arrival_direction)
            self.Edet_preds.append(Edet_pred)
            self.labels_preds.append(labels_pred)
            self.arrival_direction_preds.append(arr_dir_pred)

    def save(self, filename):
        """
        Save the predicted data to the given file.
        """

        dt = h5py.special_dtype(vlen=np.dtype("f"))
        arrival_direction_preds = [a.unit_vector for a in self.arrival_direction_preds]
        with h5py.File(filename, "w") as f:
            ppc = f.create_group("PPC")
            ppc.create_dataset("Edet", data=self.Edet)
            ppc.create_dataset(
                "arrival_direction", data=self.arrival_direction.unit_vector
            )
            ppc.create_dataset("Edet_preds", data=self.Edet_preds, dtype=dt)
            adp = ppc.create_group("arrival_direction_preds")
            for i, a in enumerate(arrival_direction_preds):
                adp.create_dataset(str(i), data=a)

    def plot(self, ppc_type=None, cmap=None):
        """
        Plot the posterior predictive check against the data
        (or original simulation) for ppc_type == 'arrival direction'
        or ppc_type == 'energy'.
        """

        if ppc_type == None:
            ppc_type = "arrival direction"

        # how many simulaitons
        N_sim = len(self.arrival_direction_preds)
        N_grid = N_sim + 1
        N_rows = ceil(np.sqrt(N_grid))
        N_cols = ceil(N_grid / N_rows)

        if ppc_type == "arrival direction":

            # plot style
            if cmap == None:
                cmap = plt.cm.get_cmap("viridis")

            # figure
            fig, ax = plt.subplots(N_rows, N_cols, figsize=(5 * N_rows, 4 * N_cols))
            flat_ax = ax.reshape(-1)

            # skymap
            skymap = AllSkyMap(projection="hammer", lon_0=0, lat_0=0)

            for i, ax in enumerate(flat_ax):

                if i < N_grid:
                    # data
                    if i == 0:
                        skymap.ax = ax
                        label = True
                        for lon, lat in np.nditer(
                            [self.arrival_direction.glons, self.arrival_direction.glats]
                        ):
                            if label:
                                skymap.tissot(
                                    lon, lat, 4.0, npts=30, alpha=0.5, label="data"
                                )
                                label = False
                            else:
                                skymap.tissot(lon, lat, 4.0, npts=30, alpha=0.5)

                    # predicted
                    else:
                        skymap.ax = ax
                        label = True
                        for lon, lat in np.nditer(
                            [
                                self.arrival_direction_preds[i - 1].lons,
                                self.arrival_direction_preds[i - 1].lats,
                            ]
                        ):
                            if label:
                                skymap.tissot(
                                    lon,
                                    lat,
                                    4.0,
                                    npts=30,
                                    alpha=0.5,
                                    color="g",
                                    label="predicted",
                                )
                                label = False
                            else:
                                skymap.tissot(
                                    lon, lat, 4.0, npts=30, alpha=0.5, color="g"
                                )
                else:
                    ax.axis("off")

        if ppc_type == "energy":

            bins = np.logspace(np.log(self.Eth), np.log(1e4), base=np.e)

            # figure
            fig, ax = plt.subplots(N_rows, N_cols, figsize=(5 * N_rows, 4 * N_cols))
            flat_ax = ax.reshape(-1)

            for i, ax in enumerate(flat_ax):

                if i < N_grid:

                    if i == 0:
                        ax.hist(
                            self.Edet, bins=bins, alpha=0.7, label="data", color="k"
                        )
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        ax.get_yaxis().set_visible(False)
                    else:
                        ax.hist(
                            self.Edet_preds[i - 1],
                            bins=bins,
                            alpha=0.7,
                            label="predicted",
                            color="g",
                        )
                        ax.set_xscale("log")
                        ax.set_yscale("log")
                        ax.get_yaxis().set_visible(False)

                else:
                    ax.axis("off")

        if ppc_type == "labels":

            bins = np.linspace(min(self.labels), max(self.labels), len(self.labels))

            # figure
            fig, ax = plt.subplots(N_rows, N_cols, figsize=(5 * N_rows, 4 * N_cols))
            flat_ax = ax.reshape(-1)

            for i, ax in enumerate(flat_ax):

                if i < N_grid:

                    if i == 0:
                        ax.hist(
                            self.labels, bins=bins, alpha=0.7, label="data", color="k"
                        )
                        ax.get_yaxis().set_visible(False)
                    else:
                        ax.hist(
                            self.labels_preds[i - 1],
                            bins=bins,
                            alpha=0.7,
                            label="predicted",
                            color="g",
                        )
                        ax.get_yaxis().set_visible(False)

                else:
                    ax.axis("off")
