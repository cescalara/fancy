import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from cmdstanpy import CmdStanModel

__all__ = ["Model", "Direction", "uv_to_coord", "coord_to_uv"]

Mpc_to_km = 3.086e19


class Model:
    """
    Simple wrapper for models defined in Stan.
    """

    def __init__(self, model_filename=None, sim_filename=None, include_paths=None):
        """
        Simple wrapper for models defined in Stan.

        :param model_filename: location of the stan code for model
        :param sim_filename: locaiton of the stan code for simulation
        """

        self.model_filename = model_filename
        self.sim_filename = sim_filename
        self.include_paths = include_paths

        self.simulation = None

    def compile(self):
        """
        Compile and cache the necessary Stan models if not already done.

        :param reset: Rerun the compilation
        """

        stanc_options = {"include-paths": self.include_paths}

        if self.model_filename:
            self.model = CmdStanModel(
                stan_file=self.model_filename,
                model_name="model",
                stanc_options=stanc_options,
            )

        if self.sim_filename:
            self.simulation = CmdStanModel(
                stan_file=self.sim_filename,
                model_name="sim",
                stanc_options=stanc_options,
            )

    def input(
        self,
        B: float = None,
        kappa: float = None,
        F_T: float = None,
        f: float = None,
        Q: float = None,
        F0: float = None,
        alpha: float = None,
        Eth: float = 50,
        ptype: str = "p",
    ):
        """
        Get simulation inputs.

        :param F_T: total flux [# km-^2 yr^-1]
        :param f: associated fraction
        :param kappa: deflection parameter
        :param B: rms B field strength [nG]
        :param Q: Total Luminosity [yr^-1]
        :param alpha: source spectral index
        :param Eth: threshold energy of study [EeV]
        :param ptype: element of composition
        """
        self.F_T = F_T
        self.f = f
        self.kappa = kappa
        self.B = B
        self.Q = Q
        self.F0 = F0
        self.alpha = alpha
        self.Eth = Eth
        self.Eth_sim = None  # To be set by Analysis
        self.ptype = ptype

    def _get_properties(self):
        """
        Convenience function to pack object into dict.
        """

        self.properties = {}
        self.properties["F_T"] = self.F_T
        self.properties["f"] = self.f
        self.properties["kappa"] = self.kappa
        self.properties["B"] = self.B
        self.properties["Q"] = self.Q
        self.properties["F0"] = self.F0
        self.properties["F0"] = self.F0
        self.properties["alpha"] = self.alpha
        self.properties["Eth"] = self.Eth
        self.properties["Eth_sim"] = self.Eth_sim

        self.properties["sim_filename"] = self.sim_filename
        self.properties["model_filename"] = self.model_filename
        self.properties["include_paths"] = self.include_paths

    def save(self, file_handle):
        """
        Save to the passed H5py file handle,
        i.e. something that cna be used with
        file_handle.create_dataset()

        :param file_handle: file handle
        """

        self._get_properties()

        for key, value in self.properties.items():
            try:
                file_handle.create_dataset(key, data=value)
            except:
                pass


class Direction:
    """
    Input the unit vector vMF samples and
    store x, y, and z and galactic coordinates
    of direction in Mpc.
    """

    def __init__(self, unit_vector_3d):
        """
        Input the unit vector samples and
        store x, y, and z and galactic coordinates
        of direction in Mpc.

        :param unit_vector_3d: a 3-dimensional unit vector.
        """

        self.unit_vector = unit_vector_3d
        transposed_uv = np.transpose(self.unit_vector)
        self.x = transposed_uv[0]
        self.y = transposed_uv[1]
        self.z = transposed_uv[2]
        self.d = SkyCoord(
            self.x,
            self.y,
            self.z,
            unit="mpc",
            representation_type="cartesian",
            frame="icrs",
        )
        self.d.representation_type = "spherical"
        self.glons = self.d.galactic.l.wrap_at(360 * u.deg).deg
        self.glats = self.d.galactic.b.wrap_at(180 * u.deg).deg

        self.ras = self.d.ra.deg
        self.decs = self.d.dec.deg


def uv_to_coord(uv):
    """
    Convert unit vector array into SkyCoord object in the ICRS frame.

    :param uv: array of 3D unit vectors
    :return: astropy SkyCoord object
    """
    transposed_uv = np.transpose(uv)
    x = transposed_uv[0]
    y = transposed_uv[1]
    z = transposed_uv[2]

    c = SkyCoord(x, y, z, unit="Mpc", representation_type="cartesian", frame="icrs")

    return c


def coord_to_uv(coord):
    """
    Convert SkyCoord object into array of unit vecotrs in the ICRS frame.
    Used for input into Stan programs.

    :param coord: astropy SkyCoord object
    :return: an array of 3D unit vectors
    """
    c = coord.icrs
    ds = np.array([c.cartesian.x, c.cartesian.y, c.cartesian.z]).T
    uv = ds / np.linalg.norm(ds, axis=0)

    return uv


def convert_scale(D, alpha_T, eps, F0=None, Q=None, to_stan=True):
    """
    Convenience function to convert parameters
    to O(1) scale for sampling in Stan.
    D [Mpc] -> (D * 3.086) / 100
    alpha_T [km^2 yr] -> alpha_T / 1000
    eps [km^2 yr] -> eps / 1000
    F [# km^-2 yr^-1] -> F * 1000
    Q [# yr^-1] -> Q / 1e39

    Can also convert back by setting to_stan = False
    """

    # Convert from physical units to Stan units
    if to_stan:

        D = [(d * 3.086) / 100 for d in D]
        alpha_T = alpha_T / 1000.0
        eps = [e / 1000.0 for e in eps]

        if F0:
            F0 = F0 * 1000.0

        if isinstance(Q, (list, np.ndarray)):
            Q = Q / 1.0e39

    # Convert from Stan units to physical units
    else:

        D = [(d / 3.086) * 100 for d in D]
        alpha_T = alpha_T * 1000.0
        eps = [e * 1000.0 for e in eps]

        if F0:
            F0 = F0 / 1000.0

        if isinstance(Q, (list, np.ndarray)):
            Q = Q * 1.0e39

    if F0 and isinstance(Q, (list, np.ndarray)):

        return D, alpha_T, eps, F0, Q

    else:

        return D, alpha_T, eps


def get_simulation_input(Nsim, f, D, M, alpha_T):
    """
    For a given associated fraction and
    detector exposure, find the background flux and
    source luminosity as input to the simulation.

    :param Nsim: N simulated, ignoring exposure effects.
    :param f: Associated fraction.
    :param D: List of distances to sources [Mpc].
    :param M: Integral over the angular exposure [sr].
    :param alpha_T: Total exposure [km^2 sr yr].
    """

    FT = (Nsim * M) / alpha_T  # km^-2 yr^-1
    Fs = f * FT
    F0 = (1 - f) * FT

    # Assume equal luminosities
    Q = Fs / (sum([1 / (4 * np.pi * (d * Mpc_to_km) ** 2) for d in D]))
    Q = np.tile(Q, len(D))  # yr^-1

    return Q, F0
