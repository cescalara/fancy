import os
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz

try:

    import crpropa

except:

    crpropa = None


from fancy.utils.package_data import (
    get_path_to_stan_file,
    get_path_to_stan_includes,
    get_path_to_lens,
)
from fancy.interfaces.stan import Model, uv_to_coord, coord_to_uv
from fancy.detector.vMF.vmf import sample_vMF
from fancy.interfaces.utils import f_theta


class GMFDeflections:
    """
    Interface to GMF modelling and lensing with CRPropa3
    """

    _defined_models = ["JF12", "PT11", "TF17"]

    _defined_deflection_types = ["simple", "shifted"]

    def __init__(self, model="JF12", deflection_type="simple"):
        """
        Interface to GMF modelling and lensing with CRPropa3
        """

        if not crpropa:

            raise ImportError("CRPropa3 must be installed to use this functionality")

        if model in self._defined_models:

            if model == "JF12":

                self.lens_name = "JF12full_Gamale"

            else:

                raise NotImplementedError()

        else:

            raise ValueError("GMF model not recognised")

        if deflection_type in self._defined_deflection_types:

            self.deflection_type = deflection_type

        else:

            raise ValueError("Deflection type not recognised")

        if self.deflection_type == "simple":

            # Set up relationship between kappa and theta for use in kappa_gmf
            self._P = 0.683
            self._kappa_grid = np.logspace(-5, 8, 2000)
            self._thetaP_grid = f_theta(self._kappa_grid, P=self._P)
            self._f_kappa = interp1d(
                self._thetaP_grid,
                self._kappa_grid,
                kind="cubic",
                fill_value=(1e12, 1e-12),
                bounds_error=False,
            )

    def get_kappa_gmf_per_source(self, varpi, kappa_ex, E_ex, A, Z):
        """
        Find effective kappa for GMF deflections. Used in
        calculations for simple deflections.

        NB: Approximate version.

        :param varpi: source direction as a unit vector
        :param kappa_ex: Expected kappa from extragalactic deflections
        """

        # Simulate events from a single source
        N = 1000

        omega_gb = sample_vMF(varpi, kappa_ex, N)
        coords_gb = uv_to_coord(omega_gb)

        energies_gb = np.tile(E_ex, N) * crpropa.EeV  # UHECR energy at gal. boundary
        pid = crpropa.nucleusId(A, Z)

        rigidities_gb = np.array(
            [np.float64(energy_gb) / Z for energy_gb in energies_gb]
        )

        # Apply lens
        coords, _, _ = self._apply_lens(coords_gb, rigidities_gb, pid)

        # Calculate kappa_gmf
        kappa_gmf = self._get_kappa_gmf(coords, varpi)

        return kappa_gmf

    def get_kappa_gmf_per_source_composition(self, varpi, kappa_ex, REs, dREs, J_REs, crp_threads=1):
        """
        Find effective kappa for GMF deflections. Used in
        calculations for simple deflections.

        NB: Approximate version.

        :param varpi: source direction as a unit vector
        :param kappa_ex: Expected kappa from extragalactic deflections
        """

        # Simulate events from a single source
        N = 1000
        # set number of threads for CRPropa, should be == 1 and parallize this whole function
        os.environ["OMP_NUM_THREADS"] = str(crp_threads) 

        # Initialise lens (cannot initialize globally since CRPropa instance 
        # needs to be launched per thread)
        path_to_lens = str(get_path_to_lens(self.lens_name))
        self.lens = crpropa.MagneticLens(path_to_lens)
        self.lens.normalizeLens()

        omega_gb = sample_vMF(varpi, kappa_ex, N)
        coords_gb = uv_to_coord(omega_gb)
        
        # fine to set as EeV, input anyways needs to be energy
        # rigidities_gb = np.tile(Rex, N) * crpropa.EeV  # UHECR energy at gal. boundary

        # sample from arrival rigidity distribution
        rng = np.random.default_rng()
        rigidities_gb = rng.choice(REs, size=N, p=J_REs * dREs) * crpropa.EeV
        pid = crpropa.nucleusId(1, 1)  # take protons, since no nuclear effects

        # Apply lens
        coords, _, _ = self._apply_lens(coords_gb, rigidities_gb, pid)

        # Calculate kappa_gmf
        kappa_gmf = self._get_kappa_gmf(coords, varpi)

        return kappa_gmf

    def _get_kappa_gmf_per_source_full(self, varpi, B, Eth, alpha, D, A, Z):
        """
        Find effective kappa for GMF deflections.
        Used in calculations for simple deflections.

        :param varpi: source direction as a unit vector
        :param B: EGMF strength in nG
        :param Eth: threshold energy of source in EeV
        :param alpha: spectral index
        :param D: distance in Mpc
        :param Z: atomic number
        """

        # Simulate events from a single source
        N = 1000

        sim_input = {}
        sim_input["N"] = N
        sim_input["D"] = (D * 3.086) / 100  # Convert scale for Stan
        sim_input["varpi"] = varpi
        sim_input["alpha"] = alpha
        sim_input["B"] = B
        sim_input["Eth"] = Eth
        sim_input["Z"] = Z

        stan_path = str(get_path_to_stan_includes())
        sim_file_name = str(get_path_to_stan_file("single_source_sim.stan"))
        simulation = Model(sim_filename=sim_file_name, include_paths=stan_path)
        simulation.compile()

        sim_output = simulation.simulation.sample(
            data=sim_input,
            iter_sampling=1,
            chains=1,
            fixed_param=True,
            show_console=False,
        )

        # Simulate deflections based on kappa
        kappas = sim_output.stan_variable("kappa")[0]
        omega_gb = []
        for k in kappas:
            omega_gb.append(sample_vMF(varpi, k, 1))
        coords_gb = uv_to_coord(omega_gb)[0]

        energies_gb = (
            sim_output.stan_variable("Earr")[0] * crpropa.EeV
        )  # UHECR energy at gal. boundary
        pid = crpropa.nucleusId(A, sim_input["Z"])

        rigidities_gb = np.array(
            [np.float64(energy_gb) / sim_input["Z"] for energy_gb in energies_gb]
        )

        # Apply lens
        coords, _, _ = self._apply_lens(coords_gb, rigidities_gb, pid)

        # Calculate kappa_gmf
        kappa_gmf = self._get_kappa_gmf(coords, varpi)

        return kappa_gmf

    def _apply_lens(self, coords, rigidities, pid):
        """
        Apply lens to input particles and sample new properties.
        """

        particles = crpropa.ParticleMapsContainer()

        N = len(rigidities)

        for i in range(N):

            # convert to vector3d, flip due to initialisation in CRPropa for Vector3d -> lonlat
            coord_xyz = coords[i].cartesian.xyz.value * (-1)
            vector3d = crpropa.Vector3d(*coord_xyz)

            particles.addParticle(
                pid, rigidities[i], vector3d
            )

        particles.applyLens(self.lens)

        pids, energies, lons, lats = particles.getRandomParticles(N)
        coords = SkyCoord(lons * u.rad, lats * u.rad, frame="galactic")

        return coords, energies, pids

    def _get_kappa_gmf(self, coords, varpi):
        """
        Get kappa_gmf for simple deflections.
        """

        # Angle between source and new directions
        arr_dirs = coord_to_uv(coords)
        cos_thetas = np.dot(arr_dirs, varpi)
        thetas = np.arccos(cos_thetas)

        # Find theta that contains P
        counts, theta_bins = np.histogram(thetas, bins=300, density=True)
        dx = theta_bins[1] - theta_bins[0]
        cumulative = cumtrapz(counts, dx=dx)
        cumul_P_idx = np.argwhere(cumulative <= self._P)[-1]
        thetaP = theta_bins[cumul_P_idx][0]

        # Convert to kappa
        kappa_gmf = float(self._f_kappa(thetaP))

        return kappa_gmf
