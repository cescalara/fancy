import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
from matplotlib import pyplot as plt
import h5py
from tqdm import tqdm as progress_bar
from multiprocessing import Pool, cpu_count
from scipy.stats import bernoulli
import os

import stan_utility

from ..interfaces.integration import ExposureIntegralTable
from ..interfaces.stan import Direction, convert_scale, coord_to_uv, uv_to_coord
from ..interfaces.data import Uhecr
from ..interfaces.utils import get_nucleartable
# from ..plotting import AllSkyMap
from ..plotting import AllSkyMapCartopy as AllSkyMap
from ..propagation.energy_loss import get_Eth_src, get_kappa_ex, get_Eex, get_Eth_sim, get_arrival_energy, get_arrival_energy_vec
from ..detector.vMF.vmf import sample_vMF, sample_sphere
from ..detector.exposure import m_dec

# import crpropa
import sys

sys.path.append("/opt/CRPropa3/lib/python3.8/site-packages")
import crpropa

__all__ = ['Analysis']


class Analysis():
    """
    To manage the running of simulations and fits based on Data and Model objects.
    """

    nthreads = int(cpu_count() * 0.75)

    def __init__(self,
                 data,
                 model,
                 analysis_type=None,
                 filename=None,
                 summary=b''):
        """
        To manage the running of simulations and fits based on Data and Model objects.

        :param data: a Data object
        :param model: a Model object
        :param analysis_type: type of analysis
        """

        self.data = data
        self.model = model
        self.filename = filename

        # Initialise file
        if self.filename:

            with h5py.File(self.filename, 'w') as f:
                desc = f.create_group('description')
                desc.attrs['summary'] = summary

        self.simulation_input = None
        self.fit_input = None

        self.simulation = None
        self.fit = None

        # Simulation outputs
        self.source_labels = None
        self.E = None
        self.Earr = None
        self.Edet = None
        self.defl_plotvars = None

        self.arr_dir_type = 'arrival_direction'
        self.E_loss_type = 'energy_loss'
        self.joint_type = 'joint'
        self.gmf_type = "joint_gmf"

        if analysis_type == None:
            analysis_type = self.arr_dir_type

        self.analysis_type = analysis_type

        if self.analysis_type.find('joint') != -1:

            # find lower energy threshold for the simulation, given Eth and Eerr
            self.model.Eth_sim = get_Eth_sim(
                self.data.detector.energy_uncertainty, self.model.Eth)

            # find correspsonding Eth_src
            self.Eth_src = get_Eth_src(self.model.Eth_sim,
                                       self.data.source.distance)

        # Set up integral tables
        params = self.data.detector.params
        varpi = self.data.source.unit_vector
        self.tables = ExposureIntegralTable(varpi=varpi, params=params)

        # table containing (A, Z) of each element
        self.nuc_table = get_nucleartable()

    def build_tables(self,
                     num_points=50,
                     sim_only=False,
                     fit_only=False,
                     parallel=True):
        """
        Build the necessary integral tables.
        """

        if sim_only:

            # kappa_true table for simulation
            if self.analysis_type == self.arr_dir_type or self.analysis_type == self.E_loss_type:
                kappa_true = self.model.kappa
                D_src = self.data.source.distance

            if self.analysis_type == self.joint_type:
                D_src = self.data.source.distance
                self.Eex = get_Eex(self.Eth_src, self.model.alpha)
                self.kappa_ex = get_kappa_ex(self.Eex, self.model.B, D_src)
                kappa_true = self.kappa_ex

            if self.analysis_type == self.gmf_type:
                # shift by 0.02 to get kappa_ex at g.b.
                D_src = self.data.source.distance - 0.02
                self.Eex = get_Eex(self.Eth_src, self.model.alpha)

                self.kappa_ex = get_kappa_ex(self.Eex, self.model.B, D_src)
                kappa_true = self.kappa_ex

                # evaluate for kappa_d

            if parallel:
                self.tables.build_for_sim_parallel(kappa_true,
                                                   self.model.alpha,
                                                   self.model.B, D_src)
            else:
                self.tables.build_for_sim(kappa_true, self.model.alpha,
                                          self.model.B, D_src)

        if fit_only:

            # logarithmically spcaed array with 60% of points between KAPPA_MIN and 100
            kappa_first = np.logspace(np.log(1),
                                      np.log(10),
                                      int(num_points * 0.7),
                                      base=np.e)
            kappa_second = np.logspace(np.log(10),
                                       np.log(100),
                                       int(num_points * 0.2) + 1,
                                       base=np.e)
            kappa_third = np.logspace(np.log(100),
                                      np.log(1000),
                                      int(num_points * 0.1) + 1,
                                      base=np.e)
            kappa = np.concatenate(
                (kappa_first, kappa_second[1:], kappa_third[1:]), axis=0)

            # full table for fit
            if parallel:
                self.tables.build_for_fit_parallel(kappa)
            else:
                self.tables.build_for_fit(kappa)

    def build_energy_table(self,
                           num_points=50,
                           table_file=None,
                           parallel=True):
        """
        Build the energy interpolation tables.
        """

        self.E_grid = np.logspace(np.log(self.model.Eth),
                                  np.log(1.0e4),
                                  num_points,
                                  base=np.e)
        self.Earr_grid = []

        if parallel:

            args_list = [(self.E_grid, d) for d in self.data.source.distance]
            # parallelize for each source distance
            with Pool(self.nthreads) as mpool:
                results = list(
                    progress_bar(mpool.imap(get_arrival_energy_vec, args_list),
                                 total=len(args_list),
                                 desc='Precomputing energy grids'))

                self.Earr_grid = results

        else:
            for i in progress_bar(range(len(self.data.source.distance)),
                                  desc='Precomputing energy grids'):
                d = self.data.source.distance[i]
                self.Earr_grid.append(
                    [get_arrival_energy(e, d)[0] for e in self.E_grid])

        if table_file:
            with h5py.File(table_file, 'r+') as f:
                E_group = f.create_group('energy')
                E_group.create_dataset('E_grid', data=self.E_grid)
                E_group.create_dataset('Earr_grid', data=self.Earr_grid)

    def use_tables(self, input_filename, main_only=True):
        """
        Pass in names of integral tables that have already been made.
        Only the main table is read in by default, the simulation table 
        must be recalculated every time the simulation parameters are 
        changed.
        """

        if main_only:
            input_table = ExposureIntegralTable(input_filename=input_filename)
            self.tables.table = input_table.table
            self.tables.kappa = input_table.kappa

            with h5py.File(input_filename, 'r') as f:
                self.E_grid = f['energy/E_grid'][()]
                self.Earr_grid = f['energy/Earr_grid'][()]

        else:
            self.tables = ExposureIntegralTable(input_filename=input_filename)

    def _get_zenith_angle(self, c_icrs, loc, time):
        """
        Calculate the zenith angle of a known point 
        in ICRS (equatorial coords) for a given 
        location and time.
        """
        c_altaz = c_icrs.transform_to(AltAz(obstime=time, location=loc))
        return (np.pi / 2 - c_altaz.alt.rad)

    def _simulate_zenith_angles(self, start_year=2004):
        """
        Simulate zenith angles for a set of arrival_directions.

        :params: start_year: year in which measurements started.
        """

        if len(self.arrival_direction.d.icrs) == 1:
            c_icrs = self.arrival_direction.d.icrs[0]
        else:
            c_icrs = self.arrival_direction.d.icrs

        time = []
        zenith_angles = []
        stuck = []

        j = 0
        first = True
        for d in c_icrs:
            za = 99
            i = 0
            while (za > self.data.detector.threshold_zenith_angle.rad):
                dt = np.random.exponential(1 / self.N)
                if (first):
                    t = start_year + dt
                else:
                    t = time[-1] + dt
                tdy = Time(t, format='decimalyear')
                za = self._get_zenith_angle(d, self.data.detector.location,
                                            tdy)

                i += 1
                if (i > 100):
                    za = self.data.detector.threshold_zenith_angle.rad
                    stuck.append(1)
            time.append(t)
            first = False
            zenith_angles.append(za)
            j += 1
            #print(j , za)

        if (len(stuck) > 1):
            print('Warning: % of zenith angles stuck is',
                  len(stuck) / len(zenith_angles) * 100)

        return zenith_angles

    def simulate(self, seed=None, Eth_sim=None):
        """
        Run a simulation.

        :param seed: seed for RNG
        :param Eth_sim: the minimun energy simulated
        :param gmf: enable galactic magnetic field deflections
        """

        eps = self.tables.sim_table

        # handle selected sources
        if (self.data.source.N < len(eps)):
            eps = [eps[i] for i in self.data.source.selection]

        # convert scale for sampling
        D = self.data.source.distance
        alpha_T = self.data.detector.alpha_T
        L = self.model.L
        F0 = self.model.F0
        D, alpha_T, eps, F0, L = convert_scale(D, alpha_T, eps, F0, L)

        if self.analysis_type == self.joint_type \
            or self.analysis_type == self.E_loss_type \
                or self.analysis_type == self.gmf_type:
            # find lower energy threshold for the simulation, given Eth and Eerr
            if Eth_sim:
                self.model.Eth_sim = Eth_sim

        # compile inputs from Model and Data
        self.simulation_input = {
            'kappa_d': self.data.detector.kappa_d,
            'Ns': len(self.data.source.distance),
            'varpi': self.data.source.unit_vector,
            'D': D,
            'A': self.data.detector.area,
            'a0': self.data.detector.location.lat.rad,
            'lon': self.data.detector.location.lon.rad,
            'theta_m': self.data.detector.threshold_zenith_angle.rad,
            'alpha_T': alpha_T,
            'eps': eps
        }

        self.simulation_input['L'] = L
        self.simulation_input['F0'] = F0
        self.simulation_input['distance'] = self.data.source.distance

        if self.analysis_type == self.arr_dir_type or self.analysis_type == self.E_loss_type:

            self.simulation_input['kappa'] = self.model.kappa

        if self.analysis_type == self.E_loss_type:

            self.simulation_input['alpha'] = self.model.alpha
            self.simulation_input['Eth'] = self.model.Eth_sim
            self.simulation_input[
                'Eerr'] = self.data.detector.energy_uncertainty

        if self.analysis_type == self.joint_type:

            self.simulation_input['B'] = self.model.B
            self.simulation_input['alpha'] = self.model.alpha
            self.simulation_input['Eth'] = self.model.Eth_sim
            self.simulation_input[
                'Eerr'] = self.data.detector.energy_uncertainty

        if self.analysis_type == self.gmf_type:

            self.simulation_input['B'] = self.model.B
            self.simulation_input['alpha'] = self.model.alpha
            self.simulation_input['Eth'] = self.model.Eth_sim
            self.simulation_input[
                'Eerr'] = self.data.detector.energy_uncertainty

            # get particle type we intialize simulation with
            self.ptype = self.model.ptype
            _, Z = self.nuc_table[self.ptype]
            self.simulation_input["Z"] = Z

        try:
            if self.data.source.flux:
                self.simulation_input['flux'] = self.data.source.flux
            else:
                self.simulation_input['flux'] = np.zeros(self.data.source.N)
        except:
            self.simulation_input['flux'] = np.zeros(self.data.source.N)

        # run simulation
        print('Running Stan simulation...')
        self.simulation = self.model.simulation.sampling(
            data=self.simulation_input,
            iter=1,
            chains=1,
            algorithm="Fixed_param",
            seed=seed)

        # extract output
        print('Extracting output...')

        self.Nex_sim = self.simulation.extract(['Nex_sim'])['Nex_sim']
        # source_labels: to which source label each UHECR is associated with
        self.source_labels = (
            self.simulation.extract(['lambda'])['lambda'][0] - 1).astype(int)

        if self.analysis_type == self.arr_dir_type:
            arrival_direction = self.simulation.extract(
                ['arrival_direction'])['arrival_direction'][0]

        elif self.analysis_type == self.joint_type \
            or self.analysis_type == self.E_loss_type \
                or self.analysis_type == self.gmf_type:

            self.Earr = self.simulation.extract(
                ['Earr'])['Earr'][0]  # arrival energy
            self.E = self.simulation.extract(
                ['E'])['E'][0]  # sampled from spectrum

            # simulate with deflections with GMF
            if self.analysis_type == self.gmf_type:
                kappas = self.simulation.extract(['kappa'])['kappa'][0]
                print("Simulating deflections...")
                arrival_direction, self.Edet = self._simulate_deflections(
                    kappas)

            else:
                arrival_direction = self.simulation.extract(
                    ['arrival_direction'])['arrival_direction'][0]

                self.Edet = self.simulation.extract(['Edet'])['Edet'][0]

            # make cut on Eth
            inds = np.where(self.Edet >= self.model.Eth)
            self.Edet = self.Edet[inds]
            arrival_direction = arrival_direction[inds]
            self.source_labels = self.source_labels[inds]

        # convert to Direction object
        self.arrival_direction = Direction(arrival_direction)
        self.N = len(self.arrival_direction.unit_vector)

        # simulate the zenith angles
        print('Simulating zenith angles...')
        self.zenith_angles = self._simulate_zenith_angles(
            self.data.detector.start_year)

        # Make uhecr object
        uhecr_properties = {}
        uhecr_properties['label'] = self.data.detector.label
        uhecr_properties['N'] = self.N
        uhecr_properties['unit_vector'] = self.arrival_direction.unit_vector
        uhecr_properties['energy'] = self.Edet
        uhecr_properties['zenith_angle'] = self.zenith_angles
        uhecr_properties['A'] = np.tile(self.data.detector.area, self.N)
        uhecr_properties['source_labels'] = self.source_labels

        uhecr_properties[
            "ptype"] = self.model.ptype if self.analysis_type == self.gmf_type else 'p'

        new_uhecr = Uhecr()
        new_uhecr.from_simulation(uhecr_properties)

        # evaluate kappa_gmf manually here
        if self.analysis_type == self.gmf_type:
            print("Computing kappa_gmf...")
            new_uhecr.kappa_gmf, _, _ = new_uhecr.eval_kappa_gmf(
                particle_type=new_uhecr.ptype,
                Nrand=100,
                gmf="JF12",
                plot=False)

        self.data.uhecr = new_uhecr

        print('Done!')

    def _simulate_deflections(self, kappas, Nrand=100, path_to_lens=None):
        '''
        Simulate the forward propagation of UHECR in the galactic magnetic field using
        CRPropa healpix maps and galactic lensing. The simulation performs (in essence)
        the following:
        1. sample from vMF distribution with the given kappa (deflections from EGMF)
        2. Create map of particles and apply lensing given from CRPropa
        3. Sample energies and directions from lensed map
        4. Limit directions and corrsp. energies based on exposure of detector
        This then returns the simulated arrival directions and detected energies
        at Earth after deflection.

        Composition is accounted for by using the rigidities instead of energies in the lensing process.

        The corresponding coordinates at each point in the simulation is also recorded and written
        into a dictionary. This will be written to the corresponding simulation output file. One can
        use this to plot the different deflections in the future.

        :param kappas: array of vMF spread parameter for each UHECR originating from a source
        :param Nrand: number of samples performed per true UHECR for vMF, higher means more precise lensing
        :param path_to_lens: path to the GMF lensing directory
        '''
        N_uhecr = len(kappas)  # number of UHECRS at gal. boundary
        self.defl_plotvars = {
        }  # container for plotting variables for deflection simulations

        # sample from vMF distribution with obtained kappas to get
        # coordinates at boundary
        omega_gb, src_indices, bg_indices = self._sample_at_gb(kappas, Nrand)
        coords_gb = np.array(
            [uv_to_coord(omega_gb_i) for omega_gb_i in omega_gb])

        # apply magnetic lensing to get energies and coordinates at Earth
        # path to lens is by default "./JF12full_Gamale/lens.cfg", but can be set to anywhere
        # if one wishes to construct more lenses
        if path_to_lens is None:
            path_to_lens = os.path.join(
                os.path.abspath(os.path.dirname(__file__)), "JF12full_Gamale",
                "lens.cfg")
        coords_earth, energies_earth, crmap_unlensed, crmap_lensed = self._apply_lensing(
            coords_gb, N_uhecr, Nrand, path_to_lens)

        # limit using detector exposure
        # energies needed since those will also be truncated by rejected UHECRs
        omega_det_exp_limited, energies_exp_limited = self._apply_exposure_limits(
            coords_earth, energies_earth, N_uhecr)

        # apply normal sampling to energies based on detector energy uncertainty
        energies_exp_limited = energies_exp_limited * 1e-18  # convert back to EeV
        Eerr = self.data.detector.energy_uncertainty
        energies_det_exp_limited = np.random.normal(loc=energies_exp_limited,
                                                    scale=Eerr *
                                                    energies_exp_limited)

        # append to dictionary
        self.defl_plotvars = {
            "omega_gb": omega_gb,
            "src_indices": src_indices,
            "bg_indices": bg_indices,
            "omega_earth": np.array(coord_to_uv(coords_earth)),
            "energies_earth": energies_earth,
            "omega_det_exp_limited": omega_det_exp_limited,
            "energies_det_exp_limited": energies_det_exp_limited,
            "map_unlensed": crmap_unlensed,
            "map_lensed": crmap_lensed
        }

        return omega_det_exp_limited, energies_det_exp_limited

    def _sample_at_gb(self, kappas, Nrand):
        '''
        Sample from the vMF distribution at the galactic boundary. 

        This is identical to what is done in the no_gmf simulation at Earth. 
        '''
        source_labels = self.source_labels
        varpi = self.data.source.unit_vector

        # get source vector in which uhecr is associated with
        # otherwise the UHECR is associated with background
        varpi_per_uhecr = []
        for lmbda in source_labels:
            if lmbda == len(varpi):  # if background
                varpi_per_uhecr.append(None)
            else:
                varpi_per_uhecr.append(np.array(varpi[lmbda]))

        # get UHECR indices correlating to background and source
        src_indices = np.argwhere(source_labels != len(varpi))
        bg_indices = np.argwhere(source_labels == len(varpi))

        # obtain arrival directions at the galactic boundary
        omega_gb = []
        for i, lmbda in enumerate(source_labels):
            if lmbda == len(
                    varpi):  # from background, sample from sphere uniformly
                omega_gb_bg = sample_sphere(1, Nrand)
                omega_gb.append(omega_gb_bg)
                # coords_gb.append(uv_to_coord(omega_gb_bg))
            else:  # from source, sample from vmf
                omega_gb_src = sample_vMF(varpi_per_uhecr[i], kappas[i], Nrand)
                omega_gb.append(omega_gb_src)
                # coords_gb.append(uv_to_coord(omega_gb_src))

        return omega_gb, src_indices, bg_indices

    def _apply_lensing(self, coords_gb, N_uhecr, Nrand, path_to_lens):
        '''
        Apply lensing from GMF obtained from https://www.desy.de/~crpropa/data/magnetic_lenses/ 
        to the arrival directions at the galactic boundary. We use the JF12 model by default.
        Returns energies and arrival directions (SkyCoord) at Earth after deflection.

        Note that we can also construct our own magnetic field lens, however, this can be performed
        for future projects.
        (check https://crpropa.github.io/CRPropa3/pages/example_notebooks/galactic_backtracking/galactic_backtracking.v4.html#
        for details.)

        :param coords_gb: list of SkyCoord arrival directions at galactic boundary
        '''

        energies_gb = self.Earr * crpropa.EeV  # UHECR energy at gal. boundary
        A, Z = self.nuc_table[self.ptype]
        pid = crpropa.nucleusId(A, Z)

        map_container = crpropa.ParticleMapsContainer()

        # add particle to map container
        # coordinate transformation is set to that based on making the final sampling
        # from lensed map to be correct.
        # divide energy by Z to get rigidity (to account for composition)
        # this part takes ~30 secs
        for i in progress_bar(range(N_uhecr),
                              total=N_uhecr,
                              desc="Adding UHECR to Map Container"):
            for j in range(Nrand):
                c_gal = coords_gb[i][j].galactic
                map_container.addParticle(pid,
                                          np.float64(energies_gb[i]) / Z,
                                          np.pi - c_gal.l.rad, c_gal.b.rad)

        # evaluate map used to plot unlensed map using healpy
        NPIX = map_container.getNumberOfPixels(
        )  # hardset to 49152 via CRPropa, correleats to ang. res of 0.92 deg
        crMap_unlensed = np.zeros(NPIX)
        rigidities = map_container.getEnergies(
            int(pid))  # actually getting rigidity
        for rigidity in rigidities:
            crMap_unlensed += map_container.getMap(int(pid),
                                                   rigidity * crpropa.eV)

        # apply lens of b-field model to get map of uhecrs at earth
        # full lens to account for turbulent effects
        lens = crpropa.MagneticLens(path_to_lens)
        lens.normalizeLens()
        map_container.applyLens(lens)

        # same as above, but for lensed version
        crMap_lensed = np.zeros(NPIX)
        rigidities = map_container.getEnergies(int(pid))
        for rigidity in rigidities:
            crMap_lensed += map_container.getMap(int(pid),
                                                 rigidity * crpropa.eV)

        # now generate individual particles from lensed map
        # lon \in [-pi, pi], lats \in [-pi/2, pi/2]
        _, rigidities_earth, lons_earth, lats_earth = map_container.getRandomParticles(
            N_uhecr)

        # multiply by charge to get back energy from rigidity
        energies_earth = Z * rigidities_earth

        # convert to SkyCoord coordinates
        # lon \in [0, 2pi], lats \in [-pi/2, pi/2]
        coords_earth = SkyCoord((np.pi - lons_earth) * u.rad,
                                lats_earth * u.rad,
                                frame="galactic")

        return coords_earth, energies_earth, crMap_unlensed, crMap_lensed

    def _apply_exposure_limits(self,
                               coords_earth,
                               energies_earth,
                               N_uhecr,
                               count_limit=1e3):
        '''
        Apply the exposure from the corresponding detector to the coordinates and energies
        obtained at Earth after lensing. This is done via rejection sampling with as with 
        `exposure_limited_vMF_rng` in `vMF.stan`. 
        Returns an array of unit vectors + energies, corresponding to those limited by the exposure
        of the detector.

        Basic algorithm for rejection + vMF sampling (the Pythonic way, based on same stan code):
        0. Initialize an accepted-rejected container, full of zeros. This will be updated for each iteration with 1's, and 
        loop terminates when this array only contains 1's (i.e. all directions accepted).
        1. sample from vMF distribution with ang_err as spread for ALL directions
        2. evaluate the probability to detect the UHECR at that declination based on exposure function m(dec_det) (given as pdet)
        3. use Bernoulli distribution (2-D categorical distribution) to sample accepted (1) or rejected (0) for
        each sampled direction, based on pdet above.
        4. find the indices that are (a) are not accepted by the accepted-rejected container, and (b) where the sampled values
        are only the accepted ones
        5. append them to the container with same indices
        6. append the corresponding omega_det with those indices only.
        7. terminate when either (a) all values in container are accepted ones, or (b) the count limit is exceeded (1e7 in stan code)
        '''

        # get unit vector of coordinates at earth
        omega_true = np.array(coord_to_uv(coords_earth))

        # initialization of algorithm
        omega_det_exp_limited = np.zeros((N_uhecr, 3))
        accepted_rejected_container = np.zeros(N_uhecr)
        count = 0

        print("Performing truncations due to exposure...")
        while len(np.nonzero(accepted_rejected_container)) != N_uhecr:
            # sample from vMF distribution with angular uncertainty
            omega_det = np.array([
                sample_vMF(omega_true_i, self.data.detector.kappa_d, 1)
                for omega_true_i in omega_true
            ])
            omega_det = omega_det[:, 0, :]  # to collapse the array size

            # evaluate probability to detect omega with given exposure using exposure function
            dec_det = np.pi / 2. - np.arccos(
                omega_det[:, 2])  # shift since arccos \in [0, pi]
            m_omega = np.array(
                [m_dec(d, self.data.detector.params) for d in dec_det])
            pdet = (m_omega / self.data.detector.exposure_max)

            # sample from bernoulli distribution (2-D categorical distribution)
            samples = bernoulli.rvs(pdet, size=N_uhecr)

            # get indices where samples != 0 and where container == 0
            # i.e. accepted indices in this loop which are not accounted for in accepted yet
            sample_nonzero_indices = np.argwhere(
                (accepted_rejected_container == 0) & (samples != 0))[:, 0]

            # append to locations where accepted_rejected_container == 0
            accepted_rejected_container[sample_nonzero_indices] = samples[
                sample_nonzero_indices]

            # append the evaluated omega_det in this iteration to the same locations where
            # sampling != 0 and container == 0
            omega_det_exp_limited[sample_nonzero_indices, :] = omega_det[
                sample_nonzero_indices, :]

            count += 1
            # if count exceeds some truncation limit, break the loop
            if count > count_limit:
                break

        # remove all directions that are stuck (i.e. container == 0), as well as corresp. energies
        container_nonzero_indices = np.nonzero(accepted_rejected_container)[0]
        omega_det_exp_limited = omega_det_exp_limited[
            container_nonzero_indices]
        energies_exp_limited = energies_earth[container_nonzero_indices]

        return omega_det_exp_limited, energies_exp_limited

    def _prepare_fit_inputs(self):
        """
        Gather inputs from Model, Data and IntegrationTables.
        """

        eps_fit = self.tables.table
        kappa_grid = self.tables.kappa
        E_grid = self.E_grid
        Earr_grid = list(self.Earr_grid)

        # KW: due to multiprocessing appending,
        # collapse dimension from (1, 23, 50) -> (23, 50)
        eps_fit.resize(self.Earr_grid.shape)

        # handle selected sources
        if (self.data.source.N < len(eps_fit)):
            eps_fit = [eps_fit[i] for i in self.data.source.selection]
            Earr_grid = [Earr_grid[i] for i in self.data.source.selection]

        # add E interpolation for background component (possible extension with Dbg)
        Earr_grid.append([0 for e in E_grid])

        # convert scale for sampling
        D = self.data.source.distance
        alpha_T = self.data.detector.alpha_T
        D, alpha_T, eps_fit = convert_scale(D, alpha_T, eps_fit)

        # prepare fit inputs
        self.fit_input = {
            'Ns': self.data.source.N,
            'varpi': self.data.source.unit_vector,
            'D': D,
            'N': self.data.uhecr.N,
            'arrival_direction': self.data.uhecr.unit_vector,
            'A': self.data.uhecr.A,
            'alpha_T': alpha_T,
            'Ngrid': len(kappa_grid),
            'eps': eps_fit,
            'kappa_grid': kappa_grid,
            'zenith_angle': self.data.uhecr.zenith_angle
        }

        if self.analysis_type == self.joint_type \
                or self.analysis_type == self.E_loss_type \
                or self.analysis_type == self.gmf_type:

            self.fit_input['Edet'] = self.data.uhecr.energy
            self.fit_input['Eth'] = self.model.Eth
            self.fit_input['Eerr'] = self.data.detector.energy_uncertainty
            self.fit_input['E_grid'] = E_grid
            self.fit_input['Earr_grid'] = Earr_grid

        if self.analysis_type == self.gmf_type:
            ptype = str(self.data.uhecr.ptype)
            _, self.fit_input["Z"] = self.nuc_table[ptype]
            self.fit_input["kappa_gmf"] = self.data.uhecr.kappa_gmf
        else:
            self.fit_input["kappa_d"] = self.data.detector.kappa_d

    def save(self):
        """
        Write the analysis to file.
        """

        with h5py.File(self.filename, 'r+') as f:

            source_handle = f.create_group('source')
            if self.data.source:
                self.data.source.save(source_handle)

            uhecr_handle = f.create_group('uhecr')
            if self.data.uhecr:
                self.data.uhecr.save(uhecr_handle)

            detector_handle = f.create_group('detector')
            if self.data.detector:
                self.data.detector.save(detector_handle)

            model_handle = f.create_group('model')
            if self.model:
                self.model.save(model_handle)

            fit_handle = f.create_group('fit')
            if self.fit:

                # fit inputs
                fit_input_handle = fit_handle.create_group('input')
                for key, value in self.fit_input.items():
                    fit_input_handle.create_dataset(key, data=value)

                # samples
                samples = fit_handle.create_group('samples')
                for key, value in self.chain.items():
                    samples.create_dataset(key, data=value)

            else:
                plotvars_handle = f.create_group("plotvars")

                if self.analysis_type == self.gmf_type:
                    for key, value in list(self.defl_plotvars.items()):
                        plotvars_handle.create_dataset(key, data=value)

    def plot(self, type=None, cmap=None):
        """
        Plot the data associated with the analysis object.

        type == 'arrival direction':
        Plot the arrival directions on a skymap, 
        with a colour scale describing which source 
        the UHECR is from.

        type == 'energy'
        Plot the simulated energy spectrum from the 
        source, to after propagation (arrival) and 
        detection
        """

        # plot style
        if cmap == None:
            cmap = plt.cm.get_cmap('viridis')

        # plot arrival directions by default
        if type == None:
            type == 'arrival_direction'

        if type == 'arrival_direction':

            # figure
            fig, ax = plt.subplots()
            fig.set_size_inches((12, 6))

            # skymap
            skymap = AllSkyMap(projection='hammer', lon_0=0, lat_0=0)

            self.data.source.plot(skymap)
            self.data.detector.draw_exposure_lim(skymap)
            self.data.uhecr.plot(skymap)

            # standard labels and background
            skymap.draw_standard_labels()

            # legend
            ax.legend(frameon=False, bbox_to_anchor=(0.85, 0.85))

        if type == 'energy':

            bins = np.logspace(np.log(self.model.Eth), np.log(1e4), base=np.e)

            fig, ax = plt.subplots()

            if isinstance(self.E, (list, np.ndarray)):
                ax.hist(self.E,
                        bins=bins,
                        alpha=0.7,
                        label=r'$\tilde{E}$',
                        color=cmap(0.0))
            if isinstance(self.Earr, (list, np.ndarray)):
                ax.hist(self.Earr,
                        bins=bins,
                        alpha=0.7,
                        label=r'$E$',
                        color=cmap(0.5))

            ax.hist(self.data.uhecr.energy,
                    bins=bins,
                    alpha=0.7,
                    label=r'$\hat{E}$',
                    color=cmap(1.0))

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(frameon=False)

    def use_crpropa_data(self, energy, unit_vector):
        """
        Build fit inputs from the UHECR dataset.
        """

        self.N = len(energy)
        self.arrival_direction = Direction(unit_vector)

        # simulate the zenith angles
        print('Simulating zenith angles...')
        self.zenith_angles = self._simulate_zenith_angles()
        print('Done!')

        # Make Uhecr object
        uhecr_properties = {}
        uhecr_properties['label'] = 'sim_uhecr'
        uhecr_properties['N'] = self.N
        uhecr_properties['unit_vector'] = self.arrival_direction.unit_vector
        uhecr_properties['energy'] = energy
        uhecr_properties['zenith_angle'] = self.zenith_angles
        uhecr_properties['A'] = np.tile(self.data.detector.area, self.N)

        new_uhecr = Uhecr()
        new_uhecr.from_properties(uhecr_properties)

        self.data.uhecr = new_uhecr

    def fit_model(self,
                  iterations=1000,
                  chains=4,
                  seed=None,
                  sample_file=None,
                  warmup=None):
        """
        Fit a model.

        :param iterations: number of iterations
        :param chains: number of chains
        :param seed: seed for RNG
        """

        # Prepare fit inputs
        self._prepare_fit_inputs()

        # fit
        print("Performing fitting...")
        self.fit = self.model.model.sampling(data=self.fit_input,
                                             iter=iterations,
                                             chains=chains,
                                             seed=seed,
                                             sample_file=sample_file,
                                             warmup=warmup)

        # Diagnositics
        print("Checking all diagnostics...")
        stan_utility.utils.check_all_diagnostics(self.fit)

        self.chain = self.fit.extract(permuted=True)
        print("Done!")
        return self.fit
