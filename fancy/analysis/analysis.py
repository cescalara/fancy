import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
from matplotlib import pyplot as plt
import h5py

import stan_utility

from ..interfaces.integration import ExposureIntegralTable
from ..interfaces.stan import Direction, convert_scale
from ..interfaces.data import Uhecr
from ..plotting import AllSkyMap
from ..propagation.energy_loss import get_Eth_src, get_kappa_ex, get_Eex, get_Eth_sim, get_arrival_energy


__all__ = ['Analysis']


class Analysis():
    """
    To manage the running of simulations and fits based on Data and Model objects.
    """

    def __init__(self, data, model, analysis_type = None, filename = None, summary = b''):
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

        self.arr_dir_type = 'arrival direction'
        self.E_loss_type = 'energy loss'
        self.joint_type = 'joint'

        if analysis_type == None:
            analysis_type = self.arr_dir_type
            
        self.analysis_type = analysis_type

        if self.analysis_type == 'joint':

            # find lower energy threshold for the simulation, given Eth and Eerr
            self.model.Eth_sim = get_Eth_sim(self.data.detector.energy_uncertainty, self.model.Eth)

            # find correspsonding Eth_src
            self.Eth_src = get_Eth_src(self.model.Eth_sim, self.data.source.distance)

        # Set up integral tables
        params = self.data.detector.params
        varpi = self.data.source.unit_vector
        self.tables = ExposureIntegralTable(varpi = varpi, params = params)
        
            
    def build_tables(self, num_points = 50, sim_only = False, fit_only = False):
        """
        Build the necessary integral tables.
        """

        if sim_only:
            
            # kappa_true table for simulation
            if self.analysis_type == self.arr_dir_type or self.analysis_type == self.E_loss_type:
                kappa_true = self.model.kappa

            if self.analysis_type == self.joint_type:
                self.Eex = get_Eex(self.Eth_src, self.model.alpha)
                self.kappa_ex = get_kappa_ex(self.Eex, self.model.B, self.data.source.distance)        
                kappa_true = self.kappa_ex

            self.tables.build_for_sim(kappa_true, self.model.alpha, self.model.B, self.data.source.distance)
    
        if fit_only:

            # logarithmically spcaed array with 60% of points between KAPPA_MIN and 100
            kappa_first = np.logspace(np.log(1), np.log(10), int(num_points * 0.7), base = np.e)
            kappa_second = np.logspace(np.log(10), np.log(100), int(num_points * 0.2) + 1, base = np.e)
            kappa_third = np.logspace(np.log(100), np.log(1000), int(num_points * 0.1) + 1, base = np.e)
            kappa = np.concatenate((kappa_first, kappa_second[1:], kappa_third[1:]), axis = 0)
        
            # full table for fit
            self.tables.build_for_fit(kappa)

    def build_energy_table(self, num_points = 50, input_filename = None):
        """
        Build the energy interpolation tables.
        """

        self.E_grid = np.logspace(np.log(self.model.Eth), np.log(1.0e4), num_points, base = np.e)
        self.Earr_grid = []
        
        for i, d in enumerate(self.data.source.distance):
            print(i, d)
            self.Earr_grid.append([get_arrival_energy(e, d)[0] for e in self.E_grid])

        if input_filename:
            with h5py.File(input_filename, 'r+') as f:
                E_group = f.create_group('energy')
                E_group.create_dataset('E_grid', data = self.E_grid)
                E_group.create_dataset('Earr_grid', data = self.Earr_grid)
                
            
    def use_tables(self, input_filename, main_only = True):
        """
        Pass in names of integral tables that have already been made.
        Only the main table is read in by default, the simulation table 
        must be recalculated every time the simulation parameters are 
        changed.
        """

        if main_only:
            input_table = ExposureIntegralTable(input_filename = input_filename)
            self.tables.table = input_table.table
            self.tables.kappa = input_table.kappa

            if self.analysis_type == self.joint_type:
                with h5py.File(input_filename, 'r') as f:
                    self.E_grid = f['energy/E_grid'].value
                    self.Earr_grid = f['energy/Earr_grid'].value
            
        else:
            self.tables = ExposureIntegralTable(input_filename = input_filename)

        
    def _get_zenith_angle(self, c_icrs, loc, time):
        """
        Calculate the zenith angle of a known point 
        in ICRS (equatorial coords) for a given 
        location and time.
        """
        c_altaz = c_icrs.transform_to(AltAz(obstime = time, location = loc))
        return (np.pi/2 - c_altaz.alt.rad)


    def _simulate_zenith_angles(self):
        """
        Simulate zenith angles for a set of arrival_directions.
        """

        start_time = 2004

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
                    t = start_time + dt
                else:
                    t = time[-1] + dt
                tdy = Time(t, format = 'decimalyear')
                za = self._get_zenith_angle(d, self.data.detector.location, tdy)
        
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
            print('Warning: % of zenith angles stuck is', len(stuck)/len(zenith_angles) * 100)

        return zenith_angles

    
    def simulate(self, seed = None, Eth_sim = None):
        """
        Run a simulation.

        :param seed: seed for RNG
        :param Eth_sim: the minimun energy simulated
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
            

        if self.analysis_type == self.joint_type or self.analysis_type == self.E_loss_type:
            # find lower energy threshold for the simulation, given Eth and Eerr
            if Eth_sim:
                self.model.Eth_sim = Eth_sim

            
        # compile inputs from Model and Data
        self.simulation_input = {
            'kappa_d' : self.data.detector.kappa_d, 
            'Ns' : len(self.data.source.distance),
            'varpi' : self.data.source.unit_vector, 
            'D' : D,
            'A' : self.data.detector.area,
            'a0' : self.data.detector.location.lat.rad,
            'lon' : self.data.detector.location.lon.rad,
            'theta_m' : self.data.detector.threshold_zenith_angle.rad, 
            'alpha_T' : alpha_T,
            'eps' : eps}

        self.simulation_input['L'] = L
        self.simulation_input['F0'] = F0
        self.simulation_input['distance'] = self.data.source.distance
        if self.analysis_type == self.arr_dir_type or self.analysis_type == self.E_loss_type:

            self.simulation_input['kappa'] = self.model.kappa

        if self.analysis_type == self.E_loss_type:

            self.simulation_input['alpha'] = self.model.alpha
            self.simulation_input['Eth'] = self.model.Eth_sim
            self.simulation_input['Eerr'] = self.data.detector.energy_uncertainty
            
        if self.analysis_type == self.joint_type:
            
            self.simulation_input['B'] = self.model.B    
            self.simulation_input['alpha'] = self.model.alpha
            self.simulation_input['Eth'] = self.model.Eth_sim
            self.simulation_input['Eerr'] = self.data.detector.energy_uncertainty

        try:
            if self.data.source.flux:
                self.simulation_input['flux'] = self.data.source.flux
            else:
                self.simulation_input['flux'] = np.zeros(self.data.source.N)
        except:
            self.simulation_input['flux'] = np.zeros(self.data.source.N)
        
        # run simulation
        print('Running stan simulation...')
        self.simulation = self.model.simulation.sampling(data = self.simulation_input, iter = 1,
                                                         chains = 1, algorithm = "Fixed_param", seed = seed)

        # extract output
        print('Extracting output...')
        self.Nex_sim = self.simulation.extract(['Nex_sim'])['Nex_sim']
        arrival_direction = self.simulation.extract(['arrival_direction'])['arrival_direction'][0]
        self.source_labels = (self.simulation.extract(['lambda'])['lambda'][0] - 1).astype(int)
    
        if self.analysis_type == self.joint_type or self.analysis_type == self.E_loss_type:
            
            self.Edet = self.simulation.extract(['Edet'])['Edet'][0]
            self.Earr = self.simulation.extract(['Earr'])['Earr'][0]
            self.E = self.simulation.extract(['E'])['E'][0]

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
        self.zenith_angles = self._simulate_zenith_angles()
        print('Done!')

        # Make uhecr object
        uhecr_properties = {}
        uhecr_properties['label'] = 'sim_uhecr'
        uhecr_properties['N'] = self.N
        uhecr_properties['unit_vector'] = self.arrival_direction.unit_vector
        uhecr_properties['energy'] = self.Edet
        uhecr_properties['zenith_angle'] = self.zenith_angles
        uhecr_properties['A'] = np.tile(self.data.detector.area, self.N)       

        new_uhecr = Uhecr()
        new_uhecr.from_properties(uhecr_properties)
        
        self.data.uhecr = new_uhecr
        

    def _prepare_fit_inputs(self):
        """
        Gather inputs from Model, Data and IntegrationTables.
        """
        
        eps_fit = self.tables.table
        kappa_grid = self.tables.kappa
        E_grid = self.E_grid
        Earr_grid = list(self.Earr_grid)
        
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
        self.fit_input = {'Ns' : self.data.source.N, 
                          'varpi' :self.data.source.unit_vector,
                          'D' : D, 
                          'N' : self.data.uhecr.N, 
                          'arrival_direction' : self.data.uhecr.unit_vector, 
                          'A' : self.data.uhecr.A,
                          'kappa_d' : self.data.detector.kappa_d,
                          'alpha_T' : alpha_T, 
                          'Ngrid' : len(kappa_grid), 
                          'eps' : eps_fit, 
                          'kappa_grid' : kappa_grid,
                          'zenith_angle' : self.data.uhecr.zenith_angle}

        if self.analysis_type == self.joint_type or self.analysis_type == self.E_loss_type:
            
            self.fit_input['Edet'] = self.data.uhecr.energy
            self.fit_input['Eth'] = self.model.Eth
            self.fit_input['Eerr'] = self.data.detector.energy_uncertainty
            self.fit_input['E_grid'] = E_grid
            self.fit_input['Earr_grid'] = Earr_grid
            
        
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
                    fit_input_handle.create_dataset(key, data = value)

                # samples
                samples = fit_handle.create_group('samples')
                for key, value in self.chain.items():
                    samples.create_dataset(key, data = value)                
     
            
    def plot(self, type = None, cmap = None):
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
            type == 'arrival direction'
        
        if type == 'arrival direction':
    
            # figure
            fig, ax = plt.subplots();
            fig.set_size_inches((12, 6))

            # skymap
            skymap = AllSkyMap(projection = 'hammer', lon_0 = 0, lat_0 = 0);

            self.data.source.plot(skymap)
            self.data.detector.draw_exposure_lim(skymap)
            self.data.uhecr.plot(skymap, source_labels = self.source_labels)
        
           # standard labels and background
            skymap.draw_standard_labels()

            # legend
            ax.legend(frameon = False, bbox_to_anchor = (0.85, 0.85))

        if type == 'energy':

            bins = np.logspace(np.log(self.model.Eth), np.log(1e4), base = np.e)

            fig, ax = plt.subplots()

            if isinstance(self.E, (list, np.ndarray)):
                ax.hist(self.E, bins = bins, alpha = 0.7, label = r'$\tilde{E}$', color = cmap(0.0))
            if isinstance(self.Earr, (list, np.ndarray)):
                ax.hist(self.Earr, bins = bins, alpha = 0.7, label = r'$E$', color = cmap(0.5))

            ax.hist(self.data.uhecr.energy, bins = bins, alpha = 0.7, label = r'$\hat{E}$', color = cmap(1.0))

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend(frameon = False)
    

                
    def use_uhecr_data(self):
        """
        Build fit inputs from the UHECR dataset.
        """

        eps_fit = self.tables.table
        kappa_grid = self.tables.kappa

        if self.analysis_type == self.joint_type:
            E_grid = self.E_grid
            Earr_grid = list(self.Earr_grid)
        
        # handle selected sources
        if (self.data.source.N < len(eps_fit)):
            eps_fit = [eps_fit[i] for i in self.data.source.selection]
            if self.analysis_type == self.joint_type:
                Earr_grid = [Earr_grid[i] for i in self.data.source.selection]

        if self.analysis_type == self.joint_type:
            # add E interpolation for background component (possible extension with Dbg)
            Earr_grid.append([0 for e in E_grid])
            
        # convert scale for sampling
        D = self.data.source.distance
        alpha_T = self.data.detector.alpha_T
        D, alpha_T, eps_fit = convert_scale(D, alpha_T, eps_fit)
                
        print('preparing fit inputs...')
        self.fit_input = {'Ns' : self.data.source.N,
                          'varpi' :self.data.source.unit_vector,
                          'D' : D,
                          'N' : self.data.uhecr.N,
                          'arrival_direction' : self.data.uhecr.unit_vector,
                          'A' : self.data.uhecr.A,
                          'kappa_d' : self.data.detector.kappa_d,
                          'alpha_T' : alpha_T,
                          'Ngrid' : len(kappa_grid),
                          'eps' : eps_fit,
                          'kappa_grid' : kappa_grid,
                          'zenith_angle' : np.deg2rad(self.data.uhecr.zenith_angle)}
        
        try:
            self.fit_input['flux'] = self.data.source.flux
        except:
            print('No flux weights available for sources.')
        
        if self.analysis_type == self.joint_type:

            self.fit_input['Edet'] = self.data.uhecr.energy
            self.fit_input['Eth'] = self.model.Eth
            self.fit_input['Eerr'] = self.data.detector.energy_uncertainty
            self.fit_input['E_grid'] = E_grid
            self.fit_input['Earr_grid'] = Earr_grid
            
        print('done')

    def use_crpropa_data(self, energy, arrival_direction):
        """
        Build fit inputs from the UHECR dataset.
        """

        self.N = len(energy)
        self.arrival_direction = arrival_direction 
        self.energy = energy
        
        eps_fit = self.tables.table
        kappa_grid = self.tables.kappa
        E_grid = self.E_grid
        Earr_grid = list(self.Earr_grid)
        
        # handle selected sources
        if (self.data.source.N < len(eps_fit)):
            eps_fit = [eps_fit[i] for i in self.data.source.selection]
            Earr_grid = [Earr_grid[i] for i in self.data.source.selection]

        # add E interpolation for background component
        Earr_grid.append([0 for e in E_grid])
            
        # convert scale for sampling
        D = self.data.source.distance
        alpha_T = self.data.detector.alpha_T
        D, alpha_T, eps_fit = convert_scale(D, alpha_T, eps_fit)

        # simulate the zenith angles
        print('simulating zenith angles...')
        self.zenith_angles = self._simulate_zenith_angles()
        print('done')
        
        print('preparing fit inputs...')
        self.fit_input = {'Ns' : self.data.source.N,
                          'varpi' :self.data.source.unit_vector,
                          'D' : D,
                          'N' : self.N,
                          'arrival_direction' : self.arrival_direction.unit_vector,
                          'A' : np.tile(self.data.detector.area, self.N),
                          'kappa_d' : self.data.detector.kappa_d,
                          'alpha_T' : alpha_T,
                          'Ngrid' : len(kappa_grid),
                          'eps' : eps_fit,
                          'kappa_grid' : kappa_grid,
                          'zenith_angle' : self.zenith_angles}
        
        try:
            self.fit_input['flux'] = self.data.source.flux
        except:
            print('No flux weights available for sources.')
        
        if self.analysis_type == self.joint_type:

            self.fit_input['Edet'] = energy
            self.fit_input['Eth'] = self.model.Eth
            self.fit_input['Eerr'] = self.data.detector.energy_uncertainty
            self.fit_input['E_grid'] = E_grid
            self.fit_input['Earr_grid'] = Earr_grid
            
        print('done')

        
    def fit_model(self, iterations = 1000, chains = 4, seed = None, sample_file = None, warmup = None):
        """
        Fit a model.

        :param iterations: number of iterations
        :param chains: number of chains
        :param seed: seed for RNG
        """

        # Prepare fit inputs
        self._prepare_fit_inputs()
        
        # fit
        self.fit = self.model.model.sampling(data = self.fit_input, iter = iterations, chains = chains, seed = seed,
                                             sample_file = sample_file, warmup = warmup)

        # Diagnositics
        stan_utility.utils.check_all_diagnostics(self.fit)
        
        self.chain = self.fit.extract(permuted = True)
        return self.fit
        
        
