import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
from matplotlib import pyplot as plt
import h5py

from ..interfaces.integration import ExposureIntegralTable
from ..interfaces.stan import Direction, convert_scale
from ..interfaces import stan_utility
from ..utils import PlotStyle
from ..plotting import AllSkyMap
from ..propagation.energy_loss import get_Eth_src, get_kappa_ex, get_Eex, get_Eth_sim, get_arrival_energy


__all__ = ['Analysis']


class Analysis():
    """
    To manage the running of simulations and fits based on Data and Model objects.
    """

    def __init__(self, data, model, analysis_type = None, filename = None):
        """
        To manage the running of simulations and fits based on Data and Model objects.
        
        :param data: a Data object
        :param model: a Model object
        :param analysis_type: type of analysis
        """

        self.data = data
        self.model = model
        self.filename = filename
        if self.filename:
            with h5py.File(self.filename, 'r+') as f:
                f.create_group('input')
                f.create_group('output')
            
        self.simulation_input = None
        self.fit_input = None
        
        self.simulation = None
        self.fit = None

        self.arr_dir_type = 'arrival direction'
        self.E_loss_type = 'energy loss'
        self.joint_type = 'joint'

        if analysis_type == None:
            analysis_type = self.arr_dir_type
        self.analysis_type = analysis_type

        if self.analysis_type == 'joint':
            # find lower energy threshold for the simulation, given Eth and Eerr
            self.Eth_sim = get_Eth_sim(self.model.Eerr, self.model.Eth)
            #self.Eth_sim = 52
            # find correspsonding Eth_src
            self.Eth_src = get_Eth_src(self.Eth_sim, self.data.source.distance)

        params = self.data.detector.params
        varpi = self.data.source.unit_vector
        self.tables = ExposureIntegralTable(varpi = varpi, params = params)
     
            
    def build_tables(self, num_points = 50, sim_only = False, fit_only = False):
        """
        Build the necessary integral tables.
        """

        if not fit_only:
            
            # kappa_true table for simulation
            if self.analysis_type == self.arr_dir_type or self.analysis_type == self.E_loss_type:
                kappa_true = self.model.kappa

            if self.analysis_type == self.joint_type:
                self.Eex = get_Eex(self.Eth_src, self.model.alpha)
                self.kappa_ex = get_kappa_ex(self.Eex, self.model.B, self.data.source.distance)        
                kappa_true = self.kappa_ex

            self.tables.build_for_sim(kappa_true, self.model.alpha, self.model.B, self.data.source.distance)
    
        if not sim_only:

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
        F1 = self.model.F1
        Dbg = self.model.Dbg
        D, Dbg, alpha_T, eps, F0, F1, L = convert_scale(D, Dbg, alpha_T, eps, F0, F1, L)
            

        if self.analysis_type == self.joint_type or self.analysis_type == self.E_loss_type:
            # find lower energy threshold for the simulation, given Eth and Eerr
            if Eth_sim:
                self.Eth_sim = Eth_sim
            print('simulating down to', self.Eth_sim, 'EeV...')

            
        # compile inputs from Model and Data
        self.simulation_input = {
                       'kappa_c' : self.data.detector.kappa_c, 
                       'Ns' : len(self.data.source.distance),
                       'varpi' : self.data.source.unit_vector, 
                       'D' : D,
                       'A' : self.data.detector.area,
                       'a0' : self.data.detector.location.lat.rad,
                       'theta_m' : self.data.detector.threshold_zenith_angle.rad, 
                       'alpha_T' : alpha_T,
                       'eps' : eps}

        self.simulation_input['L'] = L
        self.simulation_input['F0'] = F0
        self.simulation_input['F1'] = F1
        self.simulation_input['Dbg'] = Dbg
          
        if self.analysis_type == self.arr_dir_type or self.analysis_type == self.E_loss_type:

            self.simulation_input['kappa'] = self.model.kappa

        if self.analysis_type == self.E_loss_type:

            self.simulation_input['alpha'] = self.model.alpha
            self.simulation_input['Eth'] = self.Eth_sim
            self.simulation_input['Eerr'] = self.model.Eerr
            
        if self.analysis_type == self.joint_type:
            
            self.simulation_input['B'] = self.model.B    
            self.simulation_input['alpha'] = self.model.alpha
            self.simulation_input['Eth'] = self.Eth_sim
            self.simulation_input['Eerr'] = self.model.Eerr

        try:
            if self.data.source.flux:
                self.simulation_input['flux'] = self.data.source.flux
            else:
                self.simulation_input['flux'] = np.zeros(self.data.source.N)
        except:
            self.simulation_input['flux'] = np.zeros(self.data.source.N)
            print('No flux weights used in simulation.')
        
        # run simulation
        print('running stan simulation...')
        self.simulation = self.model.simulation.sampling(data = self.simulation_input, iter = 1,
                                                         chains = 1, algorithm = "Fixed_param", seed = seed)

        print('done')

        # extract output
        print('extracting output...')
        self.Nex_sim = self.simulation.extract(['Nex_sim'])['Nex_sim']
        arrival_direction = self.simulation.extract(['arrival_direction'])['arrival_direction'][0]
        self.labels = (self.simulation.extract(['lambda'])['lambda'][0] - 1).astype(int)
    
        if self.analysis_type == self.joint_type or self.analysis_type == self.E_loss_type:
            
            self.Edet = self.simulation.extract(['Edet'])['Edet'][0]
            self.Earr = self.simulation.extract(['Earr'])['Earr'][0]
            self.E = self.simulation.extract(['E'])['E'][0]

            # make cut on Eth
            print('making cut at Eth =', self.model.Eth, 'EeV...')
            inds = np.where(self.Edet >= self.model.Eth)
            self.Edet = self.Edet[inds]
            arrival_direction = arrival_direction[inds]
            self.labels = self.labels[inds]
            print(len(arrival_direction), 'events above', self.model.Eth, 'EeV...')
        
        # convert to Direction object
        self.arrival_direction = Direction(arrival_direction)
        self.N = len(self.arrival_direction.unit_vector)
        print('done')

        
        # simulate the zenith angles
        print('simulating zenith angles...')
        self.zenith_angles = self._simulate_zenith_angles()
        print('done')
        
        eps_fit = self.tables.table
        kappa_grid = self.tables.kappa
        E_grid = self.E_grid
        Earr_grid = list(self.Earr_grid)
        
        # handle selected sources
        if (self.data.source.N < len(eps_fit)):
            eps_fit = [eps_fit[i] for i in self.data.source.selection]
            Earr_grid = [Earr_grid[i] for i in self.data.source.selection]

        # add E interpolation for Dbg
        #Earr_grid.append([get_arrival_energy(e, self.model.Dbg) for e in E_grid])    
        Earr_grid.append([0 for e in E_grid])
        
        # convert scale for sampling
        D = self.data.source.distance
        alpha_T = self.data.detector.alpha_T
        L = self.model.L
        F0 = self.model.F0
        F1 = self.model.F1
        Dbg = self.model.Dbg
        D, Dbg, alpha_T, eps_fit, F0, F1, L = convert_scale(D, Dbg, alpha_T, eps_fit, F0, F1, L)
            
        # prepare fit inputs
        print('preparing fit inputs...')
        self.fit_input = {'Ns' : self.data.source.N, 
                          'varpi' :self.data.source.unit_vector,
                          'D' : D, 
                          'N' : self.N, 
                          'arrival_direction' : self.arrival_direction.unit_vector, 
                          'A' : np.tile(self.data.detector.area, self.N),
                          'kappa_c' : self.data.detector.kappa_c,
                          'alpha_T' : alpha_T, 
                          'Ngrid' : len(kappa_grid), 
                          'eps' : eps_fit, 
                          'kappa_grid' : kappa_grid,
                          'zenith_angle' : self.zenith_angles,
                          'Dbg' : Dbg}


        if self.analysis_type == self.joint_type or self.analysis_type == self.E_loss_type:
            
            self.fit_input['Edet'] = self.Edet
            self.fit_input['Eth'] = self.model.Eth
            self.fit_input['Eerr'] = self.model.Eerr
            self.fit_input['E_grid'] = E_grid
            self.fit_input['Earr_grid'] = Earr_grid

        try:
            if self.data.source.flux:
                self.simulation_input['flux'] = self.data.source.flux
            else:
                self.simulation_input['flux'] = np.zeros(self.data.source.N)
        except:
            print('No flux weights available for sources.')
      
            
        print('done')
        
        
    def save_simulation(self):
        """
        Write the simulated data to file.
        """
        if self.fit_input != None:
            
            with h5py.File(self.filename, 'r+') as f:

                # inputs
                sim_inputs = f['input'].create_group('simulation')
                for key, value in self.simulation_input.items():
                    sim_inputs.create_dataset(key, data = value)
                sim_inputs.create_dataset('kappa_ex', data = self.kappa_ex)

                # outputs
                sim_outputs = f['output'].create_group('simulation')
                sim_outputs.create_dataset('E', data = self.E)
                sim_outputs.create_dataset('Earr', data = self.Earr)
                sim_outputs.create_dataset('Edet', data = self.Edet)
                sim_outputs.create_dataset('Nex_sim', data = self.Nex_sim)                
                sim_fit_inputs = f['output/simulation'].create_group('fit_input')
                for key, value in self.fit_input.items():
                    sim_fit_inputs.create_dataset(key, data = value)
        else:
            print("Error: nothing to save!")

            
    def plot_simulation(self, type = None, cmap = None):
        """
        Plot the simulated data.
        
        type == 'arrival direction':
        Plot the arrival directions on a skymap, 
        with a colour scale describing which source 
        the UHECR is from (background in black).

        type == 'energy'
        Plot the simulated energy spectrum from the 
        source, to after propagation (arrival) and 
        detection
        """

        # plot arrival directions by default
        if type == None:
            type == 'arrival direction'
        
        if type == 'arrival direction':

            # plot style
            if cmap == None:
                style = PlotStyle()
            else:
                style = PlotStyle(cmap_name = cmap)
            
            # figure
            fig = plt.figure(figsize = (12, 6));
            ax = plt.gca()

            # skymap
            skymap = AllSkyMap(projection = 'hammer', lon_0 = 0, lat_0 = 0);

            self.data.source.plot(style, skymap)
            self.data.detector.draw_exposure_lim(skymap)
       
            Ns = self.data.source.N
            cmap = plt.cm.get_cmap('plasma', Ns + 2) 
            label = True

            try:
                self.lables = self.labels
            except:
                self.labels = np.ones(len(self.arrival_direction.lons))

            for lon, lat, lab in np.nditer([self.arrival_direction.lons, self.arrival_direction.lats, self.labels]):
                color = cmap(lab)
                if label:
                    skymap.tissot(lon, lat, 4.0, npts = 30, facecolor = color,
                                  alpha = 0.5, label = 'simulated data')
                    label = False
                else:
                    skymap.tissot(lon, lat, 4.0, npts = 30, facecolor = color, alpha = 0.5)

            # standard labels and background
            skymap.draw_standard_labels(style.cmap, style.textcolor)

            # legend
            plt.legend(bbox_to_anchor = (0.85, 0.85))
            leg = ax.get_legend()
            frame = leg.get_frame()
            frame.set_linewidth(0)
            frame.set_facecolor('None')
            for text in leg.get_texts():
                plt.setp(text, color = style.textcolor)

        if type == 'energy':

            bins = np.logspace(np.log(self.model.Eth), np.log(1e4), base = np.e)
            plt.hist(self.E, bins = bins, alpha = 0.7, label = 'source')
            plt.hist(self.Earr, bins = bins, alpha = 0.7, label = 'arrival')
            plt.hist(self.Edet, bins = bins, alpha = 0.7, label = 'detection')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()
    
        
    def use_simulation(self, input_filename):
        """
        Read in simulated data from a file to create fit_input.
        """

        self.simulation_input = {}
        self.fit_input = {}
        with h5py.File(input_filename, 'r') as f:
            

            sim_input = f['input/simulation']
            for key in sim_input:
                self.simulation_input[key] = sim_input[key].value
                
            sim_output = f['output/simulation']
            self.E = sim_output['E'].value
            self.Earr = sim_output['Earr'].value
            self.Edet = sim_output['Edet'].value
                
            sim_fit_input = sim_output['fit_input']
            for key in sim_fit_input:
                self.fit_input[key] = sim_fit_input[key].value
            self.arrival_direction = Direction(self.fit_input['arrival_direction'])
            

                
    def use_uhecr_data(self):
        """
        Build fit inputs from the UHECR dataset.
        """

        eps_fit = self.tables.table
        kappa_grid = self.tables.kappa
        E_grid = self.E_grid
        Earr_grid = list(self.Earr_grid)
        
        # handle selected sources
        if (self.data.source.N < len(eps_fit)):
            eps_fit = [eps_fit[i] for i in self.data.source.selection]
            Earr_grid = [Earr_grid[i] for i in self.data.source.selection]

        # add E interpolation for Dbg
        #Earr_grid.append([get_arrival_energy(e, self.model.Dbg) for e in E_grid])
        Earr_grid.append([0 for e in E_grid])
            
        # convert scale for sampling
        D = self.data.source.distance
        alpha_T = self.data.detector.alpha_T
        Dbg = self.model.Dbg
        D, Dbg, alpha_T, eps_fit = convert_scale(D, Dbg, alpha_T, eps_fit)
                
        print('preparing fit inputs...')
        self.fit_input = {'Ns' : self.data.source.N,
                          'varpi' :self.data.source.unit_vector,
                          'D' : D,
                          'N' : self.data.uhecr.N,
                          'arrival_direction' : self.data.uhecr.unit_vector,
                          'A' : self.data.uhecr.A,
                          'kappa_c' : self.data.detector.kappa_c,
                          'alpha_T' : alpha_T,
                          'Ngrid' : len(kappa_grid),
                          'eps' : eps_fit,
                          'kappa_grid' : kappa_grid,
                          'zenith_angle' : np.deg2rad(self.data.uhecr.incidence_angle)}
        
        try:
            self.fit_input['flux'] = self.data.source.flux
        except:
            print('No flux weights available for sources.')
        
        if self.analysis_type == self.joint_type:

            self.fit_input['Edet'] = self.data.uhecr.energy
            self.fit_input['Eth'] = self.model.Eth
            self.fit_input['Eerr'] = self.model.Eerr
            self.fit_input['Dbg'] = Dbg
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

        # add E interpolation for Dbg
        #Earr_grid.append([get_arrival_energy(e, self.model.Dbg) for e in E_grid])
        Earr_grid.append([0 for e in E_grid])
            
        # convert scale for sampling
        D = self.data.source.distance
        alpha_T = self.data.detector.alpha_T
        Dbg = self.model.Dbg
        D, Dbg, alpha_T, eps_fit = convert_scale(D, Dbg, alpha_T, eps_fit)

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
                          'kappa_c' : self.data.detector.kappa_c,
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
            self.fit_input['Eerr'] = self.model.Eerr
            self.fit_input['Dbg'] = Dbg
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

        # fit
        self.fit = self.model.model.sampling(data = self.fit_input, iter = iterations, chains = chains, seed = seed,
                                             sample_file = sample_file, warmup = warmup)

        # Diagnositics
        self.fit_treedepth = stan_utility.check_treedepth(self.fit)
        self.fit_div = stan_utility.check_div(self.fit)
        self.fit_energy = stan_utility.check_energy(self.fit)
        self.n_eff = stan_utility.check_n_eff(self.fit)
        self.rhat = stan_utility.check_rhat(self.fit)
        
        self.chain = self.fit.extract(permuted = True)
        return self.fit

    def save_fit(self):

        if self.fit:

            with h5py.File(self.filename, 'r+') as f:

                fit_input = f['input'].create_group('fit')
                for key, value in self.fit_input.items():
                    fit_input.create_dataset(key, data = value)
                fit_input.create_dataset('params', data = self.data.detector.params)
                fit_input.create_dataset('theta_m', data = self.data.detector.threshold_zenith_angle.rad)
                fit_input.create_dataset('a0', data = self.data.detector.location.lat.rad)
                
                fit_output = f['output'].create_group('fit')
                diagnostics = fit_output.create_group('diagnostics')
                diagnostics.create_dataset('treedepth', data = self.fit_treedepth)
                diagnostics.create_dataset('divergence', data = self.fit_div)
                diagnostics.create_dataset('energy', data = self.fit_energy)
                rhat = diagnostics.create_group('rhat')
                for key, value in self.rhat.items():
                    rhat.create_dataset(key, data = value)
                n_eff = diagnostics.create_group('n_eff')
                for key, value in self.n_eff.items():
                    n_eff.create_dataset(key, data = value)      
                samples = fit_output.create_group('samples')
                for key, value in self.chain.items():
                    samples.create_dataset(key, data = value)
                
        else:
            print('Error: no fit to save')
        
        
    def ppc(self, seed = None):
        """
        Run a posterior predictive check.
        Use the fit parameters to simulate a dataset.
        Meant to be a quick check having just run a fit.
        """

        if self.analysis_type == 'arrival direction':
            print('No PPC implemented for arrival direction only analysis :( ')

        if self.analysis_type == 'joint':

            # extract fitted parameters
            chain = self.fit.extract(permuted = True)
            self.B_fit = np.mean(chain['B'])
            self.alpha_fit = np.mean(chain['alpha'])
            self.F0_fit = np.mean(chain['F0'])
            self.L_fit = np.mean(np.transpose(chain['L']), axis = 1)
        
            # calculate eps integral
            print('precomputing exposure integrals...')
            self.Eex = get_Eex(self.Eth_src, self.alpha_fit)
            self.kappa_ex = get_kappa_ex(self.Eex, self.B_fit, self.data.source.distance)        
            kappa_true = self.kappa_ex
            varpi = self.data.source.unit_vector
            params = self.data.detector.params
            self.ppc_table = ExposureIntegralTable(varpi = varpi, params = params)
            self.ppc_table.build_for_sim(kappa, self.alpha_fit, self.B_fit, self.data.source.distance)
            
            eps = self.ppc_table.sim_table

            # convert scale for sampling
            D = self.data.source.distance
            alpha_T = self.data.detector.alpha_T
            L = self.model.L
            F0 = self.model.F0
            Dbg = self.model.Dbg
            D, Dbg, alpha_T, eps, F0, L = convert_scale(D, Dbg, alpha_T, eps, F0, L)
            
            # compile inputs from Model, Data and self.fit
            self.ppc_input = {
                'kappa_c' : self.data.detector.kappa_c,
                'Ns' : self.data.source.N,
                'varpi' : self.data.source.unit_vector,
                'D' : D,
                'A' : self.data.detector.area,
                'a0' : self.data.detector.location.lat.rad,
                'theta_m' : self.data.detector.threshold_zenith_angle.rad,
                'alpha_T' : alpha_T,
                'eps' : eps}

            self.ppc_input['B'] = self.B_fit
            self.ppc_input['L'] = self.L_fit
            self.ppc_input['F0'] = self.F0_fit
            self.ppc_input['alpha'] = self.alpha_fit
            
            self.ppc_input['Eth'] = self.model.Eth
            self.ppc_input['Eerr'] = self.model.Eerr
            self.ppc_input['Dbg'] = Dbg

            # run simulation
            print('running posterior predictive simulation...')
            self.posterior_predictive = self.model.simulation.sampling(data = self.ppc_input, iter = 1,
                                                                       chains = 1, algorithm = "Fixed_param", seed = seed)
            print('done')

            # extract output
            print('extracting output...')
            arrival_direction = self.posterior_predictive.extract(['arrival_direction'])['arrival_direction'][0]
            self.arrival_direction_pred = Direction(arrival_direction)
            self.Edet_pred = self.posterior_predictive.extract(['Edet'])['Edet'][0]
            print('done')

        return self.arrival_direction_pred, self.Edet_pred
        
    def plot_ppc(self, ppc_type = None, cmap = None, use_sim_data = False):
        """
        Plot the posterior predictive check against the data 
        (or original simulation) for ppc_type == 'arrival direction' 
        or ppc_type == 'energy'.
        """

        if ppc_type == None:
            ppc_type = 'arrival direction'

        if ppc_type == 'arrival direction':

            # plot style
            if cmap == None:
                style = PlotStyle()
            else:
                style = PlotStyle(cmap_name = cmap)
            
            # figure
            fig = plt.figure(figsize = (12, 6));
            ax = plt.gca()

            # skymap
            skymap = AllSkyMap(projection = 'hammer', lon_0 = 0, lat_0 = 0);

            self.data.source.plot(style, skymap)
            self.data.detector.draw_exposure_lim(skymap)
       
            label = True
            if use_sim_data:
                for lon, lat in np.nditer([self.arrival_direction.lons, self.arrival_direction.lats]):
                    if label:
                        skymap.tissot(lon, lat, 4.0, npts = 30, alpha = 0.5, label = 'data')
                        label = False
                    else:
                        skymap.tissot(lon, lat, 4.0, npts = 30, alpha = 0.5)
            else:
                for lon, lat in np.nditer([self.data.uhecr.coord.galactic.l.deg, self.data.uhecr.coord.galactic.b.deg]):
                    if label:
                        skymap.tissot(lon, lat, self.data.uhecr.coord_uncertainty, npts = 30, alpha = 0.5, label = 'data')
                        label = False
                    else:
                        skymap.tissot(lon, lat, self.data.uhecr.coord_uncertainty, npts = 30, alpha = 0.5)
                
            label = True
            for lon, lat in np.nditer([self.arrival_direction_pred.lons, self.arrival_direction_pred.lats]):
                if label: 
                    skymap.tissot(lon, lat, self.data.uhecr.coord_uncertainty, npts = 30, alpha = 0.5,
                                  color = 'g', label = 'predicted')
                    label = False
                else:
                    skymap.tissot(lon, lat, self.data.uhecr.coord_uncertainty, npts = 30, alpha = 0.5, color = 'g')

            # standard labels and background
            skymap.draw_standard_labels(style.cmap, style.textcolor)

            # legend
            plt.legend(bbox_to_anchor = (0.85, 0.85))
            leg = ax.get_legend()
            frame = leg.get_frame()
            frame.set_linewidth(0)
            frame.set_facecolor('None')
            for text in leg.get_texts():
                plt.setp(text, color = style.textcolor)

        if ppc_type == 'energy':

            bins = np.logspace(np.log(self.model.Eth), np.log(1e4), base = np.e)
            plt.hist(self.Edet, bins = bins, alpha = 0.7, label = 'data')
            plt.hist(self.Edet_pred, bins = bins, alpha = 0.7, label = 'predicted')
            plt.xscale('log')
            plt.yscale('log')
            plt.legend()

    
