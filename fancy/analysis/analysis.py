import numpy as np
import pystan
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy import units as u
from matplotlib import pyplot as plt

from ..interfaces.integration import ExposureIntegralTable
from ..interfaces.stan import Direction
from ..interfaces import stan_utility

__all__ = ['Analysis']


class Analysis():
    """
    To manage the running of simulations and fits based on Data and Model objects.
    """

    def __init__(self, data, model):
        """
        To manage the running of simulations and fits based on Data and Model objects.
        
        :param data: a Data object
        :param model: a Model object
        """

        self.data = data

        self.model = model

        self.fit_input = None
        self.sim_input = None
        
        self.simulation = None
        self.fit = None
        
        
    def build_tables(self, num_points, table_filename, sim_table_filename):
        """
        Build the necessary integral tables.
        """

        self.sim_table_filename = sim_table_filename
        self.table_filename = table_filename 
        
        kappa = np.linspace(10, 1000, num_points)

        params = self.data.detector.params

        kappa_true = self.model.kappa

        varpi = self.data.source.unit_vector

        # kappa_true table for simulation
        self.sim_table = ExposureIntegralTable(kappa_true, varpi, params, self.sim_table_filename)
        self.sim_table.build()
        
        # full table for fit
        self.table = ExposureIntegralTable(kappa, varpi, params, self.table_filename)
        self.table.build()
        

    def use_tables(self, table_filename, sim_table_filename):
        """
        Pass in names of integral tables that have already been made.
        """
        self.sim_table_filename = sim_table_filename
        self.table_filename = table_filename 
        

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
        c_icrs = self.detected.d.icrs 

        time = []
        zenith_angles = []
        stuck = []

        j = 0
        first = True
        for d in c_icrs:
            za = 99
            i = 0
            while (za > self.data.detector.threshold_zenith_angle.rad):
                dt = np.random.exponential(1 / self.Nex_sim)
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
            print(j , za)
            
        if (len(stuck) > 1):
            print('Warning: % of zenith angles stuck is', len(stuck)/len(zenith_angles) * 100)

        return zenith_angles

    
    def simulate(self, seed = None):
        """
        Run a simulation.

        :param seed: seed for RNG
        """

        eps = pystan.read_rdump(self.sim_table_filename)['table'][0]
        
        # compile inputs from Model and Data
        self.simulation_input = {'F_T' : self.model.F_T,
                       'f' : self.model.f,
                       'kappa' : self.model.kappa,
                       'kappa_c' : self.model.kappa_c, 
                       'N_A' : len(self.data.source.distance),
                       'varpi' : self.data.source.unit_vector, 
                       'D' : self.data.source.distance,
                       'A' : self.data.detector.area,
                       'a_0' : self.data.detector.a_0,
                       'theta_m' : self.data.detector.theta_m, 
                       'alpha_T' : self.data.detector.alpha_T,
                       'M' : self.data.detector.M,
                       'eps' : eps}

        # run simulation
        self.simulation = self.model.simulation.sampling(data = self.simulation_input, iter = 1,
                                                                chains = 1, algorithm = "Fixed_param", seed = seed)
        # extract output
        self.pdet = self.simulation.extract(['pdet'])['pdet'][0]
        self.theta = self.simulation.extract(['theta'])['theta'][0]
        self.event = self.simulation.extract(['event'])['event'][0]
        self.Nex_sim = self.simulation.extract(['Nex_sim'])['Nex_sim']
        self.detected = Direction(self.event)
        
        # simulate the zenith angles
        self.zenith_angles = self._simulate_zenith_angles()

        eps_fit = pystan.read_rdump(self.table_filename)['table']
        kappa_grid = pystan.read_rdump(self.table_filename)['kappa']
        
        # prepare fit inputs
        self.fit_input = {'N_A' : len(self.data.source.distance), 
                          'varpi' :self.data.source.unit_vector,
                          'D' : self.data.source.distance, 
                          'N' : len(self.event), 
                          'detected' : self.event, 
                          'A' : self.data.detector.area,
                          'a_0' : self.data.detector.a_0,
                          'theta_m' : self.data.detector.theta_m, 
                          'alpha_T' : self.data.detector.alpha_T, 
                          'M' : self.data.detector.M, 
                          'Ngrid' : len(kappa_grid), 
                          'eps' : eps_fit, 
                          'kappa_grid' : kappa_grid,
                          'zenith_angle' : self.zenith_angles}

        
    def save_simulated_data(self, filename):
        """
        Write the simulated data to file.
        """
        if self.fit_input != None:
            pystan.stan_rdump(self.fit_input, filename)
        else:
            print("Error: nothing to save!")

            
    def plot_simulation(self):
        """
        Plot the simulated data on a skymap
        """
        labels = (self.simulation.extract(['lambda'])['lambda'][0] - 1).astype(int)

        fig, skymap = self.data.show()

        cmap = plt.cm.get_cmap('plasma', 17) 
        label = True
        for lon, lat, lab in np.nditer([self.detected.lons, self.detected.lats, labels]):
            if (lab == 17):
                color = 'k'
            else:
                color = cmap(lab)
            if label:
                skymap.tissot(lon, lat, 2, npts = 30, facecolor = color,
                              alpha = 0.5, label = 'simulated data')
                label = False
            else:
                skymap.tissot(lon, lat, 2, npts = 30, facecolor = color, alpha = 0.5)   


    def use_simulated_data(self, filename):
        """
        Read in simulated data from a file.
        """

        self.fit_input = pystan.read_rdump(filename)
                
    def fit_model(self, seed = None):
        """
        Fit a model.

        :param seed: seed for RNG
        """

        # fit
        self.fit = self.model.model.sampling(data = self.fit_input, iter = 1000, chains = 4, seed = seed)

        # Diagnositics
        stan_utility.check_treedepth(self.fit)
        stan_utility.check_div(self.fit)
        stan_utility.check_energy(self.fit)

        return self.fit
