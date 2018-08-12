import h5py
from math import ceil
import numpy as np

from ..interfaces import stan_utility
from ..interfaces.integration import ExposureIntegralTable


__all__ = ['Results', 'PPC']


class Results():
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
        with h5py.File(self.filename, 'r') as f:
            fit_output = f['output/fit/samples']
            for key in list_of_keys:
                chain[key] = fit_output[key].value

        return chain

    def get_truths(self, list_of_keys):
        """
        For the case where the analysis was based on simulated 
        data, return input values or 'truths' for desired 
        parameters specified by list_of_keys.
        """

        truths = {}
        with h5py.File(self.filename, 'r') as f:

            try:
                sim_input = f['input/simulation']
                for key in list_of_keys:
                    truths[key] = sim_input[key].value

            except:
                print('Error: file does not contain simulation inputs.')
        return truths
    

    def traceplot(self, list_of_keys):
        """
        Plot the traceplot for parameters specified by list_of_keys.
        """
        return 0
    

    def summary(self):
        """
        Print a summary of the results.
        """
        return 0

    def get_fit_parameters(self):
        """
        Return mean values of all main fit parameters.
        """

        list_of_keys = ['B', 'alpha', 'L', 'F0']
        chain = get_chain(list_of_keys)

        fit_parameters = {}
        fit_parameters['B'] = np.mean(chain['B'])
        fit_parameters['alpha'] = np.mean(chain['alpha'])
        fit_parameters['F0'] = np.mean(chain['F0'])
        fit_parameters['L'] = np.mean(np.transpose(chain['L']), axis = 1)
        
        return fit_parameters
        
    def get_input_data(self):
        """
        Return fit input data.
        """

        fit_input = {}
        with h5py.File(self.filename, 'r') as f:
            fit_input_from_file = f['input/fit']
            for key in fit_input_from_file:
                fit_input[key] = fit_input_from_file[key].value
                
        return fit_input
    
    def ppc(self, stan_sim_file, N = 3):
        """
        Run N posterior predictive simulations.
        """

        fit_parameters = get_fit_parameters()
        input_data = get_input_data()

        self.ppc = PPC(stan_sim_file, ppc_type)

        self.arrival_direction_preds = []
        self.Edet_preds = []
        for i in range(N):
            arr_dir_pred, Edet_pred = self.ppc.simulate(fit_parameters, input_data)
            self.arrival_direction_preds.append(arr_dir_pred)
            self.Edet_preds.append(Edet_pred)

    
    

class PPC():
    """
    Handles posterior predictive checks.
    """

    def __init__(self, stan_sim_file):
        """
        Handles posterior predictive checks.
        :param stan_sim_file: the stan file to use to run the simulation
        """
    
        # compile the stan model
        self.simulaiton = stan_utilty.compile_model(stan_sim_file)

        self.type = ppc_type

        self.arrival_direction_preds = []
        self.Edet_preds = []
        
        
    def simulate(self, fit_parameters, input_data):
        """
        Simulate from the posterior predictive distribution. 
        """

        self.alpha = fit_parameters['alpha']
        self.B = fit_parameters['B']
        self.F0 = fit_parameter['F0']
        self.L = fit_parameter['L']

        
        # calculate eps integral
        print('precomputing exposure integrals...')
        self.Eth_src = get_Eth_src(input_data['Eth'], input_data['D'])
        self.Eex = get_Eex(self.Eth_src, self.alpha)
        self.kappa_ex = get_kappa_ex(self.Eex, self.B, input_data['D'])        
        
        varpi = input_data['varpi']
        params = input_data['params']
        self.ppc_table = ExposureIntegralTable(varpi = varpi, params = params)
        self.ppc_table.build_for_sim(self.kappa_ex, self.alpha, self.B, input_data['D'])
            
        eps = self.ppc_table.sim_table

        # compile inputs 
        self.ppc_input = {
            'kappa_c' : input_data['kappa_c'],
            'Ns' : input_data['Ns'],
            'varpi' : input_data['varpi'],
            'D' : input_data['D'],
            'A' : input_data['A'],
            'a0' : input_data['a0'],
            'theta_m' : input_data['theta_m'],
            'alpha_T' : input_data['alpha_T'],
            'eps' : eps}
        
        self.ppc_input['B'] = self.B
        self.ppc_input['L'] = self.L
        self.ppc_input['F0'] = self.F0
        self.ppc_input['alpha'] = self.alpha
        
        self.ppc_input['Eth'] = input_data['Eth']
        self.ppc_input['Eerr'] = input_data['Eerr']
        self.ppc_input['Dbg'] = input_data['Dbg']
        
        # run simulation
        print('running posterior predictive simulation...')
        self.posterior_predictive = self.simulation.sampling(data = self.ppc_input, iter = 1,
                                                             chains = 1, algorithm = "Fixed_param", seed = seed)
        print('done')
        
        # extract output
        print('extracting output...')
        arrival_direction = self.posterior_predictive.extract(['arrival_direction'])['arrival_direction'][0]
        self.arrival_direction_preds.append(Direction(arrival_direction))
        self.Edet_preds.append(self.posterior_predictive.extract(['Edet'])['Edet'][0])
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

            # how many simulaitons
            N_sim = len(self.arrival_direction_preds)
            N_grid = N_sim + 1
            N_rows = ceil(np.sqrt(N_grid))
            N_cols = ceil(N_grid / N_rows)
            
            # figure
            fig, ax = plt.subplots(N_rows, N_cols, figsize = (10, 10))
            flat_ax = ax.reshape(-1)
                                   
            # skymap
            skymap = AllSkyMap(projection = 'hammer', lon_0 = 0, lat_0 = 0);
                                   
            for i, ax in enumerate(flat_ax):

                # data
                if i == 0:
                    skymap.ax = ax
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

                # predicted
                else:
                    skymap.ax = ax
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
                ax.legend(bbox_to_anchor = (0.85, 0.85))
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
