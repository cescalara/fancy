from scipy import integrate, interpolate
import h5py

from ..detector.exposure import *

__all__ = ['ExposureIntegralTable']

class ExposureIntegralTable():
    """
    Handles the building and storage of exposure integral tables 
    that are passed to Stan to be interpolated over.
    """

    def __init__(self, varpi = None, params = None, input_filename = None):
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
            for v in self.varpi:
                result, err = integrate.dblquad(integrand, 0, np.pi,
                                                lambda phi : 0, lambda phi : 2 * np.pi,
                                                args = (v, k, self.params))

                print(k, result, err)
                results.append(result)
            self.sim_table = results
            print()

        # different kappa for each source
        else:
            results = []
            for i, v in enumerate(self.varpi):
                result, err = integrate.dblquad(integrand, 0, np.pi,
                                                lambda phi : 0, lambda phi : 2 * np.pi,
                                                args = (v, self.sim_kappa[i], self.params))

                print(self.sim_kappa[i], result, err)
                results.append(result)

            self.sim_table = results
            print()

            
    def build_for_fit(self, kappa):
        """
        Build the tabulated integrals to be interpolated over in the fit.
        Save with filename given.
        
        Expects self.kappa to be a list of kappa values to evaluate the integral for, 
        for each source individually.
        """

        self.kappa = kappa
        for v in self.varpi:
            results = []
            for k in self.kappa:
                result, err = integrate.dblquad(integrand, 0, np.pi,
                                                lambda phi : 0, lambda phi : 2 * np.pi,
                                                args = (v, k, self.params))
                
                print(k, result, err)
                results.append(result)
            self.table.append(np.asarray(results))
            print()
        self.table = np.asarray(self.table)


    def init_from_file(self, input_filename):
        """
        Initialise the object from the given file.
        """

        with h5py.File(input_filename, 'r') as f:

            self.varpi = f['varpi'].value
            self.params = f['params'].value
            self.kappa = f['main']['kappa'].value
            self.table = f['main']['table'].value
            
            if f['simulation']['kappa'].value is not h5py.Empty('f'):
                try:
                    self.sim_kappa = f['simulation']['kappa'].value
                    self.sim_table = f['simulation']['table'].value
                    self.sim_alpha = f['simulation']['alpha'].value
                    self.sim_B = f['simulation']['B'].value
                    self.sim_D = f['simulation']['D'].value
                except:
                    print("skipped simulation values")
        
    def save(self, output_filename):
        """
        Save the computed integral table(s) to a HDF5 file 
        for later use as inputs.
        
        If no table is found, create an empty dataset.
 
        :param output_filename: the name of the file to write to 
        """

        with h5py.File(output_filename, 'w') as f:

            # common params
            f.create_dataset('varpi', data = self.varpi)
            f.create_dataset('params', data = self.params)

            # main interpolation table
            main = f.create_group('main')
            if self.table != []:
                main.create_dataset('kappa', data = self.kappa)
                main.create_dataset('table', data = self.table)
            else:
                main.create_dataset('kappa', data = h5py.Empty('f'))
                main.create_dataset('table', data = h5py.Empty('f'))

            # simulation table
            sim = f.create_group('simulation')
            if self.sim_table != []:                
                sim.create_dataset('kappa', data = self.sim_kappa)
                sim.create_dataset('table', data = self.sim_table)
                sim.create_dataset('alpha', data = self.sim_alpha)
                sim.create_dataset('B', data = self.sim_B)
                sim.create_dataset('D', data = self.sim_D)
            else:
                sim.create_dataset('kappa', data = h5py.Empty('f'))
                sim.create_dataset('table', data = h5py.Empty('f'))
                sim.create_dataset('alpha', data = h5py.Empty('f'))
                sim.create_dataset('B', data = h5py.Empty('f'))
                sim.create_dataset('D', data = h5py.Empty('f'))
         
