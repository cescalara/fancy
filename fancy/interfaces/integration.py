from scipy import integrate, interpolate
import pystan

from ..detector.exposure import *

__all__ = ['ExposureIntegralTable']

class ExposureIntegralTable():
    """
    Handles the building and storage of exposure integral tables 
    that are passed to Stan to be interpolated over.
    """

    def __init__(self, kappa, varpi, params, filename = None):
        """
        Handles the building and storage of integral tables 
        that are passed to Stan to be interpoloated over.
        
        :param integral: the Integral object to tabulate 
        :param kappa: an array of kappa values to evaluate the integral for
        :param varpi: an array of 3D unit vectors to pass to the integrand
        :param params: an array of other parameters to pass to the integrand
        :param filename: the filename to save the table in
        """
        
        self.kappa = kappa

        self.varpi = varpi

        self.params = params
        
        if filename != None:
            self.filename = filename
        else:
            self.filename = 'ExposureIntegralTable' + str(np.random.uniform(1, 1000))

        self.table = []
        self.sim_table = []


    def build_for_sim(self):
        """
        Build the tabulated integrals to be used for simulations and posterior predictive checks.
        Save with the filename given.
        
        Expects self.kappa to be either a fixed value or a list of values of length equal to the 
        total number of sources. The integral is evaluated once for each source. 
        """

        # single fixed kappa
        if isinstance(self.kappa, int) or isinstance(self.kappa, float):
            k = self.kappa
            results = []
            for v in self.varpi:
                result, err = integrate.dblquad(integrand, 0, np.pi,
                                                lambda phi : 0, lambda phi : 2 * np.pi,
                                                args = (v, k, self.params))

                print(k, result, err)
                results.append(result)
            self.table.append(np.asarray(results))
            print()

        # different kappa for each source
        else:
            results = []
            for i, v in enumerate(self.varpi):
                result, err = integrate.dblquad(integrand, 0, np.pi,
                                                lambda phi : 0, lambda phi : 2 * np.pi,
                                                args = (v, self.kappa[i], self.params))

                print(self.kappa[i], result, err)
                results.append(result)

            self.table.append(np.asarray(results))
            print()

        # save to file
        pystan.stan_rdump({'table' : self.table, 'kappa' : self.kappa}, self.filename)

            
    def build_for_fit(self):
        """
        Build the tabulated integrals to be interpolated over in the fit.
        Save with filename given.
        
        Expects self.kappa to be a list of kappa values to evaluate the integral for, 
        for each source individually.
        """

        for k in self.kappa:
            results = []
            for v in self.varpi:
                result, err = integrate.dblquad(integrand, 0, np.pi,
                                                lambda phi : 0, lambda phi : 2 * np.pi,
                                                args = (v, k, self.params))
                
                print(k, result, err)
                results.append(result)
            self.table.append(np.asarray(results))
            print()
        self.table = np.asarray(self.table).transpose()
            
        # save to file
        pystan.stan_rdump({'table' : self.table, 'kappa' : self.kappa}, self.filename)
        
