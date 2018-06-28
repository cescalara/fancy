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

        
    def build(self):
        """
        Build the integral table and save with filename given.
        """

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

        else:
        
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
        
