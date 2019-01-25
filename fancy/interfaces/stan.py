import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from pandas import read_csv
import h5py

import stan_utility  

__all__ = ['Model', 'Direction', 'uv_to_coord', 'coord_to_uv']
        

class Model():
    """
    Simple wrapper for models defined in Stan.
    """

    def __init__(self, model_filename, sim_filename, include_paths =  None):
        """
        Simple wrapper for models defined in Stan.
       
        :param model_filename: location of the stan code for model
        :param sim_filename: locaiton of the stan code for simulation
        """

        self.model_filename = model_filename
        self.sim_filename = sim_filename
        self.include_paths = include_paths
        
        self.simulation = None
        self.fit_input = None
        
        
        
    def compile(self):
        """
        Compile the models if not already done.
        """
        self.model = stan_utility.compile_model(filename = self.model_filename, model_name = 'sim',
                                                include_paths = self.include_paths)
        self.simulation = stan_utility.compile_model(filename = self.sim_filename, model_name = 'model',
                                                     include_paths = self.include_paths)

    def input(self, B = None, kappa = None,
                          F_T = None, f = None, L = None, F0 = None,
                          alpha = None, Eth = None, Eerr = None):
        """
        Get simulation inputs.

        :param F_T: total flux [# km-^2 yr^-1]
        :param f: associated fraction
        :param kappa: deflection parameter 
        :param B: rms B field strength [nG]
        :param kappa_c: reconstruction parameter
        :param alpha: source spectral index
        :param Eth: threshold energy of study [EeV]
        :param Eerr: energy reconstruction uncertainty = Eerr * E 
        """
        self.F_T = F_T
        self.f = f
        self.kappa = kappa
        self.B = B
        self.L = L 
        self.F0 = F0
        self.alpha = alpha
        self.Eth = Eth
        self.Eerr = Eerr
        
        
class Direction():
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
        self.d = SkyCoord(self.x, self.y, self.z, 
                          unit = 'mpc', 
                          representation_type = 'cartesian', 
                          frame = 'icrs')
        self.d.representation_type = 'spherical'
        self.lons = self.d.galactic.l.wrap_at(360 * u.deg).deg
        self.lats = self.d.galactic.b.wrap_at(180 * u.deg).deg



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
       
    c = SkyCoord(x, y, z, unit = 'mpc', representation_type = 'cartesian',
                 frame = 'icrs')

    return c
        

def coord_to_uv(coord):
    """
    Convert SkyCoord object into array of unit vecotrs in the ICRS frame.
    Used for input into Stan programs.
    
    :param coord: astropy SkyCoord object
    :return: an array of 3D unit vectors
    """
    c = coord.icrs
    ds = [c.cartesian.x, c.cartesian.y, c.cartesian.z]
    uv = [d / np.linalg.norm(d) for d in np.transpose(ds)]

    return uv

def convert_scale(D, alpha_T, eps, F0 = None, L = None):
    """
    Convenience function to convert parameters 
    to O(1) scale for sampling in Stan.
    D [Mpc] -> (D * 3.086) / 100
    alpha_T [km^2 yr] -> alpha_T / 1000
    eps [km^2 yr] -> eps / 1000
    F [# km^-2 yr^-1] -> F * 1000
    L [# yr^-1] -> L / 1e39
    """

    D = [(d * 3.086) / 100 for d in D]
    alpha_T = alpha_T / 1000.0
    eps = [e / 1000.0 for e in eps]
    
    if F0:
        F0 = F0 * 1000.0

    if isinstance(L, (list, np.ndarray)):
        L = L / 1.0e39

    if F0 and isinstance(L, (list, np.ndarray)):
        return D, alpha_T, eps, F0, L
    else:
        return D, alpha_T, eps
