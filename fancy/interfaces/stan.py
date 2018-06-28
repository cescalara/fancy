import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

from .stan_utility import *

__all__ = ['Model', 'Direction', 'uv_to_coord', 'coord_to_uv']
        

class Model():
    """
    Simple wrapper for models defined in Stan.
    """

    def __init__(self, model_filename, sim_filename):
        """
        Simple wrapper for models defined in Stan.
       
        :param model_filename: location of the stan code for model
        :param sim_filename: locaiton of the stan code for simulation
        """

        self.model_filename = model_filename
        self.sim_filename = sim_filename

        self.simulation = None
        self.fit_input = None
        
        
    def compile(self):
        """
        Compile the models if not already done.
        """
        self.model = compile_model(self.model_filename)
        self.simulation = compile_model(self.sim_filename)

    def simulation_inputs(self, F_T, f, kappa, kappa_c):
        """
        Get simulation inputs.

        :param F_T: total flux
        :param f: associated fraction
        :param kappa: deflection parameter
        :param kappa_c: reconstruction parameter
        """
        self.F_T = F_T
        self.f = f
        self.kappa = kappa
        self.kappa_c = kappa_c
        
        
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
