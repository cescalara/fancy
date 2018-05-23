import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u


__all__ = ['Direction']


class Direction():
    """
    Input the unit vector vMF samples and 
    store x, y, and z and galactic coordinates 
    of direction in Mpc.
    """
    
    def __init__(self, unit_vector_3d):
        """
        Input the unit vector vMF samples and 
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
                          frame = 'galactic')
        self.d.representation_type = 'spherical'
        self.lons = self.d.galactic.l.wrap_at(360 * u.deg).deg
        self.lats = self.d.galactic.b.wrap_at(180 * u.deg).deg
