import numpy as np

__all__ = ['coord_to_uv']

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
