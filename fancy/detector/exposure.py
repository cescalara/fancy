import numpy as np

"""
Exposure functions for a ground based UHECR observatory.

The exposure is calculated using the method described in 
Sommers, P., 2000. Cosmic Ray Anisotropy Analysis with a Full-Sky Observatory. arXiv.org.

Here, the exposure functions are written in terms of the spherical 
coordinate theta, where theta = pi/2 - declination and theta [pi, 0], dec [-pi/2, pi/2]. 
This expression is for convenience when translating to the Stan run code in which 
everything takes place on the unit sphere. 

For completeness, the exposure as a function of declination is also included 
with _dec appended to the function name. 

@author Francesca Capel 
@date June 2018
"""

def xi(theta, p):
    return (p[2] - (p[1] * np.cos(theta))) / (p[0] * np.sin(theta)) 

def alpha_m(theta, p):
    xi_val = xi(theta, p)
    if (xi_val > 1):
        return 0
    elif (xi_val < -1):
        return np.pi
    else:
        return np.arccos(xi_val)
    
def m(theta, p):
    return (p[0] * np.sin(theta) * np.sin(alpha_m(theta, p)) 
            + alpha_m(theta, p) * p[1] * np.cos(theta))

def integrand(phi, theta, varpi, kappa, p):
    """
    Integrand for \int d(omega) m(omega) * rho(omega | kappa).

    Expressed as an integral in spherical coordinates for 
    theta [0, pi] and phi [0, 2pi]. For use with 
    scipy.integrate.dblquad

    NB: the alpha_T / M factor from eps(omega) is included 
    NB: the kappa / 4pi*sinh(kappa) from the vMF is included 
    """
    omega = [np.sin(theta) * np.cos(phi),
             np.sin(theta) * np.sin(phi),
             np.cos(theta)]
    if kappa > 100:
        integ = (p[3] / p[4]) * np.exp( kappa * np.dot(omega, varpi) + np.log(kappa) - np.log(4 * np.pi / 2)
                                        - kappa ) * m(theta, p) * np.sin(theta)    
    else:    
        integ = (p[3] / p[4]) * constant_val(kappa) * np.exp(kappa * np.dot(omega, varpi)) * m(theta, p) * np.sin(theta)
    return integ

def integrand_vMF(phi, theta, varpi, kappa):
    """
    Integrand for \int d(omega) * rho(omega | kappa).

    Expressed as an integral in spherical coordinates for 
    theta [0, pi] and phi [0, 2pi]. For use with 
    scipy.integrate.dblquad
    """
    omega = [np.sin(theta) * np.cos(phi),
             np.sin(theta) * np.sin(phi),
             np.cos(theta)]
    if kappa > 100:
        integ = np.exp( kappa * np.dot(omega, varpi) + np.log(kappa) - np.log(4 * np.pi / 2)
                        - kappa ) * np.sin(theta)    
    else:    
        integ = constant_val(kappa) * np.exp(kappa * np.dot(omega, varpi)) * np.sin(theta)
    return integ


def integrand_approx(phi, theta, varpi, kappa, p):
    """
    Integrand for \int d(omega) m(omega) * rho(omega | kappa).

    Expressed as an integral in spherical coordinates for 
    theta [0, pi] and phi [0, 2pi]

    Approximation used to avoid numerical overflow at large kappa.
    """
    omega = [np.sin(theta) * np.cos(phi),
             np.sin(theta) * np.sin(phi),
             np.cos(theta)]
    integ = np.exp( kappa * np.dot(omega, varpi) + np.log(kappa) - np.log(4 * np.pi / 2)
                       - kappa ) * m(theta, p) * np.sin(theta)
    return integ

def alpha(theta, phi, varpi):
    """
    The angle for which omega.varpi = cos(alpha).

    The angle between omega and varpi unit vectors.
    """
    inner = (varpi[0] * np.sin(theta) * np.cos(phi)
             + varpi[1] * np.sin(theta) * np.sin(phi)
             + varpi[2] * np.cos(theta))
    return np.arccos(inner)

def m_integrand(theta, p):
    """
    Integrand for \int d(omega) m(omega).

    Expressed as an integral in spherical coordinates for 
    theta [0, pi] and phi [0, 2pi]
    """
    return 2 * np.pi * m(theta, p) * np.sin(theta)


def constant_val(kappa):
    """
    Constant in front of integral.

    Approximates for large kappa to avoid numerical overflow.
    """
    if kappa > 100:
        return kappa / (4 * np.pi * np.exp(kappa - np.log(2)))
    else:
        return kappa / (4 * np.pi * np.sinh(kappa))

"""
Exposure as a funciton of declination. 
"""

def xi_dec(dec, p):
    return (p[2] - p[1] * np.sin(dec)) / (p[0] * np.cos(dec))

def alpha_m_dec(dec, p):
    xi_val = xi_dec(dec, p)
    if (xi_val > 1):
        return 0
    elif (xi_val < -1):
        return np.pi
    else:
        return np.arccos(xi_val)
    
def m_dec(dec, p):
    return (p[0] * np.cos(dec) * np.sin(alpha_m_dec(dec, p)) 
            + alpha_m_dec(dec, p) * p[1] * np.sin(dec))
