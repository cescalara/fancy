import numpy as np
from datetime import date
from scipy import integrate
from astropy import units as u
from astropy.coordinates import EarthLocation

from .exposure import m_integrand

"""
Constants from information provided in Auger Collabration publications on the 2014 dataset.

* latitude and longitude taken from Auger Collaboration 2008.
* Information on periods and exposures from Auger Collaboration et al. 2014.
* M values calculated by integrating the m_integrand in exposure.py
over the unit sphere for certain values of theta_m. 

@author Francesca Capel
@date July 2018
"""


# position of the PAO [rad]
lat = np.deg2rad(-35.2)
lon = np.deg2rad(-69.4)
auger_location = EarthLocation(lat = lat * u.rad, lon = lon * u.rad,
                               height = 1400 * u.m)

# threshold zenith angle
theta_m = np.deg2rad(80)

# define periods 1 - 3 based on Abreu et al. 2010.
# define period 4 based on Collaboration et al. 2014
period_1_start = date(2004, 1, 1)
period_1_end = date(2006, 5, 26)
period_2_start = date(2006, 5, 27)
period_2_end = date(2007, 8, 31)
period_3_start = date(2007, 9, 1)
period_3_end = date(2009, 12, 31)
period_4_start = date(2010, 1, 1)
period_4_end = date(2014, 3, 31)

# find length of each period in units of years
deltat1 = (period_1_end - period_1_start).days / 365.25
deltat2 = (period_2_end - period_2_start).days / 365.25
deltat3 = (period_3_end - period_3_start).days / 365.25
deltat4 = (period_4_end - period_4_start).days / 365.25
deltat = (period_4_end - period_1_start).days / 365.25

# define total exposures [km^2 sr year]
alpha_T_vert = 51753
alpha_T_incl = 14699
ratio_iv = alpha_T_incl / alpha_T_vert
alpha_T = alpha_T_vert + alpha_T_incl

# define exposures in each period
alpha_T_1 = 4390
alpha_T_1_incl = alpha_T_1 * ratio_iv 
alpha_T_2 = 4500
alpha_T_2_incl = alpha_T_2 * ratio_iv 
alpha_T_3 = 11480
alpha_T_3_incl = alpha_T_3 * ratio_iv
alpha_T_4 = (alpha_T_vert - (alpha_T_1 + alpha_T_2 + alpha_T_3))
alpha_T_4_incl = alpha_T_4 * ratio_iv

# calculate M (integral over exposure factor) [sr]
auger_params = []
auger_params.append(np.cos(lat))
auger_params.append(np.sin(lat))
auger_params.append(np.cos(theta_m))
auger_params.append(alpha_T)
M, Merr = integrate.quad(m_integrand, 0, np.pi, args = auger_params)
auger_params.append(M)

# calculate areas for each period [km^2]
A1 = alpha_T_1 / (M * deltat1)
A1_incl = alpha_T_1_incl / (M * deltat1)
A2 = alpha_T_2 / (M * deltat2)
A2_incl = alpha_T_2_incl / (M * deltat2)
A3 = alpha_T_3 / (M * deltat3)
A3_incl = alpha_T_3 / (M * deltat3)
A4 = alpha_T_4 / (M * deltat4)
A4_incl = alpha_T_4_incl / (M * deltat4)
A = alpha_T / (M * deltat)

# reconstruction uncertainty for arrival direction
kappa_c = 9323
