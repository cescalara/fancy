import numpy as np
from datetime import date
from scipy import integrate
from astropy import units as u
from astropy.coordinates import EarthLocation

from .exposure import m_integrand, kappa_dval

"""
Constants from information provided in Auger Collabration publications on the 2010 dataset.

* latitude and longitude taken from Auger Collaboration 2008.
* Information on periods and exposures from Abreu et al. 2010.
* M values calculated by integrating the m_integrand in exposure.py
over the unit sphere for certain values of theta_m. 

NB: theta_m is up to 80 in later papers.

@author Francesca Capel
@date July 2018
"""

# position of the PAO [rad]
lat = np.deg2rad(-35.2)
lon = np.deg2rad(-69.4)
height = 1400 # [m]
auger_location = EarthLocation(lat = lat * u.rad, lon = lon * u.rad,
                               height = height * u.m)

# threshold incidence angle [rad]
theta_m = np.deg2rad(60)

# define periods based on Abreu et al. 2010.
period_start = date(2004, 1, 1)
period_1_start = period_start
period_1_end = date(2006, 5, 26)
period_2_start = date(2006, 5, 27)
period_2_end = date(2007, 8, 31)
period_3_start = date(2007, 9, 1)
period_3_end = date(2009, 12, 31)

# start year of observation
start_year = 2004

# find length of each period in units of years
deltat1 = (period_1_end - period_1_start).days / 365.25
deltat2 = (period_2_end - period_2_start).days / 365.25
deltat3 = (period_3_end - period_3_start).days / 365.25
deltat = (period_3_end - period_1_start).days / 365.25

# define total exposures [km^2 sr year]
alpha_T_1 = 4390
alpha_T_2 = 4500
alpha_T_3 = 11480
alpha_T = 20370

# calculate M (integral over exposure factor) [sr]
detector_params = []
detector_params.append(np.cos(lat))
detector_params.append(np.sin(lat))
detector_params.append(np.cos(theta_m))
detector_params.append(alpha_T)
M, Merr = integrate.quad(m_integrand, 0, np.pi, args = detector_params)
detector_params.append(M)

# calculate areas for each period [km^2]
A1 = alpha_T_1 / (M * deltat1)
A2 = alpha_T_2 / (M * deltat2)
A3 = alpha_T_3 / (M * deltat3)
A = alpha_T / (M * deltat)

# reconstruction uncertainty for arrival direction
sig_omega = 0.9
kappa_d = kappa_dval(sig_omega)

# reconstruction uncertainty for energy
f_E = 0.12

# threshold energy [EeV]
Eth = 52

# For convenience
detector_properties = {}
detector_properties['label'] = 'auger2010'
detector_properties['lat'] = lat
detector_properties['lon'] = lon
detector_properties['height'] = height
detector_properties['theta_m'] = theta_m
detector_properties['kappa_d'] = kappa_d
detector_properties['f_E'] = f_E
detector_properties['A'] = A
detector_properties['alpha_T'] = alpha_T
detector_properties['Eth'] = Eth
detector_properties["start_year"] = start_year

