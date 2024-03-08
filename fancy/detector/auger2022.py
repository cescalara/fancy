import numpy as np
from datetime import date
from scipy import integrate
from astropy import units as u
from astropy.coordinates import EarthLocation

from .exposure import m_integrand, kappa_dval

"""
Constants from information provided in Auger Collabration publications on the 2022 dataset.

* latitude and longitude taken from Auger Collaboration 2008.
* Information on periods and exposures from Auger Collaboration et al. 2022 (arXiv:2206.13492).
* M values calculated by integrating the m_integrand in exposure.py
over the unit sphere for certain values of theta_m. 

@author Keito Watanabe
@date September 2022
"""

# position of the PAO [rad]
lat = np.deg2rad(-35.2)
lon = np.deg2rad(-69.4)
height = 1400 # [m]
auger_location = EarthLocation(lat = lat * u.rad, lon = lon * u.rad,
                               height = height * u.m)

# threshold zenith angle
theta_m = np.deg2rad(80)

# start year of observation
start_year = 2004

# get total period as the Eth is different from 2014 dataset
period_start = date(2004, 1, 1)
period_end = date(2020, 12, 31)

# length of each period in years
deltat = (period_end - period_start).days / 365.25

# total exposure, vertical and inclined in km^2 sr yr
alpha_T_vert = 95700
alpha_T_incl = 26300
alpha_T = alpha_T_vert + alpha_T_incl

# calculate M (integral over exposure factor) [sr]
detector_params = []
detector_params.append(np.cos(lat))
detector_params.append(np.sin(lat))
detector_params.append(np.cos(theta_m))
detector_params.append(alpha_T)
M, Merr = integrate.quad(m_integrand, 0, np.pi, args = detector_params)
detector_params.append(M)

# calculate effective area in km^2
A = alpha_T / (M * deltat)
A_incl = alpha_T_incl / (M * deltat)
A_vert = alpha_T_vert / (M * deltat)

# reconstruction uncertainty for arrival direction
sig_omega = 1.0
kappa_d = kappa_dval(sig_omega)

# reconstruction uncertainty for energy
# calibration unc ~ 14%, SD resolution ~ 7%, add in quadrature
f_E = 0.156

# threshold energy [EeV]
Eth = 32

# For convenience
detector_properties = {}
detector_properties['label'] = 'auger2022'
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