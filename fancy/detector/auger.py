import numpy as np
from datetime import date
from scipy import integrate

from .exposure import m_integrand

"""
Constants from information provided in Auger Collabration publications.

* latitude and longitude taken from Auger Collaboration 2008.
* Information on periods and exposures from Abreu et al. 2010.
* M values calculated by integrating the m_integrand in exposure.py
over the unit sphere for certain values of theta_m. 

NB: theta_m is up to 80 in later papers.

@author Francesca Capel
@date July 2018
"""

# position of the PAO [deg]
lat = -35.2
lon = -69.4

# threshold incidence angle [deg]
theta_m = 60

# define periods based on Abreu et al. 2010.
period_1_start = date(2004, 1, 1)
period_1_end = date(2006, 5, 26)
period_2_start = date(2006, 5, 27)
period_2_end = date(2007, 8, 31)
period_3_start = date(2007, 9, 1)
period_3_end = date(2009, 12, 31)

# find length of each period in units of years
deltat1 = (period_1_end - period_1_start).days / 365.25
deltat2 = (period_2_end - period_2_start).days / 365.25
deltat3 = (period_3_end - period_3_start).days / 365.25

# define total exposures [km^2 sr year]
alpha_T_1 = 4390
alpha_T_2 = 4500
alpha_T_3 = 11480
alpha_T = 20370

# calculate M (integral over exposure factor) [sr]
auger_params_for_M_calc = []
auger_params_for_M_calc.append(np.cos(np.deg2rad(lat)))
auger_params_for_M_calc.append(np.sin(np.deg2rad(lat)))
auger_params_for_M_calc.append(np.cos(np.deg2rad(theta_m)))

M, Merr = integrate.quad(m_integrand, 0, np.pi, args = auger_params_for_M_calc)

# calculate areas for each period [km^2]
A1 = alpha_T_1 / (M * deltat1)
A2 = alpha_T_2 / (M * deltat2)
A3 = alpha_T_3 / (M * deltat3)
