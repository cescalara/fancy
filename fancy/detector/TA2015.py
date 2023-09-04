import numpy as np
from datetime import date
from scipy import integrate
from astropy import units as u
from astropy.coordinates import EarthLocation

from .exposure import m_integrand, kappa_dval

"""
Constants from information provided in TA Collaboration from
May 11, 2008 to May 11, 2015 as given by Abbasi et al. 2018 and 
Abu-Zayyad et al. 2013.
This information will be used by Abbasi et. al. 2014. as we have the 
public TA data from this paper.

The exposures are evaluated as with the Auger ones provided by Francesca Capel.
i.e. "M values calculated by integrating the m_integrand in exposure.py
over the unit sphere for certain values of theta_m."

* The latitudes, longitudes, and height are taken from Abbasi et. al. 2014.
* Information on total exposure from Abbasi et al. 2018 and Abu-Zayyad et al. 2013.
* Only the period of observation is provided, and no specific on / off-time are 
*   given in these articles. As os such, the detector off-time is neglected 
*   as it is < 1% (Ivanov PhD Thesis, 2012); this should be corrected for if this is a 
*   major concern.

For reference:
- Abbasi et al. 2014 is the TA hotspot paper
- Abbasi et al. 2018 is the Declination Dependence evidence paper
- Abu-Zayyad et al. 2013 is the paper that the hotspot paper refers to.

@author Keito Watanabe
@date July 2021
"""
# position of TA [rad]
lat = np.deg2rad(39.3)  # +39.3 deg N
lon = np.deg2rad(-112.91)  # 112.91 deg W
height = 1400. # [m]
auger_location = EarthLocation(lat = lat * u.rad, lon = lon * u.rad,
                               height = height * u.m)

# threshold incidence angle [rad]
# above 10^19 eV, theta_m = 55
# below this, theta_m = 45
theta_m = np.deg2rad(55)

# define observation period based on Abu-Zayyad et al. 2013 and Abbasi et al. 2018
# assume no off-time
period_start = date(2008, 5, 11)  # first observation date
period_1_start = period_start
period_1_end = date(2012, 5, 20)
period_2_start = date(2012, 5, 20)  # assuming no off-time
period_2_end = date(2015, 5, 11)
period_last_end = period_2_end  # last observation date

# start year & date of observation
start_year = 2008

# find length of each period in units of years
deltat1 = (period_1_end - period_1_start).days / 365.25
deltat2 = (period_2_end - period_2_start).days / 365.25
deltat = (period_last_end - period_1_start).days / 365.25

# total exposures in [km^2 sr yr]
alpha_T = 8300  # 2008 - 2015
alpha_T_1 = 3690  # 2008 - 2012
alpha_T_2 = alpha_T - alpha_T_1  # 2012 - 2015

# calculate M (integral over exposure factor) [sr]
detector_params = []
detector_params.append(np.cos(lat))
detector_params.append(np.sin(lat))
detector_params.append(np.cos(theta_m))
detector_params.append(alpha_T)
M, Merr = integrate.quad(m_integrand, 0, np.pi, args = detector_params)
detector_params.append(M)

# calculate areas for each period [km^2]
A = alpha_T / (M * deltat)
A1 = alpha_T_1 / (M * deltat1)
A2 = alpha_T_2 / (M * deltat2)

# reconstruction uncertainty for arrival direction
# ~1.7 as given by Abbasi et al. 2014
# > 1.5 for E > 10^19 eV from Abbasi et al. 2018
sig_omega = 1.7
kappa_d = kappa_dval(sig_omega)

# reconstruction uncertainty for energy
f_E = 0.20

# threshold energy [EeV]
Eth = 57

# For convenience
detector_properties = {}
detector_properties['label'] = 'TA2015'
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