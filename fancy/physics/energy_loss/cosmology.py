from astropy.constants import c
from astropy import units as u

H0 = 67.3 * u.km / (u.s * u.Mpc)
Om = 0.272
Ol = 1 - Om

DH = (c / H0).to(u.Mpc)
