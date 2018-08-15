import numpy as np
from scipy import integrate, optimize
from matplotlib import pyplot as plt

"""
Energy loss functions for propagation of UHECR protons. 
Based on the semi-analytical results of the continuous loss approximation, 
as described in detail in de Domenico and Insolia (2013) and also used by 
Khanin and Mortlock (2016).

The beta_* functions are loss rates for different processes. Units are as follows, unless otherwise stated:

beta_pi and beta_adi
--------------------
Energy, E [eV]
Redshift, z [dimensionless]
Loss rate, beta_* [yr^-1]

beta_bh
-------
Energy, E [EeV]
Redshift z, [dimnesionless]
Loss rate, beta_bh [s^-1]

NB: beta_bh is calculated in these units to avoid issues with numerical overflow.
NB: the forumla for beta_pi in de Domenico and Insolia (2013) is missing a minus sign 
    to have the correct for of Aexp(-B/E).

@author Francesca Capel
@date July 2018
"""

# ignore warnings
np.warnings.filterwarnings('ignore')

# Globals
Mpc_in_m = 3.084e22 # [m]
H0 = 70 # [km s^-1 Mpc^-1]
H0 = H0 * (1e3 / Mpc_in_m) # [s^-1]
c = 3.0e8 # [m s^-1]
DH = (c / H0) / Mpc_in_m # [Mpc]

def beta_pi(z, E):
    """
    GZK losses due to photomeson production.
    Originally parametrised by Anchordorqui et al. (1997)
    Based on Berezinsky and Grigor'eva (1988) suggestion of Aexp(-B/E) form.
    """
    
    p = [3.66e-8, (2.87e20 / 1e18), 2.42e-8]
    check = 6.86 * np.exp(-0.807 * z) * 1e20

    if (E <= check):
        return p[0] * (1 + z)**3 * np.exp(-p[1] / ((1 + z) * (E / 1e18)))
    if (E > check):
         return p[2] * (1 + z)**3

def beta_adi(z):
    """
    Losses due to adiabatic expansion. 
    Makes use of the hubble constant converted into units of [yr^-1].
    """
    
    H0 = 70.4 # s^-1 km/Mpc
    H0 = ((H0 / 3.086e22) * 1e3) * 3.154e7 # yr^-1
    lCDM = [0.272, 0.728]
    
    a = lCDM[0] * (1 + z)**3
    b = 1 - sum(lCDM) * (1 + z)**2

    return H0 * (a + lCDM[1] + b)**(0.5)

def phi_inf(xi):
    """
    The approximation of the phi(xi) function as xi->inf.
    Used in calculation of beta_bh.
    Described in Chodorowski et al. (1992).
    """
    
    d = [-86.07, 50.96, -14.45, 8.0 / 3.0]
    sum_term = sum([d_i * np.log(xi)**i for i, d_i in enumerate(d)])

    return xi * sum_term

def phi(xi):
    """
    Approximation of the phi(xi) function for different regimes
    of xi.
    Used in calculation of beta_bh.
    Described in Chodorowski et al. (1992). 
    """
    
    if (xi == 2):
        return (np.pi / 12) * (xi - 2)**4

    elif (xi < 25):

        c = [0.8048, 0.1459, 1.137e-3, -3.879e-6]
        sum_term = 0

        for i in range(len(c)):
            sum_term += c[i] * (xi - 2)**(i+1) 

        return (np.pi / 12) * (xi - 2)**4 / (1 + sum_term)

    elif (xi >= 25):
        
        phi_inf_term = phi_inf(xi)
        f = [2.910, 78.35, 1837]
        sum_term = sum([(f_i * xi**(-(i + 1))) for i, f_i in enumerate(f)])

        return phi_inf_term / (1 - sum_term)
        
def integrand(xi, E, z):
    """
    Integrand as a functional of phi(xi) used in calcultion of beta_bh.
    Described in de Domenico and Insolia (2013).
    """
    
    num = phi(xi)
    B = 1.02
    denom = np.exp( (B * xi) / ((1 + z) * E) ) - 1

    return num / denom
        
def beta_bh(z, E):
    """
    Losses due to Bethe-Heitler pair production.
    Described in de Domenico amnd Insolia (2013).
    """
    
    A = 3.44e-18 
    integ, err = integrate.quad(integrand, 2, np.inf, args = (E, z))

    return (A  / E**3) * integ 

def Ltot(z, E):
    """
    The total energy loss length
    """
    
    c = 3.064e-7 # Mpc/yr
    L = c / (beta_pi(z, E) + beta_adi(z) + (3.154e7 * beta_bh(z, E / 1e18)))
    return L


def make_energy_loss_plot(z, E):
    """
    Recreate figure 2 in de Domenico and Insolia (2013).
    """

    c = 3.064e-7 # Mpc/yr

    # adiabatic and GZK
    L_pi = [(c / beta_pi(z, e)) for e in E]
    L_adi = [(c / beta_adi(z)) for e in E]

    # pair production
    L_bh = [(c / (3.154e7 * beta_bh(z, e / 1e18))) for e in E]

    # total
    L_tot = [c / (beta_pi(z, e) + beta_adi(z) + (3.154e7*beta_bh(z, e/1e18)) ) for e in E]

    plt.figure(figsize = (10, 7))

    plt.plot(E, L_pi, linewidth = 5, alpha = 0.7, label = 'Photomeson production')
    plt.plot(E, L_adi, linewidth = 5, alpha = 0.7, label = 'Adiabatic')
    plt.plot(E, L_bh, linewidth = 5, alpha = 0.7, label = 'Pair production')
    plt.plot(E, L_tot, '--', linewidth = 5, alpha = 0.7, label = 'Total')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(5, 1e4)
    plt.xlabel('E [eV]')
    plt.ylabel('L [Mpc]')
    plt.legend()

def dEdr(r, E):
    """
    The ODE to solve for propagation energy losses.
    """
    z = r / DH
    return - E / Ltot(z, E) 

def dEdr_rev(r, E, D):
    """
    The ODE to solve for propagation energy losses.
    """
    r_rev = D - r
    z = r_rev / DH
    return E / Ltot(z, E) 


def get_arrival_energy(E, D):
    """
    Get the arrival energy for a given initial energy.
    Takes into account all propagation affects.
    Solves the ODE dE/dr = - E / Lloss. 
    NB: input and output E in EeV!
    """

    E = E * 1.0e18
    integrator = integrate.ode(dEdr).set_integrator('lsoda', method = 'bdf')
    integrator.set_initial_value(E, 0)
    r1 = D
    dr = 1
    
    while integrator.successful() and integrator.t < r1:
        integrator.integrate(integrator.t + dr)
        #print("%g %g" % (integrator.t, integrator.y))

    Earr = integrator.y / 1.0e18
    return Earr

def get_source_threshold_energy(Eth, D):
    """
    Get the equivalent source energy for a given arrival energy.
    Takes into account all propagation affects.
    Solves the ODE dE/dr = E / Lloss. 
    NB: input Eth and output E in EeV! 
    """

    Eth = Eth * 1.0e18
    integrator = integrate.ode(dEdr_rev).set_integrator('lsoda', method = 'bdf')
    integrator.set_initial_value(Eth, 0).set_f_params(D)
    r1 = D
    dr = 1
    
    while integrator.successful() and integrator.t < r1:
        integrator.integrate(integrator.t + dr)
        #print("%g %g" % (integrator.t, integrator.y))

    Eth_src = integrator.y / 1.0e18 
    return Eth_src

def get_Eth_src(Eth, D):
    """
    Find the source threshold energies for all sources.
    NB: input Eth in EeV!
    """

    Eth_src = []
    for d in D:
        Eth_src.append(get_source_threshold_energy(Eth, d)[0])

    return Eth_src


def get_Eex(Eth_src, alpha):
    """
    Find median E for a power law distribution
    described by slope alpha and minium Eth_src.
    """

    Eex = []
    for e in Eth_src:
        Eex.append(2**(1 / (alpha - 1)) * e)

    return Eex
    

def get_kappa_ex(E, B, D):
    """
    Find kappa_ex for a given B field.
    Based on the deflection approximation used in 
    Achterberg et al. 1999.
    """

    l = 1; # Mpc

    kappa_ex = []
    for i, e in enumerate(E):
        theta_p = 0.0401 * (e / 50)**(-1) * B * (D[i] / 10)**0.5 * l**0.5
        kappa_ex.append(2.3 / theta_p**2)

    return kappa_ex;
