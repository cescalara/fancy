/**
 * Simple simulation of an extragalactic point source.
 * For use with GMF lensing calculations
 *
 * @author Francesca Capel
 * @date December 2022
 */

functions {

#include energy_spectrum.stan
#include uhecr_propagation.stan
#include vMF.stan
#include observatory_exposure.stan
#include utils.stan
  
}

data {

  /* number of events to simulate */
  int<lower=0> N;

  /* source positions */
  unit_vector[3] varpi;
  real D;

  /* source spectrum */
  real alpha;
  real<lower=0> Eth;
  
  /* deflection */
  real<lower=0> B;
  int Z; 
  
}

transformed data {
  
  /* definitions */
  array[1] real x_r;
  array[0] int x_i;
  real Eth_src;
  array[1, 1] real D_in;
  real D_kappa;
  
  /* 
  D in Mpc / 10 for kappa calculation 
  Subtract by 20kpc to account for galactic boundary
  */
  D_in[1, 1] = (D / 3.086) * 100 - 0.02; 
  D_kappa = ((D / 3.086) * 10) - 0.2; // Mpc / 10
 
  /* Eth_src */
  x_r[1] = 1.0e4; // approx inf
  Eth_src = get_Eth_src_sim(Eth, D_in, x_r, x_i)[1];
  
}

generated quantities {

  array[N] real E;
  array[N] real kappa;
  array[N] real Earr;
  
  for (i in 1:N) {

    E[i] = spectrum_rng(alpha, Eth_src);
    kappa[i] = get_kappa(E[i], B, D_kappa, Z);
    Earr[i] = get_arrival_energy_sim(E[i], D_in[1], x_r, x_i);

  }
  
}

