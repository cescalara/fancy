/* simplified model for the UHECR model of Soiaporn et al. (2012)
 * Francesca Capel April 2018
 */

data {
  int<lower=0> N_A; // number of AGN
  real varpi[J][J]; // coordinates of AGN
  real D[j]; // distance to AGN
  real omega[J][J]; // coordinates of observed UHECR
}

parameters {
  real F_T; // total flux
  //real f; // associated fraction
  real kappa<lower=1, upper=10>; // concentration parameter

  /* hyperparameters */
  real s = 0.01;
  int a = 1;
  int b = 10;
}

transformed parameters {
}

model {
  // prior on F_T
  F_T ~ exponential(s)
  // prior on f
  // prior on kappa
  kappa ~ uniform(a, b)
  omega ~ von_mises(mu, kappa)
}
