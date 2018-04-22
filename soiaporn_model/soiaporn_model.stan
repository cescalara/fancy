// hiearchical model for UHECR arrival directions
// Paper: Soiaporn, K. et al., 2012. Multilevel Bayesian framework for modeling the production,
// propagation and detection of ultra-high energy cosmic rays. arXiv.org, astro-ph.HE(3), pp.1249–1285.
// Data: https://www.auger.org/index.php/document-centre/viewdownload/115-data/
// 2354-list-of-ultra-high-energy-cosmic-ray-events-2014


data {
  // UHECRs
  int<lower=0> i; // number of detected UHECR events
  real d[i, i]; // glon and glat of the detected UHECR events

  // sources
  int<lower=0> k; // number of potential sources
  real D[k]; // distance to the sources in Mpc
  real varpi[k, k]; // glon and glat of the sources
}

transformed data {


}

parameters {
  real F_t; // total flux
  real f; // fraction associated with sources
  //real kappa[i]; // concentration parameter of defelction
  int<lower=0> lamda[i]; // source label

  // hyperparameters
  real s; // scale of exponential prior for F_T
  int a, b; //shape parameters of the beta prior for f
}

model {

  // priors
  s = 0.01 * 4 * 3.14;
  F_t ~ exponential(s);

  a = 1;
  b = 1;
  f ~ beta(a, b)

  //lambda = 

  // likelihood
  
  
}

generated quantities {


}
