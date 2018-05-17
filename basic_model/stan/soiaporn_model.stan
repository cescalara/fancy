/**
 * Hierarchcial model for UHECR 
 * Based on the work by Soiaporn et al. (2012)
 * @author Francesca Capel
 * @date May 2018
 */

functions {

  /* factor for association of UHECR i to source k */
  //
  
  /* get multiplicity for a given lambda */
  /*
  int multiplicity(vector lambda, int k) {
    int m = 0;

    for (i in 1:num_elements(lambda)) {
      if (lambda[i] == k) {
	m += 1;
      }
    }
    return m;
  }
  */
}

data {

  /* sources */
  int<lower=1> N_A;
  unit_vector[3] varpi[N_A]; 

  /* uhecr */
  int<lower=1> N; 
  unit_vector[3] detected[N]; 

}

parameters { 

  /* associated fraction */
  real<lower=0, upper=1> f;
  
  /* sources */
  real<lower=0> F_T; 
  simplex[N_A + 1] w;
  int<lower=0> lambda[i];
  
  /* deflection */
  real<lower=0> kappa;  
  real<lower=0> kappa_c;  

}

transformed parameters {

  /* source flux */
  real F = F_T * f;
  real F_A[N_A];

  for (k in 1:N_A) { 
    F_A[k] = w[k] * F;
  }

}

model {

  real f_ki_factor, f_ki_inner, f_ki;
  
  /* priors */
  F_T ~ normal(N, 10);
  f ~ beta(1, 1);
  kappa ~ normal(100, 20);
  kappa_c ~ normal(1000, 10);

  /* labels */
  lambda ~ categorical(w);

  /* likelihood */
  sum_F_A = sum(F_A);
  target += log(-sum_F_A) + (N * log(sum_F_A)); 

  f_ki = 1;
  for (i in 1:N) {
    if (lambda[i] > 0) {
      f_ki_factor = (kappa_c * kappa) / (4 * pi() * sinh(kappa_c) * sinh(kappa));
      f_ki_inner = (kappa_c * detected[i]) + (kappa * varpi[lambda[i]]);
      f_ki = sinh(f_ki_inner) / f_ki_inner;
    }
    else {
      f_ki = 1 / (4 * pi());
    }
    target += log(f_ki);
  }

  /* complicated likelihood */
  /*
  fk_fac = 1;
  for (k in 1:N_A) {
    m_k = multiplicity(lambda, k); 
    Fk_fac *= pow(F_A[k], m_k) * exp(-F_A[k]);
  }
  f_ki = 1;
  for (i in 1:N) {
    if (lambda[i] > 0) {
      f_ki_factor = (kappa_c * kappa) / (4 * pi() * sinh(kappa_c) * sinh(kappa));
      f_ki_inner = (kappa_c * detected[i]) + (kappa * varpi[lambda[i]]);
      f_ki *= sinh(f_ki_inner) / f_ki_inner;
    }
    else {
      f_ki *= 1 / (4 * pi());
    }
  }
  */
  
  
}
