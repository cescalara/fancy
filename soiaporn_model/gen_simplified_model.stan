functions {
  /* vMF RNG */
  // vector vMF_rng(vector varpi, real kappa) {

    
  //  return omega;
  //}

}


data {

  /* sources */
  int<lower=1> N_A;
  unit_vector[3] varpi[N_A]; 

  /* known parameters */
  int<lower=0> F_T;
  real<lower=0, upper=1> f;
  simplex[N_A + 1] w;
  real<lower=0> kappa;
  
}

transformed data {

  real F = f * F_T;
  real F_A[N_A];
  
  for (i in 1:N_A) { 
    F_A[i] = w[i] * F;
  }
 
}

parameters {

  unit_vector[3] omega[F_T]; 
  
}

model {
  vector[N_A + 1] log_w = log(w);

  /* priors */
  //F_T ~ normal(N, 10);
  //f ~ beta(1, 1);
  //kappa ~ normal(100, 20);
  
  /* FMM of vMF with isotropic component */
  for (n in 1:F_T) {
    vector[N_A + 1] lps = log_w;

    for (n_a in 1:(N_A + 1)) {
      
      if (n_a < N_A + 1) {
	lps[n_a] += kappa * dot_product(omega[n], varpi[n_a]) + log(kappa) - log(4 * pi() * sinh(kappa));	
      }
      else {
	lps[n_a] += log(1 / ( 4 * pi() ));
      }
      
    }
    
    target += log_sum_exp(lps);
  }

}
