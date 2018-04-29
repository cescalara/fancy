data {
  int<lower=1> N_A; // number of sources 
  int<lower=1> N; // number of data points
  
  unit_vector[3] varpi[N_A]; // source locations
  unit_vector[3] omega[N]; // UHECR locations
  real w[N_A]; // weights
}

parameters {
  //simplex[N_A] lambda; // mixing proportions
  //unit_vector[3] mu[N_A]; // source centres
  
  real F_T; 
  real<lower=0> kappa; 
}

transformed parameters {
  real F_A[N_A];

  for (i in 1:N_A) { 
    F_A[i] = w[i] * F_T;
  }
}

model {
  real log_w[N_A] = log(w); 

  F_T ~ normal(500, 20);
  kappa ~ uniform(1, 10);

  for (n in 1:N) {
    real lps[N_A] = log_w;
    for (n_a in 1:N_A) {
      lps[n_a] += kappa * dot_product(omega[n], varpi[n_a]) + log(kappa) - log(4 * pi() * sinh(kappa));
    }
    target += log_sum_exp(lps);
  }   
}
