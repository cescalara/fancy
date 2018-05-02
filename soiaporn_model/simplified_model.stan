data {
  int<lower=1> N_A; /* number of sources */ 
  int<lower=1> N; /* number of data points */
  
  unit_vector[3] varpi[N_A]; /* source locations */
  unit_vector[3] omega[N]; /* uhecr locations */
  real w[N_A]; /* weights */
}

parameters {
  //simplex[N_A] lambda; // mixing proportions (weights)
  //unit_vector[3] mu[N_A]; // source centres

  real<lower=0> f;
  real<lower=0> F_T; 
  real<lower=0> kappa; 
}

transformed parameters {
  real F = f * F_T;
  real F_A[N_A];

  for (i in 1:N_A) { 
    F_A[i] = w[i] * F;
  }
}

model {
  /* cache the weights */
  real log_w[N_A] = log(w); 

  /* priors */
  F_T ~ normal(500, 200);
  f ~ uniform(0, 1);
  kappa ~ uniform(1, 10);
  
  /* isotropic component */
  target += log (1 / (4 * pi()));

  /* mixture of source components */
  for (n in 1:N) {
    real lps[N_A] = log_w;
    for (n_a in 1:N_A) {
      lps[n_a] += kappa * dot_product(omega[n], varpi[n_a]) + log(kappa) - log(4 * pi() * sinh(kappa));
    }
    target += log_sum_exp(lps);
  }   
}
