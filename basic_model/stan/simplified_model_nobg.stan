
data {

  int<lower=1> N_A;
  int<lower=1> N; 
 
  unit_vector[3] varpi[N_A]; 
  unit_vector[3] omega[N]; 

}

parameters { 

  real<lower=0> F_T; 
  real<lower=0> kappa;
  simplex[N_A] w;

}

transformed parameters {

  real F_A[N_A];

  for (i in 1:N_A) { 
    F_A[i] = w[i] * F_T;
  }

}

model {
  vector[N_A] log_w = log(w);
 
  /* priors */
  F_T ~ normal(N, 200);
  kappa ~ normal(100, 10);
  
  /* FMM of vMF */
  for (n in 1:N) {
    vector[N_A] lps = log_w;

    for (n_a in 1:N_A) {
      lps[n_a] += kappa * dot_product(omega[n], varpi[n_a]) + log(kappa) - log(4 * pi() * sinh(kappa));
    }
    
    target += log_sum_exp(lps);
  }

}
