data {
  int<lower=1> N_A; /* number of sources */ 
  int<lower=1> N; /* number of data points */
  
  unit_vector[3] varpi[N_A]; /* source locations */
  unit_vector[3] omega[N]; /* uhecr locations */
  simplex[N_A] w;
  real<lower=0> kappa;
}

parameters { 
  real<lower=0> F_T; 
  //real<lower=0> kappa;

  real<lower=0, upper=1> f;
  // simplex[2] f;
}

transformed parameters {
  real F = f * F_T;
  real F_A[N_A];

  for (i in 1:N_A) { 
    F_A[i] = w[i] * F;
  }
}

model {
  vector[N_A] log_w = log(w);

  real lps_sum = 0;
 
  /* priors */
  F_T ~ normal(N, 200);
  f ~ uniform(0.7, 1);
  //kappa ~ uniform(1, 20);
  
  /* mixture of source components */
  for (n in 1:N) {
    vector[N_A] lps = log_w;
    real lpb = log(1 - f) + log( 1 / (4 * pi()) );

    for (n_a in 1:N_A) {
      lps[n_a] += kappa * dot_product(omega[n], varpi[n_a]) + log(kappa) - log(4 * pi() * sinh(kappa));
    }
    lps_sum += log_sum_exp(lps);
    lps_sum += log(f);
    target += log_sum_exp(lpb, lps_sum);
  }

  /* target */
}
