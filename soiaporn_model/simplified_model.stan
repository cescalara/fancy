
data {

  /* sources */
  int<lower=1> N_A;
  unit_vector[3] varpi[N_A]; 

  /* uhecr */
  int<lower=1> N; 
  unit_vector[3] omega[N]; 

}

parameters { 

  real<lower=0> F_T; 
  real<lower=0> kappa;

  simplex[N_A + 1] w;
  real<lower=0, upper=1> f;
  // simplex[2] f;
}

transformed parameters {

  real F = f * F_T;
  real F_A[N_A];

  for (i in 2:N_A + 1) { 
    F_A[i] = w[i] * F;
  }
}

model {
  vector[N_A + 1] log_w = log(w);

  //real lpb = log(1 - f) + log( 1 / (4 * pi()) );
  //real lps_sum = 0;
 
  /* priors */
  F_T ~ normal(N, 200);
  f ~ normal(0.9, 0.1);
  kappa ~ normal(100, 20);
  
  /* FMM of vMF */
  for (n in 1:N) {
    vector[N_A + 1] lps = log_w;

    for (n_a in 1:(N_A + 1)) {
      
      if (n_a == 1) {
	lps[n_a] += log(1 / ( 4 * pi() ))
      }
      else {
	lps[n_a] += kappa * dot_product(omega[n], varpi[n_a]) + log(kappa) - log(4 * pi() * sinh(kappa));	
      }
      
    }
    
    target += log_sum_exp(lps);
  }

  //lps_sum += log(f);
 
  /* target */
  //target += log_sum_exp(lpb, lps_sum);

}
