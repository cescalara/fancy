
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

  int<lower=0> lambda[i];
}

transformed parameters {

  //real<lower=0, upper=1> f;
  real F_A[N_A] + 1;

  for (i in 1:N_A + 1) { 
    F_A[i] = w[i] * F_T;
  }
}

model {
  vector[N_A + 1] log_w = log(w);

  /* priors */
  F_T ~ normal(N, 10);
  f ~ beta(1, 1);
  kappa ~ normal(100, 20);

  /* model */
  /* rough idea, need to handle lambda properly */
  lambda ~ categorial(w);
  F ~ poisson(F_A);

  /* */
  //f_ik
}
