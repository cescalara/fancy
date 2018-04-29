data {
  //int<lower=0> N_A;
  
  unit_vector[3] varpi; 
  unit_vector[3] omega; 
  real w;
}

transformed data {
  real s = 0.01;
  int a = 1;
  int b = 10;
}

parameters {
  real F_T; 
 
  real<lower=0> kappa; 
}

transformed parameters {
  real F_A;

  //for (i in 1:N_A) 
  F_A = w * F_T;
}

model {
  F_T ~ exponential(s);
  kappa ~ uniform(a, b);

  target += kappa * dot_product(omega, varpi) + log(kappa) - log(4 * pi() * sinh(kappa));
}
