/**
 * Hierarchcial model for UHECR 
 * Based on the work by Soiaporn et al. (2012)
 * @author Francesca Capel
 * @date May 2018
 */

functions {

  /* compute the absolute value of a vector */
  real abs_val(vector input_vector) {
    real av;
    int n = num_elements(input_vector);

    real sum_squares = 0;
    for (i in 1:n) {
      sum_squares += (input_vector[i] * input_vector[i]);
    }
    av = sqrt(sum_squares);

    return av;
  }
  
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

  //int<lower=0,upper=N_A> lambda[N];
  real f_ki_factor;
  real f_ki_inner;
  real f_ki;
  real sum_F_A;
  
  /* priors */
  F_T ~ normal(N, 10);
  f ~ beta(1, 1);
  kappa ~ normal(100, 20);
  kappa_c ~ normal(1000, 10);

  /* labels */
  //lambda ~ categorical(w);

  /* likelihood */
  sum_F_A = sum(F_A);
  target += log(-sum_F_A) + (N * log(sum_F_A)); 

  f_ki = 1;
  for (k in 1:N_A) {
    for (i in 1:N) {
      if (k > 0) {
	f_ki_factor = (kappa_c * kappa) / (4 * pi() * sinh(kappa_c) * sinh(kappa));
	f_ki_inner = abs_val( (kappa_c * detected[i]) + (kappa * varpi[k]) );
	f_ki = sinh(f_ki_inner) / f_ki_inner;
      }
      else {
	f_ki = 1 / (4 * pi());
      }
      target += log(f_ki);
    }
  }
  
  
}
