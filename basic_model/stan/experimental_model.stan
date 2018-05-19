/**
 * Experimental model for UHECR 
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

  /* sources */
  real<lower=0> F_T; 
  simplex[N_A + 1] w;
  
  /* deflection */
  real<lower=0> kappa;  
  real<lower=0> kappa_c;  

}

transformed parameters {

  /* associated fraction */
  real<lower=0, upper=1> f = 1 - w[N_A + 1];
  
  /* source flux */
  vector[N_A + 1] F_A;

  for (k in 1:N_A + 1) {
    F_A[k] = w[k] * F_T;
  }

}

model {

  vector[N_A + 1] log_w = log(w);

  /* priors */
  F_T ~ normal(N, 10);
  f ~ beta(1, 1);
  kappa ~ normal(100, 20);
  
  /* FMM of vMF with isotropic component */
  for (n in 1:N) {
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
