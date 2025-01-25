/**
/*
/* Estimate parameters of vMF from samples.
/* 
*/

functions {
    real vMF_lpdf(real cos_theta, real kappa) {

    real lprob;
    if (kappa > 100) {
      lprob = kappa * cos_theta + log(kappa) - log(4 * pi()) - kappa + log(2);
    }
    else {
      lprob = kappa * cos_theta + log(kappa) - log(4 * pi() * sinh(kappa));
    }
    return lprob;
    
  }
}

data {
    int<lower=0> N;  /* number of samples */
    array[N] unit_vector[3] n;  /* samples */
    unit_vector[3] mu;  /* mean direction */
}

transformed data {
   array[N] real cos_thetas;  /* dot product between mean and samples */
   
   for (i in 1:N) {
        cos_thetas[i] = dot_product(n[i], mu);
   }

}

parameters {
    // unit_vector[3] mu;  /* mean direction */
    real<lower=0> kappa;  /* concentration parameter */
}

model {
    for (i in 1:N) {
        cos_thetas[i] ~ vMF(kappa);
    }
}