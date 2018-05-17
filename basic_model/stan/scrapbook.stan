/* place to store some test code */


/* complicated likelihood */
  /*
  // for all possible lambda...  
  fk_fac = 1;
  for (k in 1:N_A) {
    m_k = multiplicity(lambda, k); 
    Fk_fac *= pow(F_A[k], m_k) * exp(-F_A[k]);
  }
  f_ki = 1;
  for (i in 1:N) {
    if (lambda[i] > 0) {
      f_ki_factor = (kappa_c * kappa) / (4 * pi() * sinh(kappa_c) * sinh(kappa));
      f_ki_inner = (kappa_c * detected[i]) + (kappa * varpi[lambda[i]]);
      f_ki *= sinh(f_ki_inner) / f_ki_inner;
    }
    else {
      f_ki *= 1 / (4 * pi());
    }
  }
  */


 /* get multiplicity for a given lambda */
  /*
  int multiplicity(vector lambda, int k) {
    int m = 0;

    for (i in 1:num_elements(lambda)) {
      if (lambda[i] == k) {
	m += 1;
      }
    }
    return m;
  }
  */

