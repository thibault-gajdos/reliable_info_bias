// Hierarchical linear likelihood model of information aggregation
// with sequential effect

data {
  int<lower=1> N; // number of subjects
  int<lower=1> T_max; // maximum number of trials among all subjects
  int<lower=1> I; // number of info by trials
  int Tsubj[N]; // number of trials for each subject
  int<lower=1, upper=2> color[N,T_max, I]; // sequences of info colors: blue=1, red=2
  real proba[N,T_max, I]; // sequences of probabilities
  int<lower=1, upper=2> choice[N,T_max]; // choices for each subject
}

parameters {
  //hypergroup param
  vector[I-1] mu_w; // sequential weights mean (for  i:1-5; w6 normaliszed to 1)
  vector[3] mu_pr; // alpha, beta, prior proba means
  vector<lower=0>[I-1] sigma_w; // sequential weights var 
  vector<lower=0>[3] sigma; // alpha, beta, bias 

  //  subject level param
  vector[N] alpha_raw;
  vector[N] beta_raw;
  vector[N] bias_raw;  
  matrix[N, I-1] w_raw;
}

transformed parameters {
  vector<lower = 0>[N] alpha;
  vector[N] beta;
  vector<lower = 0, upper=1>[N] bias;  
  matrix<lower=0, upper=2>[N, I-1] w;
  
  for (n in 1:N){
     alpha[n] = Phi_approx(mu_pr[1]+ sigma[1]*alpha_raw[n])*20; 
     beta[n] = mu_pr[2]+sigma[2]*beta_raw[n]; 
     bias[n] = Phi_approx(mu_pr[3]+ sigma[3]*bias_raw[n]);
     
      for (i in 1:5){
        	w[n,i] = Phi_approx(mu_w[i]+ sigma_w[i]*w_raw[n,i])*2;
      }
  }
}

model {
  // Hypermarameters
  mu_w    ~ std_normal();
  sigma_w ~ normal(0,.2);
  mu_pr   ~ std_normal();
  sigma   ~ normal(0,.2);

  // individual parameters
  alpha_raw      ~ std_normal();
  beta_raw       ~ std_normal();
  bias_raw       ~ std_normal();
  to_vector(w_raw) ~ std_normal();
  

  for (n in 1:N) {
    for (t in 1:Tsubj[n]) {
      vector[2] evidence;
      real weight;
      vector[2] val;
      evidence = rep_vector(0, 2);
      
      // bias instead of prior - free parameter represents subjective prior belief in favour of blue 
      
      evidence[1] += log(bias[n]/ (1 - bias[n])); // add the bias expressed as negative (<50% - red bias) or positive (>50% - blue bias) bias in favour of blue

      
      // likelihood
      for (i in 1:I) {
        	if (i == 6){
        	  weight = 1;
        	}
        	else{
        	  weight = w[n,i];
        	}
        	
          if (color[n, t, i] == 1) { // color = blue
            evidence[1] += weight*(alpha[n] * log(proba[n, t, i] / (1 - proba[n, t, i])) + beta[n]);
          }
          if (color[n, t, i] == 2) { // color = red
            evidence[2] += weight*(alpha[n] * log(proba[n, t, i] / (1 - proba[n, t, i])) + beta[n]);
          }
      }
      
      // softmax choice
      val = softmax(evidence);
      choice[n,t] ~ categorical(val);
    }
  }
}

generated quantities {
  // For group level parameters
  real mu_alpha;
  real mu_beta;
  real mu_bias;  
  real  mu_w1;
  real  mu_w2;
  real  mu_w3;
  real  mu_w4;
  real  mu_w5;  

  // For log likelihood calculation
  real log_lik[N,T_max];
  
  // For posterior predictive check
  real y_pred[N, T_max];
  
  // Set all posterior predictions to -1 (avoids NULL values)
  for (n in 1:N) {
    for (t in 1:T_max) {
      y_pred[n, t] = -1;
      log_lik[n , t] = 0;
  	}
  }
  
  mu_alpha   = Phi_approx(mu_pr[1])*20;
  mu_beta    = mu_pr[2]; 
  mu_bias    = Phi_approx(mu_pr[3]);
  mu_w1      = Phi_approx(mu_w[1])*2;
  mu_w2      = Phi_approx(mu_w[2])*2;
  mu_w3      = Phi_approx(mu_w[3])*2;
  mu_w4      = Phi_approx(mu_w[4])*2;
  mu_w5      = Phi_approx(mu_w[5])*2;
  
  { // local section, this saves time and space
    for (n in 1:N) {
      // define values
      vector[2] evidence;
      real weight;
      vector[2] val;
      evidence = rep_vector(0, 2);
      for (t in 1:Tsubj[n]) {
      	evidence = rep_vector(0, 2);
      	
      	evidence[1] += log(bias[n]/ (1 - bias[n]));
      	
        	for (i in 1:I) {
        	  if (i == 6){
        	    weight = 1;
        	  }
        	  else{
        	    weight = w[n,i];
        	  }
        	  
        	  if (color[n, t, i] == 1) { // color = blue
        	    evidence[1] += weight*( alpha[n] * log(proba[n, t, i] / (1 - proba[n, t, i])) + beta[n]);
        	  }
        	  if (color[n, t, i] == 2) {  // color = red
        	    evidence[2] += weight*(alpha[n] * log(proba[n, t, i] / (1 - proba[n, t, i])) + beta[n]);
        	  }
        	}
        	
      	// Calculate log likelihood and store it in the array
      	val = softmax(evidence);
      	log_lik[n,t] = categorical_lpmf(choice[n, t] | val);
      	y_pred[n,t] = categorical_rng(val);
      }
    }
  }
}

