functions {
  vector clamp_vector(vector x, real lo, real hi) {
    vector[num_elements(x)] out;
    for (i in 1:num_elements(x)) {
      out[i] = fmin(fmax(x[i], lo), hi);
    }
    return out;
  }

  // Worker function for reduce_sum
  real partial_sum(array[] int slice_indices,
                   int start, int end,
                   vector mu_pr, vector sigma_pr,
                   array[] int Tsubj,
                   array[,] int sample,
                   array[,,] int color,
                   array[,,] real proba,
                   array[,] int choice,
                   matrix param_raw){
    
    real lp = 0;
    
    // Determine the number of subject-level parameters (now 6: alpha, beta, lambda, theta, psi, bias)
    int K = dims(param_raw)[2]; 

    for (i in 1:size(slice_indices)) {
      int n = slice_indices[i];
      
      // Transform parameters (K=6 in the updated model)
      vector[K] params; 
      
      // 1. alpha (Mixing Weight) - Bounded [0, 1]
      params[1] = Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n, 1]); 
      // 2. beta (Baseline Bias/Intercept) - Unbounded 
      params[2] = mu_pr[2] + sigma_pr[2] * param_raw[n, 2];      
      // 3. lambda (Time Decay) - Bounded [0, 1]
      params[3] = Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n, 3]); 
      // 4. theta (Inverse Temperature/Choice Sensitivity) - Bounded [0, 5]
      params[4] = Phi_approx(mu_pr[4] + sigma_pr[4] * param_raw[n, 4]) * 5; 
      // 5. psi (Log-Odds Weight) - Bounded [0, 5]
      params[5] = Phi_approx(mu_pr[5] + sigma_pr[5] * param_raw[n, 5])*5; 

      //  CHANGE: New Bias Parameter Transformation 
      // 6. bias (Subjective Prior Probability) - Bounded [0, 1]
      real bias_n = Phi_approx(mu_pr[6] + sigma_pr[6] * param_raw[n, 6]);
      // Convert bias probability to log-odds (prior evidence)
      real prior_log_odds = log(bias_n / (1 - bias_n));

      for (t in 1:Tsubj[n]) {
        vector[2] evidence = rep_vector(0.0, 2);
        int sample_size = sample[n, t];

        //  CHANGE: Add the subjective prior bias to the evidence for Choice 1
        evidence[1] += prior_log_odds;

        for (s in 1:sample_size) {
          real p = proba[n, t, s];
          real l = logit(p);
          int color_val = color[n, t, s];
          
          // Log-odds of sample based on alpha, psi, beta
          real log_odds = params[1] * l * params[5] + (1-params[1])* params[2]; 
          
          // Accumulate time-weighted evidence
          evidence[color_val] += exp(params[3] * (s - sample_size)) * log_odds;
        }
        vector[2] evidence_safe = clamp_vector(evidence, -100, 100);
        lp += categorical_lpmf(choice[n, t] | softmax(params[4]*evidence_safe));
      }
    }
    return lp;
  }
  

  vector compute_evidence(int sample_size, array[] int color_data, array[] real proba_data, 
                          real alpha, real beta, real lambda, real theta, real psi, real bias_p) { //  CHANGE: Added bias_p
    vector[2] evidence = rep_vector(0.0, 2);
    
    // CHANGE: Add the subjective prior bias
    evidence[1] += log(bias_p / (1 - bias_p)); 
    
    for (s in 1:sample_size) {
      real p = proba_data[s];
      real l = logit(p);
      real log_odds = alpha * l * psi + (1-alpha) * beta;
      evidence[color_data[s]] += exp(lambda * (s - sample_size)) * log_odds;
    }
    return evidence;
  }

  real compute_log_lik(int sample_size, array[] int color_data, array[] real proba_data, 
                       int choice, real alpha, real beta, real lambda, real theta, real psi, real bias_p) { //  CHANGE: Added bias_p
    
    // CHANGE: Updated call to compute_evidence with bias_p
    vector[2] evidence = compute_evidence(sample_size, color_data, proba_data, alpha, beta, lambda, theta, psi, bias_p); 
    vector[2] evidence_safe = clamp_vector(evidence, -100, 100);
    return categorical_lpmf(choice | softmax(theta*evidence_safe));
  }
}

data {
  int<lower=1> N;
  int<lower=1> T_max;
  int<lower=1> I_max;
  array[N] int<lower=1> Tsubj;
  array[N, T_max] int<lower=-1> sample;
  array[N, T_max, I_max] int<lower=-1, upper=2> color;
  array[N, T_max, I_max] real<lower=-1, upper=1> proba;
  array[N, T_max] int<lower=-1, upper=2> choice;

  // grainsize for reduce_sum
  int<lower=5> grainsize;
}

parameters {
  //  CHANGE: Increased dimension from [5] to [6] for the new bias parameter
  vector[6] mu_pr;
  vector<lower=0>[6] sigma_pr;
  matrix[N, 6] param_raw;
}

model {
  // Priors for 6 parameters (all standard normal)
  mu_pr ~ std_normal();
  sigma_pr ~ std_normal();
  to_vector(param_raw) ~ std_normal();
  
  // Create array of indices for parallelization
  array[N] int indices;
  for (n in 1:N) indices[n] = n;
  
  // Use reduce_sum for parallelization
  // The signature of reduce_sum remains the same, but the vector/matrix sizes passed (mu_pr, sigma_pr, param_raw)
  // now correspond to the 6 parameters defined above.
  target += reduce_sum(partial_sum, indices, grainsize,
                       mu_pr, sigma_pr,
                       Tsubj, sample, color, proba, choice, param_raw);
}

generated quantities {
  // CHANGE: Increased dimension from [N, 5] to [N, 6]
  matrix[N, 6] params;
  array[N, T_max] real y_pred = rep_array(-1.0, N, T_max);
  vector[sum(Tsubj)] log_lik;
  
  // CHANGE: Added output for the group mean of the bias parameter
  real mu_alpha = Phi_approx(mu_pr[1]);
  real mu_beta = mu_pr[2];
  real mu_lambda = Phi_approx(mu_pr[3]);
  real mu_theta = Phi_approx(mu_pr[4]) * 5;
  real mu_psi = Phi_approx(mu_pr[5])*5;
  real mu_bias = Phi_approx(mu_pr[6]); // NEW: Group mean P(Bias)

  int k = 0;
  for (n in 1:N) {
    // CHANGE: Transformation for all 6 parameters
    params[n, 1] = Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n, 1]); // alpha
    params[n, 2] = mu_pr[2] + sigma_pr[2] * param_raw[n, 2]; // beta
    params[n, 3] = Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n, 3]); // lambda
    params[n, 4] = Phi_approx(mu_pr[4] + sigma_pr[4] * param_raw[n, 4]) * 5; // theta
    params[n, 5] = Phi_approx(mu_pr[5] + sigma_pr[5] * param_raw[n, 5])*5; // psi
    params[n, 6] = Phi_approx(mu_pr[6] + sigma_pr[6] * param_raw[n, 6]); // NEW: bias (P)

    for (t in 1:Tsubj[n]) {
      k += 1;

      array[I_max] int color_trial;
      array[I_max] real proba_trial;

      for (i in 1:I_max) {
        color_trial[i] = color[n, t, i];
        proba_trial[i] = proba[n, t, i];
      }
      
      // CHANGE: Updated call to compute_evidence with the 6th parameter (bias)
      vector[2] evidence = compute_evidence(sample[n, t], color_trial, proba_trial,
                                            params[n, 1], params[n, 2], params[n, 3], 
                                            params[n, 4], params[n, 5], params[n, 6]); // params[n, 6] is the bias_p
      
      vector[2] evidence_safe = clamp_vector(evidence, -100, 100);
      vector[2] prob = softmax(params[n,4]*evidence_safe);
      y_pred[n, t] = categorical_rng(prob);
      
      // CHANGE: Updated call to compute_log_lik with the 6th parameter (bias)
      log_lik[k] = compute_log_lik(sample[n, t], color_trial, proba_trial,
                                   choice[n, t],
                                   params[n, 1], params[n, 2], params[n, 3], 
                                   params[n, 4], params[n, 5], params[n, 6]);
    }
  }
}
