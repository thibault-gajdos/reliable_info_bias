functions {
  vector clamp_vector(vector x, real lo, real hi) {
    vector[num_elements(x)] out;
    for (i in 1:num_elements(x)) {
      out[i] = fmin(fmax(x[i], lo), hi);
    }
    return out;
  }

  real partial_sum(array[] int slice_indices,
                   int start, int end,
                   vector mu_pr, vector sigma_pr,
                   array[] int Tsubj,
                   array[,] int sample,
                   array[,,] int color,
                   array[,,] real proba,
                   array[,] int choice,
                   matrix param_raw,
                   array[,] int feedback) { 
    
    real lp = 0;
    real V_b_min = 0.001;
    real V_b_max = 0.999;
    
    for (i in 1:size(slice_indices)) {
      int n = slice_indices[i];
    
      vector[4] params;
      params[1] = Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n, 1]);       // alpha
      params[2] = mu_pr[2] + sigma_pr[2] * param_raw[n, 2];                   // beta
      params[3] = Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n, 3]);       // lambda
      params[4] = Phi_approx(mu_pr[4] + sigma_pr[4] * param_raw[n, 4]) * 2.0; // delta [0,2]

      real beliefcount_blue = 1.0; 
      real beliefcount_red = 1.0;  
      real V_b = 0.5; 

      for (t in 1:Tsubj[n]) {
        vector[2] evidence = rep_vector(0.0, 2);
        int sample_size = sample[n, t];
        
        real V_b_clamped = fmin(fmax(V_b, V_b_min), V_b_max);
        evidence[1] += log(V_b_clamped / (1 - V_b_clamped)); 

        for (s in 1:sample_size) {
          real p = proba[n, t, s];
          real l = logit(p);
          int color_val = color[n, t, s];
          
          // PSI REMOVED
          real log_odds = params[1] * l + (1 - params[1]) * params[2]; 
          evidence[color_val] += exp(params[3] * (s - sample_size)) * log_odds;
        }
        
        vector[2] evidence_safe = clamp_vector(evidence, -100, 100);
        
        // THETA FIXED TO 1.0
        lp += categorical_lpmf(choice[n, t] | softmax(1.0 * evidence_safe));
        
        int x = feedback[n, t];
        // MATH AS ORIGINALLY REQUESTED (Clamps Removed)
        beliefcount_blue = params[4] * (beliefcount_blue - 1) + x + 1;
        beliefcount_red  = params[4] * (beliefcount_red - 1) + (1 - x) + 1;
        
        V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red);
      }
    }
    return lp;
  }
  
  vector compute_evidence(int sample_size, array[] int color_data, array[] real proba_data, 
                          real alpha, real beta, real lambda, real V_b) {
    vector[2] evidence = rep_vector(0.0, 2);
    real V_b_clamped = fmin(fmax(V_b, 0.001), 0.999);
    evidence[1] += log(V_b_clamped / (1 - V_b_clamped)); 

    for (s in 1:sample_size) {
      real p = proba_data[s];
      real l = logit(p);
      real log_odds = alpha * l + (1 - alpha) * beta;
      evidence[color_data[s]] += exp(lambda * (s - sample_size)) * log_odds;
    }
    return evidence;
  }

  real compute_log_lik(int sample_size, array[] int color_data, array[] real proba_data, 
                       int choice, real alpha, real beta, real lambda, real V_b) {
    vector[2] evidence = compute_evidence(sample_size, color_data, proba_data, alpha, beta, lambda, V_b); 
    vector[2] evidence_safe = clamp_vector(evidence, -100, 100);
    return categorical_lpmf(choice | softmax(1.0 * evidence_safe));
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
  array[N, T_max] int<lower=0, upper=1> feedback; 
  int<lower=5> grainsize;
}

parameters {
  vector[4] mu_pr;
  vector<lower=0>[4] sigma_pr;
  matrix[N, 4] param_raw;
}

model {
  mu_pr ~ std_normal();
  sigma_pr ~ std_normal();
  to_vector(param_raw) ~ std_normal();
  
  array[N] int indices;
  for (n in 1:N) indices[n] = n;
  
  target += reduce_sum(partial_sum, indices, grainsize,
                       mu_pr, sigma_pr,
                       Tsubj, sample, color, proba, choice, param_raw, feedback);
}

generated quantities {
    matrix[N, 4] ind_params;
    array[N, T_max] real y_pred = rep_array(-1.0, N, T_max);
    vector[sum(Tsubj)] log_lik;

    real mu_alpha  = Phi_approx(mu_pr[1]);
    real mu_beta   = mu_pr[2];
    real mu_lambda = Phi_approx(mu_pr[3]);
    real mu_delta  = Phi_approx(mu_pr[4]) * 2.0;

    int k = 0;
    for (n in 1:N) {
        ind_params[n, 1] = Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n, 1]); 
        ind_params[n, 2] = mu_pr[2] + sigma_pr[2] * param_raw[n, 2];             
        ind_params[n, 3] = Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n, 3]); 
        ind_params[n, 4] = Phi_approx(mu_pr[4] + sigma_pr[4] * param_raw[n, 4]) * 2.0; 

        real beliefcount_blue = 1.0;
        real beliefcount_red = 1.0;
        real V_b = 0.5;

        for (t in 1:Tsubj[n]) {
            k += 1;
            array[I_max] int color_trial;
            array[I_max] real proba_trial;

            for (i in 1:I_max) {
                color_trial[i] = color[n, t, i];
                proba_trial[i] = proba[n, t, i];
            }

            log_lik[k] = compute_log_lik(sample[n, t], color_trial, proba_trial,
                                         choice[n, t],
                                         ind_params[n, 1], ind_params[n, 2], ind_params[n, 3], V_b);

            vector[2] ev = compute_evidence(sample[n, t], color_trial, proba_trial,
                                            ind_params[n, 1], ind_params[n, 2], ind_params[n, 3], V_b);
            
            vector[2] ev_safe = clamp_vector(ev, -100, 100);
            y_pred[n, t] = categorical_rng(softmax(1.0 * ev_safe));

            int x = feedback[n, t];
            beliefcount_blue = ind_params[n, 4] * (beliefcount_blue - 1) + x + 1;
            beliefcount_red  = ind_params[n, 4] * (beliefcount_red - 1) + (1 - x) + 1;
            
            V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red);
        }
    }
}


