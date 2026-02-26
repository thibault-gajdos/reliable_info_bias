# MODEL REVISION: Sequential Belief & Confirmation Bias (kappa)

# Anchored Identifiability
# To resolve identifiability issues where parameters "swallow" each other:
# - theta (params[4]): FIXED at 1.0. No longer scales the softmax.
# - psi (params[5]): FIXED at 1.0. No longer scales the logit(proba).
# Any change in choice consistency is now forced into alpha or kappa.


#  "Confirmation Bias"
# - Constraint: [1.0, inf) via 1.0 + Phi_approx(...)
# - Logic: Only boosts the 'log_odds' if the cue color matches the 
#   current 'prior_log_odds'. 
# - This targets "Confirming Evidence" specifically rather than scaling all cues.

# SEQUENTIAL BELIEF STATE:
# - prior_log_odds: The dynamic starting point for trial 't', 
#   derived from 'V_b' (the belief state).
# - delta (params[6]): "Bias-Neglect". Controls the leak in the 
#   symmetric update of 'beliefcount_blue/red'.

# --- PARAMETER MAPPING ---
# [1] alpha  : Subjective Reliability 
# [2] beta   : Unreliable info bias / shift
# [3] lambda : Recency / Decay of cues within a trial
# [4] theta  : FIXED (1.0) - Choice consistency
# [5] psi    : FIXED (1.0) 
# [6] delta  : Bias-Neglect - Cross-trial belief memory
# [7] kappa  : Confirmation Bias - Selective boost for prior-consistent cues

functions {
  vector clamp_vector(vector x, real lo, real hi) {
    vector[num_elements(x)] out;
    for (i in 1:num_elements(x)) {
      out[i] = fmin(fmax(x[i], lo), hi);
    }
    return out;
  }

  // 1. Worker function for reduce_sum (Parallelized Likelihood)
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
    
      vector[7] params;
      params[1] = Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n,1]); // alpha (reliability)
      params[2] = mu_pr[2] + sigma_pr[2] * param_raw[n,2];             // beta (bias)
      params[3] = Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n,3]); // lambda (recency)
      
      // ANCHORS: Hardcoded to 1.0 to solve identifiability
      params[4] = 1.0; // theta (Softmax Inverse Temperature)
      params[5] = 1.0; // psi (Reliability scaling)
      
      params[6] = Phi_approx(mu_pr[6] + sigma_pr[6] * param_raw[n,6]);      // delta (bias-neglect)
      params[7] = 1.0 + Phi_approx(mu_pr[7] + sigma_pr[7] * param_raw[n,7]); // kappa (confirmation boost)

      // Initialize sequential belief state
      real beliefcount_blue = 1.0;
      real beliefcount_red  = 1.0;
      real V_b = 0.5;

      for (t in 1:Tsubj[n]) {
        vector[2] evidence = rep_vector(0.0, 2);
        int S = sample[n,t];

        // Add Dynamic Prior to Evidence
        real V_b_clamped = fmin(fmax(V_b, V_b_min), V_b_max);
        real prior_log_odds = log(V_b_clamped / (1 - V_b_clamped));
        evidence[1] += prior_log_odds;

        // Cue accumulation loop
        for (s in 1:S) {
          real l = logit(proba[n,t,s]);
          int c = color[n,t,s];
          
          // Evidence weight: alpha scales logit, beta shifts the mean
          real log_odds = params[1] * l + (1 - params[1]) * params[2];

          // Confirmation Bias (Kappa): Boost evidence only if it matches prior
          real k_boost = 1.0;
          if ((prior_log_odds > 0 && c == 1) || (prior_log_odds < 0 && c == 2))
            k_boost = params[7];

          evidence[c] += exp(params[3] * (s - S)) * log_odds * k_boost;
        }

        // Softmax Choice Probability
        lp += categorical_lpmf(choice[n,t] | softmax(clamp_vector(evidence, -100, 100)));

        // Sequential Bayesian Update (using feedback)
        int x = feedback[n,t];
        beliefcount_blue = params[6] * (beliefcount_blue - 1) + x + 1;
        beliefcount_red  = params[6] * (beliefcount_red  - 1) + (1 - x) + 1;
        V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red);
      }
    }
    return lp;
  }

  // 2. Helper for Evidence Calculation
  vector compute_evidence(int S, array[] int color_data, array[] real proba_data,
                          real alpha, real beta, real lambda,
                          real V_b, real kappa) {

    vector[2] evidence = rep_vector(0.0, 2);
    real prior_log_odds = logit(fmin(fmax(V_b, 0.001), 0.999));
    evidence[1] += prior_log_odds;

    for (s in 1:S) {
      real l = logit(proba_data[s]);
      int c = color_data[s];
      real log_odds = alpha * l + (1-alpha) * beta;

      real k_boost = 1.0;
      if ((prior_log_odds > 0 && c == 1) || (prior_log_odds < 0 && c == 2))
        k_boost = kappa;

      evidence[c] += exp(lambda * (s - S)) * log_odds * k_boost;
    }
    return evidence;
  }

  // 3. Helper for Log-Likelihood Calculation
  real compute_log_lik(int S, array[] int color_data, array[] real proba_data,
                       int choice, real alpha, real beta, real lambda,
                       real V_b, real kappa) {

    vector[2] evidence = compute_evidence(S, color_data, proba_data, alpha, beta, lambda, V_b, kappa);
    return categorical_lpmf(choice | softmax(clamp_vector(evidence, -100, 100)));
  }
}

data {
  int<lower=1> N;
  int<lower=1> T_max;
  int<lower=1> I_max;
  array[N] int<lower=1> Tsubj;
  array[N, T_max] int sample;
  array[N, T_max, I_max] int color;
  array[N, T_max, I_max] real proba;
  array[N, T_max] int choice;
  array[N, T_max] int feedback;
  int<lower=5> grainsize;
}

parameters {
  vector[7] mu_pr;
  vector<lower=0>[7] sigma_pr;
  matrix[N, 7] param_raw;
}

model {
  // Group-level priors (mu_pr[4] and [5] will not influence)
  mu_pr ~ std_normal();
  sigma_pr ~ std_normal();
  to_vector(param_raw) ~ std_normal();

  array[N] int indices;
  for (n in 1:N) indices[n] = n;

  target += reduce_sum(
    partial_sum,
    indices,
    grainsize,
    mu_pr, sigma_pr,
    Tsubj, sample, color, proba, choice,
    param_raw, feedback
  );
}

generated quantities {
  matrix[N, 7] params;
  array[N, T_max] real y_pred = rep_array(-1.0, N, T_max);
  vector[sum(Tsubj)] log_lik;

  // Group means (4 and 5 are fixed)
  real mu_alpha  = Phi_approx(mu_pr[1]);
  real mu_beta   = mu_pr[2];
  real mu_lambda = Phi_approx(mu_pr[3]);
  real mu_theta  = 1.0; 
  real mu_psi    = 1.0; 
  real mu_delta  = Phi_approx(mu_pr[6]);
  real mu_kappa  = 1.0 + Phi_approx(mu_pr[7]);

  int k = 0;
  for (n in 1:N) {
    params[n,1] = Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n,1]);
    params[n,2] = mu_pr[2] + sigma_pr[2] * param_raw[n,2];
    params[n,3] = Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n,3]);
    params[n,4] = 1.0; 
    params[n,5] = 1.0; 
    params[n,6] = Phi_approx(mu_pr[6] + sigma_pr[6] * param_raw[n,6]);
    params[n,7] = 1.0 + Phi_approx(mu_pr[7] + sigma_pr[7] * param_raw[n,7]);

    real beliefcount_blue = 1.0;
    real beliefcount_red  = 1.0;
    real V_b = 0.5;

    for (t in 1:Tsubj[n]) {
      k += 1;
      array[I_max] int color_trial = color[n,t];
      array[I_max] real proba_trial = proba[n,t];

      log_lik[k] = compute_log_lik(sample[n,t], color_trial, proba_trial, choice[n,t],
                                   params[n,1], params[n,2], params[n,3], V_b, params[n,7]);

      vector[2] evidence = compute_evidence(sample[n,t], color_trial, proba_trial,
                                            params[n,1], params[n,2], params[n,3], V_b, params[n,7]);

      y_pred[n,t] = categorical_rng(softmax(clamp_vector(evidence, -100, 100)));

      // Sequential Update for Generated Quantities
      int x = feedback[n,t];
      beliefcount_blue = params[n,6] * (beliefcount_blue - 1) + x + 1;
      beliefcount_red  = params[n,6] * (beliefcount_red  - 1) + (1 - x) + 1;
      V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red);
    }
  }
}

