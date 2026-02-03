// MODEL REVISION: Sequential Belief & Continuous Confirmation Bias (Eta)

// 1. CONTINUOUS CONFIRMATION BOOST (params[7] = eta)
// - Replaced the binary "Kappa" switch with a continuous "Eta" ramp.
// - Boost = exp(eta * a), where 'a' is the alignment between cue and prior.
// - If V_b = 0.5: Alignment (a) is 0. Boost = exp(0) = 1.0. (No effect).
// - If V_b > 0.5 (Prior favors Blue):
//    * Blue cues get a BOOST (multiplier > 1.0).
//     * Red cues get a PENALTY (multiplier < 1.0).
// - If V_b < 0.5 (Prior favors Red):
//     * Red cues get a BOOST (multiplier > 1.0).
//     * Blue cues get a PENALTY (multiplier < 1.0).
//- Constraint: eta is bounded [0, 1] to prevent deterministic likelihoods.
//- INTERPRETATION: 
//      * eta = 0: No bias (Treats all cues fairly regardless of belief).
//      * eta = 1: Max bias (Prior strongly filters cues; contradicting info is ignored).

//2.
// - To fix the identifiability conflict between theta, psi, and alpha:
// - theta (params[4]): FIXED at 1.0 (Softmax Inverse Temperature).
// - psi   (params[5]): FIXED at 1.0 (Thurstonian Scale / Logit scaling).



functions {
  vector clamp_vector(vector x, real lo, real hi) {
    vector[num_elements(x)] out;
    for (i in 1:num_elements(x)) {
      out[i] = fmin(fmax(x[i], lo), hi);
    }
    return out;
  }

  // Continuous confirmation boost logic following Thibault's email
  real get_boost(int color, real V_b, real eta) {
    // a = 2p-1 if Blue, -(2p-1) if Red
    real a = (color == 1) ? (2 * V_b - 1) : -(2 * V_b - 1);
    return exp(eta * a);
  }
  
  
//    - Eta (params[7]): The "Bias Strength"" estimated for each subject [0, 1].
//     - a (Alignment): How well the cue matches the current belief [-1, 1].
//  Alignment measures the agreement between the cue color (c) and the current 
//  belief probability (V_b). It scales linearly:
// 
//    If Cue is Blue (c=1): a =  (2 * V_b - 1)
//     If Cue is Red  (c=2): a = -(2 * V_b - 1)
// 
//      If subject is 90% sure the underlying colour is Blue (V_b = 0.9):
//    - Blue cue: a = +0.8 (Strong Agreement)
//    - Red cue:  a = -0.8 (Strong Disagreement)

//   The term 'exp(Eta * a)' transforms alignment into a multiplicative weight:
//  
//     - NEUTRAL (V_b = 0.5): a = 0, exp(0) = 1.0. 
//       Evidence is weighted objectively (no bias).
//  
//     - AGREEMENT (a > 0): Multiplier > 1.0. 
//        The evidence is amplified (Confirmation Boost).
//  
//     - DISAGREEMENT (a < 0): Multiplier < 1.0. 
//       The evidence is supressed (Disconfirmation Penalty).

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
    
    for (i in 1:size(slice_indices)) {
      int n = slice_indices[i];
    
      vector[7] params;
      params[1] = Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n,1]); // alpha
      params[2] = mu_pr[2] + sigma_pr[2] * param_raw[n,2];             // beta
      params[3] = Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n,3]); // lambda
      params[4] = 1.0; // theta (FIXED)
      params[5] = 1.0; // psi (FIXED)
      params[6] = Phi_approx(mu_pr[6] + sigma_pr[6] * param_raw[n,6]); // delta
      params[7] = Phi_approx(mu_pr[7] + sigma_pr[7] * param_raw[n,7]); // eta

      real b_blue = 1.0;
      real b_red  = 1.0;
      real V_b = 0.5;

      for (t in 1:Tsubj[n]) {
        vector[2] evidence = rep_vector(0.0, 2);
        int S = sample[n,t];

        // Apply Clamping for stability
        real V_b_clamped = fmin(fmax(V_b, 0.001), 0.999);
        
        // Add Dynamic Prior to Evidence
        evidence[1] += log(V_b_clamped / (1 - V_b_clamped));

        for (s in 1:S) {
          real l = logit(proba[n,t,s]);
          int c = color[n,t,s];
          
          // Weighted log-odds
          real log_odds = params[1] * l + (1 - params[1]) * params[2];

          // Continuous Confirmation Boost (eta)
          real eta_boost = get_boost(c, V_b_clamped, params[7]);

          evidence[c] += exp(params[3] * (s - S)) * log_odds * eta_boost;
        }

        lp += categorical_lpmf(choice[n,t] | softmax(clamp_vector(evidence, -100, 100)));

        // Sequential Update (delta)
        int x = feedback[n,t];
        b_blue = params[6] * (b_blue - 1) + x + 1;
        b_red  = params[6] * (b_red  - 1) + (1 - x) + 1;
        V_b = b_blue / (b_blue + b_red);
      }
    }
    return lp;
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

  // Group-level summary statistics
  real mu_alpha  = Phi_approx(mu_pr[1]);
  real mu_beta   = mu_pr[2];
  real mu_lambda = Phi_approx(mu_pr[3]);
  real mu_delta  = Phi_approx(mu_pr[6]);
  real mu_eta    = Phi_approx(mu_pr[7]);

  int k = 0;
  for (n in 1:N) {
    // Map individual parameters
    params[n,1] = Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n,1]);
    params[n,2] = mu_pr[2] + sigma_pr[2] * param_raw[n,2];
    params[n,3] = Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n,3]);
    params[n,4] = 1.0; 
    params[n,5] = 1.0; 
    params[n,6] = Phi_approx(mu_pr[6] + sigma_pr[6] * param_raw[n,6]);
    params[n,7] = Phi_approx(mu_pr[7] + sigma_pr[7] * param_raw[n,7]);

    real b_blue = 1.0;
    real b_red  = 1.0;
    real V_b = 0.5;

    for (t in 1:Tsubj[n]) {
      k += 1;
      vector[2] evidence = rep_vector(0.0, 2);
      real V_b_clamped = fmin(fmax(V_b, 0.001), 0.999);
      
      evidence[1] += log(V_b_clamped / (1 - V_b_clamped));

      for (s in 1:sample[n,t]) {
        real l = logit(proba[n,t,s]);
        int c = color[n,t,s];
        real log_odds = params[n,1] * l + (1 - params[n,1]) * params[n,2];
        real eta_boost = get_boost(c, V_b_clamped, params[n,7]);
        evidence[c] += exp(params[n,3] * (s - sample[n,t])) * log_odds * eta_boost;
      }

      // Calculate Likelihood and Predictions
      log_lik[k] = categorical_lpmf(choice[n,t] | softmax(clamp_vector(evidence, -100, 100)));
      y_pred[n,t] = categorical_rng(softmax(clamp_vector(evidence, -100, 100)));

      // Sequential Update for the next trial
      int x = feedback[n,t];
      b_blue = params[n,6] * (b_blue - 1) + x + 1;
      b_red  = params[n,6] * (b_red  - 1) + (1 - x) + 1;
      V_b = b_blue / (b_blue + b_red);
    }
  }
}

