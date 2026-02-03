// safety utility - in bayesian models, extreme params can create infinite evidence.
//The softmax function can't handle number that large and will return NaN, crashing the model. 
functions {
  vector clamp_vector(vector x, real lo, real hi) {
    vector[num_elements(x)] out;
    for (i in 1:num_elements(x)) {
      out[i] = fmin(fmax(x[i], lo), hi);
    }
    return out;
  }

// for parallelization, instead of calculating one subject at a time, 
// stan split the subejct into slices to run on different CPU cores. 
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
      
 // mu_pr - the group average 
 // sigma_pr - the group spread
 // param_raw - the individual's deviation from the group 
 // Phi_approx = Cumulative Normal Distribution, it squashes any number into the range (0,1). 

      vector[6] params;              // FIX: psi removed, theta fixed
      params[1] = Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n, 1]); // alpha
      params[2] = mu_pr[2] + sigma_pr[2] * param_raw[n, 2];             // beta
      params[3] = Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n, 3]); // lambda
      params[4] = Phi_approx(mu_pr[4] + sigma_pr[4] * param_raw[n, 4]) * 2; // delta
      // CHANGE 1: Removed 1.0 + Phi_approx.
      //  params[5] is now 'eta' (sensitivity to confirmation).
      //  We allow this to be negative or positive to test for confirmation vs disconfirmation.
      params[5] = mu_pr[5] + sigma_pr[5] * param_raw[n, 5]; // eta

// initializing beliefs - every subject starts with a flat prior, a count
// of 1 for each color represents a Beta distribition Beta (1,1), 
// which is a uniform distribution.
// beliefcount represents pseudo counts from previous experience, starting at 1 and 1 means the subject is 50/50
// V_b is the subjects current probability that the blue state is true. 

      real beliefcount_blue = 1.0; 
      real beliefcount_red = 1.0;  
      real V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red); 

      for (t in 1:Tsubj[n]) {
        vector[2] evidence = rep_vector(0.0, 2);
        int sample_size = sample[n, t];
        
// prior_log_odds converts probability to the log-odds scale>
        real V_b_clamped = fmin(fmax(V_b, V_b_min), V_b_max);
        real prior_log_odds = log(V_b_clamped / (1 - V_b_clamped));
        evidence[1] += prior_log_odds;


// p - the reliabolity of the cue 
// l - the strenght of the cue in the log-space 
// colour_val - which side the cue points to (1 for blue, 2 for red)
        for (s in 1:sample_size) {
          real p = proba[n, t, s];
          real l = logit(p);
          int color_val = color[n, t, s];
          real log_odds = params[1] * l + (1 - params[1]) * params[2];
          
         //
          //  CHANGE 2: SMOOTH KAPPA MECHANISM
          //  Instead of the IF/ELSE threshold, we calculate 'a' (belief direction/strength)
          //  and use the exponential of eta * a.
            //
          real a;
          if (color_val == 1) {
            a = 2 * V_b_clamped - 1; // Positive if belief favors blue
          } else {
            a = 2 * (1 - V_b_clamped) - 1; // Positive if belief favors red
          }
          real current_kappa = exp(params[5] * a);

          evidence[color_val] += exp(params[3] * (s - sample_size)) * log_odds * current_kappa;
        }
        
        vector[2] evidence_safe = clamp_vector(evidence, -100, 100);
        lp += categorical_lpmf(choice[n, t] | softmax(evidence_safe));
// turn the evidence vector into probabilities that sum to 100% 

// x - the outcome (1 or 0) 
        int x = feedback[n, t];
        beliefcount_blue = params[4] * (beliefcount_blue - 1) + x + 1;
        beliefcount_red  = params[4] * (beliefcount_red  - 1) + (1 - x) + 1;
        // updates the probabolity of the blue state Vb, that will be the starting point
        // the prior for the next trial's cue accumulation
        V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red);
      }
    }
    return lp;
  }
  
  // functions identical in logic to the main loop
  // compute_evidence replicated the within trial cue loop, it returns a vector of 
  // 2 values (blue evidence, red evidence)
  
  vector compute_evidence(int sample_size, array[] int color_data, array[] real proba_data, 
                          real alpha, real beta, real lambda, real V_b, real eta) {
    vector[2] evidence = rep_vector(0.0, 2);
    real V_b_clamped = fmin(fmax(V_b, 0.001), 0.999);
    real prior_log_odds = log(V_b_clamped / (1 - V_b_clamped));
    evidence[1] += prior_log_odds;

    for (s in 1:sample_size) {
      real l = logit(proba_data[s]);
      int color_val = color_data[s];
      real log_odds = alpha * l + (1 - alpha) * beta;
      
      // CHANGE 3: Integrated smooth kappa into helper function 
      real a;
      if (color_val == 1) {
          a = 2 * V_b_clamped - 1;
      } else {
          a = 2 * (1 - V_b_clamped) - 1;
      }
      real current_kappa = exp(eta * a);
      
      evidence[color_val] += exp(lambda * (s - sample_size)) * log_odds * current_kappa;
    }
    return evidence;
  }

 real compute_log_lik(int sample_size, array[] int color_data, array[] real proba_data, 
                       int choice, real alpha, real beta, real lambda, 
                       real V_b, real eta) { 
    vector[2] evidence = compute_evidence(sample_size, color_data, proba_data, 
                                          alpha, beta, lambda, V_b, eta); 
    vector[2] evidence_safe = clamp_vector(evidence, -100, 100);
    // calculates the log proability of the subjects actual choice given the model's parameters
    return categorical_lpmf(choice | softmax(evidence_safe));
  }
}


// defines what we're feeding into Stan 
data {
  int<lower=1> N; // number of subjects 
  int<lower=1> T_max; // maximum number of trials
  int<lower=1> I_max; // maximum number of cues per trial 
  array[N] int<lower=1> Tsubj;
  array[N, T_max] int sample;  // many many cues were actually shown in that specific trial 
  array[N, T_max, I_max] int color;
  array[N, T_max, I_max] real proba; // the actual cue colourrs and their associated p-values. 
  array[N, T_max] int choice;
  array[N, T_max] int feedback; // the binary otucome (0,1) seen at the end of each trial 
  int<lower=5> grainsize; 
}

// define global and local parameters

parameters {
  vector[5] mu_pr;                // FIX: psi, theta removed
  vector<lower=0>[5] sigma_pr;    
  matrix[N, 5] param_raw;
}

// where Bayesian inference actually happens 
// Prior - we use std_normal (normal(0,1)). Because the parameters are transformed
// using phi approx, a standard normal prior is weakly informative so it allows the data
// to speak for itself while keeping the paramters within a realistic range. 
model {
  mu_pr ~ std_normal();
  sigma_pr ~ normal(0, 1);
  to_vector(param_raw) ~ std_normal();
  
  array[N] int indices;
  for (n in 1:N) indices[n] = n;
  
  //tells stan to sum up all the choice probabilities calculated in partial_sum
  
  target += reduce_sum(partial_sum, indices, grainsize,
                       mu_pr, sigma_pr,
                       Tsubj, sample, color, proba, choice, param_raw, feedback);
}

// runs after the model has finished fitting

generated quantities {
    matrix[N, 5] params;
    array[N, T_max] real y_pred = rep_array(-1.0, N, T_max);
    vector[sum(Tsubj)] log_lik;

    int k = 0;
    for (n in 1:N) {
        params[n, 1] = Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n, 1]); // alpha
        params[n, 2] = mu_pr[2] + sigma_pr[2] * param_raw[n, 2];             // beta
        params[n, 3] = Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n, 3]); // lambda
        params[n, 4] = Phi_approx(mu_pr[4] + sigma_pr[4] * param_raw[n, 4]); // delta
        params[n, 5] = mu_pr[5] + sigma_pr[5] * param_raw[n, 5];             // eta

        real beliefcount_blue = 1.0;
        real beliefcount_red = 1.0;
        real V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red);

        for (t in 1:Tsubj[n]) {
            k += 1;
            int sample_size = sample[n, t];
            array[I_max] int color_trial;
            array[I_max] real proba_trial;
            for (i in 1:I_max) {
                color_trial[i] = color[n, t, i];
                proba_trial[i] = proba[n, t, i];
            }

            log_lik[k] = compute_log_lik(sample_size, color_trial, proba_trial,
                                         choice[n, t],
                                         params[n, 1], params[n, 2], params[n, 3],
                                         V_b, params[n, 5]);

            vector[2] evidence = compute_evidence(sample_size, color_trial, proba_trial,
                                                  params[n, 1], params[n, 2], params[n, 3],
                                                  V_b, params[n, 5]);

            vector[2] evidence_safe = clamp_vector(evidence, -100, 100);
            y_pred[n, t] = categorical_rng(softmax(evidence_safe));

            int x = feedback[n, t];
            beliefcount_blue = params[n, 4] * (beliefcount_blue - 1) + x + 1;
            beliefcount_red  = params[n, 4] * (beliefcount_red  - 1) + (1 - x) + 1;
            V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red);
        }
    }
}

