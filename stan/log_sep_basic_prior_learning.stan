// Hierarchical cue integration model with Bayesian dynamic prior updating
// Implements bias-neglect parameter delta to discount past evidence in Beta-Binomial updates

// Defines the full trial structure: cue colors, probabilities, subject choices, and feedback (whether blue was correct). 
data {
  int<lower=1> N;                      // Number of subjects
  int<lower=1> T_max;                  // Max trials across all subjects
  int<lower=1> I;                      // Number of cues per trial
  int Tsubj[N];                        // Number of trials per subject
  int<lower=1, upper=2> color[N, T_max, I];  // Cue colors: blue=1, red=2
  real proba[N, T_max, I];                     // Cue probabilities
  int<lower=1, upper=2> choice[N, T_max];      // Subject choices per trial
  int<lower=0, upper=1> feedback[N, T_max];    // Feedback: 1=blue correct, 0=red correct
}

// mu_w, mu_pr, mu_delta: Group-level means for weights, cue distortion/bias (α, β), and neglect parameter (δ). 
// sigma_w, sigma, sigma_delta: Group-level SDs for above.
//
// *_raw: Subject-level unconstrained parameters. 
// *They’re transformed later via Phi_approx to meaningful scales.
parameters {
  // Group-level means
  vector[I-1] mu_w;        // Mean cue weights (first 5 cues)
  vector[2] mu_pr;         // Means for alpha (cue distortion) and beta (cue bias)
  real mu_delta;           // Mean of bias-neglect parameter delta
  

  // Group-level standard deviations
  vector<lower=0>[I-1] sigma_w;    // SD cue weights
  vector<lower=0>[2] sigma;        // SD alpha and beta
  real<lower=0> sigma_delta;       // SD of delta
 

  // Subject-level raw parameters (unconstrained)
  vector[N] alpha_raw;      // Cue distortion per subject
  vector[N] beta_raw;       // Cue bias per subject
  vector[N] delta_raw;      // Bias-neglect per subject
  matrix[N, I-1] w_raw;     // Raw cue weights for first 5 cues

}

// Each subject’s: alpha = sigmoid-scaled cue distortion factor // beta = cue bias 
// delta = bias-neglect parameter ∈ [0,1] (i.e., how much prior evidence is discounted)
// w = sequential cue weights (first 5 estimated, 6th fixed to 1)
transformed parameters {
  vector<lower=0>[N] alpha;                 // Transformed cue distortion (scaled)
  vector[N] beta;                           // Transformed cue bias
  vector<lower=0, upper=1>[N] delta;        // Bias-neglect parameter ∈ [0,1]
  matrix<lower=0, upper=2>[N, I-1] w;       // Transformed cue weights

  for (n in 1:N) {
    alpha[n] = Phi_approx(mu_pr[1] + sigma[1] * alpha_raw[n]) * 20;
    beta[n]  = mu_pr[2] + sigma[2] * beta_raw[n];
    delta[n] = Phi_approx(mu_delta + sigma_delta * delta_raw[n]);


    for (i in 1:(I - 1))
      w[n, i] = Phi_approx(mu_w[i] + sigma_w[i] * w_raw[n, i]) * 2;
  }
}

// Initialises Beta distribution parameters beliefcount_blue and beliefcount_red to (1,1) → uniform prior
// Computes belief as:
// V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red)
// Incorporates this belief as the log-odds prior into the softmax decision:
//   evidence[1] += log(V_b / (1 - V_b));
// Cue evidence is added as usual (weighted, distorted by α and shifted by β)
model {
  // Priors on group-level parameters
  mu_pr ~ std_normal();
  sigma ~ normal(0, 0.2);

  mu_delta ~ std_normal();
  sigma_delta ~ normal(0, 0.2);

  mu_w ~ std_normal();
  sigma_w ~ normal(0, 0.2);


  // Priors on subject-level parameters
  alpha_raw ~ std_normal();
  beta_raw ~ std_normal();
  delta_raw ~ std_normal();
  to_vector(w_raw) ~ std_normal();


  // Main model loop: per subject
  for (n in 1:N) {
    // Initialise Beta prior parameters for belief about "blue" being correct
    real beliefcount_blue = 1;
    real beliefcount_red = 1;
    real V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red);  // Initial belief = 0.5 (uniform)

    // Loop over trials for subject n
    for (t in 1:Tsubj[n]) {
      vector[2] evidence = rep_vector(0, 2);
      vector[2] val;
      real weight;

      // Incorporate current prior belief as log-odds into evidence for blue
      evidence[1] += log(V_b / (1 - V_b));

      // Accumulate cue-based evidence weighted by cue weights, distortion, and bias
      for (i in 1:I) {
        weight = (i == 6) ? 1 : w[n, i];  // 6th cue weight fixed to 1
        real log_odds = alpha[n] * log(proba[n, t, i] / (1 - proba[n, t, i])) + beta[n];

        if (color[n, t, i] == 1) {
          evidence[1] += weight * log_odds;  // Blue cue evidence
        } else {
          evidence[2] += weight  * log_odds;  // Red cue evidence
        }
      }

      // Compute choice probabilities via softmax over evidence for blue and red
      val = softmax(evidence);

      // Observed choice likelihood
      choice[n, t] ~ categorical(val);

      // Bayesian updating of Beta prior parameters beliefcount_blue, beliefcount_red with feedback and bias neglect delta
      // feedback[n,t] = 1 if blue correct, 0 if red correct
      beliefcount_blue = delta[n] * (beliefcount_blue - 1) + feedback[n, t] + 1;
      beliefcount_red = delta[n] * (beliefcount_red - 1) + (1 - feedback[n, t]) + 1;

      // Update belief for next trial
      V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red);

      // Clamp belief to avoid numerical issues with log(0)
      V_b = fmin(fmax(V_b, 0.001), 0.999);
    }
  }
}

generated quantities {
  // Summary statistics for group-level parameters
  real mu_alpha      = Phi_approx(mu_pr[1]) * 20;
  real mu_beta       = mu_pr[2];
  real mu_delta_val  = Phi_approx(mu_delta);
  // real mu_gamma_val = mu_gamma;   // group-level mean asymmetry

  real mu_w1 = Phi_approx(mu_w[1]) * 2;
  real mu_w2 = Phi_approx(mu_w[2]) * 2;
  real mu_w3 = Phi_approx(mu_w[3]) * 2;
  real mu_w4 = Phi_approx(mu_w[4]) * 2;
  real mu_w5 = Phi_approx(mu_w[5]) * 2;

  // Variables to hold log-likelihoods, predictions, and belief trajectories
  real log_lik[N, T_max];
  real y_pred[N, T_max];
  real V_b_hist[N, T_max];

  // Initialise values
  for (n in 1:N) {
    for (t in 1:T_max) {
      y_pred[n, t] = -1;
      log_lik[n, t] = 0;
      V_b_hist[n, t] = -1;
    }
  }

  // Simulate belief dynamics and predictions per subject
  for (n in 1:N) {
    real beliefcount_blue = 1;
    real beliefcount_red = 1;
    real V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red);

    for (t in 1:Tsubj[n]) {
      vector[2] evidence = rep_vector(0, 2);
      vector[2] val;
      real weight;

      // prior log-odds
      evidence[1] += log(V_b / (1 - V_b));

      // cue evidence
      for (i in 1:I) {
        weight = (i == 6) ? 1 : w[n, i];
        real log_odds = alpha[n] * log(proba[n, t, i] / (1 - proba[n, t, i])) + beta[n];

        if (color[n, t, i] == 1) {
          evidence[1] += weight * log_odds;
        } else {
          evidence[2] += weight * log_odds;
        }
      }

      val = softmax(evidence);
      log_lik[n, t] = categorical_lpmf(choice[n, t] | val);
      y_pred[n, t]  = categorical_rng(val);

      V_b_hist[n, t] = V_b;

      // Update beta prior with feedback and bias neglect
      beliefcount_blue = delta[n] * (beliefcount_blue - 1) + feedback[n, t] + 1;
      beliefcount_red = delta[n] * (beliefcount_red - 1) + (1 - feedback[n, t]) + 1;

      V_b = beliefcount_blue / (beliefcount_blue + beliefcount_red);
      V_b = fmin(fmax(V_b, 0.001), 0.999);
    }
  }
}

