  functions {
  
  real KAPPA() { return 10.0; }

  vector clamp_vector(vector x, real lo, real hi) {
    vector[num_elements(x)] out;
    for (i in 1:num_elements(x)) {
      out[i] = fmin(fmax(x[i], lo), hi);
    }
    return out;
  }

  // Work out the current Blue belief from previous feedback and the instruction.
  // Rho sets the starting instruction weight. Delta controls how long it lasts.
  //
  // 1 = Blue more likely
  // 2 = Red more likely
  // 3 = one colour more likely, direction unspecified
  // 4 = no instruction / neutral

  real compute_Vb(int instr,
                  real rho,   // How much weight the instruction gets at the start
                  real Delta, // How far from 0.5 the undirected instruction suggests the base rate is
                  real blue_feedback,
                  real red_feedback,
                  real instruction_decay) {

    // Start from an even belief and use discounted feedback to update it.
    real a = 1.0 + blue_feedback;
    real b = 1.0 + red_feedback;

    real V_base = a / (a + b);

    // No instruction: use feedback alone.
    if (instr == 4) {
      return V_base;
    }

    // Blue instruction: give extra weight to possible base rates above 0.5.
    // Rho mixes the Blue instruction with the initial even belief.

    if (instr == 1) {

      real prob_region =
        1.0 - beta_cdf(0.5 | a, b);

      real prob_region_shift =
        1.0 - beta_cdf(0.5 | a + 1.0, b);

      // Inside/outside weight ratio is phi^(delta^t). Phi comes from rho.
      // The instruction becomes weaker after each trial, using delta^t.
      real outside_weight =
        pow(1.0 - rho, instruction_decay);

      real inside_weight =
        pow(1.0 + rho, instruction_decay);

      real denominator =
        outside_weight
        + (inside_weight - outside_weight) * prob_region;

      real numerator =
        V_base *
        (outside_weight
         + (inside_weight - outside_weight) * prob_region_shift);

      return numerator / denominator;
    }

    // Red instruction: give extra weight to possible base rates below 0.5.
    // Rho mixes the Red instruction with the initial even belief.

    if (instr == 2) {

      real prob_region =
        beta_cdf(0.5 | a, b);

      real prob_region_shift =
        beta_cdf(0.5 | a + 1.0, b);

      // Inside/outside weight ratio is phi^(delta^t). Phi comes from rho.
      // The instruction becomes weaker after each trial, using delta^t.
      real outside_weight =
        pow(1.0 - rho, instruction_decay);

      real inside_weight =
        pow(1.0 + rho, instruction_decay);

      real denominator =
        outside_weight
        + (inside_weight - outside_weight) * prob_region;

      real numerator =
        V_base *
        (outside_weight
         + (inside_weight - outside_weight) * prob_region_shift);

      return numerator / denominator;
    }

    // Undirected instruction: favour either colour being clearly more likely.
    // rho * Uniform on the two outer regions
    // + (1-rho) * Uniform(0,1)

    {
      real lower_bound = 0.5 - Delta;
      real upper_bound = 0.5 + Delta;

      real region_length = 1.0 - 2.0 * Delta;

      real prob_region =
          beta_cdf(lower_bound | a, b)
        + 1.0
        - beta_cdf(upper_bound | a, b);

      real prob_region_shift =
          beta_cdf(lower_bound | a + 1.0, b)
        + 1.0
        - beta_cdf(upper_bound | a + 1.0, b);

      // Inside/outside weight ratio is phi^(delta^t). Phi comes from rho.
      // The instruction becomes weaker after each trial, using delta^t.
      real outside_weight =
        pow(1.0 - rho, instruction_decay);

      real inside_weight =
        pow(1.0 - rho + rho / region_length, instruction_decay);

      real denominator =
        outside_weight
        + (inside_weight - outside_weight) * prob_region;

      real numerator =
        V_base *
        (outside_weight
         + (inside_weight - outside_weight) * prob_region_shift);

      return numerator / denominator;
    }
  }

  real partial_sum(array[] int slice_indices,
                   int start, int end,
                   vector mu_pr, vector sigma_pr,
                   real Delta,
                   array[] int Tsubj,
                   array[,] int sample,
                   array[,,] int color,
                   array[,,] real proba,
                   array[,] int choice,
                   matrix param_raw,
                   array[,] int feedback,
                   array[] int instr) {

    real lp = 0;
    real V_b_min = 0.001;
    real V_b_max = 0.999;

    for (i in 1:size(slice_indices)) {
      int n = slice_indices[i];

      vector[8] params;

      params[1] = Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n, 1]) * 6; // alpha
      params[2] = mu_pr[2] + sigma_pr[2] * param_raw[n, 2];                 // beta
      params[3] = Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n, 3]);     // lambda
      params[4] = mu_pr[4] + sigma_pr[4] * param_raw[n, 4];                 // eta
      params[5] = Phi_approx(mu_pr[5] + sigma_pr[5] * param_raw[n, 5]) * 6; // alpha0
      params[6] = mu_pr[6] + sigma_pr[6] * param_raw[n, 6];                 // beta0
      params[7] = Phi_approx(mu_pr[7] + sigma_pr[7] * param_raw[n, 7]);     // delta
      params[8] = Phi_approx(mu_pr[8] + sigma_pr[8] * param_raw[n, 8]);     // rho

      // Start with no feedback. The instruction sets the first belief.
      real feedback_blue = 0.0;
      real feedback_red = 0.0;

      // The instruction has its full weight before the first trial.
      real instruction_decay = 1.0;

      real V_b = compute_Vb(instr[n],
                            params[8],
                            Delta,
                            feedback_blue,
                            feedback_red,
                            instruction_decay);

      for (t in 1:Tsubj[n]) {
        vector[2] evidence = rep_vector(0.0, 2);
        int sample_size = sample[n, t];

        real V_b_clamped =
          fmin(fmax(V_b, V_b_min), V_b_max);

        real P =
          params[5] * logit(V_b_clamped) + params[6];

        evidence[1] += P;

        // Use a smooth measure of which colour the decision belief favours.
        // This also lets Stan estimate the parameters without a sudden jump.
        real A = tanh(0.5 * KAPPA() * P);

        for (s in 1:sample_size) {
          real p = proba[n, t, s];
          real l = logit(p);
          int color_val = color[n, t, s];

          real log_odds =
            params[1] * l + params[2];

          real a =
            (color_val == 1) ? A : -A;

          real gain =
            exp(params[4] * a);

          evidence[color_val] +=
            exp(params[3] * (s - sample_size))
            * log_odds
            * gain;
        }

        vector[2] evidence_safe =
          clamp_vector(evidence, -100, 100);

        lp += categorical_lpmf(
          choice[n, t] | softmax(evidence_safe)
        );

        // Previous feedback loses some weight before we add the new result.
        int x = feedback[n, t];

        feedback_blue =
          params[7] * feedback_blue + x;

        feedback_red =
          params[7] * feedback_red + (1 - x);

        // The original instruction also loses some weight after each trial.
        instruction_decay *= params[7];

        // This updated belief is used at the start of the next trial.
        V_b = compute_Vb(instr[n],
                         params[8],
                         Delta,
                         feedback_blue,
                         feedback_red,
                         instruction_decay);
      }
    }

    return lp;
  }

  vector compute_evidence(int sample_size,
                          array[] int color_data,
                          array[] real proba_data,
                          real alpha,
                          real beta,
                          real lambda,
                          real V_b,
                          real eta,
                          real alpha0,
                          real beta0) {

    vector[2] evidence = rep_vector(0.0, 2);

    real V_b_clamped =
      fmin(fmax(V_b, 0.001), 0.999);

    real P =
      alpha0 * logit(V_b_clamped) + beta0;

    evidence[1] += P;

    // Check how much each sample agrees with the current decision belief.
    real A = tanh(0.5 * KAPPA() * P);

    for (s in 1:sample_size) {
      real l = logit(proba_data[s]);
      int color_val = color_data[s];

      real log_odds =
        alpha * l + beta;

      real a =
        (color_val == 1) ? A : -A;

      real gain =
        exp(eta * a);

      evidence[color_val] +=
        exp(lambda * (s - sample_size))
        * log_odds
        * gain;
    }

    return evidence;
  }

  real compute_log_lik(int sample_size,
                       array[] int color_data,
                       array[] real proba_data,
                       int choice,
                       real alpha,
                       real beta,
                       real lambda,
                       real V_b,
                       real eta,
                       real alpha0,
                       real beta0) {

    vector[2] evidence =
      compute_evidence(sample_size,
                       color_data,
                       proba_data,
                       alpha,
                       beta,
                       lambda,
                       V_b,
                       eta,
                       alpha0,
                       beta0);

    vector[2] evidence_safe =
      clamp_vector(evidence, -100, 100);

    return categorical_lpmf(
      choice | softmax(evidence_safe)
    );
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

  // 1 = Blue more likely
  // 2 = Red more likely
  // 3 = one colour more likely, unspecified
  // 4 = neutral / no instruction
  array[N] int<lower=1, upper=4> instr;

  int<lower=5> grainsize;
}

parameters {
  // Eight parameters can vary between participants. Delta is shared by the group.
  vector[8] mu_pr;
  vector<lower=0>[8] sigma_pr;
  matrix[N, 8] param_raw;
  real Delta_raw;
}

model {
  array[N] int indices;
  real Delta = 0.5 * Phi_approx(Delta_raw);

  mu_pr[1:3] ~ std_normal();

  mu_pr[4] ~ normal(0, 0.5); // eta
  mu_pr[5] ~ std_normal();   // alpha0
  mu_pr[6] ~ normal(0, 0.5); // beta0
  mu_pr[7] ~ std_normal();   // delta
  mu_pr[8] ~ std_normal();   // rho
  Delta_raw ~ std_normal(); // One Delta for the undirected group

  sigma_pr ~ normal(0, 1);

  to_vector(param_raw) ~ std_normal();

  for (n in 1:N)
    indices[n] = n;

  target += reduce_sum(
    partial_sum,
    indices,
    grainsize,
    mu_pr,
    sigma_pr,
    Delta,
    Tsubj,
    sample,
    color,
    proba,
    choice,
    param_raw,
    feedback,
    instr
  );
}

generated quantities {
  real mu_alpha =
    Phi_approx(mu_pr[1]) * 6;

  real mu_beta =
    mu_pr[2];

  real mu_lambda =
    Phi_approx(mu_pr[3]);

  real mu_eta =
    mu_pr[4];

  real mu_alpha0 =
    Phi_approx(mu_pr[5]) * 6;

  real mu_beta0 =
    mu_pr[6];

  real mu_delta =
    Phi_approx(mu_pr[7]);

  real mu_rho =
    Phi_approx(mu_pr[8]);

  // Delta is shared, rather than estimated separately for each participant.
  real mu_Delta = 0.5 * Phi_approx(Delta_raw);

  matrix[N, 8] params;

  array[N, T_max] real y_pred =
    rep_array(-1.0, N, T_max);

  array[N, T_max] real V_b_pred =
    rep_array(-1.0, N, T_max);

  vector[sum(Tsubj)] log_lik;

  int k = 0;

  for (n in 1:N) {

    params[n, 1] =
      Phi_approx(mu_pr[1] + sigma_pr[1] * param_raw[n, 1]) * 6; // alpha

    params[n, 2] =
      mu_pr[2] + sigma_pr[2] * param_raw[n, 2]; // beta

    params[n, 3] =
      Phi_approx(mu_pr[3] + sigma_pr[3] * param_raw[n, 3]); // lambda

    params[n, 4] =
      mu_pr[4] + sigma_pr[4] * param_raw[n, 4]; // eta

    params[n, 5] =
      Phi_approx(mu_pr[5] + sigma_pr[5] * param_raw[n, 5]) * 6; // alpha0

    params[n, 6] =
      mu_pr[6] + sigma_pr[6] * param_raw[n, 6]; // beta0

    params[n, 7] =
      Phi_approx(mu_pr[7] + sigma_pr[7] * param_raw[n, 7]); // delta

    params[n, 8] =
      Phi_approx(mu_pr[8] + sigma_pr[8] * param_raw[n, 8]); // rho

    real feedback_blue = 0.0;
    real feedback_red = 0.0;

    // Start with the full instruction weight before trial 1.
    real instruction_decay = 1.0;

    real V_b =
      compute_Vb(instr[n],
                 params[n, 8],
                 mu_Delta,
                 feedback_blue,
                 feedback_red,
                 instruction_decay);

    for (t in 1:Tsubj[n]) {

      k += 1;

      int sample_size =
        sample[n, t];

      array[I_max] int color_trial;
      array[I_max] real proba_trial;

      for (i in 1:I_max) {
        color_trial[i] =
          color[n, t, i];

        proba_trial[i] =
          proba[n, t, i];
      }

      // Keep the belief that was actually used for this trial.
      V_b_pred[n, t] = V_b;

      log_lik[k] =
        compute_log_lik(
          sample_size,
          color_trial,
          proba_trial,
          choice[n, t],
          params[n, 1],
          params[n, 2],
          params[n, 3],
          V_b,
          params[n, 4],
          params[n, 5],
          params[n, 6]
        );

      vector[2] evidence =
        compute_evidence(
          sample_size,
          color_trial,
          proba_trial,
          params[n, 1],
          params[n, 2],
          params[n, 3],
          V_b,
          params[n, 4],
          params[n, 5],
          params[n, 6]
        );

      vector[2] evidence_safe =
        clamp_vector(evidence, -100, 100);

      y_pred[n, t] =
        categorical_rng(softmax(evidence_safe));

      // Discount older feedback before adding the outcome from this trial.
      int x = feedback[n, t];

      feedback_blue =
        params[n, 7] * feedback_blue + x;

      feedback_red =
        params[n, 7] * feedback_red + (1 - x);

      // The instruction also fades at the same rate as feedback.
      instruction_decay *= params[n, 7];

      // Use the updated belief for the following trial.
      V_b =
        compute_Vb(
          instr[n],
          params[n, 8],
          mu_Delta,
          feedback_blue,
          feedback_red,
          instruction_decay
        );
    }
  }
}
