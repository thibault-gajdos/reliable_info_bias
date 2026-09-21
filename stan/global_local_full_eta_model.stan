functions {

  vector clamp_vector(vector x, real lo, real hi) {
    vector[num_elements(x)] out;

    for (i in 1:num_elements(x)) {
      out[i] = fmin(fmax(x[i], lo), hi);
    }

    return out;
  }


  vector combine_prior_with_global_eta(
      vector raw_sample_evidence,
      vector local_sample_evidence,
      real V_b,
      real eta_global) {

    // Determine the direction of the samples before either eta
    // mechanism changes their overall direction.
    real sample_log_odds =
      raw_sample_evidence[1] -
      raw_sample_evidence[2];

    real global_alignment;
    real global_kappa;
    real prior_log_odds;

    vector[2] evidence;


    if (sample_log_odds > 0) {

      // The original combined samples favour Blue.
      global_alignment =
        2 * V_b - 1;

    } else if (sample_log_odds < 0) {

      // The original combined samples favour Red.
      global_alignment =
        2 * (1 - V_b) - 1;

    } else {

      // Global eta has no effect when the original
      // combined samples are exactly balanced.
      global_alignment = 0;
    }


    global_kappa =
      exp(
        eta_global *
        global_alignment
      );


    // Apply global eta to the sample evidence after
    // local eta has modified the individual samples.
    evidence =
      local_sample_evidence *
      global_kappa;


    // Add the learned prior afterwards so that neither
    // local nor global eta multiplies the prior.
    prior_log_odds =
      log(
        V_b /
        (1 - V_b)
      );

    evidence[1] +=
      prior_log_odds;


    return evidence;
  }


  real partial_sum(
      array[] int slice_indices,
      int start,
      int end,
      vector mu_pr,
      vector sigma_pr,
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

      int n =
        slice_indices[i];

      vector[6] params;


      params[1] =
        Phi_approx(
          mu_pr[1] +
          sigma_pr[1] *
          param_raw[n, 1]
        ) * 6;
        // alpha


      params[2] =
        mu_pr[2] +
        sigma_pr[2] *
        param_raw[n, 2];
        // beta


      params[3] =
        Phi_approx(
          mu_pr[3] +
          sigma_pr[3] *
          param_raw[n, 3]
        );
        // lambda


      params[4] =
        Phi_approx(
          mu_pr[4] +
          sigma_pr[4] *
          param_raw[n, 4]
        ) * 2;
        // delta


      params[5] =
        mu_pr[5] +
        sigma_pr[5] *
        param_raw[n, 5];
        // eta_local


      params[6] =
        mu_pr[6] +
        sigma_pr[6] *
        param_raw[n, 6];
        // eta_global


      real beliefcount_blue = 1.0;

      real beliefcount_red = 1.0;

      real V_b =
        beliefcount_blue /
        (
          beliefcount_blue +
          beliefcount_red
        );


      for (t in 1:Tsubj[n]) {

        // This vector contains the samples before either
        // eta mechanism is applied. It is used to classify
        // the original global direction of the evidence.
        vector[2] raw_sample_evidence =
          rep_vector(0.0, 2);


        // This vector contains the samples after local eta
        // has been applied separately to every sample.
        vector[2] local_sample_evidence =
          rep_vector(0.0, 2);


        int sample_size =
          sample[n, t];


        real V_b_clamped =
          fmin(
            fmax(
              V_b,
              V_b_min
            ),
            V_b_max
          );


        for (s in 1:sample_size) {

          real p =
            proba[n, t, s];

          real l =
            logit(p);

          int color_val =
            color[n, t, s];


          real log_odds =
            params[1] * l +
            params[2];


          // Apply reliability transformation and recency.
          real sample_contribution =
            exp(
              params[3] *
              (s - sample_size)
            ) *
            log_odds;


          // Determine whether this individual sample
          // agrees with the learned prior.
          real local_alignment;

          if (color_val == 1) {

            // Individual sample is Blue.
            local_alignment =
              2 * V_b_clamped - 1;

          } else {

            // Individual sample is Red.
            local_alignment =
              2 * (1 - V_b_clamped) - 1;
          }


          // Apply eta_local to this individual sample.
          real local_kappa =
            exp(
              params[5] *
              local_alignment
            );


          // Keep an unchanged copy for classification
          // of the original global sample direction.
          raw_sample_evidence[color_val] +=
            sample_contribution;


          // Separately accumulate the locally weighted samples.
          local_sample_evidence[color_val] +=
            sample_contribution *
            local_kappa;
        }


        // The function:
        //
        // 1. Determines global alignment using raw_sample_evidence.
        // 2. Applies eta_global to local_sample_evidence.
        // 3. Adds the learned prior afterwards.
        vector[2] evidence =
          combine_prior_with_global_eta(
            raw_sample_evidence,
            local_sample_evidence,
            V_b_clamped,
            params[6]
          );


        vector[2] evidence_safe =
          clamp_vector(
            evidence,
            -100,
            100
          );


        lp +=
          categorical_lpmf(
            choice[n, t] |
            softmax(evidence_safe)
          );


        // Update the learned prior after the current
        // choice has been evaluated.
        int x =
          feedback[n, t];


        beliefcount_blue =
          params[4] *
          (beliefcount_blue - 1) +
          x +
          1;


        beliefcount_red =
          params[4] *
          (beliefcount_red - 1) +
          (1 - x) +
          1;


        V_b =
          beliefcount_blue /
          (
            beliefcount_blue +
            beliefcount_red
          );
      }
    }


    return lp;
  }


  vector compute_evidence(
      int sample_size,
      array[] int color_data,
      array[] real proba_data,
      real alpha,
      real beta,
      real lambda,
      real V_b,
      real eta_local,
      real eta_global) {

    vector[2] raw_sample_evidence =
      rep_vector(0.0, 2);

    vector[2] local_sample_evidence =
      rep_vector(0.0, 2);


    real V_b_clamped =
      fmin(
        fmax(
          V_b,
          0.001
        ),
        0.999
      );


    for (s in 1:sample_size) {

      real l =
        logit(
          proba_data[s]
        );

      int color_val =
        color_data[s];


      real log_odds =
        alpha * l +
        beta;


      real sample_contribution =
        exp(
          lambda *
          (s - sample_size)
        ) *
        log_odds;


      real local_alignment;

      if (color_val == 1) {

        local_alignment =
          2 * V_b_clamped - 1;

      } else {

        local_alignment =
          2 * (1 - V_b_clamped) - 1;
      }


      real local_kappa =
        exp(
          eta_local *
          local_alignment
        );


      raw_sample_evidence[color_val] +=
        sample_contribution;


      local_sample_evidence[color_val] +=
        sample_contribution *
        local_kappa;
    }


    return combine_prior_with_global_eta(
      raw_sample_evidence,
      local_sample_evidence,
      V_b_clamped,
      eta_global
    );
  }


  real compute_log_lik(
      int sample_size,
      array[] int color_data,
      array[] real proba_data,
      int choice,
      real alpha,
      real beta,
      real lambda,
      real V_b,
      real eta_local,
      real eta_global) {

    vector[2] evidence =
      compute_evidence(
        sample_size,
        color_data,
        proba_data,
        alpha,
        beta,
        lambda,
        V_b,
        eta_local,
        eta_global
      );


    vector[2] evidence_safe =
      clamp_vector(
        evidence,
        -100,
        100
      );


    return categorical_lpmf(
      choice |
      softmax(evidence_safe)
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

  int<lower=5> grainsize;
}


parameters {

  // Parameter order:
  // 1 = alpha
  // 2 = beta
  // 3 = lambda
  // 4 = delta
  // 5 = eta_local
  // 6 = eta_global

  vector[6] mu_pr;

  vector<lower=0>[6] sigma_pr;

  matrix[N, 6] param_raw;
}


model {

  mu_pr ~
    std_normal();

  sigma_pr ~
    normal(0, 1);

  to_vector(param_raw) ~
    std_normal();


  array[N] int indices;


  for (n in 1:N) {
    indices[n] = n;
  }


  target +=
    reduce_sum(
      partial_sum,
      indices,
      grainsize,
      mu_pr,
      sigma_pr,
      Tsubj,
      sample,
      color,
      proba,
      choice,
      param_raw,
      feedback
    );
}


generated quantities {

  real mu_alpha =
    Phi_approx(
      mu_pr[1]
    ) * 6;


  real mu_beta =
    mu_pr[2];


  real mu_lambda =
    Phi_approx(
      mu_pr[3]
    );


  real mu_delta =
    Phi_approx(
      mu_pr[4]
    ) * 2;


  real mu_eta_local =
    mu_pr[5];


  real mu_eta_global =
    mu_pr[6];


  matrix[N, 6] params;


  array[N, T_max] real y_pred =
    rep_array(
      -1.0,
      N,
      T_max
    );


  vector[sum(Tsubj)] log_lik;


  int k = 0;


  for (n in 1:N) {

    params[n, 1] =
      Phi_approx(
        mu_pr[1] +
        sigma_pr[1] *
        param_raw[n, 1]
      ) * 6;
      // alpha


    params[n, 2] =
      mu_pr[2] +
      sigma_pr[2] *
      param_raw[n, 2];
      // beta


    params[n, 3] =
      Phi_approx(
        mu_pr[3] +
        sigma_pr[3] *
        param_raw[n, 3]
      );
      // lambda


    params[n, 4] =
      Phi_approx(
        mu_pr[4] +
        sigma_pr[4] *
        param_raw[n, 4]
      ) * 2;
      // delta


    params[n, 5] =
      mu_pr[5] +
      sigma_pr[5] *
      param_raw[n, 5];
      // eta_local


    params[n, 6] =
      mu_pr[6] +
      sigma_pr[6] *
      param_raw[n, 6];
      // eta_global


    real beliefcount_blue = 1.0;

    real beliefcount_red = 1.0;

    real V_b =
      beliefcount_blue /
      (
        beliefcount_blue +
        beliefcount_red
      );


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
          params[n, 5],
          params[n, 6]
        );


      vector[2] evidence_safe =
        clamp_vector(
          evidence,
          -100,
          100
        );


      y_pred[n, t] =
        categorical_rng(
          softmax(evidence_safe)
        );


      int x =
        feedback[n, t];


      beliefcount_blue =
        params[n, 4] *
        (beliefcount_blue - 1) +
        x +
        1;


      beliefcount_red =
        params[n, 4] *
        (beliefcount_red - 1) +
        (1 - x) +
        1;


      V_b =
        beliefcount_blue /
        (
          beliefcount_blue +
          beliefcount_red
        );
    }
  }
}