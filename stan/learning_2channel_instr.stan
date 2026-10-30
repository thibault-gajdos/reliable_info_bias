// Two-channel learning model with instruction-based prior.
//
// Modifications:
//   - delta is estimated separately for each participant in (0, 2).
//   - rho is FIXED at 0.90 (not estimated).
//   - Delta (the undirected exclusion band) is REMOVED.
//   - instr = 3 (undirected) and instr = 4 (no instruction) both begin
//     with the same uniform Beta(1,1) prior.
//
// Beliefs about the frequency p of Blue:
//   instr = 1: rho * Uniform(0.5,1) + (1-rho) * Uniform(0,1).
//   instr = 2: rho * Uniform(0,0.5) + (1-rho) * Uniform(0,1).
//   instr = 3 or 4: Uniform(0,1), independent of rho.
//
// With rho=0.9, initial V_B = 0.725 (instr 1), 0.275 (instr 2),
// and 0.500 (instr 3/4). Unlike rho=1, both halves of the prior
// have positive support, so contradictory feedback can reverse the
// learned base-rate belief.
//
// Updating rule: f_(t+1)(p) proportional to Pr(F_t|p) * f_t(p)^delta.
// Discounted counts: N_B <- delta*N_B+F_t; N_R <- delta*N_R+(1-F_t).
// Instruction contribution is weighted by pw = delta^(t-1).
// Therefore f_t(p) is proportional to Beta(1+N_B,1+N_R)
// multiplied by h_1(p)^pw, where h_1 is the instruction density.
//
// Decision prior channel: P = alpha0 * logit(V_B) + beta0.
// Sample evidence: (alpha * logit(pi) + beta) * exp(lambda*(s-S)).
//
// Six estimated subject parameters: alpha, beta, lambda, alpha0,
// beta0, delta. 
//
// CAUTION: delta > 1 amplifies past feedback and instruction terms;
// counts can become extremely large over 240 decisions and cause
// instability in Beta CDF calculations. Inspect diagnostics.

functions {
  vector clamp_vector(vector x, real lo, real hi) {
    vector[num_elements(x)] out;
    for (i in 1:num_elements(x)) {
      out[i] = fmin(fmax(x[i], lo), hi);
    }
    return out;
  }

  // Posterior belief E[p] under the original mixture instruction
  // prior with rho fixed at 0.90. All calculations are on the log
  // scale to avoid overflow in pow(h, delta^(t-1)) when delta > 1.
  // For instr=1, density is 1-rho below 0.5 and 1+rho above 0.5.
  // For instr=2, the densities are reversed.
  real belief_V(real NB, real NR, int instr, real pw) {
    real a = 1 + NB;
    real b = 1 + NR;
    real m = a / (a + b);

    // No directional information: simple Beta posterior mean.
    if (instr == 3 || instr == 4) return m;

    // Fixed instruction trust. Do not estimate rho.
    real rho = 0.90;
    real log_h_lo = (instr == 1) ? pw * log1m(rho) : pw * log1p(rho);
    real log_h_hi = (instr == 1) ? pw * log1p(rho) : pw * log1m(rho);

    // Normalisation: probability mass below and above 0.5 under
    // Beta(a,b), reweighted by the instruction density.
    real log_den = log_sum_exp(
      log_h_lo + beta_lcdf(0.5 | a, b),
      log_h_hi + beta_lccdf(0.5 | a, b)
    );

    // First moment: m times a corresponding Beta(a+1,b) mixture.
    real log_num = log_sum_exp(
      log_h_lo + beta_lcdf(0.5 | a + 1, b),
      log_h_hi + beta_lccdf(0.5 | a + 1, b)
    );

    return m * exp(log_num - log_den);
  }

  vector compute_evidence(int sample_size, array[] int color_data,
                          array[] real proba_data,
                          real alpha, real beta, real lambda, real V_b,
                          real alpha0, real beta0) {
    vector[2] evidence = rep_vector(0.0, 2);
    real V_b_clamped = fmin(fmax(V_b, 0.001), 0.999);
    real P = alpha0 * logit(V_b_clamped) + beta0;
    evidence[1] += P;

    for (s in 1:sample_size) {
      real l = logit(proba_data[s]);
      int color_val = color_data[s];
      real log_odds = alpha * l + beta;
      evidence[color_val] += exp(lambda * (s - sample_size)) * log_odds;
    }
    return evidence;
  }

  // Six estimated subject parameters, plus fixed rho=0.9 in position 7.
  vector subject_params(vector mu_pr, vector sigma_pr, row_vector raw) {
    vector[7] p;
    p[1] = Phi_approx(mu_pr[1] + sigma_pr[1] * raw[1]) * 6; // alpha
    p[2] = mu_pr[2] + sigma_pr[2] * raw[2];                 // beta
    p[3] = Phi_approx(mu_pr[3] + sigma_pr[3] * raw[3]);     // lambda
    p[4] = Phi_approx(mu_pr[4] + sigma_pr[4] * raw[4]) * 6; // alpha0
    p[5] = mu_pr[5] + sigma_pr[5] * raw[5];                 // beta0
    p[6] = 2 * Phi_approx(mu_pr[6] + sigma_pr[6] * raw[6]); // delta (0,2)
    p[7] = 0.90;                                            // rho FIXED
    return p;
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
                   array[,] int feedback,
                   array[] int instr) {
    real lp = 0;

    for (i in 1:size(slice_indices)) {
      int n = slice_indices[i];
      vector[7] params = subject_params(mu_pr, sigma_pr, param_raw[n]);

      real NB = 0;
      real NR = 0;
      real pw = 1;  // delta^(t-1)

      for (t in 1:Tsubj[n]) {
        real V_b = belief_V(NB, NR, instr[n], pw);
        vector[2] evidence = compute_evidence(sample[n, t], color[n, t],
                                              proba[n, t], params[1],
                                              params[2], params[3], V_b,
                                              params[4], params[5]);
        lp += categorical_lpmf(choice[n, t] |
                               softmax(clamp_vector(evidence, -100, 100)));

        int x = feedback[n, t];
        NB = params[6] * NB + x;
        NR = params[6] * NR + (1 - x);
        pw *= params[6];
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
  array[N] int<lower=1, upper=4> instr;  // 1=B, 2=R, 3=U, 4=N
  int<lower=5> grainsize;
}

parameters {
  vector[6] mu_pr;
  vector<lower=0>[6] sigma_pr;
  matrix[N, 6] param_raw;
}

model {
  mu_pr[1:3] ~ std_normal();
  mu_pr[4] ~ std_normal();     // alpha0
  mu_pr[5] ~ normal(0, 0.5);   // beta0
  mu_pr[6] ~ std_normal();     // delta
  sigma_pr ~ normal(0, 1);
  to_vector(param_raw) ~ std_normal();

  array[N] int indices;
  for (n in 1:N) indices[n] = n;

  target += reduce_sum(partial_sum, indices, grainsize,
                       mu_pr, sigma_pr, Tsubj, sample, color,
                       proba, choice, param_raw, feedback, instr);
}

generated quantities {
  real mu_alpha  = Phi_approx(mu_pr[1]) * 6;
  real mu_beta   = mu_pr[2];
  real mu_lambda = Phi_approx(mu_pr[3]);
  real mu_alpha0 = Phi_approx(mu_pr[4]) * 6;
  real mu_beta0  = mu_pr[5];
  real mu_delta  = 2 * Phi_approx(mu_pr[6]);
  real mu_rho    = 0.90;  // fixed; included for output compatibility

  matrix[N, 7] params;    // seventh column contains fixed rho=0.90
  array[N, T_max] real V_B = rep_array(-1.0, N, T_max);
  array[N, T_max] real y_pred = rep_array(-1.0, N, T_max);
  vector[sum(Tsubj)] log_lik;

  {
    int k = 0;
    for (n in 1:N) {
      params[n] = subject_params(mu_pr, sigma_pr, param_raw[n])';

      real NB = 0;
      real NR = 0;
      real pw = 1;  // delta^(t-1)

      for (t in 1:Tsubj[n]) {
        k += 1;
        real V_b = belief_V(NB, NR, instr[n], pw);
        V_B[n, t] = V_b;
        vector[2] evidence = compute_evidence(sample[n, t], color[n, t],
                                              proba[n, t], params[n, 1],
                                              params[n, 2], params[n, 3], V_b,
                                              params[n, 4], params[n, 5]);
        vector[2] prob = softmax(clamp_vector(evidence, -100, 100));
        log_lik[k] = categorical_lpmf(choice[n, t] | prob);
        y_pred[n, t] = categorical_rng(prob);

        int x = feedback[n, t];
        NB = params[n, 6] * NB + x;
        NR = params[n, 6] * NR + (1 - x);
        pw *= params[n, 6];
      }
    }
  }
}