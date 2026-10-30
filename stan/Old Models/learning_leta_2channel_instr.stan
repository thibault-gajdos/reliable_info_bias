// Two-channel learning model with instruction-based prior
// (reliability.tex, Section "Instructed base rates and a two-channel model").
//
// Belief about the frequency p of Blue:
//   default prior Beta(1,1); an instruction is an event E_k on p;
//   initial belief f_1 = rho * Beta(1,1) | E_k + (1 - rho) * Beta(1,1),
//   with rho the trust in the instruction.
//   Conditions (data `instr`): 1 = B ("Blue more probable", E = (1/2,1)),
//                              2 = R ("Red more probable",  E = (0,1/2)),
//                              3 = U ("one colour more probable",
//                                     E = (0,1/2-Delta] U [1/2+Delta,1)),
//                              4 = N (no instruction).
//   Update: f_{t+1}(p) ‚àù Pr(F_t | p) f_t(p)^delta, which gives
//   f_{t+1} ‚àù h_1^(delta^t) * Beta(1 + N_B, 1 + N_R), with discounted feedback
//   counts N_B = delta * N_B + F_t, N_R = delta * N_R + (1 - F_t).
//   V_B = E[p] is computed in closed form with inc_beta (Beta CDF).
//
// Decision: P = alpha0 * logit(V_B) + beta0; A = tanh(KAPPA/2 * P);
//   each sample: (alpha * logit(pi) + beta) * exp(lambda (s - S)) * exp(eta * sigma * A).
//
// Parameters (subject level, non-centred):
//   1 alpha (0,6), 2 beta, 3 lambda (0,1), 4 eta, 5 alpha0 (0,6), 6 beta0,
//   7 delta (0,1), 8 rho (0,1).
// Group level: Delta in (0, 1/2), used by condition U only.

functions {
  real KAPPA() { return 10.0; }

  vector clamp_vector(vector x, real lo, real hi) {
    vector[num_elements(x)] out;
    for (i in 1:num_elements(x)) {
      out[i] = fmin(fmax(x[i], lo), hi);
    }
    return out;
  }

  // Subjective probability of Blue given the discounted counts,
  // the instruction condition, the trust rho, the band Delta and pw = delta^(t-1).
  // G0(x) = G_{a,b}(x) and G1(x) = G_{a+1,b}(x) = G0(x) - x^a (1-x)^b / (a B(a,b)).
  real belief_V(real NB, real NR, int instr, real rho, real Delta, real pw) {
    real a = 1 + NB;
    real b = 1 + NR;
    real m = a / (a + b);
    if (instr == 4) return m;

    // once the instruction has been forgotten (h = 1 up to 1e-8 on the log scale), V = m
    real lr = (instr == 3) ? abs(log1m(rho) - log(1 - rho + rho / (1 - 2 * Delta)))
                           : log1p(rho) - log1m(rho);
    if (pw * lr < 1e-8) return m;

    real lcst = -log(a) - lbeta(a, b);

    if (instr == 1 || instr == 2) {
      // values of h on (0,1/2) and (1/2,1)
      real h_lo = pow(1 - rho, pw);
      real h_hi = pow(1 + rho, pw);
      if (instr == 2) {
        real tmp = h_lo;
        h_lo = h_hi;
        h_hi = tmp;
      }
      real G0 = inc_beta(a, b, 0.5);
      real G1 = G0 - exp((a + b) * log(0.5) + lcst);
      return m * (h_lo * G1 + h_hi * (1 - G1)) / (h_lo * G0 + h_hi * (1 - G0));
    }

    // instr == 3 (U): intervals (0,x1), (x1,x2), (x2,1), with x1 = 1/2 - Delta, x2 = 1/2 + Delta
    {
      real x1 = 0.5 - Delta;
      real x2 = 0.5 + Delta;
      real h_out = pow(1 - rho + rho / (1 - 2 * Delta), pw);
      real h_in  = pow(1 - rho, pw);
      real G0_1 = inc_beta(a, b, x1);
      real G0_2 = inc_beta(a, b, x2);
      real G1_1 = G0_1 - exp(a * log(x1) + b * log1m(x1) + lcst);
      real G1_2 = G0_2 - exp(a * log(x2) + b * log1m(x2) + lcst);
      real num = h_out * G1_1 + h_in * (G1_2 - G1_1) + h_out * (1 - G1_2);
      real den = h_out * G0_1 + h_in * (G0_2 - G0_1) + h_out * (1 - G0_2);
      return m * num / den;
    }
  }

  vector compute_evidence(int sample_size, array[] int color_data, array[] real proba_data,
                          real alpha, real beta, real lambda, real V_b, real eta,
                          real alpha0, real beta0) {
    vector[2] evidence = rep_vector(0.0, 2);
    real V_b_clamped = fmin(fmax(V_b, 0.001), 0.999);
    real P = alpha0 * logit(V_b_clamped) + beta0;
    evidence[1] += P;

    real A = tanh(0.5 * KAPPA() * P);

    for (s in 1:sample_size) {
      real l = logit(proba_data[s]);
      int color_val = color_data[s];
      real log_odds = alpha * l + beta;
      real a = (color_val == 1) ? A : -A;
      real gain = exp(eta * a);
      evidence[color_val] += exp(lambda * (s - sample_size)) * log_odds * gain;
    }
    return evidence;
  }

  vector subject_params(vector mu_pr, vector sigma_pr, row_vector raw) {
    vector[8] p;
    p[1] = Phi_approx(mu_pr[1] + sigma_pr[1] * raw[1]) * 6;   // alpha
    p[2] = mu_pr[2] + sigma_pr[2] * raw[2];                   // beta
    p[3] = Phi_approx(mu_pr[3] + sigma_pr[3] * raw[3]);       // lambda
    p[4] = mu_pr[4] + sigma_pr[4] * raw[4];                   // eta
    p[5] = Phi_approx(mu_pr[5] + sigma_pr[5] * raw[5]) * 6;   // alpha0
    p[6] = mu_pr[6] + sigma_pr[6] * raw[6];                   // beta0
    p[7] = Phi_approx(mu_pr[7] + sigma_pr[7] * raw[7]);       // delta
    p[8] = fmin(Phi_approx(mu_pr[8] + sigma_pr[8] * raw[8]), 1 - 1e-9); // rho
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
                   array[] int instr,
                   real Delta) {
    real lp = 0;

    for (i in 1:size(slice_indices)) {
      int n = slice_indices[i];
      vector[8] params = subject_params(mu_pr, sigma_pr, param_raw[n]);

      real NB = 0;
      real NR = 0;
      real pw = 1;   // delta^(t-1)

      for (t in 1:Tsubj[n]) {
        real V_b = belief_V(NB, NR, instr[n], params[8], Delta, pw);
        vector[2] evidence = compute_evidence(sample[n, t], color[n, t], proba[n, t],
                                              params[1], params[2], params[3], V_b,
                                              params[4], params[5], params[6]);
        lp += categorical_lpmf(choice[n, t] | softmax(clamp_vector(evidence, -100, 100)));

        int x = feedback[n, t];
        NB = params[7] * NB + x;
        NR = params[7] * NR + (1 - x);
        pw *= params[7];
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
  array[N] int<lower=1, upper=4> instr;   // 1 = B, 2 = R, 3 = U, 4 = N
  int<lower=5> grainsize;
}

parameters {
  vector[8] mu_pr;
  vector<lower=0>[8] sigma_pr;
  matrix[N, 8] param_raw;
  // FIXED-DELTA: original estimated parameter
  // real Delta_raw;
}

transformed parameters {
  // FIXED-DELTA DIAGNOSTIC: original transformation 
  // real<lower=0, upper=0.5> Delta = 0.5 * fmin(fmax(Phi_approx(Delta_raw), 1e-6), 1 - 1e-6);
  real<lower=0, upper=0.5> Delta = 0.10;
}

model {
  mu_pr[1:3] ~ std_normal();
  mu_pr[4] ~ normal(0, 0.5);   // eta
  mu_pr[5] ~ std_normal();     // alpha0 
  mu_pr[6] ~ normal(0, 0.5);   // beta0
  mu_pr[7] ~ std_normal();     // delta
  mu_pr[8] ~ std_normal();     // rho
  sigma_pr ~ normal(0, 1);
  to_vector(param_raw) ~ std_normal();
  // FIXED-DELTA DIAGNOSTIC: original prior 
  // Delta_raw ~ std_normal();

  array[N] int indices;
  for (n in 1:N) indices[n] = n;

  target += reduce_sum(partial_sum, indices, grainsize,
                       mu_pr, sigma_pr,
                       Tsubj, sample, color, proba, choice, param_raw, feedback,
                       instr, Delta);
}

generated quantities {
  real mu_alpha  = Phi_approx(mu_pr[1]) * 6;
  real mu_beta   = mu_pr[2];
  real mu_lambda = Phi_approx(mu_pr[3]);
  real mu_eta    = mu_pr[4];
  real mu_alpha0 = Phi_approx(mu_pr[5]) * 6;
  real mu_beta0  = mu_pr[6];
  real mu_delta  = Phi_approx(mu_pr[7]);
  real mu_rho    = Phi_approx(mu_pr[8]);

  matrix[N, 8] params;
  array[N, T_max] real V_B = rep_array(-1.0, N, T_max);
  array[N, T_max] real y_pred = rep_array(-1.0, N, T_max);
  vector[sum(Tsubj)] log_lik;

  {
    int k = 0;
    for (n in 1:N) {
      params[n] = subject_params(mu_pr, sigma_pr, param_raw[n])';

      real NB = 0;
      real NR = 0;
      real pw = 1;

      for (t in 1:Tsubj[n]) {
        k += 1;
        real V_b = belief_V(NB, NR, instr[n], params[n, 8], Delta, pw);
        V_B[n, t] = V_b;
        vector[2] evidence = compute_evidence(sample[n, t], color[n, t], proba[n, t],
                                              params[n, 1], params[n, 2], params[n, 3], V_b,
                                              params[n, 4], params[n, 5], params[n, 6]);
        vector[2] prob = softmax(clamp_vector(evidence, -100, 100));
        log_lik[k] = categorical_lpmf(choice[n, t] | prob);
        y_pred[n, t] = categorical_rng(prob);

        int x = feedback[n, t];
        NB = params[n, 7] * NB + x;
        NR = params[n, 7] * NR + (1 - x);
        pw *= params[n, 7];
      }
    }
  }
}

