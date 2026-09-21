# ============================================================================

# PARAMETER RECOVERY FOR FULL LEARNING CHOICE MODEL


# This script runs parameter recovery for the full learning version of the
# sequential choice model.
#
# The aim is to test whether the model can recover known parameter values when
# choices are generated from the same model that is later fitted back to those
# simulated choices.
# The script follows the same general logic as the simpler recovery script:
#
# 1. Load the real experimental task structure.
# 2. Choose one row of a parameter grid using command-line index k.
# 3. Simulate subject-level parameters around the selected group-level values.
# 4. Generate simulated choices from those known parameters.
# 5. Fit the Stan model to the simulated choices.
# 6. Save the true simulated values, fitted posterior summaries, and diagnostics.
#
# The real data are used to preserve the actual task structure:
# - number of subjects
# - number of trials per subject
# - sample colours on each trial
# - stated reliability values on each trial
# - feedback / correct colour on each trial
#
# The important difference from the simple model is that this learning model
# carries a belief across trials. The model starts each subject with an unbiased
# belief about the base-rate colour, then updates that belief after each trial
# using the observed feedback.
#
# In the simple model, choices depend only on:
#
# - reliability distortion: alpha and beta
# - sample-position weights: w1 to w5, with the final sample fixed as reference
#
# In this learning model, choices depend on:
#
# - alpha: reliability distortion slope
# - beta: reliability distortion intercept / 50% evidence bias
# - lambda: exponential recency weighting within a trial
# - delta: feedback-based updating of the modelled base-rate belief
# - eta: belief-dependent modulation of incoming evidence
#
#
# Feedback coding:
#
# feedback = 1 means blue was objectively correct on that trial
# feedback = 0 means red was objectively correct on that trial
#
# This feedback variable is the correct-colour sequence used for belief updating.

# The parameter grid also differs from the simple recovery script. The simple
# recovery script used:
# 4 alpha values × 4 beta values × 4 w profiles = 64 runs
# This script is for the Implicit Unaware Base Rate group.
# The grid values are taken directly from the Implicit Unaware posterior summary:
#
#   alpha  = 3.66 [2.33, 4.96]
#   beta   = 0.645 [0.381, 0.914]
#   lambda = 0.0704 [0.0401, 0.103]
#   delta  = 0.765 [0.229, 1.30]
#   eta    = -0.245 [-0.346, -0.162]
#
# Each parameter is treated separately and varied independently across three
# values: the lower 95% interval bound, the posterior mean, and the upper 95%
# interval bound.
#
# This gives:
#   3 alpha × 3 beta × 3 lambda × 3 delta × 3 eta = 243 runs
#
# The learning model has five separate psychological parameters rather than
# alpha, beta, and a temporal-weight profile. Therefore, this script uses a
# full independent grid over the learning-model parameters instead of bundling
# lambda, delta, and eta into fixed profiles.
# ===========================================================================


rm(list = ls(all = TRUE))
setwd(getwd())
renv::load()

library(cmdstanr)
library(dplyr)
library(tibble)

args <- commandArgs(trailingOnly = TRUE)
k    <- as.numeric(args[1])

# ============================================================================
# 0. LOAD PRE-COMPILED MODEL
# ============================================================================
#
# CHANGED FROM SIMPLE SCRIPT:
# The simple script used:
#   ./stan/log_seq_basic.stan
#
# This script uses the full learning model, which includes:
#   alpha  = reliability distortion slope
#   beta   = reliability distortion intercept / 50% evidence bias
#   lambda = recency
#   delta  = feedback-based base-rate learning
#   eta    = belief-dependent evidence weighting

model <- cmdstan_model(
  stan_file     = './stan/log_trunc_simplified_boost_learning.stan',
  cpp_options   = list(stan_threads = TRUE),
  stanc_options = list("O1")
)

# ============================================================================
# 1. LOAD STIMULUS DATA FROM ACTUAL EXPERIMENT
# ============================================================================
#
# CHANGED FROM SIMPLE SCRIPT:
# The simple script used:
#   ./data/data_reliability.rdata
#
# This learning model uses the Implicit Unaware prior-belief data because it needs
# the feedback/correct-colour sequence to update the modelled base-rate belief.
#
# 
#
# The real data are used only to preserve the actual task structure:
#   subjects
#   trials
#   sample colours
#   sample reliabilities
#   feedback/correct colour on each trial

load(file = './data/data_priorbelief_unaware_exp11.rdata')

# CHANGED FROM SIMPLE SCRIPT:
# This now matches the colour, choice, sample-number, and feedback coding used
# in the learning-model fitting script.
#
# choice:
#   blue choice = 1
#   red choice  = 2
#
# color:
#   blue sample = 1
#   red sample  = 2
#
# feedback:
#   blue correct = 1
#   red correct  = 0
#
# Real participant choices are recoded here to keep this script close to the
# simple recovery script, but they are NOT used for fitting in recovery.
# The model is fitted to simulated choices later: choice_sim.

data <- data %>%
  arrange(ParticipantPrivateID, TrialNumber) %>%
  mutate(
    choice = case_when(
      ResponseButtonOrder == 1 & Response == 0 ~ 2,
      ResponseButtonOrder == 1 & Response == 1 ~ 1,
      ResponseButtonOrder == 0 & Response == 0 ~ 1,
      ResponseButtonOrder == 0 & Response == 1 ~ 2,
      TRUE ~ NA_real_
    )
  ) %>%
  mutate(
    across(
      starts_with("color"),
      ~ case_when(
        . == "blue" ~ 1,
        . == "red"  ~ 2,
        TRUE ~ NA_real_
      )
    )
  ) %>%
  rowwise() %>%
  mutate(
    sample_number = sum(!is.na(c_across(starts_with("proba_"))))
  ) %>%
  ungroup() %>%
  mutate(
    feedback = ifelse(CorrectResponse == 1, 1, 0)
  )

if (any(is.na(data$choice))) {
  stop("Some choices could not be recoded.")
}

if (any(is.na(data$feedback))) {
  stop("Some feedback values could not be recoded.")
}


subjs  <- unique(data$ParticipantPrivateID)
N      <- length(subjs)
T_max  <- max(data$TrialNumber)
I      <- max(data$sample_number)

d      <- data %>% group_by(ParticipantPrivateID) %>% summarise(t_subjs = n())
Tsubj  <- d$t_subjs

## Stimulus arrays
##
## CHANGED FROM SIMPLE SCRIPT:
## The simple script only needed:
##   color_arr
##   proba_arr
##
## The learning model also needs:
##   sample_arr   = number of samples on each trial
##   feedback_arr = correct colour / feedback on each trial

color_arr    <- array(1L,  c(N, T_max, I))
proba_arr    <- array(0.5, c(N, T_max, I))
sample_arr   <- array(I,   c(N, T_max))
feedback_arr <- array(0L,  c(N, T_max))

for (n in seq_len(N)) {
  t_n       <- Tsubj[n]
  data_subj <- data %>% filter(ParticipantPrivateID == subjs[n])
  
  for (j in seq_len(t_n)) {
    
    feedback_arr[n, j] <- data_subj$feedback[j]
    sample_arr[n, j]   <- data_subj$sample_number[j]
    
    for (i in seq_len(data_subj$sample_number[j])) {
      color_arr[n, j, i] <- data_subj[[paste0("color_", i)]][j]
      
      p_raw <- data_subj[[paste0("proba_", i)]][j]
      
      if (!is.na(p_raw) && p_raw > 1) {
        proba_arr[n, j, i] <- p_raw / 100
      } else {
        proba_arr[n, j, i] <- p_raw
      }
    }
  }
}

for (n in seq_len(N)) {
  for (j in seq_len(Tsubj[n])) {
    stopifnot(all(color_arr[n, j, seq_len(sample_arr[n, j])] %in% c(1L, 2L)))
    stopifnot(all(proba_arr[n, j, seq_len(sample_arr[n, j])] > 0 &
                    proba_arr[n, j, seq_len(sample_arr[n, j])] < 1))
  }
}
stopifnot(all(feedback_arr %in% c(0L, 1L)))

# ============================================================================
# 2. PARAMETER GRID 
# ============================================================================
#
# Simple script:
#   4 alpha values × 4 beta values × 4 w profiles = 64 runs
#
# CHANGED FROM SIMPLE SCRIPT:
# This recovery script is for the Implicit Unaware Base Rate group.
#
# The grid values are taken directly from the Implicit Unaware posterior summary:
#
#   alpha  = 3.66 [2.33, 4.96]
#   beta   = 0.645 [0.381, 0.914]
#   delta  = 0.765 [0.229, 1.30]
#   eta    = -0.245 [-0.346, -0.162]
#   lambda = 0.0704 [0.0401, 0.103]
#
# Each parameter is varied independently across:
#   lower 95% interval, posterior mean, upper 95% interval
#
# This gives:
#   3 alpha × 3 beta × 3 lambda × 3 delta × 3 eta = 243 runs
#
# This is not the same as the simple model's 64-run grid. The learning model has
# five separate parameters, so a full independent grid is necessarily larger.


alpha_range  <- c(2.33, 3.66, 4.96)
beta_range   <- c(0.381, 0.645, 0.914)
lambda_range <- c(0.0401, 0.0704, 0.103)
delta_range  <- c(0.229, 0.765, 1.30)
eta_range    <- c(-0.346, -0.245, -0.162)

parameters <- expand.grid(
  alpha_idx  = seq_along(alpha_range),
  beta_idx   = seq_along(beta_range),
  lambda_idx = seq_along(lambda_range),
  delta_idx  = seq_along(delta_range),
  eta_idx    = seq_along(eta_range)
)

stopifnot(k >= 1, k <= nrow(parameters))

mu_alpha  <- alpha_range[parameters$alpha_idx[k]]
mu_beta   <- beta_range[parameters$beta_idx[k]]
mu_lambda <- lambda_range[parameters$lambda_idx[k]]
mu_delta  <- delta_range[parameters$delta_idx[k]]
mu_eta    <- eta_range[parameters$eta_idx[k]]

# CHANGED FROM SIMPLE SCRIPT:
# The learning Stan model uses Phi_approx rather than exact Phi.
# These functions reproduce the Stan transform in R.
#
# Phi_approx(x) = inv_logit(0.07056*x^3 + 1.5976*x)

Phi_approx_R <- function(x) {
  plogis(0.07056 * x^3 + 1.5976 * x)
}

Phi_approx_inv_R <- function(p) {
  if (p <= 0 || p >= 1) {
    stop("Phi_approx inverse requires p strictly between 0 and 1.")
  }
  
  uniroot(
    f = function(x) Phi_approx_R(x) - p,
    interval = c(-20, 20)
  )$root
}

## Convert to raw space to match Stan parameterization
##
## CHANGED FROM SIMPLE SCRIPT:
## Simple model transforms:
##   alpha = 20 * Phi(...)
##   beta  = identity
##   w     = 2 * Phi(...)
##
## Learning model transforms:
##   alpha  = 6 * Phi_approx(...)
##   beta   = identity
##   lambda = Phi_approx(...)
##   delta  = 2 * Phi_approx(...)
##   eta    = identity

mu_pr <- c(
  Phi_approx_inv_R(mu_alpha / 6),     # alpha = 6 * Phi_approx(mu_pr[1])
  mu_beta,                            # beta  = mu_pr[2] identity
  Phi_approx_inv_R(mu_lambda),        # lambda = Phi_approx(mu_pr[3])
  Phi_approx_inv_R(mu_delta / 2),     # delta = 2 * Phi_approx(mu_pr[4])
  mu_eta                              # eta = mu_pr[5] identity
)

## Group-level SDs in raw space
##
## CHANGED FROM SIMPLE SCRIPT:
## The simple model had separate sigma_pr and sigma_w.
## The learning model has one vector for the five group-level parameters.

sigma_pr <- c(0.35, 0.25, 0.35, 0.35, 0.25)

## True group-level values for recovery comparison
group_sim   <- c(mu_alpha, mu_beta, mu_lambda, mu_delta, mu_eta)
param_names <- c("alpha", "beta", "lambda", "delta", "eta")

cat(sprintf("[k=%d] alpha=%.3f beta=%.3f lambda=%.4f delta=%.3f eta=%.3f\n",
            k, mu_alpha, mu_beta, mu_lambda, mu_delta, mu_eta))

# ============================================================================
# 3. SIMULATE RESPONSES 
# ============================================================================
#
# As in the simple script:
#   1. Draw individual parameters around the selected group-level values.
#   2. Generate simulated choices from those known parameters.
#   3. Fit the model to those simulated choices.
#
# CHANGED FROM SIMPLE SCRIPT:
# The simulated choice now depends not only on reliability and sample order,
# but also on:
#   current prior belief
#   feedback-based belief updating
#   recency via lambda
#   belief-dependent weighting via eta

safe_logit <- function(p) {
  p <- pmax(pmin(p, 1 - 1e-6), 1e-6)
  log(p / (1 - p))
}

softmax_R <- function(x) {
  x <- pmin(pmax(x, -100), 100)
  val <- exp(x - max(x))
  val / sum(val)
}

set.seed(42 + k)

## Draw individual parameters via Stan's non-centered parameterization
##
## CHANGED FROM SIMPLE SCRIPT:
## The simple script drew 7 individual parameters:
##   alpha, beta, w1, w2, w3, w4, w5
##
## The learning model draws 5 individual parameters:
##   alpha, beta, lambda, delta, eta

params_indiv <- matrix(NA_real_, nrow = N, ncol = 5,
                       dimnames = list(NULL, c("alpha", "beta",
                                               "lambda", "delta", "eta")))

for (n in seq_len(N)) {
  raw <- rnorm(5, 0, 1)
  
  params_indiv[n, 1] <- 6 * Phi_approx_R(mu_pr[1] + sigma_pr[1] * raw[1])  # alpha
  params_indiv[n, 2] <- mu_pr[2] + sigma_pr[2] * raw[2]                    # beta
  params_indiv[n, 3] <- Phi_approx_R(mu_pr[3] + sigma_pr[3] * raw[3])      # lambda
  params_indiv[n, 4] <- 2 * Phi_approx_R(mu_pr[4] + sigma_pr[4] * raw[4])  # delta
  params_indiv[n, 5] <- mu_pr[5] + sigma_pr[5] * raw[5]                    # eta
}

## Simulate choices
choice_sim <- array(1L, c(N, T_max))

for (n in seq_len(N)) {
  alpha_n  <- params_indiv[n, 1]
  beta_n   <- params_indiv[n, 2]
  lambda_n <- params_indiv[n, 3]
  delta_n  <- params_indiv[n, 4]
  eta_n    <- params_indiv[n, 5]
  
  # CHANGED FROM SIMPLE SCRIPT:
  # The learning model carries a belief across trials.
  #
  # It starts from equal counts:
  #   blue count = 1
  #   red count  = 1
  #
  # Therefore the starting belief is:
  #   V_b = 0.5
  #
  # V_b means the model's current belief that blue is the high-base-rate colour.
  
  beliefcount_blue <- 1.0
  beliefcount_red  <- 1.0
  V_b <- beliefcount_blue / (beliefcount_blue + beliefcount_red)
  
  for (j in seq_len(Tsubj[n])) {
    
    evidence <- c(0.0, 0.0)
    
    # CHANGED FROM SIMPLE SCRIPT:
    # Add the current prior belief before integrating the current trial samples.
    #
    # If V_b > 0.5, this adds evidence toward blue.
    # If V_b < 0.5, this adds evidence toward red because the blue log-odds is negative.
    
    V_b_clamped <- min(max(V_b, 0.001), 0.999)
    prior_log_odds <- log(V_b_clamped / (1 - V_b_clamped))
    evidence[1] <- evidence[1] + prior_log_odds
    
    for (s in seq_len(I)) {
      c_s <- color_arr[n, j, s]
      p_s <- proba_arr[n, j, s]
      
      if (c_s < 1 || c_s > 2 || p_s <= 0 || p_s >= 1) next
      
      # Same reliability transform idea as the simple script:
      #
      #   m_s = alpha * logit(reliability) + beta
      #
      # alpha controls sensitivity to reliability.
      # beta controls whether even 50% samples behave like evidence.
      
      m_s <- alpha_n * safe_logit(p_s) + beta_n
      
      # CHANGED FROM SIMPLE SCRIPT:
      # Belief-congruence term.
      #
      # a is positive if the sample favours the currently believed colour.
      # a is negative if the sample favours the less believed colour.
      #
      # eta controls how this changes sample weighting.
      #
      # If eta > 0:
      #   belief-congruent samples are upweighted.
      #
      # If eta < 0:
      #   belief-incongruent samples are upweighted.
      
      if (c_s == 1) {
        a <- 2 * V_b_clamped - 1
      } else {
        a <- 2 * (1 - V_b_clamped) - 1
      }
      
      current_kappa <- exp(eta_n * a)
      
      # CHANGED FROM SIMPLE SCRIPT:
      # Recency is now controlled by lambda rather than free sample weights.
      #
      # The final sample has weight:
      #   exp(lambda * (I - I)) = 1
      #
      # Earlier samples have lower weight when lambda > 0.
      
      recency_weight <- exp(lambda_n * (s - I))
      
      evidence[c_s] <- evidence[c_s] + recency_weight * current_kappa * m_s
    }
    
    ## softmax(evidence) — no temperature, matching Stan
    val <- softmax_R(evidence)
    choice_sim[n, j] <- sample(1:2, 1, prob = val)
    
    # CHANGED FROM SIMPLE SCRIPT:
    # Update belief after observing feedback/correct colour.
    #
    # x = 1 means blue was correct on this trial.
    # x = 0 means red was correct on this trial.
    #
    # This is the "correct colour sequence".
    # It is not the participant's real choice and not the simulated choice.
    
    x <- feedback_arr[n, j]
    
    beliefcount_blue <- delta_n * (beliefcount_blue - 1) + x + 1
    beliefcount_red  <- delta_n * (beliefcount_red  - 1) + (1 - x) + 1
    
    V_b <- beliefcount_blue / (beliefcount_blue + beliefcount_red)
  }
}

# ============================================================================
# 4. FIT
# ============================================================================
#


dir.create('./results/choice_learning_implicit_unaware', recursive = TRUE, showWarnings = FALSE)

out_file <- sprintf('./results/choice_learning_implicit_unaware/recover_%d.rds', k)
if (file.exists(out_file)) {
  cat(sprintf("[k=%d] Output already exists — skipping.\n", k))
  quit(save = "no", status = 0)
}

# CHANGED FROM SIMPLE SCRIPT:
# The learning Stan model expects:
#   I_max    instead of I
#   sample   = number of samples per trial
#   feedback = correct colour / feedback sequence
#   grainsize for within-chain threading

fit <- model$sample(
  data = list(
    N        = N,
    T_max    = T_max,
    I_max    = I,
    Tsubj    = Tsubj,
    sample   = sample_arr,
    color    = color_arr,
    proba    = proba_arr,
    choice   = choice_sim,
    feedback = feedback_arr,
    grainsize = 5
  ),
  iter_sampling     = 3000,
  iter_warmup       = 2000,
  chains            = 4,
  parallel_chains   = 4,
  threads_per_chain = max(1L, as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "4")) %/% 4),
  seed              = 12345 + k,
  adapt_delta       = 0.95,
  max_treedepth     = 12,
  refresh           = 500
)

# ============================================================================
# 5. SAVE
# ============================================================================
#
# CHANGED FROM SIMPLE SCRIPT:
# The saved group-level parameters are now:
#   mu_alpha
#   mu_beta
#   mu_lambda
#   mu_delta
#   mu_eta
#
# The saved individual-level parameters are extracted as:
#   params[n,1] = alpha
#   params[n,2] = beta
#   params[n,3] = lambda
#   params[n,4] = delta
#   params[n,5] = eta

## Group-level recovery
group_param_names <- c('mu_alpha', 'mu_beta', 'mu_lambda', 'mu_delta', 'mu_eta')
group_fitted <- fit$summary(variables = group_param_names)

## Individual-level recovery: params[n,1..5]
indiv_vars <- unlist(lapply(seq_along(param_names), function(p) {
  paste0("params[", seq_len(N), ",", p, "]")
}))

indiv_fitted <- fit$summary(variables = indiv_vars)

## Simulated individual values, matching order above
indiv_sim_vec <- c(
  params_indiv[, 1],   # alpha for all N
  params_indiv[, 2],   # beta for all N
  params_indiv[, 3],   # lambda for all N
  params_indiv[, 4],   # delta for all N
  params_indiv[, 5]    # eta for all N
)

## Diagnostics
diag_summary <- fit$diagnostic_summary(quiet = TRUE)

# CHANGED FROM SIMPLE SCRIPT:
# Save the selected grid row as well.
# This makes it easier later to check exactly which true parameter values
# were used for each recovery run.

parameter_grid_row <- parameters[k, ] %>%
  mutate(
    alpha  = mu_alpha,
    beta   = mu_beta,
    lambda = mu_lambda,
    delta  = mu_delta,
    eta    = mu_eta
  )

results <- list(
  k                  = k,
  parameter_grid_row = parameter_grid_row,
  group_sim          = group_sim,
  group_fitted       = group_fitted,
  indiv_sim          = indiv_sim_vec,
  indiv_fitted       = indiv_fitted,
  params_indiv_sim   = params_indiv,
  choice_sim         = choice_sim,
  param_names        = param_names,
  n_divergent        = sum(diag_summary$num_divergent),
  n_max_td           = sum(diag_summary$num_max_treedepth),
  max_rhat           = max(fit$summary()$rhat, na.rm = TRUE),
  min_ess            = min(fit$summary()$ess_bulk, na.rm = TRUE)
)

saveRDS(results, out_file)
cat(sprintf("[k=%d] Done. divergences=%d max_rhat=%.3f\n",
            k, results$n_divergent, results$max_rhat))