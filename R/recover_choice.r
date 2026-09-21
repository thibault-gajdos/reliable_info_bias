rm(list = ls(all = TRUE))

# ============================================================================
# PARAMETER RECOVERY FOR FULL CHOICE MODEL
# alpha, beta, lambda, delta, eta
# ============================================================================

setwd("~/reliable_info/param_recovery_choice")

if (requireNamespace("renv", quietly = TRUE)) {
  renv::load()
}

library(cmdstanr)
library(dplyr)
library(tibble)
library(stringr)

# ============================================================================
# 0. ARRAY INDEX
# ============================================================================

args <- commandArgs(trailingOnly = TRUE)

k <- if (length(args) >= 1) {
  as.integer(args[1])
} else {
  as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID", "1"))
}

stopifnot(!is.na(k), k >= 1)

# ============================================================================
# 1. COMPILE MODEL
# ============================================================================

model <- cmdstan_model(
  stan_file     = "./stan/log_seq_full.stan",   # EDIT if your Stan file has another name
  cpp_options   = list(stan_threads = TRUE),
  stanc_options = list("O1")
)

# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

# Stan's Phi_approx is approximately inv_logit(0.07056*x^3 + 1.5976*x)
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

safe_logit <- function(p) {
  p <- pmax(pmin(p, 1 - 1e-6), 1e-6)
  log(p / (1 - p))
}

softmax_R <- function(x) {
  x <- pmin(pmax(x, -100), 100)
  ex <- exp(x - max(x))
  ex / sum(ex)
}

recode_colour_1_2 <- function(x) {
  if (is.character(x) || is.factor(x)) {
    x <- tolower(as.character(x))
    out <- ifelse(x == "blue", 1L,
                  ifelse(x == "red", 2L, NA_integer_))
    return(out)
  }
  
  x <- as.integer(x)
  
  # Assume already coded 1 = blue, 2 = red
  if (all(na.omit(x) %in% c(1L, 2L))) {
    return(x)
  }
  
  stop("Colour values must be blue/red or 1/2.")
}

recode_feedback_1_0 <- function(x) {
  if (is.character(x) || is.factor(x)) {
    x <- tolower(as.character(x))
    out <- ifelse(x == "blue", 1L,
                  ifelse(x == "red", 0L, NA_integer_))
    return(out)
  }
  
  x <- as.integer(x)
  
  # Assume already coded 1 = blue, 0 = red
  if (all(na.omit(x) %in% c(0L, 1L))) {
    return(x)
  }
  
  # If coded 1 = blue, 2 = red
  if (all(na.omit(x) %in% c(1L, 2L))) {
    return(ifelse(x == 1L, 1L, 0L))
  }
  
  stop("Feedback must be blue/red, 1/0, or 1/2.")
}

# ============================================================================
# 3. LOAD ACTUAL STIMULUS DATA
# ============================================================================

load(file = "./data/data_reliability.rdata")

# Assumes loaded object is called data
stopifnot(exists("data"))

data <- data %>%
  arrange(ParticipantPrivateID, TrialNumber)

subjs <- unique(data$ParticipantPrivateID)
N <- length(subjs)

Tsubj <- data %>%
  group_by(ParticipantPrivateID) %>%
  summarise(t_subjs = n(), .groups = "drop") %>%
  pull(t_subjs)

T_max <- max(Tsubj)

color_cols <- grep("^color_", names(data), value = TRUE)
proba_cols <- grep("^proba_", names(data), value = TRUE)

stopifnot(length(color_cols) > 0)
stopifnot(length(color_cols) == length(proba_cols))

I_max <- length(color_cols)

# ---------------------------------------------------------------------------
# Feedback column
# ---------------------------------------------------------------------------
# Your Stan model needs feedback[n,t] coded as:
#   1 = blue correct
#   0 = red correct
#
# If autodetection fails, manually set this, e.g.
# feedback_col <- "CorrectColour"

feedback_col <- Sys.getenv("FEEDBACK_COL", unset = "")

if (feedback_col == "") {
  candidate_feedback_cols <- c(
    "feedback", "Feedback",
    "correct_colour", "CorrectColour",
    "correct_color", "CorrectColor",
    "page_colour", "PageColour",
    "page_color", "PageColor",
    "true_colour", "TrueColour",
    "true_color", "TrueColor",
    "CorrectAnswer", "correct_answer"
  )
  
  candidate_feedback_cols <- candidate_feedback_cols[
    candidate_feedback_cols %in% names(data)
  ]
  
  if (length(candidate_feedback_cols) == 0) {
    stop(
      "Could not find feedback column. Set feedback_col manually or set FEEDBACK_COL environment variable. ",
      "It must code the true/correct page colour as blue/red, 1/0, or 1/2."
    )
  }
  
  feedback_col <- candidate_feedback_cols[1]
}

cat("Using feedback column:", feedback_col, "\n")

# ============================================================================
# 4. BUILD STAN ARRAYS FROM ACTUAL TASK STRUCTURE
# ============================================================================

sample_arr   <- array(1L,   dim = c(N, T_max))
color_arr    <- array(1L,   dim = c(N, T_max, I_max))
proba_arr    <- array(0.5,  dim = c(N, T_max, I_max))
feedback_arr <- array(0L,   dim = c(N, T_max))

for (n in seq_len(N)) {
  
  data_subj <- data %>%
    filter(ParticipantPrivateID == subjs[n]) %>%
    arrange(TrialNumber)
  
  for (t in seq_len(Tsubj[n])) {
    
    row_t <- data_subj[t, ]
    
    colour_values <- unlist(row_t[color_cols], use.names = FALSE)
    proba_values  <- as.numeric(unlist(row_t[proba_cols], use.names = FALSE)) / 100
    
    colour_values <- recode_colour_1_2(colour_values)
    
    valid_samples <- which(
      !is.na(colour_values) &
        !is.na(proba_values) &
        proba_values > 0 &
        proba_values < 1
    )
    
    sample_size <- length(valid_samples)
    
    if ("sample" %in% names(data_subj)) {
      sample_size <- as.integer(row_t$sample)
    }
    
    sample_arr[n, t] <- sample_size
    
    for (s in seq_len(sample_size)) {
      color_arr[n, t, s] <- as.integer(colour_values[s])
      proba_arr[n, t, s] <- as.numeric(proba_values[s])
    }
    
    feedback_arr[n, t] <- recode_feedback_1_0(row_t[[feedback_col]])
  }
}

# ============================================================================
# 5. PARAMETER GRID
# ============================================================================
# This defines known generating values.
#
# These are group-level locations on the interpretable scale:
#   mu_alpha  in [0, 6]
#   mu_beta   unbounded
#   mu_lambda in [0, 1]
#   mu_delta  in [0, 2]
#   mu_eta    unbounded

alpha_range <- c(1.5, 2.5, 3.5, 4.5)
beta_range  <- c(0.00, 0.25, 0.50, 0.75)

# Four profiles for lambda, delta, eta.
# This keeps the default grid at 4 x 4 x 4 = 64 simulations.
lde_profiles <- tibble(
  lambda = c(0.00, 0.25, 0.50, 0.75),
  delta  = c(0.75, 1.00, 1.25, 1.50),
  eta    = c(0.00, -0.25, -0.50, 0.30)
)

parameters <- expand.grid(
  alpha_idx = seq_along(alpha_range),
  beta_idx  = seq_along(beta_range),
  lde_idx   = seq_len(nrow(lde_profiles))
)

if (k > nrow(parameters)) {
  stop(sprintf("k=%d but only %d parameter combinations exist.", k, nrow(parameters)))
}

mu_alpha  <- alpha_range[parameters$alpha_idx[k]]
mu_beta   <- beta_range[parameters$beta_idx[k]]
mu_lambda <- lde_profiles$lambda[parameters$lde_idx[k]]
mu_delta  <- lde_profiles$delta[parameters$lde_idx[k]]
mu_eta    <- lde_profiles$eta[parameters$lde_idx[k]]

group_sim <- c(
  mu_alpha  = mu_alpha,
  mu_beta   = mu_beta,
  mu_lambda = mu_lambda,
  mu_delta  = mu_delta,
  mu_eta    = mu_eta
)

cat("\nGenerating parameters:\n")
print(group_sim)

# Convert interpretable group values into the latent mu_pr scale used by Stan.
# Important: because Stan uses Phi_approx, we invert Phi_approx, not pnorm.

mu_pr <- c(
  Phi_approx_inv_R(mu_alpha / 6),    # alpha = 6 * Phi_approx(mu_pr[1])
  mu_beta,                           # beta = mu_pr[2]
  Phi_approx_inv_R(mu_lambda),       # lambda = Phi_approx(mu_pr[3])
  Phi_approx_inv_R(mu_delta / 2),    # delta = 2 * Phi_approx(mu_pr[4])
  mu_eta                             # eta = mu_pr[5]
)

# True group-level SDs on the latent/raw scale.
# These define between-subject variability in the simulated data.

sigma_pr <- c(
  0.35,  # alpha
  0.25,  # beta
  0.35,  # lambda
  0.35,  # delta
  0.25   # eta
)

# ============================================================================
# 6. SIMULATE INDIVIDUAL PARAMETERS
# ============================================================================

set.seed(42 + k)

param_names <- c("alpha", "beta", "lambda", "delta", "eta")

params_indiv <- matrix(
  NA_real_,
  nrow = N,
  ncol = 5,
  dimnames = list(as.character(subjs), param_names)
)

param_raw_sim <- matrix(
  rnorm(N * 5, 0, 1),
  nrow = N,
  ncol = 5
)

for (n in seq_len(N)) {
  
  params_indiv[n, "alpha"] <-
    6 * Phi_approx_R(mu_pr[1] + sigma_pr[1] * param_raw_sim[n, 1])
  
  params_indiv[n, "beta"] <-
    mu_pr[2] + sigma_pr[2] * param_raw_sim[n, 2]
  
  params_indiv[n, "lambda"] <-
    Phi_approx_R(mu_pr[3] + sigma_pr[3] * param_raw_sim[n, 3])
  
  params_indiv[n, "delta"] <-
    2 * Phi_approx_R(mu_pr[4] + sigma_pr[4] * param_raw_sim[n, 4])
  
  params_indiv[n, "eta"] <-
    mu_pr[5] + sigma_pr[5] * param_raw_sim[n, 5]
}

# ============================================================================
# 7. SIMULATE CHOICES FROM YOUR FULL GENERATIVE MODEL
# ============================================================================

choice_sim <- array(1L, dim = c(N, T_max))

for (n in seq_len(N)) {
  
  alpha_n  <- params_indiv[n, "alpha"]
  beta_n   <- params_indiv[n, "beta"]
  lambda_n <- params_indiv[n, "lambda"]
  delta_n  <- params_indiv[n, "delta"]
  eta_n    <- params_indiv[n, "eta"]
  
  beliefcount_blue <- 1.0
  beliefcount_red  <- 1.0
  V_b <- beliefcount_blue / (beliefcount_blue + beliefcount_red)
  
  for (t in seq_len(Tsubj[n])) {
    
    sample_size <- sample_arr[n, t]
    
    evidence <- c(0.0, 0.0)
    
    V_b_clamped <- min(max(V_b, 0.001), 0.999)
    prior_log_odds <- log(V_b_clamped / (1 - V_b_clamped))
    
    # Stan adds the prior log-odds to blue evidence only
    evidence[1] <- evidence[1] + prior_log_odds
    
    for (s in seq_len(sample_size)) {
      
      p_s <- proba_arr[n, t, s]
      c_s <- color_arr[n, t, s]
      
      l_s <- safe_logit(p_s)
      
      log_odds <- alpha_n * l_s + beta_n
      
      if (c_s == 1) {
        a <- 2 * V_b_clamped - 1
      } else {
        a <- 2 * (1 - V_b_clamped) - 1
      }
      
      current_kappa <- exp(eta_n * a)
      recency_weight <- exp(lambda_n * (s - sample_size))
      
      evidence[c_s] <- evidence[c_s] +
        recency_weight * log_odds * current_kappa
    }
    
    p_choice <- softmax_R(evidence)
    
    choice_sim[n, t] <- sample(
      x = 1:2,
      size = 1,
      prob = p_choice
    )
    
    # Belief update from actual task feedback, matching Stan
    x <- feedback_arr[n, t]
    
    beliefcount_blue <- delta_n * (beliefcount_blue - 1) + x + 1
    beliefcount_red  <- delta_n * (beliefcount_red  - 1) + (1 - x) + 1
    
    V_b <- beliefcount_blue / (beliefcount_blue + beliefcount_red)
  }
}

# ============================================================================
# 8. FIT MODEL TO SIMULATED CHOICES
# ============================================================================

dir.create("./results/choice", recursive = TRUE, showWarnings = FALSE)

out_file <- sprintf("./results/choice/recover_%03d.rds", k)

if (file.exists(out_file)) {
  cat(sprintf("\n[k=%d] Output already exists — skipping.\n", k))
  quit(save = "no", status = 0)
}

threads_total <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "4"))
threads_per_chain <- max(1L, threads_total %/% 4)

fit <- model$sample(
  data = list(
    N        = N,
    T_max    = T_max,
    I_max    = I_max,
    Tsubj    = Tsubj,
    sample   = sample_arr,
    color    = color_arr,
    proba    = proba_arr,
    choice   = choice_sim,
    feedback = feedback_arr,
    grainsize = 1
  ),
  iter_sampling     = 3000,
  iter_warmup       = 2000,
  chains            = 4,
  parallel_chains   = 4,
  threads_per_chain = threads_per_chain,
  seed              = 12345 + k,
  adapt_delta       = 0.95,
  max_treedepth     = 12,
  refresh           = 500
)

# ============================================================================
# 9. EXTRACT RECOVERY SUMMARIES
# ============================================================================

group_param_names <- c(
  "mu_alpha",
  "mu_beta",
  "mu_lambda",
  "mu_delta",
  "mu_eta"
)

group_fitted <- fit$summary(variables = group_param_names)

group_recovery <- group_fitted %>%
  mutate(
    true = group_sim[variable],
    bias = mean - true
  ) %>%
  select(variable, true, mean, median, sd, q5, q95, bias, rhat, ess_bulk, ess_tail)

# Individual-level generated quantities:
# params[n,1] = alpha
# params[n,2] = beta
# params[n,3] = lambda
# params[n,4] = delta
# params[n,5] = eta

indiv_vars <- unlist(lapply(seq_along(param_names), function(p) {
  paste0("params[", seq_len(N), ",", p, "]")
}))

indiv_fitted <- fit$summary(variables = indiv_vars)

indiv_true <- c(
  params_indiv[, "alpha"],
  params_indiv[, "beta"],
  params_indiv[, "lambda"],
  params_indiv[, "delta"],
  params_indiv[, "eta"]
)

indiv_recovery <- indiv_fitted %>%
  mutate(
    subject = rep(seq_len(N), times = length(param_names)),
    subject_id = rep(subjs, times = length(param_names)),
    parameter = rep(param_names, each = N),
    true = indiv_true,
    bias = mean - true
  ) %>%
  select(variable, subject, subject_id, parameter,
         true, mean, median, sd, q5, q95, bias,
         rhat, ess_bulk, ess_tail)

# ============================================================================
# 10. DIAGNOSTICS
# ============================================================================

diag_summary <- fit$diagnostic_summary(quiet = TRUE)
full_summary <- fit$summary()

diagnostics <- list(
  n_divergent = sum(diag_summary$num_divergent),
  n_max_treedepth = sum(diag_summary$num_max_treedepth),
  max_rhat = max(full_summary$rhat, na.rm = TRUE),
  min_ess_bulk = min(full_summary$ess_bulk, na.rm = TRUE),
  min_ess_tail = min(full_summary$ess_tail, na.rm = TRUE)
)

cat("\nDiagnostics:\n")
print(diagnostics)

# ============================================================================
# 11. SAVE
# ============================================================================

results <- list(
  k = k,
  parameter_grid_row = parameters[k, ],
  group_sim = group_sim,
  sigma_pr_sim = sigma_pr,
  mu_pr_sim = mu_pr,
  params_indiv_sim = params_indiv,
  param_raw_sim = param_raw_sim,
  choice_sim = choice_sim,
  sample = sample_arr,
  color = color_arr,
  proba = proba_arr,
  feedback = feedback_arr,
  group_fitted = group_fitted,
  group_recovery = group_recovery,
  indiv_fitted = indiv_fitted,
  indiv_recovery = indiv_recovery,
  diagnostics = diagnostics,
  param_names = param_names
)

saveRDS(results, out_file)

cat(sprintf(
  "\n[k=%d] Done. Saved to %s. Divergences=%d, max Rhat=%.3f\n",
  k,
  out_file,
  diagnostics$n_divergent,
  diagnostics$max_rhat
))