# rm(list = ls(all = TRUE))

library(tidyverse)
library(posterior)
library(bayesplot)
library(cmdstanr)
library(loo)

##################################################
## PREPARE THE DATA
##################################################

load("/Users/bty615/Documents/GitHub/reliable_info_bias/data/data_priorbelief_aware_exp11.rdata")

## Response coding:
## If ResponseButtonOrder = 1: blue -> 1, red -> 0
## If ResponseButtonOrder = 0: blue -> 0, red -> 1
##
## Recode choice:
## blue = 1
## red  = 2

data <- data %>%
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
  ##mutate(
  ## feedback = ifelse(CorrectResponse == 1, 1, 0)
  ## )

mutate(
  ## CorrectResponse is button-coded, so it must also
  ## be converted into the objectively correct colour.
  feedback = case_when(
   ResponseButtonOrder == 1 & CorrectResponse == 1 ~ 1L, # Blue
   ResponseButtonOrder == 1 & CorrectResponse == 0 ~ 0L, # Red
   ResponseButtonOrder == 0 & CorrectResponse == 0 ~ 1L, # Blue
   ResponseButtonOrder == 0 & CorrectResponse == 1 ~ 0L, # Red
    TRUE ~ NA_integer_
  )
)
##################################################
## CHECK BASIC DATA STRUCTURE
##################################################

N <- length(unique(data$ParticipantPrivateID))
T_max <- max(data$TrialNumber)
I_max <- max(data$sample_number)

d <- data %>%
  group_by(ParticipantPrivateID) %>%
  summarise(t_subjs = n(), .groups = "drop")

t_subjs <- d$t_subjs
subjs <- unique(data$ParticipantPrivateID)

cat("N subjects:", N, "\n")
cat("T max:", T_max, "\n")
cat("I max:", I_max, "\n")
cat("Total rows:", nrow(data), "\n")

##################################################
## INITIALISE ARRAYS FOR STAN
##################################################

choice   <- array(-1, c(N, T_max))
color    <- array(-1, c(N, T_max, I_max))
proba    <- array(-1, c(N, T_max, I_max))
sample   <- array(-1, c(N, T_max))
feedback <- array(0,  c(N, T_max))

##################################################
## FILL ARRAYS
##################################################

for (n in 1:N) {
  
  t <- t_subjs[n]
  
  data_subj <- data %>%
    filter(ParticipantPrivateID == subjs[n]) %>%
    arrange(TrialNumber)
  
  choice[n, 1:t] <- data_subj$choice
  feedback[n, 1:t] <- data_subj$feedback
  
  for (k in 1:t) {
    
    data_subj_t <- data_subj[k, ]
    
    sample[n, k] <- data_subj_t$sample_number
    
    for (i in 1:data_subj_t$sample_number) {
      
      color_var <- paste0("color_", i)
      proba_var <- paste0("proba_", i)
      
      color[n, k, i] <- data_subj[[color_var]][k]
      proba[n, k, i] <- data_subj[[proba_var]][k] / 100
    }
  }
}

##################################################
## CREATE DATA LIST FOR STAN
##################################################

data_list <- list(
  N = N,
  T_max = T_max,
  I_max = I_max,
  Tsubj = t_subjs,
  color = color,
  proba = proba,
  choice = choice,
  sample = sample,
  feedback = feedback,
  grainsize = 5
)

##################################################
## FIT THE MODEL
##################################################

setwd("/Users/bty615/Documents/GitHub/reliable_info_bias/stan")

model <- cmdstan_model(
  stan_file = "./log_trunc_simplified_boost_learning.stan",
  force_recompile = TRUE,
  cpp_options = list(
    stan_opencl = FALSE,
    stan_threads = TRUE
  ),
  stanc_options = list("O1"),
  compile_model_methods = TRUE
)

fit <- model$sample(
  data = data_list,
  seed = 4321,
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 5,
  iter_warmup = 2000,
  iter_sampling = 1000,
  max_treedepth = 12,
  adapt_delta = 0.9,
  save_warmup = FALSE
)

##################################################
## COMPUTE LOO
##################################################

loo_result <- fit$loo(
  cores = 10,
  moment_match = TRUE
)

print(loo_result)

##################################################
## SAVE RESULTS
##################################################

dir.create("./results/fits/exp11_unaware/", recursive = TRUE, showWarnings = FALSE)
dir.create("./results/loo/exp11_unaware/", recursive = TRUE, showWarnings = FALSE)

save(
  fit,
  file = "./results/fits/exp11_unaware/fit_trunc_boost_unaware_exp11.rdata"
)

save(
  loo_result,
  file = "./results/loo/exp11_unaware/loo_trunc_boost_unaware_exp11.rdata"
)

cat("\nSaved fit to:\n")
cat("./results/fits/exp11_unaware/fit_trunc_boost_unaware_exp11.rdata\n")

cat("\nSaved LOO to:\n")
cat("./results/loo/exp11_unaware/loo_trunc_boost_unaware_exp11.rdata\n")

































####################################################
#  RESULTS ANALYSIS
####################################################


# ---------------------------------------------------
# DEFINE AND LOAD EXPERIMENT
# ---------------------------------------------------

exp <- 'exp11'
models <- c('basic','theta','trunc_simplified')
models <- c('basic','theta','trunc','full')
#load(paste0('./data/data_list_',exp,'.rdata'))
load("/Users/bty615/Documents/GitHub/reliable_info_bias/data/data_priorbelief_aware_exp11.rdata")
N <- data_list$N

# ---------------------------------------------------
# PAIRS AND TRACES PLOTS
# ---------------------------------------------------

## LOAD MODELS FITS
fits <- vector("list", length(models))
for (i in 1:length(models)){
    load(paste0('./results/fits/',exp,'/fit_',models[i],'_',exp,'.rdata'))
    fits[[i]] <- fit
}

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_boost_aware_exp11.rdata")
fit_trunc_simplified <- fit
## Pairs plots
for (i in 1:length(models)) {
  # Extract the current fit
 # fit <- fit[[i]]
  posterior_samples <- fit$draws()
  posterior_df <- as_draws_df(posterior_samples)
  selected_params <- posterior_df[, grepl("^mu", colnames(posterior_df)) & !grepl("^mu_pr", colnames(posterior_df))]
  plot <- mcmc_pairs(selected_params)
  plot_file <- paste0('./results/plots/',exp,'/pairs_plot_', models[i], "_", exp, ".pdf")
  ggsave(plot, file = plot_file)
}







library(posterior)
library(bayesplot)
library(ggplot2)

# Load the CmdStanR fit object
load("/Users/imogen/Documents/GitHub/reliable_info/results/fits/Exp12/fit_trunc_simplified_learning_aware_exp12.rdata")

# Extract draws properly
posterior_df <- as_draws_df(fit$draws())

# Pick the parameters you want (example: 'mu' but not 'mu_pr')
selected_params <- posterior_df[, 
                                grepl("^mu", names(posterior_df)) & !grepl("^mu_pr", names(posterior_df))
]

# Plot
p <- mcmc_pairs(selected_params)
print(p)





selected_params <- posterior_df[, grepl("param_raw", colnames(posterior_df)) ]
s1 <- fit$summary(
  variables = c('param_raw'),
  posterior::default_summary_measures(),
  extra_quantiles = ~posterior::quantile2(., probs = c(.0275, 0.5, .975))
)



selected_params <- posterior_df[, grepl("param_raw", colnames(posterior_df)) ]
s1 <- fit$summary(
  #variables = c('mu_beta'),
  variable = grepl("^mu", colnames(posterior_df))
  posterior::default_summary_measures(),
  extra_quantiles = ~posterior::quantile2(., probs = c(.0275, 0.5, .975))
)


library(tidyverse)
library(posterior)
library(cmdstanr)

# =====================================================================
# 1. SETUP: Define Paths and Files to Process
# =====================================================================

# --- CRITICAL: Define the shared directory where RData files are located ---
BASE_DIR <- "/Users/imogen/Documents/GitHub/reliable_info/results/fits/Exp12" 

# --- Define the three files to process ---
FILES_TO_PROCESS <- tribble(
  ~rdata_file, ~group_suffix,
  "fit_trunc_simplified_learning_aware_exp11.rdata", "aware_exp11",
  "fit_trunc_simplified_learning_unaware_exp11.rdata", "unaware_exp11",
  "fit_trunc_simplified_learning_aware_exp12.rdata", "aware_exp12"
)


# =====================================================================
# 2. FUNCTION DEFINITION: process_fit_to_csv (DEFINITIVE FIX)
# =====================================================================

# Function to extract, summarize, and save parameters for a single fit object.
process_fit_to_csv <- function(fit, group_suffix, output_path) {
  
  filename_prefix <- paste0("param_trunc_simplified_learning_", group_suffix)
  
  # The desired final output order (alpha, beta, lambda, theta, psi, delta)
  FINAL_PARAM_ORDER <- c("alpha", "beta", "lambda", "theta", "psi", "delta")
  
  # The FIXED mapping to correct the observed swap of index 5 (psi) and 6 (delta)
  STAN_INDEX_NAMES_FIXED <- c("alpha", "beta", "lambda", "theta", "delta", "psi")
  
  # --- A. Individual Parameters Extraction (params[n,j]) ---
  
  cat("-> Extracting Individual Parameters...\n")
  
  # 1a. Extract draws for all subject-level parameters
  posterior_df <- fit$draws(variables = "params") %>% as_draws_df()
  param_cols <- posterior_df %>% select(matches("^params\\["))
  
  # 1b. Convert to long format and map indices to names
  df_long <- param_cols %>%
    as_draws_df() %>%
    pivot_longer(
      cols = everything(),
      names_to = "var",
      values_to = "value"
    ) %>%
    # Parse variable name "params[i,j]" into indices
    separate(
      var,
      into = c("drop", "i", "j"),
      sep = "\\[|,|\\]",
      extra = "drop",
      fill = "right"
    ) %>%
    mutate(
      subj = as.numeric(i),
      par  = as.numeric(j)
    ) %>%
    filter(!is.na(subj), !is.na(par)) %>%
    select(-drop, -i, -j) %>%
    
    mutate(param = STAN_INDEX_NAMES_FIXED[par]) %>% 
    select(subj, param, value)
  
  # 1c. Compute posterior summaries (mean, median, SD, quantiles)
  param_individual <- df_long %>%
    group_by(subj, param) %>%
    summarise(
      mean    = mean(value, na.rm = TRUE),
      median  = median(value, na.rm = TRUE),
      sd      = sd(value, na.rm = TRUE),
      mad     = mad(value, na.rm = TRUE),
      q5      = quantile(value, 0.05, na.rm = TRUE), 
      q95     = quantile(value, 0.95, na.rm = TRUE), 
      .groups = "drop"
    ) %>%
    mutate(
      rhat = NA, ess_bulk = NA, ess_tail = NA,
      # Convert param to a factor using the defined output order (FINAL_PARAM_ORDER)
      param = factor(param, levels = FINAL_PARAM_ORDER) 
    ) %>%
    # Sorts by subject first, then by the factor level (enforcing MATLAB's order)
    arrange(subj, param) 
  
  # 1d. SAVE Individual CSV
  write_csv(
    param_individual,
    file.path(output_path, paste0(filename_prefix, "_individual.csv"))
  )
  
  # --- B. Group Parameters Extraction (mu_alpha, mu_beta, etc.) ---
  
  cat("  -> Extracting Group Parameters...\n")
  # 2a. Extract transformed means from Stan's 'generated quantities'
  draw_vars <- c("mu_alpha", "mu_beta", "mu_lambda", "mu_theta","mu_psi", "mu_delta")
  draws_all <- fit$draws(variables = draw_vars)
  
  group_df <- draws_all %>%
    as_draws_df() %>%
    # Rename variables by removing the 'mu_' prefix
    rename_with(~ sub("^mu_", "", .), starts_with("mu_")) %>%
    # Select all 6 parameters
    select(alpha, beta, psi, lambda, theta, delta) 
  
  # 2b. Compute summary statistics
  param_group <- group_df %>%
    pivot_longer(everything(), names_to="param", values_to="value") %>%
    group_by(param) %>%
    summarise(
      mean   = mean(value, na.rm = TRUE),
      median = median(value, na.rm = TRUE),
      sd     = sd(value, na.rm = TRUE),
      mad    = mad(value, na.rm = TRUE),
      q5     = quantile(value, .05, na.rm = TRUE),
      q95    = quantile(value, .95, na.rm = TRUE),
      .groups="drop"
    ) %>%
    mutate(
      rhat = NA, ess_bulk = NA, ess_tail = NA,
      # Convert param to a factor using the defined output order (FINAL_PARAM_ORDER)
      param = factor(param, levels = FINAL_PARAM_ORDER) 
    ) %>%
    # Sorts by factor level, ensuring the Group CSV is also in the correct order
    arrange(param) %>% 
    select(param, mean, median, sd, mad, q5, q95, rhat, ess_bulk, ess_tail)
  
  # 2c. SAVE Group CSV
  write_csv(
    param_group,
    file.path(output_path, paste0(filename_prefix, "_group.csv"))
  )
  
  cat(paste0("  -> Saved CSVs with prefix: ", filename_prefix, "\n"))
  return(NULL)
}


# =====================================================================
# 3. EXECUTION LOOP
# =====================================================================

cat("Starting Stan results processing...\n")

for (i in 1:nrow(FILES_TO_PROCESS)) {
  
  file_name <- FILES_TO_PROCESS$rdata_file[i] 
  
  group_suffix <- FILES_TO_PROCESS$group_suffix[i]
  
  full_path <- file.path(BASE_DIR, file_name)
  
  cat(paste("--- Processing:", file_name, "---\n"))
  
  # Load the fit object (assumes the object is named 'fit' inside the RData file)
  load(full_path)
  
  # Process and save the CSVs
  process_fit_to_csv(fit, group_suffix, BASE_DIR)
  
  # Clean up the fit object to prepare for the next load
  rm(fit) 
}

cat("----------------------------------------------------------------\n")
cat(paste("SUCCESS: All CSVs saved to:", BASE_DIR, "\n"))
cat("----------------------------------------------------------------\n")






## Traces
for (i in 1:length(models)) {
  # Extract the current fit
  fit <- fits[[i]]
  posterior_samples <- fit$draws()
  posterior_df <- as_draws_df(posterior_samples)
  selected_params <- posterior_df[, grepl("^mu", colnames(posterior_df))]
  plot <- mcmc_trace(selected_params)
  plot_file <- paste0('./results/plots/',exp,'/trace_plot_', models[i], "_", exp, ".pdf")
  ggsave(plot, file = plot_file)
}

# ---------------------------------------------------
#  MODELS SUMMARIES 
# ---------------------------------------------------

## Parameters values
all_summaries <- list()
for (i in 1:length(models)) {
    fit <- fits[[i]]
    posterior_samples <- fit$draws()
    posterior_df <- as_draws_df(posterior_samples)
    selected_params <- posterior_df[, grepl("^mu", colnames(posterior_df)) & !grepl("^mu_pr", colnames(posterior_df))]
    summary_stats <- posterior_summary(selected_params)
    summary_df <- as.data.frame(summary_stats)
    all_summaries[[i]] <- summary_df
    html_file <- paste0("./results/summary/",exp,"/summary_mu_", models[i], "_", exp, ".html")
    latex_file <- paste0("./results/summary/",exp,"/summary_mu_", models[i], "_", exp, ".tex")
    html_content <- kable(summary_df, format = "html", table.attr = "class='table table-bordered'")
    writeLines(html_content, html_file)
    latex_content <- kable(summary_df, format = "latex", booktabs = TRUE)
    writeLines(latex_content, latex_file)
}


# ---------------------------------------------------
# MODEL COMPARISON
# ---------------------------------------------------

## Model Comparison
loos <- vector("list", length(models))
for (i in 1:length(models)){
    load(paste0('./results/loo/',exp,'/loo_',models[i],'_',exp,'.rdata'))
    loos[[i]] <- loo
}
loo_comparison <- loo_compare(loos)
html_content <- kable(loo_comparison, format = "html", table.attr = "class='table table-bordered'")
writeLines(html_content, paste0('./results/summary/',exp,'/loo_',exp,'.html'))

## PLOT
# Extract elpd_loo and se_elpd_loo for each model
colnames(loo_comparison) <- c("elpd_diff", "se_diff", "elpd_loo", "se_elpd_loo", 
                              "p_loo", "se_p_loo", "looic", "se_looic")
model_digits <- gsub("model", "", rownames(loo_comparison))  # Extract digits (e.g., "3", "2", "1")
model_indices <- as.numeric(model_digits)  # Convert to numeric indices
model_names <- models[model_indices]  # This will give the correct names like 'full', 'trunc', 'theta'

# Adjust the rownames of loo_comparison to use the correct model names
rownames(loo_comparison) <- model_names
elpd_diff_values <- loo_comparison[, "elpd_diff"]
se_diff_values <- loo_comparison[, "se_diff"]
df <- data.frame(
  model = rownames(loo_comparison),   # Model names (adjusted)
  elpd_diff = elpd_diff_values,       # ELPD difference values
  se_elpd_diff = se_diff_values      # Standard errors for ELPD difference
)


plot_loo <- ggplot(df, aes(x = reorder(model, elpd_diff), y = elpd_diff, fill = model)) +
  geom_bar(stat = "identity", show.legend = FALSE) +  # Bar plot for elpd_diff values
  geom_errorbar(aes(ymin = elpd_diff - se_elpd_diff, ymax = elpd_diff + se_elpd_diff), 
                width = 0.2, color = "black") +  # Error bars
  labs(
    title = "Model Comparison: ELPD Difference with Error Bars",
    x = "Model",
    y = "ELPD Difference"
  ) +
  theme_minimal()  # Minimal theme for aesthetics
ggsave(plot_loo, file = paste0('./results/plots/',exp,'/loo_',exp,'.pdf'))

# ---------------------------------------------------
# PARAMETERS ANALYSIS
# ---------------------------------------------------

param_indiv <- list()
for (i in 1:length(models)) {
    fit <- fits[[i]]
    draws <- fit$draws()  
    variables <- variables(draws)
    params_vars <- grep("^params\\[", variables, value = TRUE)
    median <- sapply(params_vars, function(var) median(as_draws_df(draws)[[var]]))
    d <- data.frame(median = median) 
    vars <- colnames(as_draws_df(draws)) %>%
        .[!grepl("^mu_pr", .)] %>%       # Remove variables starting with "mu_pr"
        grep("^mu_", ., value = TRUE) %>% # Keep only variables starting with "mu_"
        sub("^mu_", "", .)                # Remove "mu_" prefix
    d$subj <- rep(c(1:N),length(vars))
    d$param <- d$param <- rep(vars, each = N)
    d <- pivot_wider(d, 
                      id_cols = subj, 
                      names_from = param, 
                      values_from = median)
    d$model <- models[i]
    param_indiv[[i]] <- d
}
params <- bind_rows(param_indiv)
save(params, file = paste0('./results/summary/',exp,'/params_indiv_',exp,'.rdata'))


### theta plot
plot_theta <- ggplot(
  params %>%
    group_by(model) %>%
    arrange(theta, .by_group = TRUE) %>%
    mutate(order = row_number()),
  aes(x = order, y = theta)
) +
  geom_point(color = "steelblue") +
  geom_hline(yintercept = 1, color = "red", linetype = "dashed", linewidth = 0.8) +
  facet_wrap(~ model, scales = "free_y") +
  labs(title = "Alpha Values by Model",
       x = "Sorted Order", y = "Theta") +
  theme_minimal()
ggsave(plot_theta, file = paste0('./results/plots/',exp,'/theta_',exp,'.pdf'))


################################################################################
##                 PROBA TRANSFORMATION
################################################################################
f_basic <- function(p, alpha, beta) {
    l = alpha*log(p/(1-p))+beta
    fp = exp(l)/(1+exp(l))
  return(fp)
}
f_trunc <- function(p, psi, l_inf, l_diff, alpha, beta) {
    l = log(p/(1-p))
    if (l<l_inf)
        l = l_inf
    else if (l>l_inf + l_diff)
        l = l_inf + l_diff    
    ll = (2*alpha*psi)/l_diff*(l-l_inf-l_diff/2) + (1-alpha)*beta
    fp = exp(ll)/(1+exp(ll))
  return(fp)
}

f_trunc_simplified <- function(p, psi, l_inf, l_diff, alpha, beta) {
    l = log(p/(1-p)) 
    ll = alpha*psi*l + (1-alpha)*beta
    fp = exp(ll)/(1+exp(ll))
  return(fp)
}
f_full <- function(p, psi, l_inf, l_diff, kappa, beta) {
    l = log(p/(1-p))
    if (l<l_inf)
        l = l_inf
    else if (l>l_inf + l_diff)
        l = l_inf + l_diff   
    pp = exp(l)/(1+exp(l))
    alpha = 1/(1+kappa*pp*(1-pp))
    ll = (2*alpha*psi)/l_diff*(l-l_inf-l_diff/2) + (1-alpha)*beta
    fp = exp(ll)/(1+exp(ll))
  return(fp)
}




# ---------------------------------------------------
# BASIC MODEL
# ---------------------------------------------------


## GROUP LEVEL
parameters <- all_summaries[[1]]
p = c(1:99)/100
alpha = parameters["mu_alpha", "Estimate"]
beta = parameters["mu_beta", "Estimate"]
fp <- sapply(p,f_basic, alpha =alpha, beta=beta)

df <- data.frame(p = p, fp = fp)
plot_f_basic <- ggplot(df, aes(x = p, y = fp)) +
    geom_line(color = "blue", size = 1) +        # function curve
    geom_abline(intercept = 0, slope = 1,        # diagonal y = x
                color = "red", linetype = "dashed") +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_minimal()
ggsave(plot_f_basic, file = paste0('./results/plots/',exp,'/f_basic_',exp,'.pdf'))

## INDIVIDUAL
parameters <-  param_indiv[[1]] 
plot_data <- list()
for (i in 1:nrow(parameters)) {
  subj_data <- parameters[i, ]
  p <- c(1:99) / 100  # p values from 0.01 to 0.99
  alpha <- subj_data$alpha
  beta <- subj_data$beta
  fp <- sapply(p, f_basic, alpha = alpha, beta = beta)
  df_plot <- data.frame(p = p, fp = fp, subj = subj_data$subj)
  plot_data[[i]] <- df_plot
}

# Combine all the individual subject data frames into one
df_all <- do.call(rbind, plot_data)

# Create a single plot with all subjects
plot_f_basic_all <- ggplot(df_all, aes(x = p, y = fp, color = factor(subj))) +
  geom_line(size = 1) +       # Function curve
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +  # Diagonal y = x
  facet_wrap(~ subj, scales = "free_y", ncol = 5) + # Create a plot for each subject
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  labs(title = "Proba Transformation by Subject", x = "p", y = "f_basic(p)") +
  theme(legend.position = "none")

# Save the combined plot for all subjects
ggsave(plot_f_basic_all, file = paste0('./results/plots/', exp, '/f_basic_all_subjects_', exp, '.pdf'))


# ---------------------------------------------------
# THETA MODEL
# ---------------------------------------------------


## GROUP LEVEL
parameters <- all_summaries[[2]]
p = c(1:99)/100
alpha = parameters["mu_alpha", "Estimate"]
beta = parameters["mu_beta", "Estimate"]
fp <- sapply(p,f_basic, alpha =alpha, beta=beta)

df <- data.frame(p = p, fp = fp)
plot_f_theta <- ggplot(df, aes(x = p, y = fp)) +
    geom_line(color = "blue", size = 1) +        # function curve
    geom_abline(intercept = 0, slope = 1,        # diagonal y = x
                color = "red", linetype = "dashed") +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_minimal()
ggsave(plot_f_theta, file = paste0('./results/plots/',exp,'/f_theta_',exp,'.pdf'))

## INDIVIDUAL
parameters <-  param_indiv[[2]] 
plot_data <- list()
for (i in 1:nrow(parameters)) {
  subj_data <- parameters[i, ]
  p <- c(1:99) / 100  # p values from 0.01 to 0.99
  alpha <- subj_data$alpha
  beta <- subj_data$beta
  fp <- sapply(p, f_basic, alpha = alpha, beta = beta)
  df_plot <- data.frame(p = p, fp = fp, subj = subj_data$subj)
  plot_data[[i]] <- df_plot
}

# Combine all the individual subject data frames into one
df_all <- do.call(rbind, plot_data)

# Create a single plot with all subjects
plot_f_theta_all <- ggplot(df_all, aes(x = p, y = fp, color = factor(subj))) +
  geom_line(size = 1) +       # Function curve
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +  # Diagonal y = x
  facet_wrap(~ subj, scales = "free_y", ncol = 5) + # Create a plot for each subject
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  labs(title = "Proba Transformation by Subject", x = "p", y = "f_theta(p)") +
  theme(legend.position = "none")

# Save the combined plot for all subjects
ggsave(plot_f_theta_all, file = paste0('./results/plots/', exp, '/f_theta_all_subjects_', exp, '.pdf'))

# ---------------------------------------------------
# TRUNC MODEL
# ---------------------------------------------------
####
## GROUP LEVEL
parameters <- all_summaries[[3]]
p = c(1:99)/100
l_inf = parameters["mu_l_inf", "Estimate"]
l_diff = parameters["mu_l_diff", "Estimate"]
alpha = parameters["mu_alpha", "Estimate"]
beta = parameters["mu_beta", "Estimate"]
psi = parameters["mu_psi", "Estimate"]
fp <- sapply(p, f_trunc, psi = psi, l_inf = l_inf, l_diff=l_diff, alpha =alpha, beta=beta)

df <- data.frame(p = p, fp = fp)
plot_f_trunc <- ggplot(df, aes(x = p, y = fp)) +
    geom_line(color = "blue", size = 1) +        # function curve
    geom_abline(intercept = 0, slope = 1,        # diagonal y = x
                color = "red", linetype = "dashed") +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_minimal()
ggsave(plot_f_trunc, file = paste0('./results/plots/',exp,'/f_trunc_',exp,'.pdf'))

## INDIVIDUAL
parameters <-  param_indiv[[3]] 
plot_data <- list()
for (i in 1:nrow(parameters)) {
  subj_data <- parameters[i, ]
  p <- c(1:99) / 100  # p values from 0.01 to 0.99
  alpha <- subj_data$alpha
  beta <- subj_data$beta
  l_inf <-subj_data$l_inf
  l_diff <- subj_data$l_diff
  psi <-subj_data$psi
  fp <- sapply(p, f_trunc, psi = psi, l_inf = l_inf, l_diff = l_diff, alpha = alpha, beta = beta)
  df_plot <- data.frame(p = p, fp = fp, subj = subj_data$subj)
  plot_data[[i]] <- df_plot
}

# Combine all the individual subject data frames into one
df_all <- do.call(rbind, plot_data)

# Create a single plot with all subjects
plot_f_trunc_all <- ggplot(df_all, aes(x = p, y = fp, color = factor(subj))) +
  geom_line(size = 1) +       # Function curve
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +  # Diagonal y = x
  facet_wrap(~ subj, scales = "free_y", ncol = 5) + # Create a plot for each subject
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  labs(title = "Proba Transformation by Subject", x = "p", y = "f_trunc(p)") +
  theme(legend.position = "none")

# Save the combined plot for all subjects
ggsave(plot_f_trunc_all, file = paste0('./results/plots/', exp, '/f_trunc_all_subjects_', exp, '.pdf'))

# ---------------------------------------------------
## TRUNC SIMPLIFIED
# ---------------------------------------------------

## GROUP LEVEL
parameters <- all_summaries[[3]]
p = c(1:99)/100
alpha = parameters["mu_alpha", "Estimate"]
beta = parameters["mu_beta", "Estimate"]
psi = parameters["mu_psi", "Estimate"]
fp <- sapply(p, f_trunc_simplified, psi = psi, alpha =alpha, beta=beta)

df <- data.frame(p = p, fp = fp)
plot_f_trunc_simplified <- ggplot(df, aes(x = p, y = fp)) +
    geom_line(color = "blue", size = 1) +        # function curve
    geom_abline(intercept = 0, slope = 1,        # diagonal y = x
                color = "red", linetype = "dashed") +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_minimal()
ggsave(plot_f_trunc_simplified, file = paste0('./results/plots/',exp,'/f_trunc_simplified_',exp,'.pdf'))

## INDIVIDUAL
parameters <-  param_indiv[[3]] 
plot_data <- list()
for (i in 1:nrow(parameters)) {
  subj_data <- parameters[i, ]
  p <- c(1:99) / 100  # p values from 0.01 to 0.99
  alpha <- subj_data$alpha
  beta <- subj_data$beta
  psi <-subj_data$psi
  fp <- sapply(p, f_trunc_simplified, psi = psi,  alpha = alpha, beta = beta)
  df_plot <- data.frame(p = p, fp = fp, subj = subj_data$subj)
  plot_data[[i]] <- df_plot
}

# Combine all the individual subject data frames into one
df_all <- do.call(rbind, plot_data)

# Create a single plot with all subjects
plot_f_trunc_simplified_all <- ggplot(df_all, aes(x = p, y = fp, color = factor(subj))) +
  geom_line(size = 1) +       # Function curve
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +  # Diagonal y = x
  facet_wrap(~ subj, scales = "free_y", ncol = 5) + # Create a plot for each subject
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  labs(title = "Proba Transformation by Subject", x = "p", y = "f_trunc_simp(p)") +
  theme(legend.position = "none")

# Save the combined plot for all subjects
ggsave(plot_f_trunc_simplified_all, file = paste0('./results/plots/', exp,'/f_trunc_simplified_all_subjects_', exp, '.pdf'))



# ---------------------------------------------------
# FULL MODEL
# ---------------------------------------------------
## GROUP LEVEL
parameters <- all_summaries[[4]]
p = c(1:99)/100
l_inf = parameters["mu_l_inf", "Estimate"]
l_diff = parameters["mu_l_diff", "Estimate"]
kappa =1/parameters["mu_kappa", "Estimate"]
beta = parameters["mu_beta", "Estimate"]
psi = parameters["mu_psi", "Estimate"]
fp <- sapply(p, f_full, psi = psi, l_inf = l_inf, l_diff=l_diff, kappa =kappa, beta=beta)

df <- data.frame(p = p, fp = fp)
plot_f_full <- ggplot(df, aes(x = p, y = fp)) +
    geom_line(color = "blue", size = 1) +        # function curve
    geom_abline(intercept = 0, slope = 1,        # diagonal y = x
                color = "red", linetype = "dashed") +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_minimal()
ggsave(plot_f_full, file = paste0('./results/plots/',exp,'/f_full_',exp,'.pdf'))

## INDIVIDUAL
parameters <-  param_indiv[[4]] 
plot_data <- list()
for (i in 1:nrow(parameters)) {
  subj_data <- parameters[i, ]
  p <- c(1:99) / 100  # p values from 0.01 to 0.99
  kappa <- 1/subj_data$kappa
  beta <- subj_data$beta
  l_inf <-subj_data$l_inf
  l_diff <- subj_data$l_diff
  psi <-subj_data$psi
  fp <- sapply(p, f_full, psi = psi, l_inf = l_inf, l_diff = l_diff, kappa = kappa, beta = beta)
  df_plot <- data.frame(p = p, fp = fp, subj = subj_data$subj)
  plot_data[[i]] <- df_plot
}

# Combine all the individual subject data frames into one
df_all <- do.call(rbind, plot_data)

# Create a single plot with all subjects
plot_f_full_all <- ggplot(df_all, aes(x = p, y = fp, color = factor(subj))) +
  geom_line(size = 1) +       # Function curve
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +  # Diagonal y = x
  facet_wrap(~ subj, scales = "free_y", ncol = 5) + # Create a plot for each subject
  scale_x_continuous(limits = c(0, 1)) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  labs(title = "Proba Transformation by Subject", x = "p", y = "f_full(p)") +
  theme(legend.position = "none")

# Save the combined plot for all subjects
ggsave(plot_f_full_all, file = paste0('./results/plots/', exp, '/f_full_all_subjects_', exp, '.pdf'))












library(tidyverse)
library(posterior)
library(ggplot2)

# -------------------------------------------------------------------
# Load fits 
# -------------------------------------------------------------------
load("/Users/imogen/Documents/GitHub/reliable_info/results/fits/Exp12/fit_trunc_simplified_learning_aware_red_exp12.rdata")
fit_unaware <- fit

load("/Users/imogen/Documents/GitHub/reliable_info/results/fits/Exp12/fit_trunc_simplified_learning_aware_exp11.rdata")
fit_aware <- fit

load("/Users/imogen/Documents/GitHub/reliable_info/results/fits/Exp12/fit_trunc_simplified_learning_aware_exp12.rdata")
fit_explicit <- fit


# -------------------------------------------------------------------
# Function: extract  the transformed mu_ parameters
# -------------------------------------------------------------------
extract_mu <- function(fit, label) {
  
  d <- as_draws_df(fit$draws())
  

  keep_params <- c(
    "mu_alpha",
    "mu_beta",
    "mu_lambda",
    "mu_theta",
    "mu_psi",
    "mu_delta"
  )
  
  
  keep_params <- keep_params[keep_params %in% colnames(d)]
  
  mu_df <- d %>%
    select(all_of(keep_params)) %>%
    mutate(group = label) %>%
    pivot_longer(
      cols = all_of(keep_params),
      names_to = "param",
      values_to = "value"
    )
  
  return(mu_df)
}


# -------------------------------------------------------------------
# Extract and Set Order
# -------------------------------------------------------------------
df_unaware  <- extract_mu(fit_unaware,  "Implicit Unaware")
df_aware    <- extract_mu(fit_aware,    "Implicit Aware")
df_explicit <- extract_mu(fit_explicit, "Explicit Aware")

df_all <- bind_rows(df_unaware, df_aware, df_explicit)

# CRITICAL FIX: Convert 'group' to a factor and set the order.
# Plotting order: Darkest -> Medium -> Lightest (so lightest plots on top)
df_all$group <- factor(df_all$group, levels = c(
  "Implicit Aware",     # Dark Green (plotted first/bottom)
  "Explicit Aware",     # Orange (plotted second/middle)
  "Implicit Unaware"    # Light Green (plotted last/top)
))


# -------------------------------------------------------------------
# Define Custom Labels and Colors
# -------------------------------------------------------------------
param_labels <- c(
  "mu_alpha" = expression(paste(mu[alpha] )),
  "mu_beta"  = expression(paste(mu[beta])),
  "mu_lambda"= expression(paste(mu[lambda], " (Sequential Decay)")),
  "mu_psi"   = expression(paste(mu[psi], " (Distortion Scaling)")),
  "mu_theta" = expression(paste(mu[theta], " (Response Noise)")),
  "mu_delta" = expression(paste(mu[delta], " (Learning Rate)"))
)

custom_colors <- c(
  "Explicit Aware"   = "#E69F00",  # Orange
  "Implicit Aware"   = "#1B5E20",  # Dark Green
  "Implicit Unaware" = "#A8D08D"   # Light Green
)


# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
p <- ggplot(df_all, aes(x = value, fill = group)) +
  geom_histogram(
    position = "identity",
    bins = 80,
    alpha = 0.55,           # Slight adjustment to transparency
    color = "black",        #  Add a black outline to define boundaries
    linewidth = 0.1         # Make the outline thin
  ) +
  facet_wrap(~param, scales = "free", ncol = 3, labeller = as_labeller(param_labels, default = label_parsed)) +
  
  # Apply the custom colors
  scale_fill_manual(values = custom_colors) +
  
  theme_bw(base_size = 16) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    
    strip.background = element_rect(fill = "gray90", color = "gray50"),
    strip.text = element_text(face = "bold", size = 12),
    
    legend.position = "bottom",
    legend.key.size = unit(0.8, "cm")
  ) +
  labs(
    title = "Posterior Parameter Distributions",
    x = "Posterior Sample Value",
    y = "Frequency",
    fill = "Awareness Group"
  )

print(p)

# ===============================================================
# SECTION 2: BAYESIAN EXCEEDANCE PROBABILITY TESTS 
# ===============================================================

# ----------------------------------------------------------------
# 2.1 Extract pooled posterior draws
# ----------------------------------------------------------------

extract_mu_draws <- function(fit) {
  
  d <- as_draws_df(fit$draws())
  
  keep_params <- c(
    "mu_alpha",
    "mu_beta",
    "mu_lambda",
    "mu_theta",
    "mu_psi",
    "mu_delta"
  )
  
  keep_params <- keep_params[keep_params %in% colnames(d)]
  
  if (length(keep_params) == 0) {
    stop("No mu_* parameters found in this fit.")
  }
  
  d %>% select(all_of(keep_params))
}

draws_unaware  <- extract_mu_draws(fit_unaware)
draws_aware    <- extract_mu_draws(fit_aware)
draws_explicit <- extract_mu_draws(fit_explicit)

# ----------------------------------------------------------------
# 2.2 Identify common parameters across all models
# ----------------------------------------------------------------

mu_names <- Reduce(
  intersect,
  list(
    colnames(draws_explicit),
    colnames(draws_aware),
    colnames(draws_unaware)
  )
)



cat("\n--- PARAMETERS USED FOR EXCEEDANCE TESTS ---\n")
print(mu_names)
cat("------------------------------------------\n")

# ----------------------------------------------------------------
# 2.3 Exceedance Probability Function
# ----------------------------------------------------------------

calculate_exceedance <- function(draws_X, draws_Y, param_name) {
  
  diff <- draws_X[[param_name]] - draws_Y[[param_name]]
  
  exceedP_XY <- mean(diff > 0)
  mean_diff  <- mean(diff)
  
  SE <- if (mean_diff > 0) mean(diff < 0) else exceedP_XY
  
  list(
    ExceedP_XY = exceedP_XY,
    SE_value   = SE,
    Mean_Diff  = mean_diff
  )
}

# ----------------------------------------------------------------
# 2.4 Ordered Group Comparisons
# Explicit → Implicit Aware → Implicit Unaware
# ----------------------------------------------------------------

group_pairs <- list(
  list("Explicit Aware",  draws_explicit, "Implicit Aware",   draws_aware),
  list("Explicit Aware",  draws_explicit, "Implicit Unaware", draws_unaware),
  list("Implicit Aware",  draws_aware,    "Implicit Unaware", draws_unaware)
)

# ----------------------------------------------------------------
# 2.5 Run Comparisons
# ----------------------------------------------------------------

results <- list()
k <- 1

for (p in mu_names) {
  for (pair in group_pairs) {
    
    group1 <- pair[[1]]
    draws1 <- pair[[2]]
    group2 <- pair[[3]]
    draws2 <- pair[[4]]
    
    res <- calculate_exceedance(draws1, draws2, p)
    
    results[[k]] <- tibble(
      Parameter = p,
      Group1 = group1,
      Group2 = group2,
      Mean1 = mean(draws1[[p]]),
      Mean2 = mean(draws2[[p]]),
      Mean_Diff = res$Mean_Diff,
      P_Exceed_G1_G2 = res$ExceedP_XY,
      P_Exceed_Smaller = res$SE_value
    )
    
    k <- k + 1
  }
}

results_df <- bind_rows(results) %>%
  mutate(
    Significance = P_Exceed_Smaller < 0.05,
    Comparison = paste(Group1, ">", Group2)
  ) %>%
  select(
    Parameter, Comparison,
    Mean1, Mean2, Mean_Diff,
    P_Exceed_G1_G2, P_Exceed_Smaller, Significance
  )

# ----------------------------------------------------------------
# 2.6 Print Results
# ----------------------------------------------------------------

cat("\n==============================================================\n")
cat("BAYESIAN EXCEEDANCE PROBABILITY TEST RESULTS\n")
cat("Ordering: Explicit Aware > Implicit Aware > Implicit Unaware\n")
cat("==============================================================\n")
print(results_df, n = Inf)

















library(tidyverse)
library(posterior)
library(bayesplot)
library(ggplot2)

# -------------------------------------------------------------------
# 1. DEFINE FILES AND LABELS
# -------------------------------------------------------------------
base_path <- "/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/"

# Mapping the filenames from your screenshot
file_list <- list(
  "Exp11 Aware"   = "fit_trunc_simplified_learning_boost_aware_exp11.rdata",
  "Exp11 Unaware" = "fit_trunc_simplified_learning_boost_unaware_exp11.rdata",
  "Exp12 Aware"   = "fit_trunc_simplified_learning_boost_aware_exp12.rdata"
)

# -------------------------------------------------------------------
# 2. EXTRACTION FUNCTION (Updated for 5 params)
# -------------------------------------------------------------------
extract_mu_all <- function(path, label) {
  if(!file.exists(path)) {
    message("File missing: ", path)
    return(NULL)
  }
  
  load(path) # Loads the 'fit' object
  
  d <- as_draws_df(fit$draws())
  
  # Mapping: 1=alpha, 2=beta, 3=lambda, 4=delta, 5=eta
  mu_map <- c(
    "mu_pr[1]" = "mu_alpha",
    "mu_pr[2]" = "mu_beta",
    "mu_pr[3]" = "mu_lambda",
    "mu_pr[4]" = "mu_delta",
    "mu_pr[5]" = "mu_eta"
  )
  
  mu_df <- d %>%
    select(all_of(names(mu_map))) %>%
    rename(!!!setNames(names(mu_map), mu_map)) %>%
    mutate(Condition = label) %>%
    pivot_longer(
      cols = starts_with("mu_"), 
      names_to = "param", 
      values_to = "value"
    )
  
  return(mu_df)
}

# Combine all datasets
all_data <- map2_df(file_list, names(file_list), ~ {
  full_path <- file.path(base_path, .x)
  extract_mu_all(full_path, .y)
})

# -------------------------------------------------------------------
# 3. PARAMETER LABELS (PARSED)
# -------------------------------------------------------------------
param_labels <- c(
  "mu_alpha"  = expression(paste(mu[alpha], " (Evidence Sensitivity)")),
  "mu_beta"   = expression(paste(mu[beta], " (Bias)")),
  "mu_lambda" = expression(paste(mu[lambda], " (Decay)")),
  "mu_delta"  = expression(paste(mu[delta], " (Persistence)")),
  "mu_eta"    = expression(paste(mu[eta], " (Noise/Eta)"))
)

# Using your original Green as the anchor
model_color_aware   <- "#1B5E20" # Original Dark Green
model_color_unaware <- "#81C784" # Lighter Green for contrast
model_color_exp12   <- "#2E7D32" # Medium Green for Exp12

# -------------------------------------------------------------------
# 4. OVERLAY HISTOGRAM PLOT
# -------------------------------------------------------------------
p_overlay <- ggplot(all_data, aes(x = value, fill = Condition)) +
  geom_histogram(
    bins = 80, 
    alpha = 0.6, 
    position = "identity", # This overlays them instead of stacking them
    color = "black", 
    linewidth = 0.1
  ) +
  facet_wrap(
    ~ param, 
    scales = "free", 
    ncol = 3, 
    labeller = as_labeller(param_labels, default = label_parsed)
  ) +
  scale_fill_manual(values = c(
    "Exp11 Aware"   = model_color_aware,
    "Exp11 Unaware" = model_color_unaware,
    "Exp12 Aware"   = model_color_exp12
  )) +
  theme_bw(base_size = 16) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    strip.background = element_rect(fill = "gray90", color = "gray50"),
    strip.text       = element_text(face = "bold", size = 11),
    legend.position  = "bottom"
  ) +
  labs(
    title = "Posterior Overlay: Aware vs. Unaware",
    subtitle = "Boost Learning Group-level parameters",
    x = "Posterior Sample Value (Latent Scale)",
    y = "Frequency"
  )

print(p_overlay)



library(tidyverse)
library(posterior)
library(ggplot2)

# -------------------------------------------------------------------
# 1. Load fits 
# -------------------------------------------------------------------
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_boost_aware_exp12.rdata")
fit_unaware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_boost_aware_exp11.rdata")
fit_aware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_boost_unaware_exp11.rdata")
fit_explicit <- fit

# -------------------------------------------------------------------
# 2. Define Param Mapping (Stan index -> Descriptive name)
# -------------------------------------------------------------------
# If your Stan model uses mu_pr[1] through [5], we map them here:
param_map <- c(
  "mu_pr[1]" = "mu_alpha",
  "mu_pr[2]" = "mu_beta",
  "mu_pr[3]" = "mu_lambda",
  "mu_pr[4]" = "mu_delta",
  "mu_pr[5]" = "mu_eta"
)

# -------------------------------------------------------------------
# 3. Extraction Function
# -------------------------------------------------------------------
extract_mu <- function(fit, label) {
  # Get draws as a data frame
  d <- as_draws_df(fit$draws())
  
  # Find which columns in 'd' match our 'param_map' keys
  # (handles both 'mu_pr[1]' and 'mu_pr.1.' formats)
  actual_cols <- intersect(names(param_map), colnames(d))
  
  # If empty, try the dot format
  if(length(actual_cols) == 0) {
    actual_cols <- colnames(d)[grepl("mu_pr", colnames(d))][1:5]
  }
  
  mu_df <- d %>%
    select(all_of(actual_cols))
  
  # Force set names to ensure they match 'param_labels'
  colnames(mu_df) <- unname(param_map[1:ncol(mu_df)])
  
  mu_df <- mu_df %>%
    mutate(group = label) %>%
    pivot_longer(
      cols = -group, 
      names_to = "param",
      values_to = "value"
    )
  
  return(mu_df)
}

# -------------------------------------------------------------------
# 4. Prepare Data
# -------------------------------------------------------------------
df_all <- bind_rows(
  extract_mu(fit_unaware,  "Implicit Unaware"),
  extract_mu(fit_aware,    "Implicit Aware"),
  extract_mu(fit_explicit, "Explicit Aware")
)

# Crucial: Ensure 'param' is a factor that matches the keys in 'param_labels'
df_all$param <- factor(df_all$param, levels = unname(param_map))

df_all$group <- factor(df_all$group, levels = c(
  "Implicit Aware",
  "Explicit Aware",
  "Implicit Unaware"
))

# -------------------------------------------------------------------
# 5. Plotting Definitions
# -------------------------------------------------------------------
param_labels <- c(
  "mu_alpha"  = "mu[alpha] ~ (Weighting)",
  "mu_beta"   = "mu[beta] ~ (Intercept)",
  "mu_lambda" = "mu[lambda] ~ (Sequential ~ Decay)",
  "mu_delta"  = "mu[delta] ~ (Learning ~ Rate)",
  "mu_eta"    = "mu[eta] ~ (Confirmation ~ Sensitivity)"
)

custom_colors <- c(
  "Explicit Aware"   = "#E69F00",
  "Implicit Aware"   = "#1B5E20",
  "Implicit Unaware" = "#A8D08D"
)

# -------------------------------------------------------------------
# 6. Final Plot
# -------------------------------------------------------------------
p <- ggplot(df_all, aes(x = value, fill = group)) +
  geom_histogram(
    position = "identity",
    bins = 60,
    alpha = 0.55,
    color = "black",
    linewidth = 0.1
  ) +
  # Use label_parsed so the math expressions show up correctly
  facet_wrap(~param, scales = "free", ncol = 3, 
             labeller = as_labeller(param_labels, default = label_parsed)) +
  scale_fill_manual(values = custom_colors) +
  theme_bw(base_size = 14) +
  theme(
    legend.position = "bottom",
    strip.background = element_rect(fill = "gray95"),
    strip.text = element_text(face = "bold")
  ) +
  labs(
    title = "Posterior Parameter Distributions",
    x = "Posterior Sample Value",
    y = "Frequency",
    fill = "Awareness Group"
  )

print(p)






library(tidyverse)
library(posterior)
library(ggplot2)

# -------------------------------------------------------------------
# 1. Load fits 
# -------------------------------------------------------------------
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_boost_aware_exp12.rdata")
fit_explicit<- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_boost_aware_exp11.rdata")
fit_aware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_boost_unaware_exp11.rdata")
fit_unaware <- fit

# -------------------------------------------------------------------
# 2. Define Param Mapping (Stan index -> Descriptive name)
# -------------------------------------------------------------------
param_map <- c(
  "mu_pr[1]" = "mu_alpha",
  "mu_pr[2]" = "mu_beta",
  "mu_pr[3]" = "mu_lambda",
  "mu_pr[4]" = "mu_delta",
  "mu_pr[5]" = "mu_eta"
)

# -------------------------------------------------------------------
# 3. Extraction Function (With Transformations)
# -------------------------------------------------------------------
extract_mu <- function(fit, label) {
  d <- as_draws_df(fit$draws())
  
  # 1. Handle potential naming differences ([1] vs .1.)
  actual_cols <- intersect(names(param_map), colnames(d))
  if(length(actual_cols) == 0) {
    actual_cols <- colnames(d)[grepl("mu_pr", colnames(d))][1:5]
  }
  
  mu_df <- d %>% select(all_of(actual_cols))
  
  # 2. Rename to internal names for easier math
  colnames(mu_df) <- unname(param_map[1:ncol(mu_df)])
  
  # 3. Apply the Stan transformations to get the "Physical" values
  mu_df <- mu_df %>%
    mutate(
      mu_alpha  = pnorm(mu_alpha),       # Squashed 0 to 1
      mu_lambda = pnorm(mu_lambda),      # Squashed 0 to 1
      mu_delta  = pnorm(mu_delta) * 2    # Squashed 0 to 1, then stretched to 2
      # mu_beta and mu_eta remain untransformed per your Stan code
    ) %>%
    mutate(group = label) %>%
    pivot_longer(cols = -group, names_to = "param", values_to = "value")
  
  return(mu_df)
}

# -------------------------------------------------------------------
# 4. Prepare Data
# -------------------------------------------------------------------
df_all <- bind_rows(
  extract_mu(fit_unaware,  "Implicit Unaware"),
  extract_mu(fit_aware,    "Implicit Aware"),
  extract_mu(fit_explicit, "Explicit Aware")
)

# Ensure 'param' matches labels for the facet titles
df_all$param <- factor(df_all$param, levels = unname(param_map))
df_all$group <- factor(df_all$group, levels = c("Implicit Aware", "Explicit Aware", "Implicit Unaware"))

# -------------------------------------------------------------------
# 5. Plotting Definitions
# -------------------------------------------------------------------
param_labels <- c(
  "mu_alpha"  = "mu[alpha]",
  "mu_beta"   = "mu[beta]",
  "mu_lambda" = "mu[lambda]",
  "mu_delta"  = "mu[delta]",
  "mu_eta"    = "mu[eta]"
)

custom_colors <- c(
  "Explicit Aware"   = "#E69F00",
  "Implicit Aware"   = "#1B5E20",
  "Implicit Unaware" = "#A8D08D"
)

# -------------------------------------------------------------------
# 6. Final Plot
# -------------------------------------------------------------------
ggplot(df_all, aes(x = value, fill = group)) +
  geom_histogram(position = "identity", bins = 60, alpha = 0.55, color = "black", linewidth = 0.1) +
  facet_wrap(~param, scales = "free", ncol = 3, 
             labeller = as_labeller(param_labels, default = label_parsed)) +
  scale_fill_manual(values = custom_colors) +
  theme_bw(base_size = 14) +
  theme(legend.position = "bottom") +
  labs(title = "Transformed Posterior Parameter Distributions",
       x = "Parameter Value (Transformed Scale)", y = "Frequency")




library(dplyr)
library(tidyr)

# ---------------------------------------------------------------
# 7. Posterior summaries against optimal values
# ---------------------------------------------------------------
ref_values <- c(mu_delta = 1, mu_eta = 0)

posterior_vs_optimal <- df_plot %>%
  group_by(group, param) %>%
  summarise(
    ref = ref_values[as.character(first(param))],
    mean = mean(value),
    median = median(value),
    l95 = quantile(value, 0.025),
    u95 = quantile(value, 0.975),
    p_above_ref = mean(value > ref),
    p_below_ref = mean(value < ref),
    mean_diff = mean(value - ref),
    median_diff = median(value - ref),
    l95_diff = quantile(value - ref, 0.025),
    u95_diff = quantile(value - ref, 0.975),
    .groups = "drop"
  )

print(posterior_vs_optimal)





library(tidyverse)
library(posterior)
library(ggplot2)

# -------------------------------------------------------------------
# 1. Load fits
# -------------------------------------------------------------------
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_boost_model_aware_exp12.rdata")
fit_explicit <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_boost_model_aware_exp11.rdata")
fit_aware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_boost_model_unaware_exp11.rdata")
fit_unaware <- fit

# -------------------------------------------------------------------
# 2. Extraction function: keep only delta and eta
# -------------------------------------------------------------------
extract_delta_eta <- function(fit, label) {
  d <- as_draws_df(fit$draws())
  
  if (all(c("mu_delta", "mu_eta") %in% colnames(d))) {
    out <- d %>%
      select(mu_delta, mu_eta)
  } else {
    mu_cols <- colnames(d)[grepl("^mu_pr(\\.|\\[)", colnames(d))]
    if (length(mu_cols) < 5) {
      stop("Could not find enough mu_pr columns to extract mu_delta and mu_eta.")
    }
    
    out <- d %>%
      select(all_of(mu_cols[c(4, 5)]))
    
    colnames(out) <- c("mu_delta", "mu_eta")
    
    out <- out %>%
      mutate(
        mu_delta = pnorm(mu_delta) * 2.0
      )
  }
  
  out %>%
    mutate(group = label) %>%
    pivot_longer(
      cols = c(mu_delta, mu_eta),
      names_to = "param",
      values_to = "value"
    )
}

# -------------------------------------------------------------------
# 3. Prepare data
# -------------------------------------------------------------------
df_plot <- bind_rows(
  extract_delta_eta(fit_unaware,  "Implicit Unaware"),
  extract_delta_eta(fit_aware,    "Implicit Aware"),
  extract_delta_eta(fit_explicit, "Explicit Aware")
)

df_plot$group <- factor(
  df_plot$group,
  levels = c("Implicit Aware", "Explicit Aware", "Implicit Unaware")
)

df_plot$param <- factor(
  df_plot$param,
  levels = c("mu_delta", "mu_eta")
)

# -------------------------------------------------------------------
# 4. Colours
# -------------------------------------------------------------------
custom_colors <- c(
  "Explicit Aware"   = "#E69F00",
  "Implicit Aware"   = "#1B5E20",
  "Implicit Unaware" = "#A8D08D"
)

# -------------------------------------------------------------------
# 5. Reference lines
# -------------------------------------------------------------------
ref_lines <- tibble(
  param = c("mu_delta", "mu_eta"),
  xint  = c(1, 0),
  label = "Optimal Bayesian\nobserver"
)

panel_titles <- tibble(
  param = c("mu_delta", "mu_eta"),
  x = c(Inf, Inf),
  y = c(Inf, Inf),
  label = c("Delta", "Eta")
)

ggplot(df_plot, aes(x = value, fill = group)) +
  geom_histogram(
    position = "identity",
    bins = 60,
    alpha = 0.55,
    color = "black",
    linewidth = 0.1
  ) +
  geom_vline(
    data = ref_lines,
    aes(xintercept = xint),
    color = "red",
    linewidth = 1,
    inherit.aes = FALSE
  ) +
  geom_text(
    data = ref_lines,
    aes(x = xint, y = 0, label = label),
    color = "red",
    vjust = 2.8,
    hjust = 0.5,
    size = 4,
    inherit.aes = FALSE
  ) +
  geom_text(
    data = panel_titles,
    aes(x = x, y = y, label = label),
    hjust = 1.1,
    vjust = 1.5,
    size = 5,
    fontface = "bold",
    inherit.aes = FALSE
  ) +
  facet_wrap(
    ~param,
    scales = "free",
    ncol = 1
  ) +
  scale_fill_manual(values = custom_colors) +
  scale_y_continuous(name = "Frequency") +
  coord_cartesian(clip = "off") +
  labs(
    x = "Parameter value"
  ) +
  theme_bw(base_size = 14) +
  theme(
    strip.background = element_blank(),
    strip.text = element_blank(),
    
    legend.position = "inside",
    legend.position.inside = c(0.05, 0.18),
    legend.justification = c(0, 0),
    legend.background = element_rect(fill = scales::alpha("white", 0.75), color = NA),
    
    plot.margin = margin(10, 10, 20, 10),
    
    axis.title = element_text(size = 20),
    axis.text  = element_text(size = 14)
  )
















library(tidyverse)
library(posterior)
library(ggplot2)

# -------------------------------------------------------------------
# 1. Load fits 
# -------------------------------------------------------------------
# Ensure these files contain the 5-parameter (alpha, beta, lambda, delta, eta) model
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_boost_model_aware_exp12.rdata")
fit_explicit <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_boost_model_aware_exp11.rdata")
fit_aware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_boost_model_unaware_exp11.rdata")
fit_unaware <- fit

# -------------------------------------------------------------------
# 2. Define Param Mapping (Updated for 5 parameters)
# -------------------------------------------------------------------
param_map <- c(
  "mu_pr[1]" = "mu_alpha",
  "mu_pr[2]" = "mu_beta",
  "mu_pr[3]" = "mu_lambda",
  "mu_pr[4]" = "mu_delta",
  "mu_pr[5]" = "mu_eta"
)

# -------------------------------------------------------------------
# 3. Extraction Function
# -------------------------------------------------------------------
extract_mu <- function(fit, label, alpha_scale = 6) {
  d <- as_draws_df(fit$draws())
  
  # 1. Check if generated quantities already exist (mu_alpha, etc.)
  gen_quant_names <- c("mu_alpha", "mu_beta", "mu_lambda", "mu_delta", "mu_eta")
  
  if (all(gen_quant_names %in% colnames(d))) {
    mu_df <- d %>% select(all_of(gen_quant_names))
  } else {
    # 2. Otherwise pull mu_pr[1:5] and transform manually
    actual_cols <- intersect(names(param_map), colnames(d))
    
    # Handle Stan naming variations like mu_pr.1. vs mu_pr[1]
    if (length(actual_cols) == 0) {
      mu_cols <- colnames(d)[grepl("^mu_pr(\\.|\\[)", colnames(d))]
      actual_cols <- mu_cols[1:5]
    }
    
    mu_df <- d %>% select(all_of(actual_cols))
    colnames(mu_df) <- unname(param_map[1:ncol(mu_df)])
    
    # 3. Apply Transformations to match your Stan model
    mu_df <- mu_df %>%
      mutate(
        mu_alpha  = pnorm(mu_alpha) * alpha_scale, # Sensitivity [0, 6]
        mu_lambda = pnorm(mu_lambda),               # Decay [0, 1]
        mu_delta  = pnorm(mu_delta) * 2.0           # Memory [0, 2]
        # mu_beta and mu_eta stay untransformed (intercepts)
      )
  }
  
  mu_df %>%
    mutate(group = label) %>%
    pivot_longer(cols = -group, names_to = "param", values_to = "value")
}

# -------------------------------------------------------------------
# 4. Prepare Data
# -------------------------------------------------------------------
df_all <- bind_rows(
  extract_mu(fit_unaware,  "Implicit Unaware", alpha_scale = 6),
  extract_mu(fit_aware,    "Implicit Aware",   alpha_scale = 6),
  extract_mu(fit_explicit, "Explicit Aware",   alpha_scale = 6)
)

# Set factor levels for clean plotting
df_all$param <- factor(df_all$param, levels = unname(param_map))
df_all$group <- factor(df_all$group, levels = c("Implicit Aware", "Explicit Aware", "Implicit Unaware"))

# -------------------------------------------------------------------
# 5. Plot labels + colours
# -------------------------------------------------------------------
param_labels <- c(
  "mu_alpha"  = "mu[alpha]",
  "mu_beta"   = "mu[beta]",
  "mu_lambda" = "mu[lambda]",
  "mu_delta"  = "mu[delta]",
  "mu_eta"    = "mu[eta]"
)

custom_colors <- c(
  "Explicit Aware"   = "#E69F00",
  "Implicit Aware"   = "#1B5E20",
  "Implicit Unaware" = "#A8D08D"
)

# -------------------------------------------------------------------
# 6. Final Plot
# -------------------------------------------------------------------
ggplot(df_all, aes(x = value, fill = group)) +
  geom_histogram(position = "identity", bins = 60, alpha = 0.55, color = "black", linewidth = 0.1) +
  facet_wrap(~param, scales = "free", ncol = 2,
             labeller = as_labeller(param_labels, default = label_parsed)) +
  scale_fill_manual(values = custom_colors) +
  theme_bw(base_size = 14) +
  theme(legend.position = "bottom",
        strip.text = element_text(face = "bold")) +
  labs(title = "Transformed Posterior Parameter Distributions",
       subtitle = "5-Parameter Model: Includes delta and Confirmation Bias (eta)",
       x = "Parameter Value (Transformed Scale)",
       y = "Frequency")
# -------------------------------------------------------------------
# 7. Function: Extract group-level posterior draws (WIDE)
# -------------------------------------------------------------------
extract_mu_draws <- function(fit, alpha_scale = 6) {
  d <- as_draws_df(fit$draws())
  
  keep_params <- c("mu_alpha", "mu_beta", "mu_lambda", "mu_delta", "mu_eta")
  
  # Prefer generated quantities if present
  if (all(keep_params %in% colnames(d))) {
    out <- d %>% select(all_of(keep_params))
  } else {
    # Otherwise pull mu_pr[1:5] (handles mu_pr[1] or mu_pr.1.)
    mu_cols <- colnames(d)[grepl("^mu_pr(\\.|\\[)", colnames(d))]
    if (length(mu_cols) < 5) stop("No mu_* GQs found and <5 mu_pr columns found in fit draws().")
    
    out <- d %>% select(all_of(mu_cols[1:5]))
    colnames(out) <- keep_params
    
    # Apply same transforms as your Stan model
    out <- out %>%
      mutate(
        mu_alpha  = pnorm(mu_alpha) * alpha_scale,  # [0, 6]
        mu_lambda = pnorm(mu_lambda),               # [0, 1]
        mu_delta  = pnorm(mu_delta) * 2.0           # [0, 2]
        # mu_beta and mu_eta remain untransformed
      )
  }
  
  out
}

draws_unaware  <- extract_mu_draws(fit_unaware,  alpha_scale = 6)
draws_aware    <- extract_mu_draws(fit_aware,    alpha_scale = 6)
draws_explicit <- extract_mu_draws(fit_explicit, alpha_scale = 6)

# -------------------------------------------------------------------
# 8. Identify common parameters across all fits
# -------------------------------------------------------------------
mu_names <- Reduce(intersect, list(colnames(draws_explicit),
                                   colnames(draws_aware),
                                   colnames(draws_unaware)))
cat("Parameters used for Bayesian comparisons:\n")
print(mu_names)

# -------------------------------------------------------------------
# 9. Bayesian exceedance probability function (same logic as your template)
# -------------------------------------------------------------------
calculate_exceedance <- function(draws_X, draws_Y, param_name) {
  x <- draws_X[[param_name]]
  y <- draws_Y[[param_name]]
  
  # Make robust if draw lengths differ
  n <- min(length(x), length(y))
  if (length(x) != n) x <- sample(x, n, replace = TRUE)
  if (length(y) != n) y <- sample(y, n, replace = TRUE)
  
  diff <- x - y
  exceedP_XY <- mean(diff > 0)          # P(Group1 > Group2)
  mean_diff  <- mean(diff)              # mean difference
  SE <- if (mean_diff > 0) mean(diff < 0) else exceedP_XY  # smaller tail prob
  
  list(ExceedP_XY = exceedP_XY, SE_value = SE, Mean_Diff = mean_diff)
}

# -------------------------------------------------------------------
# 10. Define ordered group comparisons
# -------------------------------------------------------------------
group_pairs <- list(
  list("Explicit Aware",  draws_explicit, "Implicit Aware",   draws_aware),
  list("Explicit Aware",  draws_explicit, "Implicit Unaware", draws_unaware),
  list("Implicit Aware",  draws_aware,    "Implicit Unaware", draws_unaware)
)

# -------------------------------------------------------------------
# 11. Run exceedance probability comparisons
# -------------------------------------------------------------------
results <- list()
k <- 1

for (p in mu_names) {
  for (pair in group_pairs) {
    group1 <- pair[[1]]; draws1 <- pair[[2]]
    group2 <- pair[[3]]; draws2 <- pair[[4]]
    
    res <- calculate_exceedance(draws1, draws2, p)
    
    results[[k]] <- tibble(
      Parameter = p,
      Group1 = group1,
      Group2 = group2,
      Mean1 = mean(draws1[[p]]),
      Mean2 = mean(draws2[[p]]),
      Mean_Diff = res$Mean_Diff,
      P_Exceed_G1_G2 = res$ExceedP_XY,
      P_Exceed_Smaller = res$SE_value
    )
    k <- k + 1
  }
}

results_df <- bind_rows(results) %>%
  mutate(
    Significance = P_Exceed_Smaller < 0.05,
    Comparison = paste(Group1, ">", Group2)
  ) %>%
  select(Parameter, Comparison, Mean1, Mean2, Mean_Diff,
         P_Exceed_G1_G2, P_Exceed_Smaller, Significance)

# -------------------------------------------------------------------
# 12. Print results
# -------------------------------------------------------------------
cat("\n==============================================================\n")
cat("BAYESIAN EXCEEDANCE PROBABILITY TEST RESULTS\n")
cat("Model: TRUNC Simplified BOOST (mu_alpha, mu_beta, mu_lambda, mu_delta, mu_eta)\n")
cat("Ordering: Explicit Aware > Implicit Aware > Implicit Unaware\n")
cat("==============================================================\n")
print(results_df, n = Inf)



# -------------------------------------------------------------------
# 6. Split plots: (alpha+beta), (lambda), (eta+delta)
# -------------------------------------------------------------------

# Helper to keep consistent styling
base_hist <- function(dat, ncol = 2, title = NULL, subtitle = NULL) {
  ggplot(dat, aes(x = value, fill = group)) +
    geom_histogram(position = "identity", bins = 60, alpha = 0.55,
                   color = "black", linewidth = 0.1) +
    facet_wrap(~param, scales = "free", ncol = ncol,
               labeller = as_labeller(param_labels, default = label_parsed)) +
    scale_fill_manual(values = custom_colors) +
    theme_bw(base_size = 14) +
    theme(
      legend.position = "bottom",
      strip.text = element_text(face = "bold")
    ) +
    labs(
      title = title,
      subtitle = subtitle,
      x = "Parameter Value ",
      y = "Frequency"
    )
}

# ---- Plot 1: alpha + beta ----
df_ab <- df_all %>% filter(param %in% c("mu_alpha", "mu_beta"))
p_ab <- base_hist(
  df_ab, ncol = 2,
  title = "Posterior Distributions: Sensitivity & Intercept",
  subtitle = "α and β"
)

# ---- Plot 2: lambda only ----
df_lam <- df_all %>% filter(param %in% c("mu_lambda"))
p_lam <- base_hist(
  df_lam, ncol = 1,
  title = "Posterior Distributions: Within-trial Recency",
  subtitle = "λ"
)

# ---- Plot 3: eta + delta ----
df_ed <- df_all %>% filter(param %in% c("mu_eta", "mu_delta"))
p_ed <- base_hist(
  df_ed, ncol = 2,
  title = "Posterior Distributions: Confirmation Bias & Learning",
  subtitle = "η and δ"
)

# Print them (one after the other)
p_ab
p_lam
p_ed
ggsave("posterior_alpha_beta.png", p_ab, width = 10, height = 6, dpi = 300)
ggsave("posterior_lambda.png",     p_lam, width = 8,  height = 6, dpi = 300)
ggsave("posterior_eta_delta.png",  p_ed, width = 10, height = 6, dpi = 300)



# ===============================================================
# FULL SCRIPT: Posterior Distributions + Bayesian Exceedance
#              (3-Param TRUNC SIMPLIFIED MODEL)
# ===============================================================

# -------------------------------------------------------------------
# 0. Load libraries
# -------------------------------------------------------------------
library(tidyverse)
library(posterior)
library(ggplot2)

# -------------------------------------------------------------------
# 1. Load fits (3-parameter model)
# -------------------------------------------------------------------
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_model_unaware_exp11.rdata")
fit_unaware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_model_aware_exp11.rdata")
fit_aware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_model_aware_exp12.rdata")
fit_explicit <- fit

# -------------------------------------------------------------------
# 2. Function: Extract transformed mu_ parameters for plotting
#     - Prefers generated quantities: mu_alpha, mu_beta, mu_lambda
#     - Falls back to mu_pr[1:3] and applies same transforms as Stan:
#         mu_alpha  = pnorm(mu_pr1) * 6
#         mu_lambda = pnorm(mu_pr3)
#         mu_beta raw
# -------------------------------------------------------------------
extract_mu <- function(fit, label, alpha_scale = 6) {
  d <- as_draws_df(fit$draws())
  
  if (all(c("mu_alpha","mu_beta","mu_lambda") %in% colnames(d))) {
    mu_df <- d %>% select(mu_alpha, mu_beta, mu_lambda)
  } else {
    # fallback to mu_pr[1:3] or mu_pr.1. style
    mu_cols <- colnames(d)[grepl("^mu_pr(\\.|\\[)", colnames(d))]
    if (length(mu_cols) < 3) stop("No mu_alpha/beta/lambda or mu_pr[1:3] found in fit draws()")
    
    mu_df <- d %>% select(all_of(mu_cols[1:3]))
    colnames(mu_df) <- c("mu_alpha", "mu_beta", "mu_lambda")
    
    mu_df <- mu_df %>%
      mutate(
        mu_alpha  = pnorm(mu_alpha) * alpha_scale,
        mu_lambda = pnorm(mu_lambda)
        # mu_beta stays raw
      )
  }
  
  mu_df %>%
    mutate(group = label) %>%
    pivot_longer(cols = c(mu_alpha, mu_beta, mu_lambda),
                 names_to = "param",
                 values_to = "value")
}

# -------------------------------------------------------------------
# 3. Extract draws for plotting
# -------------------------------------------------------------------
df_unaware  <- extract_mu(fit_unaware,  "Implicit Unaware", alpha_scale = 6)
df_aware    <- extract_mu(fit_aware,    "Implicit Aware",   alpha_scale = 6)
df_explicit <- extract_mu(fit_explicit, "Explicit Aware",   alpha_scale = 6)

df_all <- bind_rows(df_unaware, df_aware, df_explicit)

# Set factor order for plotting
df_all$group <- factor(df_all$group, levels = c(
  "Implicit Unaware",
  "Implicit Aware",
  "Explicit Aware"
))

# -------------------------------------------------------------------
# 4. Plot posterior distributions
# -------------------------------------------------------------------
param_labels <- c(
  "mu_alpha"  = expression(mu[alpha]),
  "mu_beta"   = expression(mu[beta]),
  "mu_lambda" = expression(mu[lambda])
)

custom_colors <- c(
  "Explicit Aware"   = "#E69F00",
  "Implicit Aware"   = "#1B5E20",
  "Implicit Unaware" = "#A8D08D"
)

ggplot(df_all, aes(x = value, fill = group)) +
  geom_histogram(position = "identity", bins = 60, alpha = 0.55,
                 color = "black", linewidth = 0.1) +
  facet_wrap(~param, scales = "free", ncol = 3,
             labeller = as_labeller(param_labels, default = label_parsed)) +
  scale_fill_manual(values = custom_colors) +
  theme_bw(base_size = 14) +
  theme(legend.position = "bottom",
        strip.background = element_blank(),
        strip.text = element_text(face = "bold")) +
  labs(title = "Posterior Parameter Distributions (3-Param Simplified Model)",
       subtitle = "Alpha (Sensitivity), Beta (Additive Bias), Lambda (Recency)",
       x = "Parameter Value (Transformed Scale)",
       y = "Frequency")

# -------------------------------------------------------------------
# 5. Function: Extract group-level posterior draws (WIDE)
#     (same transform logic as above)
# -------------------------------------------------------------------
extract_mu_draws <- function(fit, alpha_scale = 6) {
  d <- as_draws_df(fit$draws())
  
  if (all(c("mu_alpha","mu_beta","mu_lambda") %in% colnames(d))) {
    out <- d %>% select(mu_alpha, mu_beta, mu_lambda)
  } else {
    mu_cols <- colnames(d)[grepl("^mu_pr(\\.|\\[)", colnames(d))]
    if (length(mu_cols) < 3) stop("No mu_alpha/beta/lambda or mu_pr[1:3] found in fit draws()")
    
    out <- d %>% select(all_of(mu_cols[1:3]))
    colnames(out) <- c("mu_alpha", "mu_beta", "mu_lambda")
    
    out <- out %>%
      mutate(
        mu_alpha  = pnorm(mu_alpha) * alpha_scale,
        mu_lambda = pnorm(mu_lambda)
      )
  }
  out
}

draws_unaware  <- extract_mu_draws(fit_unaware,  alpha_scale = 6)
draws_aware    <- extract_mu_draws(fit_aware,    alpha_scale = 6)
draws_explicit <- extract_mu_draws(fit_explicit, alpha_scale = 6)

# -------------------------------------------------------------------
# 6. Identify common parameters across all fits
# -------------------------------------------------------------------
mu_names <- Reduce(intersect, list(colnames(draws_explicit),
                                   colnames(draws_aware),
                                   colnames(draws_unaware)))
cat("Parameters used for Bayesian comparisons:\n")
print(mu_names)

# -------------------------------------------------------------------
# 7. Bayesian exceedance probability function
#     - Matches your learning script:
#         P_Exceed_G1_G2 = mean(diff > 0)
#         P_Exceed_Smaller = tail probability on the opposite side
#         Mean_Diff = mean(diff)
# -------------------------------------------------------------------
calculate_exceedance <- function(draws_X, draws_Y, param_name) {
  # robust to unequal draw counts (just in case)
  x <- draws_X[[param_name]]
  y <- draws_Y[[param_name]]
  n <- min(length(x), length(y))
  if (length(x) != n) x <- sample(x, n, replace = TRUE)
  if (length(y) != n) y <- sample(y, n, replace = TRUE)
  
  diff <- x - y
  exceedP_XY <- mean(diff > 0)          # Posterior probability Group1 > Group2
  mean_diff  <- mean(diff)              # Posterior mean difference
  SE <- if (mean_diff > 0) mean(diff < 0) else exceedP_XY  # your "smaller tail"
  
  list(ExceedP_XY = exceedP_XY, SE_value = SE, Mean_Diff = mean_diff)
}

# -------------------------------------------------------------------
# 8. Define ordered group comparisons (same ordering as your script)
# -------------------------------------------------------------------
group_pairs <- list(
  list("Explicit Aware",  draws_explicit, "Implicit Aware",   draws_aware),
  list("Explicit Aware",  draws_explicit, "Implicit Unaware", draws_unaware),
  list("Implicit Aware",  draws_aware,    "Implicit Unaware", draws_unaware)
)

# -------------------------------------------------------------------
# 9. Run exceedance probability comparisons
# -------------------------------------------------------------------
results <- list()
k <- 1

for (p in mu_names) {
  for (pair in group_pairs) {
    group1 <- pair[[1]]; draws1 <- pair[[2]]
    group2 <- pair[[3]]; draws2 <- pair[[4]]
    
    res <- calculate_exceedance(draws1, draws2, p)
    
    results[[k]] <- tibble(
      Parameter = p,
      Group1 = group1,
      Group2 = group2,
      Mean1 = mean(draws1[[p]]),
      Mean2 = mean(draws2[[p]]),
      Mean_Diff = res$Mean_Diff,
      P_Exceed_G1_G2 = res$ExceedP_XY,
      P_Exceed_Smaller = res$SE_value
    )
    k <- k + 1
  }
}

results_df <- bind_rows(results) %>%
  mutate(
    Significance = P_Exceed_Smaller < 0.05,
    Comparison = paste(Group1, ">", Group2)
  ) %>%
  select(Parameter, Comparison, Mean1, Mean2, Mean_Diff,
         P_Exceed_G1_G2, P_Exceed_Smaller, Significance)

# -------------------------------------------------------------------
# 10. Print results
# -------------------------------------------------------------------
cat("\n==============================================================\n")
cat("BAYESIAN EXCEEDANCE PROBABILITY TEST RESULTS\n")
cat("Ordering: Explicit Aware > Implicit Aware > Implicit Unaware\n")
cat("Model: TRUNC Simplified (mu_alpha, mu_beta, mu_lambda)\n")
cat("==============================================================\n")
print(results_df, n = Inf)










# ===============================================================
# FULL SCRIPT: Posterior Distributions + Bayesian Exceedance
# ===============================================================

# -------------------------------------------------------------------
# 0. Load libraries
# -------------------------------------------------------------------
library(tidyverse)
library(posterior)
library(ggplot2)

# -------------------------------------------------------------------
# 1. Load new fits
# -------------------------------------------------------------------
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_unaware_exp11.rdata")
fit_unaware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_aware_exp11.rdata")
fit_aware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_aware_exp12.rdata")
fit_explicit <- fit

# -------------------------------------------------------------------
# 2. Function: Extract transformed mu_ parameters for plotting
# -------------------------------------------------------------------
extract_mu <- function(fit, label) {
  d <- as_draws_df(fit$draws())
  
  keep_params <- c("mu_alpha", "mu_beta", "mu_lambda", "mu_delta")
  keep_params <- keep_params[keep_params %in% colnames(d)]
  
  mu_df <- d %>%
    select(all_of(keep_params)) %>%
    mutate(group = label) %>%
    pivot_longer(cols = all_of(keep_params),
                 names_to = "param",
                 values_to = "value")
  
  return(mu_df)
}

# -------------------------------------------------------------------
# 3. Extract draws for plotting
# -------------------------------------------------------------------
df_unaware  <- extract_mu(fit_unaware,  "Implicit Unaware")
df_aware    <- extract_mu(fit_aware,    "Implicit Aware")
df_explicit <- extract_mu(fit_explicit, "Explicit Aware")

df_all <- bind_rows(df_unaware, df_aware, df_explicit)

# Set factor order for plotting
df_all$group <- factor(df_all$group, levels = c(
  "Implicit Unaware",
  "Implicit Aware",
  "Explicit Aware"
))

# -------------------------------------------------------------------
# 4. Plot posterior distributions
# -------------------------------------------------------------------
param_labels <- c(
  "mu_alpha"  = expression(mu[alpha]),
  "mu_beta"   = expression(mu[beta]),
  "mu_lambda" = expression(mu[lambda]),
  "mu_delta"  = expression(mu[delta])
)

custom_colors <- c(
  "Explicit Aware"   = "#E69F00",
  "Implicit Aware"   = "#1B5E20",
  "Implicit Unaware" = "#A8D08D"
)

ggplot(df_all, aes(x = value, fill = group)) +
  geom_histogram(position = "identity", bins = 60, alpha = 0.55, color = "black", linewidth = 0.1) +
  facet_wrap(~param, scales = "free", ncol = 2, labeller = as_labeller(param_labels, default = label_parsed)) +
  scale_fill_manual(values = custom_colors) +
  theme_bw(base_size = 14) +
  theme(legend.position = "bottom",
        strip.background = element_blank(),
        strip.text = element_text(face = "bold")) +
  labs(title = "Posterior Parameter Distributions (4-Param Learning Model)",
       x = "Parameter Value",
       y = "Frequency")

# -------------------------------------------------------------------
# 5. Function: Extract group-level posterior draws
# -------------------------------------------------------------------
extract_mu_draws <- function(fit) {
  d <- as_draws_df(fit$draws())
  keep_params <- c("mu_alpha", "mu_beta", "mu_lambda", "mu_delta")
  keep_params <- keep_params[keep_params %in% colnames(d)]
  if(length(keep_params) == 0) stop("No mu_* parameters found in fit")
  d %>% select(all_of(keep_params))
}

draws_unaware  <- extract_mu_draws(fit_unaware)
draws_aware    <- extract_mu_draws(fit_aware)
draws_explicit <- extract_mu_draws(fit_explicit)

# -------------------------------------------------------------------
# 6. Identify common parameters across all models
# -------------------------------------------------------------------
mu_names <- Reduce(intersect, list(colnames(draws_explicit), colnames(draws_aware), colnames(draws_unaware)))
cat("Parameters used for Bayesian comparisons:\n")
print(mu_names)

# -------------------------------------------------------------------
# 7. Bayesian exceedance probability function
# -------------------------------------------------------------------
calculate_exceedance <- function(draws_X, draws_Y, param_name) {
  diff <- draws_X[[param_name]] - draws_Y[[param_name]]
  exceedP_XY <- mean(diff > 0)          # Posterior probability Group1 > Group2
  mean_diff  <- mean(diff)              # Posterior mean difference
  SE <- if(mean_diff > 0) mean(diff < 0) else exceedP_XY
  list(ExceedP_XY = exceedP_XY, SE_value = SE, Mean_Diff = mean_diff)
}

# -------------------------------------------------------------------
# 8. Define ordered group comparisons
# -------------------------------------------------------------------
group_pairs <- list(
  list("Explicit Aware",  draws_explicit, "Implicit Aware",   draws_aware),
  list("Explicit Aware",  draws_explicit, "Implicit Unaware", draws_unaware),
  list("Implicit Aware",  draws_aware,    "Implicit Unaware", draws_unaware)
)

# -------------------------------------------------------------------
# 9. Run exceedance probability comparisons
# -------------------------------------------------------------------
results <- list()
k <- 1

for(p in mu_names) {
  for(pair in group_pairs) {
    group1 <- pair[[1]]; draws1 <- pair[[2]]
    group2 <- pair[[3]]; draws2 <- pair[[4]]
    
    res <- calculate_exceedance(draws1, draws2, p)
    
    results[[k]] <- tibble(
      Parameter = p,
      Group1 = group1,
      Group2 = group2,
      Mean1 = mean(draws1[[p]]),
      Mean2 = mean(draws2[[p]]),
      Mean_Diff = res$Mean_Diff,
      P_Exceed_G1_G2 = res$ExceedP_XY,
      P_Exceed_Smaller = res$SE_value
    )
    k <- k + 1
  }
}

results_df <- bind_rows(results) %>%
  mutate(
    Significance = P_Exceed_Smaller < 0.05,
    Comparison = paste(Group1, ">", Group2)
  ) %>%
  select(Parameter, Comparison, Mean1, Mean2, Mean_Diff, P_Exceed_G1_G2, P_Exceed_Smaller, Significance)

# -------------------------------------------------------------------
# 10. Print results
# -------------------------------------------------------------------
cat("\n==============================================================\n")
cat("BAYESIAN EXCEEDANCE PROBABILITY TEST RESULTS\n")
cat("Ordering: Explicit Aware > Implicit Aware > Implicit Unaware\n")
cat("==============================================================\n")
print(results_df, n = Inf)




# ===============================================================
# FULL SCRIPT: Posterior Distributions + Bayesian Exceedance
# 4-PARAM MODEL: alpha, beta, lambda, eta
# DELTA REMOVED
# ===============================================================

# -------------------------------------------------------------------
# 0. Load libraries
# -------------------------------------------------------------------
library(tidyverse)
library(posterior)
library(ggplot2)

# -------------------------------------------------------------------
# 1. Load new fits
# -------------------------------------------------------------------
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_eta_model_unaware_exp11.rdata")
fit_unaware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_eta_model_aware_exp11.rdata")
fit_aware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_eta_model_aware_exp12.rdata")
fit_explicit <- fit

# -------------------------------------------------------------------
# 2. Function: Extract transformed mu_ parameters for plotting
# -------------------------------------------------------------------
extract_mu <- function(fit, label) {
  d <- as_draws_df(fit$draws())
  
  keep_params <- c("mu_alpha", "mu_beta", "mu_lambda", "mu_eta")
  keep_params <- keep_params[keep_params %in% colnames(d)]
  
  if (length(keep_params) == 0) {
    stop(paste("No matching mu_* parameters found in fit for group:", label))
  }
  
  mu_df <- d %>%
    select(all_of(keep_params)) %>%
    mutate(group = label) %>%
    pivot_longer(
      cols = all_of(keep_params),
      names_to = "param",
      values_to = "value"
    )
  
  return(mu_df)
}

# -------------------------------------------------------------------
# 3. Extract draws for plotting
# -------------------------------------------------------------------
df_unaware  <- extract_mu(fit_unaware,  "Implicit Unaware")
df_aware    <- extract_mu(fit_aware,    "Implicit Aware")
df_explicit <- extract_mu(fit_explicit, "Explicit Aware")

df_all <- bind_rows(df_unaware, df_aware, df_explicit)

# Set factor order for plotting
df_all$group <- factor(df_all$group, levels = c(
  "Implicit Unaware",
  "Implicit Aware",
  "Explicit Aware"
))

# Set parameter order for plotting
df_all$param <- factor(df_all$param, levels = c(
  "mu_alpha",
  "mu_beta",
  "mu_lambda",
  "mu_eta"
))

# -------------------------------------------------------------------
# 4. Plot posterior distributions
# -------------------------------------------------------------------
param_labels <- c(
  "mu_alpha"  = expression(mu[alpha]),
  "mu_beta"   = expression(mu[beta]),
  "mu_lambda" = expression(mu[lambda]),
  "mu_eta"    = expression(mu[eta])
)

custom_colors <- c(
  "Explicit Aware"   = "#E69F00",
  "Implicit Aware"   = "#1B5E20",
  "Implicit Unaware" = "#A8D08D"
)

ggplot(df_all, aes(x = value, fill = group)) +
  geom_histogram(
    position = "identity",
    bins = 60,
    alpha = 0.55,
    color = "black",
    linewidth = 0.1
  ) +
  facet_wrap(
    ~param,
    scales = "free",
    ncol = 2,
    labeller = as_labeller(param_labels, default = label_parsed)
  ) +
  scale_fill_manual(values = custom_colors) +
  theme_bw(base_size = 14) +
  theme(
    legend.position = "bottom",
    strip.background = element_blank(),
    strip.text = element_text(face = "bold")
  ) +
  labs(
    title = "Posterior Parameter Distributions (4-Param Model: Delta set to 1)",
    x = "Parameter Value",
    y = "Frequency",
    fill = "Group"
  )

# -------------------------------------------------------------------
# 5. Function: Extract group-level posterior draws
# -------------------------------------------------------------------
extract_mu_draws <- function(fit, label = NULL) {
  d <- as_draws_df(fit$draws())
  
  keep_params <- c("mu_alpha", "mu_beta", "mu_lambda", "mu_eta")
  keep_params <- keep_params[keep_params %in% colnames(d)]
  
  if (length(keep_params) == 0) {
    stop(paste("No matching mu_* parameters found in fit:", label))
  }
  
  d %>% select(all_of(keep_params))
}

draws_unaware  <- extract_mu_draws(fit_unaware,  "Implicit Unaware")
draws_aware    <- extract_mu_draws(fit_aware,    "Implicit Aware")
draws_explicit <- extract_mu_draws(fit_explicit, "Explicit Aware")

# -------------------------------------------------------------------
# 6. Identify common parameters across all models
# -------------------------------------------------------------------
mu_names <- Reduce(intersect, list(
  colnames(draws_explicit),
  colnames(draws_aware),
  colnames(draws_unaware)
))

cat("Parameters used for Bayesian comparisons:\n")
print(mu_names)

# -------------------------------------------------------------------
# 7. Bayesian exceedance probability function
# -------------------------------------------------------------------
calculate_exceedance <- function(draws_X, draws_Y, param_name) {
  diff <- draws_X[[param_name]] - draws_Y[[param_name]]
  
  exceedP_XY <- mean(diff > 0)          # Posterior probability Group1 > Group2
  mean_diff  <- mean(diff)              # Posterior mean difference
  
  # Smaller tail probability in direction opposite the observed mean difference
  SE <- if (mean_diff > 0) {
    mean(diff < 0)
  } else {
    mean(diff > 0)
  }
  
  list(
    ExceedP_XY = exceedP_XY,
    SE_value = SE,
    Mean_Diff = mean_diff
  )
}

# -------------------------------------------------------------------
# 8. Define ordered group comparisons
# -------------------------------------------------------------------
group_pairs <- list(
  list("Explicit Aware",  draws_explicit, "Implicit Aware",   draws_aware),
  list("Explicit Aware",  draws_explicit, "Implicit Unaware", draws_unaware),
  list("Implicit Aware",  draws_aware,    "Implicit Unaware", draws_unaware)
)

# -------------------------------------------------------------------
# 9. Run exceedance probability comparisons
# -------------------------------------------------------------------
results <- list()
k <- 1

for (p in mu_names) {
  for (pair in group_pairs) {
    group1 <- pair[[1]]
    draws1 <- pair[[2]]
    group2 <- pair[[3]]
    draws2 <- pair[[4]]
    
    res <- calculate_exceedance(draws1, draws2, p)
    
    results[[k]] <- tibble(
      Parameter = p,
      Group1 = group1,
      Group2 = group2,
      Mean1 = mean(draws1[[p]]),
      Mean2 = mean(draws2[[p]]),
      Mean_Diff = res$Mean_Diff,
      P_Exceed_G1_G2 = res$ExceedP_XY,
      P_Exceed_Smaller = res$SE_value
    )
    
    k <- k + 1
  }
}

results_df <- bind_rows(results) %>%
  mutate(
    Significance = P_Exceed_Smaller < 0.05,
    Comparison = paste(Group1, ">", Group2),
    Parameter_Label = case_when(
      Parameter == "mu_alpha"  ~ "mu_alpha",
      Parameter == "mu_beta"   ~ "mu_beta",
      Parameter == "mu_lambda" ~ "mu_lambda",
      Parameter == "mu_eta"    ~ "mu_eta",
      TRUE ~ Parameter
    )
  ) %>%
  select(
    Parameter,
    Comparison,
    Mean1,
    Mean2,
    Mean_Diff,
    P_Exceed_G1_G2,
    P_Exceed_Smaller,
    Significance
  )

# -------------------------------------------------------------------
# 10. Print results
# -------------------------------------------------------------------
cat("\n==============================================================\n")
cat("BAYESIAN EXCEEDANCE PROBABILITY TEST RESULTS\n")
cat("4-PARAM MODEL: alpha, beta, lambda, eta\n")
cat("Delta removed\n")
cat("Ordering: Explicit Aware > Implicit Aware > Implicit Unaware\n")
cat("==============================================================\n")

print(results_df, n = Inf)

# ===============================================================
# Plot ELPD comparison across groups
# ===============================================================

# Nice labels for plotting
loo_summary$model_label <- dplyr::case_when(
  loo_summary$model == "simple_model"   ~ "Simple",
  loo_summary$model == "learning_model" ~ "Learning",
  loo_summary$model == "eta_model"      ~ "Eta",
  loo_summary$model == "learning_boost" ~ "Learning + boost",
  TRUE ~ loo_summary$model
)

loo_summary$group_label <- dplyr::case_when(
  loo_summary$group == "Exp11 Unaware" ~ "Implicit Unaware",
  loo_summary$group == "Exp11 Aware"   ~ "Implicit Aware",
  loo_summary$group == "Exp12 Aware"   ~ "Explicit Aware",
  TRUE ~ loo_summary$group
)

# Keep consistent model order
loo_summary$model_label <- factor(
  loo_summary$model_label,
  levels = c("Simple", "Learning", "Eta", "Learning + boost")
)

loo_summary$group_label <- factor(
  loo_summary$group_label,
  levels = c("Implicit Unaware", "Implicit Aware", "Explicit Aware")
)

# Colours matching your group palette
group_cols <- c(
  "Implicit Unaware" = rgb(0.56, 0.93, 0.56),
  "Implicit Aware"   = rgb(0.00, 0.50, 0.00),
  "Explicit Aware"   = rgb(1.00, 0.65, 0.00)
)

# ---------------------------------------------------------------
# Plot 1: ΔELPD relative to best model within each group
# Best model is always 0. Worse models are negative.
# ---------------------------------------------------------------

p_elpd_diff <- ggplot(
  loo_summary,
  aes(
    x = model_label,
    y = elpd_diff,
    fill = group_label
  )
) +
  geom_col(width = 0.75, colour = "black", linewidth = 0.2) +
  geom_errorbar(
    aes(
      ymin = elpd_diff - se_diff,
      ymax = elpd_diff + se_diff
    ),
    width = 0.18,
    linewidth = 0.7
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.7) +
  facet_wrap(~ group_label, nrow = 1) +
  scale_fill_manual(values = group_cols) +
  labs(
    x = "Model",
    y = expression(Delta * ELPD ~ "(vs best model)"),
    title = "LOO model comparison"
  ) +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "none",
    strip.background = element_blank(),
    strip.text = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 35, hjust = 1),
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

print(p_elpd_diff)

# Save figure
ggsave(
  filename = file.path(loo_dir, "loo_elpd_diff_4models_3groups.png"),
  plot = p_elpd_diff,
  width = 11,
  height = 4.5,
  dpi = 300
)

ggsave(
  filename = file.path(loo_dir, "loo_elpd_diff_4models_3groups.pdf"),
  plot = p_elpd_diff,
  width = 11,
  height = 4.5
)


# ---------------------------------------------------------------
# Plot 2: raw ELPD values
# Higher is better, but values are negative.
# ---------------------------------------------------------------

p_elpd_raw <- ggplot(
  loo_summary,
  aes(
    x = model_label,
    y = elpd_loo,
    fill = group_label
  )
) +
  geom_col(width = 0.75, colour = "black", linewidth = 0.2) +
  geom_errorbar(
    aes(
      ymin = elpd_loo - se_elpd_loo,
      ymax = elpd_loo + se_elpd_loo
    ),
    width = 0.18,
    linewidth = 0.7
  ) +
  facet_wrap(~ group_label, nrow = 1, scales = "free_y") +
  scale_fill_manual(values = group_cols) +
  labs(
    x = "Model",
    y = "ELPD-LOO",
    title = "Raw ELPD-LOO by model"
  ) +
  theme_classic(base_size = 14) +
  theme(
    legend.position = "none",
    strip.background = element_blank(),
    strip.text = element_text(size = 14, face = "bold"),
    axis.text.x = element_text(angle = 35, hjust = 1),
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

print(p_elpd_raw)

ggsave(
  filename = file.path(loo_dir, "loo_elpd_raw_4models_3groups.png"),
  plot = p_elpd_raw,
  width = 11,
  height = 4.5,
  dpi = 300
)

ggsave(
  filename = file.path(loo_dir, "loo_elpd_raw_4models_3groups.pdf"),
  plot = p_elpd_raw,
  width = 11,
  height = 4.5
)










############################################
## DELTA–ETA RELATIONSHIP / TRADE-OFF CHECK
############################################

rm(list = ls(all = TRUE))

library(tidyverse)
library(posterior)
library(ggplot2)

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

base_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias"
fits_dir <- file.path(base_dir, "results", "fits", "Exp12")
fig_dir  <- file.path(base_dir, "results", "figures")

if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

# -------------------------------------------------------------------------
# Fit files
# Use the BOOST model here because this is the model containing delta + eta
# from your latest parameter results.
# -------------------------------------------------------------------------

fit_files <- tibble(
  group = c(
    "Implicit Unaware Base Rate",
    "Implicit Aware Base Rate",
    "Explicit Undirected Base Rate",
    "Explicit True Base Rate",
    "Explicit Deceptive Base Rate"
  ),
  file = c(
    "fit_trunc_boost_model_unaware_exp11.rdata",
    "fit_trunc_boost_model_aware_exp11.rdata",
    "fit_trunc_boost_model_aware_exp12.rdata",
    "fit_trunc_boost_truthful_exp13.rdata",
    "fit_trunc_boost_deceptive_exp13.rdata"
  )
) %>%
  mutate(path = file.path(fits_dir, file))

# -------------------------------------------------------------------------
# Extract delta and eta from one fit
# -------------------------------------------------------------------------

extract_delta_eta <- function(path, group_name) {
  
  if (!file.exists(path)) {
    stop("Missing fit file: ", path)
  }
  
  env <- new.env()
  load(path, envir = env)
  
  if (!exists("fit", envir = env)) {
    stop("No object called 'fit' found in: ", path)
  }
  
  draws <- as_draws_df(env$fit$draws())
  
  # Case 1: transformed parameters already exist
  if (all(c("mu_delta", "mu_eta") %in% names(draws))) {
    
    out <- draws %>%
      select(mu_delta, mu_eta)
    
  } else {
    
    # Case 2: raw mu_pr columns
    mu_cols <- names(draws)[grepl("^mu_pr(\\[|\\.)", names(draws))]
    
    if (length(mu_cols) < 5) {
      stop("Could not find mu_delta/mu_eta or enough mu_pr columns in: ", path)
    }
    
    out <- draws %>%
      select(all_of(mu_cols[c(4, 5)]))
    
    names(out) <- c("mu_delta", "mu_eta")
    
    out <- out %>%
      mutate(
        mu_delta = pnorm(mu_delta) * 2
        # eta remains untransformed
      )
  }
  
  out %>%
    mutate(group = group_name)
}

# -------------------------------------------------------------------------
# Combine draws across groups
# -------------------------------------------------------------------------

delta_eta_draws <- map2_dfr(
  fit_files$path,
  fit_files$group,
  extract_delta_eta
)

group_levels <- c(
  "Implicit Unaware Base Rate",
  "Implicit Aware Base Rate",
  "Explicit Undirected Base Rate",
  "Explicit True Base Rate",
  "Explicit Deceptive Base Rate"
)

delta_eta_draws <- delta_eta_draws %>%
  mutate(group = factor(group, levels = group_levels))

# -------------------------------------------------------------------------
# Correlation table
# -------------------------------------------------------------------------

delta_eta_corrs <- delta_eta_draws %>%
  group_by(group) %>%
  summarise(
    mean_delta = mean(mu_delta, na.rm = TRUE),
    mean_eta   = mean(mu_eta, na.rm = TRUE),
    r_delta_eta = cor(mu_delta, mu_eta, use = "complete.obs"),
    .groups = "drop"
  )

cat("\n====================================================\n")
cat("DELTA–ETA POSTERIOR CORRELATIONS\n")
cat("====================================================\n")
print(delta_eta_corrs, n = Inf)

# -------------------------------------------------------------------------
# Plot delta against eta
# -------------------------------------------------------------------------

p <- ggplot(delta_eta_draws, aes(x = mu_delta, y = mu_eta)) +
  geom_point(alpha = 0.12, size = 0.45) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 0.8) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "red") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  facet_wrap(~group, scales = "free") +
  theme_bw(base_size = 13) +
  labs(
    title = "Posterior relationship between delta and eta",
    subtitle = "Checks whether belief updating and confirmation-asymmetry parameters covary",
    x = "delta",
    y = "eta"
  ) +
  theme(
    strip.text = element_text(face = "bold", size = 10),
    axis.title = element_text(face = "bold"),
    plot.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

print(p)

ggsave(
  filename = file.path(fig_dir, "delta_eta_posterior_relationship_5groups.png"),
  plot = p,
  width = 10,
  height = 7,
  dpi = 300
)

ggsave(
  filename = file.path(fig_dir, "delta_eta_posterior_relationship_5groups.pdf"),
  plot = p,
  width = 10,
  height = 7
)

# -------------------------------------------------------------------------
# Optional: flag possible trade-offs
# -------------------------------------------------------------------------

cat("\nInterpretation guide:\n")
cat("r close to 0 = little evidence of posterior trade-off between delta and eta.\n")
cat("|r| around .5 or larger = possible parameter trade-off / identifiability concern.\n")









############################################
## CHECK WHETHER CHOICE-AGAINST-PRIOR TRIALS
## DRIVE PRIOR-INCONGRUENT EVIDENCE EFFECTS
############################################

rm(list = ls(all = TRUE))

library(tidyverse)
library(ggplot2)
library(broom)

base_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias"
data_dir <- file.path(base_dir, "data")
fig_dir  <- file.path(base_dir, "results", "figures")

if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

# ------------------------------------------------------------
# Data files
# ------------------------------------------------------------

data_files <- tibble(
  group = c(
    "Implicit Unaware Base Rate",
    "Implicit Aware Base Rate",
    "Explicit Undirected Base Rate",
    "Explicit True Base Rate",
    "Explicit Deceptive Base Rate"
  ),
  file = c(
    "data_priorbelief_unaware_exp11.rdata",
    "data_priorbelief_aware_exp11.rdata",
    "data_priorbelief_aware_exp12.rdata",
    "data_priorbelief_truthful_exp13.rdata",
    "data_priorbelief_deceptive_exp13.rdata"
  )
) %>%
  mutate(path = file.path(data_dir, file))

# ------------------------------------------------------------
# Load and prepare data
# ------------------------------------------------------------

load_one_group <- function(path, group_name) {
  
  env <- new.env()
  load(path, envir = env)
  
  data <- env$data
  
  # Exp12 sometimes uses Manipulation_ResponseButtonOrder
  if (!"ResponseButtonOrder" %in% names(data) &&
      "Manipulation_ResponseButtonOrder" %in% names(data)) {
    data <- data %>%
      rename(ResponseButtonOrder = Manipulation_ResponseButtonOrder)
  }
  
  data <- data %>%
    mutate(
      choice = case_when(
        ResponseButtonOrder == 0 & Response == 0 ~ 1, # blue
        ResponseButtonOrder == 0 & Response == 1 ~ 2, # red
        ResponseButtonOrder == 1 & Response == 0 ~ 2, # red
        ResponseButtonOrder == 1 & Response == 1 ~ 1, # blue
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
    )
  
  # Prior-aligned evidence:
  # positive = evidence favours prior
  # negative = evidence favours opposite colour
  data <- data %>%
    rowwise() %>%
    mutate(
      evidence_prior = sum(
        qlogis(c_across(starts_with("proba_")) / 100) *
          ifelse(c_across(starts_with("color_")) == Prior_Belief, 1, -1),
        na.rm = TRUE
      ),
      choose_prior = as.numeric(choice == Prior_Belief),
      choose_against_prior = as.numeric(choice != Prior_Belief)
    ) %>%
    ungroup() %>%
    mutate(
      group = group_name,
      evidence_bin = case_when(
        evidence_prior < -1.0 ~ "Strong evidence against prior",
        evidence_prior >= -1.0 & evidence_prior < -0.25 ~ "Weak evidence against prior",
        evidence_prior >= -0.25 & evidence_prior <= 0.25 ~ "Neutral / balanced",
        evidence_prior > 0.25 & evidence_prior <= 1.0 ~ "Weak evidence for prior",
        evidence_prior > 1.0 ~ "Strong evidence for prior",
        TRUE ~ NA_character_
      )
    )
  
  data
}

all_data <- map2_dfr(
  data_files$path,
  data_files$group,
  load_one_group
)

# ------------------------------------------------------------
# 1. How often do people choose against the prior by evidence bin?
# ------------------------------------------------------------

choice_against_summary <- all_data %>%
  group_by(group, evidence_bin) %>%
  summarise(
    p_choose_against_prior = mean(choose_against_prior, na.rm = TRUE),
    p_choose_prior = mean(choose_prior, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    evidence_bin = factor(
      evidence_bin,
      levels = c(
        "Strong evidence against prior",
        "Weak evidence against prior",
        "Neutral / balanced",
        "Weak evidence for prior",
        "Strong evidence for prior"
      )
    )
  )

cat("\n====================================================\n")
cat("CHOICE AGAINST PRIOR BY EVIDENCE BIN\n")
cat("====================================================\n")
print(choice_against_summary, n = Inf)

# ------------------------------------------------------------
# 2. Plot: choice against prior as function of prior-aligned evidence
# ------------------------------------------------------------

p1 <- ggplot(choice_against_summary,
             aes(x = evidence_bin, y = p_choose_against_prior)) +
  geom_col(width = 0.7) +
  facet_wrap(~group, nrow = 1) +
  labs(
    title = "Choice against prior by prior-aligned evidence",
    x = "Evidence direction",
    y = "P(choose against prior)"
  ) +
  theme_bw(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1, face = "bold"),
    strip.text = element_text(face = "bold"),
    axis.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

print(p1)

ggsave(
  filename = file.path(fig_dir, "choice_against_prior_by_evidence_bin.png"),
  plot = p1,
  width = 14,
  height = 5,
  dpi = 300
)

# ------------------------------------------------------------
# 3. Logistic model:
# Does prior-inconsistent evidence predict choosing against the prior?
# ------------------------------------------------------------

models_choice_against <- all_data %>%
  group_by(group) %>%
  group_modify(~ {
    
    m <- glm(
      choose_against_prior ~ evidence_prior,
      data = .x,
      family = binomial()
    )
    
    tidy(m)
  }) %>%
  ungroup()

cat("\n====================================================\n")
cat("LOGISTIC MODEL: CHOOSE AGAINST PRIOR ~ PRIOR-ALIGNED EVIDENCE\n")
cat("====================================================\n")
print(models_choice_against, n = Inf)

# Interpretation:
# evidence_prior is positive when evidence favours prior.
# Therefore, a NEGATIVE coefficient means:
# stronger evidence for the prior reduces choosing against prior.
# equivalently, evidence against the prior increases choosing against prior.

# ------------------------------------------------------------
# 4. Compare anti-prior choices vs prior choices
# ------------------------------------------------------------

anti_prior_distribution <- all_data %>%
  group_by(group, choose_against_prior) %>%
  summarise(
    mean_evidence_prior = mean(evidence_prior, na.rm = TRUE),
    median_evidence_prior = median(evidence_prior, na.rm = TRUE),
    l95 = quantile(evidence_prior, 0.025, na.rm = TRUE),
    u95 = quantile(evidence_prior, 0.975, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  mutate(
    choice_type = ifelse(
      choose_against_prior == 1,
      "Chose against prior",
      "Chose prior"
    )
  )

cat("\n====================================================\n")
cat("EVIDENCE DISTRIBUTION FOR PRIOR VS AGAINST-PRIOR CHOICES\n")
cat("====================================================\n")
print(anti_prior_distribution, n = Inf)

p2 <- ggplot(all_data,
             aes(x = factor(choose_against_prior),
                 y = evidence_prior)) +
  geom_boxplot(outlier.shape = NA, width = 0.65) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "red") +
  facet_wrap(~group, nrow = 1) +
  scale_x_discrete(
    labels = c("0" = "Chose prior", "1" = "Chose against prior")
  ) +
  labs(
    title = "Prior-aligned evidence on prior vs against-prior choices",
    x = NULL,
    y = "Evidence favouring prior"
  ) +
  theme_bw(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 25, hjust = 1, face = "bold"),
    strip.text = element_text(face = "bold"),
    axis.title = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

print(p2)

ggsave(
  filename = file.path(fig_dir, "evidence_prior_vs_against_prior_choices.png"),
  plot = p2,
  width = 14,
  height = 5,
  dpi = 300
)



data <- data %>%
  mutate(
    analysis_prior = case_when(
      group == "Explicit Deceptive Base Rate" & "InstructedPrior" %in% names(data) ~ InstructedPrior,
      TRUE ~ Prior_Belief
    )
  )

data <- data %>%
  mutate(
    analysis_prior = case_when(
      group == "Explicit Deceptive Base Rate" & "InstructedPrior" %in% names(data) ~ InstructedPrior,
      TRUE ~ Prior_Belief
    )
  )