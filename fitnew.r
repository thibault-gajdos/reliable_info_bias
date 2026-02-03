
#rm(list=ls(all=TRUE))  ## efface les données
#source('~/thib/projects/tools/R_lib.r')
#setwd('~/thib/projects/reliable_info')
#source('~/thib/projects/reliable_info/utils.r')

library(tidyverse)
library(posterior)
library(bayesplot)
library(cmdstanr)

##################################################
##PREPARE THE DATA
##################################################

#
load("/Users/bty615/Documents/GitHub/reliable_info_bias/data/data_priorbelief_aware_exp11.rdata")


## if Response ResponseButtonOrder= 1: blue->1, red->0
## if Response ResponseButtonOrder= 0: blue->0, red->1
## we recode: blue ->1, red ->2
data <- data %>%
  mutate(choice = case_when(
    (ResponseButtonOrder == 1 & Response == 0) ~ 2,
    (ResponseButtonOrder == 1 & Response == 1) ~ 1,
    (ResponseButtonOrder == 0 & Response == 0) ~ 1,
    (ResponseButtonOrder == 0 & Response == 1) ~ 2
  )) %>%
  mutate_at(vars(starts_with("color")), ~ ifelse(. == "blue", 1, 2)) %>%
  rowwise() %>%
  mutate(sample_number = sum(!is.na(c_across(starts_with("proba_"))))) %>%
  ungroup() %>% # <-- PIPE ADDED HERE!
  mutate(feedback = ifelse(CorrectResponse == 1, 1, 0)) # 1=Blue correct, 0=Red correct


N = length(unique(data$ParticipantPrivateID))
T_max = max(data$TrialNumber)
I_max <- max(data$sample_number) ## max number of samples/trial
## compute trials by subject
d <- data %>%
    group_by(ParticipantPrivateID) %>%
    summarise(t_subjs = n())
t_subjs <- d$t_subjs
subjs <- unique(data$ParticipantPrivateID)

    
## Initialize data arrays
choice  <- array(-1, c(N, T_max))
color <- array( -1, c(N, T_max, I_max))
proba <- array(-1, c(N, T_max, I_max))
sample <- array(-1, c(N, T_max))
# NEW: Initialize the feedback array
feedback <- array(0, c(N, T_max))
## fill the  arrays
for (n in 1:N) { ## loop through subjects
  t <- t_subjs[n] ## number of trials for subj i
  data_subj <- data %>% filter(ParticipantPrivateID == subjs[n])
  choice[n, 1:t] <- data_subj$choice 
  #NEW: Populate the feedback array.
  feedback[n, 1:t] <- data_subj$feedback[1:t]
  for (k in 1:t) { ## loop through trials
      data_subj_t <- data_subj[k,]
      sample[n,k] <- data_subj_t$sample_number
      for (i in 1:data_subj_t$sample_number) {
          color_var <- paste0("color_", i)
          proba_var <- paste0("proba_", i)
          color[n, k, i] <- data_subj[[color_var]][k]
          proba[n, k, i] <- data_subj[[proba_var]][k]/100
      }
  }
}


data_list <- list(
    N = N,
    T_max = T_max,
    I_max = I_max,
    Tsubj = t_subjs,
    color = color,
    proba = proba,
    choice = choice,
    sample = sample,
    # NEW: Add the feedback array to the list
    feedback = feedback
)



##save(data_list, file = './data/data_list_8.rdata')

#####################################################
##  FIT THE MODEL
####################################################

setwd("/Users/bty615/Documents/GitHub/reliable_info_bias")
data_list$grainsize = 5 ## specify grainsize for within chain parallelization

## Compile the model
model <- cmdstan_model(
  stan_file = './log_trunc_simplified_boost_learning.stan', 
    force_recompile = TRUE, ## necessary if you change the mode
    cpp_options = list(stan_opencl = FALSE, stan_threads = TRUE), ## within chain parallel
    stanc_options = list("O1"), ## fastest sampling
    compile_model_methods = TRUE ## necessary for loo moment matching
)


## Sampling
fit <- model$sample(
  data = data_list,
  ##seed = 1234,
  seed = 4321,
  ##init = list(inits_chain,inits_chain, inits_chain,inits_chain),
  chains = 4,
  parallel_chains = 4,
  threads_per_chain = 5,
  iter_warmup = 2000,
  iter_sampling = 1000,
  max_treedepth = 12,
  adapt_delta = .9,
  save_warmup = FALSE
)


## Compute LOO
loo <- fit$loo(cores = 10, moment_match = TRUE)

# Save results
dir.create('./results/fits/exp12/', recursive = TRUE, showWarnings = FALSE)
save(fit, file = './results/fits/exp12/fit_trunc_simplified_learning_eta_aware_exp11.rdata')

save(loo, file = './results/loo/loo_trunc_simplified_learning_eta_aware.rdata')





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
load("/Users/imogen/Documents/GitHub/reliable_info/data/data_priorbelief_unaware_exp11.rdata")
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

load("/Users/imogen/Documents/GitHub/reliable_info/results/fits/Exp11/fit_trunc_simplified_aware_exp11.rdata")
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
load("/Users/imogen/Documents/GitHub/reliable_info/results/fits/Exp12/fit_trunc_simplified_learning2_aware_red_exp12.rdata")
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
library(ggplot2)

# -------------------------------------------------------------------
# Load Fit (Asymmetric Model)
# -------------------------------------------------------------------
# FIXED: Changed the closing single quote to a double quote to match the opening
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/exp12/fit_trunc_simplified_learning2_unaware_blue_exp11.rdata")

# -------------------------------------------------------------------
# Function: extract the transformed mu_ parameters
# -------------------------------------------------------------------
extract_mu <- function(fit, label) {
  
  d <- as_draws_df(fit$draws())
  
  # UPDATED: List now includes mu_deltaB and mu_deltaR
  keep_params <- c(
    "mu_alpha",
    "mu_beta",
    "mu_lambda",
    "mu_theta",
    "mu_psi",
    "mu_deltaB",
    "mu_deltaR"
  )
  
  # Ensure only parameters present in the fit are selected
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
# Extract Data
# -------------------------------------------------------------------
# Ensure 'fit' matches the object name inside your .rdata file
df_plot <- extract_mu(fit, "Asymmetric Model")

# -------------------------------------------------------------------
# Define Custom Labels and Colors
# -------------------------------------------------------------------
param_labels <- c(
  "mu_alpha"  = expression(paste(mu[alpha])),
  "mu_beta"   = expression(paste(mu[beta])),
  "mu_lambda" = expression(paste(mu[lambda], " (Sequential Decay)")),
  "mu_psi"    = expression(paste(mu[psi], " (Distortion Scaling)")),
  "mu_theta"  = expression(paste(mu[theta], " (Response Noise)")),
  "mu_deltaB" = expression(paste(mu[delta[B]], " (Persistence Blue)")),
  "mu_deltaR" = expression(paste(mu[delta[R]], " (Persistence Red)"))
)

# Using a single color since there is only one group
model_color <- "#1B5E20" # Dark Green

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
p <- ggplot(df_plot, aes(x = value)) +
  geom_histogram(
    fill = model_color,
    position = "identity",
    bins = 80,
    alpha = 0.7,
    color = "black", 
    linewidth = 0.1
  ) +
  facet_wrap(~param, scales = "free", ncol = 3, 
             labeller = as_labeller(param_labels, default = label_parsed)) +
  
  theme_bw(base_size = 16) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    
    strip.background = element_rect(fill = "gray90", color = "gray50"),
    strip.text = element_text(face = "bold", size = 10),
    
    legend.position = "none" 
  ) +
  labs(
    title = "Posterior Distributions",

    x = "Posterior Sample Value",
    y = "Frequency"
  )

print(p)






library(tidyverse)
library(posterior)
library(bayesplot)
library(ggplot2)

# =====================================================================
# 1. SETUP & DATA LOADING
# =====================================================================

fit_path <- "/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_boost_learning_aware_exp11.rdata"
load(fit_path)

# CmdStanR draws
posterior_draws <- fit$draws()

# ---------------------------------------------------------------------
# PARAMETER INDEX MAP (from Stan model)
# mu_pr[1] = alpha
# mu_pr[2] = beta
# mu_pr[3] = lambda
# mu_pr[4] = theta
# mu_pr[5] = psi
# mu_pr[6] = delta
# mu_pr[7] = kappa
# ---------------------------------------------------------------------

target_mu_params <- c(
  "mu_pr[1]",  # alpha
  "mu_pr[2]",  # beta
  "mu_pr[3]",  # lambda
  "mu_pr[4]",  # theta
  "mu_pr[5]",  # psi
  "mu_pr[6]",  # delta
  "mu_pr[7]"   # kappa
)

# =====================================================================
# 2. POSTERIOR DISTRIBUTIONS (Density / Areas Plot)
# =====================================================================

plot_posteriors <- mcmc_areas(
  posterior_draws,
  pars = target_mu_params,
  prob = 0.95,
  prob_outer = 0.99,
  point_est = "median"
) +
  labs(
    title = "Group-Level Posterior Distributions",
    subtitle = "95% Credible Intervals and Medians",
    x = "Parameter Value"
  ) +
  theme_minimal()

print(plot_posteriors)

# =====================================================================
# 3. PARAMETER CORRELATIONS (Pairs Plot)
# =====================================================================

plot_correlations <- mcmc_pairs(
  posterior_draws,
  pars = target_mu_params,
  off_diag_args = list(size = 0.5, alpha = 0.2),
  diag_fun = "hist"
)

print(plot_correlations)

# =====================================================================
# 4. NUMERICAL SUMMARY & CORRELATION MATRIX
# =====================================================================

# Group-level numerical summary
stats_summary <- fit$summary(variables = target_mu_params)
print(stats_summary)

# Correlation matrix from posterior draws
draws_df <- as_draws_df(posterior_draws) %>%
  select(all_of(target_mu_params))

cor_matrix <- cor(draws_df)

cat("\n--- POSTERIOR CORRELATION MATRIX (mu_pr) ---\n")
print(round(cor_matrix, 3))





library(tidyverse)
library(posterior)
library(bayesplot)
library(ggplot2)

# -------------------------------------------------------------------
# 1. LOAD FIT
# -------------------------------------------------------------------
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_boost_learning_aware_exp11.rdata")

# -------------------------------------------------------------------
# 2. EXTRACT & RENAME GROUP-LEVEL PARAMETERS
# -------------------------------------------------------------------
extract_mu <- function(fit, label) {
  
  d <- as_draws_df(fit$draws())
  
  # Stan index -> readable name
  mu_map <- c(
    "mu_pr[1]" = "mu_alpha",
    "mu_pr[2]" = "mu_beta",
    "mu_pr[3]" = "mu_lambda",
    "mu_pr[4]" = "mu_theta",
    "mu_pr[5]" = "mu_psi",
    "mu_pr[6]" = "mu_delta",
    "mu_pr[7]" = "mu_kappa"
  )
  
  mu_df <- d %>%
    select(all_of(names(mu_map))) %>%
    rename(!!!setNames(names(mu_map), mu_map)) %>%
    mutate(group = label) %>%
    pivot_longer(
      cols = all_of(unname(mu_map)),
      names_to = "param",
      values_to = "value"
    )
  
  mu_df
}

df_plot <- extract_mu(fit, "Boost Learning Model")

# -------------------------------------------------------------------
# 3. PARAMETER LABELS (PARSED, PUBLICATION-READY)
# -------------------------------------------------------------------
param_labels <- c(
  "mu_alpha"  = expression(paste(mu[alpha], " (Evidence Sensitivity)")),
  "mu_beta"   = expression(paste(mu[beta], " (Bias / Intercept)")),
  "mu_lambda" = expression(paste(mu[lambda], " (Sequential Decay)")),
  "mu_psi"    = expression(paste(mu[psi], " (Distortion Scaling)")),
  "mu_theta"  = expression(paste(mu[theta], " (Response Noise)")),
  "mu_delta"  = expression(paste(mu[delta], " (Belief Persistence)")),
  "mu_kappa"  = expression(paste(mu[kappa], " (Confirmation Boost)"))
)

model_color <- "#1B5E20"  # dark green

# -------------------------------------------------------------------
# 4. POSTERIOR DISTRIBUTIONS (HISTOGRAM FACETS)
# -------------------------------------------------------------------
p_post <- ggplot(df_plot, aes(x = value)) +
  geom_histogram(
    fill = model_color,
    bins = 80,
    alpha = 0.7,
    color = "black",
    linewidth = 0.1
  ) +
  facet_wrap(
    ~ param,
    scales = "free",
    ncol = 3,
    labeller = as_labeller(param_labels, default = label_parsed)
  ) +
  theme_bw(base_size = 16) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    strip.background = element_rect(fill = "gray90", color = "gray50"),
    strip.text       = element_text(face = "bold", size = 11),
    legend.position  = "none"
  ) +
  labs(
    title = "Posterior Distributions",
    subtitle = "Group-level parameters",
    x = "Posterior Sample Value",
    y = "Frequency"
  )

print(p_post)

# -------------------------------------------------------------------
# 5. CORRELATION PAIRS PLOT
# -------------------------------------------------------------------
draws_wide <- as_draws_df(fit$draws()) %>%
  select(
    `mu_pr[1]`, `mu_pr[2]`, `mu_pr[3]`,
    `mu_pr[4]`, `mu_pr[5]`, `mu_pr[6]`, `mu_pr[7]`
  ) %>%
  rename(
    mu_alpha  = `mu_pr[1]`,
    mu_beta   = `mu_pr[2]`,
    mu_lambda = `mu_pr[3]`,
    mu_theta  = `mu_pr[4]`,
    mu_psi    = `mu_pr[5]`,
    mu_delta  = `mu_pr[6]`,
    mu_kappa  = `mu_pr[7]`
  )

color_scheme_set("green")

p_corr <- mcmc_pairs(
  draws_wide,
  pars = colnames(draws_wide),
  off_diag_args = list(size = 0.5, alpha = 0.25),
  diag_fun = "hist"
)

print(p_corr)

# -------------------------------------------------------------------
# 6. NUMERIC CORRELATION MATRIX (OPTIONAL)
# -------------------------------------------------------------------
cor_matrix <- cor(draws_wide)

cat("\n--- POSTERIOR CORRELATION MATRIX (mu parameters) ---\n")
print(round(cor_matrix, 3))










library(tidyverse)
library(posterior)
library(ggplot2)
library(ggridges)

# -------------------------------------------------------------------
# 1. LOAD FIT
# -------------------------------------------------------------------
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_boost_learning_aware_exp11.rdata")

draws <- as_draws_df(fit$draws())

# -------------------------------------------------------------------
# 2. PARAMETER MAP (INDEX → NAME)
# -------------------------------------------------------------------
param_map <- tibble(
  idx = 1:7,
  name = c("alpha", "beta", "lambda", "theta", "psi", "delta", "kappa")
)

# -------------------------------------------------------------------
# 3. GROUP-LEVEL PARAMETERS (TRANSFORMED)
# -------------------------------------------------------------------
group_params <- draws %>%
  transmute(
    alpha  = pnorm(`mu_pr[1]`),
    beta   = `mu_pr[2]`,
    lambda = exp(`mu_pr[3]`),
    theta  = 5 * pnorm(`mu_pr[4]`),
    psi    = 5 * pnorm(`mu_pr[5]`),
    delta  = pnorm(`mu_pr[6]`),
    kappa  = 1 + 2 * pnorm(`mu_pr[7]`)
  ) %>%
  pivot_longer(
    everything(),
    names_to = "parameter",
    values_to = "value"
  )

# -------------------------------------------------------------------
# 4. INDIVIDUAL-LEVEL PARAMETERS (TRANSFORMED)
# -------------------------------------------------------------------
# param_raw[n, p] → subject n, parameter p

individual_params <- draws %>%
  select(starts_with("param_raw[")) %>%
  pivot_longer(
    everything(),
    names_to = "param",
    values_to = "z"
  ) %>%
  extract(
    param,
    into = c("subject", "idx"),
    regex = "param_raw\\[(\\d+),(\\d+)\\]",
    convert = TRUE
  ) %>%
  left_join(param_map, by = "idx") %>%
  mutate(
    value = case_when(
      name == "alpha"  ~ pnorm(z),
      name == "beta"   ~ z,
      name == "lambda" ~ exp(z),
      name == "theta"  ~ 5 * pnorm(z),
      name == "psi"    ~ 5 * pnorm(z),
      name == "delta"  ~ pnorm(z),
      name == "kappa"  ~ 1 + 2 * pnorm(z)
    )
  )

# -------------------------------------------------------------------
# 5. PLOT 1: GROUP-LEVEL POSTERIORS (INTERPRETABLE SCALE)
# -------------------------------------------------------------------
p_group <- ggplot(group_params, aes(x = value)) +
  geom_histogram(
    bins = 60,
    fill = "#1B5E20",
    color = "black",
    alpha = 0.75
  ) +
  facet_wrap(~parameter, scales = "free", ncol = 3) +
  theme_bw(base_size = 15) +
  labs(
    title = "Group-level Posterior Distributions",
    subtitle = "Interpretable parameter scale",
    x = "Parameter value",
    y = "Frequency"
  )

print(p_group)

# -------------------------------------------------------------------
# 6. PLOT 2: INDIVIDUAL DIFFERENCES (RIDGE PLOTS)
# -------------------------------------------------------------------
p_indiv <- individual_params %>%
  filter(parameter %in% c("delta", "kappa", "alpha")) %>%
  ggplot(aes(x = value, y = factor(subject))) +
  geom_density_ridges(
    scale = 2,
    rel_min_height = 0.01,
    fill = "#1B5E20",
    alpha = 0.6
  ) +
  facet_wrap(~parameter, scales = "free_x") +
  theme_ridges(base_size = 14) +
  labs(
    title = "Individual-level Parameter Distributions",
    x = "Parameter value",
    y = "Participant"
  )

print(p_indiv)

# -------------------------------------------------------------------
# 7. OPTIONAL: GROUP vs INDIVIDUAL OVERLAY (DELTA & KAPPA)
# -------------------------------------------------------------------
p_overlay <- individual_params %>%
  filter(parameter %in% c("delta", "kappa")) %>%
  ggplot(aes(x = value)) +
  geom_density(
    data = group_params %>% filter(parameter %in% c("delta", "kappa")),
    aes(x = value),
    color = "black",
    linewidth = 1.2
  ) +
  geom_density(
    aes(group = subject),
    alpha = 0.15,
    fill = "#1B5E20"
  ) +
  facet_wrap(~parameter, scales = "free") +
  theme_bw(base_size = 14) +
  labs(
    title = "Group vs Individual Parameter Distributions",
    x = "Parameter value",
    y = "Density"
  )

print(p_overlay)











library(tidyverse)
library(posterior)
library(ggplot2)
library(ggridges)

# -------------------------------------------------------------------
# 1. LOAD FIT
# -------------------------------------------------------------------
# Ensure this matches your latest file name
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_eta_aware_exp11.rdata")

draws <- as_draws_df(fit$draws())

# -------------------------------------------------------------------
# 2. PARAMETER MAP (INDEX → NAME)
# -------------------------------------------------------------------
param_map <- tibble(
  idx = 1:7,
  name = c("alpha", "beta", "lambda", "theta", "psi", "delta", "eta")
)

# -------------------------------------------------------------------
# 3. GROUP-LEVEL PARAMETERS (TRANSFORMED)
# -------------------------------------------------------------------
group_params <- draws %>%
  transmute(
    alpha  = pnorm(`mu_pr[1]`),
    beta   = `mu_pr[2]`,
    lambda = pnorm(`mu_pr[3]`),
    theta  = 1.0, # FIXED
    psi    = 1.0, # FIXED
    delta  = pnorm(`mu_pr[6]`),
    eta    = pnorm(`mu_pr[7]`)
  ) %>%
  pivot_longer(
    everything(),
    names_to = "parameter",
    values_to = "value"
  )

# -------------------------------------------------------------------
# 4. INDIVIDUAL-LEVEL PARAMETERS (TRANSFORMED)
# -------------------------------------------------------------------
individual_params <- draws %>%
  select(starts_with("param_raw[")) %>%
  pivot_longer(
    everything(),
    names_to = "param_string",
    values_to = "z"
  ) %>%
  extract(
    param_string,
    into = c("subject", "idx"),
    regex = "param_raw\\[(\\d+),(\\d+)\\]",
    convert = TRUE
  ) %>%
  left_join(param_map, by = "idx") %>%
  # We use the mu_pr and sigma_pr columns from 'draws' for each sample
  mutate(
    mu = case_when(
      idx == 1 ~ draws[[paste0("mu_pr[1]")]][1:n()],
      idx == 2 ~ draws[[paste0("mu_pr[2]")]][1:n()],
      idx == 3 ~ draws[[paste0("mu_pr[3]")]][1:n()],
      idx == 6 ~ draws[[paste0("mu_pr[6]")]][1:n()],
      idx == 7 ~ draws[[paste0("mu_pr[7]")]][1:n()],
      TRUE     ~ 0
    ),
    sigma = case_when(
      idx == 1 ~ draws[[paste0("sigma_pr[1]")]][1:n()],
      idx == 2 ~ draws[[paste0("sigma_pr[2]")]][1:n()],
      idx == 3 ~ draws[[paste0("sigma_pr[3]")]][1:n()],
      idx == 6 ~ draws[[paste0("sigma_pr[6]")]][1:n()],
      idx == 7 ~ draws[[paste0("sigma_pr[7]")]][1:n()],
      TRUE     ~ 0
    )
  ) %>%
  mutate(
    value = case_when(
      name == "alpha"  ~ pnorm(mu + sigma * z),
      name == "beta"   ~ mu + sigma * z,
      name == "lambda" ~ pnorm(mu + sigma * z),
      name == "theta"  ~ 1.0,
      name == "psi"    ~ 1.0,
      name == "delta"  ~ pnorm(mu + sigma * z),
      name == "eta"    ~ pnorm(mu + sigma * z)
    )
  )

# -------------------------------------------------------------------
# 5. PLOT 1: GROUP-LEVEL POSTERIORS
# -------------------------------------------------------------------
# We filter out theta and psi because they are constants (value = 1.0)
p_group <- group_params %>% 
  filter(!parameter %in% c("theta", "psi")) %>%
  ggplot(aes(x = value)) +
  geom_histogram(bins = 50, fill = "#2E7D32", color = "white", alpha = 0.8) +
  facet_wrap(~parameter, scales = "free", ncol = 3) +
  theme_minimal(base_size = 14) +
  labs(
    title = "Group-level Posterior Distributions",
    subtitle = "Continuous Eta Model (Anchored theta/psi = 1.0)",
    x = "Parameter Value",
    y = "Frequency"
  )

print(p_group)

# -------------------------------------------------------------------
# 6. PLOT 2: INDIVIDUAL DIFFERENCES (RIDGE PLOTS)
# -------------------------------------------------------------------
# IMPORTANT: We only plot parameters with variance (skip theta/psi)

p_indiv <- individual_params %>%
  filter(name %in% c("alpha", "delta", "eta")) %>%
  ggplot(aes(x = value, y = factor(subject), fill = name)) +
  geom_density_ridges(
    scale = 1.5, 
    rel_min_height = 0.01, 
    alpha = 0.7, 
    color = "white"
  ) +
  facet_wrap(~name, scales = "free_x") +
  scale_fill_viridis_d(option = "mako", begin = 0.3, end = 0.7) +
  theme_ridges(base_size = 12) +
  theme(legend.position = "none") +
  labs(
    title = "Individual Subject Posteriors",
    subtitle = "Key dynamic parameters: Alpha, Delta, and Eta",
    x = "Parameter Value",
    y = "Subject ID"
  )

print(p_indiv)