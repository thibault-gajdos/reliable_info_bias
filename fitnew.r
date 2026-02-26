
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
load("/Users/bty615/Documents/GitHub/reliable_info_bias/data/data_priorbelief_aware_exp12.rdata")


## if Response ResponseButtonOrder= 1: blue->1, red->0
## if Response ResponseButtonOrder= 0: blue->0, red->1
## we recode: blue ->1, red ->2
data <- data %>%
  mutate(choice = case_when(
    (Manipulation_ResponseButtonOrder == 1 & Response == 0) ~ 2,
    (Manipulation_ResponseButtonOrder == 1 & Response == 1) ~ 1,
    (Manipulation_ResponseButtonOrder == 0 & Response == 0) ~ 1,
    (Manipulation_ResponseButtonOrder == 0 & Response == 1) ~ 2
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

setwd("/Users/bty615/Documents/GitHub/reliable_info_bias/stan")
data_list$grainsize = 5 ## specify grainsize for within chain parallelization

## Compile the model
model <- cmdstan_model(
  stan_file = './log_trunc_simplified_model.stan', 
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
save(fit, file = './results/fits/exp12/fit_trunc_simplified_model_aware_exp12.rdata')

save(loo, file = './results/loo/loo_trunc_simplified_model_aware_exp12.rdata')





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







library(tidyverse)
library(posterior)
library(ggplot2)

# -------------------------------------------------------------------
# 1. Load fits 
# -------------------------------------------------------------------
# Ensure these files contain the 5-parameter (alpha, beta, lambda, delta, eta) model
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_boost_aware_exp12.rdata")
fit_explicit <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_boost_aware_exp11.rdata")
fit_aware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_boost_unaware_exp11.rdata")
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
# LOO comparisons for:
#   - Exp11 UNAWARE
#   - Exp11 AWARE
#   - Exp12 AWARE
# Comparing: simple vs learn vs boost
# ===============================================================

library(loo)

load_loo <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
  e <- new.env()
  load(path, envir = e)
  if (!exists("loo", envir = e)) stop("No object named `loo` found in: ", path)
  e$loo
}

run_group <- function(label, simple_path, learn_path, boost_path) {
  
  loo_simple <- load_loo(simple_path)
  loo_learn  <- load_loo(learn_path)
  loo_boost  <- load_loo(boost_path)
  
  cat("\n====================================================\n")
  cat(label, "\n")
  cat("====================================================\n")
  
  Ns <- c(
    simple = length(loo_simple$pointwise[, "elpd_loo"]),
    learn  = length(loo_learn$pointwise[, "elpd_loo"]),
    boost  = length(loo_boost$pointwise[, "elpd_loo"])
  )
  cat("\nN observations used (should match):\n")
  print(Ns)
  if (length(unique(Ns)) != 1) warning(label, ": N differs across models — not comparable!")
  
  cat("\nPareto-k table — SIMPLE:\n"); print(pareto_k_table(loo_simple))
  cat("\nPareto-k table — LEARN:\n");  print(pareto_k_table(loo_learn))
  cat("\nPareto-k table — BOOST:\n");  print(pareto_k_table(loo_boost))
  
  comp <- loo_compare(list(simple = loo_simple, learn = loo_learn, boost = loo_boost))
  cat("\nLOO comparison (higher elpd_loo is better):\n")
  print(comp, digits = 2, simplify = FALSE)
  
  invisible(list(simple = loo_simple, learn = loo_learn, boost = loo_boost, comp = comp))
}

# ----------------------------
# Your intentional mapping kept exactly
# ----------------------------

res_exp11_unaware <- run_group(
  label       = "Exp11 — UNAWARE",
  simple_path = "./results/loo/loo_trunc_simplified_model_unaware_exp11.rdata",
  learn_path  = "./results/loo/loo_trunc_simplified_learning_unaware_exp11.rdata",
  boost_path  = "./results/loo/loo_trunc_simplified_boost_unaware_exp11.rdata"
)

res_exp11_aware <- run_group(
  label       = "Exp11 — AWARE",
  simple_path = "./results/loo/loo_trunc_simplified_model_aware_exp11.rdata",
  learn_path  = "./results/loo/loo_trunc_simplified_learning_aware_exp11.rdata",
  boost_path  = "./results/loo/loo_trunc_simplified_boost_aware_exp11.rdata"
)

res_exp12_aware <- run_group(
  label       = "Exp12 — AWARE",
  simple_path = "./results/loo/loo_trunc_simplified_model_aware_exp12.rdata",
  learn_path  = "./results/loo/loo_trunc_simplified_learning_aware_exp12.rdata",
  boost_path  = "./results/loo/loo_trunc_simplified_boost_aware_exp12.rdata"
)

# Quick access
comp_exp11_unaware <- res_exp11_unaware$comp
comp_exp11_aware   <- res_exp11_aware$comp
comp_exp12_aware   <- res_exp12_aware$comp







# ===============================================================
# Full script: PSIS-LOO comparisons + BAR CHARTS (base R)
# Groups:
#   - Exp11 UNAWARE
#   - Exp11 AWARE
#   - Exp12 AWARE
# Models (your intentional mapping preserved):
#   simple_path = ...model...
#   learn_path  = ...learning...
#   boost_path  = ...boost...
# ===============================================================

rm(list = ls(all = TRUE))
library(loo)

# ----------------------------
# Helper: safely load `loo` from an .rdata file
# ----------------------------
load_loo <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
  e <- new.env()
  load(path, envir = e)
  if (!exists("loo", envir = e)) stop("No object named `loo` found in: ", path)
  e$loo
}

# ----------------------------
# Helper: run diagnostics + comparison
# ----------------------------
run_group <- function(label, simple_path, learn_path, boost_path) {
  
  loo_simple <- load_loo(simple_path)
  loo_learn  <- load_loo(learn_path)
  loo_boost  <- load_loo(boost_path)
  
  cat("\n====================================================\n")
  cat(label, "\n")
  cat("====================================================\n")
  
  # Alignment check: same number of observations?
  Ns <- c(
    simple = length(loo_simple$pointwise[, "elpd_loo"]),
    learn  = length(loo_learn$pointwise[, "elpd_loo"]),
    boost  = length(loo_boost$pointwise[, "elpd_loo"])
  )
  cat("\nN observations used (should match):\n")
  print(Ns)
  if (length(unique(Ns)) != 1) warning(label, ": N differs across models — not comparable!")
  
  # Pareto-k diagnostics
  cat("\nPareto-k table — SIMPLE:\n"); print(pareto_k_table(loo_simple))
  cat("\nPareto-k table — LEARN:\n");  print(pareto_k_table(loo_learn))
  cat("\nPareto-k table — BOOST:\n");  print(pareto_k_table(loo_boost))
  
  # LOO compare (named properly)
  comp <- loo_compare(list(simple = loo_simple, learn = loo_learn, boost = loo_boost))
  cat("\nLOO comparison (higher elpd_loo is better):\n")
  print(comp, digits = 2, simplify = FALSE)
  
  invisible(list(simple = loo_simple, learn = loo_learn, boost = loo_boost, comp = comp))
}

# ----------------------------
# BAR CHART helpers (base R)
# ----------------------------
bar_loo_diff <- function(comp, main = "") {
  d <- as.data.frame(comp)
  models <- rownames(d)
  
  # Order so best (0) is first/top
  ord <- order(d$elpd_diff, decreasing = TRUE)
  d <- d[ord, , drop = FALSE]
  models <- models[ord]
  
  y <- d$elpd_diff
  se <- d$se_diff
  
  mids <- barplot(y,
                  names.arg = models,
                  horiz = TRUE,
                  las = 1,
                  xlab = "Δelpd (vs best; higher is better)",
                  main = main)
  
  segments(y - se, mids, y + se, mids, lwd = 2)
  abline(v = 0, lty = 2, col = "gray40")
}

bar_loo_elpd <- function(comp, main = "") {
  d <- as.data.frame(comp)
  models <- rownames(d)
  
  # Order by absolute elpd_loo (higher is better)
  ord <- order(d$elpd_loo, decreasing = TRUE)
  d <- d[ord, , drop = FALSE]
  models <- models[ord]
  
  y <- d$elpd_loo
  se <- d$se_elpd_loo
  
  mids <- barplot(y,
                  names.arg = models,
                  horiz = TRUE,
                  las = 1,
                  xlab = "elpd_loo (higher is better)",
                  main = main)
  
  segments(y - se, mids, y + se, mids, lwd = 2)
}

# ===============================================================
# Run all 3 groups (YOUR intentional mapping kept)
# ===============================================================

res_exp11_unaware <- run_group(
  label       = "Exp11 — UNAWARE",
  simple_path = "./results/loo/loo_trunc_simplified_model_unaware_exp11.rdata",
  learn_path  = "./results/loo/loo_trunc_simplified_learning_unaware_exp11.rdata",
  boost_path  = "./results/loo/loo_trunc_simplified_boost_unaware_exp11.rdata"
)

res_exp11_aware <- run_group(
  label       = "Exp11 — AWARE",
  simple_path = "./results/loo/loo_trunc_simplified_model_aware_exp11.rdata",
  learn_path  = "./results/loo/loo_trunc_simplified_learning_aware_exp11.rdata",
  boost_path  = "./results/loo/loo_trunc_simplified_boost_aware_exp11.rdata"
)

res_exp12_aware <- run_group(
  label       = "Exp12 — AWARE",
  simple_path = "./results/loo/loo_trunc_simplified_model_aware_exp12.rdata",
  learn_path  = "./results/loo/loo_trunc_simplified_learning_aware_exp12.rdata",
  boost_path  = "./results/loo/loo_trunc_simplified_boost_aware_exp12.rdata"
)

# Quick access to compare tables
comp_exp11_unaware <- res_exp11_unaware$comp
comp_exp11_aware   <- res_exp11_aware$comp
comp_exp12_aware   <- res_exp12_aware$comp

# ===============================================================
# PLOTS
#   1) Δelpd bar charts (recommended)
#   2) Absolute elpd_loo bar charts (optional)
# ===============================================================

# ---- 1) Δelpd (vs best) ----
par(mfrow = c(1,3), mar = c(4,8,3,1))
bar_loo_diff(comp_exp11_unaware, "Exp11 — Unaware (Δelpd)")
bar_loo_diff(comp_exp11_aware,   "Exp11 — Aware (Δelpd)")
bar_loo_diff(comp_exp12_aware,   "Exp12 — Aware (Δelpd)")
par(mfrow = c(1,1))

# ---- 2) Absolute elpd_loo (optional) ----
par(mfrow = c(1,3), mar = c(4,8,3,1))
bar_loo_elpd(comp_exp11_unaware, "Exp11 — Unaware (elpd_loo)")
bar_loo_elpd(comp_exp11_aware,   "Exp11 — Aware (elpd_loo)")
bar_loo_elpd(comp_exp12_aware,   "Exp12 — Aware (elpd_loo)")
par(mfrow = c(1,1))








# ===============================================================
# LOO comparisons for:
#   - Exp11 UNAWARE
#   - Exp11 AWARE
#   - Exp12 AWARE
# Comparing: simple vs learn vs boost
# PLUS: pairwise deltas for slide narration:
#   learn vs simple  (δ-only extension)
#   boost vs learn   (η added on top of δ)
#   boost vs simple  (full model vs baseline)
# ===============================================================

rm(list=ls(all=TRUE))
library(loo)

load_loo <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
  e <- new.env()
  load(path, envir = e)
  if (!exists("loo", envir = e)) stop("No object named `loo` found in: ", path)
  e$loo
}

run_group <- function(label, simple_path, learn_path, boost_path) {
  
  loo_simple <- load_loo(simple_path)
  loo_learn  <- load_loo(learn_path)
  loo_boost  <- load_loo(boost_path)
  
  cat("\n====================================================\n")
  cat(label, "\n")
  cat("====================================================\n")
  
  Ns <- c(
    simple = nrow(loo_simple$pointwise),
    learn  = nrow(loo_learn$pointwise),
    boost  = nrow(loo_boost$pointwise)
  )
  cat("\nN observations used (must match):\n")
  print(Ns)
  if (length(unique(Ns)) != 1) warning(label, ": N differs across models — not comparable!")
  
  cat("\nPareto-k table — SIMPLE:\n"); print(pareto_k_table(loo_simple))
  cat("\nPareto-k table — LEARN:\n");  print(pareto_k_table(loo_learn))
  cat("\nPareto-k table — BOOST:\n");  print(pareto_k_table(loo_boost))
  
  comp <- loo_compare(list(simple = loo_simple, learn = loo_learn, boost = loo_boost))
  cat("\nLOO comparison (ΔELPD vs best; best has 0):\n")
  print(comp, digits = 2, simplify = FALSE)
  
  # ----------------------------
  # Pairwise comparisons for narration
  # ----------------------------
  pair_ls <- loo_compare(list(learn = loo_learn, simple = loo_simple))  # learn vs simple
  pair_bl <- loo_compare(list(boost = loo_boost, learn  = loo_learn))   # boost vs learn
  pair_bs <- loo_compare(list(boost = loo_boost, simple = loo_simple))  # boost vs simple  <-- NEW
  
  d_ls <- as.data.frame(pair_ls)
  d_bl <- as.data.frame(pair_bl)
  d_bs <- as.data.frame(pair_bs)
  
  # Signed ΔELPDs from point estimates
  elpd_simple <- loo_simple$estimates["elpd_loo","Estimate"]
  elpd_learn  <- loo_learn$estimates["elpd_loo","Estimate"]
  elpd_boost  <- loo_boost$estimates["elpd_loo","Estimate"]
  
  delta_learn_simple <- elpd_learn - elpd_simple
  delta_boost_learn  <- elpd_boost - elpd_learn
  delta_boost_simple <- elpd_boost - elpd_simple  # <-- NEW
  
  # SE magnitudes for pairwise differences:
  # take se_diff from the non-best row in each 2-model compare
  se_from_pair <- function(df) {
    best <- rownames(df)[which.max(df$elpd_loo)]
    abs(df$se_diff[rownames(df) != best])
  }
  
  se_learn_simple <- se_from_pair(d_ls)
  se_boost_learn  <- se_from_pair(d_bl)
  se_boost_simple <- se_from_pair(d_bs)  # <-- NEW
  
  # Slide sentence
  cat("\nSLIDE SENTENCE:\n")
  cat(sprintf(
    paste0(
      "%s: ΔELPD(learn−simple) = %.1f (SE≈%.1f). ",
      "ΔELPD(boost−learn) = %.1f (SE≈%.1f). ",
      "ΔELPD(boost−simple) = %.1f (SE≈%.1f).\n"
    ),
    label,
    delta_learn_simple, se_learn_simple,
    delta_boost_learn,  se_boost_learn,
    delta_boost_simple, se_boost_simple
  ))
  
  invisible(list(
    simple = loo_simple, learn = loo_learn, boost = loo_boost,
    comp = comp,
    deltas = c(learn_minus_simple = delta_learn_simple,
               boost_minus_learn  = delta_boost_learn,
               boost_minus_simple = delta_boost_simple),
    se = c(se_learn_simple = se_learn_simple,
           se_boost_learn  = se_boost_learn,
           se_boost_simple = se_boost_simple)
  ))
}

# ----------------------------
# Your intentional mapping kept exactly
# ----------------------------

res_exp11_unaware <- run_group(
  label       = "Exp11 — UNAWARE",
  simple_path = "./results/loo/loo_trunc_simplified_model_unaware_exp11.rdata",
  learn_path  = "./results/loo/loo_trunc_simplified_learning_unaware_exp11.rdata",
  boost_path  = "./results/loo/loo_trunc_simplified_boost_unaware_exp11.rdata"
)

res_exp11_aware <- run_group(
  label       = "Exp11 — AWARE",
  simple_path = "./results/loo/loo_trunc_simplified_model_aware_exp11.rdata",
  learn_path  = "./results/loo/loo_trunc_simplified_learning_aware_exp11.rdata",
  boost_path  = "./results/loo/loo_trunc_simplified_boost_aware_exp11.rdata"
)

res_exp12_aware <- run_group(
  label       = "Exp12 — AWARE",
  simple_path = "./results/loo/loo_trunc_simplified_model_aware_exp12.rdata",
  learn_path  = "./results/loo/loo_trunc_simplified_learning_aware_exp12.rdata",
  boost_path  = "./results/loo/loo_trunc_simplified_boost_aware_exp12.rdata"
)

rm(list = ls(all = TRUE))

rm(list = ls(all = TRUE))
library(loo)

# ---- load .rdata that contains an object named `loo`
load_loo <- function(path) {
  if (!file.exists(path)) stop("File not found: ", path)
  e <- new.env()
  load(path, envir = e)
  if (!exists("loo", envir = e)) stop("No object named `loo` found in: ", path)
  e$loo
}

# ---- compute signed ΔELPD and its SE for A - B
delta_elpd <- function(loo_A, loo_B) {
  comp <- loo_compare(list(A = loo_A, B = loo_B))
  df <- as.data.frame(comp)
  
  elpd_A <- loo_A$estimates["elpd_loo", "Estimate"]
  elpd_B <- loo_B$estimates["elpd_loo", "Estimate"]
  delta  <- elpd_A - elpd_B
  
  best <- rownames(df)[which.max(df$elpd_loo)]
  se   <- abs(df$se_diff[rownames(df) != best])
  
  c(delta = as.numeric(delta), se = as.numeric(se))
}

# ---- build contrasts for one group from file paths
group_contrasts <- function(simple_path, learn_path, boost_path) {
  loo_simple <- load_loo(simple_path)
  loo_learn  <- load_loo(learn_path)
  loo_boost  <- load_loo(boost_path)
  
  Ns <- c(simple = nrow(loo_simple$pointwise),
          learn  = nrow(loo_learn$pointwise),
          boost  = nrow(loo_boost$pointwise))
  if (length(unique(Ns)) != 1) warning("Nobs differs across models: ", paste(Ns, collapse = ", "))
  
  ls <- delta_elpd(loo_learn, loo_simple)  # learn - simple  (δ only)
  bs <- delta_elpd(loo_boost, loo_simple)  # boost - simple  (δ + η)
  bl <- delta_elpd(loo_boost, loo_learn)   # boost - learn   (η on top)
  
  list(
    deltas = c(ls["delta"], bs["delta"], bl["delta"]),
    ses    = c(ls["se"],    bs["se"],    bl["se"]),
    Ns = Ns
  )
}

# ---- plot one panel (group-colored bars)
plot_panel <- function(title, deltas, ses, ylim, bar_col) {
  
  labels <- c("learn − simple\n(δ only)",
              "boost − simple\n(δ + η)",
              "boost − learn\n(η on top)")
  
  mids <- barplot(
    deltas,
    names.arg = labels,
    las = 2,
    ylab = expression(Delta*ELPD),
    main = title,
    ylim = ylim,
    border = NA,
    col = bar_col
  )
  
  arrows(mids, deltas - ses, mids, deltas + ses,
         angle = 90, code = 3, length = 0.05, lwd = 2)
  
  abline(h = 0, lty = 2, col = "gray40", lwd = 1.5)
}

# ===============================================================
# Group colours (your palette)
# ===============================================================
col_implicit_unaware <- rgb(0.56, 0.93, 0.56)
col_implicit_aware   <- rgb(0.00, 0.50, 0.00)
col_explicit_aware   <- rgb(1.00, 0.65, 0.00)

# ===============================================================
# Paths + titles (renamed to awareness labels)
# ===============================================================
paths <- list(
  implicit_unaware = list(
    title  = "Implicit Unaware",
    color  = col_implicit_unaware,
    simple = "./results/loo/loo_trunc_simplified_model_unaware_exp11.rdata",
    learn  = "./results/loo/loo_trunc_simplified_learning_unaware_exp11.rdata",
    boost  = "./results/loo/loo_trunc_simplified_boost_unaware_exp11.rdata"
  ),
  implicit_aware = list(
    title  = "Implicit Aware",
    color  = col_implicit_aware,
    simple = "./results/loo/loo_trunc_simplified_model_aware_exp11.rdata",
    learn  = "./results/loo/loo_trunc_simplified_learning_aware_exp11.rdata",
    boost  = "./results/loo/loo_trunc_simplified_boost_aware_exp11.rdata"
  ),
  explicit_aware = list(
    title  = "Explicit Aware",
    color  = col_explicit_aware,
    simple = "./results/loo/loo_trunc_simplified_model_aware_exp12.rdata",
    learn  = "./results/loo/loo_trunc_simplified_learning_aware_exp12.rdata",
    boost  = "./results/loo/loo_trunc_simplified_boost_aware_exp12.rdata"
  )
)

# ===============================================================
# Compute results + shared y-limits
# ===============================================================
res <- lapply(paths, \(p) group_contrasts(p$simple, p$learn, p$boost))

all_vals <- unlist(lapply(res, \(r) c(r$deltas - r$ses, r$deltas + r$ses, 0)))
ylim_shared <- range(all_vals) * 1.08

# ===============================================================
# Plot 3 panels
# ===============================================================
op <- par(mfrow = c(1, 3), mar = c(9, 5, 4, 1))
on.exit(par(op), add = TRUE)

plot_panel(paths$implicit_unaware$title, res$implicit_unaware$deltas, res$implicit_unaware$ses, ylim_shared, paths$implicit_unaware$color)
plot_panel(paths$implicit_aware$title,   res$implicit_aware$deltas,   res$implicit_aware$ses,   ylim_shared, paths$implicit_aware$color)
plot_panel(paths$explicit_aware$title,   res$explicit_aware$deltas,   res$explicit_aware$ses,   ylim_shared, paths$explicit_aware$color)

# Optional: print extracted values
cat("\n=== Pairwise ΔELPD ± SE extracted from loo objects ===\n")
for (nm in names(res)) {
  cat("\n", paths[[nm]]$title, "\n", sep = "")
  print(data.frame(
    contrast = c("learn-simple (δ only)", "boost-simple (δ+η)", "boost-learn (η on top)"),
    delta = res[[nm]]$deltas,
    se = res[[nm]]$ses
  ), row.names = FALSE)
}


rm(list=ls(all=TRUE))
library(loo)

load_loo <- function(path){
  e <- new.env(); load(path, envir=e)
  if(!exists("loo", envir=e)) stop("No `loo` in ", path)
  e$loo
}

plot_vs_best <- function(title, loo_simple, loo_learn, loo_boost, col, ylim=NULL){
  
  comp <- loo_compare(list(simple=loo_simple, learn=loo_learn, boost=loo_boost))
  d <- as.data.frame(comp)
  
  # order by best to worst (best first)
  d <- d[order(d$elpd_loo, decreasing=TRUE), , drop=FALSE]
  
  y  <- d$elpd_diff
  se <- d$se_diff
  labs <- rownames(d)
  
  if(is.null(ylim)){
    ylim <- range(c(y-se, y+se, 0)) * 1.1
  }
  
  mids <- barplot(y, names.arg=labs, las=2, main=title,
                  ylab=expression(Delta*ELPD~"(vs best)"),
                  col=col, border=NA, ylim=ylim)
  
  arrows(mids, y-se, mids, y+se, angle=90, code=3, length=0.05, lwd=2)
  abline(h=0, lty=2, col="gray40")
  invisible(list(comp=comp, ylim=ylim))
}

# colours (your palette)
col_unaware <- rgb(0.56,0.93,0.56)
col_iaware  <- rgb(0.00,0.50,0.00)
col_eaware  <- rgb(1.00,0.65,0.00)

# load loo objects
loo_u_simple <- load_loo("./results/loo/loo_trunc_simplified_model_unaware_exp11.rdata")
loo_u_learn  <- load_loo("./results/loo/loo_trunc_simplified_learning_unaware_exp11.rdata")
loo_u_boost  <- load_loo("./results/loo/loo_trunc_simplified_boost_unaware_exp11.rdata")

loo_ia_simple <- load_loo("./results/loo/loo_trunc_simplified_model_aware_exp11.rdata")
loo_ia_learn  <- load_loo("./results/loo/loo_trunc_simplified_learning_aware_exp11.rdata")
loo_ia_boost  <- load_loo("./results/loo/loo_trunc_simplified_boost_aware_exp11.rdata")

loo_ea_simple <- load_loo("./results/loo/loo_trunc_simplified_model_aware_exp12.rdata")
loo_ea_learn  <- load_loo("./results/loo/loo_trunc_simplified_learning_aware_exp12.rdata")
loo_ea_boost  <- load_loo("./results/loo/loo_trunc_simplified_boost_aware_exp12.rdata")

# compute shared ylim across all panels (so they compare nicely)
get_ylim <- function(looS, looL, looB){
  d <- as.data.frame(loo_compare(list(simple=looS, learn=looL, boost=looB)))
  y <- d$elpd_diff; se <- d$se_diff
  c(y-se, y+se, 0)
}
all_vals <- c(get_ylim(loo_u_simple, loo_u_learn, loo_u_boost),
              get_ylim(loo_ia_simple, loo_ia_learn, loo_ia_boost),
              get_ylim(loo_ea_simple, loo_ea_learn, loo_ea_boost))
ylim_shared <- range(all_vals) * 1.08

par(mfrow=c(1,3), mar=c(8,5,4,1))
plot_vs_best("Implicit Unaware", loo_u_simple, loo_u_learn, loo_u_boost, col_unaware, ylim_shared)
plot_vs_best("Implicit Aware",   loo_ia_simple, loo_ia_learn, loo_ia_boost, col_iaware,  ylim_shared)
plot_vs_best("Explicit Aware",   loo_ea_simple, loo_ea_learn, loo_ea_boost, col_eaware,  ylim_shared)
par(mfrow=c(1,1))


# ===============================================================
# Probability distortion curves (3 awareness groups)
# Uses posterior draws from Stan fits (CmdStanR .rds or cmdstan csv)
# Distortion: p' = inv_logit(alpha * logit(p) + beta)
# ===============================================================

rm(list = ls(all = TRUE))

library(posterior)
# cmdstanr is optional if you only readRDS CmdStanMCMC objects
# library(cmdstanr)

# -----------------------------
# Helpers
# -----------------------------
logit <- function(p) log(p / (1 - p))

# Load a CmdStanR fit saved as .rds
read_fit <- function(path) {
  if (!file.exists(path)) stop("Missing fit file: ", path)
  readRDS(path)
}

# Extract posterior draws for mu_alpha and mu_beta
# Works if generated quantities include mu_alpha/mu_beta,
# otherwise falls back to mu_pr[1], mu_pr[2] with your transforms:
#   mu_alpha = Phi(mu_pr[1]) * 6
#   mu_beta  = mu_pr[2]
extract_mu_alpha_beta <- function(fit) {
  # Try to grab mu_alpha/mu_beta directly
  vars_try <- c("mu_alpha", "mu_beta", "mu_pr[1]", "mu_pr[2]")
  draws <- posterior::as_draws_df(fit$draws(variables = vars_try))
  
  has_mu_alpha <- "mu_alpha" %in% names(draws)
  has_mu_beta  <- "mu_beta"  %in% names(draws)
  
  # Alpha
  if (has_mu_alpha) {
    alpha <- draws$mu_alpha
  } else if ("mu_pr[1]" %in% names(draws)) {
    alpha <- pnorm(draws$`mu_pr[1]`) * 6
  } else {
    stop("Couldn't find mu_alpha or mu_pr[1] in draws().")
  }
  
  # Beta
  if (has_mu_beta) {
    beta <- draws$mu_beta
  } else if ("mu_pr[2]" %in% names(draws)) {
    beta <- draws$`mu_pr[2]`
  } else {
    stop("Couldn't find mu_beta or mu_pr[2] in draws().")
  }
  
  list(alpha = as.numeric(alpha), beta = as.numeric(beta))
}

# Compute distortion curve summary from posterior draws
curve_from_draws <- function(alpha, beta, p_grid) {
  lg <- logit(p_grid)
  
  # matrix: ndraws x ngrid
  linpred <- tcrossprod(alpha, lg) + beta
  pprime  <- plogis(linpred)
  
  mu <- colMeans(pprime)
  lo <- apply(pprime, 2, quantile, probs = 0.025)
  hi <- apply(pprime, 2, quantile, probs = 0.975)
  
  list(mu = mu, lo = lo, hi = hi)
}

# Optional: summary at specific reliability points
points_from_draws <- function(alpha, beta, p_pts) {
  lg <- logit(p_pts)
  linpred <- tcrossprod(alpha, lg) + beta
  pprime  <- plogis(linpred)
  
  mu <- colMeans(pprime)
  lo <- apply(pprime, 2, quantile, probs = 0.025)
  hi <- apply(pprime, 2, quantile, probs = 0.975)
  
  data.frame(p = p_pts, mu = mu, lo = lo, hi = hi)
}

# -----------------------------
# SET YOUR FIT PATHS HERE
# Use best-fitting model per group if you want:
#   - Implicit Unaware: Simple (αβλ)
#   - Implicit Aware:   Boost  (αβλδη)
#   - Explicit Aware:   Boost  (αβλδη)
# -----------------------------
fit_paths <- list(
  implicit_unaware = "./results/fits/FIT_FILE_FOR_SIMPLE_unaware_exp11.rds",
  implicit_aware   = "./results/fits/FIT_FILE_FOR_BOOST_aware_exp11.rds",
  explicit_aware   = "./results/fits/FIT_FILE_FOR_BOOST_aware_exp12.rds"
)

# Colours (your palette)
cols <- list(
  implicit_unaware = rgb(0.56, 0.93, 0.56),
  implicit_aware   = rgb(0.00, 0.50, 0.00),
  explicit_aware   = rgb(1.00, 0.65, 0.00)
)

titles <- c(
  implicit_unaware = "Implicit Unaware",
  implicit_aware   = "Implicit Aware",
  explicit_aware   = "Explicit Aware"
)

# -----------------------------
# Build curves
# -----------------------------
p_grid <- seq(0.01, 0.99, length.out = 400)

fits <- lapply(fit_paths, read_fit)
pars <- lapply(fits, extract_mu_alpha_beta)

curves <- mapply(
  function(pr) curve_from_draws(pr$alpha, pr$beta, p_grid),
  pars,
  SIMPLIFY = FALSE
)

# Optional: reliability points (50/55/65)
p_pts <- c(0.50, 0.55, 0.65)
pt_summ <- mapply(
  function(pr) points_from_draws(pr$alpha, pr$beta, p_pts),
  pars,
  SIMPLIFY = FALSE
)

# -----------------------------
# Plot: single panel with 3 curves
# -----------------------------
png("probability_distortion_3groups.png", width = 1400, height = 900, res = 150)

plot(p_grid * 100, curves[[1]]$mu * 100,
     type = "n",
     xlab = "True reliability (%)",
     ylab = "Distorted reliability (%)",
     xlim = c(0, 100),
     ylim = c(0, 100),
     asp = 1,
     bty = "n")

abline(0, 1, lty = 2, col = "gray40", lwd = 2)

for (nm in names(curves)) {
  col <- cols[[nm]]
  cu  <- curves[[nm]]
  
  # 95% band
  polygon(
    x = c(p_grid, rev(p_grid)) * 100,
    y = c(cu$lo, rev(cu$hi)) * 100,
    col = adjustcolor(col, alpha.f = 0.20),
    border = NA
  )
  
  # mean line
  lines(p_grid * 100, cu$mu * 100, col = col, lwd = 3)
  
  # optional points at 50/55/65 with 95% CI
  pts <- pt_summ[[nm]]
  points(pts$p * 100, pts$mu * 100, pch = 16, cex = 1.2, col = col)
  arrows(pts$p * 100, pts$lo * 100, pts$p * 100, pts$hi * 100,
         angle = 90, code = 3, length = 0.03, lwd = 2, col = col)
}

legend("topleft",
       legend = titles[names(curves)],
       col = unlist(cols[names(curves)]),
       lwd = 3, bty = "n")

dev.off()

# -----------------------------
# Plot: 3-panel version (optional)
# -----------------------------
png("probability_distortion_3panels.png", width = 1700, height = 700, res = 150)

op <- par(mfrow = c(1, 3), mar = c(5, 5, 4, 1))
on.exit(par(op), add = TRUE)

for (nm in names(curves)) {
  col <- cols[[nm]]
  cu  <- curves[[nm]]
  
  plot(p_grid * 100, cu$mu * 100,
       type = "n",
       main = titles[[nm]],
       xlab = "True reliability (%)",
       ylab = "Distorted reliability (%)",
       xlim = c(0, 100),
       ylim = c(0, 100),
       asp = 1,
       bty = "n")
  
  abline(0, 1, lty = 2, col = "gray40", lwd = 2)
  
  polygon(c(p_grid, rev(p_grid)) * 100,
          c(cu$lo, rev(cu$hi)) * 100,
          col = adjustcolor(col, alpha.f = 0.20),
          border = NA)
  
  lines(p_grid * 100, cu$mu * 100, col = col, lwd = 3)
  
  pts <- pt_summ[[nm]]
  points(pts$p * 100, pts$mu * 100, pch = 16, cex = 1.2, col = col)
  arrows(pts$p * 100, pts$lo * 100, pts$p * 100, pts$hi * 100,
         angle = 90, code = 3, length = 0.03, lwd = 2, col = col)
}

dev.off()

cat("Saved:\n  probability_distortion_3groups.png\n  probability_distortion_3panels.png\n")