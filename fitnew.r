
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
save(fit, file = './results/fits/exp12/fit_trunc_simplified_simple_aware_exp11.rdata')

save(loo, file = './results/loo/loo_trunc_simplified_simple_aware_exp11.rdata')





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
# Update this path to your new fit file
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_simplified_learning_boost_unaware_exp11.rdata")

# -------------------------------------------------------------------
# 2. EXTRACT GROUP-LEVEL PARAMETERS (TRANSFORMED)
# -------------------------------------------------------------------
extract_mu <- function(fit, label) {
  
  d <- as_draws_df(fit$draws())
  
  # In the new model, we grab the explicitly named generated quantities.
  # These are already on the 'Physical' scale (0-1, 0-2, etc.)
  target_params <- c(
    "mu_alpha", 
    "mu_beta", 
    "mu_lambda", 
    "mu_delta", 
    "mu_eta"
  )
  
  # Ensure we only grab parameters that exist in the fit object
  existing_params <- intersect(target_params, colnames(d))
  
  mu_df <- d %>%
    select(all_of(existing_params)) %>%
    mutate(group = label) %>%
    pivot_longer(
      cols = -group,
      names_to = "param",
      values_to = "value"
    )
  
  mu_df
}

df_plot <- extract_mu(fit, "Hierarchical Boost Model")

# -------------------------------------------------------------------
# 3. PARAMETER LABELS (REFLECTING NEW ETA/DELTA LOGIC)
# -------------------------------------------------------------------
param_labels <- c(
  "mu_alpha"  = expression(paste(mu[alpha], " (alpha: 0-1)")),
  "mu_beta"   = expression(paste(mu[beta], " (beta)")),
  "mu_lambda" = expression(paste(mu[lambda], " (lambda: 0-1)")),
  "mu_delta"  = expression(paste(mu[delta], " (delta: 0-2)")),
  "mu_eta"    = expression(paste(mu[eta], " (eta)"))
)

model_color <- "#2E7D32"  # Forest green

# -------------------------------------------------------------------
# 4. POSTERIOR DISTRIBUTIONS
# -------------------------------------------------------------------
p_post <- ggplot(df_plot, aes(x = value)) +
  geom_histogram(
    fill = model_color,
    bins = 60,
    alpha = 0.7,
    color = "white",
    linewidth = 0.1
  ) +
  facet_wrap(
    ~ param,
    scales = "free",
    ncol = 3,
    labeller = as_labeller(param_labels, default = label_parsed)
  ) +
  theme_bw(base_size = 14) +
  theme(
    strip.background = element_rect(fill = "gray95"),
    strip.text       = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  ) +
  labs(
    title = "Transformed Group-Level Posteriors",
  
    x = "Parameter Value",
    y = "Density"
  )

print(p_post)

# -------------------------------------------------------------------
# 5. CORRELATION PAIRS PLOT
# -------------------------------------------------------------------
# Extract wide format for pairs plot
draws_wide <- as_draws_df(fit$draws()) %>%
  select(any_of(c("mu_alpha", "mu_beta", "mu_lambda", "mu_delta", "mu_eta")))

color_scheme_set("green")

p_corr <- mcmc_pairs(
  draws_wide,
  off_diag_args = list(size = 0.5, alpha = 0.2),
  diag_fun = "hist"
)

print(p_corr)

# -------------------------------------------------------------------
# 6. NUMERIC SUMMARY
# -------------------------------------------------------------------
cat("\n--- POSTERIOR SUMMARY 
summary_stats <- draws_wide %>%
  pivot_longer(cols = everything()) %>%
  group_by(name) %>%
  summarize(
    mean = mean(value),
    median = median(value),
    sd = sd(value),
    q5 = quantile(value, 0.05),
    q95 = quantile(value, 0.95)
  )

print(summary_stats)












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
# (Make sure these .rdata files contain the NEW 4-parameter 'fit' objects)
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/Exp12/fit_trunc_simplified_learning_aware_new_exp12.rdata")
fit_explicit <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/Exp12/fit_trunc_simplified_learning_aware_new_exp11.rdata")
fit_aware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/Exp12/fit_trunc_simplified_learning_unaware_new_exp11.rdata")
fit_unaware <- fit

# -------------------------------------------------------------------
# 2. Define Param Mapping (Stan index -> Descriptive name)
# -------------------------------------------------------------------
# Updated to exactly 4 parameters to match your new Stan model
param_map <- c(
  "mu_pr[1]" = "mu_alpha",
  "mu_pr[2]" = "mu_beta",
  "mu_pr[3]" = "mu_lambda",
  "mu_pr[4]" = "mu_delta"
)

# -------------------------------------------------------------------
# 3. Extraction Function (With Transformations)
# -------------------------------------------------------------------
extract_mu <- function(fit, label) {
  # Get draws as a data frame
  d <- as_draws_df(fit$draws())
  
  # 1. Handle potential naming differences ([1] vs .1.)
  # This finds which names from param_map exist in the columns
  actual_cols <- intersect(names(param_map), colnames(d))
  
  # If the names match Stan's internal "mu_pr.1." style instead of "mu_pr[1]"
  if(length(actual_cols) == 0) {
    # Specifically grab only the first 4 mu_pr columns
    mu_cols <- colnames(d)[grepl("mu_pr", colnames(d))]
    actual_cols <- mu_cols[1:4] 
  }
  
  mu_df <- d %>% select(all_of(actual_cols))
  
  # 2. Rename to internal names (mu_alpha, mu_beta, etc.)
  # We use the first 4 elements of param_map
  colnames(mu_df) <- unname(param_map[1:ncol(mu_df)])
  
  # 3. Apply the Stan transformations 
  mu_df <- mu_df %>%
    mutate(
      mu_alpha  = pnorm(mu_alpha),       # [0, 1]
      mu_lambda = pnorm(mu_lambda),      # [0, 1]
      mu_delta  = pnorm(mu_delta) * 2.0  # [0, 2]
      # mu_beta remains untransformed
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
  "mu_delta"  = "mu[delta]"
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
  facet_wrap(~param, scales = "free", ncol = 2, # Changed to 2 columns for 4 params
             labeller = as_labeller(param_labels, default = label_parsed)) +
  scale_fill_manual(values = custom_colors) +
  theme_bw(base_size = 14) +
  theme(legend.position = "bottom") +
  labs(title = "Transformed Posterior Parameter Distributions",
       subtitle = "4-Parameter Model: alpha, beta, lambda, delta [0,2]",
       x = "Parameter Value (Transformed Scale)", 
       y = "Frequency")
