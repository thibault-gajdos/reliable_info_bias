
rm(list=ls(all=TRUE))  ## efface les données
setwd("/Users/bty/Documents/GitHub/reliable_info_bias")
source('utils.r')

library(rstan)
library(tidyverse)

exp <- 12

#################################################
##      COMPUTE AIC FOR LLO AND LINEAR
##      EXTRACT GROUP PARAMETERS
################################################

### GROUP Parameter ####
if (exp == 12){
  
  log_seq_basic_prior.group <- extract_group('../results/fits/Exp12/fit_trunc_simplified_learning2_aware_exp12.rds', parameters =  c('mu_alpha', 'mu_beta', "mu_delta", 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
  save(log_seq_basic_prior.group, file = '../results/summary/summary_group_log_trunc_simplified_aware_exp12.rdata')
  
}

### individual parameters ######
if (exp == 12){
  
  log_seq_basic_prior.individual <- extract_individual('../results/fits/Exp12/log_trunc_simplified_aware_exp12.rds', parameters =  c('alpha', 'beta', 'delta', 'w'))
  save(log_seq_basic_prior.individual, file = '../results/summary/summary_individual_log_trunc_simplified_aware_exp12.rdata')
  
} 


### individual parameters ######
if (exp == 12){
  
  log_seq_basic_prior.individual <- extract_individual('../results/fits/Exp12/log_trunc_simplified_aware_exp12.rds', parameters =  c('alpha', 'beta', 'delta', 'w'))
  save(log_seq_basic_prior.individual, file = '../results/summary/summary_individual_log_trunc_simplified_aware_exp12.rdata')
  
  ########################################################
  #   AWARENESS-SPECIFIC GROUP PARAMETER EXTRACTION

  
  awareness_groups <- c("aware", "unaware")
  
  for (grp in awareness_groups) {
    message("\n--- Processing awareness group: ", grp, " ---")
    
    # === .rds file path for the group ===
    fit_path <- paste0("../results/fits/Exp12/log_seq_basic_prior_", grp, "_exp12.rds")
    
    # === Output RData paths ===
    group_rdata_path      <- paste0("../results/summary/Exp12/summary_group_log_seq_basic_prior", grp, "_exp12.rdata")
    individual_rdata_path <- paste0("../results/summary/Exp12/summary_individual_log_seq_basic_prior", grp, "_exp12.rdata")
    
    # === Output CSV paths (optional) ===
    group_csv_path        <- paste0("param_group_prior_exp12_", grp, ".csv")
    individual_csv_path   <- paste0("param_individual_prior_exp12_", grp, ".csv")
    
    # === Extract GROUP-LEVEL parameters ===
    group_params <- extract_group(
      fit_path,
      parameters = c('mu_alpha', 'mu_beta', "mu_bias",
                     'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5',
                     'sigma', 'sigma_w')
    )
    group_varname <- paste0("group_params_", grp)
    assign(group_varname, group_params)
    save(list = group_varname, file = group_rdata_path)
    
    group_df <- group_params %>%
      mutate(model = 'log_seq_basic_prior_', group = grp)
    write.csv(group_df, file = group_csv_path, row.names = FALSE)
    
    # === Extract INDIVIDUAL-LEVEL parameters ===
    individual_params <- extract_individual(
      fit_path,
      parameters = c('alpha', 'beta', "bias", 'w')
    )
    indiv_varname <- paste0("individual_params_", grp)
    assign(indiv_varname, individual_params)
    save(list = indiv_varname, file = individual_rdata_path)
    
    individual_df <- individual_params %>%
      mutate(model = 'log_seq_basic_prior_', group = grp)
    write.csv(individual_df, file = individual_csv_path, row.names = FALSE)
  }
  
  
  
  
  rm(list = ls(all = TRUE))  ## efface les données
  setwd("/Users/bty615/Documents/GitHub/reliable_info_bias")
  source("utils.r")
  
  library(rstan)
  library(tidyverse)
  
  exp <- 12
  
  #################################################
  ##      COMPUTE AIC FOR LLO AND LINEAR
  ##      EXTRACT GROUP PARAMETERS
  #################################################
  
  ### GROUP parameters ####
  if (exp == 12){
    
    log_trunc_simplified.group <- extract_group(
      "../results/fits/Exp12/log_trunc_simplified_aware_exp12.rds",
      parameters = c(
        "mu_alpha",
        "mu_beta",
        "mu_lambda",
        "mu_theta",
        "mu_psi",
        "mu_deltaB",
        "mu_deltaR"
      )
    )
    
    save(
      log_trunc_simplified.group,
      file = "../results/summary/summary_group_log_trunc_simplified_aware_exp12.rdata"
    )
  }
  
  #################################################
  ##      EXTRACT INDIVIDUAL PARAMETERS
  #################################################
  
  ### individual parameters ####
  if (exp == 12){
    
    log_trunc_simplified.individual <- extract_individual(
      "../results/fits/Exp12/log_trunc_simplified_aware_exp12.rds",
      parameters = c(
        "alpha",
        "beta",
        "lambda",
        "theta",
        "psi",
        "deltaB",
        "deltaR"
      )
    )
    
    save(
      log_trunc_simplified.individual,
      file = "../results/summary/summary_individual_log_trunc_simplified_aware_exp12.rdata"
    )
  }