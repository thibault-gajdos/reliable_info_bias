library(tidyverse)
library(kableExtra)

exp <- 12

########################################################
#    GROUP PARAMETERS
#######################################################
setwd("/Users/Imogen/Documents/GitHub/reliable_info/results/summary")

if (exp == 12){
  
  load(paste0('Exp',exp,'/summary_group_log_seq_basic_prior_learning_aware_exp12.rdata'))
  
  group <- log_seq_basic_prior.group %>%
    mutate(model = 'log_seq_basic_prior_learning_aware')
  
  write.csv(group,
            file = 'param_group_prior_learning_aware_exp12.csv',
            row.names = FALSE)
}

########################################################
#   INDIVIDUAL PARAMETERS
########################################################

if (exp == 12){
  
  load(paste0('Exp',exp,'/summary_individual_log_seq_basic_prior_learning_aware_exp12.rdata'))
  
  individual <- log_seq_basic_prior.individual %>%
    mutate(model = 'log_seq_basic_prior_learning_aware')
  
  write.csv(individual,
            file = 'param_individual_prior_learning_aware_exp12.csv',
            row.names = FALSE)
}

#   AWARENESS SUBGROUP PARAMETER EXPORTS
########################################################
#awareness_groups <- c("aware", "unaware")

#for (grp in awareness_groups) {

# === GROUP parameters ===
#group_path <- paste0('Exp', exp, '/summary_group_log_seq_basic_prior_', grp, '_exp11.rdata')
#load(group_path)  # loads: group_params_aware OR group_params_unaware

#group_var <- get(paste0("group_params_", grp)) %>%
#  mutate(model = 'log_seq_basic_prior', group = grp)
#write.csv(group_var, file = paste0('param_group_prior_exp11_', grp, '.csv'), row.names = FALSE)

# === INDIVIDUAL parameters ===
#indiv_path <- paste0('Exp', exp, '/summary_individual_log_seq_basic_prior_', grp, '_exp11.rdata')
#load(indiv_path)  # loads: individual_params_aware OR individual_params_unaware
#
#indiv_var <- get(paste0("individual_params_", grp)) %>%
#  mutate(model = 'log_seq_basic_prior', group = grp)
#write.csv(indiv_var, file = paste0('param_individual_prior_exp11_', grp, '.csv'), row.names = FALSE)
#}

########################################################
#   MODELS COMPARISON
#######################################################
setwd('/Users/Imogen/Documents/GitHub/reliable_info/results/loo')
library('loo')
if (exp == 12){
  load(paste0('Exp',exp,'/loo_log_seq_basic_prior_exp12.rdata')) 
  loo_log_seq_basic_prior <- loo
  
  load(paste0('Exp',exp,'/loo_log_seq_basic_exp12.rdata')) 
  loo_log_seq_basic <- loo
  
  comp = loo_compare(loo_log_sep_basic, loo_log_seq_basic_prior)
  # should be ordered from the worst,..., to the 2nd best, to the best model 
  
  print(comp, digits = 2, simplify = FALSE)
  kable(print(comp, digits = 2, simplify = FALSE), digits= 2)
  write.csv(comp, file = 'loo_comp_prior_exp12.csv')
  
} 
