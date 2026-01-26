
rm(list=ls(all=TRUE))  ## efface les données
setwd(/Users/bty615/Documents/GitHub/)
source('utils.r')

library(rstan)
library(tidyverse)

exp <- 12

fit.all <- readRDS('../results/fits/Exp11/log_seq_basic_prior_exp11.rds')
#fit.aware <- readRDS('../results/fits/Exp11/log_seq_basic_prior_aware_exp11.rds')

###################################################
### create a matrix of MCMC output plots 
### A matrix of group (global) parameters##########
###################################################

# all participants
pairs(fit.all, pars = c('mu_bias','mu_alpha', 'mu_beta'))
pairs(fit.all, pars = c('mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5'))

# aware subgroup only
file.name <- paste('../results/fits/Exp11/corrplot_aware','.png', sep='')
png(file.name, width=1600,height=1600, res=300)
pairs(fit.aware, pars = c('mu_bias','mu_alpha', 'mu_beta'))
dev.off()

file.name <- paste('../results/fits/Exp11/corrplot_weight_aware','.png', sep='')
png(file.name, width=2000,height=2000, res=300)
pairs(fit.aware, pars = c('mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5'))
dev.off()









rm(list=ls(all=TRUE)) 

# 1. Set Working Directory (Ensure path is in quotes)
setwd("/Users/bty615/Documents/GitHub/reliable_info_bias")

# 2. Load Libraries
library(tidyverse)
library(cmdstanr) # You are using cmdstanr based on your previous snippet
library(bayesplot)

# 3. Load the Fit Object
# Note: Since you used cmdstanr to fit, use read_cmdstan_csv or load the .rdata
# Based on your previous code, you saved as .rdata:
load('./results/fits/exp12/fit_trunc_simplified_learning2_unaware_blue_exp11.rdata')

# If 'fit' is a cmdstanr object, we convert to a format 'pairs' or 'bayesplot' likes
# Extract draws for plotting
draws <- fit$draws()

###################################################
### CREATE MCMC OUTPUT PLOTS
###################################################

# Mapping of your Stan parameters to their descriptive names:
# mu_pr[1] -> alpha (reliability weight)
# mu_pr[2] -> beta (constant bias)
# mu_pr[3] -> lambda (recency)
# mu_pr[4] -> theta (sensitivity)
# mu_pr[5] -> psi (scaling)
# mu_pr[6] -> delta_B (blue learning)
# mu_pr[7] -> delta_R (red learning)

# Define the global hyper-parameters (mu_pr)
core_params <- c("mu_pr[1]", "mu_pr[2]", "mu_pr[3]", "mu_pr[4]")
learning_params <- c("mu_pr[5]", "mu_pr[6]", "mu_pr[7]")

# --- Plotting to Screen ---
# Pairs plot for core mechanisms
mcmc_pairs(draws, pars = core_params, off_diag_fun = "hex")

# --- Saving to Disk ---
# Create directory if it doesn't exist
dir.create('./results/plots/exp12/', recursive = TRUE, showWarnings = FALSE)

# Correlation plot for Learning Parameters
file_name <- './results/plots/exp12/corrplot_learning.png'
png(file_name, width=2000, height=2000, res=300)
print(mcmc_pairs(draws, pars = learning_params))
dev.off()

# Correlation plot for Sensory Parameters
file_name_2 <- './results/plots/exp12/corrplot_sensory.png'
png(file_name_2, width=2000, height=2000, res=300)
print(mcmc_pairs(draws, pars = core_params))
dev.off()

cat("Plots saved to ./results/plots/exp12/")





