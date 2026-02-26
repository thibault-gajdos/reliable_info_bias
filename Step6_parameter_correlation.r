
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





