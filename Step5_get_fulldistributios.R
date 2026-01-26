rm(list=ls(all=TRUE))## efface les données

library(rstan)
library(tidyverse)
library("writexl")

exp <- 11
if (exp==11){
  fit <- readRDS("/Users/Imogen/Documents/GitHub/reliable_info/results/fits/Exp11/log_seq_basic_prior_unaware_exp11.rds") 
}

params <- as.matrix(fit)
alpha <- params[, "mu_alpha"]
beta <- params[, "mu_beta"] 
bias <- params[, "mu_bias"] 
w1 <- params[, "mu_w1"]
w2 <- params[, "mu_w2"]
w3 <- params[, "mu_w3"] 
w4 <- params[, "mu_w4"]
w5 <- params[, "mu_w5"]
# hist(w1)
# summary(fit, 'mu_w1')

df <- data.frame(alpha,beta,bias,w1,w2,w3,w4,w5)

if (exp==11){
  write_xlsx(df,  "full_distributions_unaware_exp11.xlsx")
}




