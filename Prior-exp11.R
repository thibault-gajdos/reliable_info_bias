rm(list=ls(all=TRUE))  ## efface les données
## source('~/thib/projects/tools/R_lib.r')
## setwd('~/thib/projects/reliable_info')
## source('~/thib/projects/reliable_info/utils.r')
library('tidyverse')
library(rstan)
library(bayesplot)
library(ggplot2)

##################################################
##        PREPARE THE DATA       
##################################################
setwd("/Users/Imogen/Documents/GitHub/reliable_info/data/")

load('data_priorbelief_exp11.rdata')
##  if Response  ResponseButtonOrder= 1:  blue->1, red->0
##  if Response  ResponseButtonOrder= 0:  blue->0, red->1
##  we recode: blue ->1, red ->2
data <- data %>%
  mutate(choice = case_when(
    (ResponseButtonOrder == 1 & Response == 0) ~ 2, # red
    (ResponseButtonOrder == 1 & Response == 1) ~ 1, # blue
    (ResponseButtonOrder == 0 & Response == 0) ~ 1, # blue
    (ResponseButtonOrder == 0 & Response == 1) ~ 2  # red
  )) %>%
  mutate_at(vars(starts_with("color")), ~ ifelse(. == "blue", 1, 2)) # 1 = blue, 2 = red


N = length(unique(data$ParticipantPrivateID))
T_max = max(data$TrialNumber)
I =  sum(grepl("color", names(data))) ## number of samples/trial
## compute trials by subject
d <- data %>%
  group_by(ParticipantPrivateID) %>%
  summarise(t_subjs = n())
t_subjs <- d$t_subjs
subjs <- unique(data$ParticipantPrivateID)

## Initialize data arrays
choice  <- array(-1, c(N, T_max))
color <- array( -1, c(N, T_max, I))
proba <- array(-1, c(N, T_max, I))


## Fill the arrays
for (n in 1:N) {
  t <- t_subjs[n] ## number of trials for subj i
  data_subj <- data %>% filter(ParticipantPrivateID == subjs[n])
  choice[n, 1:t] <- data_subj$choice
  for (k in 1:t) {
    for (i in 1:I) {
      color_var <- paste0("color_", i)
      proba_var <- paste0("proba_", i)
      color[n, k, i] <- data_subj[[color_var]][k]
      proba[n, k, i] <- data_subj[[proba_var]][k]/100
    }
  }
}




# Create the data_list
data_list <- list(
  N = N,
  T_max = T_max,
  I = I,
  Tsubj = t_subjs,
  color = color,
  proba = proba,
  choice = choice
  
)


################################################
##  FIT LOG   model
##############################################
setwd('/Users/imogen/Documents/GitHub/reliable_info/stan/')

##### log odds model, sequential effects, prior proba, fixed theta #####
fit <- stan('log_seq_basic_prior.stan',
            data = data_list,
            iter = 4000,
            warmup = 2000,
            chains = 4,
            cores = 4,
            init =  "random",
            seed = 12345,
            control = list(adapt_delta = .8,  max_treedepth = 12)
)
write_rds(fit, '../results/fits/log_seq_basic_prior_exp11.rds')
loo <- loo(fit,moment_match = T)
save(loo, file = '../results/loo/loo_log_seq_basic_prior_exp11.rdata')




setwd("/Users/Imogen/Documents/GitHub/reliable_info/results/fits/Exp11")
fit <- readRDS("log_seq_basic_prior_exp11.rds")
loo <- loo(fit,moment_match = T)

fit <- readRDS("log_seq_basic_prior_exp11.rds")


# pr <- extract(fit, 'mu_pr')
# histogram(pr[["mu_pr" ]][1])
# 
# launch_shinystan(fit)
# 
# 
# ##### the normative Bayesian model #####
# fit <- stan('normative_bayes.stan',
#             data = data_list,
#             iter = 4000,
#             warmup = 2000,
#             chains = 4,
#             cores = 4,
#             init =  "random",
#             seed = 12345,
#             control = list(adapt_delta = .8,  max_treedepth = 12)
# )
# write_rds(fit, '../results/fits/normative_bayes_exp3.rds')
# loo <- loo(fit)
# save(loo, file = '../results/loo/loo_normative_bayes_exp3.rdata')
# 
# # ##### linear with fixed theta #####
# # fit <- stan('linear_seq_basic.stan',
# #             data = data_list,
# #             iter = 4000,
# #             warmup = 2000,
# #             chains = 4,
# #             cores = 4,
# #             init =  "random",
# #             seed = 12345,
# #             control = list(adapt_delta = .8,  max_treedepth = 12)
# # )
# # write_rds(fit, '../results/fits/linear_seq_basic_exp3.rds')
# # loo <- loo(fit)
# # save(loo, file = '../results/loo/loo_linear_seq_basic_exp3.rdata')
# 
# 
# # log model with sequential effects (theta fixed to 1) 
# fit <- stan('log_seq_basic.stan',
#             data = data_list,
#             iter = 4000,
#             warmup = 2000,
#             chains = 4,
#             cores = 4,
#             init =  "random",
#             seed = 12345,
#             control = list(adapt_delta = .8,  max_treedepth = 12)
#             )
# write_rds(fit, '../results/fits/log_seq_basic_exp3.rds')
# loo <- loo(fit)
# save(loo, file = '../results/loo/loo_log_seq_basic_exp3.rdata')
# 
# 
# ##### linear in log odds with fixed theta (no sequential effects) #####
# fit <- stan('log_noseq_basic.stan',
#             data = data_list,
#             iter = 4000,
#             warmup = 2000,
#             chains = 4,
#             cores = 4,
#             init =  "random",
#             seed = 12345,
#             control = list(adapt_delta = .8,  max_treedepth = 12)
# )
# write_rds(fit, '../results/fits/log_noseq_basic_exp3.rds')
# loo <- loo(fit)
# save(loo, file = '../results/loo/log_noseq_basic_exp3.rdata')
# 
# 
# ##### linear in log odds with fixed theta (sequential effects only) #####
# fit <- stan('log_onlyseq_basic.stan',
#             data = data_list,
#             iter = 4000,
#             warmup = 2000,
#             chains = 4,
#             cores = 4,
#             init =  "random",
#             seed = 12345,
#             control = list(adapt_delta = .8,  max_treedepth = 12)
# )
# write_rds(fit, '../results/fits/log_onlyseq_basic_exp3.rds')
# loo <- loo(fit)
# save(loo, file = '../results/loo/log_onlyseq_basic_exp3.rdata')
# 