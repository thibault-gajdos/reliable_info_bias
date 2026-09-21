## rm(list=ls(all=TRUE))## efface les données

library(rstan)
library(tidyverse)

for (exp in c(2,3,4,6,7,11)){
  
  #################################################
  ##      COMPUTE AIC FOR LLO AND LINEAR
  ##      EXTRACT GROUP PARAMETERS
  ################################################
  
  # fit.name <- '../results/fits/Exp2/normative_bayes_exp2.rds'
  # fit.bayes <- readRDS(fit.name)
  # fit.name <- '../results/fits/Exp2/linear_seq_basic_exp2.rds'
  # fit.lin <- readRDS(fit.name)
  # fit.name <- '../results/fits/Exp2/log_seq_basic_exp2.rds'
  # fit.log <- readRDS(fit.name)
  # fit.bayes@model_pars
  # fit.lin@model_pars
  # fit.log@model_pars
  # a<- loo::extract_log_lik(fit.log,"sigma_w", merge_chains = TRUE)
  
  # alpha and alpha_raw how they differ
  
  
  ### GROUP Parameter ####
  if (exp == 2){
    
    normative_bayes.group <- extract_group('../results/fits/Exp2/normative_bayes_exp2.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(normative_bayes.group, file = '../results/summary/summary_group_normative_bayes_exp2.rdata')
    
    linear_seq_basic.group <- extract_group('../results/fits/Exp2/linear_seq_basic_exp2.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(linear_seq_basic.group, file = '../results/summary/summary_group_linear_seq_basic_exp2.rdata')
    
    log_seq_basic.group <- extract_group('../results/fits/Exp2/log_seq_basic_exp2.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_seq_basic.group, file = '../results/summary/summary_group_log_seq_basic_exp2.rdata')
    
    log_noseq_basic.group <- extract_group('../results/fits/Exp2/log_noseq_basic_exp2.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_noseq_basic.group, file = '../results/summary/summary_group_log_noseq_basic_exp2.rdata')
    
    log_onlyseq_basic.group <- extract_group('../results/fits/Exp2/log_onlyseq_basic_exp2.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_onlyseq_basic.group, file = '../results/summary/summary_group_log_onlyseq_basic_exp2.rdata')
    
  } else if (exp == 3) {
    
    normative_bayes.group <- extract_group('../results/fits/Exp3/normative_bayes_exp3.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(normative_bayes.group, file = '../results/summary/summary_group_normative_bayes_exp3.rdata')
    
    linear_seq_basic.group <- extract_group('../results/fits/Exp3/linear_seq_basic_exp3.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(linear_seq_basic.group, file = '../results/summary/summary_group_linear_seq_basic_exp3.rdata')
    
    log_seq_basic.group <- extract_group('../results/fits/Exp3/log_seq_basic_exp3.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_seq_basic.group, file = '../results/summary/summary_group_log_seq_basic_exp3.rdata')
    
    log_noseq_basic.group <- extract_group('../results/fits/Exp3/log_noseq_basic_exp3.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_noseq_basic.group, file = '../results/summary/summary_group_log_noseq_basic_exp3.rdata')
    
    log_onlyseq_basic.group <- extract_group('../results/fits/Exp3/log_onlyseq_basic_exp3.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_onlyseq_basic.group, file = '../results/summary/summary_group_log_onlyseq_basic_exp3.rdata')
    
  } else if (exp == 4){
    
    normative_bayes.group <- extract_group('../results/fits/Exp4/normative_bayes_exp4.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(normative_bayes.group, file = '../results/summary/summary_group_normative_bayes_exp4.rdata')
    
    linear_seq_basic.group <- extract_group('../results/fits/Exp4/linear_seq_basic_exp4.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(linear_seq_basic.group, file = '../results/summary/summary_group_linear_seq_basic_exp4.rdata')
    
    log_seq_basic.group <- extract_group('../results/fits/Exp4/log_seq_basic_exp4.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w','sigma','sigma_w'))
    save(log_seq_basic.group, file = '../results/summary/summary_group_log_seq_basic_exp4.rdata')
    
    # log_seq_basic_2slopes.group <- extract_group('../results/fits/Exp4/log_seq_basic_2slopes_exp4.rds', parameters =  c('mu_alpha_pos', 'mu_alpha_neg', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    # save(log_seq_basic_2slopes.group, file = '../results/summary/summary_group_log_seq_basic_2slopes_exp4.rdata')
    # 
    # log_seq_basic_2offsets.group <- extract_group('../results/fits/Exp4/log_seq_basic_2offsets_exp4.rds', parameters =  c('mu_alpha_pos', 'mu_alpha_neg', 'mu_beta_pos', 'mu_beta_neg', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    # save(log_seq_basic_2offsets.group, file = '../results/summary/summary_group_log_seq_basic_2offsets_exp4.rdata')
    
    log_noseq_basic.group <- extract_group('../results/fits/Exp4/log_noseq_basic_exp4.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_noseq_basic.group, file = '../results/summary/summary_group_log_noseq_basic_exp4.rdata')
    
    log_onlyseq_basic.group <- extract_group('../results/fits/Exp4/log_onlyseq_basic_exp4.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_onlyseq_basic.group, file = '../results/summary/summary_group_log_onlyseq_basic_exp4.rdata')
    
  } else if (exp == 6){
    
    normative_bayes.group <- extract_group('../results/fits/Exp6/normative_bayes_exp6.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(normative_bayes.group, file = '../results/summary/summary_group_normative_bayes_exp6.rdata')
    
    linear_seq_basic.group <- extract_group('../results/fits/Exp6/linear_seq_basic_exp6.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(linear_seq_basic.group, file = '../results/summary/summary_group_linear_seq_basic_exp6.rdata')
    
    log_seq_basic.group <- extract_group('../results/fits/Exp6/log_seq_basic_exp6.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_seq_basic.group, file = '../results/summary/summary_group_log_seq_basic_exp6.rdata')
    
    log_noseq_basic.group <- extract_group('../results/fits/Exp6/log_noseq_basic_exp6.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_noseq_basic.group, file = '../results/summary/summary_group_log_noseq_basic_exp6.rdata')
    
    log_onlyseq_basic.group <- extract_group('../results/fits/Exp6/log_onlyseq_basic_exp6.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_onlyseq_basic.group, file = '../results/summary/summary_group_log_onlyseq_basic_exp6.rdata')
    
  } else if (exp == 7){
    
    normative_bayes.group <- extract_group('../results/fits/Exp7/normative_bayes_exp7.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(normative_bayes.group, file = '../results/summary/summary_group_normative_bayes_exp7.rdata')
    
    linear_seq_basic.group <- extract_group('../results/fits/Exp7/linear_seq_basic_exp7.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(linear_seq_basic.group, file = '../results/summary/summary_group_linear_seq_basic_exp7.rdata')
    
    log_seq_basic.group <- extract_group('../results/fits/Exp7/log_seq_basic_exp7.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_seq_basic.group, file = '../results/summary/summary_group_log_seq_basic_exp7.rdata')
    
    # log_seq_basic_2slopes.group <- extract_group('../results/fits/Exp7/log_seq_basic_2slopes_exp7.rds', parameters =  c('mu_alpha_pos', 'mu_alpha_neg', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    # save(log_seq_basic_2slopes.group, file = '../results/summary/summary_group_log_seq_basic_2slopes_exp7.rdata')
    # 
    # log_seq_basic_2offsets.group <- extract_group('../results/fits/Exp7/log_seq_basic_2offsets_exp7.rds', parameters =  c('mu_alpha_pos', 'mu_alpha_neg', 'mu_beta_pos', 'mu_beta_neg', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    # save(log_seq_basic_2offsets.group, file = '../results/summary/summary_group_log_seq_basic_2offsets_exp7.rdata')
    
    log_noseq_basic.group <- extract_group('../results/fits/Exp7/log_noseq_basic_exp7.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_noseq_basic.group, file = '../results/summary/summary_group_log_noseq_basic_exp7.rdata')
    
    log_onlyseq_basic.group <- extract_group('../results/fits/Exp7/log_onlyseq_basic_exp7.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_onlyseq_basic.group, file = '../results/summary/summary_group_log_onlyseq_basic_exp7.rdata')
    
  } else if (exp == 8){
    
    normative_bayes.group <- extract_group('../results/fits/Exp8/normative_bayes_exp8.rds', parameters =  c('mu_alpha', 'mu_beta','sigma'))
    save(normative_bayes.group, file = '../results/summary/summary_group_normative_bayes_exp8.rdata')
    
    linear_basic.group <- extract_group('../results/fits/Exp8/linear_basic_exp8.rds', parameters =  c('mu_alpha', 'mu_beta','sigma'))
    save(linear_basic.group, file = '../results/summary/summary_group_linear_basic_exp8.rdata')
    
    log_basic.group <- extract_group('../results/fits/Exp8/log_basic_exp8.rds', parameters =  c('mu_alpha', 'mu_beta','sigma'))
    save(log_basic.group, file = '../results/summary/summary_group_log_basic_exp8.rdata')
    
    log_basic_2slopes.group <- extract_group('../results/fits/Exp8/log_basic_2slopes_exp8.rds', parameters =  c('mu_alpha_pos', 'mu_alpha_neg','mu_beta','sigma'))
    save(log_basic_2slopes.group, file = '../results/summary/summary_group_log_basic_2slopes_exp8.rdata')
    
  } else if (exp == 11){

    log_seq_basic.group2 <- extract_group('../results/fits/Exp11/log_seq_basic_exp11.rds', parameters = c('mu_w','mu_pr','alpha','beta','w','alpha_raw','beta_raw','w_raw')) 

    log_seq_basic.group <- extract_group('../results/fits/Exp11/log_seq_basic_exp11.rds', parameters =  c('mu_alpha', 'mu_beta', 'mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5','sigma','sigma_w'))
    save(log_seq_basic.group, file = '../results/summary/summary_group_log_seq_basic_exp11.rdata')
  }
  
  runthis<-1
  if (runthis == 1){
    ### individual parameters ######
    if (exp == 2){
      
      normative_bayes.individual <- extract_individual('../results/fits/Exp2/normative_bayes_exp2.rds', parameters =  c('alpha', 'beta', 'w'))
      save(normative_bayes.individual, file = '../results/summary/summary_individual_normative_bayes_exp2.rdata')
      
      linear_seq_basic.individual <- extract_individual('../results/fits/Exp2/linear_seq_basic_exp2.rds', parameters =  c('alpha', 'beta', 'w'))
      save(linear_seq_basic.individual, file = '../results/summary/summary_individual_linear_seq_basic_exp2.rdata')
      
      log_seq_basic.individual <- extract_individual('../results/fits/Exp2/log_seq_basic_exp2.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_seq_basic.individual, file = '../results/summary/summary_individual_log_seq_basic_exp2.rdata')
      
      log_noseq_basic.individual <- extract_individual('../results/fits/Exp2/log_noseq_basic_exp2.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_noseq_basic.individual, file = '../results/summary/summary_individual_log_noseq_basic_exp2.rdata')
      
      log_onlyseq_basic.individual <- extract_individual('../results/fits/Exp2/log_onlyseq_basic_exp2.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_onlyseq_basic.individual, file = '../results/summary/summary_individual_log_onlyseq_basic_exp2.rdata')
      
    } else if (exp == 3) {
      
      normative_bayes.individual <- extract_individual('../results/fits/Exp3/normative_bayes_exp3.rds', parameters =  c('alpha', 'beta', 'w'))
      save(normative_bayes.individual, file = '../results/summary/summary_individual_normative_bayes_exp3.rdata')
      
      linear_seq_basic.individual <- extract_individual('../results/fits/Exp3/linear_seq_basic_exp3.rds', parameters =  c('alpha', 'beta', 'w'))
      save(linear_seq_basic.individual, file = '../results/summary/summary_individual_linear_seq_basic_exp3.rdata')
      
      log_seq_basic.individual <- extract_individual('../results/fits/Exp3/log_seq_basic_exp3.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_seq_basic.individual, file = '../results/summary/summary_individual_log_seq_basic_exp3.rdata')
      
      log_noseq_basic.individual <- extract_individual('../results/fits/Exp3/log_noseq_basic_exp3.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_noseq_basic.individual, file = '../results/summary/summary_individual_log_noseq_basic_exp3.rdata')
      
      log_onlyseq_basic.individual <- extract_individual('../results/fits/Exp3/log_onlyseq_basic_exp3.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_onlyseq_basic.individual, file = '../results/summary/summary_individual_log_onlyseq_basic_exp3.rdata')
      
    } else if (exp == 4){
      
      normative_bayes.individual <- extract_individual('../results/fits/Exp4/normative_bayes_exp4.rds', parameters =  c('alpha', 'beta', 'w'))
      save(normative_bayes.individual, file = '../results/summary/summary_individual_normative_bayes_exp4.rdata')
      
      linear_seq_basic.individual <- extract_individual('../results/fits/Exp4/linear_seq_basic_exp4.rds', parameters =  c('alpha', 'beta', 'w'))
      save(linear_seq_basic.individual, file = '../results/summary/summary_individual_linear_seq_basic_exp4.rdata')
      
      log_seq_basic.individual <- extract_individual('../results/fits/Exp4/log_seq_basic_exp4.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_seq_basic.individual, file = '../results/summary/summary_individual_log_seq_basic_exp4.rdata')
      
      # log_seq_basic_2slopes.individual <- extract_individual('../results/fits/Exp4/log_seq_basic_2slopes_exp4.rds', parameters =  c('alpha_pos', 'alpha_neg','beta', 'w'))
      # save(log_seq_basic_2slopes.individual, file = '../results/summary/summary_individual_log_seq_basic_2slopes_exp4.rdata')
      # 
      # log_seq_basic_2offsets.individual <- extract_individual('../results/fits/Exp4/log_seq_basic_2offsets_exp4.rds', parameters =  c('alpha_pos', 'alpha_neg','beta_pos', 'beta_neg', 'w'))
      # save(log_seq_basic_2offsets.individual, file = '../results/summary/summary_individual_log_seq_basic_2offsets_exp4.rdata')
      
      log_noseq_basic.individual <- extract_individual('../results/fits/Exp4/log_noseq_basic_exp4.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_noseq_basic.individual, file = '../results/summary/summary_individual_log_noseq_basic_exp4.rdata')
      
      log_onlyseq_basic.individual <- extract_individual('../results/fits/Exp4/log_onlyseq_basic_exp4.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_onlyseq_basic.individual, file = '../results/summary/summary_individual_log_onlyseq_basic_exp4.rdata')
      
    } else if (exp == 6){
      
      normative_bayes.individual <- extract_individual('../results/fits/Exp6/normative_bayes_exp6.rds', parameters =  c('alpha', 'beta', 'w'))
      save(normative_bayes.individual, file = '../results/summary/summary_individual_normative_bayes_exp6.rdata')
      
      linear_seq_basic.individual <- extract_individual('../results/fits/Exp6/linear_seq_basic_exp6.rds', parameters =  c('alpha', 'beta', 'w'))
      save(linear_seq_basic.individual, file = '../results/summary/summary_individual_linear_seq_basic_exp6.rdata')
      
      log_seq_basic.individual <- extract_individual('../results/fits/Exp6/log_seq_basic_exp6.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_seq_basic.individual, file = '../results/summary/summary_individual_log_seq_basic_exp6.rdata')
      
      log_noseq_basic.individual <- extract_individual('../results/fits/Exp6/log_noseq_basic_exp6.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_noseq_basic.individual, file = '../results/summary/summary_individual_log_noseq_basic_exp6.rdata')
      
      log_onlyseq_basic.individual <- extract_individual('../results/fits/Exp6/log_onlyseq_basic_exp6.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_onlyseq_basic.individual, file = '../results/summary/summary_individual_log_onlyseq_basic_exp6.rdata')
      
    } else if (exp == 7){
      
      normative_bayes.individual <- extract_individual('../results/fits/Exp7/normative_bayes_exp7.rds', parameters =  c('alpha', 'beta', 'w'))
      save(normative_bayes.individual, file = '../results/summary/summary_individual_normative_bayes_exp7.rdata')
      
      linear_seq_basic.individual <- extract_individual('../results/fits/Exp7/linear_seq_basic_exp7.rds', parameters =  c('alpha', 'beta', 'w'))
      save(linear_seq_basic.individual, file = '../results/summary/summary_individual_linear_seq_basic_exp7.rdata')
      
      log_seq_basic.individual <- extract_individual('../results/fits/Exp7/log_seq_basic_exp7.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_seq_basic.individual, file = '../results/summary/summary_individual_log_seq_basic_exp7.rdata')
      
      # log_seq_basic_2slopes.individual <- extract_individual('../results/fits/Exp7/log_seq_basic_2slopes_exp7.rds', parameters =  c('alpha_pos', 'alpha_neg','beta', 'w'))
      # save(log_seq_basic_2slopes.individual, file = '../results/summary/summary_individual_log_seq_basic_2slopes_exp7.rdata')
      # 
      # log_seq_basic_2offsets.individual <- extract_individual('../results/fits/Exp7/log_seq_basic_2offsets_exp7.rds', parameters =  c('alpha_pos', 'alpha_neg','beta_pos', 'beta_neg', 'w'))
      # save(log_seq_basic_2offsets.individual, file = '../results/summary/summary_individual_log_seq_basic_2offsets_exp7.rdata')
      
      log_noseq_basic.individual <- extract_individual('../results/fits/Exp7/log_noseq_basic_exp7.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_noseq_basic.individual, file = '../results/summary/summary_individual_log_noseq_basic_exp7.rdata')
      
      log_onlyseq_basic.individual <- extract_individual('../results/fits/Exp7/log_onlyseq_basic_exp7.rds', parameters =  c('alpha', 'beta', 'w'))
      save(log_onlyseq_basic.individual, file = '../results/summary/summary_individual_log_onlyseq_basic_exp7.rdata')
      
    } else if (exp == 8){
      
      normative_bayes.individual <- extract_individual('../results/fits/Exp8/normative_bayes_exp8.rds', parameters =  c('alpha', 'beta'))
      save(normative_bayes.individual, file = '../results/summary/summary_individual_normative_bayes_exp8.rdata')
      
      linear_basic.individual <- extract_individual('../results/fits/Exp8/linear_basic_exp8.rds', parameters =  c('alpha', 'beta'))
      save(linear_basic.individual, file = '../results/summary/summary_individual_linear_basic_exp8.rdata')
      
      log_basic.individual <- extract_individual('../results/fits/Exp8/log_basic_exp8.rds', parameters =  c('alpha', 'beta'))
      save(log_basic.individual, file = '../results/summary/summary_individual_log_basic_exp8.rdata')
      
      log_basic_2slopes.individual <- extract_individual('../results/fits/Exp8/log_basic_2slopes_exp8.rds', parameters =  c('alpha_pos', 'alpha_neg','beta'))
      save(log_basic_2slopes.individual, file = '../results/summary/summary_individual_log_basic_2slopes_exp8.rdata')
      
    }
    
  }
  
  
}



