# rm(list=ls(all=TRUE))## efface les données


library(tidyverse)
library(kableExtra)

exp <- 6

########################################################
#    GROUP PARAMETERS
#######################################################
setwd('D:/reliable_info/results/summary')

if (exp == 2){
  load(paste0('Exp',exp,'/summary_group_normative_bayes_exp2.rdata')) 
  normative_bayes.group <- normative_bayes.group %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_group_linear_seq_basic_exp2.rdata')) 
  linear_seq_basic.group <- linear_seq_basic.group %>% mutate(model = 'lin_seq_basic') 
    
  load(paste0('Exp',exp,'/summary_group_log_seq_basic_exp2.rdata')) 
  log_seq_basic.group <- log_seq_basic.group %>% mutate(model = 'log_seq_basic') 

  load(paste0('Exp',exp,'/summary_group_log_noseq_basic_exp2.rdata')) 
  log_noseq_basic.group <- log_noseq_basic.group %>% mutate(model = 'log_noseq_basic') 

  load(paste0('Exp',exp,'/summary_group_log_onlyseq_basic_exp2.rdata')) 
  log_onlyseq_basic.group <- log_onlyseq_basic.group %>% mutate(model = 'log_onlyseq_basic') 
  
  group <- rbind(normative_bayes.group, linear_seq_basic.group, log_seq_basic.group, log_noseq_basic.group, log_onlyseq_basic.group)
  write.csv(group, file = 'param_group_exp2.csv')
  
} else if (exp == 3) {
  load(paste0('Exp',exp,'/summary_group_normative_bayes_exp3.rdata')) 
  normative_bayes.group <- normative_bayes.group %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_group_linear_seq_basic_exp3.rdata')) 
  linear_seq_basic.group <- linear_seq_basic.group %>% mutate(model = 'lin_seq_basic') 
  
  load(paste0('Exp',exp,'/summary_group_log_seq_basic_exp3.rdata')) 
  log_seq_basic.group <- log_seq_basic.group %>% mutate(model = 'log_seq_basic') 
  
  # load(paste0('Exp',exp,'/summary_group_log_seq_basic_2slopes_exp3.rdata')) 
  # log_seq_basic_2slopes.group <- log_seq_basic_2slopes.group %>% mutate(model = '2slopes') 
  # 
  # load(paste0('Exp',exp,'/summary_group_log_seq_basic_2offsets_exp3.rdata')) 
  # log_seq_basic_2offsets.group <- log_seq_basic_2offsets.group %>% mutate(model = '2offsets') 
  
  load(paste0('Exp',exp,'/summary_group_log_noseq_basic_exp3.rdata')) 
  log_noseq_basic.group <- log_noseq_basic.group %>% mutate(model = 'log_noseq_basic') 
  
  load(paste0('Exp',exp,'/summary_group_log_onlyseq_basic_exp3.rdata')) 
  log_onlyseq_basic.group <- log_onlyseq_basic.group %>% mutate(model = 'log_onlyseq_basic') 
  
  group <- rbind(normative_bayes.group, linear_seq_basic.group, log_seq_basic.group, log_noseq_basic.group, log_onlyseq_basic.group)
  write.csv(group, file = 'param_group_exp3.csv')
  
} else if (exp == 4){
  load(paste0('Exp',exp,'/summary_group_normative_bayes_exp4.rdata')) 
  normative_bayes.group <- normative_bayes.group %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_group_linear_seq_basic_exp4.rdata')) 
  linear_seq_basic.group <- linear_seq_basic.group %>% mutate(model = 'lin_seq_basic') 
  
  load(paste0('Exp',exp,'/summary_group_log_seq_basic_exp4.rdata')) 
  log_seq_basic.group <- log_seq_basic.group %>% mutate(model = 'log_seq_basic') 
  
  # load(paste0('Exp',exp,'/summary_group_log_seq_basic_2slopes_exp4.rdata')) 
  # log_seq_basic_2slopes.group <- log_seq_basic_2slopes.group %>% mutate(model = '2slopes') 
  # 
  # load(paste0('Exp',exp,'/summary_group_log_seq_basic_2offsets_exp4.rdata')) 
  # log_seq_basic_2offsets.group <- log_seq_basic_2offsets.group %>% mutate(model = '2offsets') 
  
  load(paste0('Exp',exp,'/summary_group_log_noseq_basic_exp4.rdata')) 
  log_noseq_basic.group <- log_noseq_basic.group %>% mutate(model = 'log_noseq_basic') 
  
  load(paste0('Exp',exp,'/summary_group_log_onlyseq_basic_exp4.rdata')) 
  log_onlyseq_basic.group <- log_onlyseq_basic.group %>% mutate(model = 'log_onlyseq_basic') 
  
  group <- rbind(normative_bayes.group, linear_seq_basic.group, log_seq_basic.group, log_noseq_basic.group, log_onlyseq_basic.group)
  write.csv(group, file = 'param_group_exp4.csv')
  

} else if (exp == 6){
  load(paste0('Exp',exp,'/summary_group_normative_bayes_exp6.rdata')) 
  normative_bayes.group <- normative_bayes.group %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_group_linear_seq_basic_exp6.rdata')) 
  linear_seq_basic.group <- linear_seq_basic.group %>% mutate(model = 'lin_seq_basic') 
  
  load(paste0('Exp',exp,'/summary_group_log_seq_basic_exp6.rdata')) 
  log_seq_basic.group <- log_seq_basic.group %>% mutate(model = 'log_seq_basic') 
  
  load(paste0('Exp',exp,'/summary_group_log_noseq_basic_exp6.rdata')) 
  log_noseq_basic.group <- log_noseq_basic.group %>% mutate(model = 'log_noseq_basic') 
  
  load(paste0('Exp',exp,'/summary_group_log_onlyseq_basic_exp6.rdata')) 
  log_onlyseq_basic.group <- log_onlyseq_basic.group %>% mutate(model = 'log_onlyseq_basic') 
  
  group <- rbind(normative_bayes.group, linear_seq_basic.group, log_seq_basic.group, log_noseq_basic.group, log_onlyseq_basic.group)
  write.csv(group, file = 'param_group_exp6.csv')
  
} else if (exp == 7){
  load(paste0('Exp',exp,'/summary_group_normative_bayes_exp7.rdata')) 
  normative_bayes.group <- normative_bayes.group %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_group_linear_seq_basic_exp7.rdata')) 
  linear_seq_basic.group <- linear_seq_basic.group %>% mutate(model = 'lin_seq_basic') 

  load(paste0('Exp',exp,'/summary_group_log_seq_basic_exp7.rdata')) 
  log_seq_basic.group <- log_seq_basic.group %>% mutate(model = 'log_seq_basic') 
  

  # load(paste0('Exp',exp,'/summary_group_log_seq_basic_2slopes_exp7.rdata')) 
  # log_seq_basic_2slopes.group <- log_seq_basic_2slopes.group %>% mutate(model = '2slopes') 
  # 
  # load(paste0('Exp',exp,'/summary_group_log_seq_basic_2offsets_exp7.rdata')) 
  # log_seq_basic_2offsets.group <- log_seq_basic_2offsets.group %>% mutate(model = '2offsets') 
  
  
  load(paste0('Exp',exp,'/summary_group_log_noseq_basic_exp7.rdata')) 
  log_noseq_basic.group <- log_noseq_basic.group %>% mutate(model = 'log_noseq_basic') 
  
  load(paste0('Exp',exp,'/summary_group_log_onlyseq_basic_exp7.rdata')) 
  log_onlyseq_basic.group <- log_onlyseq_basic.group %>% mutate(model = 'log_onlyseq_basic') 
  
  group <- rbind(normative_bayes.group, linear_seq_basic.group, log_seq_basic.group, log_noseq_basic.group, log_onlyseq_basic.group)
  write.csv(group, file = 'param_group_exp7.csv')
  
}  else if (exp == 8){
  load(paste0('Exp',exp,'/summary_group_normative_bayes_exp8.rdata')) 
  normative_bayes.group <- normative_bayes.group %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_group_linear_basic_exp8.rdata')) 
  linear_basic.group <- linear_basic.group %>% mutate(model = 'lin_basic') 
  
  load(paste0('Exp',exp,'/summary_group_log_basic_exp8.rdata')) 
  log_basic.group <- log_basic.group %>% mutate(model = 'log_basic') 
  
  load(paste0('Exp',exp,'/summary_group_log_basic_2slopes_exp8.rdata')) 
  log_basic_2slopes.group <- log_basic_2slopes.group %>% mutate(model = 'log_basic_2slopes') 
  
  group <- rbind(normative_bayes.group, linear_basic.group, log_basic.group, log_basic_2slopes.group)
  write.csv(group, file = 'param_group_exp8.csv')
  
}


########################################################
#   INDIVIDUAL PARAMETERS
########################################################
if (exp == 2){
  load(paste0('Exp',exp,'/summary_individual_normative_bayes_exp2.rdata')) 
  normative_bayes.individual <- normative_bayes.individual %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_individual_linear_seq_basic_exp2.rdata')) 
  linear_seq_basic.individual <- linear_seq_basic.individual %>% mutate(model = 'lin_seq_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_seq_basic_exp2.rdata')) 
  log_seq_basic.individual <- log_seq_basic.individual %>% mutate(model = 'log_seq_basic') 

  load(paste0('Exp',exp,'/summary_individual_log_noseq_basic_exp2.rdata')) 
  log_noseq_basic.individual <- log_noseq_basic.individual %>% mutate(model = 'log_noseq_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_onlyseq_basic_exp2.rdata')) 
  log_onlyseq_basic.individual <- log_onlyseq_basic.individual %>% mutate(model = 'log_onlyseq_basic') 
  
  individual <- rbind(normative_bayes.individual, linear_seq_basic.individual, log_seq_basic.individual, log_noseq_basic.individual, log_onlyseq_basic.individual)
  write.csv(individual, file = 'param_individual_exp2.csv')
  
} else if (exp == 3){
  load(paste0('Exp',exp,'/summary_individual_normative_bayes_exp3.rdata')) 
  normative_bayes.individual <- normative_bayes.individual %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_individual_linear_seq_basic_exp3.rdata')) 
  linear_seq_basic.individual <- linear_seq_basic.individual %>% mutate(model = 'lin_seq_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_seq_basic_exp3.rdata')) 
  log_seq_basic.individual <- log_seq_basic.individual %>% mutate(model = 'log_seq_basic') 
  
  # load(paste0('Exp',exp,'/summary_individual_log_seq_basic_2slopes_exp3.rdata')) 
  # log_seq_basic_2slopes.individual <- log_seq_basic_2slopes.individual %>% mutate(model = '2slopes') 
  # 
  # load(paste0('Exp',exp,'/summary_individual_log_seq_basic_2offsets_exp3.rdata')) 
  # log_seq_basic_2offsets.individual <- log_seq_basic_2offsets.individual %>% mutate(model = '2offsets') 
  
  load(paste0('Exp',exp,'/summary_individual_log_noseq_basic_exp3.rdata')) 
  log_noseq_basic.individual <- log_noseq_basic.individual %>% mutate(model = 'log_noseq_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_onlyseq_basic_exp3.rdata')) 
  log_onlyseq_basic.individual <- log_onlyseq_basic.individual %>% mutate(model = 'log_onlyseq_basic') 
  
  individual <- rbind(normative_bayes.individual, linear_seq_basic.individual, log_seq_basic.individual, log_noseq_basic.individual, log_onlyseq_basic.individual)
  write.csv(individual, file = 'param_individual_exp3.csv')
  
}  else if (exp == 4){
  load(paste0('Exp',exp,'/summary_individual_normative_bayes_exp4.rdata')) 
  normative_bayes.individual <- normative_bayes.individual %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_individual_linear_seq_basic_exp4.rdata')) 
  linear_seq_basic.individual <- linear_seq_basic.individual %>% mutate(model = 'lin_seq_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_seq_basic_exp4.rdata')) 
  log_seq_basic.individual <- log_seq_basic.individual %>% mutate(model = 'log_seq_basic') 
  
  # load(paste0('Exp',exp,'/summary_individual_log_seq_basic_2slopes_exp4.rdata')) 
  # log_seq_basic_2slopes.individual <- log_seq_basic_2slopes.individual %>% mutate(model = '2slopes') 
  # 
  # load(paste0('Exp',exp,'/summary_individual_log_seq_basic_2offsets_exp4.rdata')) 
  # log_seq_basic_2offsets.individual <- log_seq_basic_2offsets.individual %>% mutate(model = '2offsets') 

  load(paste0('Exp',exp,'/summary_individual_log_noseq_basic_exp4.rdata')) 
  log_noseq_basic.individual <- log_noseq_basic.individual %>% mutate(model = 'log_noseq_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_onlyseq_basic_exp4.rdata')) 
  log_onlyseq_basic.individual <- log_onlyseq_basic.individual %>% mutate(model = 'log_onlyseq_basic') 
  
  individual <- rbind(normative_bayes.individual, linear_seq_basic.individual, log_seq_basic.individual, log_noseq_basic.individual, log_onlyseq_basic.individual)
  write.csv(individual, file = 'param_individual_exp4.csv')
  
} else if (exp == 6){
  load(paste0('Exp',exp,'/summary_individual_normative_bayes_exp6.rdata')) 
  normative_bayes.individual <- normative_bayes.individual %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_individual_linear_seq_basic_exp6.rdata')) 
  linear_seq_basic.individual <- linear_seq_basic.individual %>% mutate(model = 'lin_seq_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_seq_basic_exp6.rdata')) 
  log_seq_basic.individual <- log_seq_basic.individual %>% mutate(model = 'log_seq_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_noseq_basic_exp6.rdata')) 
  log_noseq_basic.individual <- log_noseq_basic.individual %>% mutate(model = 'log_noseq_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_onlyseq_basic_exp6.rdata')) 
  log_onlyseq_basic.individual <- log_onlyseq_basic.individual %>% mutate(model = 'log_onlyseq_basic') 
  
  individual <- rbind(normative_bayes.individual, linear_seq_basic.individual, log_seq_basic.individual, log_noseq_basic.individual, log_onlyseq_basic.individual)
  write.csv(individual, file = 'param_individual_exp6.csv')
  
  individual <- rbind(normative_bayes.individual, linear_seq_basic.individual, log_seq_basic.individual)
  write.csv(individual, file = 'param_individual_exp6.csv')
  
} else if (exp == 7){
  load(paste0('Exp',exp,'/summary_individual_normative_bayes_exp7.rdata')) 
  normative_bayes.individual <- normative_bayes.individual %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_individual_linear_seq_basic_exp7.rdata')) 
  linear_seq_basic.individual <- linear_seq_basic.individual %>% mutate(model = 'lin_seq_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_seq_basic_exp7.rdata')) 
  log_seq_basic.individual <- log_seq_basic.individual %>% mutate(model = 'log_seq_basic') 
  
  # load(paste0('Exp',exp,'/summary_individual_log_seq_basic_2slopes_exp7.rdata')) 
  # log_seq_basic_2slopes.individual <- log_seq_basic_2slopes.individual %>% mutate(model = '2slopes') 
  # 
  # load(paste0('Exp',exp,'/summary_individual_log_seq_basic_2offsets_exp7.rdata')) 
  # log_seq_basic_2offsets.individual <- log_seq_basic_2offsets.individual %>% mutate(model = '2offsets') 
   
  load(paste0('Exp',exp,'/summary_individual_log_noseq_basic_exp7.rdata')) 
  log_noseq_basic.individual <- log_noseq_basic.individual %>% mutate(model = 'log_noseq_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_onlyseq_basic_exp7.rdata')) 
  log_onlyseq_basic.individual <- log_onlyseq_basic.individual %>% mutate(model = 'log_onlyseq_basic') 
  
  individual <- rbind(normative_bayes.individual, linear_seq_basic.individual, log_seq_basic.individual, log_noseq_basic.individual, log_onlyseq_basic.individual)
  write.csv(individual, file = 'param_individual_exp7.csv')
  
} else if (exp == 8){
  load(paste0('Exp',exp,'/summary_individual_normative_bayes_exp8.rdata')) 
  normative_bayes.individual <- normative_bayes.individual %>% mutate(model = 'normative_bayes') 
  
  load(paste0('Exp',exp,'/summary_individual_linear_basic_exp8.rdata')) 
  linear_basic.individual <- linear_basic.individual %>% mutate(model = 'lin_basic') 
  
  load(paste0('Exp',exp,'/summary_individual_log_basic_exp8.rdata')) 
  log_basic.individual <- log_basic.individual %>% mutate(model = 'log_basic') 

  load(paste0('Exp',exp,'/summary_individual_log_basic_2slopes_exp8.rdata')) 
  log_basic_2slopes.individual <- log_basic_2slopes.individual %>% mutate(model = 'log_basic_2slopes') 
  
  individual <- rbind(normative_bayes.individual, linear_basic.individual, log_basic.individual, log_basic_2slopes.individual)
  write.csv(individual, file = 'param_individual_exp8.csv')

} 

########################################################
#   MODELS COMPARISON
#######################################################
setwd('D:/reliable_info/results/loo')
library('loo')
if (exp == 2){
  load(paste0('Exp',exp,'/loo_normative_bayes_exp2.rdata')) 
  loo_normative_bayes <- loo
  load(paste0('Exp',exp,'/loo_linear_seq_basic_exp2.rdata')) 
  loo_lin_seq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_seq_basic_exp2.rdata')) 
  loo_log_seq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_noseq_basic_exp2.rdata')) 
  loo_log_noseq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_onlyseq_basic_exp2.rdata')) 
  loo_log_onlyseq_basic <- loo
  
  comp = loo_compare(loo_normative_bayes, loo_log_onlyseq_basic, loo_lin_seq_basic,  loo_log_noseq_basic, loo_log_seq_basic)
  # should be ordered from the worst,..., to the 2nd best, to the best model 
  
  print(comp, digits = 2, simplify = FALSE)
  kable(print(comp, digits = 2, simplify = FALSE), digits= 2)
  write.csv(comp, file = 'loo_comp_exp2.csv')
  
} else if (exp == 3){
  load(paste0('Exp',exp,'/loo_normative_bayes_exp3.rdata')) 
  loo_normative_bayes <- loo
  load(paste0('Exp',exp,'/loo_linear_seq_basic_exp3.rdata')) 
  loo_lin_seq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_seq_basic_exp3.rdata')) 
  loo_log_seq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_noseq_basic_exp3.rdata')) 
  loo_log_noseq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_onlyseq_basic_exp3.rdata')) 
  loo_log_onlyseq_basic <- loo
  
  comp = loo_compare(loo_normative_bayes, loo_log_onlyseq_basic, loo_lin_seq_basic,  loo_log_noseq_basic, loo_log_seq_basic)
  
  # should be ordered from the worst,..., to the 2nd best, to the best model 
  
  print(comp, digits = 2, simplify = FALSE)
  kable(print(comp, digits = 2, simplify = FALSE), digits= 2)
  write.csv(comp, file = 'loo_comp_exp3.csv')
  
} else if (exp == 4){
  load(paste0('Exp',exp,'/loo_normative_bayes_exp4.rdata')) 
  loo_normative_bayes <- loo
  load(paste0('Exp',exp,'/loo_linear_seq_basic_exp4.rdata')) 
  loo_lin_seq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_seq_basic_exp4.rdata')) 
  loo_log_seq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_noseq_basic_exp4.rdata')) 
  loo_log_noseq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_onlyseq_basic_exp4.rdata')) 
  loo_log_onlyseq_basic <- loo
  
  comp = loo_compare(loo_normative_bayes, loo_log_onlyseq_basic, loo_lin_seq_basic,  loo_log_noseq_basic, loo_log_seq_basic)
  
  # should be ordered from the worst,..., to the 2nd best, to the best model 
  
  print(comp, digits = 2, simplify = FALSE)
  kable(print(comp, digits = 2, simplify = FALSE), digits= 2)
  write.csv(comp, file = 'loo_comp_exp4.csv')
  
} else if (exp == 6){
    load(paste0('Exp',exp,'/loo_normative_bayes_exp6.rdata')) 
    loo_normative_bayes <- loo
    load(paste0('Exp',exp,'/loo_linear_seq_basic_exp6.rdata')) 
    loo_lin_seq_basic <- loo
    load(paste0('Exp',exp,'/loo_log_seq_basic_exp6.rdata')) 
    loo_log_seq_basic <- loo
    load(paste0('Exp',exp,'/loo_log_noseq_basic_exp6.rdata')) 
    loo_log_noseq_basic <- loo
    load(paste0('Exp',exp,'/loo_log_onlyseq_basic_exp6.rdata')) 
    loo_log_onlyseq_basic <- loo
    
    comp = loo_compare(loo_normative_bayes, loo_log_onlyseq_basic, loo_lin_seq_basic,  loo_log_noseq_basic, loo_log_seq_basic)
    
    # should be ordered from the worst,..., to the 2nd best, to the best model 
    
    print(comp, digits = 2, simplify = FALSE)
    kable(print(comp, digits = 2, simplify = FALSE), digits= 2)
    write.csv(comp, file = 'loo_comp_exp6.csv')

} else if (exp == 7){
  load(paste0('Exp',exp,'/loo_normative_bayes_exp7.rdata')) 
  loo_normative_bayes <- loo
  load(paste0('Exp',exp,'/loo_linear_seq_basic_exp7.rdata')) 
  loo_lin_seq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_seq_basic_exp7.rdata')) 
  loo_log_seq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_noseq_basic_exp7.rdata')) 
  loo_log_noseq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_onlyseq_basic_exp7.rdata')) 
  loo_log_onlyseq_basic <- loo
  
  comp = loo_compare(loo_normative_bayes, loo_log_onlyseq_basic, loo_lin_seq_basic,  loo_log_noseq_basic, loo_log_seq_basic)
  
  # should be ordered from the worst,..., to the 2nd best, to the best model 
  
  print(comp, digits = 2, simplify = FALSE)
  kable(print(comp, digits = 2, simplify = FALSE), digits= 2)
  write.csv(comp, file = 'loo_comp_exp7.csv')
} else if (exp == 8){
  load(paste0('Exp',exp,'/loo_normative_bayes_exp8.rdata')) 
  loo_normative_bayes <- loo
  load(paste0('Exp',exp,'/loo_linear_basic_exp8.rdata')) 
  loo_lin_seq_basic <- loo
  load(paste0('Exp',exp,'/loo_log_basic_exp8.rdata')) 
  loo_log_seq_basic <- loo
  
  comp = loo_compare(loo_normative_bayes, loo_lin_seq_basic, loo_log_seq_basic)
  # should be ordered from the worst,..., to the 2nd best, to the best model 
  
  print(comp, digits = 2, simplify = FALSE)
  kable(print(comp, digits = 2, simplify = FALSE), digits= 2)
  write.csv(comp, file = 'loo_comp_exp8.csv')
}






