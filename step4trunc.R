library(tidyverse)
library(kableExtra)

exp <- 12

setwd("/Users/Imogen/Documents/GitHub/reliable_info/results/summary")

############################################
# GROUP PARAMETERS
############################################
if (exp == 12){

  load(paste0('Exp',exp,'/summary_group_log_trunc_simplified_aware_exp12.rdata'))
  # object: log_trunc.group (as data.frame)

  group <- log_trunc.group %>%
    mutate(model = 'log_trunc_simplified_aware') %>%
    mutate(param = recode(param,
                          'mu_alpha'  = 'alpha',
                          'mu_beta'   = 'beta',
                          'mu_lambda' = 'lambda',
                          'mu_theta'  = 'theta',
                          'mu_psi'    = 'psi'))

  write.csv(group,
            file = 'param_group_log_trunc_simplified_aware_exp12.csv',
            row.names = FALSE)
}

############################################
# INDIVIDUAL PARAMETERS (renaming params[n,k])
############################################
if (exp == 12){

  load(paste0('Exp',exp,'/summary_individual_log_trunc_simplified_aware_exp12.rdata'))
  # object: log_trunc.individual (long format)

  # Extract subject and param index from e.g. params[12,3]
  individual <- log_trunc.individual %>%
    mutate(
      subject = as.integer(str_extract(param, "(?<=\\[)\\d+")),
      index   = as.integer(str_extract(param, "(?<=,)\\d+(?=\\])"))
    ) %>%
    mutate(parameter = case_when(
      index == 1 ~ "alpha",
      index == 2 ~ "beta",
      index == 3 ~ "lambda",
      index == 4 ~ "theta",
      index == 5 ~ "psi"
    )) %>%
    select(subject, parameter, mean, sd, `2.5%`, `50%`, `97.5%`) %>%
    pivot_wider(
      id_cols = subject,
      names_from = parameter,
      values_from = c(mean, sd, `2.5%`, `50%`, `97.5%`)
    ) %>%
    arrange(subject)

  write.csv(individual,
            file = 'param_individual_log_trunc_simplified_aware_exp12.csv',
            row.names = FALSE)
}