rm(list = ls(all = TRUE))

setwd("/Users/bty615/Documents/GitHub/reliable_info_bias")

library(cmdstanr)
library(posterior)
library(tidyverse)

exp <- 13

#################################################
##
##      LOAD MODEL FIT
##
#################################################

fit_path <- paste0(
  "stan/results/fits/exp11_unaware/",
  "fit_trunc_boost_deceptive_exp13.rdata"
)

if (!file.exists(fit_path)) {
  stop("Fit file not found: ", fit_path)
}

load(fit_path)

#################################################
##
##      CREATE OUTPUT FOLDER
##
#################################################

summary_folder <- "stan/results/summary/exp11_unaware"

dir.create(
  summary_folder,
  recursive = TRUE,
  showWarnings = FALSE
)

#################################################
##
##      EXTRACT GROUP PARAMETERS
##
#################################################

if (exp == 13) {
  
  log_trunc_boost.group <- fit$summary(
    variables = c(
      "mu_alpha",
      "mu_beta",
      "mu_lambda",
      "mu_delta",
      "mu_eta"
    ),
    "mean",
    "median",
    "sd",
    q2.5 = ~unname(quantile(.x, 0.025)),
    q97.5 = ~unname(quantile(.x, 0.975)),
    "rhat",
    "ess_bulk",
    "ess_tail"
  )
  
  save(
    log_trunc_boost.group,
    file = file.path(
      summary_folder,
      "summary_group_log_boost_deceptive_exp13.rdata"
    )
  )
}

#################################################
##
##      EXTRACT INDIVIDUAL PARAMETERS
##
#################################################

if (exp == 13) {
  
  log_trunc_boost.individual <- fit$summary(
    variables = "params",
    "mean",
    "median",
    "sd",
    q2.5 = ~unname(quantile(.x, 0.025)),
    q97.5 = ~unname(quantile(.x, 0.975)),
    "rhat",
    "ess_bulk",
    "ess_tail"
  ) %>%
    
    tidyr::extract(
      variable,
      into = c("participant", "parameter_number"),
      regex = "params\\[([0-9]+),([0-9]+)\\]",
      convert = TRUE
    ) %>%
    
    mutate(
      parameter = c(
        "alpha",
        "beta",
        "lambda",
        "delta",
        "eta"
      )[parameter_number]
    ) %>%
    
    arrange(
      participant,
      parameter_number
    ) %>%
    
    select(
      participant,
      parameter,
      mean,
      median,
      sd,
      q2.5,
      q97.5,
      rhat,
      ess_bulk,
      ess_tail
    )
  
  save(
    log_trunc_boost.individual,
    file = file.path(
      summary_folder,
      "summary_individual_log_trunc_boost_deceptive_exp13.rdata"
    )
  )
}

#################################################
##
##      DISPLAY RESULTS
##
#################################################

print(log_trunc_boost.group)
print(log_trunc_boost.individual)

cat("\nGroup results saved to:\n")
cat(
  file.path(
    summary_folder,
    "summary_group_log_trunc_boost_deceptive_exp13.rdata"
  ),
  "\n"
)

cat("\nIndividual results saved to:\n")
cat(
  file.path(
    summary_folder,
    "summary_individual_log_trunc_boost_deceptive_exp13.rdata"
  ),
  "\n"
)
  
  
#################################################
##
##      SAVE INDIVIDUAL PARAMETERS AS CSV
##
#################################################

individual_csv <- file.path(
  summary_folder,
  "param_individual_boost_deceptive_exp13.csv"
)

write_csv(
  log_trunc_boost.individual,
  individual_csv
)

cat("\nIndividual parameter CSV saved to:\n")
cat(individual_csv, "\n")

cat("\nCHECK:\n")
cat("Rows:", nrow(log_trunc_boost.individual), "\n")
cat(
  "Participants:",
  n_distinct(log_trunc_boost.individual$participant),
  "\n"
)

print(table(log_trunc_boost.individual$parameter))
  
  
  
  
  