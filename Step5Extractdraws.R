library(rstan)
library(posterior)
library(tidyverse)

extract_rstan_wide <- function(fit, group_label) {
  
  draws <- posterior::as_draws_df(fit)
  
  # detect parameter names (vector/matrix)
  vars <- names(draws)
  
  keep <- vars[
    grepl("^alpha\\[", vars) |
      grepl("^beta\\[",  vars) |
      grepl("^delta\\[", vars) |
      grepl("^w\\[",     vars)
  ]
  
  # keep only relevant parameters, wide format
  df <- draws %>%
    select(.chain, .iteration, .draw, all_of(keep)) %>%
    mutate(group = group_label)
  
  return(df)
}

# LOAD FITS
fit_unaware_11  <- readRDS("/Users/imogen/Documents/GitHub/reliable_info/results/fits/Exp11/log_seq_basic_prior_learning_unaware_exp11.rds")
fit_aware_11    <- readRDS("/Users/imogen/Documents/GitHub/reliable_info/results/fits/Exp11/log_seq_basic_prior_learning_aware_exp11.rds")
fit_explicit_12 <- readRDS("/Users/imogen/Documents/GitHub/reliable_info/results/fits/Exp12/log_seq_basic_prior_learning_aware_exp12.rds")

# EXTRACT WIDE
draws_unaware  <- extract_rstan_wide(fit_unaware_11,  "Implicit-Unaware")
draws_aware    <- extract_rstan_wide(fit_aware_11,    "Implicit-Aware")
draws_explicit <- extract_rstan_wide(fit_explicit_12, "Explicit-Aware")

# SAVE
write_csv(draws_unaware,  "posterior_unaware_exp11_wide.csv")
write_csv(draws_aware,    "posterior_aware_exp11_wide.csv")
write_csv(draws_explicit, "posterior_explicit_exp12_wide.csv")



