rm(list=ls(all=TRUE)) ## efface les données

library(rstan)
library(tidyverse)
library(writexl)

fit_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/exp12"

fit_files <- c(
  "fit_trunc_boost_model_unaware_exp11.rdata",
  "fit_trunc_boost_model_aware_exp11.rdata",
  "fit_trunc_boost_model_aware_exp12.rdata",
  "fit_trunc_boost_truthful_exp13.rdata",
  "fit_trunc_boost_deceptive_exp13.rdata"
)

wanted_params <- c("mu_alpha", "mu_beta", "mu_delta", "mu_eta", "mu_lambda")

find_params <- function(x) {
  
  test <- try(as.matrix(x), silent = TRUE)
  
  if (!inherits(test, "try-error")) {
    if (all(wanted_params %in% colnames(test))) {
      return(test)
    }
  }
  
  if (is.environment(x)) {
    for (name in ls(x)) {
      out <- find_params(get(name, envir = x))
      if (!is.null(out)) return(out)
    }
  }
  
  if (is.list(x)) {
    for (i in seq_along(x)) {
      out <- find_params(x[[i]])
      if (!is.null(out)) return(out)
    }
  }
  
  return(NULL)
}

for (file in fit_files) {
  
  rm(list = setdiff(ls(), c("fit_dir", "fit_files", "wanted_params", "find_params", "file")))
  
  loaded_names <- load(file.path(fit_dir, file))
  
  params <- NULL
  
  for (name in loaded_names) {
    params <- find_params(get(name))
    if (!is.null(params)) break
  }
  
  if (is.null(params)) {
    stop(paste("Could not find posterior draws in", file))
  }
  
  alpha <- params[, "mu_alpha"]
  beta <- params[, "mu_beta"]
  delta <- params[, "mu_delta"]
  eta <- params[, "mu_eta"]
  lambda <- params[, "mu_lambda"]
  
  df <- data.frame(alpha, beta, delta, eta, lambda)
  
  out_name <- paste0(
    "full_distributions_",
    tools::file_path_sans_ext(file),
    ".xlsx"
  )
  
  write_xlsx(df, file.path(fit_dir, out_name))
  
  print(paste("Saved:", out_name))
}