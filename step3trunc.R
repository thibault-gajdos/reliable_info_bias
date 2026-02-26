# ===============================================================
# Extract GROUP + INDIVIDUAL parameter CSVs with columns:
# param mean se_mean sd 2.50% 50% 97.50% n_eff Rhat AIC l model
# Works with CmdStanMCMC stored in .rdata under ./results/fits/...
# ===============================================================

rm(list = ls(all = TRUE))
setwd("/Users/bty615/Documents/GitHub/reliable_info_bias")

suppressPackageStartupMessages({
  library(cmdstanr)
  library(posterior)
  library(loo)
  library(dplyr)
  library(tibble)
  library(stringr)
})

fits_root <- file.path(getwd(), "results", "fits")
out_dir   <- file.path(getwd(), "results", "summary", "params")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---------------------------------------------------------------
# Load CmdStanMCMC from .rdata (whatever the object name is)
# ---------------------------------------------------------------
load_cmdstan_from_rdata <- function(path) {
  e <- new.env()
  obj_names <- load(path, envir = e)
  for (nm in obj_names) {
    obj <- e[[nm]]
    if (inherits(obj, "CmdStanMCMC")) return(obj)
  }
  stop("No CmdStanMCMC object found inside: ", path,
       "\nObjects were: ", paste(obj_names, collapse = ", "))
}

# ---------------------------------------------------------------
# Summarise draws into EXACT columns you want
# ---------------------------------------------------------------
summarise_draws_table <- function(draws) {
  # draws: posterior draws object (keeps chain structure => ESS/Rhat works)
  df <- posterior::summarise_draws(
    draws,
    mean     = mean,
    se_mean  = posterior::mcse_mean,
    sd       = sd,
    `2.50%`  = function(x) unname(stats::quantile(x, 0.025)),
    `50%`    = function(x) unname(stats::quantile(x, 0.50)),
    `97.50%` = function(x) unname(stats::quantile(x, 0.975)),
    n_eff    = posterior::ess_bulk,
    Rhat     = posterior::rhat
  )
  as.data.frame(df)
}

# ---------------------------------------------------------------
# Compute l and AIC from log_lik
# l = elpd_loo estimate (matches your old utils intent)
# AIC = -2*l + 2*k
# ---------------------------------------------------------------
compute_l_and_AIC <- function(fit, k_params) {
  # log_lik draws -> matrix draws x Nobs
  ll_draws <- fit$draws(variables = "log_lik")
  ll_mat   <- posterior::as_draws_matrix(ll_draws)
  
  el <- loo::elpd(ll_mat)$estimates
  # robust: grab elpd_loo estimate if present
  if ("elpd_loo" %in% rownames(el)) {
    l <- as.numeric(el["elpd_loo", "Estimate"])
  } else {
    # fallback: first estimate
    l <- as.numeric(el[1, "Estimate"])
  }
  AIC <- -2 * l + 2 * k_params
  list(l = l, AIC = AIC)
}

# ---------------------------------------------------------------
# Extract GROUP parameters CSV for one fit
# ---------------------------------------------------------------
extract_group_csv <- function(fit, params_needed, model_name) {
  
  # parameter summaries
  draws <- fit$draws(variables = params_needed)
  tab <- summarise_draws_table(draws)
  
  # cmdstanr uses "variable" column name
  tab <- tab %>% rename(param = variable)
  
  # l + AIC
  la <- compute_l_and_AIC(fit, k_params = length(params_needed))
  
  tab$AIC  <- la$AIC
  tab$l    <- la$l
  tab$model <- model_name
  
  # Keep column order exactly as requested
  tab %>%
    select(param, mean, se_mean, sd, `2.50%`, `50%`, `97.50%`, n_eff, Rhat, AIC, l, model)
}

# ---------------------------------------------------------------
# Extract INDIVIDUAL parameters from params[subject,k] or ind_params[subject,k]
# Returns same summary columns + subject + parameter label
# ---------------------------------------------------------------
extract_individual_csv <- function(fit, model_name) {
  
  # Decide K map by model
  k_map <- switch(model_name,
                  model    = c("alpha","beta","lambda"),
                  learning = c("alpha","beta","lambda","delta"),
                  boost    = c("alpha","beta","lambda","delta","eta"),
                  stop("Unknown model: ", model_name))
  
  # Prefer "params", else "ind_params"
  vars_all <- posterior::variables(fit$draws())
  mat_name <- if (any(grepl("^params\\[", vars_all))) "params" else
    if (any(grepl("^ind_params\\[", vars_all))) "ind_params" else
      stop("No params[ , ] or ind_params[ , ] found in draws() for model: ", model_name)
  
  draws <- fit$draws(variables = mat_name)
  tab <- summarise_draws_table(draws) %>% as_tibble()
  
  tab <- tab %>% rename(param = variable)
  
  # parse subject,k from "params[12,3]"
  re <- paste0("^", mat_name, "\\[(\\d+),(\\d+)\\]$")
  m <- regexec(re, tab$param)
  g <- regmatches(tab$param, m)
  ok <- lengths(g) == 3
  tab <- tab[ok, , drop = FALSE]
  
  tab$subject <- as.integer(sapply(g[ok], `[[`, 2))
  tab$k       <- as.integer(sapply(g[ok], `[[`, 3))
  tab$parameter <- k_map[tab$k]
  
  tab$model <- model_name
  
  tab %>%
    select(subject, parameter,
           mean, se_mean, sd, `2.50%`, `50%`, `97.50%`, n_eff, Rhat,
           model)
}

# ===============================================================
# RUN ALL: 3 groups × 3 models
# (adjust if you only want best model per group)
# ===============================================================
spec <- list(
  list(exp=11, aw="unaware"),
  list(exp=11, aw="aware"),
  list(exp=12, aw="aware")
)
models <- c("model","learning","boost")

for (g in spec) {
  for (m in models) {
    
    stem <- sprintf("fit_trunc_simplified_%s_%s_exp%d", m, g$aw, g$exp)
    
    # find the .rdata file (recursive)
    file <- list.files(fits_root,
                       pattern = paste0("^", stem, "\\.rdata$"),
                       recursive = TRUE, full.names = TRUE, ignore.case = TRUE)
    if (!length(file)) stop("Missing fit file for: ", stem)
    file <- file[1]
    
    message("\n=== ", stem, " ===\n", file)
    
    fit <- load_cmdstan_from_rdata(file)
    
    # which group params exist depends on model
    group_params <- switch(m,
                           model    = c("mu_alpha","mu_beta","mu_lambda"),
                           learning = c("mu_alpha","mu_beta","mu_lambda","mu_delta"),
                           boost    = c("mu_alpha","mu_beta","mu_lambda","mu_delta","mu_eta")
    )
    
    gdf <- extract_group_csv(fit, group_params, model_name = m) %>%
      mutate(experiment = g$exp, awareness = g$aw)
    
    idf <- extract_individual_csv(fit, model_name = m) %>%
      mutate(experiment = g$exp, awareness = g$aw)
    
    g_out <- file.path(out_dir, paste0("param_group_", stem, ".csv"))
    i_out <- file.path(out_dir, paste0("param_individual_", stem, ".csv"))
    
    write.csv(gdf, g_out, row.names = FALSE)
    write.csv(idf, i_out, row.names = FALSE)
    
    message("Saved:\n  ", g_out, "\n  ", i_out)
  }
}

cat("\n✅ Done. CSVs are in:\n", out_dir, "\n", sep="")