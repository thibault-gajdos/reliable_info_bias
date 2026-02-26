# utils_clean.r  (NO external source(), NO setwd())

library(loo)
library(rstan)
library(dplyr)
library(tibble)
library(stringr)

# ----------------------------
# Load a stanfit from either .rds or .rdata
# ----------------------------
load_stanfit <- function(path) {
  if (!file.exists(path)) stop("Missing fit file: ", path)
  
  if (grepl("\\.rds$", path, ignore.case = TRUE)) {
    fit <- readRDS(path)
    if (!inherits(fit, "stanfit")) stop("RDS is not a stanfit: ", path)
    return(fit)
  }
  
  if (grepl("\\.rdata$|\\.rda$", path, ignore.case = TRUE)) {
    e <- new.env()
    load(path, envir = e)
    objs <- ls(e)
    for (nm in objs) {
      obj <- e[[nm]]
      if (inherits(obj, "stanfit")) return(obj)
    }
    stop("No stanfit object found inside: ", path,
         "\nObjects inside were: ", paste(objs, collapse=", "))
  }
  
  stop("Unsupported file type (expected .rds or .rdata/.rda): ", path)
}

# ----------------------------
# GROUP parameter extraction
# returns a data.frame with mean/quantiles + ELPD_loo + n_divergent
# ----------------------------
extract_group <- function(fit_path, parameters) {
  fit <- load_stanfit(fit_path)
  
  # pointwise log_lik -> elpd_loo estimate
  log_lik <- loo::extract_log_lik(fit, parameter_name = "log_lik", merge_chains = TRUE)
  elpd_est <- loo::elpd(log_lik)$estimates
  elpd_loo <- as.numeric(elpd_est["elpd_loo", "Estimate"])
  
  # divergent transitions
  sampler_params <- rstan::get_sampler_params(fit, inc_warmup = FALSE)
  n_div <- sum(sapply(sampler_params, function(x) sum(x[, "divergent__"])))
  
  # posterior summaries for requested params
  summ <- as.data.frame(
    rstan::summary(fit, pars = parameters, probs = c(0.025, 0.5, 0.975))$summary
  ) |>
    rownames_to_column("param")
  
  summ$elpd_loo <- elpd_loo
  summ$n_divergent <- n_div
  summ
}

# ----------------------------
# INDIVIDUAL extraction from generated quantities matrices:
# - tries params[N,K] first, then ind_params[N,K]
# - returns long-format: subject, parameter, mean, 2.5%, 50%, 97.5%
# ----------------------------
extract_individual <- function(fit_path, param_names = c("alpha","beta","lambda","delta","eta")) {
  fit <- load_stanfit(fit_path)
  
  # detect matrix name in fit
  all_pars <- fit@sim$fnames_oi
  has_params    <- any(grepl("^params\\[", all_pars))
  has_indparams <- any(grepl("^ind_params\\[", all_pars))
  
  mat_name <- if (has_params) "params" else if (has_indparams) "ind_params" else NULL
  if (is.null(mat_name)) {
    stop("Couldn't find params[...] or ind_params[...] in fit: ", fit_path)
  }
  
  arr <- rstan::extract(fit, pars = mat_name)[[1]]  # iterations x N x K
  if (length(dim(arr)) != 3) stop("Unexpected shape for ", mat_name, " in: ", fit_path)
  
  n_subj <- dim(arr)[2]
  k_dim  <- dim(arr)[3]
  
  # map columns -> parameter names
  # K=3: alpha,beta,lambda
  # K=4: alpha,beta,lambda,delta
  # K=5: alpha,beta,lambda,delta,eta
  default_map <- list(
    `3` = c("alpha","beta","lambda"),
    `4` = c("alpha","beta","lambda","delta"),
    `5` = c("alpha","beta","lambda","delta","eta")
  )
  pmap <- default_map[[as.character(k_dim)]]
  if (is.null(pmap)) stop("Unexpected K dimension (", k_dim, ") in ", mat_name, " for: ", fit_path)
  
  # summarise per subject x param
  out <- list()
  idx <- 1
  for (k in seq_len(k_dim)) {
    draws_k <- arr[, , k, drop=FALSE]  # iters x N x 1
    draws_k <- draws_k[, , 1]          # iters x N
    
    m  <- apply(draws_k, 2, mean)
    q2 <- apply(draws_k, 2, quantile, probs=0.025)
    q5 <- apply(draws_k, 2, quantile, probs=0.5)
    q9 <- apply(draws_k, 2, quantile, probs=0.975)
    
    out[[idx]] <- data.frame(
      subject = 1:n_subj,
      parameter = pmap[k],
      mean = m,
      q2.5 = q2,
      q50  = q5,
      q97.5 = q9
    )
    idx <- idx + 1
  }
  
  dplyr::bind_rows(out)
}