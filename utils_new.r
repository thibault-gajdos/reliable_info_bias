
library(posterior)
library(dplyr)
library(tibble)
library(loo)

##############################################
#  FLUID EXTRACT GROUP PARAMETERS
##############################################
extract_group <- function(fit, parameters) {
  
  if (is.character(fit)) {
    tmp_env <- new.env()
    load(fit, envir = tmp_env)
    fit_obj <- tmp_env[[ls(tmp_env)[1]]] 
  } else {
    fit_obj <- fit
  }
  
  if (inherits(fit_obj, "CmdStanMCMC")) {
    # Directly summarize the requested variables
    results.group <- fit_obj$summary(
      variables = parameters,
      ~quantile(.x, probs = c(0.025, 0.50, 0.975)),
      mean, sd, rhat
    )
    
    # AIC Calculation
    if ("log_lik" %in% fit_obj$metadata()$variables) {
      draws_ll <- fit_obj$draws("log_lik", format = "matrix")
      l_val <- loo::elpd(draws_ll)$estimates[[1]]
      results.group <- results.group %>% 
        mutate(l = l_val, AIC = -2*l_val + 2*length(parameters))
    }
    
  } else {
    results.group <- as.data.frame(summary(fit_obj, pars = parameters)$summary) %>%
      rownames_to_column("variable")
  }
  
  # Ensure we always have a column named 'param' for consistency
  if("variable" %in% names(results.group)) results.group <- rename(results.group, param = variable)
  
  return(results.group)
}

##############################################
#  FLUID EXTRACT INDIVIDUAL PARAMETERS
##############################################
extract_individual <- function(fit, parameters = "ind_params") {
  
  if (is.character(fit)) {
    tmp_env <- new.env()
    load(fit, envir = tmp_env)
    fit_obj <- tmp_env[[ls(tmp_env)[1]]]
  } else {
    fit_obj <- fit
  }
  
  param_names <- c("alpha", "beta", "lambda", "delta")
  
  if (inherits(fit_obj, "CmdStanMCMC")) {
    # Use the $summary method's built-in variable selection
    results.individual <- fit_obj$summary(variables = parameters, mean)
    
    # If the above is empty, it means 'ind_params' needs to be selected by index
    if (nrow(results.individual) == 0) {
      all_vars <- fit_obj$metadata()$variables
      target_vars <- all_vars[grepl(paste0("^", parameters, "\\["), all_vars)]
      results.individual <- fit_obj$summary(variables = target_vars, mean)
    }
  } else {
    results.individual <- as.data.frame(summary(fit_obj, pars = parameters)$summary) %>%
      rownames_to_column("variable")
  }
  
  # Final cleanup and indexing
  results.individual <- results.individual %>%
    rename(param = variable) %>%
    mutate(
      s = as.numeric(gsub(".*\\[(\\d+),.*", "\\1", param)),   
      p_idx = as.numeric(gsub(".*,(\\d+)\\].*", "\\1", param)) 
    ) %>%
    mutate(
      parameter_name = param_names[p_idx]
    )
  
  return(results.individual)
}