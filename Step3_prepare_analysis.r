rm(list = ls(all = TRUE))

setwd("/Users/bty615/Documents/GitHub/reliable_info_bias")

library(cmdstanr)
library(posterior)
library(tidyverse)

#################################################
##
##      FOLDERS
##
#################################################

fit_folder <- "stan/results/fits/exp11_unaware"

summary_folder <- "stan/results/summary/exp11_unaware"

dir.create(
  summary_folder,
  recursive = TRUE,
  showWarnings = FALSE
)

#################################################
##
##      FIND ALL LETA_2CHANNELS FITS
##
#################################################

fit_files <- list.files(
  path = fit_folder,
  pattern = "\\.rdata$",
  full.names = TRUE,
  ignore.case = TRUE
)

fit_files <- sort(
  fit_files[
    grepl(
      "leta_2channels",
      basename(fit_files),
      ignore.case = TRUE
    )
  ]
)

if (length(fit_files) == 0L) {
  stop("No leta_2channels fits found in: ", fit_folder)
}

cat("\nFits to export:\n")
print(basename(fit_files))

#################################################
##
##      PARAMETER ORDER IN YOUR STAN MODEL
##
#################################################

parameter_names <- c(
  "alpha",
  "beta",
  "lambda",
  "eta",
  "alpha0",
  "beta0",
  "delta"
)

group_variables <- paste0("mu_", parameter_names)

#################################################
##
##      SUMMARY FUNCTION
##
#################################################

summarise_parameters <- function(fit, variables) {
  fit$summary(
    variables = variables,
    "mean",
    "median",
    "sd",
    q2.5 = ~unname(quantile(.x, 0.025)),
    q97.5 = ~unname(quantile(.x, 0.975)),
    "rhat",
    "ess_bulk",
    "ess_tail"
  )
}

#################################################
##
##      EXPORT EACH FIT
##
#################################################

export_fit <- function(fit_path) {
  
  cat("\nLoading:", basename(fit_path), "\n")
  
  fit_environment <- new.env()
  
  loaded_objects <- load(
    fit_path,
    envir = fit_environment
  )
  
  fit_names <- loaded_objects[
    vapply(
      loaded_objects,
      function(object_name) {
        inherits(
          get(object_name, envir = fit_environment),
          "CmdStanMCMC"
        )
      },
      logical(1)
    )
  ]
  
  if (length(fit_names) != 1L) {
    stop("Expected exactly one CmdStanMCMC fit.")
  }
  
  fit <- get(
    fit_names[[1]],
    envir = fit_environment
  )
  
  output_name <- tools::file_path_sans_ext(
    basename(fit_path)
  )
  
  output_name <- sub(
    "^fit_",
    "",
    output_name,
    ignore.case = TRUE
  )
  
  #################################################
  ## GROUP PARAMETERS
  #################################################
  
  leta_2channels.group <- summarise_parameters(
    fit,
    group_variables
  ) %>%
    arrange(match(variable, group_variables))
  
  if (
    nrow(leta_2channels.group) != 7L ||
    !setequal(
      leta_2channels.group$variable,
      group_variables
    )
  ) {
    stop("Fit does not contain the seven expected group parameters.")
  }
  
  #################################################
  ## INDIVIDUAL PARAMETERS
  #################################################
  
  leta_2channels.individual <- summarise_parameters(
    fit,
    "params"
  ) %>%
    tidyr::extract(
      variable,
      into = c("participant", "parameter_number"),
      regex = "^params\\[([0-9]+),([0-9]+)\\]$",
      convert = TRUE
    )
  
  if (
    anyNA(leta_2channels.individual$participant) ||
    anyNA(leta_2channels.individual$parameter_number) ||
    !setequal(
      leta_2channels.individual$parameter_number,
      seq_along(parameter_names)
    )
  ) {
    stop("Fit does not match the seven-parameter model.")
  }
  
  participant_check <- leta_2channels.individual %>%
    group_by(participant) %>%
    summarise(
      rows = n(),
      parameters = n_distinct(parameter_number),
      .groups = "drop"
    )
  
  if (
    any(participant_check$rows != 7L) ||
    any(participant_check$parameters != 7L)
  ) {
    stop("Each participant must have seven parameter summaries.")
  }
  
  leta_2channels.individual <- leta_2channels.individual %>%
    mutate(
      parameter = parameter_names[parameter_number]
    ) %>%
    arrange(participant, parameter_number) %>%
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
  
  #################################################
  ## SAVE GROUP AND INDIVIDUAL CSVs
  #################################################
  
  group_csv <- file.path(
    summary_folder,
    paste0("param_group_", output_name, ".csv")
  )
  
  individual_csv <- file.path(
    summary_folder,
    paste0("param_individual_", output_name, ".csv")
  )
  
  write_csv(
    leta_2channels.group,
    group_csv
  )
  
  write_csv(
    leta_2channels.individual,
    individual_csv
  )
  
  #################################################
  ## ALSO SAVE RDATA SUMMARIES
  #################################################
  
  save(
    leta_2channels.group,
    file = file.path(
      summary_folder,
      paste0("summary_group_", output_name, ".rdata")
    )
  )
  
  save(
    leta_2channels.individual,
    file = file.path(
      summary_folder,
      paste0("summary_individual_", output_name, ".rdata")
    )
  )
  
  cat("Group CSV:", group_csv, "\n")
  cat("Individual CSV:", individual_csv, "\n")
  
  tibble(
    fit_file = basename(fit_path),
    participants = n_distinct(
      leta_2channels.individual$participant
    ),
    group_rows = nrow(leta_2channels.group),
    individual_rows = nrow(leta_2channels.individual),
    status = "Saved",
    detail = NA_character_
  )
}

#################################################
##
##      RUN ALL FITS
##
##      If one fails, continue with the others.
##
#################################################

export_results <- lapply(
  fit_files,
  function(fit_path) {
    tryCatch(
      export_fit(fit_path),
      error = function(e) {
        
        message(
          "FAILED: ",
          basename(fit_path),
          " — ",
          conditionMessage(e)
        )
        
        tibble(
          fit_file = basename(fit_path),
          participants = NA_integer_,
          group_rows = NA_integer_,
          individual_rows = NA_integer_,
          status = "Failed",
          detail = conditionMessage(e)
        )
      }
    )
  }
) %>%
  bind_rows()

#################################################
##
##      DISPLAY AND SAVE EXPORT CHECK
##
#################################################

print(export_results, n = Inf, width = Inf)

write_csv(
  export_results,
  file.path(
    summary_folder,
    "leta_2channels_export_check.csv"
  )
)

cat(
  "\nSuccessfully exported:",
  sum(export_results$status == "Saved"),
  "of",
  nrow(export_results),
  "fits.\n"
)

cat("\nOutput folder:\n", summary_folder, "\n")



























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
  "fit_trunc_boost_aware_exp12.rdata"
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
      "summary_group_log_boost_aware_exp12.rdata"
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
      "summary_individual_log_trunc_boost_aware_exp12.rdata"
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
    "summary_group_log_trunc_boost_aware_exp12.rdata"
  ),
  "\n"
)

cat("\nIndividual results saved to:\n")
cat(
  file.path(
    summary_folder,
    "summary_individual_log_trunc_boost_aware_exp12.rdata"
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
  "param_individual_boost_aware_exp12.csv"
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
  
  
  
  
  