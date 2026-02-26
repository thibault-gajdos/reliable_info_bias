rm(list=ls(all=TRUE))

# 1) Always start from repo root
setwd("/Users/bty615/Documents/GitHub/reliable_info_bias")

source("utils_new.r")  # your existing utils (after you changed the thib source)

fits_root <- file.path(getwd(), "results", "fits")
out_dir   <- file.path(getwd(), "results", "summary", "params")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 2) Robust file resolver: finds a file anywhere under results/fits/
find_fit <- function(stem_or_filename) {
  # allow either "fit_trunc_simplified_model_aware_exp11" or "...rdata"
  stem <- sub("\\.(rds|rda|rdata)$", "", stem_or_filename, ignore.case = TRUE)
  
  hits <- list.files(
    fits_root,
    pattern = paste0("^", stem, "\\.(rds|rda|rdata)$"),
    recursive = TRUE,
    full.names = TRUE,
    ignore.case = TRUE
  )
  
  if (!length(hits)) {
    stop(
      "Could not find fit: ", stem, "\nSearched under: ", fits_root,
      "\nTip: run list.files(fits_root, recursive=TRUE) to see what’s there."
    )
  }
  hits[1]
}

# 3) Define what to extract per model
params_group <- list(
  model    = c("mu_alpha","mu_beta","mu_lambda"),
  learning = c("mu_alpha","mu_beta","mu_lambda","mu_delta"),
  boost    = c("mu_alpha","mu_beta","mu_lambda","mu_delta","mu_eta")
)

params_indiv <- list(
  model    = c("alpha","beta","lambda"),
  learning = c("alpha","beta","lambda","delta"),
  boost    = c("alpha","beta","lambda","delta","eta")
)

# 4) Everything you want to process
spec <- list(
  list(exp=11, aw="unaware", model="model"),
  list(exp=11, aw="unaware", model="learning"),
  list(exp=11, aw="unaware", model="boost"),
  list(exp=11, aw="aware",   model="model"),
  list(exp=11, aw="aware",   model="learning"),
  list(exp=11, aw="aware",   model="boost"),
  list(exp=12, aw="aware",   model="model"),
  list(exp=12, aw="aware",   model="learning"),
  list(exp=12, aw="aware",   model="boost")
)

# 5) Loop + write CSVs
for (s in spec) {
  stem <- sprintf("fit_trunc_simplified_%s_%s_exp%d", s$model, s$aw, s$exp)
  fit_path <- find_fit(stem)
  
  message("\n--- ", stem, " ---\n", fit_path)
  
  g <- extract_group(fit_path, parameters = params_group[[s$model]]) |>
    dplyr::mutate(model=s$model, experiment=s$exp, awareness=s$aw)
  
  i <- extract_individual(fit_path, parameters = params_indiv[[s$model]]) |>
    dplyr::mutate(model=s$model, experiment=s$exp, awareness=s$aw)
  
  write.csv(g, file.path(out_dir, paste0("param_group_", stem, ".csv")), row.names=FALSE)
  write.csv(i, file.path(out_dir, paste0("param_individual_", stem, ".csv")), row.names=FALSE)
}

cat("\nSaved CSVs to:\n", out_dir, "\n")