######################################
## prepare Exp13 data for fits etc.
## simple version matching Exp11/Exp12
#####################################

rm(list = ls(all = TRUE))

library(tidyverse)

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

matlight_dir <- "/Users/bty615/Library/CloudStorage/OneDrive-QueenMary,UniversityofLondon/Prior Belief/Experiment_PriorBelief/Subject/matlight"

out_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias/data"

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

# -------------------------------------------------------------------------
# Files
# -------------------------------------------------------------------------

files <- list(
  truthful  = file.path(matlight_dir, "data_priorbelief_truthful_exp13.csv"),
  deceptive = file.path(matlight_dir, "data_priorbelief_deceptive_exp13.csv")
)

out_names <- list(
  truthful  = "data_priorbelief_truthful_exp13",
  deceptive = "data_priorbelief_deceptive_exp13"
)

# -------------------------------------------------------------------------
# Convert each file
# -------------------------------------------------------------------------

for (condition in names(files)) {
  
  infile  <- files[[condition]]
  outname <- out_names[[condition]]
  
  data <- read.csv(infile)
  
  data <- data %>%
    separate(
      Sample_Reliability,
      into = paste0("proba_", 1:6),
      sep = "\\s+"
    ) %>%
    mutate(
      across(starts_with("proba_"), as.numeric)
    ) %>%
    mutate(
      color = str_extract_all(Sample_Color, "blue|red")
    ) %>%
    unnest_wider(
      color,
      names_sep = "_"
    )
  
  write.csv(
    data,
    file = file.path(out_dir, paste0(outname, ".csv")),
    row.names = FALSE
  )
  
  save(
    data,
    file = file.path(out_dir, paste0(outname, ".rdata"))
  )
  
  cat("Saved:", outname, "\n")
}