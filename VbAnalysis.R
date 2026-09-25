
############################################################
# Vb TRAJECTORIES — FIVE MODELS, FIVE GROUPS
#
# One graph per model:
#   1. Learning
#   2. Local eta
#   3. Learning + Local eta
#   4. 2 channels
#   5. Local eta + 2 channels
#
#
# Vb is calculated BEFORE each trial's feedback and aligned
# to the true base-rate colour:
#
#   0.5  = neutral
#   >0.5 = belief toward the true base-rate colour
#   <0.5 = belief away from the true base-rate colour
#
# Uses each participant's posterior-mean delta.
# These are trajectories calculated at the mean parameter,
# not posterior mean trajectories or credible intervals.


rm(list = ls(all.names = TRUE))

suppressPackageStartupMessages({
  library(cmdstanr)
  library(tidyverse)
})


# ============================================================
# 1. SETTINGS
# ============================================================

project_dir <- path.expand(
  "~/Documents/GitHub/reliable_info_bias"
)

data_dir <- file.path(
  project_dir,
  "data"
)

fits_dir <- file.path(
  project_dir,
  "stan/results/fits/exp11_unaware"
)

TMAX <- 240L

INITIAL_BLUE_COUNT <- 1
INITIAL_RED_COUNT  <- 1

# Participant order is assumed to match the fitting script:
#
# unique(data$ParticipantPrivateID)
#
# This order is preserved before sorting trials.


# ============================================================
# 2. GROUP INFORMATION
# ============================================================

groups <- tribble(
  ~group, ~suffix, ~data_file, ~prior_col, ~colour,
  
  "Implicit Unaware Base Rate",
  "unaware_exp11",
  "data_priorbelief_unaware_exp11.rdata",
  "Prior_Belief",
  "#F4A3A3",
  
  "Implicit Aware Base Rate",
  "aware_exp11",
  "data_priorbelief_aware_exp11.rdata",
  "Prior_Belief",
  "#8FD694",
  
  "Explicit Undirected Base Rate",
  "aware_exp12",
  "data_priorbelief_aware_exp12.rdata",
  "Prior_Belief",
  "#E69F00",
  
  "Explicit True Base Rate",
  "truthful_exp13",
  "data_priorbelief_truthful_exp13.rdata",
  "TruePrior",
  "#1B7837",
  
  "Explicit Deceptive Base Rate",
  "deceptive_exp13",
  "data_priorbelief_deceptive_exp13.rdata",
  "TruePrior",
  "#8B0000"
)

group_levels <- groups$group

group_cols <- setNames(
  groups$colour,
  groups$group
)


# ============================================================
# 3. MODEL INFORMATION
#
# Learning:
#   ind_params[,4] = delta, range [0,2]
#
# Local eta:
#   no fitted delta; accumulation is equivalent to delta = 1
#
# Learning + Local eta:
#   params[,4] = delta, range [0,2]
#
# 2 channels:
#   params[,6] = delta, range [0,1]
#
# Local eta + 2 channels:
#   params[,7] = delta, range [0,1]
#
# Read transformed parameters directly.
# Do not multiply the extracted delta values again.
# ============================================================

models <- tribble(
  ~model, ~prefix, ~n_params, ~delta_index, ~container, ~delta_upper,
  
  "Learning",
  "learning",
  4L,
  4L,
  "ind_params",
  2,
  
  "Local eta",
  "localeta",
  4L,
  NA_integer_,
  "params",
  1,
  
  "Learning + Local eta",
  "learning_localeta",
  5L,
  4L,
  "params",
  2,
  
  "2 channels",
  "2channels",
  6L,
  6L,
  "params",
  1,
  
  "Local eta + 2 channels",
  "leta_2channels",
  7L,
  7L,
  "params",
  1
)

model_levels <- models$model


# ============================================================
# 4. LOAD AN RDATA OBJECT
# ============================================================

load_object <- function(
    path,
    kind = c("data", "fit")
) {
  
  kind <- match.arg(kind)
  
  if (!file.exists(path)) {
    stop(
      "File not found:\n",
      path
    )
  }
  
  e <- new.env(parent = globalenv())
  
  loaded <- load(
    path,
    envir = e
  )
  
  is_valid <- if (kind == "data") {
    
    function(x) {
      is.data.frame(x)
    }
    
  } else {
    
    function(x) {
      inherits(x, "CmdStanMCMC")
    }
  }
  
  candidates <- loaded[
    vapply(
      loaded,
      function(nm) {
        is_valid(e[[nm]])
      },
      logical(1)
    )
  ]
  
  if (kind %in% candidates) {
    return(e[[kind]])
  }
  
  if (length(candidates) == 1L) {
    return(e[[candidates]])
  }
  
  stop(
    "Cannot identify exactly one ",
    kind,
    " object in:\n",
    path,
    "\nObjects: ",
    paste(loaded, collapse = ", ")
  )
}


# ============================================================
# 5. NORMALISE TRUE-PRIOR CODING
# ============================================================

normalise_prior <- function(x) {
  
  z <- tolower(
    trimws(
      as.character(x)
    )
  )
  
  case_when(
    z %in% c(
      "1", "blue", "b", "65_blue", "blueprior"
    ) ~ 1L,
    
    z %in% c(
      "2", "red", "r", "65_red", "redprior"
    ) ~ 2L,
    
    TRUE ~ NA_integer_
  )
}


# ============================================================
# 6. PREPARE DATA AND CORRECT FEEDBACK
#
# feedback = 1: Blue objectively correct
# feedback = 0: Red objectively correct
#
# CorrectResponse uses response-button coding:
#
# RBO = 1:
#   CorrectResponse 1 = Blue
#   CorrectResponse 0 = Red
#
# RBO = 0:
#   CorrectResponse 0 = Blue
#   CorrectResponse 1 = Red
# ============================================================

prepare_data <- function(
    d,
    prior_col,
    group_name
) {
  
  if (
    !"ResponseButtonOrder" %in% names(d) &&
    "Manipulation_ResponseButtonOrder" %in% names(d)
  ) {
    
    d <- d %>%
      rename(
        ResponseButtonOrder =
          Manipulation_ResponseButtonOrder
      )
  }
  
  required <- c(
    "ParticipantPrivateID",
    "TrialNumber",
    "ResponseButtonOrder",
    "CorrectResponse"
  )
  
  missing <- setdiff(
    required,
    names(d)
  )
  
  if (length(missing)) {
    stop(
      group_name,
      ": missing ",
      paste(missing, collapse = ", ")
    )
  }
  
  if (anyNA(d$ParticipantPrivateID)) {
    stop(
      group_name,
      ": missing participant IDs."
    )
  }
  
  # Preserve participant order BEFORE sorting or filtering.
  
  ids <- unique(
    d$ParticipantPrivateID
  )
  
  d <- d %>%
    mutate(
      participant_index = match(
        ParticipantPrivateID,
        ids
      ),
      
      TrialNumber = suppressWarnings(
        as.numeric(
          as.character(TrialNumber)
        )
      ),
      
      order = suppressWarnings(
        as.integer(
          as.character(ResponseButtonOrder)
        )
      ),
      
      correct = suppressWarnings(
        as.integer(
          as.character(CorrectResponse)
        )
      )
    )
  
  if (
    any(!is.finite(d$TrialNumber)) ||
    any(
      d$TrialNumber < 1 |
      d$TrialNumber != floor(d$TrialNumber)
    )
  ) {
    stop(
      group_name,
      ": TrialNumber must contain positive integer trial indices."
    )
  }
  
  d <- d %>%
    filter(
      TrialNumber <= TMAX
    )
  
  if (
    !setequal(
      unique(d$participant_index),
      seq_along(ids)
    )
  ) {
    stop(
      group_name,
      ": a participant has no trials within TMAX."
    )
  }
  
  if (
    any(!d$order %in% 0:1) ||
    any(!d$correct %in% 0:1)
  ) {
    stop(
      group_name,
      ": invalid button-order or CorrectResponse coding."
    )
  }
  
  d <- d %>%
    mutate(
      feedback = as.integer(
        correct == order
      )
    )
  
  # Use the assigned TRUE prior where available.
  # For Exp13 this is TruePrior, not the instruction.
  #
  # If the column is absent, retain the original script's
  # feedback-majority inference and print a warning.
  
  if (prior_col %in% names(d)) {
    
    d$TrueDirection <- normalise_prior(
      d[[prior_col]]
    )
    
    if (anyNA(d$TrueDirection)) {
      stop(
        group_name,
        ": unrecognised prior labels in ",
        prior_col,
        ". Expected 1/Blue or 2/Red; inspect the actual coding."
      )
    }
    
    d$direction_source <- prior_col
    
  } else {
    
    warning(
      group_name,
      ": ",
      prior_col,
      " is absent. True direction is inferred from feedback, ",
      "as in the old script.",
      call. = FALSE
    )
    
    d <- d %>%
      group_by(
        participant_index
      ) %>%
      mutate(
        TrueDirection = case_when(
          mean(feedback) > 0.5 ~ 1L,
          mean(feedback) < 0.5 ~ 2L,
          TRUE ~ NA_integer_
        ),
        
        direction_source =
          "feedback majority (inferred)"
      ) %>%
      ungroup()
    
    if (anyNA(d$TrueDirection)) {
      stop(
        group_name,
        ": tied feedback; true prior is required."
      )
    }
  }
  
  d <- d %>%
    arrange(
      participant_index,
      TrialNumber
    )
  
  for (n in seq_along(ids)) {
    
    rows <- d[
      d$participant_index == n,
    ]
    
    if (
      n_distinct(rows$TrueDirection) != 1L
    ) {
      stop(
        group_name,
        ": inconsistent true-prior direction for participant ",
        ids[n]
      )
    }
    
    if (
      !identical(
        as.integer(rows$TrialNumber),
        seq_len(nrow(rows))
      )
    ) {
      stop(
        group_name,
        ": repeated, reset, or missing trials for participant ",
        ids[n],
        ". Use the same trial rows/order used for fitting; ",
        "do not renumber silently."
      )
    }
  }
  
  list(
    data = d,
    ids = ids
  )
}


# ============================================================
# 7. EXTRACT PARTICIPANT DELTA
# ============================================================

extract_delta <- function(
    fit,
    spec,
    n_subjects,
    filename
) {
  
  container <- spec$container
  
  s <- tryCatch(
    
    as_tibble(
      fit$summary(
        variables = container
      )
    ),
    
    error = function(e) {
      stop(
        "Cannot read ",
        container,
        " from ",
        filename,
        ": ",
        conditionMessage(e),
        "\nCheck that this fit matches the supplied Stan model ",
        "and its CSV files remain available."
      )
    }
  )
  
  parsed <- str_match(
    s$variable,
    paste0(
      "^",
      container,
      "\\[([0-9]+),([0-9]+)\\]$"
    )
  )
  
  s <- s %>%
    mutate(
      subject_index = as.integer(
        parsed[, 2]
      ),
      
      parameter_index = as.integer(
        parsed[, 3]
      )
    ) %>%
    filter(
      !is.na(subject_index),
      !is.na(parameter_index)
    )
  
  if (
    !setequal(
      unique(s$subject_index),
      seq_len(n_subjects)
    ) ||
    !setequal(
      unique(s$parameter_index),
      seq_len(spec$n_params)
    ) ||
    nrow(s) != n_subjects * spec$n_params
  ) {
    stop(
      "Participant/parameter dimensions differ from the ",
      "model specification in ",
      filename,
      ". Expected ",
      n_subjects,
      " participants and ",
      spec$n_params,
      " parameters. Check the fit and model table."
    )
  }
  
  # Local eta has no fitted delta.
  # Its update is equivalent to delta = 1.
  
  if (is.na(spec$delta_index)) {
    return(
      rep(1, n_subjects)
    )
  }
  
  s <- s %>%
    filter(
      parameter_index == spec$delta_index
    ) %>%
    arrange(
      subject_index
    )
  
  if (
    any(!is.finite(s$mean)) ||
    any(
      s$mean < 0 |
      s$mean > spec$delta_upper
    )
  ) {
    stop(
      "Delta lies outside [0, ",
      spec$delta_upper,
      "] in ",
      filename,
      ". Check the model version and parameter order."
    )
  }
  
  s$mean
}


# ============================================================
# 8. RECONSTRUCT Vb
#
# Vb is recorded BEFORE the current trial's feedback.
#
# Local eta:
#   delta = 1
#   B <- B + feedback
#   R <- R + (1 - feedback)
#
# Other models:
#   use fitted participant delta.
#
# Alpha0/beta0 transform the prior's contribution to the
# decision in the two-channel models. They do not enter
# these feedback-count updates.
# ============================================================

calculate_Vb <- function(
    feedback,
    delta
) {
  
  B <- INITIAL_BLUE_COUNT
  R <- INITIAL_RED_COUNT
  
  out <- numeric(
    length(feedback)
  )
  
  for (t in seq_along(feedback)) {
    
    out[t] <- B / (B + R)
    
    B <- delta * (B - 1) +
      feedback[t] +
      1
    
    R <- delta * (R - 1) +
      (1 - feedback[t]) +
      1
    
    if (!is.finite(B + R)) {
      stop(
        "Belief counts overflowed; ",
        "check delta and update rule."
      )
    }
  }
  
  out
}


# ============================================================
# 9. CHECK ALL FIT PATHS
# ============================================================

manifest <- crossing(
  group = group_levels,
  model = model_levels
) %>%
  left_join(
    groups %>%
      select(
        group,
        suffix
      ),
    by = "group"
  ) %>%
  left_join(
    models %>%
      select(
        model,
        prefix
      ),
    by = "model"
  ) %>%
  mutate(
    path = file.path(
      fits_dir,
      paste0(
        prefix,
        "_",
        suffix,
        ".rdata"
      )
    )
  )

missing_fits <- manifest$path[
  !file.exists(manifest$path)
]

if (length(missing_fits)) {
  stop(
    "Missing fit files:\n",
    paste(
      missing_fits,
      collapse = "\n"
    )
  )
}


# ============================================================
# 10. RUN ALL FIVE GROUPS AND FIVE MODELS
# ============================================================

all_Vb <- list()
delta_tables <- list()

for (g in seq_len(nrow(groups))) {
  
  info <- groups[g, ]
  
  current_data <- load_object(
    file.path(
      data_dir,
      info$data_file
    ),
    kind = "data"
  )
  
  prepared <- prepare_data(
    d = current_data,
    prior_col = info$prior_col,
    group_name = info$group
  )
  
  d <- prepared$data
  ids <- prepared$ids
  
  cat(
    "\n",
    info$group,
    ": ",
    length(ids),
    " participants\n",
    sep = ""
  )
  
  for (m in seq_len(nrow(models))) {
    
    spec <- models[m, ]
    
    filename <- paste0(
      spec$prefix,
      "_",
      info$suffix,
      ".rdata"
    )
    
    fit <- load_object(
      file.path(
        fits_dir,
        filename
      ),
      kind = "fit"
    )
    
    delta <- extract_delta(
      fit = fit,
      spec = spec,
      n_subjects = length(ids),
      filename = filename
    )
    
    delta_tables[[
      length(delta_tables) + 1L
    ]] <- tibble(
      group = info$group,
      model = spec$model,
      participant_index = seq_along(ids),
      ParticipantPrivateID = ids,
      delta = delta
    )
    
    cat(
      "  ",
      spec$model,
      ": mean delta = ",
      round(
        mean(delta),
        3
      ),
      "\n",
      sep = ""
    )
    
    for (n in seq_along(ids)) {
      
      rows <- d %>%
        filter(
          participant_index == n
        )
      
      all_Vb[[
        length(all_Vb) + 1L
      ]] <- rows %>%
        transmute(
          group = info$group,
          model = spec$model,
          
          ParticipantPrivateID,
          participant_index,
          TrialNumber,
          TrueDirection,
          direction_source,
          feedback,
          
          delta_used = delta[n],
          
          Vb_blue = calculate_Vb(
            feedback = feedback,
            delta = delta[n]
          ),
          
          Vb_aligned = if_else(
            TrueDirection == 1L,
            Vb_blue,
            1 - Vb_blue
          )
        )
    }
    
    rm(fit)
  }
}


# ============================================================
# 11. COMBINE RESULTS
# ============================================================

Vb_all <- bind_rows(
  all_Vb
) %>%
  mutate(
    group = factor(
      group,
      levels = group_levels
    ),
    
    model = factor(
      model,
      levels = model_levels
    )
  )

delta_used <- bind_rows(
  delta_tables
)


# ============================================================
# 12. CHECK TRUE-PRIOR COUNTS
# ============================================================

direction_check <- Vb_all %>%
  distinct(
    group,
    ParticipantPrivateID,
    TrueDirection,
    direction_source
  ) %>%
  count(
    group,
    TrueDirection,
    direction_source
  )

cat(
  "\nTRUE-PRIOR DIRECTION CHECK\n",
  "1 = Blue prior; 2 = Red prior\n\n"
)

print(
  direction_check,
  n = Inf
)


# ============================================================
# 13. GROUP-MEAN Vb AT EACH TRIAL
# ============================================================

Vb_summary <- Vb_all %>%
  group_by(
    group,
    model,
    TrialNumber
  ) %>%
  summarise(
    mean_Vb = mean(
      Vb_aligned
    ),
    
    n = n(),
    
    .groups = "drop"
  )


# ============================================================
# 14. SHARED AXIS LIMITS
# ============================================================

y_limits <- range(
  c(
    0.5,
    Vb_summary$mean_Vb
  ),
  finite = TRUE
) + c(
  -0.02,
  0.02
)

y_limits <- pmax(
  0,
  pmin(
    1,
    y_limits
  )
)


# ============================================================
# 15. ONE GRAPH PER MODEL
# ============================================================

plots <- setNames(
  vector(
    "list",
    length(model_levels)
  ),
  model_levels
)

for (model_name in model_levels) {
  
  subtitle_text <- if (
    model_name == "Local eta"
  ) {
    
    paste(
      "Delta fixed at 1;",
      "accumulated belief aligned to the true base-rate colour"
    )
    
  } else {
    
    paste(
      "Participant posterior-mean delta;",
      "accumulated belief aligned to the true base-rate colour"
    )
  }
  
  plots[[model_name]] <- Vb_summary %>%
    filter(
      model == model_name
    ) %>%
    ggplot(
      aes(
        x = TrialNumber,
        y = mean_Vb,
        colour = group
      )
    ) +
    
    geom_hline(
      yintercept = 0.5,
      linetype = "dashed",
      colour = "grey45"
    ) +
    
    geom_line(
      linewidth = 1.15
    ) +
    
    scale_colour_manual(
      values = group_cols,
      drop = FALSE
    ) +
    
    scale_x_continuous(
      breaks = c(
        1,
        50,
        100,
        150,
        200,
        240
      )
    ) +
    
    coord_cartesian(
      xlim = c(
        1,
        TMAX
      ),
      ylim = y_limits
    ) +
    
    labs(
      title = paste(
        model_name,
        "— Vb trajectory"
      ),
      
      subtitle = subtitle_text,
      
      x = "Trial",
      
      y = "Belief in the true base-rate colour",
      
      colour = NULL
    ) +
    
    theme_classic(
      base_size = 13
    ) +
    
    theme(
      plot.title = element_text(
        face = "bold"
      ),
      
      axis.title = element_text(
        face = "bold"
      ),
      
      legend.position = "bottom"
    ) +
    
    guides(
      colour = guide_legend(
        nrow = 3,
        byrow = TRUE
      )
    )
  
  print(
    plots[[model_name]]
  )
}

