############################################################
# LOCAL ETA + TWO CHANNELS
#
# FIGURE 1:
# Seven-parameter pairs plot.
#
# FIGURE 2:
# Parameters versus accuracy and optimal choices.
#
# One dot = one participant's posterior mean.
# Regression lines appear only when p < .05.
# Displays figures only; no files are saved.
############################################################

suppressPackageStartupMessages({
  library(cmdstanr)
  library(tidyverse)
})


############################################################
# 1. SETTINGS
############################################################

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

P_CUTOFF <- 0.05

# "none": use uncorrected p-values for regression lines.
# "BH": use Benjamini-Hochberg corrected p-values.
P_ADJUST <- "none"

# FALSE: optimal choice uses sample evidence + 50/50 prior.
# TRUE: optimal choice uses sample evidence + true 65/35 prior.
OPTIMAL_USE_TRUE_PRIOR <- FALSE

# FALSE: use the current graphics device, usually RStudio Plots.
# TRUE: open separate graphics windows for the two figures.
OPEN_NEW_WINDOWS <- FALSE

stopifnot(
  P_ADJUST %in% c("none", "BH")
)


############################################################
# 2. GROUPS
############################################################

groups <- tribble(
  ~group, ~suffix, ~prior_col, ~colour,
  
  "Implicit Unaware",
  "unaware_exp11",
  "Prior_Belief",
  "#F4A3A3",
  
  "Implicit Aware",
  "aware_exp11",
  "Prior_Belief",
  "#8FD694",
  
  "Explicit Undirected",
  "aware_exp12",
  "Prior_Belief",
  "#E69F00",
  
  "Explicit True",
  "truthful_exp13",
  "TruePrior",
  "#1B7837",
  
  "Explicit Deceptive",
  "deceptive_exp13",
  "TruePrior",
  "#8B0000"
)

group_cols <- setNames(
  groups$colour,
  groups$group
)

group_short <- setNames(
  c("IU", "IA", "EU", "ET", "ED"),
  groups$group
)


############################################################
# 3. PARAMETER ORDER AND LABELS
############################################################

# Order in Stan params[participant, parameter].
# Must match your leta_2channels model.
param_names <- c(
  "alpha",
  "beta",
  "lambda",
  "eta",
  "alpha0",
  "beta0",
  "delta"
)

# Order in the figures.
plot_order <- c(
  "alpha",
  "beta",
  "delta",
  "lambda",
  "eta",
  "alpha0",
  "beta0"
)

plot_labels <- list(
  alpha = expression(alpha),
  beta = expression(beta),
  delta = expression(delta),
  lambda = expression(lambda),
  eta = expression(eta),
  alpha0 = expression(alpha[0]),
  beta0 = expression(beta[0]),
  accuracy = "Accuracy (%)",
  optimal = "Optimal choices (%)"
)


############################################################
# 4. PARTICIPANT ORDER
############################################################

# Default: first-appearance order of participant IDs in each
# fitting RData file.
#
# This must match the order used when creating the Stan data.
# Matching participant counts alone cannot establish this.
#
# If the fitting script used another order, enter the exact
# ID vector for each group here.

subject_ids_by_group <- list()

# Example:
# subject_ids_by_group[["Implicit Unaware"]] <- c(
#   "participant_ID_1",
#   "participant_ID_2"
# )


############################################################
# 5. HELPERS
############################################################

num <- function(x) {
  suppressWarnings(
    as.numeric(as.character(x))
  )
}


load_object <- function(path, kind) {
  
  if (!file.exists(path)) {
    stop("Missing file:\n", path)
  }
  
  e <- new.env(parent = globalenv())
  
  loaded_names <- load(
    path,
    envir = e
  )
  
  valid_names <- loaded_names[
    vapply(
      loaded_names,
      function(name) {
        if (kind == "data") {
          is.data.frame(e[[name]])
        } else {
          inherits(e[[name]], "CmdStanMCMC")
        }
      },
      logical(1)
    )
  ]
  
  if (kind %in% valid_names) {
    return(e[[kind]])
  }
  
  if (length(valid_names) != 1L) {
    stop(
      "Cannot identify exactly one ",
      kind,
      " object in:\n",
      path
    )
  }
  
  e[[valid_names]]
}


finite_range <- function(x, padding = 0.04) {
  
  x <- x[is.finite(x)]
  
  if (!length(x)) {
    return(c(0, 1))
  }
  
  limits <- range(x)
  
  if (diff(limits) == 0) {
    limits <- limits + c(-0.5, 0.5)
  }
  
  limits + c(-1, 1) * diff(limits) * padding
}


significance_stars <- function(p) {
  
  if (length(p) != 1L || is.na(p)) {
    return("")
  }
  
  if (p < 0.001) {
    "***"
  } else if (p < 0.01) {
    "**"
  } else if (p < 0.05) {
    "*"
  } else {
    ""
  }
}


############################################################
# 6. SAMPLE EVIDENCE
############################################################

evidence_log_odds <- function(d) {
  
  n <- nrow(d)
  
  colour_columns <- paste0("color_", 1:6)
  reliability_columns <- paste0("proba_", 1:6)
  
  if (
    all(
      c(
        colour_columns,
        reliability_columns
      ) %in% names(d)
    )
  ) {
    
    sample_colours <- as.matrix(
      d[colour_columns]
    )
    
    sample_reliabilities <- matrix(
      unlist(
        lapply(
          d[reliability_columns],
          num
        ),
        use.names = FALSE
      ),
      nrow = n,
      ncol = 6
    )
    
  } else {
    
    required <- c(
      "Sample_Color",
      "Sample_Reliability"
    )
    
    if (!all(required %in% names(d))) {
      stop(
        "Sample colour/reliability columns are missing."
      )
    }
    
    sample_colours <- matrix(
      NA_character_,
      nrow = n,
      ncol = 6
    )
    
    sample_reliabilities <- matrix(
      NA_real_,
      nrow = n,
      ncol = 6
    )
    
    for (i in seq_len(n)) {
      
      colours_i <- str_extract_all(
        tolower(
          as.character(d$Sample_Color[i])
        ),
        "blue|red"
      )[[1]]
      
      reliabilities_i <- num(
        str_extract_all(
          as.character(d$Sample_Reliability[i]),
          "[0-9]+(?:\\.[0-9]+)?"
        )[[1]]
      )
      
      if (
        length(colours_i) != 6L ||
        length(reliabilities_i) != 6L
      ) {
        stop(
          "Expected exactly six samples in row ",
          i
        )
      }
      
      sample_colours[i, ] <- colours_i
      sample_reliabilities[i, ] <- reliabilities_i
    }
  }
  
  sample_colours[] <- tolower(
    trimws(
      as.character(sample_colours)
    )
  )
  
  if (
    anyNA(sample_colours) ||
    any(!sample_colours %in% c("blue", "red"))
  ) {
    stop(
      "Sample colours must be blue/red. ",
      "Check numeric colour codes before converting them."
    )
  }
  
  sample_reliabilities[
    sample_reliabilities > 1
  ] <- sample_reliabilities[
    sample_reliabilities > 1
  ] / 100
  
  distance_from_valid_reliability <- pmin(
    abs(sample_reliabilities - 0.50),
    abs(sample_reliabilities - 0.55),
    abs(sample_reliabilities - 0.65)
  )
  
  if (
    anyNA(sample_reliabilities) ||
    any(!is.finite(sample_reliabilities)) ||
    any(distance_from_valid_reliability > 1e-6)
  ) {
    stop(
      "Reliabilities must be 50/55/65 ",
      "or .50/.55/.65."
    )
  }
  
  signed_evidence <- ifelse(
    sample_colours == "blue",
    1,
    -1
  )
  
  rowSums(
    signed_evidence *
      qlogis(sample_reliabilities)
  )
}


############################################################
# 7. PREPARE ONE GROUP
############################################################

prepare_group <- function(info) {
  
  cat(
    "\nLoading ",
    info$group,
    "...\n",
    sep = ""
  )
  
  data_file <- file.path(
    data_dir,
    paste0(
      "data_priorbelief_",
      info$suffix,
      ".rdata"
    )
  )
  
  d <- load_object(
    data_file,
    kind = "data"
  )
  
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
  
  required_columns <- c(
    "ParticipantPrivateID",
    "TrialNumber",
    "Response",
    "CorrectResponse",
    "ResponseButtonOrder"
  )
  
  missing_columns <- setdiff(
    required_columns,
    names(d)
  )
  
  if (length(missing_columns)) {
    stop(
      "Missing columns in ",
      info$group,
      ": ",
      paste(missing_columns, collapse = ", ")
    )
  }
  
  if (anyNA(d$ParticipantPrivateID)) {
    stop("Missing participant IDs.")
  }
  
  # Establish participant order before sorting/filtering.
  ids <- subject_ids_by_group[[info$group]]
  
  if (is.null(ids)) {
    ids <- unique(
      as.character(d$ParticipantPrivateID)
    )
  }
  
  ids <- as.character(ids)
  
  if (
    anyDuplicated(ids) ||
    !setequal(
      ids,
      as.character(d$ParticipantPrivateID)
    )
  ) {
    stop(
      "Participant-ID mapping does not match ",
      info$group
    )
  }
  
  d <- d %>%
    mutate(
      ParticipantPrivateID =
        as.character(ParticipantPrivateID),
      
      subject_index = match(
        ParticipantPrivateID,
        ids
      ),
      
      trial = num(TrialNumber)
    )
  
  if (
    anyNA(d$trial) ||
    any(!is.finite(d$trial)) ||
    any(
      d$trial < 1 |
      d$trial != floor(d$trial)
    )
  ) {
    stop("Invalid trial numbers.")
  }
  
  d <- d %>%
    filter(trial <= TMAX) %>%
    arrange(subject_index, trial)
  
  if (
    n_distinct(d$subject_index) != length(ids)
  ) {
    stop(
      "At least one participant has no retained trials."
    )
  }
  
  for (subject in seq_along(ids)) {
    
    trial_numbers <- d$trial[
      d$subject_index == subject
    ]
    
    if (
      !identical(
        as.integer(trial_numbers),
        seq_along(trial_numbers)
      )
    ) {
      stop(
        "Repeated, reset or missing trial numbers for ",
        ids[subject],
        ". Use the same trial rows used for fitting."
      )
    }
  }
  
  response <- num(d$Response)
  correct <- num(d$CorrectResponse)
  button_order <- num(d$ResponseButtonOrder)
  
  if (
    any(!correct %in% 0:1) ||
    any(!button_order %in% 0:1) ||
    any(
      !is.na(response) &
      !response %in% 0:1
    )
  ) {
    stop(
      "Expected 0/1 response and button-order codes."
    )
  }
  
  # Correct colour mapping:
  # RBO = 1: response 1 = Blue.
  # RBO = 0: response 0 = Blue.
  blue_choice <- response == button_order
  
  log_odds <- evidence_log_odds(d)
  
  if (OPTIMAL_USE_TRUE_PRIOR) {
    
    if (!info$prior_col %in% names(d)) {
      stop(
        "True-prior column missing: ",
        info$prior_col
      )
    }
    
    prior_code <- tolower(
      trimws(
        as.character(d[[info$prior_col]])
      )
    )
    
    prior_blue <- case_when(
      prior_code %in% c(
        "1", "blue", "b", "65_blue", "blueprior"
      ) ~ 0.65,
      
      prior_code %in% c(
        "2", "red", "r", "65_red", "redprior"
      ) ~ 0.35,
      
      TRUE ~ NA_real_
    )
    
    if (anyNA(prior_blue)) {
      stop("Unrecognised true-prior coding.")
    }
    
    log_odds <- log_odds + qlogis(prior_blue)
  }
  
  # Ties are excluded from optimal-choice accuracy.
  # They remain included in objective task accuracy.
  behaviour <- d %>%
    mutate(
      accurate = response == correct,
      
      optimal_choice = ifelse(
        abs(log_odds) < 1e-10,
        NA,
        blue_choice == (log_odds > 0)
      )
    ) %>%
    group_by(
      subject_index,
      ParticipantPrivateID
    ) %>%
    summarise(
      accuracy = 100 * mean(
        accurate,
        na.rm = TRUE
      ),
      
      optimal = 100 * mean(
        optimal_choice,
        na.rm = TRUE
      ),
      
      n_accuracy = sum(
        !is.na(accurate)
      ),
      
      n_optimal = sum(
        !is.na(optimal_choice)
      ),
      
      .groups = "drop"
    )
  
  fit_file <- file.path(
    fits_dir,
    paste0(
      "leta_2channels_",
      info$suffix,
      ".rdata"
    )
  )
  
  fit <- load_object(
    fit_file,
    kind = "fit"
  )
  
  # Already-transformed participant parameters.
  parameter_summary <- as_tibble(
    fit$summary(
      variables = "params"
    )
  )
  
  indices <- str_match(
    parameter_summary$variable,
    "^params\\[([0-9]+),([0-9]+)\\]$"
  )
  
  parameter_summary <- parameter_summary %>%
    mutate(
      subject_index = as.integer(
        indices[, 2]
      ),
      
      parameter_index = as.integer(
        indices[, 3]
      )
    ) %>%
    filter(
      !is.na(subject_index),
      !is.na(parameter_index)
    )
  
  if (
    nrow(parameter_summary) != length(ids) * 7L ||
    anyDuplicated(parameter_summary$variable) ||
    !setequal(
      parameter_summary$subject_index,
      seq_along(ids)
    ) ||
    !setequal(
      parameter_summary$parameter_index,
      1:7
    )
  ) {
    stop(
      "Fit dimensions do not match the participant count ",
      "and seven parameters for ",
      info$group
    )
  }
  
  if (
    any(!is.finite(parameter_summary$mean))
  ) {
    stop("Non-finite posterior means.")
  }
  
  if (
    "rhat" %in% names(parameter_summary) &&
    any(
      parameter_summary$rhat > 1.01,
      na.rm = TRUE
    )
  ) {
    warning(
      info$group,
      ": some R-hat values exceed 1.01. ",
      "Inspect fit diagnostics."
    )
  }
  
  parameters <- parameter_summary %>%
    transmute(
      subject_index,
      
      parameter = param_names[
        parameter_index
      ],
      
      value = mean
    ) %>%
    pivot_wider(
      names_from = parameter,
      values_from = value
    )
  
  left_join(
    behaviour,
    parameters,
    by = "subject_index"
  ) %>%
    mutate(
      group = info$group
    )
}


############################################################
# 8. LOAD ALL FIVE GROUPS
############################################################

participant_data <- map_dfr(
  seq_len(nrow(groups)),
  function(i) {
    prepare_group(groups[i, ])
  }
) %>%
  mutate(
    group = factor(
      group,
      levels = groups$group
    )
  )

cat("\nParticipant counts and mean performance:\n")

print(
  participant_data %>%
    group_by(group) %>%
    summarise(
      n = n(),
      
      mean_accuracy = mean(
        accuracy,
        na.rm = TRUE
      ),
      
      mean_optimal = mean(
        optimal,
        na.rm = TRUE
      ),
      
      .groups = "drop"
    )
)


############################################################
# 9. DEFINE CORRELATIONS
############################################################

parameter_pairs <- as_tibble(
  t(
    combn(param_names, 2)
  ),
  .name_repair = "minimal"
)

names(parameter_pairs) <- c(
  "xvar",
  "yvar"
)

pairs_to_test <- bind_rows(
  parameter_pairs %>%
    mutate(
      family = "Between parameters"
    ),
  
  crossing(
    xvar = param_names,
    yvar = c("accuracy", "optimal")
  ) %>%
    mutate(
      family = "Performance"
    )
) %>%
  mutate(
    pair_id = row_number()
  )

scatter_data <- map_dfr(
  seq_len(nrow(pairs_to_test)),
  function(i) {
    
    participant_data %>%
      transmute(
        group,
        ParticipantPrivateID,
        
        pair_id = pairs_to_test$pair_id[i],
        family = pairs_to_test$family[i],
        xvar = pairs_to_test$xvar[i],
        yvar = pairs_to_test$yvar[i],
        
        x = .data[[
          pairs_to_test$xvar[i]
        ]],
        
        y = .data[[
          pairs_to_test$yvar[i]
        ]]
      )
  }
) %>%
  filter(
    is.finite(x),
    is.finite(y)
  )


############################################################
# 10. CORRELATION TESTS
############################################################

test_pair <- function(d) {
  
  if (
    nrow(d) < 3L ||
    sd(d$x) == 0 ||
    sd(d$y) == 0
  ) {
    return(
      tibble(
        n = nrow(d),
        r = NA_real_,
        p = NA_real_
      )
    )
  }
  
  result <- cor.test(
    d$x,
    d$y,
    method = "pearson",
    alternative = "two.sided"
  )
  
  tibble(
    n = nrow(d),
    r = unname(result$estimate),
    p = result$p.value
  )
}

correlation_results <- scatter_data %>%
  group_by(
    group,
    family,
    pair_id,
    xvar,
    yvar
  ) %>%
  group_modify(
    ~ test_pair(.x)
  ) %>%
  ungroup() %>%
  group_by(family) %>%
  mutate(
    # Correction across all five groups within each family.
    p_BH = p.adjust(
      p,
      method = "BH"
    ),
    
    p_used = if (P_ADJUST == "BH") {
      p_BH
    } else {
      p
    }
  ) %>%
  ungroup() %>%
  mutate(
    significant =
      !is.na(p_used) &
      p_used < P_CUTOFF
  )

cat("\nAll correlation results:\n")

print(
  correlation_results,
  n = Inf,
  width = Inf
)

cat("\nSignificant correlations:\n")

print(
  correlation_results %>%
    filter(significant),
  n = Inf,
  width = Inf
)


############################################################
# 11. PLOTTING HELPERS
############################################################

lookup_test <- function(
    group_name,
    x_name,
    y_name
) {
  
  correlation_results %>%
    filter(
      as.character(group) == group_name,
      
      (
        xvar == x_name &
          yvar == y_name
      ) |
        (
          xvar == y_name &
            yvar == x_name
        )
    )
}


get_group_xy <- function(
    group_name,
    x_name,
    y_name
) {
  
  participant_data %>%
    filter(
      as.character(group) == group_name
    ) %>%
    transmute(
      x = .data[[x_name]],
      y = .data[[y_name]]
    ) %>%
    filter(
      is.finite(x),
      is.finite(y)
    )
}


correlation_label <- function(
    group_name,
    x_name,
    y_name
) {
  
  test <- lookup_test(
    group_name,
    x_name,
    y_name
  )
  
  if (
    nrow(test) != 1L ||
    !is.finite(test$r)
  ) {
    return(
      paste0(
        group_short[group_name],
        "  r = NA"
      )
    )
  }
  
  paste0(
    group_short[group_name],
    "  r = ",
    sprintf("%.2f", test$r),
    significance_stars(test$p_used)
  )
}


draw_group_scatter <- function(
    x_name,
    y_name,
    point_size = 0.7
) {
  
  for (group_name in groups$group) {
    
    d <- get_group_xy(
      group_name,
      x_name,
      y_name
    )
    
    points(
      d$x,
      d$y,
      pch = 16,
      cex = point_size,
      
      col = adjustcolor(
        group_cols[group_name],
        alpha.f = 0.65
      )
    )
    
    test <- lookup_test(
      group_name,
      x_name,
      y_name
    )
    
    if (
      nrow(test) == 1L &&
      isTRUE(test$significant)
    ) {
      
      regression <- lm(
        y ~ x,
        data = d
      )
      
      line_x <- range(d$x)
      
      lines(
        line_x,
        
        predict(
          regression,
          newdata = data.frame(
            x = line_x
          )
        ),
        
        col = group_cols[group_name],
        lwd = 2
      )
    }
  }
}


draw_legend <- function() {
  
  par(
    mar = c(0, 0, 0, 0)
  )
  
  plot.new()
  
  legend(
    "center",
    legend = groups$group,
    col = group_cols,
    pch = 16,
    lwd = 2,
    ncol = 5,
    bty = "n",
    cex = 0.9
  )
}


significance_caption <- paste0(
  "One dot per participant | Lines: ",
  if (P_ADJUST == "BH") {
    "BH-adjusted"
  } else {
    "uncorrected"
  },
  " p < ",
  P_CUTOFF,
  " | * p < .05, ** p < .01, *** p < .001"
)


############################################################
# 12. FIGURE 1 FUNCTION: PARAMETER PAIRS
############################################################

plot_parameter_pairs <- function() {
  
  if (
    OPEN_NEW_WINDOWS &&
    interactive()
  ) {
    dev.new(
      width = 15,
      height = 14
    )
  }
  
  old_par <- par(no.readonly = TRUE)
  
  on.exit(
    {
      layout(1)
      par(old_par)
    },
    add = TRUE
  )
  
  variables <- plot_order
  k <- length(variables)
  
  layout(
    rbind(
      matrix(
        seq_len(k * k),
        nrow = k,
        ncol = k,
        byrow = TRUE
      ),
      
      rep(
        k * k + 1L,
        k
      )
    ),
    
    heights = c(
      rep(1, k),
      0.5
    )
  )
  
  par(
    oma = c(0, 1, 3, 0)
  )
  
  variable_ranges <- setNames(
    lapply(
      variables,
      function(variable) {
        finite_range(
          participant_data[[variable]]
        )
      }
    ),
    variables
  )
  
  for (row in seq_len(k)) {
    
    for (column in seq_len(k)) {
      
      x_name <- variables[column]
      y_name <- variables[row]
      
      par(
        mar = c(1.5, 1.6, 1.1, 0.3),
        mgp = c(1, 0.25, 0),
        tcl = -0.15
      )
      
      ######################################################
      # DIAGONAL: DENSITIES
      ######################################################
      
      if (row == column) {
        
        densities <- setNames(
          lapply(
            groups$group,
            function(group_name) {
              
              values <- participant_data[[x_name]][
                participant_data$group == group_name
              ]
              
              values <- values[
                is.finite(values)
              ]
              
              if (
                length(values) < 2L ||
                sd(values) == 0
              ) {
                return(NULL)
              }
              
              density(
                values,
                n = 256
              )
            }
          ),
          groups$group
        )
        
        density_peaks <- vapply(
          densities,
          function(d) {
            if (is.null(d)) {
              0
            } else {
              max(d$y)
            }
          },
          numeric(1)
        )
        
        maximum_density <- max(density_peaks)
        
        if (maximum_density <= 0) {
          maximum_density <- 1
        }
        
        plot(
          NA,
          xlim = variable_ranges[[x_name]],
          
          ylim = c(
            0,
            maximum_density * 1.05
          ),
          
          axes = FALSE,
          xlab = "",
          ylab = ""
        )
        
        for (group_name in groups$group) {
          
          d <- densities[[group_name]]
          
          if (!is.null(d)) {
            
            polygon(
              c(d$x, rev(d$x)),
              
              c(
                d$y,
                rep(0, length(d$y))
              ),
              
              col = adjustcolor(
                group_cols[group_name],
                alpha.f = 0.08
              ),
              
              border = NA
            )
            
            lines(
              d$x,
              d$y,
              col = group_cols[group_name],
              lwd = 1.8
            )
            
          } else {
            
            values <- participant_data[[x_name]][
              participant_data$group == group_name
            ]
            
            values <- values[
              is.finite(values)
            ]
            
            if (length(values)) {
              abline(
                v = values[1],
                col = group_cols[group_name],
                lwd = 1.5
              )
            }
          }
        }
        
        ######################################################
        # LOWER TRIANGLE: SCATTERPLOTS
        ######################################################
        
      } else if (row > column) {
        
        plot(
          NA,
          xlim = variable_ranges[[x_name]],
          ylim = variable_ranges[[y_name]],
          axes = FALSE,
          xlab = "",
          ylab = ""
        )
        
        draw_group_scatter(
          x_name,
          y_name,
          point_size = 0.7
        )
        
        ######################################################
        # UPPER TRIANGLE: CORRELATIONS
        ######################################################
        
      } else {
        
        plot(
          NA,
          xlim = c(0, 1),
          ylim = c(0, 1),
          axes = FALSE,
          xlab = "",
          ylab = ""
        )
        
        y_positions <- seq(
          0.86,
          0.14,
          length.out = nrow(groups)
        )
        
        for (i in seq_len(nrow(groups))) {
          
          group_name <- groups$group[i]
          
          text(
            x = 0.04,
            y = y_positions[i],
            
            labels = correlation_label(
              group_name,
              x_name,
              y_name
            ),
            
            adj = 0,
            col = group_cols[group_name],
            cex = 0.8,
            font = 2
          )
        }
      }
      
      ######################################################
      # BORDERS AND LABELS
      ######################################################
      
      box(
        col = "grey75",
        lwd = 0.6
      )
      
      if (row == k) {
        axis(
          1,
          cex.axis = 0.7,
          las = 1
        )
      }
      
      if (column == 1L) {
        axis(
          2,
          cex.axis = 0.7,
          las = 1
        )
      }
      
      if (row == 1L) {
        mtext(
          plot_labels[[x_name]],
          side = 3,
          line = 0.2,
          cex = 1
        )
      }
      
      if (column == 1L) {
        mtext(
          plot_labels[[y_name]],
          side = 2,
          line = 1.1,
          cex = 1
        )
      }
    }
  }
  
  draw_legend()
  
  mtext(
    "Local eta + 2 channels: parameter relationships",
    side = 3,
    outer = TRUE,
    line = 1.6,
    cex = 1.2,
    font = 2
  )
  
  mtext(
    significance_caption,
    side = 3,
    outer = TRUE,
    line = 0.5,
    cex = 0.85
  )
  
  invisible(NULL)
}


############################################################
# 13. FIGURE 2 FUNCTION: PERFORMANCE
#
# Top row: accuracy
# Bottom row: optimal choices
# Columns: seven parameters
############################################################

plot_performance <- function() {
  
  if (
    OPEN_NEW_WINDOWS &&
    interactive()
  ) {
    dev.new(
      width = 18,
      height = 9
    )
  }
  
  old_par <- par(no.readonly = TRUE)
  
  on.exit(
    {
      layout(1)
      par(old_par)
    },
    add = TRUE
  )
  
  n_parameters <- length(plot_order)
  
  layout(
    rbind(
      matrix(
        seq_len(2 * n_parameters),
        nrow = 2,
        byrow = TRUE
      ),
      
      rep(
        2 * n_parameters + 1L,
        n_parameters
      )
    ),
    
    heights = c(
      1,
      1,
      0.25
    )
  )
  
  par(
    oma = c(0, 1, 4, 0)
  )
  
  outcomes <- c(
    "accuracy",
    "optimal"
  )
  
  for (outcome in outcomes) {
    
    outcome_values <- participant_data[[outcome]]
    
    outcome_values <- outcome_values[
      is.finite(outcome_values)
    ]
    
    if (!length(outcome_values)) {
      stop(
        "No finite values for ",
        outcome
      )
    }
    
    # Same outcome scale for all seven panels in this row.
    observed_limits <- range(outcome_values)
    y_span <- diff(observed_limits)
    
    if (y_span == 0) {
      y_span <- 1
    }
    
    # Space above the data for correlation labels.
    y_limits <- c(
      observed_limits[1] - 0.08 * y_span,
      observed_limits[2] + 0.65 * y_span
    )
    
    for (parameter in plot_order) {
      
      par(
        mar = c(3.2, 3.3, 2.1, 0.6),
        mgp = c(2, 0.6, 0),
        tcl = -0.25
      )
      
      x_limits <- finite_range(
        participant_data[[parameter]],
        padding = 0.05
      )
      
      plot(
        NA,
        xlim = x_limits,
        ylim = y_limits,
        xlab = "",
        ylab = "",
        axes = FALSE
      )
      
      axis(
        1,
        cex.axis = 0.85
      )
      
      # Keep percentage tick labels within 0–100.
      y_ticks <- pretty(observed_limits)
      
      y_ticks <- y_ticks[
        y_ticks >= 0 &
          y_ticks <= 100 &
          y_ticks >= y_limits[1] &
          y_ticks <= observed_limits[2] + 0.05 * y_span
      ]
      
      axis(
        2,
        at = y_ticks,
        cex.axis = 0.85,
        las = 1
      )
      
      box(
        col = "grey70"
      )
      
      mtext(
        plot_labels[[parameter]],
        side = 1,
        line = 2,
        cex = 1.1
      )
      
      if (parameter == plot_order[1]) {
        mtext(
          plot_labels[[outcome]],
          side = 2,
          line = 2.3,
          cex = 1.05
        )
      }
      
      draw_group_scatter(
        parameter,
        outcome,
        point_size = 0.85
      )
      
      panel_limits <- par("usr")
      
      label_x <- panel_limits[1] +
        0.04 * diff(panel_limits[1:2])
      
      label_y <- panel_limits[4] -
        seq(
          0.06,
          0.30,
          length.out = nrow(groups)
        ) * diff(panel_limits[3:4])
      
      for (i in seq_len(nrow(groups))) {
        
        group_name <- groups$group[i]
        
        text(
          x = label_x,
          y = label_y[i],
          
          labels = correlation_label(
            group_name,
            parameter,
            outcome
          ),
          
          adj = 0,
          col = group_cols[group_name],
          cex = 0.85,
          font = 2
        )
      }
    }
  }
  
  draw_legend()
  
  mtext(
    "Local eta + 2 channels: parameters and performance",
    side = 3,
    outer = TRUE,
    line = 2.3,
    cex = 1.3,
    font = 2
  )
  
  mtext(
    significance_caption,
    side = 3,
    outer = TRUE,
    line = 1.2,
    cex = 0.95
  )
  
  optimal_caption <- if (
    OPTIMAL_USE_TRUE_PRIOR
  ) {
    "Optimal choices: sample evidence and true 65/35 prior"
  } else {
    "Optimal choices: sample evidence and neutral 50/50 prior"
  }
  
  mtext(
    optimal_caption,
    side = 3,
    outer = TRUE,
    line = 0.2,
    cex = 0.9
  )
  
  invisible(NULL)
}


############################################################
# 14. DISPLAY BOTH FIGURES
############################################################

# Figure 1: parameter-only pairs plot.
plot_parameter_pairs()

# Figure 2: parameters versus accuracy and optimal choices.
plot_performance()


############################################################
# 15. RE-DISPLAY AND INSPECT
############################################################

# In RStudio, the second figure appears last.
# Use the back arrow in the Plots pane to see Figure 1.
# Use Zoom to view either figure at a larger size.
#
# Re-display Figure 1:
# plot_parameter_pairs()
#
# Re-display Figure 2:
# plot_performance()
#
# Inspect results:
# View(correlation_results)
# View(participant_data)
#
# To open separate graphics windows:
# OPEN_NEW_WINDOWS <- TRUE
# plot_parameter_pairs()
# plot_performance()
#
# These exploratory correlations use participant posterior
# means; they do not propagate parameter uncertainty.

cat(
  "\nDone: both figures displayed. No files saved.\n"
)











rm(list = ls(all = TRUE))

setwd("/Users/bty615/Documents/GitHub/reliable_info_bias")

library(tidyverse)
library(posterior)

##################################################
## HELPER: LOAD FIRST OBJECT FROM .RDATA
##################################################

load_fit <- function(file_path) {
  obj_name <- load(file_path)
  get(obj_name[1])
}

##################################################
## PATHS
##################################################

fits_dir <- "stan/results/fits/exp11_unaware"
fig_dir  <- "results/figures/global_eta_corrplots"

if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

##################################################
## LOAD ALL FIVE GLOBAL ETA FITS
##################################################

fit_unaware <- load_fit(
  file.path(fits_dir, "fit_trunc_global_eta_unaware_exp11.rdata")
)

fit_aware <- load_fit(
  file.path(fits_dir, "fit_trunc_global_eta_aware_exp11.rdata")
)

fit_explicit <- load_fit(
  file.path(fits_dir, "fit_trunc_global_eta_aware_exp12.rdata")
)

fit_truthful <- load_fit(
  file.path(fits_dir, "fit_trunc_global_eta_truthful_exp13.rdata")
)

fit_deceptive <- load_fit(
  file.path(fits_dir, "fit_trunc_global_eta_deceptive_exp13.rdata")
)

##################################################
## PUT FITS INTO A LIST
##################################################

fit_list <- list(
  unaware_exp11  = fit_unaware,
  aware_exp11    = fit_aware,
  explicit_exp12 = fit_explicit,
  truthful_exp13 = fit_truthful,
  deceptive_exp13 = fit_deceptive
)

##################################################
## EXTRACT GROUP-LEVEL POSTERIOR PARAMETERS
##################################################

extract_pars <- function(fit_obj) {
  
  draws <- as_draws_df(fit_obj$draws())
  
  # ------------------------------------------------
  # Case 1: transformed parameters already available
  # ------------------------------------------------
  
  if (all(
    c(
      "mu_alpha",
      "mu_beta",
      "mu_lambda",
      "mu_delta",
      "mu_eta"
    ) %in% colnames(draws)
  )) {
    
    pars <- draws %>%
      select(
        mu_alpha,
        mu_beta,
        mu_delta,
        mu_lambda,
        mu_eta
      )
    
  } else {
    
    # ------------------------------------------------
    # Case 2: transform raw mu_pr parameters
    # ------------------------------------------------
    
    mu_cols_bracket <- paste0("mu_pr[", 1:5, "]")
    mu_cols_dot     <- paste0("mu_pr.", 1:5, ".")
    
    if (all(mu_cols_bracket %in% colnames(draws))) {
      
      mu_cols <- mu_cols_bracket
      
    } else if (all(mu_cols_dot %in% colnames(draws))) {
      
      mu_cols <- mu_cols_dot
      
    } else {
      
      mu_cols <- colnames(draws)[grepl("^mu_pr(\\[|\\.)", colnames(draws))]
      mu_cols <- mu_cols[1:5]
    }
    
    if (length(mu_cols) < 5) {
      stop("Could not find all five group-level mu parameters.")
    }
    
    pars <- draws %>%
      select(all_of(mu_cols))
    
    colnames(pars) <- c(
      "mu_alpha",
      "mu_beta",
      "mu_lambda",
      "mu_delta",
      "mu_eta"
    )
    
    pars <- pars %>%
      mutate(
        mu_alpha  = pnorm(mu_alpha) * 6,
        mu_lambda = pnorm(mu_lambda),
        mu_delta  = pnorm(mu_delta) * 2
        # mu_beta unchanged
        # mu_eta unchanged
      ) %>%
      select(
        mu_alpha,
        mu_beta,
        mu_delta,
        mu_lambda,
        mu_eta
      )
  }
  
  pars
}

##################################################
## PANEL FUNCTIONS TO MATCH YOUR EXAMPLE
##################################################

panel.hist <- function(x, ...) {
  
  usr <- par("usr")
  on.exit(par(usr))
  
  par(usr = c(usr[1:2], 0, 1.5))
  
  h <- hist(
    x,
    plot = FALSE,
    breaks = 24
  )
  
  breaks <- h$breaks
  y <- h$counts
  y <- y / max(y)
  
  rect(
    xleft   = breaks[-length(breaks)],
    ybottom = 0,
    xright  = breaks[-1],
    ytop    = y,
    col     = "#5BA3D0",
    border  = "#2F6FA5",
    lwd     = 0.4
  )
}

panel.scatter <- function(x, y, ...) {
  points(
    x,
    y,
    pch = 21,
    bg  = rgb(0/255, 70/255, 140/255, 0.75),
    col = rgb(0/255, 45/255, 95/255, 0.95),
    cex = 0.85,
    lwd = 0.35
  )
}

##################################################
## FUNCTION TO SAVE ONE PAIRS PLOT
##################################################

save_pairs_plot <- function(fit_obj, fit_name, fig_dir) {
  
  pars <- extract_pars(fit_obj)
  
  file_name <- file.path(
    fig_dir,
    paste0("corrplot_global_eta_", fit_name, ".png")
  )
  
  png(
    filename = file_name,
    width = 2200,
    height = 2200,
    res = 300
  )
  
  op <- par(
    mar = c(3.2, 3.2, 2.2, 1.2),
    mgp = c(1.8, 0.5, 0),
    tcl = -0.25
  )
  
  pairs(
    pars,
    labels = c(
      "mu_alpha",
      "mu_beta",
      "mu_delta",
      "mu_lambda",
      "mu_eta"
    ),
    diag.panel  = panel.hist,
    lower.panel = panel.scatter,
    upper.panel = panel.scatter,
    gap = 0.25,
    cex.labels = 1.1,
    font.labels = 1,
    las = 1
  )
  
  par(op)
  dev.off()
  
  message("Saved: ", file_name)
}

##################################################
## LOOP THROUGH ALL FIVE GROUPS
##################################################

for (fit_name in names(fit_list)) {
  save_pairs_plot(
    fit_obj  = fit_list[[fit_name]],
    fit_name = fit_name,
    fig_dir  = fig_dir
  )
}

##################################################
## DONE
##################################################

cat("\nDone.\n")
cat("Plots saved in:\n")
cat(fig_dir, "\n")













############################################################
# LEARNING + LOCAL ETA + TWO CHANNELS + INSTRUCTION
#
# Diagonal: posterior densities
# Lower panels: posterior draws
# Upper panels: within-group posterior correlations
#
# Loads all five fits directly.
# Displays only. No files are saved.
############################################################

suppressPackageStartupMessages({
  library(cmdstanr)
  library(posterior)
})

############################################################
# 1. FILE LOCATION
############################################################

fits_dir <- path.expand(
  "~/Documents/GitHub/reliable_info_bias/stan/results/fits/exp11_unaware"
)

fit_paths <- c(
  "Implicit Unaware" = file.path(
    fits_dir,
    "learning_leta_2channel_instr_unaware_exp11.rdata"
  ),
  "Implicit Aware" = file.path(
    fits_dir,
    "learning_leta_2channel_instr_aware_exp11.rdata"
  ),
  "Explicit Undirected" = file.path(
    fits_dir,
    "learning_leta_2channel_instr_aware_exp12.rdata"
  ),
  "Explicit Truthful" = file.path(
    fits_dir,
    "learning_leta_2channel_instr_truthful_exp13.rdata"
  ),
  "Explicit Deceptive" = file.path(
    fits_dir,
    "learning_leta_2channel_instr_deceptive_exp13.rdata"
  )
)

missing_files <- fit_paths[!file.exists(fit_paths)]

if (length(missing_files)) {
  stop(
    "Missing fit files:\n",
    paste(missing_files, collapse = "\n")
  )
}

############################################################
# 2. GROUP COLOURS AND LABELS
############################################################

group_cols <- c(
  "Implicit Unaware"    = "#F4A6A6",
  "Implicit Aware"      = "#81C784",
  "Explicit Undirected" = "#E69F00",
  "Explicit Truthful"   = "#1B5E20",
  "Explicit Deceptive"  = "#8B1A1A"
)

group_short <- c(
  "Implicit Unaware"    = "IU",
  "Implicit Aware"      = "IA",
  "Explicit Undirected" = "EU",
  "Explicit Truthful"   = "ET",
  "Explicit Deceptive"  = "ED"
)

############################################################
# 3. PARAMETERS
############################################################

# Named, transformed group-level parameters.
# No guessing of raw parameter indices or transformations.

parameter_names <- c(
  "mu_alpha",
  "mu_beta",
  "mu_delta",
  "mu_lambda",
  "mu_eta",
  "mu_alpha0",
  "mu_beta0",
  "mu_rho",
  "Delta"
)

parameter_labels <- list(
  mu_alpha  = expression(mu[alpha]),
  mu_beta   = expression(mu[beta]),
  mu_delta  = expression(mu[delta]),
  mu_lambda = expression(mu[lambda]),
  mu_eta    = expression(mu[eta]),
  mu_alpha0 = expression(mu[alpha[0]]),
  mu_beta0  = expression(mu[beta[0]]),
  mu_rho    = expression(mu[rho]),
  Delta     = expression(Delta)
)

applicable_parameters <- function(group_name) {
  
  pars <- parameter_names
  
  # No instruction-trust parameter displayed for implicit groups.
  if (group_name %in% c("Implicit Unaware", "Implicit Aware")) {
    pars <- setdiff(pars, c("mu_rho", "Delta"))
  }
  
  # The undirected instruction band applies only to that group.
  if (group_name != "Explicit Undirected") {
    pars <- setdiff(pars, "Delta")
  }
  
  pars
}

############################################################
# 4. LOAD A SAVED FIT
############################################################

load_fit <- function(file_path) {
  
  env <- new.env()
  
  # Encoding warnings about stored comments may appear here.
  # Warnings are not suppressed.
  loaded_names <- load(file_path, envir = env)
  
  fit_names <- loaded_names[
    vapply(
      loaded_names,
      function(nm) inherits(env[[nm]], "CmdStanMCMC"),
      logical(1)
    )
  ]
  
  if (length(fit_names) != 1L) {
    stop(
      "Expected exactly one CmdStanMCMC fit in:\n",
      file_path
    )
  }
  
  env[[fit_names]]
}

############################################################
# 5. EXTRACT POSTERIOR DRAWS
############################################################

extract_pars <- function(fit_obj, group_name) {
  
  required <- applicable_parameters(group_name)
  
  available <- posterior::variables(fit_obj$draws())
  missing <- setdiff(required, available)
  
  if (length(missing)) {
    stop(
      "\nMissing parameters for ", group_name, ":\n",
      paste(missing, collapse = ", "),
      "\nThis script expects learning_leta_2channel_instr,",
      "\nincluding mu_eta. Check the selected model file."
    )
  }
  
  draws <- posterior::as_draws_matrix(
    fit_obj$draws(variables = required)
  )
  
  pars <- as.data.frame(draws)
  pars <- pars[, required, drop = FALSE]
  
  if (!nrow(pars) || any(!is.finite(as.matrix(pars)))) {
    stop("Missing or non-finite posterior draws for ", group_name)
  }
  
  # Check convergence of the plotted parameters.
  diagnostics <- fit_obj$summary(variables = required)
  
  if (
    "rhat" %in% names(diagnostics) &&
    any(diagnostics$rhat > 1.01, na.rm = TRUE)
  ) {
    warning(
      group_name,
      ": at least one plotted parameter has R-hat > 1.01."
    )
  }
  
  # Inapplicable instruction parameters are left blank.
  for (p in setdiff(parameter_names, required)) {
    pars[[p]] <- NA_real_
  }
  
  pars <- pars[, parameter_names, drop = FALSE]
  pars$Group <- group_name
  
  pars
}

############################################################
# 6. LOAD ALL FIVE GROUPS DIRECTLY
############################################################

pars_list <- setNames(
  vector("list", length(fit_paths)),
  names(fit_paths)
)

for (g in names(fit_paths)) {
  
  cat("\nLoading:", g, "\n")
  cat(fit_paths[[g]], "\n")
  
  current_fit <- load_fit(fit_paths[[g]])
  pars_list[[g]] <- extract_pars(current_fit, g)
  
  rm(current_fit)
}

all_pars <- do.call(rbind, pars_list)
rownames(all_pars) <- NULL

all_pars$Group <- factor(
  all_pars$Group,
  levels = names(group_cols)
)

cat("\nPosterior draws per group:\n")
print(table(all_pars$Group))

############################################################
# 7. PARAMETER RANGES
############################################################

param_ranges <- setNames(
  lapply(parameter_names, function(p) {
    
    x <- all_pars[[p]]
    x <- x[is.finite(x)]
    
    if (!length(x)) {
      stop("No finite draws for ", p)
    }
    
    limits <- range(x)
    
    if (diff(limits) == 0) {
      limits <- limits + c(-0.5, 0.5)
    }
    
    limits + c(-1, 1) * diff(limits) * 0.03
  }),
  parameter_names
)

############################################################
# 8. SCATTER PANEL
############################################################

draw_scatter_panel <- function(
    x, y, x_name, y_name,
    show_x_axis = FALSE,
    show_y_axis = FALSE
) {
  
  plot(
    NA,
    xlim = param_ranges[[x_name]],
    ylim = param_ranges[[y_name]],
    xlab = "", ylab = "",
    axes = FALSE,
    type = "n"
  )
  
  box(col = "grey70", lwd = 0.8)
  
  for (g in names(group_cols)) {
    
    idx <- all_pars$Group == g &
      is.finite(x) & is.finite(y)
    
    points(
      x[idx], y[idx],
      pch = 16,
      cex = 0.40,
      col = adjustcolor(
        group_cols[g],
        alpha.f = 0.25
      )
    )
  }
  
  if (show_x_axis) {
    axis(1, cex.axis = 0.65, las = 1)
  }
  
  if (show_y_axis) {
    axis(2, cex.axis = 0.65, las = 1)
  }
}

############################################################
# 9. DENSITY PANEL
############################################################

draw_density_panel <- function(
    x, parameter_name,
    show_x_axis = FALSE,
    show_y_axis = FALSE
) {
  
  dens_list <- list()
  constant_values <- list()
  
  for (g in names(group_cols)) {
    
    xx <- x[
      all_pars$Group == g & is.finite(x)
    ]
    
    if (length(xx) < 2L) next
    
    if (length(unique(xx)) == 1L) {
      constant_values[[g]] <- xx[1]
    } else {
      dens_list[[g]] <- density(xx, n = 512)
    }
  }
  
  y_max <- if (length(dens_list)) {
    max(vapply(
      dens_list,
      function(d) max(d$y),
      numeric(1)
    ))
  } else {
    1
  }
  
  plot(
    NA,
    xlim = param_ranges[[parameter_name]],
    ylim = c(0, y_max * 1.08),
    xlab = "", ylab = "",
    axes = FALSE,
    type = "n"
  )
  
  box(col = "grey70", lwd = 0.8)
  
  for (g in names(dens_list)) {
    
    d <- dens_list[[g]]
    
    polygon(
      c(d$x, rev(d$x)),
      c(d$y, rep(0, length(d$y))),
      col = adjustcolor(
        group_cols[g],
        alpha.f = 0.08
      ),
      border = NA
    )
    
    lines(
      d$x, d$y,
      col = group_cols[g],
      lwd = 2
    )
  }
  
  for (g in names(constant_values)) {
    abline(
      v = constant_values[[g]],
      col = group_cols[g],
      lwd = 2
    )
  }
  
  if (show_x_axis) {
    axis(1, cex.axis = 0.65, las = 1)
  }
  
  if (show_y_axis) {
    axis(2, cex.axis = 0.65, las = 1)
  }
}

############################################################
# 10. CORRELATION PANEL
############################################################

draw_correlation_panel <- function(x, y) {
  
  plot(
    NA,
    xlim = c(0, 1),
    ylim = c(0, 1),
    xlab = "", ylab = "",
    axes = FALSE,
    type = "n"
  )
  
  box(col = "grey75", lwd = 0.7)
  
  levs <- names(group_cols)
  y_positions <- seq(
    0.84, 0.16,
    length.out = length(levs)
  )
  
  for (i in seq_along(levs)) {
    
    g <- levs[i]
    
    idx <- all_pars$Group == g &
      is.finite(x) & is.finite(y)
    
    if (
      sum(idx) > 2L &&
      sd(x[idx]) > 0 &&
      sd(y[idx]) > 0
    ) {
      
      label <- paste0(
        group_short[g],
        "  r = ",
        sprintf("%.2f", cor(x[idx], y[idx]))
      )
      
    } else {
      
      label <- paste0(group_short[g], "  --")
    }
    
    text(
      x = 0.05,
      y = y_positions[i],
      labels = label,
      adj = c(0, 0.5),
      col = group_cols[g],
      cex = 0.68,
      font = 2
    )
  }
}

############################################################
# 11. BOTTOM LEGEND
############################################################

draw_bottom_legend <- function() {
  
  plot(
    NA,
    xlim = c(0, 1),
    ylim = c(0, 1),
    axes = FALSE,
    xlab = "", ylab = "",
    type = "n"
  )
  
  legend(
    "center",
    legend = names(group_cols),
    col = group_cols,
    pch = 16,
    lty = 1,
    lwd = 2,
    ncol = 3,
    bty = "n",
    cex = 0.8
  )
}

############################################################
# 12. COMPLETE FIGURE
############################################################

draw_complete_figure <- function() {
  
  old_par <- par(no.readonly = TRUE)
  
  on.exit({
    layout(1)
    par(old_par)
  }, add = TRUE)
  
  npar <- length(parameter_names)
  
  layout_matrix <- rbind(
    matrix(
      seq_len(npar * npar),
      nrow = npar,
      ncol = npar,
      byrow = TRUE
    ),
    rep(npar * npar + 1L, npar)
  )
  
  par(oma = c(0, 0, 2.5, 0))
  
  layout(
    layout_matrix,
    widths = rep(1, npar),
    heights = c(rep(1, npar), 0.8)
  )
  
  for (row in seq_len(npar)) {
    for (col in seq_len(npar)) {
      
      par(
        mar = c(1.15, 1.15, 0.85, 0.15),
        mgp = c(0.8, 0.2, 0),
        tcl = -0.15
      )
      
      x_name <- parameter_names[col]
      y_name <- parameter_names[row]
      
      x <- all_pars[[x_name]]
      y <- all_pars[[y_name]]
      
      if (row == col) {
        
        draw_density_panel(
          x = x,
          parameter_name = x_name,
          show_x_axis = row == npar,
          show_y_axis = col == 1
        )
        
      } else if (row > col) {
        
        draw_scatter_panel(
          x = x,
          y = y,
          x_name = x_name,
          y_name = y_name,
          show_x_axis = row == npar,
          show_y_axis = col == 1
        )
        
      } else {
        
        draw_correlation_panel(x, y)
      }
      
      if (row == 1) {
        mtext(
          parameter_labels[[x_name]],
          side = 3,
          line = 0.12,
          cex = 0.9
        )
      }
      
      if (col == 1) {
        mtext(
          parameter_labels[[y_name]],
          side = 2,
          line = 0.65,
          cex = 0.9
        )
      }
    }
  }
  
  par(mar = c(0, 0, 0, 0))
  draw_bottom_legend()
  
  mtext(
    "Learning + local eta + two channels + instruction",
    side = 3,
    outer = TRUE,
    line = 0.8,
    cex = 1.05,
    font = 2
  )
}

############################################################
# 13. DISPLAY ONLY
############################################################

draw_complete_figure()













############################################################
# LEARNING + TWO CHANNELS + INSTRUCTION (WITHOUT LOCAL ETA)
#
# Diagonal: posterior densities for all five groups
# Lower triangle: joint posterior draws
# Upper triangle: within-group posterior correlations
#
# Also prints R-hat, ESS and sampler diagnostics.
# Displays the plot only; does not save any output files.
############################################################

suppressPackageStartupMessages({
  library(cmdstanr)
  library(posterior)
})

# ==========================================================
# 1. LOCATION OF SAVED FITS
# ==========================================================
project_dir <- path.expand("~/Documents/GitHub/reliable_info_bias")
fits_root   <- file.path(project_dir, "stan", "results", "fits")
fits_old    <- file.path(fits_root, "exp11_unaware")
fits_exp12  <- file.path(fits_root, "exp12_undirected")

# The new Exp12 HPC fit can be stored as learning_2channel_instr.rdata
# in exp12_undirected. Other fits normally have group-specific names.
# For each group, the first existing candidate is used.
first_existing <- function(candidates, group_name) {
  found <- candidates[file.exists(candidates)]
  if (length(found) == 0L) {
    stop(
      "No fit found for ", group_name, ". Checked:\n",
      paste(candidates, collapse = "\n"),
      "\nEdit the candidate paths in Section 1."
    )
  }
  normalizePath(found[1L], mustWork = TRUE)
}

candidate_paths <- list(
  "Implicit Unaware" = c(
    file.path(fits_old, "learning_2channel_instr_unaware_exp11.rdata"),
    file.path(fits_old, "learning_2channel_instr_unaware_exp11.rds")
  ),
  "Implicit Aware" = c(
    file.path(fits_old, "learning_2channel_instr_aware_exp11.rdata"),
    file.path(fits_old, "learning_2channel_instr_aware_exp11.rds")
  ),
  "Explicit Undirected" = c(
    file.path(fits_exp12, "learning_2channel_instr.rdata"),
    file.path(fits_exp12, "learning_2channel_instr.rds"),
    file.path(fits_old, "learning_2channel_instr_aware_exp12.rdata"),
    file.path(fits_old, "learning_2channel_instr_aware_exp12.rds")
  ),
  "Explicit Truthful" = c(
    file.path(fits_old, "learning_2channel_instr_truthful_exp13.rdata"),
    file.path(fits_old, "learning_2channel_instr_truthful_exp13.rds")
  ),
  "Explicit Deceptive" = c(
    file.path(fits_old, "learning_2channel_instr_deceptive_exp13.rdata"),
    file.path(fits_old, "learning_2channel_instr_deceptive_exp13.rds")
  )
)

fit_paths <- vapply(names(candidate_paths), function(g) {
  first_existing(candidate_paths[[g]], g)
}, character(1))

cat("\nFITS SELECTED:\n")
print(fit_paths)

# ==========================================================
# 2. COLOURS AND LABELS
# ==========================================================
group_cols <- c(
  "Implicit Unaware"     = "#F4A6A6",
  "Implicit Aware"       = "#81C784",
  "Explicit Undirected"  = "#E69F00",
  "Explicit Truthful"    = "#1B5E20",
  "Explicit Deceptive"   = "#8B1A1A"
)

group_short <- c(
  "Implicit Unaware"     = "IU",
  "Implicit Aware"       = "IA",
  "Explicit Undirected"  = "EU",
  "Explicit Truthful"    = "ET",
  "Explicit Deceptive"   = "ED"
)

# Eight parameters. There is intentionally NO mu_eta.
parameter_names <- c(
  "mu_alpha", "mu_beta", "mu_delta", "mu_lambda",
  "mu_alpha0", "mu_beta0", "mu_rho", "Delta"
)

parameter_labels <- list(
  mu_alpha  = expression(mu[alpha]),
  mu_beta   = expression(mu[beta]),
  mu_delta  = expression(mu[delta]),
  mu_lambda = expression(mu[lambda]),
  mu_alpha0 = expression(mu[alpha[0]]),
  mu_beta0  = expression(mu[beta[0]]),
  mu_rho    = expression(mu[rho]),
  Delta     = expression(Delta)
)

applicable_parameters <- function(group_name) {
  p <- parameter_names
  if (group_name %in% c("Implicit Unaware", "Implicit Aware")) {
    p <- setdiff(p, c("mu_rho", "Delta"))
  }
  if (group_name != "Explicit Undirected") {
    p <- setdiff(p, "Delta")
  }
  p
}

# ==========================================================
# 3. LOAD FITS AND CHECK DIAGNOSTICS
# ==========================================================
load_fit <- function(path) {
  if (grepl("\\.rds$", path, ignore.case = TRUE)) {
    fit <- readRDS(path)
  } else {
    env <- new.env(parent = globalenv())
    loaded <- load(path, envir = env)
    fit_names <- loaded[vapply(loaded, function(nm) {
      inherits(env[[nm]], "CmdStanMCMC")
    }, logical(1))]
    if (length(fit_names) != 1L) {
      stop("Expected exactly one CmdStanMCMC object in: ", path)
    }
    fit <- env[[fit_names]]
  }
  if (!inherits(fit, "CmdStanMCMC")) {
    stop("Saved object is not a CmdStanMCMC fit: ", path)
  }
  fit
}

all_diagnostics <- list()
all_sampler_diagnostics <- list()
pars_list <- list()

for (g in names(fit_paths)) {
  cat("\nLoading ", g, " ...\n", sep = "")
  fit <- load_fit(fit_paths[[g]])
  required <- applicable_parameters(g)
  
  # Read ONLY the required transformed group-level variables.
  # Do not accidentally load the very large per-trial log_lik draws.
  draw_array <- tryCatch(
    fit$draws(variables = required),
    error = function(e) stop(g, ": ", conditionMessage(e))
  )
  present <- posterior::variables(draw_array)
  missing <- setdiff(required, present)
  if (length(missing)) {
    stop(g, " is missing: ", paste(missing, collapse = ", "),
         "\nCheck the model file and parameter names.")
  }
  
  pars <- as.data.frame(posterior::as_draws_matrix(draw_array))
  pars <- pars[, required, drop = FALSE]
  if (!nrow(pars) || any(!is.finite(as.matrix(pars)))) {
    stop("Missing or non-finite posterior draws: ", g)
  }
  
  # For models where rho/Delta do not apply, leave those panels empty.
  for (p in setdiff(parameter_names, required)) pars[[p]] <- NA_real_
  pars <- pars[, parameter_names, drop = FALSE]
  pars$Group <- g
  pars_list[[g]] <- pars
  
  diagnostics <- fit$summary(variables = required)
  wanted <- intersect(c("variable", "mean", "rhat", "ess_bulk", "ess_tail"),
                      names(diagnostics))
  diagnostics <- as.data.frame(diagnostics[, wanted, drop = FALSE])
  diagnostics$group <- g
  all_diagnostics[[g]] <- diagnostics
  
  sampler <- tryCatch(fit$diagnostic_summary(), error = function(e) NULL)
  all_sampler_diagnostics[[g]] <- sampler
  
  if ("rhat" %in% names(diagnostics) &&
      any(diagnostics$rhat > 1.01, na.rm = TRUE)) {
    warning(g, ": at least one R-hat > 1.01; do not assume convergence.")
  }
  if ("ess_bulk" %in% names(diagnostics) &&
      any(diagnostics$ess_bulk < 400, na.rm = TRUE)) {
    warning(g, ": at least one bulk ESS < 400.")
  }
  rm(fit, draw_array, pars)
  gc(verbose = FALSE)
}

all_pars <- do.call(rbind, pars_list)
rownames(all_pars) <- NULL
all_pars$Group <- factor(all_pars$Group, levels = names(group_cols))

cat("\nPOSTERIOR DRAWS PER GROUP:\n")
print(table(all_pars$Group))

cat("\nGROUP-LEVEL PARAMETER DIAGNOSTICS:\n")
diagnostic_table <- do.call(rbind, all_diagnostics)
rownames(diagnostic_table) <- NULL
print(diagnostic_table, row.names = FALSE, digits = 3)
cat("\nNOTE: Delta is fixed at 0.1 in the undirected fit; its R-hat/ESS may be NA.\n")

cat("\nSAMPLER DIAGNOSTICS (when available):\n")
for (g in names(all_sampler_diagnostics)) {
  cat("\n", g, ":\n", sep = "")
  if (is.null(all_sampler_diagnostics[[g]])) {
    cat("Not available from saved fit.\n")
  } else {
    print(all_sampler_diagnostics[[g]])
  }
}

# ==========================================================
# 4. AXIS RANGES
# ==========================================================
param_ranges <- setNames(lapply(parameter_names, function(p) {
  x <- all_pars[[p]]
  x <- x[is.finite(x)]
  if (!length(x)) stop("No finite draws for ", p)
  lim <- range(x)
  if (diff(lim) == 0) lim <- lim + c(-0.5, 0.5)
  lim + c(-1, 1) * diff(lim) * 0.03
}), parameter_names)

# ==========================================================
# 5. SCATTER, DENSITY AND CORRELATION PANELS
# ==========================================================
draw_scatter_panel <- function(x, y, x_name, y_name,
                               show_x_axis = FALSE, show_y_axis = FALSE) {
  plot(NA, xlim = param_ranges[[x_name]],
       ylim = param_ranges[[y_name]], xlab = "", ylab = "",
       axes = FALSE, type = "n")
  box(col = "grey70", lwd = 0.8)
  for (g in names(group_cols)) {
    idx <- all_pars$Group == g & is.finite(x) & is.finite(y)
    points(x[idx], y[idx], pch = 16, cex = 0.40,
           col = adjustcolor(group_cols[g], alpha.f = 0.25))
  }
  if (show_x_axis) axis(1, cex.axis = 0.65, las = 1)
  if (show_y_axis) axis(2, cex.axis = 0.65, las = 1)
}

draw_density_panel <- function(x, parameter_name,
                               show_x_axis = FALSE, show_y_axis = FALSE) {
  densities <- list()
  constant_values <- list()
  for (g in names(group_cols)) {
    xx <- x[all_pars$Group == g & is.finite(x)]
    if (length(xx) < 2L) next
    if (length(unique(xx)) == 1L) {
      constant_values[[g]] <- xx[1]
    } else {
      densities[[g]] <- density(xx, n = 512)
    }
  }
  ymax <- if (length(densities)) {
    max(vapply(densities, function(d) max(d$y), numeric(1)))
  } else 1
  plot(NA, xlim = param_ranges[[parameter_name]], ylim = c(0, ymax * 1.08),
       xlab = "", ylab = "", axes = FALSE, type = "n")
  box(col = "grey70", lwd = 0.8)
  for (g in names(densities)) {
    d <- densities[[g]]
    polygon(c(d$x, rev(d$x)), c(d$y, rep(0, length(d$y))),
            col = adjustcolor(group_cols[g], alpha.f = 0.08), border = NA)
    lines(d$x, d$y, col = group_cols[g], lwd = 2)
  }
  for (g in names(constant_values)) {
    abline(v = constant_values[[g]], col = group_cols[g], lwd = 2)
  }
  if (show_x_axis) axis(1, cex.axis = 0.65, las = 1)
  if (show_y_axis) axis(2, cex.axis = 0.65, las = 1)
}

draw_correlation_panel <- function(x, y) {
  plot(NA, xlim = c(0, 1), ylim = c(0, 1),
       xlab = "", ylab = "", axes = FALSE, type = "n")
  box(col = "grey75", lwd = 0.7)
  levs <- names(group_cols)
  y_positions <- seq(0.84, 0.16, length.out = length(levs))
  for (i in seq_along(levs)) {
    g <- levs[i]
    idx <- all_pars$Group == g & is.finite(x) & is.finite(y)
    if (sum(idx) > 2L && sd(x[idx]) > 0 && sd(y[idx]) > 0) {
      label <- sprintf("%s  r = %.2f", group_short[g], cor(x[idx], y[idx]))
    } else {
      label <- paste0(group_short[g], "  --")
    }
    text(0.05, y_positions[i], labels = label, adj = c(0, 0.5),
         col = group_cols[g], cex = 0.68, font = 2)
  }
}

draw_bottom_legend <- function() {
  plot(NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "", ylab = "",
       axes = FALSE, type = "n")
  legend("center", legend = names(group_cols), col = group_cols,
         pch = 16, lty = 1, lwd = 2, ncol = 3,
         bty = "n", cex = 0.8)
}

# ==========================================================
# 6. FULL CORRELATION MATRIX
# ==========================================================
draw_complete_figure <- function() {
  old_par <- par(no.readonly = TRUE)
  on.exit({ layout(1); par(old_par) }, add = TRUE)
  
  npar <- length(parameter_names)
  layout_matrix <- rbind(
    matrix(seq_len(npar * npar), nrow = npar, ncol = npar, byrow = TRUE),
    rep(npar * npar + 1L, npar)
  )
  par(oma = c(0, 0, 2.5, 0))
  layout(layout_matrix, widths = rep(1, npar),
         heights = c(rep(1, npar), 0.8))
  
  for (row in seq_len(npar)) {
    for (col in seq_len(npar)) {
      par(mar = c(1.15, 1.15, 0.85, 0.15), mgp = c(0.8, 0.2, 0), tcl = -0.15)
      x_name <- parameter_names[col]
      y_name <- parameter_names[row]
      x <- all_pars[[x_name]]
      y <- all_pars[[y_name]]
      if (row == col) {
        draw_density_panel(x, x_name, row == npar, col == 1)
      } else if (row > col) {
        draw_scatter_panel(x, y, x_name, y_name, row == npar, col == 1)
      } else {
        draw_correlation_panel(x, y)
      }
      if (row == 1) {
        mtext(parameter_labels[[x_name]], side = 3, line = 0.12, cex = 0.9)
      }
      if (col == 1) {
        mtext(parameter_labels[[y_name]], side = 2, line = 0.65, cex = 0.9)
      }
    }
  }
  par(mar = c(0, 0, 0, 0))
  draw_bottom_legend()
  mtext("Learning + two channels + instruction (without eta)",
        side = 3, outer = TRUE, line = 0.8, cex = 1.05, font = 2)
}

# ==========================================================
# 47. DISPLAY THE FIGURE
# ==========================================================
# Expand the RStudio Plot panel or click Zoom for legibility.
draw_complete_figure()







############################################################
# LEARNING + TWO CHANNELS + INSTRUCTION (NO LOCAL ETA)
# PARTICIPANT-LEVEL PARAMETER CORRELATIONS
#
# Matches the layout of your previous Local eta + 2 channels plot:
#   - Diagonal: densities of PARTICIPANT posterior means
#   - Lower triangle: one point per participant, per group
#   - Regression line ONLY when Pearson p < .05 (by default)
#   - Upper triangle: r with *, **, *** significance stars
#   - Same five group colours and abbreviated labels
#
# This is NOT the same as correlating MCMC group-level draws.
# Here, participants (not MCMC iterations) are observations.
#
# No behavioural data files are needed. No files are saved.
# Delta is fixed (0.1) and therefore omitted from correlations.
# rho appears only for explicit groups and only if individual
# participant rho estimates exist in the saved fit.
############################################################

suppressPackageStartupMessages({
  library(cmdstanr)
  library(tidyverse)
})

############################################################
# 1. SETTINGS
############################################################

project_dir <- path.expand(
  "~/Documents/GitHub/reliable_info_bias"
)

fits_root <- file.path(project_dir, "stan", "results", "fits")
fits_old  <- file.path(fits_root, "exp11_unaware")
fits_exp12 <- file.path(fits_root, "exp12_undirected")

P_CUTOFF <- 0.05
P_ADJUST <- "none"   # "none" matches your original; "BH" is optional
OPEN_NEW_WINDOW <- FALSE

stopifnot(P_ADJUST %in% c("none", "BH"))

# IMPORTANT: Check the 'params[i,j]' ordering in your actual
# learning_2channel_instr.stan model before interpreting plots.
# This is the old two-channel order with eta removed and rho added.
# IF THE STAN ORDER DIFFERS, EDIT THIS VECTOR.
PARAM_INDEX_ORDER <- c(
  "alpha", "beta", "lambda", "alpha0", "beta0", "delta", "rho"
)

# If a fit contains only six participant parameters, the script
# assumes rho is absent; it does NOT substitute group-level mu_rho.

############################################################
# 2. GROUPS AND FIT PATHS
############################################################

groups <- tribble(
  ~group,                 ~colour,    ~short,
  "Implicit Unaware",    "#F4A3A3", "IU",
  "Implicit Aware",      "#8FD694", "IA",
  "Explicit Undirected", "#E69F00", "EU",
  "Explicit True",       "#1B7837", "ET",
  "Explicit Deceptive",  "#8B0000", "ED"
)

group_cols  <- setNames(groups$colour, groups$group)
group_short <- setNames(groups$short, groups$group)

# First existing file for each group is used.
# Prefer your new HPC Exp12 model over an older Exp12 fit.
candidate_paths <- list(
  "Implicit Unaware" = c(
    file.path(fits_old, "learning_2channel_instr_unaware_exp11.rdata"),
    file.path(fits_old, "learning_2channel_instr_unaware_exp11.rds")
  ),
  "Implicit Aware" = c(
    file.path(fits_old, "learning_2channel_instr_aware_exp11.rdata"),
    file.path(fits_old, "learning_2channel_instr_aware_exp11.rds")
  ),
  "Explicit Undirected" = c(
    file.path(fits_exp12, "learning_2channel_instr.rdata"),
    file.path(fits_exp12, "learning_2channel_instr.rds"),
    file.path(fits_old, "learning_2channel_instr_aware_exp12.rdata"),
    file.path(fits_old, "learning_2channel_instr_aware_exp12.rds")
  ),
  "Explicit True" = c(
    file.path(fits_old, "learning_2channel_instr_truthful_exp13.rdata"),
    file.path(fits_old, "learning_2channel_instr_truthful_exp13.rds")
  ),
  "Explicit Deceptive" = c(
    file.path(fits_old, "learning_2channel_instr_deceptive_exp13.rdata"),
    file.path(fits_old, "learning_2channel_instr_deceptive_exp13.rds")
  )
)

first_existing <- function(paths, group_name) {
  match <- paths[file.exists(paths)]
  if (!length(match)) {
    stop("Fit missing for ", group_name, ". Checked:\n",
         paste(paths, collapse = "\n"))
  }
  normalizePath(match[1], mustWork = TRUE)
}

fit_paths <- vapply(
  names(candidate_paths),
  function(g) first_existing(candidate_paths[[g]], g),
  character(1)
)

cat("\nUSING THESE FIT FILES:\n")
print(fit_paths)

############################################################
# 3. PARAMETER ORDER AND LABELS
############################################################

plot_order <- c(
  "alpha", "beta", "delta", "lambda", "alpha0", "beta0", "rho"
)

plot_labels <- list(
  alpha  = expression(alpha),
  beta   = expression(beta),
  delta  = expression(delta),
  lambda = expression(lambda),
  alpha0 = expression(alpha[0]),
  beta0  = expression(beta[0]),
  rho    = expression(rho)
)

mu_names <- c(
  alpha = "mu_alpha", beta = "mu_beta",
  lambda = "mu_lambda", alpha0 = "mu_alpha0",
  beta0 = "mu_beta0", delta = "mu_delta", rho = "mu_rho"
)

############################################################
# 4. LOAD AND EXTRACT PARTICIPANT ESTIMATES
############################################################

load_fit <- function(path) {
  if (grepl("\\.rds$", path, ignore.case = TRUE)) {
    fit <- readRDS(path)
  } else {
    env <- new.env(parent = globalenv())
    loaded <- load(path, envir = env)
    nm <- loaded[vapply(loaded, function(z) {
      inherits(env[[z]], "CmdStanMCMC")
    }, logical(1))]
    if (length(nm) != 1L) {
      stop("Expected one CmdStanMCMC fit in: ", path)
    }
    fit <- env[[nm]]
  }
  if (!inherits(fit, "CmdStanMCMC")) {
    stop("Not a CmdStanMCMC object: ", path)
  }
  fit
}

extract_participants <- function(group_name, path) {
  cat("\nLoading: ", group_name, "\n", sep = "")
  fit <- load_fit(path)
  
  # Already-transformed individual parameters (no raw mu_pr guessing).
  smry <- as_tibble(fit$summary(variables = "params"))
  if (!all(c("variable", "mean") %in% names(smry))) {
    stop(group_name, ": missing variable/mean in params summary")
  }
  
  ind <- stringr::str_match(smry$variable, "^params\\[([0-9]+),([0-9]+)\\]$")
  smry <- smry %>%
    mutate(
      person = as.integer(ind[, 2]),
      index  = as.integer(ind[, 3])
    ) %>%
    filter(!is.na(person), !is.na(index))
  
  if (!nrow(smry)) {
    stop(group_name, ": couldn't find params[person,index] in the fit")
  }
  
  indices <- sort(unique(smry$index))
  if (!identical(indices, seq_along(indices)) ||
      !length(indices) %in% c(6L, 7L)) {
    stop(group_name, ": unexpected params indices: ",
         paste(indices, collapse = ", "),
         ". Check Stan's generated quantities.")
  }
  
  expected_n <- length(unique(smry$person)) * length(indices)
  if (nrow(smry) != expected_n || anyDuplicated(smry$variable)) {
    stop(group_name, ": incomplete/duplicate participant parameter grid")
  }
  
  if (any(!is.finite(smry$mean))) {
    stop(group_name, ": non-finite participant posterior estimates")
  }
  
  # NOTE: verify the order above against the Stan generated-quantities code.
  smry <- smry %>%
    mutate(parameter = PARAM_INDEX_ORDER[index])
  
  diagnostics <- smry %>%
    summarise(
      n_participants = n_distinct(person),
      max_rhat = if ("rhat" %in% names(smry))
        suppressWarnings(max(rhat, na.rm = TRUE)) else NA_real_,
      min_bulk_ess = if ("ess_bulk" %in% names(smry))
        suppressWarnings(min(ess_bulk, na.rm = TRUE)) else NA_real_,
      .groups = "drop"
    ) %>%
    mutate(group = group_name, .before = 1)
  
  if (is.infinite(diagnostics$max_rhat)) diagnostics$max_rhat <- NA_real_
  if (is.infinite(diagnostics$min_bulk_ess)) diagnostics$min_bulk_ess <- NA_real_
  
  if (is.finite(diagnostics$max_rhat) && diagnostics$max_rhat > 1.01) {
    warning(group_name, ": participant-parameter R-hat > 1.01. Interpret cautiously.")
  }
  if (is.finite(diagnostics$min_bulk_ess) && diagnostics$min_bulk_ess < 400) {
    warning(group_name, ": some participant parameters have bulk ESS < 400.")
  }
  
  p <- smry %>%
    select(person, parameter, value = mean) %>%
    pivot_wider(names_from = parameter, values_from = value) %>%
    arrange(person)
  
  # Only explicit groups have meaningful instruction trust in this task.
  if (group_name %in% c("Implicit Unaware", "Implicit Aware")) {
    p$rho <- NA_real_
  }
  for (z in plot_order) if (!z %in% names(p)) p[[z]] <- NA_real_
  
  # Check against group-level means to catch obvious parameter-index mistakes.
  # This is an advisory check (hierarchical group means need not equal
  # participant sample means exactly).
  mu_available <- tryCatch(
    posterior::variables(fit$draws(variables = unname(mu_names))),
    error = function(e) character()
  )
  present_mu <- intersect(unname(mu_names), mu_available)
  if (length(present_mu)) {
    mu_summ <- fit$summary(variables = present_mu)
    cat("Group-level means vs individual posterior means (check mapping):\n")
    for (z in names(mu_names)) {
      mz <- mu_names[[z]]
      if (mz %in% mu_summ$variable && any(is.finite(p[[z]]))) {
        mgroup <- mu_summ$mean[match(mz, mu_summ$variable)]
        mind <- mean(p[[z]], na.rm = TRUE)
        cat(sprintf("  %-7s group mu = %7.3f; mean participant = %7.3f\n",
                    z, mgroup, mind))
      }
    }
  }
  
  cat("Participants: ", nrow(p), "; params per person: ",
      length(indices), "\n", sep = "")
  
  rm(fit)
  list(
    data = p %>% mutate(group = group_name, .before = 1),
    diagnostics = diagnostics
  )
}

results <- purrr::imap(fit_paths, ~extract_participants(.y, .x))
participant_data <- bind_rows(purrr::map(results, "data")) %>%
  mutate(group = factor(group, levels = groups$group))

diagnostic_table <- bind_rows(purrr::map(results, "diagnostics"))
cat("\nPARTICIPANT-LEVEL SAMPLING DIAGNOSTICS:\n")
print(diagnostic_table, n = Inf, width = Inf)

# Remove rho altogether if it is unavailable for all explicit groups.
if (!any(is.finite(participant_data$rho))) {
  plot_order <- setdiff(plot_order, "rho")
  message("No participant-specific rho was found; omitting its panel.")
}

############################################################
# 5. PEARSON CORRELATIONS, SIGNIFICANCE STARS
############################################################

significance_stars <- function(p) {
  if (length(p) != 1L || !is.finite(p)) return("")
  if (p < .001) "***" else if (p < .01) "**" else if (p < .05) "*" else ""
}

pairs_to_test <- as_tibble(
  t(combn(plot_order, 2)), .name_repair = "minimal"
)
names(pairs_to_test) <- c("xvar", "yvar")

one_test <- function(group_name, xvar, yvar) {
  d <- participant_data %>%
    filter(as.character(group) == group_name) %>%
    transmute(x = .data[[xvar]], y = .data[[yvar]]) %>%
    filter(is.finite(x), is.finite(y))
  
  if (nrow(d) < 4L || sd(d$x) == 0 || sd(d$y) == 0) {
    return(tibble(group = group_name, xvar = xvar, yvar = yvar,
                  n = nrow(d), r = NA_real_, p = NA_real_))
  }
  
  t <- suppressWarnings(cor.test(d$x, d$y, method = "pearson"))
  tibble(group = group_name, xvar = xvar, yvar = yvar,
         n = nrow(d), r = unname(t$estimate), p = t$p.value)
}

correlation_results <- purrr::map_dfr(
  seq_len(nrow(pairs_to_test)), function(i) {
    purrr::map_dfr(groups$group, function(g) {
      one_test(g, pairs_to_test$xvar[i], pairs_to_test$yvar[i])
    })
  }
) %>%
  mutate(
    p_BH = p.adjust(p, method = "BH"),
    p_used = if (P_ADJUST == "BH") p_BH else p,
    significant = !is.na(p_used) & p_used < P_CUTOFF,
    stars = vapply(p_used, significance_stars, character(1))
  )

cat("\nSIGNIFICANT PARTICIPANT-LEVEL CORRELATIONS:\n")
print(correlation_results %>% filter(significant), n = Inf, width = Inf)

lookup_test <- function(g, xvar, yvar) {
  i <- correlation_results$group == g & (
    (correlation_results$xvar == xvar & correlation_results$yvar == yvar) |
      (correlation_results$xvar == yvar & correlation_results$yvar == xvar)
  )
  correlation_results[i, , drop = FALSE]
}

correlation_label <- function(g, xvar, yvar) {
  stat <- lookup_test(g, xvar, yvar)
  if (nrow(stat) != 1 || !is.finite(stat$r)) {
    return(paste0(group_short[g], "  --"))
  }
  paste0(group_short[g], "  r = ", sprintf("%.2f", stat$r), stat$stars)
}

############################################################
# 6. PLOTTING HELPERS (MATCH YOUR EXAMPLE)
############################################################

finite_range <- function(x, padding = .04) {
  x <- x[is.finite(x)]
  if (!length(x)) return(c(0, 1))
  lim <- range(x)
  if (diff(lim) == 0) lim <- lim + c(-.5, .5)
  lim + c(-1, 1) * diff(lim) * padding
}

variable_ranges <- setNames(
  lapply(plot_order, function(z) finite_range(participant_data[[z]])),
  plot_order
)

get_group_xy <- function(g, xvar, yvar) {
  participant_data %>%
    filter(as.character(group) == g) %>%
    transmute(x = .data[[xvar]], y = .data[[yvar]]) %>%
    filter(is.finite(x), is.finite(y))
}

draw_group_scatter <- function(xvar, yvar, point_size = .75) {
  for (g in groups$group) {
    d <- get_group_xy(g, xvar, yvar)
    if (!nrow(d)) next
    points(d$x, d$y, pch = 16, cex = point_size,
           col = adjustcolor(group_cols[g], alpha.f = .65))
    
    stat <- lookup_test(g, xvar, yvar)
    if (nrow(stat) == 1 && isTRUE(stat$significant)) {
      regression <- lm(y ~ x, data = d)
      xx <- range(d$x)
      lines(xx,
            predict(regression, newdata = data.frame(x = xx)),
            col = group_cols[g], lwd = 2)
    }
  }
}

draw_density <- function(xvar) {
  dl <- setNames(lapply(groups$group, function(g) {
    x <- participant_data[[xvar]][participant_data$group == g]
    x <- x[is.finite(x)]
    if (length(x) < 2L || sd(x) == 0) return(NULL)
    density(x, n = 256)
  }), groups$group)
  
  ymax <- max(1e-6, vapply(dl, function(d) {
    if (is.null(d)) 0 else max(d$y)
  }, numeric(1)))
  plot(NA, xlim = variable_ranges[[xvar]],
       ylim = c(0, 1.05 * ymax),
       axes = FALSE, xlab = "", ylab = "", type = "n")
  
  for (g in groups$group) {
    d <- dl[[g]]
    if (!is.null(d)) {
      polygon(c(d$x, rev(d$x)), c(d$y, rep(0, length(d$y))),
              col = adjustcolor(group_cols[g], alpha.f = .08),
              border = NA)
      lines(d$x, d$y, col = group_cols[g], lwd = 1.8)
    } else {
      x <- participant_data[[xvar]][participant_data$group == g]
      x <- x[is.finite(x)]
      if (length(x)) abline(v = x[1], col = group_cols[g], lwd = 1.6)
    }
  }
}

draw_correlation_labels <- function(xvar, yvar) {
  plot(NA, xlim = c(0, 1), ylim = c(0, 1),
       axes = FALSE, xlab = "", ylab = "", type = "n")
  y_pos <- seq(.86, .14, length.out = nrow(groups))
  for (i in seq_len(nrow(groups))) {
    g <- groups$group[i]
    text(.04, y_pos[i], correlation_label(g, xvar, yvar),
         adj = 0, col = group_cols[g], cex = .78, font = 2)
  }
}

draw_bottom_legend <- function() {
  par(mar = c(0, 0, 0, 0))
  plot.new()
  legend("center", legend = groups$group,
         col = group_cols, pch = 16, lty = 1, lwd = 2,
         ncol = 5, bty = "n", cex = .83)
}

############################################################
# 7. COMPLETE FIGURE
############################################################

plot_parameter_pairs <- function() {
  if (OPEN_NEW_WINDOW && interactive()) dev.new(width = 16, height = 14)
  op <- par(no.readonly = TRUE)
  on.exit({ layout(1); par(op) }, add = TRUE)
  
  k <- length(plot_order)
  layout(rbind(
    matrix(seq_len(k * k), nrow = k, ncol = k, byrow = TRUE),
    rep(k * k + 1L, k)
  ), heights = c(rep(1, k), .5))
  
  par(oma = c(0, 1, 3, 0))
  for (row in seq_len(k)) {
    for (column in seq_len(k)) {
      xvar <- plot_order[column]
      yvar <- plot_order[row]
      par(mar = c(1.5, 1.6, 1.1, .3), mgp = c(1, .25, 0), tcl = -.15)
      
      if (row == column) {
        draw_density(xvar)
      } else if (row > column) {
        plot(NA, xlim = variable_ranges[[xvar]],
             ylim = variable_ranges[[yvar]],
             axes = FALSE, xlab = "", ylab = "", type = "n")
        draw_group_scatter(xvar, yvar)
      } else {
        draw_correlation_labels(xvar, yvar)
      }
      
      box(col = "grey75", lwd = .6)
      if (row == k) axis(1, cex.axis = .7, las = 1)
      if (column == 1L) axis(2, cex.axis = .7, las = 1)
      if (row == 1L) mtext(plot_labels[[xvar]], side = 3, line = .2, cex = 1)
      if (column == 1L) mtext(plot_labels[[yvar]], side = 2, line = 1.1, cex = 1)
    }
  }
  
  draw_bottom_legend()
  mtext("Learning + 2 channels + instruction: parameter relationships",
        side = 3, outer = TRUE, line = 1.6, cex = 1.2, font = 2)
  mtext(paste0(
    "One dot per participant | Lines: ",
    if (P_ADJUST == "BH") "BH-adjusted" else "uncorrected",
    " p < ", P_CUTOFF, " | * p < .05, ** p < .01, *** p < .001"
  ), side = 3, outer = TRUE, line = .5, cex = .85)
  invisible(NULL)
}

plot_parameter_pairs()

# Useful console objects:
# View(correlation_results)
# View(participant_data)
# View(diagnostic_table)
# Re-display: plot_parameter_pairs()

cat("\nDone. Figure displayed; no output files saved.\n")

