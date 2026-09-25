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

















rm(list = ls(all = TRUE))

setwd("/Users/bty615/Documents/GitHub/reliable_info_bias")

library(tidyverse)
library(posterior)

##################################################
## PATHS
##################################################

fits_dir <- "results/fits/Exp12"
fig_dir  <- "results/figures/boost_model_corrplots"

if (!dir.exists(fig_dir)) {
  dir.create(
    fig_dir,
    recursive = TRUE
  )
}

##################################################
## HELPER:
## LOAD FIT OBJECT
##################################################

load_fit <- function(file_path) {
  
  if (!file.exists(file_path)) {
    stop(
      "\nMissing fit file:\n",
      file_path,
      "\n"
    )
  }
  
  env <- new.env()
  
  loaded_objects <- load(
    file_path,
    envir = env
  )
  
  if ("fit" %in% loaded_objects) {
    return(env$fit)
  }
  
  return(env[[loaded_objects[1]]])
}

##################################################
## LOAD ALL FIVE BOOST MODEL FITS
##################################################

fit_unaware <- load_fit(
  file.path(
    fits_dir,
    "fit_trunc_boost_model_unaware_exp11.rdata"
  )
)

fit_aware <- load_fit(
  file.path(
    fits_dir,
    "fit_trunc_boost_model_aware_exp11.rdata"
  )
)

fit_explicit <- load_fit(
  file.path(
    fits_dir,
    "fit_trunc_boost_model_aware_exp12.rdata"
  )
)

fit_truthful <- load_fit(
  file.path(
    fits_dir,
    "fit_trunc_boost_truthful_exp13.rdata"
  )
)

fit_deceptive <- load_fit(
  file.path(
    fits_dir,
    "fit_trunc_boost_deceptive_exp13.rdata"
  )
)

##################################################
## FIT LIST
##################################################

fit_list <- list(
  "Implicit Unaware"    = fit_unaware,
  "Implicit Aware"      = fit_aware,
  "Explicit Undirected" = fit_explicit,
  "Explicit Truthful"   = fit_truthful,
  "Explicit Deceptive"  = fit_deceptive
)

##################################################
## GROUP COLOURS
##################################################

group_cols <- c(
  "Implicit Unaware"    = "#F4A6A6",  # light red
  "Implicit Aware"      = "#81C784",  # light green
  "Explicit Undirected" = "#E69F00",  # orange
  "Explicit Truthful"   = "#1B5E20",  # dark green
  "Explicit Deceptive"  = "#8B1A1A"   # dark red
)

##################################################
## SHORT LABELS
##################################################

group_short <- c(
  "Implicit Unaware"    = "IU",
  "Implicit Aware"      = "IA",
  "Explicit Undirected" = "EU",
  "Explicit Truthful"   = "ET",
  "Explicit Deceptive"  = "ED"
)

##################################################
## EXTRACT GROUP-LEVEL POSTERIOR PARAMETERS
##################################################

extract_pars <- function(
    fit_obj,
    group_name
) {
  
  draws <- as_draws_df(
    fit_obj$draws()
  )
  
  transformed_names <- c(
    "mu_alpha",
    "mu_beta",
    "mu_lambda",
    "mu_delta",
    "mu_eta"
  )
  
  ################################################
  ## CASE 1:
  ## TRANSFORMED PARAMETERS EXIST
  ################################################
  
  if (all(transformed_names %in% colnames(draws))) {
    
    pars <- draws %>%
      select(
        mu_alpha,
        mu_beta,
        mu_delta,
        mu_lambda,
        mu_eta
      )
    
  } else {
    
    ################################################
    ## CASE 2:
    ## RAW mu_pr PARAMETERS
    ################################################
    
    mu_cols_bracket <- paste0(
      "mu_pr[",
      1:5,
      "]"
    )
    
    mu_cols_dot <- paste0(
      "mu_pr.",
      1:5,
      "."
    )
    
    if (all(mu_cols_bracket %in% colnames(draws))) {
      
      mu_cols <- mu_cols_bracket
      
    } else if (all(mu_cols_dot %in% colnames(draws))) {
      
      mu_cols <- mu_cols_dot
      
    } else {
      
      mu_cols <- colnames(draws)[
        grepl(
          "^mu_pr(\\[|\\.)",
          colnames(draws)
        )
      ]
      
      if (length(mu_cols) < 5) {
        stop(
          "\nCould not find all five group-level mu parameters for:\n",
          group_name,
          "\n"
        )
      }
      
      get_index <- function(x) {
        as.numeric(
          gsub(
            "[^0-9]",
            "",
            x
          )
        )
      }
      
      mu_cols <- mu_cols[
        order(
          sapply(
            mu_cols,
            get_index
          )
        )
      ]
      
      mu_cols <- mu_cols[1:5]
    }
    
    pars <- draws %>%
      select(
        all_of(mu_cols)
      )
    
    ################################################
    ## PARAMETER ORDER
    ##
    ## 1 = alpha
    ## 2 = beta
    ## 3 = lambda
    ## 4 = delta
    ## 5 = eta
    ################################################
    
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
        mu_beta   = mu_beta,
        mu_lambda = pnorm(mu_lambda),
        mu_delta  = pnorm(mu_delta) * 2,
        mu_eta    = mu_eta
      ) %>%
      select(
        mu_alpha,
        mu_beta,
        mu_delta,
        mu_lambda,
        mu_eta
      )
  }
  
  pars %>%
    as_tibble() %>%
    mutate(
      Group = group_name
    )
}

##################################################
## EXTRACT ALL FIVE GROUPS
##################################################

all_pars <- bind_rows(
  lapply(
    names(fit_list),
    function(g) {
      extract_pars(
        fit_obj = fit_list[[g]],
        group_name = g
      )
    }
  )
)

all_pars$Group <- factor(
  all_pars$Group,
  levels = c(
    "Implicit Unaware",
    "Implicit Aware",
    "Explicit Undirected",
    "Explicit Truthful",
    "Explicit Deceptive"
  )
)

##################################################
## CHECK
##################################################

cat(
  "\n============================================\n"
)
cat("BOOST MODEL — POSTERIOR CORRELATIONS\n")
cat("============================================\n\n")
cat("Posterior draws per group:\n\n")
print(table(all_pars$Group))

##################################################
## PARAMETERS
##################################################

parameter_names <- c(
  "mu_alpha",
  "mu_beta",
  "mu_delta",
  "mu_lambda",
  "mu_eta"
)

parameter_labels <- c(
  expression(mu[alpha]),
  expression(mu[beta]),
  expression(mu[delta]),
  expression(mu[lambda]),
  expression(mu[eta])
)

##################################################
## PARAMETER RANGES
##################################################

param_ranges <- lapply(
  parameter_names,
  function(p) {
    range(
      all_pars[[p]],
      finite = TRUE,
      na.rm = TRUE
    )
  }
)

names(param_ranges) <- parameter_names

##################################################
## PANEL:
## SCATTERPLOT
##################################################

draw_scatter_panel <- function(
    x,
    y,
    x_name,
    y_name,
    show_x_axis = FALSE,
    show_y_axis = FALSE
) {
  
  plot(
    NA,
    xlim = param_ranges[[x_name]],
    ylim = param_ranges[[y_name]],
    xlab = "",
    ylab = "",
    axes = FALSE,
    type = "n"
  )
  
  box(
    col = "grey70",
    lwd = 0.8
  )
  
  for (g in levels(all_pars$Group)) {
    
    idx <- (
      all_pars$Group == g &
        is.finite(x) &
        is.finite(y)
    )
    
    points(
      x[idx],
      y[idx],
      pch = 16,
      cex = 0.56,
      col = adjustcolor(
        group_cols[g],
        alpha.f = 0.25
      )
    )
  }
  
  if (show_x_axis) {
    axis(
      1,
      cex.axis = 0.92,
      las = 1
    )
  }
  
  if (show_y_axis) {
    axis(
      2,
      cex.axis = 0.92,
      las = 1
    )
  }
}

##################################################
## PANEL:
## DENSITY
##################################################

draw_density_panel <- function(
    x,
    parameter_name,
    show_x_axis = FALSE,
    show_y_axis = FALSE
) {
  
  levs <- levels(all_pars$Group)
  
  dens_list <- lapply(
    levs,
    function(g) {
      xx <- x[
        all_pars$Group == g &
          is.finite(x)
      ]
      density(
        xx,
        n = 512,
        na.rm = TRUE
      )
    }
  )
  
  names(dens_list) <- levs
  
  y_max <- max(
    sapply(
      dens_list,
      function(d) max(d$y)
    )
  )
  
  plot(
    NA,
    xlim = param_ranges[[parameter_name]],
    ylim = c(0, y_max * 1.08),
    xlab = "",
    ylab = "",
    axes = FALSE,
    type = "n"
  )
  
  box(
    col = "grey70",
    lwd = 0.8
  )
  
  for (g in levs) {
    
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
      d$x,
      d$y,
      col = group_cols[g],
      lwd = 2.6
    )
  }
  
  if (show_x_axis) {
    axis(
      1,
      cex.axis = 0.90,
      las = 1
    )
  }
  
  if (show_y_axis) {
    axis(
      2,
      cex.axis = 0.90,
      las = 1
    )
  }
}

##################################################
## PANEL:
## CORRELATION TEXT
##################################################

draw_correlation_panel <- function(
    x,
    y
) {
  
  plot(
    NA,
    xlim = c(0, 1),
    ylim = c(0, 1),
    xlab = "",
    ylab = "",
    axes = FALSE,
    type = "n"
  )
  
  box(
    col = "grey75",
    lwd = 0.7
  )
  
  y_positions <- c(
    0.84,
    0.67,
    0.50,
    0.33,
    0.16
  )
  
  levs <- levels(all_pars$Group)
  
  for (i in seq_along(levs)) {
    
    g <- levs[i]
    
    idx <- (
      all_pars$Group == g &
        is.finite(x) &
        is.finite(y)
    )
    
    if (sum(idx) > 2) {
      
      r_value <- cor(
        x[idx],
        y[idx],
        method = "pearson"
      )
      
      this_label <- paste0(
        group_short[g],
        "   r = ",
        sprintf("%.2f", r_value)
      )
      
    } else {
      
      this_label <- paste0(
        group_short[g],
        "   r = NA"
      )
    }
    
    text(
      x = 0.07,
      y = y_positions[i],
      labels = this_label,
      adj = c(0, 0.5),
      col = group_cols[g],
      cex = 1.04,
      font = 2
    )
  }
}

##################################################
## PANEL:
## LEGEND ONLY
##################################################

draw_bottom_legend <- function() {
  
  plot(
    NA,
    xlim = c(0, 1),
    ylim = c(0, 1),
    axes = FALSE,
    xlab = "",
    ylab = "",
    type = "n"
  )
  
  x_positions <- c(
    0.07,
    0.28,
    0.49,
    0.70,
    0.88
  )
  
  y0 <- 0.52
  
  text(
    x = 0.5,
    y = 0.90,
    labels = "Experimental group",
    cex = 1.18,
    font = 2
  )
  
  for (i in seq_along(group_cols)) {
    
    g <- names(group_cols)[i]
    x <- x_positions[i]
    
    segments(
      x0 = x - 0.045,
      x1 = x - 0.005,
      y0 = y0,
      y1 = y0,
      col = group_cols[g],
      lwd = 3.5
    )
    
    points(
      x = x - 0.025,
      y = y0,
      pch = 16,
      cex = 1.2,
      col = group_cols[g]
    )
    
    text(
      x = x + 0.015,
      y = y0,
      labels = g,
      adj = c(0, 0.5),
      cex = 0.95,
      font = 2
    )
  }
}

##################################################
## COMPLETE FIGURE
##################################################

draw_complete_figure <- function() {
  
  ################################################
  ## 5 x 5 MATRIX
  ## + 1 FULL-WIDTH LEGEND ROW UNDERNEATH
  ################################################
  
  layout_matrix <- rbind(
    matrix(
      1:25,
      nrow = 5,
      ncol = 5,
      byrow = TRUE
    ),
    rep(26, 5)
  )
  
  layout(
    layout_matrix,
    widths = rep(1.22, 5),
    heights = c(1, 1, 1, 1, 1, 0.42)
  )
  
  ################################################
  ## MATRIX PANELS
  ################################################
  
  for (row in 1:5) {
    
    for (col in 1:5) {
      
      par(
        mar = c(
          1.8,
          1.8,
          1.2,
          0.5
        ),
        mgp = c(
          1.35,
          0.35,
          0
        ),
        tcl = -0.20
      )
      
      x_name <- parameter_names[col]
      y_name <- parameter_names[row]
      
      x <- all_pars[[x_name]]
      y <- all_pars[[y_name]]
      
      if (row == col) {
        
        draw_density_panel(
          x = x,
          parameter_name = x_name,
          show_x_axis = (row == 5),
          show_y_axis = (col == 1)
        )
        
      } else if (row > col) {
        
        draw_scatter_panel(
          x = x,
          y = y,
          x_name = x_name,
          y_name = y_name,
          show_x_axis = (row == 5),
          show_y_axis = (col == 1)
        )
        
      } else {
        
        draw_correlation_panel(
          x = x,
          y = y
        )
      }
      
      ################################################
      ## TOP LABELS
      ################################################
      
      if (row == 1) {
        mtext(
          parameter_labels[col],
          side = 3,
          line = 0.22,
          cex = 1.42,
          font = 2
        )
      }
      
      ################################################
      ## LEFT LABELS
      ################################################
      
      if (col == 1) {
        mtext(
          parameter_labels[row],
          side = 2,
          line = 0.90,
          cex = 1.42,
          font = 2
        )
      }
    }
  }
  
  ################################################
  ## BOTTOM LEGEND
  ################################################
  
  par(
    mar = c(
      0.3,
      0.8,
      0.5,
      0.8
    )
  )
  
  draw_bottom_legend()
}

##################################################
## OUTPUT FILE
##################################################

plot_file <- file.path(
  fig_dir,
  "corrplot_BOOST_MODEL_5_GROUPS_no_overlap.png"
)

##################################################
## SAVE FIGURE
##################################################

png(
  filename = plot_file,
  width = 5000,
  height = 3800,
  res = 300
)

par(
  oma = c(1, 1, 3, 1)
)

draw_complete_figure()

mtext(
  "Posterior correlations between group-level parameters",
  side = 3,
  outer = TRUE,
  line = 1.0,
  cex = 1.75,
  font = 2
)

dev.off()

##################################################
## SHOW IN RSTUDIO
##################################################

if (interactive()) {
  
  dev.new(
    width = 16,
    height = 12
  )
  
  par(
    oma = c(1, 1, 3, 1)
  )
  
  draw_complete_figure()
  
  mtext(
    "Posterior correlations between group-level parameters",
    side = 3,
    outer = TRUE,
    line = 1.0,
    cex = 1.75,
    font = 2
  )
}

##################################################
## DONE
##################################################

cat(
  "\n============================================\n"
)
cat("DONE\n")
cat("============================================\n\n")
cat("Plot saved to:\n\n")
cat(
  normalizePath(plot_file),
  "\n\n"
)