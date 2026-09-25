
# BEHAVIOURAL POSTERIOR PREDICTIVE CHECKS
#
# Models: Learning, Local eta, Learning + local eta,
# 2 channels, Local eta + 2 channels
#
#   1. Accuracy
#   2. Overall PSE
#   3. PSE trajectory
#   4. Reliability-specific evidence integration



rm(list = ls(all.names = TRUE))

library(cmdstanr)
library(posterior)
library(tidyverse)


# ---------------------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------------------

project_root <- path.expand(
  "~/Documents/GitHub/reliable_info_bias"
)

fit_root <- file.path(
  project_root,
  "stan/results/fits/exp11_unaware"
)

MIN_TRIALS_PER_BIN <- 3
ROLLING_WINDOW <- 50
N_PPC_DRAWS <- 200
N_TRAJECTORY_DRAWS <- 20  # Rolling-window fits are computationally expensive.
TRAJECTORY_STEP <- 10      # Model curves at every tenth trial; observed at every trial.


# ---------------------------------------------------------------------
# 2. GROUP INFORMATION
# ---------------------------------------------------------------------

groups <- tribble(
  ~group, ~safe, ~data_file, ~order_col, ~prior_col, ~fit_suffix,
  
  "Implicit Unaware",
  "implicit_unaware",
  "data/DATA_Unaware_Exp11.csv",
  "ResponseButtonOrder",
  "Prior_Belief",
  "unaware_exp11",
  
  "Implicit Aware",
  "implicit_aware",
  "data/DATA_Aware_Exp11.csv",
  "ResponseButtonOrder",
  "Prior_Belief",
  "aware_exp11",
  
  "Explicit Undirected",
  "explicit_undirected",
  "data/DATA_Aware_Exp12.csv",
  "Manipulation_ResponseButtonOrder",
  "Prior_Belief",
  "aware_exp12",
  
  "Explicit True",
  "explicit_true",
  "data/data_priorbelief_truthful_exp13.csv",
  "ResponseButtonOrder",
  "TruePrior",
  "truthful_exp13",
  
  "Explicit Deceptive",
  "explicit_deceptive",
  "data/data_priorbelief_deceptive_exp13.csv",
  "ResponseButtonOrder",
  "TruePrior",
  "deceptive_exp13"
) %>%
  mutate(
    data_file = file.path(
      project_root,
      data_file
    )
  )


group_levels <- groups$group


# ---------------------------------------------------------------------
# 3. MODEL INFORMATION
# ---------------------------------------------------------------------

models <- tribble(
  ~model, ~file_prefix, ~model_colour,
  "Learning", "learning", "#CC79A7",
  "Local eta", "localeta", "#0072B2",
  "Learning + local eta", "learning_localeta", "#D55E00",
  "2 channels", "2channels", "#E69F00",
  "Local eta + 2 channels", "leta_2channels", "#009E73"
)

model_levels <- models$model

model_cols <- setNames(
  models$model_colour,
  models$model
)


# Reliability is shown by line type.
reliability_linetypes <- c(
  "50%" = "dotted",
  "55%" = "dashed",
  "65%" = "solid"
)


# ---------------------------------------------------------------------
# 4. NORMALISE PRIOR CODING
# ---------------------------------------------------------------------

normalise_prior <- function(x) {
  
  z <- tolower(
    as.character(x)
  )
  
  case_when(
    
    z %in% c(
      "1",
      "blue",
      "b",
      "65_blue",
      "blueprior"
    ) ~ 1L,
    
    z %in% c(
      "2",
      "red",
      "r",
      "65_red",
      "redprior"
    ) ~ 2L,
    
    str_detect(
      z,
      "blue"
    ) ~ 1L,
    
    str_detect(
      z,
      "red"
    ) ~ 2L,
    
    TRUE ~ suppressWarnings(
      as.integer(z)
    )
  )
}


# ---------------------------------------------------------------------
# 5. EXTRACT SAMPLE COLOURS AND RELIABILITIES
# ---------------------------------------------------------------------

extract_samples <- function(d) {
  
  n <- nrow(d)
  
  cols <- matrix(
    NA_character_,
    n,
    6
  )
  
  rels <- matrix(
    NA_real_,
    n,
    6
  )
  
  
  # Newer files:
  # color_1 ... color_6
  # proba_1 ... proba_6
  
  if (
    all(
      paste0(
        "color_",
        1:6
      ) %in% names(d)
    ) &&
    all(
      paste0(
        "proba_",
        1:6
      ) %in% names(d)
    )
  ) {
    
    cols <- as.matrix(
      d[
        paste0(
          "color_",
          1:6
        )
      ]
    )
    
    rels <- apply(
      d[
        paste0(
          "proba_",
          1:6
        )
      ],
      2,
      as.numeric
    )
    
    
    # Some Exp13 files contain
    # 50 / 55 / 65 instead of
    # .50 / .55 / .65
    
    rels[
      rels > 1
    ] <- rels[
      rels > 1
    ] / 100
    
    
  } else {
    
    # Older files:
    # Sample_Color
    # Sample_Reliability
    
    for (i in seq_len(n)) {
      
      cs <- str_extract_all(
        tolower(
          as.character(
            d$Sample_Color[i]
          )
        ),
        "blue|red"
      )[[1]]
      
      
      rs <- suppressWarnings(
        as.numeric(
          str_extract_all(
            as.character(
              d$Sample_Reliability[i]
            ),
            "[0-9]+(?:\\.[0-9]+)?"
          )[[1]]
        )
      )
      
      
      rs[
        rs > 1
      ] <- rs[
        rs > 1
      ] / 100
      
      
      if (length(cs)) {
        
        cols[
          i,
          seq_len(
            min(
              6,
              length(cs)
            )
          )
        ] <- cs[
          seq_len(
            min(
              6,
              length(cs)
            )
          )
        ]
      }
      
      
      if (length(rs)) {
        
        rels[
          i,
          seq_len(
            min(
              6,
              length(rs)
            )
          )
        ] <- rs[
          seq_len(
            min(
              6,
              length(rs)
            )
          )
        ]
      }
    }
  }
  
  
  list(
    colour = tolower(cols),
    reliability = rels
  )
}


# ---------------------------------------------------------------------
# 6. ADD EVIDENCE VARIABLES
# ---------------------------------------------------------------------

add_evidence <- function(d) {
  
  s <- extract_samples(d)
  
  
  # Neutral Bayesian probability
  # that Blue is correct.
  #
  # This uses a 0.5 / 0.5 prior.
  
  rb <- ifelse(
    s$colour == "blue",
    s$reliability,
    1 - s$reliability
  )
  
  
  rr <- ifelse(
    s$colour == "red",
    s$reliability,
    1 - s$reliability
  )
  
  
  rb[
    is.na(rb)
  ] <- 0.5
  
  rr[
    is.na(rr)
  ] <- 0.5
  
  
  rb <- pmin(
    pmax(
      rb,
      1e-8
    ),
    1 - 1e-8
  )
  
  
  rr <- pmin(
    pmax(
      rr,
      1e-8
    ),
    1 - 1e-8
  )
  
  
  d$neutral_blue_probability <- plogis(
    rowSums(
      log(
        rb / rr
      )
    )
  )
  
  
  # Net number of samples favouring Blue
  # at each reliability level.
  
  for (r in c(
    .50,
    .55,
    .65
  )) {
    
    keep <- abs(
      s$reliability - r
    ) < 1e-6
    
    
    blue <- rowSums(
      keep &
        s$colour == "blue",
      na.rm = TRUE
    )
    
    
    red <- rowSums(
      keep &
        s$colour == "red",
      na.rm = TRUE
    )
    
    
    d[[
      paste0(
        "x",
        as.integer(
          r * 100
        )
      )
    ]] <- blue - red
  }
  
  
  d
}


# ---------------------------------------------------------------------
# 7. PREPARE BEHAVIOURAL DATA
# ---------------------------------------------------------------------

prepare_group <- function(info) {
  
  d <- read_csv(
    info$data_file,
    show_col_types = FALSE,
    progress = FALSE
  )
  
  
  id_order <- unique(
    d$ParticipantPrivateID
  )
  
  
  d <- d %>%
    mutate(
      
      subject_index = match(
        ParticipantPrivateID,
        id_order
      ),
      
      trial = as.integer(
        TrialNumber
      ),
      
      order = as.integer(
        .data[[
          info$order_col
        ]]
      ),
      
      
      # Correct colour mapping:
      #
      # RBO = 1:
      #   Response 1 = Blue
      #   Response 0 = Red
      #
      # RBO = 0:
      #   Response 0 = Blue
      #   Response 1 = Red
      
      observed_choice = if_else(
        as.integer(Response) == order,
        1L,
        2L
      ),
      
      
      prior = normalise_prior(
        .data[[
          info$prior_col
        ]]
      )
    ) %>%
    
    arrange(
      subject_index,
      trial
    )
  
  
  d <- add_evidence(d)
  
  
  d <- d %>%
    mutate(
      
      # Did the participant choose
      # the true base-rate colour?
      
      observed_prior_choice = as.integer(
        observed_choice == prior
      ),
      
      # CorrectResponse uses the same response-button coding as Response.
      correct_colour = if_else(
        as.integer(CorrectResponse) == order,
        1L,
        2L
      ),
      
      # Evidence probability aligned to the true base-rate colour.
      evidence_prior = if_else(
        prior == 1L,
        neutral_blue_probability,
        1 - neutral_blue_probability
      ),
      
      
      # Align evidence with the
      # true base-rate colour.
      
      x50_prior = if_else(
        prior == 1,
        x50,
        -x50
      ),
      
      x55_prior = if_else(
        prior == 1,
        x55,
        -x55
      ),
      
      x65_prior = if_else(
        prior == 1,
        x65,
        -x65
      )
    )
  
  
  list(
    info = info,
    data = d
  )
}


behavioural_objects <- lapply(
  seq_len(
    nrow(groups)
  ),
  function(i) {
    
    prepare_group(
      groups[i, ]
    )
  }
)


# ---------------------------------------------------------------------
# 8. LOAD y_pred FROM ONE MODEL
# ---------------------------------------------------------------------

load_model_predictions <- function(
    behavioural_object,
    model_info,
    n_requested_draws = 200
) {
  
  info <- behavioural_object$info
  d <- behavioural_object$data
  
  
  fit_file <- file.path(
    fit_root,
    paste0(
      model_info$file_prefix,
      "_",
      info$fit_suffix,
      ".rdata"
    )
  )
  
  
  if (!file.exists(fit_file)) {
    
    stop(
      paste0(
        "\nCould not find model file:\n",
        fit_file
      )
    )
  }
  
  
  cat(
    "\nLoading:\n",
    fit_file,
    "\n"
  )
  
  
  fit_environment <- new.env(
    parent = emptyenv()
  )
  
  
  loaded_objects <- load(
    fit_file,
    envir = fit_environment
  )
  
  
  if (!"fit" %in% loaded_objects) {
    
    stop(
      paste0(
        "\nNo object named 'fit' in:\n",
        fit_file
      )
    )
  }
  
  
  current_fit <- fit_environment$fit
  
  
  if (!inherits(
    current_fit,
    "CmdStanMCMC"
  )) {
    
    stop(
      paste0(
        "\nThe fit in this file is not a CmdStanMCMC object:\n",
        fit_file
      )
    )
  }
  
  
  # -------------------------------------------------
  # Extract posterior predictive responses
  # -------------------------------------------------
  
  y_pred <- current_fit$draws(
    variables = "y_pred",
    format = "draws_matrix"
  )
  
  
  if (
    !all(
      unique(
        as.vector(y_pred)
      ) %in% c(
        1,
        2
      )
    )
  ) {
    
    stop(
      paste0(
        "\ny_pred contains values other than 1 and 2 in:\n",
        fit_file
      )
    )
  }
  
  
  # -------------------------------------------------
  # Match y_pred columns exactly to behavioural rows
  # -------------------------------------------------
  
  expected_names <- paste0(
    "y_pred[",
    d$subject_index,
    ",",
    d$trial,
    "]"
  )
  
  
  column_match <- match(
    expected_names,
    colnames(y_pred)
  )
  
  
  if (anyNA(column_match)) {
    
    first_missing <- expected_names[
      which(
        is.na(column_match)
      )[1]
    ]
    
    
    stop(
      paste0(
        "\nCould not match all predictions for:\n",
        model_info$model,
        " / ",
        info$group,
        "\n\nFirst missing variable:\n",
        first_missing
      )
    )
  }
  
  
  y_pred <- y_pred[
    ,
    column_match,
    drop = FALSE
  ]
  
  
  # -------------------------------------------------
  # Keep evenly spaced posterior draws
  # -------------------------------------------------
  
  n_available_draws <- nrow(
    y_pred
  )
  
  
  n_selected_draws <- min(
    n_requested_draws,
    n_available_draws
  )
  
  
  selected_draw_rows <- unique(
    as.integer(
      round(
        seq(
          from = 1,
          to = n_available_draws,
          length.out = n_selected_draws
        )
      )
    )
  )
  
  
  y_pred <- y_pred[
    selected_draw_rows,
    ,
    drop = FALSE
  ]
  
  
  list(
    group = info$group,
    model = model_info$model,
    data = d,
    pred = y_pred,
    fit_file = fit_file
  )
}


# ---------------------------------------------------------------------
# 9. LOAD AVAILABLE MODELS FOR ALL FIVE GROUPS
# ---------------------------------------------------------------------

model_objects <- list()

counter <- 1


for (g in seq_len(
  length(
    behavioural_objects
  )
)) {
  
  for (m in seq_len(
    nrow(models)
  )) {
    
    # No standalone Learning fit is present for Exp13.
    if (models$model[m] == "Learning" && g >= 4) next
    
    model_objects[[counter]] <-
      load_model_predictions(
        behavioural_object =
          behavioural_objects[[g]],
        
        model_info =
          models[m, ],
        
        n_requested_draws =
          N_PPC_DRAWS
      )
    
    
    counter <- counter + 1
  }
}


cat(
  "\n============================================================\n"
)

cat(
  "ALL MODEL PREDICTIONS LOADED\n"
)

cat(
  "============================================================\n"
)


# =====================================================================
# 10. RELIABILITY-SPECIFIC EVIDENCE INTEGRATION
# =====================================================================


evidence_one <- function(
    d,
    choice,
    group_name,
    model_name = NA_character_,
    draw = NA_integer_
) {
  
  map_dfr(
    c(
      "50",
      "55",
      "65"
    ),
    function(rr) {
      
      xcol <- paste0(
        "x",
        rr,
        "_prior"
      )
      
      
      tibble(
        
        subject =
          d$subject_index,
        
        x =
          d[[xcol]],
        
        y =
          as.integer(
            choice == d$prior
          )
      ) %>%
        
        group_by(
          subject,
          x
        ) %>%
        
        summarise(
          
          n = n(),
          
          p = mean(
            y
          ),
          
          .groups = "drop"
        ) %>%
        
        filter(
          n >= MIN_TRIALS_PER_BIN
        ) %>%
        
        group_by(
          x
        ) %>%
        
        summarise(
          
          value = mean(
            p,
            na.rm = TRUE
          ),
          
          se = sd(
            p,
            na.rm = TRUE
          ) /
            sqrt(
              sum(
                is.finite(p)
              )
            ),
          
          .groups = "drop"
        ) %>%
        
        mutate(
          reliability = paste0(
            rr,
            "%"
          )
        )
    }
  ) %>%
    
    mutate(
      group = group_name,
      model = model_name,
      draw = draw
    )
}


# ---------------------------------------------------------------------
# Behavioural evidence curves
# ---------------------------------------------------------------------

observed_ev <- map_dfr(
  behavioural_objects,
  function(o) {
    
    evidence_one(
      d = o$data,
      choice = o$data$observed_choice,
      group_name = o$info$group
    )
  }
)


# ---------------------------------------------------------------------
# Model evidence curves
# ---------------------------------------------------------------------

pred_ev <- map_dfr(
  model_objects,
  function(o) {
    
    map_dfr(
      seq_len(
        nrow(
          o$pred
        )
      ),
      function(k) {
        
        evidence_one(
          d = o$data,
          choice = o$pred[k, ],
          group_name = o$group,
          model_name = o$model,
          draw = k
        )
      }
    )
  }
)


# ---------------------------------------------------------------------
# Posterior predictive intervals
# ---------------------------------------------------------------------

pred_ev_band <- pred_ev %>%
  
  group_by(
    group,
    model,
    reliability,
    x
  ) %>%
  
  summarise(
    
    mean = mean(
      value,
      na.rm = TRUE
    ),
    
    lo = quantile(
      value,
      .025,
      na.rm = TRUE
    ),
    
    hi = quantile(
      value,
      .975,
      na.rm = TRUE
    ),
    
    .groups = "drop"
  )


# ---------------------------------------------------------------------
# Plotting labels
# ---------------------------------------------------------------------

pred_ev_band <- pred_ev_band %>%
  
  mutate(
    
    plot_row = factor(
      "Model predictions",
      levels = c(
        "Model predictions",
        "Behavioural data"
      )
    ),
    
    group_panel = factor(
      group,
      levels = group_levels
    ),
    
    model = factor(
      model,
      levels = model_levels
    ),
    
    reliability = factor(
      reliability,
      levels = c(
        "50%",
        "55%",
        "65%"
      )
    )
  )


observed_ev <- observed_ev %>%
  
  mutate(
    
    plot_row = factor(
      "Behavioural data",
      levels = c(
        "Model predictions",
        "Behavioural data"
      )
    ),
    
    group_panel = factor(
      group,
      levels = group_levels
    ),
    
    reliability = factor(
      reliability,
      levels = c(
        "50%",
        "55%",
        "65%"
      )
    )
  )


# ---------------------------------------------------------------------
# PLOT 2: EVIDENCE INTEGRATION
# ---------------------------------------------------------------------

p_evidence <- ggplot() +
  
  # MODEL PREDICTIONS ---------------------------------------------

geom_ribbon(
  data = pred_ev_band,
  aes(
    x = x,
    ymin = lo,
    ymax = hi,
    fill = model,
    group = interaction(
      model,
      reliability
    )
  ),
  alpha = .07
) +
  
  geom_line(
    data = pred_ev_band,
    aes(
      x = x,
      y = mean,
      colour = model,
      linetype = reliability,
      group = interaction(
        model,
        reliability
      )
    ),
    linewidth = 1
  ) +
  
  
  # BEHAVIOURAL DATA ---------------------------------------------

geom_errorbar(
  data = observed_ev,
  aes(
    x = x,
    ymin = value - se,
    ymax = value + se,
    group = reliability
  ),
  colour = "black",
  width = .08,
  linewidth = .5
) +
  
  geom_line(
    data = observed_ev,
    aes(
      x = x,
      y = value,
      linetype = reliability,
      group = reliability
    ),
    colour = "black",
    linewidth = 1
  ) +
  
  geom_point(
    data = observed_ev,
    aes(
      x = x,
      y = value,
      group = reliability
    ),
    colour = "black",
    fill = "white",
    shape = 21,
    size = 2.3
  ) +
  
  
  # REFERENCE LINES ----------------------------------------------

geom_vline(
  xintercept = 0,
  linetype = "dashed",
  colour = "grey55"
) +
  
  geom_hline(
    yintercept = .5,
    linetype = "dashed",
    colour = "grey55"
  ) +
  
  
  # FACETS -------------------------------------------------------

facet_grid(
  rows = vars(
    plot_row
  ),
  cols = vars(
    group_panel
  )
) +
  
  
  # SCALES -------------------------------------------------------

scale_colour_manual(
  values = model_cols
) +
  
  scale_fill_manual(
    values = model_cols
  ) +
  
  scale_linetype_manual(
    values = reliability_linetypes
  ) +
  
  coord_cartesian(
    ylim = c(
      0,
      1
    )
  ) +
  
  
  # LABELS -------------------------------------------------------

labs(
  x = "Net samples favouring true base-rate colour",
  y = "Proportion choosing true base-rate colour",
  colour = "Model",
  fill = "Model",
  linetype = "Reliability",
  title = "Reliability-specific evidence integration",
  subtitle = paste0(
    "Top: predictions from the available models. ",
    "Model identity is shown by colour and reliability by line type. ",
    "Bottom: observed behavioural data with ±1 SE."
  )
) +
  
  
  # THEME --------------------------------------------------------

theme_classic(
  base_size = 11
) +
  
  theme(
    legend.position = "bottom",
    strip.text = element_text(
      face = "bold"
    ),
    strip.text.y = element_text(
      angle = 90
    )
  )



# =====================================================================
# 11. ACCURACY, OVERALL PSE, AND ROLLING PSE
# =====================================================================

# Fit raw Blue-choice PSE against neutral Blue evidence for each participant.
# This matches the behavioural plot: Blue/Red prior split, 50-trial rolling
# windows, one overall participant PSE, and participant-level accuracy.
pse_one <- function(d, choice, start = 1L, end = 240L) {
  keep <- !is.na(d$trial) & d$trial >= start & d$trial <= end &
    is.finite(d$neutral_blue_probability) & !is.na(choice)
  if (sum(keep) < 12L) return(NA_real_)
  x <- qlogis(pmin(pmax(d$neutral_blue_probability[keep], 1e-6), 1 - 1e-6))
  y <- as.integer(choice[keep] == 1L)
  if (length(unique(x)) < 2L || length(unique(y)) < 2L) return(NA_real_)
  fit <- suppressWarnings(tryCatch(glm(y ~ x, family = binomial()),
                                   error = function(e) NULL))
  if (is.null(fit) || any(!is.finite(coef(fit))) || coef(fit)[2] <= 0)
    return(NA_real_)
  pmin(pmax(plogis(-coef(fit)[1] / coef(fit)[2]), 0), 1)
}

subject_summary <- function(d, choice, start = 1L, end = 240L) {
  ids <- unique(d$subject_index)
  tibble(subject_index = ids,
         prior = vapply(ids, function(id) {
           pp <- unique(na.omit(d$prior[d$subject_index == id]))
           if (length(pp) == 1L) pp else NA_integer_
         }, integer(1)),
         value = vapply(ids, function(id) {
           idx <- d$subject_index == id
           pse_one(d[idx, ], choice[idx], start, end)
         }, numeric(1)))
}

mean_se <- function(x) {
  x <- x[is.finite(x)]
  tibble(mean = if (length(x)) mean(x) else NA_real_,
         se = if (length(x) > 1L) sd(x) / sqrt(length(x)) else NA_real_,
         n = length(x))
}

predictive_band <- function(d, keys) {
  d %>% group_by(across(all_of(keys))) %>%
    summarise(mean = if (any(is.finite(value))) mean(value, na.rm = TRUE) else NA_real_,
              lo = if (any(is.finite(value))) quantile(value, .025, na.rm = TRUE) else NA_real_,
              hi = if (any(is.finite(value))) quantile(value, .975, na.rm = TRUE) else NA_real_,
              .groups = "drop")
}

observed_accuracy <- map_dfr(behavioural_objects, function(o) {
  o$data %>% filter(trial <= 240) %>% group_by(subject_index) %>%
    summarise(value = mean(observed_choice == correct_colour, na.rm = TRUE),
              .groups = "drop") %>%
    summarise(mean_se(value)) %>% mutate(group = o$info$group)
})
pred_accuracy <- map_dfr(model_objects, function(o) {
  keep <- !is.na(o$data$trial) & o$data$trial <= 240 &
    !is.na(o$data$correct_colour)
  tibble(group = o$group, model = o$model,
         draw = seq_len(nrow(o$pred)),
         value = vapply(seq_len(nrow(o$pred)), function(k) {
           by_subject <- tapply(o$pred[k, keep] == o$data$correct_colour[keep],
                                o$data$subject_index[keep], mean)
           mean(by_subject, na.rm = TRUE)
         }, numeric(1)))
}) %>% predictive_band(c("group", "model"))

observed_pse <- map_dfr(behavioural_objects, function(o) {
  subject_summary(o$data, o$data$observed_choice) %>%
    mutate(group = o$info$group)
}) %>% filter(prior %in% 1:2) %>%
  mutate(prior_colour = if_else(prior == 1L, "Blue prior", "Red prior")) %>%
  group_by(group, prior_colour) %>% summarise(mean_se(value), .groups = "drop")
pred_pse <- map_dfr(model_objects, function(o) {
  map_dfr(seq_len(nrow(o$pred)), function(k) {
    subject_summary(o$data, o$pred[k, ]) %>%
      mutate(group = o$group, model = o$model, draw = k)
  })
}) %>% filter(prior %in% 1:2) %>%
  mutate(prior_colour = if_else(prior == 1L, "Blue prior", "Red prior")) %>%
  group_by(group, model, draw, prior_colour) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  predictive_band(c("group", "model", "prior_colour"))

# A 50-trial window ending at each trial, as in the behavioural figure.
observed_ends <- 50:240
model_ends <- unique(c(seq(50, 240, by = TRAJECTORY_STEP), 240L))
observed_pse_time <- map_dfr(behavioural_objects, function(o) {
  map_dfr(observed_ends, function(t) {
    subject_summary(o$data, o$data$observed_choice, t - 49L, t) %>%
      mutate(group = o$info$group, trial = t)
  })
}) %>% filter(prior %in% 1:2) %>%
  mutate(prior_colour = if_else(prior == 1L, "Blue prior", "Red prior")) %>%
  group_by(group, trial, prior_colour) %>%
  summarise(mean_se(value), .groups = "drop") %>%
  group_by(group, prior_colour) %>%
  mutate(mean = as.numeric(stats::filter(mean, rep(1/3, 3), sides = 2))) %>%
  ungroup()

pred_pse_time <- map_dfr(model_objects, function(o) {
  draw_rows <- unique(as.integer(round(seq(1, nrow(o$pred),
                                           length.out = min(N_TRAJECTORY_DRAWS, nrow(o$pred))))))
  map_dfr(draw_rows, function(k) {
    map_dfr(model_ends, function(t) {
      subject_summary(o$data, o$pred[k, ], t - 49L, t) %>%
        mutate(group = o$group, model = o$model, draw = k, trial = t)
    })
  })
}) %>% filter(prior %in% 1:2) %>%
  mutate(prior_colour = if_else(prior == 1L, "Blue prior", "Red prior")) %>%
  group_by(group, model, draw, trial, prior_colour) %>%
  summarise(value = mean(value, na.rm = TRUE), .groups = "drop") %>%
  predictive_band(c("group", "model", "trial", "prior_colour"))

# The behavioural figure has one wide accuracy panel and five PSE panels.
# Repeat that layout for each model so every curve can be read clearly.
group_cols <- c("Implicit Unaware" = "#F28E8E",
                "Implicit Aware" = "#91C998",
                "Explicit Undirected" = "#E8993B",
                "Explicit True" = "#18824E",
                "Explicit Deceptive" = "#9D2929")
prior_cols <- c("Blue prior" = "#0072B2", "Red prior" = "#D55E00")

# Each dot in the upper plot is one participant, ordered by accuracy within
# their group, as in the supplied behavioural figure.
accuracy_subjects <- map_dfr(behavioural_objects, function(o) {
  o$data %>% filter(!is.na(trial), trial <= 240) %>%
    group_by(subject_index) %>%
    summarise(value = mean(observed_choice == correct_colour, na.rm = TRUE),
              .groups = "drop") %>%
    mutate(group = o$info$group)
}) %>%
  mutate(group = factor(group, levels = group_levels)) %>%
  arrange(group, value) %>%
  group_by(group) %>% mutate(rank = row_number()) %>% ungroup()

sizes <- accuracy_subjects %>% count(group) %>%
  mutate(offset = lag(cumsum(n), default = 0L), mid = offset + (n + 1) / 2)
accuracy_subjects <- accuracy_subjects %>%
  left_join(sizes %>% select(group, offset), by = "group") %>%
  mutate(position = offset + rank)
group_boundaries <- head(cumsum(sizes$n), -1) + .5

# Model accuracy per participant and draw, then participant means. These
# supply an aligned model marker in the upper behavioural-style panel.
model_subject_accuracy <- map_dfr(model_objects, function(o) {
  d <- o$data
  keep <- !is.na(d$trial) & d$trial <= 240 & !is.na(d$correct_colour)
  ids <- unique(d$subject_index[keep])
  map_dfr(ids, function(id) {
    idx <- which(keep & d$subject_index == id)
    tibble(group = o$group, model = o$model, subject_index = id,
           mean = mean(rowMeans(o$pred[, idx, drop = FALSE] ==
                                  matrix(d$correct_colour[idx], nrow(o$pred),
                                         length(idx), byrow = TRUE))))
  })
}) %>%
  left_join(accuracy_subjects %>%
              select(group, subject_index, position),
            by = c("group", "subject_index"))

p_accuracy_for <- function(model_name) {
  model_d <- model_subject_accuracy %>% filter(model == model_name)
  model_means <- pred_accuracy %>% filter(model == model_name) %>%
    left_join(sizes %>% mutate(group = as.character(group)) %>%
                select(group, mid, offset, n), by = "group")
  observed_means <- observed_accuracy %>%
    left_join(sizes %>% mutate(group = as.character(group)) %>%
                select(group, mid, offset, n), by = "group")
  ggplot(accuracy_subjects, aes(position, 100 * value)) +
    geom_hline(yintercept = 50, linetype = "dashed", colour = "grey55") +
    geom_vline(xintercept = group_boundaries, colour = "grey88") +
    geom_point(aes(colour = group), size = 1.55, alpha = .9) +
    geom_point(data = model_d, aes(position, 100 * mean),
               inherit.aes = FALSE, shape = 1, colour = "grey25", size = 1.5) +
    geom_segment(data = observed_means,
                 aes(x = offset + .5, xend = offset + n + .5,
                     y = 100 * mean, yend = 100 * mean),
                 inherit.aes = FALSE, linewidth = .65) +
    geom_segment(data = model_means,
                 aes(x = offset + .5, xend = offset + n + .5,
                     y = 100 * mean, yend = 100 * mean),
                 inherit.aes = FALSE, linetype = "dashed", linewidth = .65) +
    scale_colour_manual(values = group_cols, guide = "none") +
    scale_x_continuous(breaks = sizes$mid, labels = rep("", nrow(sizes)),
                       expand = expansion(add = 1)) +
    coord_cartesian(ylim = c(40, 85)) +
    labs(x = NULL, y = "Accuracy (%)", title = "Accuracy") +
    theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          axis.text.x = element_blank(), plot.title = element_text(face = "bold"))
}

p_trajectory_for <- function(group_name, model_name) {
  obs <- observed_pse_time %>% filter(group == group_name)
  mod <- pred_pse_time %>% filter(group == group_name, model == model_name)
  ggplot() +
    geom_hline(yintercept = .5, linetype = "dashed", colour = "grey55") +
    geom_ribbon(data = mod,
                aes(trial, ymin = lo, ymax = hi, fill = prior_colour),
                alpha = .08, na.rm = TRUE) +
    geom_line(data = mod,
              aes(trial, mean, colour = prior_colour, group = prior_colour),
              linetype = "dashed", linewidth = .8, na.rm = TRUE) +
    geom_line(data = obs,
              aes(trial, mean, colour = prior_colour, group = prior_colour),
              linewidth = 1.1, na.rm = TRUE) +
    scale_colour_manual(values = prior_cols, drop = FALSE) +
    scale_fill_manual(values = prior_cols, guide = "none") +
    scale_x_continuous(breaks = c(50, 100, 150, 200, 240),
                       limits = c(50, 240)) +
    coord_cartesian(ylim = c(.2, .8)) +
    labs(x = "Trial", y = if (group_name == group_levels[1]) "Raw PSE" else NULL,
         title = group_name) +
    theme_minimal(base_size = 10) +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(face = "bold", size = 10),
          axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = if (group_name == group_levels[5]) "bottom" else "none")
}

p_overall_for <- function(model_name) {
  obs <- observed_pse %>% mutate(group = factor(group, levels = group_levels))
  mod <- pred_pse %>% filter(model == model_name) %>%
    mutate(group = factor(group, levels = group_levels))
  ggplot() +
    geom_hline(yintercept = .5, linetype = "dashed", colour = "grey55") +
    geom_errorbar(data = mod,
                  aes(prior_colour, ymin = lo, ymax = hi, colour = prior_colour),
                  width = .17, alpha = .65) +
    geom_point(data = mod, aes(prior_colour, mean, colour = prior_colour),
               shape = 1, size = 3) +
    geom_errorbar(data = obs,
                  aes(prior_colour, ymin = mean - se, ymax = mean + se,
                      colour = prior_colour), width = .08, linewidth = .9) +
    geom_point(data = obs, aes(prior_colour, mean, colour = prior_colour),
               size = 2.5) +
    facet_wrap(~group, nrow = 1) +
    scale_colour_manual(values = prior_cols, guide = "none") +
    coord_cartesian(ylim = c(.2, .8)) +
    labs(x = NULL, y = "Raw PSE", title = paste(model_name, "— overall PSE"),
         subtitle = "Filled point ±1 SE: observed participants; hollow point and interval: model predictions") +
    theme_minimal(base_size = 11) +
    theme(strip.text = element_text(face = "bold"),
          axis.text.x = element_text(angle = 35, hjust = 1),
          panel.grid.minor = element_blank())
}

# Use grid viewports to reproduce the 1-wide / 5-small panel geometry
# without requiring any extra R packages. Nothing is saved to disk.
draw_behavioural_layout <- function(model_name) {
  grid::grid.newpage()
  grid::pushViewport(grid::viewport(
    layout = grid::grid.layout(2, 5,
                               heights = grid::unit(c(.48, .52), "null"))))
  print(p_accuracy_for(model_name),
        vp = grid::viewport(layout.pos.row = 1, layout.pos.col = 1:5),
        newpage = FALSE)
  for (g in seq_along(group_levels)) {
    print(p_trajectory_for(group_levels[g], model_name),
          vp = grid::viewport(layout.pos.row = 2, layout.pos.col = g),
          newpage = FALSE)
  }
  grid::popViewport()
}

for (model_name in intersect(model_levels, unique(pred_pse_time$model))) {
  draw_behavioural_layout(model_name)
  print(p_overall_for(model_name))
}

# =====================================================================
# 12. DISPLAY PLOTS
# =====================================================================

print(
  p_evidence
)
