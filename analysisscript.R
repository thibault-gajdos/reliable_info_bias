rm(list = ls(all.names = TRUE))

suppressPackageStartupMessages({
  library(tidyverse)
  library(posterior)
  library(ggplot2)
  library(grid)
  library(loo)
})

# ==============================================================================
# 1. CONFIGURATION — EDIT PATHS HERE ONLY
# ==============================================================================

project_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias"
stan_dir    <- file.path(project_dir, "stan")
fits_dir    <- file.path(stan_dir, "results", "fits", "exp11_unaware")
loo_dir     <- file.path(stan_dir, "results", "loo", "exp11_unaware")
output_dir  <- file.path(stan_dir, "results")
figure_dir  <- file.path(output_dir, "figures")
table_dir   <- file.path(output_dir, "tables")

dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir,  recursive = TRUE, showWarnings = FALSE)

# Optional sections. Set FALSE only when the required files are unavailable.
RUN_PARTICIPANT_TESTS <- FALSE
RUN_LOO_COMPARISON    <- TRUE

N_PERM <- 10000L
SEED   <- 12345L

group_levels <- c(
  "Implicit Unaware Base Rate",
  "Implicit Aware Base Rate",
  "Explicit Undirected Base Rate",
  "Explicit True Base Rate",
  "Explicit Deceptive Base Rate"
)

group_short <- c(
  "Implicit Unaware Base Rate"    = "Implicit Unaware",
  "Implicit Aware Base Rate"      = "Implicit Aware",
  "Explicit Undirected Base Rate" = "Explicit Undirected",
  "Explicit True Base Rate"       = "Explicit True",
  "Explicit Deceptive Base Rate"  = "Explicit Deceptive"
)

group_cols <- c(
  "Implicit Unaware Base Rate"    = "#F4A3A3",
  "Implicit Aware Base Rate"      = "#8FD694",
  "Explicit Undirected Base Rate" = "#E69F00",
  "Explicit True Base Rate"       = "#1B7837",
  "Explicit Deceptive Base Rate"  = "#8B0000"
)

group_axis_labels <- c(
  "Implicit Unaware Base Rate"    = "Implicit\nUnaware",
  "Implicit Aware Base Rate"      = "Implicit\nAware",
  "Explicit Undirected Base Rate" = "Explicit\nUndirected",
  "Explicit True Base Rate"       = "Explicit\nTrue",
  "Explicit Deceptive Base Rate"  = "Explicit\nDeceptive"
)

fit_files <- c(
  "Implicit Unaware Base Rate" = "leta_2channels_unaware_exp11.rdata",
  "Implicit Aware Base Rate" = "leta_2channels_aware_exp11.rdata",
  "Explicit Undirected Base Rate" = "leta_2channels_aware_exp12.rdata",
  "Explicit True Base Rate" = "leta_2channels_truthful_exp13.rdata",
  "Explicit Deceptive Base Rate" = "leta_2channels_deceptive_exp13.rdata"
)

# Master list used for ordering.
parameter_names <- c(
  "mu_alpha",
  "mu_beta",
  "mu_lambda",
  "mu_delta",
  "mu_eta",
  "mu_alpha0",
  "mu_beta0"
)

parameter_labels <- c(
  "mu_alpha"  = "alpha",
  "mu_beta"   = "beta",
  "mu_lambda" = "lambda",
  "mu_delta"  = "delta",
  "mu_eta"    = "eta",
  "mu_alpha0" = "alpha[0]",
  "mu_beta0"  = "beta[0]"
)

reference_values <- c(
  "mu_alpha"  = 1,
  "mu_beta"   = 0,
  "mu_lambda" = 0,
  "mu_delta"  = 1,
  "mu_eta"    = 0,
  "mu_alpha0" = 1,
  "mu_beta0"  = 0
)

# ==============================================================================
# 2. GENERAL HELPERS
# ==============================================================================

save_plot_both <- function(plot, stem, width, height) {
  ggsave(
    file.path(figure_dir, paste0(stem, ".png")),
    plot,
    width = width,
    height = height,
    dpi = 300
  )
  
  ggsave(
    file.path(figure_dir, paste0(stem, ".pdf")),
    plot,
    width = width,
    height = height
  )
}

load_single_object <- function(path, preferred_name = NULL) {
  
  if (!file.exists(path)) {
    stop("File not found: ", path)
  }
  
  e <- new.env(parent = emptyenv())
  loaded <- load(path, envir = e)
  
  if (!is.null(preferred_name) && preferred_name %in% loaded) {
    return(e[[preferred_name]])
  }
  
  if (length(loaded) == 1L) {
    return(e[[loaded]])
  }
  
  stop(
    "File contains several objects and '",
    preferred_name,
    "' was not found: ",
    path
  )
}

load_fit <- function(filename) {
  
  fit <- load_single_object(
    file.path(fits_dir, filename),
    "fit"
  )
  
  if (!inherits(fit, c("CmdStanMCMC", "CmdStanFit"))) {
    stop("Object is not a CmdStan fit: ", filename)
  }
  
  fit
}

# Identify the model from its filename.
# The order is important because "learning_localeta"
# also contains the shorter string "localeta".

get_model_spec <- function(filename) {
  
  filename_lower <- tolower(basename(filename))
  
  if (str_detect(filename_lower, "learning_localeta")) {
    return(list(
      model = "Learning + Local eta",
      parameters = c(
        "mu_alpha",
        "mu_beta",
        "mu_lambda",
        "mu_delta",
        "mu_eta"
      )
    ))
  }
  
  if (str_detect(filename_lower, "localeta")) {
    return(list(
      model = "Local eta",
      parameters = c(
        "mu_alpha",
        "mu_beta",
        "mu_lambda",
        "mu_eta"
      )
    ))
  }
  
  if (str_detect(filename_lower, "learning")) {
    return(list(
      model = "Learning",
      parameters = c(
        "mu_alpha",
        "mu_beta",
        "mu_lambda",
        "mu_delta"
      )
    ))
  }
  
  if (str_detect(filename_lower, "simple")) {
    return(list(
      model = "Simple",
      parameters = c(
        "mu_alpha",
        "mu_beta",
        "mu_lambda"
      )
    ))
  }
  
  # Test the more specific name first: leta_2channels also contains 2channels.
  if (str_detect(filename_lower, "leta_2channels")) {
    return(list(
      model = "Local eta + 2 channels",
      parameters = c(
        "mu_alpha", "mu_beta", "mu_lambda", "mu_eta",
        "mu_alpha0", "mu_beta0", "mu_delta"
      )
    ))
  }
  
  if (str_detect(filename_lower, "2channels")) {
    return(list(
      model = "2 channels",
      parameters = c(
        "mu_alpha", "mu_beta", "mu_lambda",
        "mu_alpha0", "mu_beta0", "mu_delta"
      )
    ))
  }
  
  stop(
    "Could not identify model type from filename: ",
    filename
  )
}

fits <- set_names(
  map(unname(fit_files), load_fit),
  names(fit_files)
)

model_specs <- map(
  fit_files,
  get_model_spec
)

model_types <- unique(
  map_chr(model_specs, "model")
)

if (length(model_types) != 1L) {
  stop(
    "The five group files do not all use the same model. Found: ",
    paste(model_types, collapse = ", ")
  )
}

cat(
  "Loaded ",
  model_types,
  " fits for all five groups.\n",
  sep = ""
)

# ==============================================================================
# 3. GROUP-LEVEL POSTERIOR DRAWS
# ==============================================================================

extract_group_draws <- function(
    fit,
    group_name,
    alpha_scale = 6,
    delta_scale = 2
) {
  
  d <- as_draws_df(
    fit$draws()
  )
  
  filename <- unname(
    fit_files[[group_name]]
  )
  
  spec <- get_model_spec(
    filename
  )
  
  expected_parameters <- spec$parameters
  
  is_two_channel <- str_detect(
    tolower(basename(filename)),
    "2channels"
  )
  
  # Preferred route:
  # use transformed generated quantities saved directly by Stan.
  
  if (all(expected_parameters %in% names(d))) {
    
    out <- d %>%
      select(
        all_of(expected_parameters)
      )
    
  } else {
    
    # Fallback route:
    # transform raw group-level mu_pr parameters.
    
    n_parameters <- length(
      expected_parameters
    )
    
    bracket <- paste0(
      "mu_pr[",
      seq_len(n_parameters),
      "]"
    )
    
    dotted <- paste0(
      "mu_pr.",
      seq_len(n_parameters),
      "."
    )
    
    if (all(bracket %in% names(d))) {
      
      raw_names <- bracket
      
    } else if (all(dotted %in% names(d))) {
      
      raw_names <- dotted
      
    } else {
      
      raw_names <- names(d)[
        str_detect(
          names(d),
          "^mu_pr(\\[|\\.)"
        )
      ]
      
      raw_names <- raw_names[
        seq_len(
          min(
            n_parameters,
            length(raw_names)
          )
        )
      ]
    }
    
    if (length(raw_names) < n_parameters) {
      
      stop(
        "Could not find the expected group-level parameters for ",
        group_name,
        "\nModel identified from filename: ",
        spec$model,
        "\nExpected: ",
        paste(expected_parameters, collapse = ", "),
        "\nAvailable matching names: ",
        paste(
          names(d)[
            str_detect(
              names(d),
              "mu_|alpha|beta|lambda|delta|eta"
            )
          ],
          collapse = ", "
        )
      )
    }
    
    out <- d %>%
      select(
        all_of(
          raw_names[
            seq_len(n_parameters)
          ]
        )
      )
    
    names(out) <- expected_parameters
    
    # -------------------------------------------------------------------------
    # Transform raw mu_pr parameters onto the scales used by the Stan model.
    # -------------------------------------------------------------------------
    
    if ("mu_alpha" %in% expected_parameters) {
      out$mu_alpha <-
        pnorm(out$mu_alpha) * alpha_scale
    }
    
    if ("mu_lambda" %in% expected_parameters) {
      out$mu_lambda <-
        pnorm(out$mu_lambda)
    }
    
    # NEW alpha0 parameter
    if ("mu_alpha0" %in% expected_parameters) {
      out$mu_alpha0 <-
        pnorm(out$mu_alpha0) * alpha_scale
    }
    
    if ("mu_delta" %in% expected_parameters) {
      
      # In the new 2-channel model:
      #
      # mu_delta = Phi_approx(mu_pr[7])
      #
      # therefore delta is bounded 0–1.
      
      if (is_two_channel) {
        
        out$mu_delta <-
          pnorm(out$mu_delta)
        
      } else {
        
        # Retain old transformation for old models.
        out$mu_delta <-
          pnorm(out$mu_delta) * delta_scale
      }
    }
  }
  
  out %>%
    mutate(
      draw_id = row_number(),
      group = group_name
    ) %>%
    relocate(
      group,
      draw_id
    )
}

draws_wide <- imap_dfr(
  fits,
  extract_group_draws
) %>%
  mutate(
    group = factor(
      group,
      levels = group_levels
    )
  )

draws_long <- draws_wide %>%
  pivot_longer(
    any_of(parameter_names),
    names_to = "parameter",
    values_to = "value"
  ) %>%
  filter(
    is.finite(value)
  ) %>%
  mutate(
    parameter = factor(
      parameter,
      levels = parameter_names
    ),
    parameter_clean = recode(
      as.character(parameter),
      !!!parameter_labels
    )
  )

active_parameter_names <-
  parameter_names[
    parameter_names %in%
      unique(as.character(draws_long$parameter))
  ]

stopifnot(
  all(is.finite(draws_long$value)),
  n_distinct(draws_long$group) == 5L,
  length(active_parameter_names) ==
    length(model_specs[[1]]$parameters)
)

cat(
  "\nParameters found in the selected model:\n"
)

print(
  unname(
    parameter_labels[
      active_parameter_names
    ]
  )
)

# ==============================================================================
# 4. PREPARE RELIABILITY DISTORTION AND SEQUENTIAL WEIGHTS
# ==============================================================================

# alpha and beta transform the EXPLICIT SAMPLE RELIABILITIES.
#
# alpha0 and beta0 are NOT used here because they transform the prior belief.

distort_reliability <- function(
    x_percent,
    alpha,
    beta
) {
  
  p <- x_percent / 100
  
  epsv <- 1e-6
  
  p <- pmin(
    pmax(p, epsv),
    1 - epsv
  )
  
  logitp <- log(
    p / (1 - p)
  )
  
  distorted_p <-
    1 /
    (
      1 +
        exp(
          -(
            alpha * logitp +
              beta
          )
        )
    )
  
  distorted_p * 100
}

big_plot_theme <-
  theme_bw(base_size = 12) +
  theme(
    axis.title.x = element_text(
      size = 15,
      face = "bold"
    ),
    axis.title.y = element_text(
      size = 15,
      face = "bold"
    ),
    axis.text = element_text(
      size = 11
    ),
    plot.title = element_text(
      size = 16,
      face = "bold"
    ),
    plot.subtitle = element_text(
      size = 12
    ),
    legend.title = element_blank(),
    legend.text = element_text(
      size = 10
    ),
    legend.background = element_blank(),
    legend.key.size = unit(
      0.55,
      "cm"
    ),
    legend.spacing.x = unit(
      0.15,
      "cm"
    ),
    legend.spacing.y = unit(
      0.05,
      "cm"
    ),
    panel.grid.minor = element_blank()
  )

x_grid <- seq(
  0,
  100,
  length.out = 501
)

x_dots <- c(
  50,
  55,
  65
)

distortion_curve_one_group <- function(
    df_group,
    x_vals
) {
  
  alpha_draws <- df_group$mu_alpha
  beta_draws  <- df_group$mu_beta
  
  map_dfr(
    x_vals,
    function(xi) {
      
      y_draws <- distort_reliability(
        xi,
        alpha_draws,
        beta_draws
      )
      
      tibble(
        x = xi,
        mean = mean(
          y_draws,
          na.rm = TRUE
        ),
        l95 = quantile(
          y_draws,
          0.025,
          na.rm = TRUE
        ),
        u95 = quantile(
          y_draws,
          0.975,
          na.rm = TRUE
        )
      )
    }
  )
}

distortion_df <- draws_wide %>%
  group_split(group) %>%
  map_dfr(
    function(df_g) {
      
      gname <- as.character(
        df_g$group[1]
      )
      
      distortion_curve_one_group(
        df_g,
        x_grid
      ) %>%
        mutate(
          group = gname
        )
    }
  ) %>%
  mutate(
    group = factor(
      group,
      levels = group_levels
    )
  )

distortion_dots_df <- draws_wide %>%
  group_split(group) %>%
  map_dfr(
    function(df_g) {
      
      gname <- as.character(
        df_g$group[1]
      )
      
      distortion_curve_one_group(
        df_g,
        x_dots
      ) %>%
        mutate(
          group = gname
        )
    }
  ) %>%
  mutate(
    group = factor(
      group,
      levels = group_levels
    )
  )

# ------------------------------------------------------------------------------
# Sequential-position weights from lambda
# ------------------------------------------------------------------------------

positions <- 1:6

position_weights <- draws_wide %>%
  group_split(group) %>%
  map_dfr(
    function(df) {
      
      map_dfr(
        positions,
        function(position) {
          
          values <- exp(
            df$mu_lambda *
              (position - 6)
          )
          
          tibble(
            position = position,
            mean = mean(values),
            l95 = quantile(
              values,
              .025
            ),
            u95 = quantile(
              values,
              .975
            )
          )
        }
      ) %>%
        mutate(
          group = as.character(
            df$group[1]
          )
        )
    }
  ) %>%
  mutate(
    group = factor(
      group,
      levels = group_levels
    )
  )

# ==============================================================================
# 5. ALL FIGURES
# ==============================================================================

# ------------------------------------------------------------------------------
# 5A. Posterior-distribution figure
# ------------------------------------------------------------------------------

# This plot now includes:
#
# alpha
# beta
# lambda
# delta
# eta
# alpha0
# beta0

p_distributions <- ggplot(
  draws_long,
  aes(
    value,
    fill = group,
    colour = group
  )
) +
  
  geom_histogram(
    aes(
      y = after_stat(density)
    ),
    bins = 60,
    position = "identity",
    alpha = 0.42,
    linewidth = 0.25
  ) +
  
  facet_wrap(
    ~parameter_clean,
    scales = "free",
    nrow = 1,
    labeller = label_parsed
  ) +
  
  scale_fill_manual(
    values = group_cols,
    guide = guide_legend(
      ncol = 1
    )
  ) +
  
  scale_colour_manual(
    values = group_cols,
    guide = guide_legend(
      ncol = 1
    )
  ) +
  
  labs(
    title = "Transformed Posterior Parameter Distributions",
    subtitle = paste(
      "Five groups,",
      model_types,
      "model"
    ),
    x = "Parameter value",
    y = "Density",
    fill = NULL,
    colour = NULL
  ) +
  
  theme_bw(
    base_size = 15
  ) +
  
  theme(
    panel.grid.major = element_line(
      colour = "grey92"
    ),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(
      fill = "grey88",
      colour = "grey70"
    ),
    strip.text = element_text(
      face = "bold",
      size = 14
    ),
    plot.title = element_text(
      face = "bold",
      size = 17
    ),
    plot.subtitle = element_text(
      size = 12,
      colour = "grey35"
    ),
    axis.title = element_text(
      face = "bold",
      size = 14
    ),
    axis.text = element_text(
      size = 11
    ),
    legend.position = "bottom",
    legend.text = element_text(
      size = 12,
      face = "bold"
    ),
    legend.key.size = unit(
      0.9,
      "lines"
    )
  )

print(
  p_distributions
)

save_plot_both(
  p_distributions,
  "posterior_distributions_five_groups",
  16,
  5
)

# ------------------------------------------------------------------------------
# 5A2. ALPHA / BETA / ALPHA0 / BETA0 POSTERIOR DISTRIBUTIONS
# ------------------------------------------------------------------------------

alpha_beta_parameters <- draws_long %>%
  filter(
    parameter %in% c(
      "mu_alpha",
      "mu_beta",
      "mu_alpha0",
      "mu_beta0"
    )
  ) %>%
  mutate(
    parameter_clean = factor(
      parameter_clean,
      levels = c(
        "alpha",
        "beta",
        "alpha[0]",
        "beta[0]"
      )
    )
  )

alpha_beta_reference <- tibble(
  parameter_clean = factor(
    c(
      "alpha",
      "beta",
      "alpha[0]",
      "beta[0]"
    ),
    levels = c(
      "alpha",
      "beta",
      "alpha[0]",
      "beta[0]"
    )
  ),
  reference = c(
    1,
    0,
    1,
    0
  )
)

p_alpha_beta <- ggplot(
  alpha_beta_parameters,
  aes(
    x = value,
    fill = group,
    colour = group
  )
) +
  
  geom_density(
    alpha = 0.20,
    linewidth = 1
  ) +
  
  geom_vline(
    data = alpha_beta_reference,
    aes(
      xintercept = reference
    ),
    inherit.aes = FALSE,
    linetype = "dashed",
    colour = "black",
    linewidth = 0.8
  ) +
  
  facet_wrap(
    ~parameter_clean,
    scales = "free_x",
    nrow = 1,
    labeller = label_parsed
  ) +
  
  scale_fill_manual(
    values = group_cols
  ) +
  
  scale_colour_manual(
    values = group_cols
  ) +
  
  labs(
    title = "Evidence and prior transformation parameters",
    x = "Posterior value",
    y = "Density",
    fill = NULL,
    colour = NULL
  ) +
  
  theme_bw(
    base_size = 15
  ) +
  
  theme(
    panel.grid.major = element_line(
      colour = "grey92"
    ),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(
      fill = "grey88",
      colour = "grey70"
    ),
    strip.text = element_text(
      face = "bold",
      size = 14
    ),
    plot.title = element_text(
      face = "bold",
      size = 17
    ),
    axis.title = element_text(
      face = "bold",
      size = 14
    ),
    axis.text = element_text(
      size = 11
    ),
    legend.position = "bottom",
    legend.text = element_text(
      size = 11,
      face = "bold"
    )
  )

print(
  p_alpha_beta
)

save_plot_both(
  p_alpha_beta,
  "alpha_beta_alpha0_beta0_posteriors_five_groups",
  15,
  5
)

# ------------------------------------------------------------------------------
# 5B. Reliability-distortion figure
# ------------------------------------------------------------------------------

# IMPORTANT:
# This deliberately continues to use ONLY alpha and beta.
#
# alpha0 and beta0 operate on the PRIOR, not on sample reliability.

p_distortion <- ggplot() +
  
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    linewidth = 1,
    color = "black"
  ) +
  
  geom_ribbon(
    data = distortion_df,
    aes(
      x = x,
      ymin = l95,
      ymax = u95,
      fill = group
    ),
    alpha = 0.12,
    colour = NA
  ) +
  
  geom_line(
    data = distortion_df,
    aes(
      x = x,
      y = mean,
      color = group
    ),
    linewidth = 1.4
  ) +
  
  geom_errorbar(
    data = distortion_dots_df,
    aes(
      x = x,
      ymin = l95,
      ymax = u95,
      color = group
    ),
    width = 0,
    linewidth = 0.7
  ) +
  
  geom_point(
    data = distortion_dots_df,
    aes(
      x = x,
      y = mean,
      fill = group
    ),
    shape = 21,
    size = 3.2,
    color = "black",
    stroke = 0.5
  ) +
  
  scale_color_manual(
    values = group_cols
  ) +
  
  scale_fill_manual(
    values = group_cols
  ) +
  
  coord_fixed() +
  
  labs(
    title = "Reliability distortion",
    x = "True reliability (%)",
    y = "Distorted reliability (%)"
  ) +
  
  xlim(
    0,
    100
  ) +
  
  ylim(
    0,
    100
  ) +
  
  big_plot_theme +
  
  theme(
    legend.position = "bottom",
    legend.box = "horizontal"
  )

print(
  p_distortion
)

save_plot_both(
  p_distortion,
  "reliability_distortion_5groups_base_rate_colours",
  7,
  6
)

write_csv(
  distortion_dots_df,
  file.path(
    table_dir,
    "distorted_reliability_50_55_65.csv"
  )
)

# ------------------------------------------------------------------------------
# 5C. Sequential-position weighting figure
# ------------------------------------------------------------------------------

p_weights <- ggplot(
  position_weights,
  aes(
    position,
    mean,
    colour = group,
    fill = group
  )
) +
  
  geom_ribbon(
    aes(
      ymin = l95,
      ymax = u95
    ),
    colour = NA,
    alpha = .10
  ) +
  
  geom_line(
    linewidth = 1
  ) +
  
  geom_point(
    size = 2
  ) +
  
  scale_x_continuous(
    breaks = positions
  ) +
  
  scale_colour_manual(
    values = group_cols
  ) +
  
  scale_fill_manual(
    values = group_cols
  ) +
  
  labs(
    x = "Sample position",
    y = "Weight relative to sample 6",
    colour = NULL,
    fill = NULL
  ) +
  
  theme_bw(
    base_size = 13
  ) +
  
  theme(
    legend.position = "bottom"
  )

print(
  p_weights
)

save_plot_both(
  p_weights,
  "sequential_weights_five_groups",
  10,
  7
)

write_csv(
  position_weights,
  file.path(
    table_dir,
    "sequential_position_weights.csv"
  )
)

# ------------------------------------------------------------------------------
# 5D. Delta and eta posterior figure
# ------------------------------------------------------------------------------

# These are group-level posterior distributions,
# not participant distributions.

delta_eta <- draws_long %>%
  filter(
    parameter %in% c(
      "mu_delta",
      "mu_eta"
    )
  ) %>%
  mutate(
    parameter_clean = factor(
      parameter_clean,
      levels = c(
        "delta",
        "eta"
      )
    )
  )

available_delta_eta <- intersect(
  c(
    "mu_delta",
    "mu_eta"
  ),
  active_parameter_names
)

if (length(available_delta_eta) > 0L) {
  
  reference_lines <- tibble(
    parameter = available_delta_eta,
    
    parameter_clean = factor(
      unname(
        parameter_labels[
          available_delta_eta
        ]
      ),
      levels = c(
        "delta",
        "eta"
      )
    ),
    
    reference = unname(
      reference_values[
        available_delta_eta
      ]
    )
  )
  
  p_delta_eta <- ggplot(
    delta_eta,
    aes(
      group,
      value,
      fill = group
    )
  ) +
    
    geom_boxplot(
      width = 0.60,
      colour = "black",
      linewidth = 0.40,
      outlier.shape = NA
    ) +
    
    geom_hline(
      data = reference_lines,
      aes(
        yintercept = reference
      ),
      inherit.aes = FALSE,
      linetype = "dashed",
      colour = "red"
    ) +
    
    facet_wrap(
      ~parameter_clean,
      scales = "free_y",
      nrow = 1,
      labeller = label_parsed
    ) +
    
    scale_fill_manual(
      values = group_cols
    ) +
    
    scale_x_discrete(
      labels = group_axis_labels
    ) +
    
    labs(
      x = NULL,
      y = "Posterior value",
      fill = NULL
    ) +
    
    theme_bw(
      base_size = 16
    ) +
    
    theme(
      panel.grid.major = element_line(
        colour = "grey88",
        linewidth = 0.40
      ),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(
        colour = "grey40",
        linewidth = 0.45
      ),
      strip.background = element_rect(
        fill = "grey82",
        colour = "grey40",
        linewidth = 0.45
      ),
      strip.text = element_text(
        face = "bold",
        size = 17
      ),
      axis.title.y = element_text(
        face = "bold",
        size = 17
      ),
      axis.text.x = element_text(
        face = "bold",
        size = 9.5
      ),
      axis.text.y = element_text(
        face = "bold",
        size = 10
      ),
      axis.ticks = element_line(
        linewidth = 0.35
      ),
      legend.position = "none",
      plot.margin = margin(
        10,
        12,
        10,
        10
      )
    )
  
  delta_eta_filename <-
    if (identical(
      available_delta_eta,
      "mu_eta"
    )) {
      
      "eta_posterior_five_groups"
      
    } else if (identical(
      available_delta_eta,
      "mu_delta"
    )) {
      
      "delta_posterior_five_groups"
      
    } else {
      
      "delta_eta_posteriors_five_groups"
    }
  
  print(
    p_delta_eta
  )
  
  save_plot_both(
    p_delta_eta,
    delta_eta_filename,
    12,
    5.5
  )
  
} else {
  
  cat(
    "\nNo delta or eta parameter is estimated by this model; skipping that figure.\n"
  )
}

# ==============================================================================
# 6. POSTERIOR SUMMARIES AND REFERENCE-VALUE PROBABILITIES
# ==============================================================================

posterior_summary <- draws_long %>%
  group_by(
    group,
    parameter_clean
  ) %>%
  summarise(
    mean = mean(value),
    median = median(value),
    sd = sd(value),
    l95 = quantile(
      value,
      .025
    ),
    u95 = quantile(
      value,
      .975
    ),
    .groups = "drop"
  )

writeup_table <- posterior_summary %>%
  mutate(
    result = sprintf(
      "M = %.3f, 95%% CrI [%.3f, %.3f]",
      mean,
      l95,
      u95
    )
  ) %>%
  select(
    group,
    parameter_clean,
    result
  ) %>%
  pivot_wider(
    names_from = parameter_clean,
    values_from = result
  )

reference_stats <- draws_long %>%
  mutate(
    reference = unname(
      reference_values[
        as.character(parameter)
      ]
    )
  ) %>%
  group_by(
    group,
    parameter_clean,
    reference
  ) %>%
  summarise(
    probability_above = mean(
      value > reference
    ),
    probability_below = mean(
      value < reference
    ),
    l95 = quantile(
      value,
      .025
    ),
    u95 = quantile(
      value,
      .975
    ),
    .groups = "drop"
  ) %>%
  mutate(
    credible_interval_excludes_reference =
      l95 > reference |
      u95 < reference
  )

write_csv(
  posterior_summary,
  file.path(
    table_dir,
    "group_posterior_summaries.csv"
  )
)

write_csv(
  writeup_table,
  file.path(
    table_dir,
    "group_posterior_writeup_table.csv"
  )
)

write_csv(
  reference_stats,
  file.path(
    table_dir,
    "posterior_reference_value_tests.csv"
  )
)

cat(
  "\n=========================================================\n"
)

cat(
  "GROUP POSTERIOR SUMMARIES\n"
)

cat(
  "=========================================================\n\n"
)

print(
  posterior_summary,
  n = Inf,
  width = Inf
)

cat(
  "\n=========================================================\n"
)

cat(
  "WRITE-UP TABLE: M AND 95% CrI FOR EACH GROUP × PARAMETER\n"
)

cat(
  "=========================================================\n\n"
)

print(
  writeup_table,
  n = Inf,
  width = Inf
)

cat(
  "\n=========================================================\n"
)

cat(
  "POSTERIOR PROBABILITIES AGAINST REFERENCE VALUES\n"
)

cat(
  "=========================================================\n\n"
)

print(
  reference_stats,
  n = Inf,
  width = Inf
)

# ==============================================================================
# 7. BAYESIAN PAIRWISE POSTERIOR CONTRASTS
# ==============================================================================

# Groups were fitted independently.
# Equal numbers of draws are paired only to generate
# Monte Carlo draws from the difference distribution.

posterior_contrast <- function(
    parameter_name,
    group_1,
    group_2
) {
  
  x1 <- draws_long %>%
    filter(
      parameter == parameter_name,
      group == group_1
    ) %>%
    pull(value)
  
  x2 <- draws_long %>%
    filter(
      parameter == parameter_name,
      group == group_2
    ) %>%
    pull(value)
  
  n <- min(
    length(x1),
    length(x2)
  )
  
  difference <-
    sample(
      x1,
      n
    ) -
    sample(
      x2,
      n
    )
  
  tibble(
    parameter = unname(
      parameter_labels[
        parameter_name
      ]
    ),
    group_1 = group_1,
    group_2 = group_2,
    mean_difference = mean(
      difference
    ),
    l95 = quantile(
      difference,
      .025
    ),
    u95 = quantile(
      difference,
      .975
    ),
    probability_group1_greater =
      mean(
        difference > 0
      ),
    probability_group1_lower =
      mean(
        difference < 0
      ),
    credible_interval_excludes_zero =
      l95 > 0 |
      u95 < 0
  )
}

set.seed(
  SEED
)

group_pairs <- combn(
  group_levels,
  2,
  simplify = FALSE
)

posterior_pairwise <- map_dfr(
  active_parameter_names,
  function(parameter_name) {
    
    map_dfr(
      group_pairs,
      ~posterior_contrast(
        parameter_name,
        .x[1],
        .x[2]
      )
    )
  }
)

write_csv(
  posterior_pairwise,
  file.path(
    table_dir,
    "bayesian_pairwise_parameter_contrasts.csv"
  )
)

cat(
  "\n=========================================================\n"
)

cat(
  "ALL BAYESIAN PAIRWISE PARAMETER CONTRASTS\n"
)

cat(
  "=========================================================\n\n"
)

print(
  posterior_pairwise,
  n = Inf,
  width = Inf
)

cat(
  "\n=========================================================\n"
)

cat(
  "CONTRASTS WHOSE 95% CrI EXCLUDES ZERO\n"
)

cat(
  "=========================================================\n\n"
)

print(
  posterior_pairwise %>%
    filter(
      credible_interval_excludes_zero
    ),
  n = Inf,
  width = Inf
)

# ==============================================================================
# 8. PARTICIPANT-LEVEL ESTIMATES AND RANDOMISATION TESTS
# ==============================================================================

# For the 2channels model:
#
# params[,1] = alpha
# params[,2] = beta
# params[,3] = lambda
# params[,4] = eta
# params[,5] = alpha0
# params[,6] = beta0
# params[,7] = delta

extract_participant_params <- function(
    fit,
    group_name
) {
  
  filename <- unname(
    fit_files[[group_name]]
  )
  
  spec <- get_model_spec(
    filename
  )
  
  participant_parameter_names <- unname(
    parameter_labels[
      spec$parameters
    ]
  )
  
  s <- fit$summary(
    variables = "params"
  ) %>%
    as_tibble()
  
  parsed <- str_match(
    s$variable,
    "^params\\[([0-9]+),([0-9]+)\\]$"
  )
  
  out <- tibble(
    participant_index = as.integer(
      parsed[, 2]
    ),
    parameter_index = as.integer(
      parsed[, 3]
    ),
    estimate = s$mean
  ) %>%
    filter(
      !is.na(participant_index),
      parameter_index %in%
        seq_along(
          participant_parameter_names
        )
    ) %>%
    mutate(
      parameter =
        participant_parameter_names[
          parameter_index
        ],
      group = group_name,
      participant_id = paste(
        group_name,
        participant_index,
        sep = "__"
      )
    ) %>%
    select(
      participant_id,
      participant_index,
      group,
      parameter,
      estimate
    )
  
  if (nrow(out) == 0) {
    stop(
      "No participant-level params found for ",
      group_name
    )
  }
  
  out
}

randomisation_test <- function(
    data,
    parameter_name,
    group_1,
    group_2,
    n_perm = N_PERM
) {
  
  d <- data %>%
    filter(
      parameter == parameter_name,
      group %in% c(
        group_1,
        group_2
      ),
      is.finite(estimate)
    )
  
  observed <-
    mean(
      d$estimate[
        d$group == group_1
      ]
    ) -
    mean(
      d$estimate[
        d$group == group_2
      ]
    )
  
  permuted <- replicate(
    n_perm,
    {
      
      shuffled <- sample(
        d$group,
        replace = FALSE
      )
      
      mean(
        d$estimate[
          shuffled == group_1
        ]
      ) -
        mean(
          d$estimate[
            shuffled == group_2
          ]
        )
    }
  )
  
  tibble(
    parameter = parameter_name,
    group_1 = group_1,
    group_2 = group_2,
    
    n_group_1 = sum(
      d$group == group_1
    ),
    
    n_group_2 = sum(
      d$group == group_2
    ),
    
    mean_group_1 = mean(
      d$estimate[
        d$group == group_1
      ]
    ),
    
    mean_group_2 = mean(
      d$estimate[
        d$group == group_2
      ]
    ),
    
    observed_difference = observed,
    
    randomisation_p =
      (
        sum(
          abs(permuted) >=
            abs(observed)
        ) +
          1
      ) /
      (
        n_perm +
          1
      )
  )
}

if (RUN_PARTICIPANT_TESTS) {
  
  participant_parameters <- imap_dfr(
    fits,
    extract_participant_params
  )
  
  participant_counts <-
    participant_parameters %>%
    distinct(
      group,
      participant_id
    ) %>%
    count(
      group,
      name = "N"
    )
  
  print(
    participant_counts
  )
  
  set.seed(
    SEED
  )
  
  active_parameter_clean <- unname(
    parameter_labels[
      active_parameter_names
    ]
  )
  
  randomisation_stats <- map_dfr(
    active_parameter_clean,
    function(parameter_name) {
      
      map_dfr(
        group_pairs,
        ~randomisation_test(
          participant_parameters,
          parameter_name,
          .x[1],
          .x[2]
        )
      )
    }
  ) %>%
    group_by(
      parameter
    ) %>%
    mutate(
      p_holm = p.adjust(
        randomisation_p,
        method = "holm"
      )
    ) %>%
    ungroup()
  
  write_csv(
    participant_parameters,
    file.path(
      table_dir,
      "participant_parameter_posterior_means.csv"
    )
  )
  
  write_csv(
    randomisation_stats,
    file.path(
      table_dir,
      "pairwise_parameter_randomisation_tests.csv"
    )
  )
  
  cat(
    "\n=========================================================\n"
  )
  
  cat(
    "ALL PARTICIPANT-LEVEL RANDOMISATION TESTS\n"
  )
  
  cat(
    "=========================================================\n\n"
  )
  
  print(
    randomisation_stats,
    n = Inf,
    width = Inf
  )
  
  cat(
    "\n=========================================================\n"
  )
  
  cat(
    "HOLM-CORRECTED RANDOMISATION TESTS WITH p < .05\n"
  )
  
  cat(
    "=========================================================\n\n"
  )
  
  print(
    randomisation_stats %>%
      filter(
        p_holm < .05
      ) %>%
      arrange(
        parameter,
        p_holm
      ),
    n = Inf,
    width = Inf
  )
}
# ==============================================================================
# 9. LOO MODEL COMPARISON — AVAILABLE MODELS BY GROUP
# ==============================================================================

loo_file_map <- list(
  "Implicit Unaware Base Rate" = c(
    learning = "learning_unaware_exp11.rdata",
    local_eta = "localeta_unaware_exp11.rdata",
    learning_local_eta = "loo_learning_localeta_unaware_exp11.rdata",
    two_channels = "2channels_unaware_exp11.rdata",
    local_eta_two_channels = "leta_2channels_unaware_exp11.rdata"
  ),
  
  "Implicit Aware Base Rate" = c(
    learning = "learning_aware_exp11.rdata",
    local_eta = "localeta_aware_exp11.rdata",
    learning_local_eta = "loo_learning_localeta_aware_exp11.rdata",
    two_channels = "2channels_aware_exp11.rdata",
    local_eta_two_channels = "leta_2channels_aware_exp11.rdata"
  ),
  
  "Explicit Undirected Base Rate" = c(
    learning = "learning_aware_exp12.rdata",
    local_eta = "localeta_aware_exp12.rdata",
    learning_local_eta = "loo_learning_localeta_aware_exp12.rdata",
    two_channels = "2channels_aware_exp12.rdata",
    local_eta_two_channels = "leta_2channels_aware_exp12.rdata"
  ),
  
  "Explicit True Base Rate" = c(
    learning = "learning_truthful_exp13.rdata",
    local_eta = "localeta_truthful_exp13.rdata",
    learning_local_eta = "loo_learning_localeta_truthful_exp13.rdata",
    two_channels = "2channels_truthful_exp13.rdata",
    local_eta_two_channels = "leta_2channels_truthful_exp13.rdata"
  ),
  
  "Explicit Deceptive Base Rate" = c(
    learning = "learning_deceptive_exp13.rdata",
    local_eta = "localeta_deceptive_exp13.rdata",
    learning_local_eta = "loo_learning_localeta_deceptive_exp13.rdata",
    two_channels = "2channels_deceptive_exp13.rdata",
    local_eta_two_channels = "leta_2channels_deceptive_exp13.rdata"
  )
)

load_loo_object <- function(path) {
  
  if (!file.exists(path)) {
    stop("Model file not found: ", path)
  }
  
  e <- new.env(parent = emptyenv())
  loaded <- load(path, envir = e)
  
  loo_names <- loaded[
    map_lgl(loaded, ~inherits(e[[.x]], "loo"))
  ]
  
  if (length(loo_names) == 1L) {
    return(e[[loo_names]])
  }
  
  fit_names <- loaded[
    map_lgl(
      loaded,
      ~inherits(e[[.x]], c("CmdStanMCMC", "CmdStanFit"))
    )
  ]
  
  if (length(fit_names) != 1L) {
    stop(
      "Expected one loo object or one CmdStan fit in: ",
      path
    )
  }
  
  fit <- e[[fit_names]]
  
  if (!"log_lik" %in% fit$metadata()$model_params) {
    stop(
      "No saved log_lik found in: ", path,
      "\nLOO requires a pointwise log likelihood for each observation."
    )
  }
  
  log_lik <- fit$draws(
    variables = "log_lik",
    format = "draws_array"
  )
  
  loo::loo(as.array(log_lik))
}

compare_loo_group <- function(files, group_name) {
  
  paths <- file.path(loo_dir, unname(files))
  missing <- !file.exists(paths)
  if (any(missing)) {
    message(group_name, ": skipping missing LOO files: ",
            paste(basename(paths[missing]), collapse = ", "))
  }
  files <- files[!missing]
  if (length(files) < 2L) {
    stop("Fewer than two LOO models available for ", group_name)
  }
  objects <- set_names(
    map(unname(files), ~load_loo_object(file.path(loo_dir, .x))),
    names(files)
  )
  
  n_obs <- map_int(objects, ~nrow(.x$pointwise))
  
  if (n_distinct(n_obs) != 1L) {
    stop(
      "LOO models use different observation counts for ",
      group_name
    )
  }
  
  comparison <- loo_compare(objects) %>%
    as.data.frame() %>%
    rownames_to_column("model") %>%
    as_tibble()
  
  estimates <- imap_dfr(
    objects,
    function(x, model_name) {
      tibble(
        model = model_name,
        elpd_loo = x$estimates["elpd_loo", "Estimate"],
        se_elpd_loo = x$estimates["elpd_loo", "SE"],
        p_loo = x$estimates["p_loo", "Estimate"],
        se_p_loo = x$estimates["p_loo", "SE"],
        looic = x$estimates["looic", "Estimate"],
        se_looic = x$estimates["looic", "SE"]
      )
    }
  )
  
  comparison %>%
    select(model, elpd_diff, se_diff) %>%
    left_join(estimates, by = "model") %>%
    mutate(
      group = group_name,
      n_observations = unique(n_obs),
      .before = 1
    )
}

if (RUN_LOO_COMPARISON) {
  
  loo_summary <- imap_dfr(
    loo_file_map,
    compare_loo_group
  ) %>%
    mutate(
      group = factor(group, levels = group_levels),
      model_label = recode(
        model,
        learning = "Learning",
        local_eta = "Local eta",
        learning_local_eta = "Learning + Local eta",
        two_channels = "2 channels",
        local_eta_two_channels = "Local eta + 2 channels"
      ),
      model_label = factor(
        model_label,
        levels = c(
          "Learning",
          "Local eta",
          "Learning + Local eta",
          "2 channels",
          "Local eta + 2 channels"
        )
      )
    )
  
  best_models <- loo_summary %>%
    group_by(group) %>%
    slice_max(elpd_loo, n = 1, with_ties = FALSE) %>%
    ungroup()
  
  print(loo_summary, n = Inf, width = Inf)
  print(best_models, n = Inf, width = Inf)
  
  # Reference-style downward bars: best model is zero, others negative.
  # SE is the SE of the ELPD difference returned by loo_compare().
  p_loo_diff <- ggplot(
    loo_summary,
    aes(x = model_label, y = elpd_diff, fill = group)
  ) +
    geom_hline(yintercept = 0, linetype = "dashed", linewidth = .7) +
    geom_col(width = .78, colour = "grey30", linewidth = .2) +
    geom_errorbar(
      aes(ymin = elpd_diff - se_diff, ymax = elpd_diff + se_diff),
      width = .16, linewidth = .65, colour = "black"
    ) +
    facet_grid(. ~ group) +
    scale_fill_manual(values = group_cols, guide = "none") +
    scale_y_continuous(expand = expansion(mult = c(.08, .04))) +
    labs(title = "LOO Model Comparison",
         x = "Model", y = expression(Delta*"ELPD (vs best model)")) +
    theme_classic(base_size = 12) +
    theme(
      plot.title = element_text(hjust = .5, face = "bold", size = 16),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 10),
      axis.text.x = element_text(angle = 38, hjust = 1, vjust = 1),
      panel.spacing.x = unit(.7, "lines")
    )
  
  print(p_loo_diff)
}





library(dplyr)

diagnostic_table <- diag_rows %>%
  mutate(
    status = case_when(
      is.na(rhat) ~ "CHECK",
      rhat > 1.01 ~ "CHECK",
      ess_bulk < 400 ~ "CHECK",
      ess_tail < 400 ~ "CHECK",
      TRUE ~ "OK"
    )
  ) %>%
  arrange(group, desc(rhat))

print(
  diagnostic_table,
  n = Inf,
  width = Inf
)

cat("\nPARAMETERS NEEDING ATTENTION:\n")

diagnostic_table %>%
  filter(status == "CHECK") %>%
  print(n = Inf, width = Inf)






# ============================================================================
# DIAGNOSTIC: chain separation in the five instruction-model groups
# Uses the SAVED fitted models. Does not refit Stan or run LOO.
# ============================================================================

suppressPackageStartupMessages({
  library(cmdstanr)
  library(posterior)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
})

# 1. CONFIGURATION -- edit only if the local path differs.
project_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias"
fits_dir <- file.path(project_dir, "stan", "results", "fits", "exp11_unaware")
output_dir <- file.path(project_dir, "stan", "results", "instruction_5groups_fixedDelta", "chain_diagnostics")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

fit_files <- c(
  "Implicit Unaware"    = "learning_leta_2channel_instr_unaware_exp11.rdata",
  "Implicit Aware"      = "learning_leta_2channel_instr_aware_exp11.rdata",
  "Explicit Undirected" = "learning_leta_2channel_instr_aware_exp12.rdata",
  "Explicit True"       = "learning_leta_2channel_instr_truthful_exp13.rdata",
  "Explicit Deceptive"  = "learning_leta_2channel_instr_deceptive_exp13.rdata"
)

pars <- c("mu_beta0", "mu_eta", "mu_alpha0", "mu_delta", "mu_rho")
chain_colours <- c("1"="#d9887f", "2"="#7b9e53", "3"="#329fc1", "4"="#a17cce")

# Optional expensive in-sample log-likelihood check for Exp12 only. NOT LOO.
RUN_EXP12_LOG_LIK_CHECK <- FALSE

all_draws <- list()
all_diagnostics <- list()

for (group_name in names(fit_files)) {
  filename <- file.path(fits_dir, fit_files[[group_name]])
  if (!file.exists(filename)) stop("Missing fit: ", filename)
  message("Reading: ", group_name)
  
  env <- new.env(parent = globalenv())
  load(filename, envir = env)
  if (!exists("fit", envir = env, inherits = FALSE)) {
    stop("The .rdata does not contain an object called 'fit': ", filename)
  }
  fit <- env$fit
  
  d <- fit$draws(variables = pars, format = "draws_df") |>
    as_tibble() |>
    mutate(group = group_name, chain = factor(.chain, levels = 1:4))
  
  all_draws[[group_name]] <- d
  diag <- as_tibble(fit$summary(variables = pars)) |>
    transmute(group = group_name, parameter = variable,
              rhat = rhat, ess_bulk = ess_bulk, ess_tail = ess_tail)
  all_diagnostics[[group_name]] <- diag
  
  if (RUN_EXP12_LOG_LIK_CHECK && group_name == "Explicit Undirected") {
    message("Reading large log_lik array for Exp12: this needs additional RAM")
    ll <- fit$draws(variables = "log_lik", format = "draws_array")
    # ll dimensions: iterations x chains x observations
    total_ll <- apply(ll, c(1, 2), sum)
    per_chain_ll <- tibble(
      group = group_name,
      chain = seq_len(ncol(total_ll)),
      in_sample_log_lik_mean = colMeans(total_ll),
      in_sample_log_lik_sd = apply(total_ll, 2, sd)
    )
    write_csv(per_chain_ll, file.path(output_dir, "undirected_in_sample_loglik_by_chain.csv"))
    rm(ll, total_ll)
  }
  
  rm(env, fit)
  gc(verbose = FALSE)
}

all_d <- bind_rows(all_draws)
all_diag <- bind_rows(all_diagnostics)

# 2. Chain-specific summaries -- helps reveal persistent offsets between chains.
chain_summary <- all_d |>
  group_by(group, chain) |>
  summarise(across(all_of(pars),
                   list(mean = mean, sd = sd), .names = "{.col}_{.fn}"),
            n_draws = n(), .groups = "drop")
write_csv(chain_summary, file.path(output_dir, "chain_specific_parameter_means.csv"))
print(chain_summary, n = Inf, width = Inf)

# 3. Within-chain vs pooled correlations for beta0 and eta.
# A pooled correlation can be misleading if distinct chains form separate clusters.
within_cor <- all_d |>
  group_by(group, chain) |>
  summarise(beta0_eta_cor = cor(mu_beta0, mu_eta), .groups = "drop")
pooled_cor <- all_d |>
  group_by(group) |>
  summarise(pooled_beta0_eta_cor = cor(mu_beta0, mu_eta), .groups = "drop")
write_csv(within_cor, file.path(output_dir, "within_chain_beta0_eta_correlations.csv"))
write_csv(pooled_cor, file.path(output_dir, "pooled_beta0_eta_correlations.csv"))
cat("\nWITHIN-CHAIN CORRELATIONS (beta0 versus eta):\n")
print(within_cor, n = Inf)
cat("\nPOOLED CORRELATIONS (do not use as convergence evidence):\n")
print(pooled_cor, n = Inf)

# 4. Convergence diagnostics for these five parameters in every group.
write_csv(all_diag, file.path(output_dir, "key_parameter_rhat_ess.csv"))
cat("\nR-HAT AND ESS BY GROUP:\n")
print(all_diag, n = Inf)

# 5. Scatterplots per group and combined across all five groups.
make_plot <- function(data, x, y, title) {
  ggplot(data, aes(x = .data[[x]], y = .data[[y]], colour = chain)) +
    geom_point(alpha = 0.22, size = 0.7) +
    scale_colour_manual(values = chain_colours, drop = FALSE) +
    labs(title = title, x = x, y = y, colour = "Chain") +
    theme_bw(base_size = 12) +
    theme(panel.grid.minor = element_blank())
}

for (group_name in names(fit_files)) {
  d <- filter(all_d, group == group_name)
  stem <- gsub("[^a-z0-9]+", "_", tolower(group_name))
  
  p1 <- make_plot(d, "mu_beta0", "mu_eta", paste(group_name, "beta0 vs eta"))
  p2 <- make_plot(d, "mu_beta0", "mu_alpha0", paste(group_name, "beta0 vs alpha0"))
  ggsave(file.path(output_dir, paste0(stem, "_beta0_eta.png")), p1,
         width = 8, height = 6, dpi = 300)
  ggsave(file.path(output_dir, paste0(stem, "_beta0_alpha0.png")), p2,
         width = 8, height = 6, dpi = 300)
  
  # rho does not enter the likelihood for the two uninstructed groups.
  if (!group_name %in% c("Implicit Unaware", "Implicit Aware")) {
    p3 <- make_plot(d, "mu_rho", "mu_delta", paste(group_name, "rho vs delta"))
    ggsave(file.path(output_dir, paste0(stem, "_rho_delta.png")), p3,
           width = 8, height = 6, dpi = 300)
  }
}

p_all <- ggplot(all_d, aes(mu_beta0, mu_eta, colour = chain)) +
  geom_point(alpha = 0.14, size = 0.55) +
  facet_wrap(~ group, ncol = 3, scales = "free") +
  scale_colour_manual(values = chain_colours) +
  labs(title = "beta0 versus eta: five groups, coloured by MCMC chain",
       x = expression(beta[0]), y = expression(eta), colour = "Chain") +
  theme_bw(base_size = 11) +
  theme(strip.text = element_text(face = "bold"), panel.grid.minor = element_blank())
print(p_all)
ggsave(file.path(output_dir, "ALL_five_groups_beta0_eta_by_chain.png"),
       p_all, width = 15, height = 10, dpi = 300)

cat("\nFinished. Results saved in:\n", output_dir, "\n")
cat("No Stan models were refitted. No LOO comparison was performed.\n")







# ============================================================================
# DIAGNOSTIC: chain separation in the five instruction-model groups
# Uses the SAVED fitted models. Does not refit Stan or run LOO.
# ============================================================================

suppressPackageStartupMessages({
  library(cmdstanr)
  library(posterior)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
})

# 1. CONFIGURATION -- edit only if the local path differs.
project_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias"
fits_dir <- file.path(project_dir, "stan", "results", "fits", "exp11_unaware")
output_dir <- file.path(project_dir, "stan", "results", "instruction_5groups_fixedDelta", "chain_diagnostics")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

fit_files <- c(
  "Implicit Unaware"    = "learning_leta_2channel_instr_unaware_exp11.rdata",
  "Implicit Aware"      = "learning_leta_2channel_instr_aware_exp11.rdata",
  "Explicit Undirected" = "learning_leta_2channel_instr_aware_exp12.rdata",
  "Explicit True"       = "learning_leta_2channel_instr_truthful_exp13.rdata",
  "Explicit Deceptive"  = "learning_leta_2channel_instr_deceptive_exp13.rdata"
)

pars <- c("mu_beta0", "mu_eta", "mu_alpha0", "mu_delta", "mu_rho")
chain_colours <- c("1"="#d9887f", "2"="#7b9e53", "3"="#329fc1", "4"="#a17cce")

# Optional expensive in-sample log-likelihood check for Exp12 only. NOT LOO.
RUN_EXP12_LOG_LIK_CHECK <- FALSE

all_draws <- list()
all_diagnostics <- list()

for (group_name in names(fit_files)) {
  filename <- file.path(fits_dir, fit_files[[group_name]])
  if (!file.exists(filename)) stop("Missing fit: ", filename)
  message("Reading: ", group_name)
  
  env <- new.env(parent = globalenv())
  load(filename, envir = env)
  if (!exists("fit", envir = env, inherits = FALSE)) {
    stop("The .rdata does not contain an object called 'fit': ", filename)
  }
  fit <- env$fit
  
  d <- fit$draws(variables = pars, format = "draws_df") |>
    as_tibble() |>
    mutate(group = group_name, chain = factor(.chain, levels = 1:4))
  
  all_draws[[group_name]] <- d
  diag <- as_tibble(fit$summary(variables = pars)) |>
    transmute(group = group_name, parameter = variable,
              rhat = rhat, ess_bulk = ess_bulk, ess_tail = ess_tail)
  all_diagnostics[[group_name]] <- diag
  
  if (RUN_EXP12_LOG_LIK_CHECK && group_name == "Explicit Undirected") {
    message("Reading large log_lik array for Exp12: this needs additional RAM")
    ll <- fit$draws(variables = "log_lik", format = "draws_array")
    # ll dimensions: iterations x chains x observations
    total_ll <- apply(ll, c(1, 2), sum)
    per_chain_ll <- tibble(
      group = group_name,
      chain = seq_len(ncol(total_ll)),
      in_sample_log_lik_mean = colMeans(total_ll),
      in_sample_log_lik_sd = apply(total_ll, 2, sd)
    )
    write_csv(per_chain_ll, file.path(output_dir, "undirected_in_sample_loglik_by_chain.csv"))
    rm(ll, total_ll)
  }
  
  rm(env, fit)
  gc(verbose = FALSE)
}

all_d <- bind_rows(all_draws)
all_diag <- bind_rows(all_diagnostics)

# 2. Chain-specific summaries -- helps reveal persistent offsets between chains.
chain_summary <- all_d |>
  group_by(group, chain) |>
  summarise(across(all_of(pars),
                   list(mean = mean, sd = sd), .names = "{.col}_{.fn}"),
            n_draws = n(), .groups = "drop")
write_csv(chain_summary, file.path(output_dir, "chain_specific_parameter_means.csv"))
print(chain_summary, n = Inf, width = Inf)

# 3. Within-chain vs pooled correlations for beta0 and eta.
# A pooled correlation can be misleading if distinct chains form separate clusters.
within_cor <- all_d |>
  group_by(group, chain) |>
  summarise(beta0_eta_cor = cor(mu_beta0, mu_eta), .groups = "drop")
pooled_cor <- all_d |>
  group_by(group) |>
  summarise(pooled_beta0_eta_cor = cor(mu_beta0, mu_eta), .groups = "drop")
write_csv(within_cor, file.path(output_dir, "within_chain_beta0_eta_correlations.csv"))
write_csv(pooled_cor, file.path(output_dir, "pooled_beta0_eta_correlations.csv"))
cat("\nWITHIN-CHAIN CORRELATIONS (beta0 versus eta):\n")
print(within_cor, n = Inf)
cat("\nPOOLED CORRELATIONS (do not use as convergence evidence):\n")
print(pooled_cor, n = Inf)

# 4. Convergence diagnostics for these five parameters in every group.
write_csv(all_diag, file.path(output_dir, "key_parameter_rhat_ess.csv"))
cat("\nR-HAT AND ESS BY GROUP:\n")
print(all_diag, n = Inf)

# 5. Scatterplots per group and combined across all five groups.
make_plot <- function(data, x, y, title) {
  ggplot(data, aes(x = .data[[x]], y = .data[[y]], colour = chain)) +
    geom_point(alpha = 0.22, size = 0.7) +
    scale_colour_manual(values = chain_colours, drop = FALSE) +
    labs(title = title, x = x, y = y, colour = "Chain") +
    theme_bw(base_size = 12) +
    theme(panel.grid.minor = element_blank())
}

for (group_name in names(fit_files)) {
  d <- filter(all_d, group == group_name)
  stem <- gsub("[^a-z0-9]+", "_", tolower(group_name))
  
  p1 <- make_plot(d, "mu_beta0", "mu_eta", paste(group_name, "beta0 vs eta"))
  p2 <- make_plot(d, "mu_beta0", "mu_alpha0", paste(group_name, "beta0 vs alpha0"))
  ggsave(file.path(output_dir, paste0(stem, "_beta0_eta.png")), p1,
         width = 8, height = 6, dpi = 300)
  ggsave(file.path(output_dir, paste0(stem, "_beta0_alpha0.png")), p2,
         width = 8, height = 6, dpi = 300)
  
  # rho does not enter the likelihood for the two uninstructed groups.
  if (!group_name %in% c("Implicit Unaware", "Implicit Aware")) {
    p3 <- make_plot(d, "mu_rho", "mu_delta", paste(group_name, "rho vs delta"))
    ggsave(file.path(output_dir, paste0(stem, "_rho_delta.png")), p3,
           width = 8, height = 6, dpi = 300)
  }
}

p_all <- ggplot(all_d, aes(mu_beta0, mu_eta, colour = chain)) +
  geom_point(alpha = 0.14, size = 0.55) +
  facet_wrap(~ group, ncol = 3, scales = "free") +
  scale_colour_manual(values = chain_colours) +
  labs(title = "beta0 versus eta: five groups, coloured by MCMC chain",
       x = expression(beta[0]), y = expression(eta), colour = "Chain") +
  theme_bw(base_size = 11) +
  theme(strip.text = element_text(face = "bold"), panel.grid.minor = element_blank())
print(p_all)
ggsave(file.path(output_dir, "ALL_five_groups_beta0_eta_by_chain.png"),
       p_all, width = 15, height = 10, dpi = 300)

cat("\nFinished. Results saved in:\n", output_dir, "\n")
cat("No Stan models were refitted. No LOO comparison was performed.\n")



















library(cmdstanr)
library(posterior)
library(dplyr)
library(ggplot2)

# Location of your five fitted instruction models
fits_dir <- paste0(
  "/Users/bty615/Documents/GitHub/reliable_info_bias/",
  "stan/results/fits/exp11_unaware"
)

file <- file.path(
  fits_dir,
  "learning_leta_2channel_instr_aware_exp12.rdata"
)

# Load the fitted model
e <- new.env(parent = globalenv())
load(file, envir = e)
fit <- e$fit

# Only load the five relevant group parameters
pars <- c(
  "mu_beta0",
  "mu_eta",
  "mu_alpha0",
  "mu_delta",
  "mu_rho"
)

d <- fit$draws(
  variables = pars,
  format = "draws_df"
) |>
  as_tibble()

d$chain <- factor(d$.chain)

# --------------------------------------------------
# 1. Chain-specific parameter means
# --------------------------------------------------

chain_means <- d |>
  group_by(chain) |>
  summarise(
    across(all_of(pars), mean),
    .groups = "drop"
  )

print(chain_means, width = Inf)

# --------------------------------------------------
# 2. beta0 versus eta
# --------------------------------------------------

p1 <- ggplot(
  d,
  aes(
    x = mu_beta0,
    y = mu_eta,
    colour = chain
  )
) +
  geom_point(alpha = 0.20, size = 0.65) +
  theme_bw(base_size = 13) +
  labs(
    title = "Undirected: beta0 versus eta",
    x = expression(beta[0]),
    y = expression(eta),
    colour = "Chain"
  )

print(p1)

# --------------------------------------------------
# 3. beta0 versus alpha0
# --------------------------------------------------

p2 <- ggplot(
  d,
  aes(
    x = mu_beta0,
    y = mu_alpha0,
    colour = chain
  )
) +
  geom_point(alpha = 0.20, size = 0.65) +
  theme_bw(base_size = 13) +
  labs(
    title = "Undirected: beta0 versus alpha0",
    x = expression(beta[0]),
    y = expression(alpha[0]),
    colour = "Chain"
  )

print(p2)

# --------------------------------------------------
# 4. rho versus delta
# --------------------------------------------------

p3 <- ggplot(
  d,
  aes(
    x = mu_rho,
    y = mu_delta,
    colour = chain
  )
) +
  geom_point(alpha = 0.20, size = 0.65) +
  theme_bw(base_size = 13) +
  labs(
    title = "Undirected: rho versus delta",
    x = expression(rho),
    y = expression(delta),
    colour = "Chain"
  )

print(p3)

# Save the figures
output_dir <- file.path(
  dirname(fits_dir),
  "undirected_diagnostics"
)

dir.create(
  output_dir,
  recursive = TRUE,
  showWarnings = FALSE
)

ggsave(file.path(output_dir, "beta0_eta.png"),
       p1, width = 8, height = 6, dpi = 300)

ggsave(file.path(output_dir, "beta0_alpha0.png"),
       p2, width = 8, height = 6, dpi = 300)

ggsave(file.path(output_dir, "rho_delta.png"),
       p3, width = 8, height = 6, dpi = 300)







library(bayesplot)
library(ggplot2)

fits_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias/stan/results/fits/exp11_unaware"

e <- new.env(parent = globalenv())

load(
  file.path(
    fits_dir,
    "learning_leta_2channel_instr_aware_exp12.rdata"
  ),
  envir = e
)

fit <- e$fit

parameters <- c(
  "mu_beta0",
  "mu_eta",
  "mu_alpha0",
  "mu_delta",
  "mu_rho"
)

draws <- fit$draws(
  variables = parameters,
  format = "draws_array"
)

p <- mcmc_trace(as.array(draws))

print(p)

ggsave(
  "undirected_problem_parameters_trace.png",
  p,
  width = 14,
  height = 10,
  dpi = 300
)





# ============================================================================
# INSTRUCTION MODEL ANALYSIS — FIVE BASE-RATE GROUPS
# Uses the model learning_leta_2channel_instr.stan
# Undirected Exp12: Delta FIXED at 0.10 (NOT an estimated posterior)
# All five .rdata fit files are in ONE folder.
# This script reads fits; it does not run MCMC or overwrite existing fits.
# ============================================================================
rm(list = ls(all.names = TRUE))
suppressPackageStartupMessages({
  library(tidyverse)
  library(posterior)
  library(cmdstanr)
})

# 1. FILE PATHS — adjust this block only if your folder is elsewhere
project_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias"
fits_dir <- file.path(project_dir, "stan", "results", "fits", "exp11_unaware")
results_dir <- file.path(project_dir, "stan", "results", "instruction_5groups_fixedDelta")
figure_dir <- file.path(results_dir, "figures")
table_dir <- file.path(results_dir, "tables")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

fit_files <- c(
  "Implicit Unaware" = "learning_leta_2channel_instr_unaware_exp11.rdata",
  "Implicit Aware" = "learning_leta_2channel_instr_aware_exp11.rdata",
  "Explicit Undirected" = "learning_leta_2channel_instr_aware_exp12.rdata",
  "Explicit True" = "learning_leta_2channel_instr_truthful_exp13.rdata",
  "Explicit Deceptive" = "learning_leta_2channel_instr_deceptive_exp13.rdata"
)
missing_files <- file.path(fits_dir, fit_files)[!file.exists(file.path(fits_dir, fit_files))]
if (length(missing_files)) stop("Missing fit files:\n", paste(missing_files, collapse = "\n"),
                                "\nUpdate fits_dir or fit_files at the top of this script.")

group_cols <- c("Implicit Unaware" = "#F4A3A3", "Implicit Aware" = "#8FD694",
                "Explicit Undirected" = "#E69F00", "Explicit True" = "#1B7837",
                "Explicit Deceptive" = "#8B0000")
param_labels <- c(mu_alpha="alpha", mu_beta="beta", mu_lambda="lambda", mu_delta="delta",
                  mu_eta="eta", mu_alpha0="alpha[0]", mu_beta0="beta[0]", mu_rho="rho")
param_order <- names(param_labels)
reference <- c(mu_alpha=1, mu_beta=0, mu_lambda=0, mu_delta=1,
               mu_eta=0, mu_alpha0=1, mu_beta0=0, mu_rho=0)

save_plot <- function(p, stem, w=12, h=7) {
  print(p)
  ggsave(file.path(figure_dir, paste0(stem, ".png")), p, width=w, height=h, dpi=300)
  ggsave(file.path(figure_dir, paste0(stem, ".pdf")), p, width=w, height=h)
}
plot_theme <- theme_bw(base_size=13) + theme(panel.grid.minor=element_blank(),
                                             strip.text=element_text(face="bold"), legend.position="bottom", legend.title=element_blank())

# 2. LOAD GROUP-LEVEL GENERATED PARAMETERS
# Only these 8 named quantities are loaded, avoiding very large log_lik/V_B arrays.
read_group <- function(filename, group_name) {
  env <- new.env(parent=globalenv())
  load(file.path(fits_dir, filename), envir=env)
  if (!exists("fit", envir=env, inherits=FALSE)) stop("No 'fit' object in ", filename)
  fit <- env$fit
  available <- fit$metadata()$model_params
  missing <- setdiff(param_order, available)
  if (length(missing)) stop("Missing parameters in ", filename, ": ", paste(missing, collapse=", "))
  d <- fit$draws(variables=param_order, format="draws_df") |> as_tibble()
  if (!all(is.finite(as.matrix(d[, param_order])))) stop("Non-finite group draws in ", filename)
  cat(sprintf("Loaded %-22s: %d posterior draws\n", group_name, nrow(d)))
  d |> select(all_of(param_order)) |>
    mutate(group=group_name, draw_id=row_number(), .before=1)
}

draws <- purrr::imap_dfr(fit_files, read_group) |>
  mutate(group=factor(group, levels=names(fit_files)))

# rho in instr=4 (no instruction) does not enter the likelihood: do not interpret it.
draws_long <- draws |>
  pivot_longer(cols=all_of(param_order), names_to="parameter", values_to="value") |>
  filter(!(parameter == "mu_rho" & group %in% c("Implicit Unaware", "Implicit Aware"))) |>
  mutate(parameter=factor(parameter, levels=param_order),
         parameter_label=unname(param_labels[as.character(parameter)]))

# 3. PARAMETER POSTERIORS: densities and box plots
p_density <- ggplot(draws_long, aes(value, colour=group, fill=group)) +
  geom_density(alpha=.18, linewidth=.65) +
  facet_wrap(~parameter, scales="free", ncol=4,
             labeller=as_labeller(param_labels, default=label_parsed)) +
  scale_colour_manual(values=group_cols) + scale_fill_manual(values=group_cols) +
  labs(title="Instruction-model group parameter posteriors", x="Parameter value", y="Density") + plot_theme
save_plot(p_density, "01_group_parameter_posteriors", 15, 9)

p_boxes <- ggplot(draws_long, aes(group, value, fill=group)) +
  geom_boxplot(outlier.shape=NA, width=.65) +
  facet_wrap(~parameter, scales="free_y", ncol=4,
             labeller=as_labeller(param_labels, default=label_parsed)) +
  scale_fill_manual(values=group_cols) +
  scale_x_discrete(labels=function(x) stringr::str_wrap(x, width=12)) +
  labs(title="Group-level posterior distributions (boxes summarise draws)", x=NULL, y="Parameter value") +
  plot_theme + theme(legend.position="none", axis.text.x=element_text(size=7, angle=35, hjust=1))
save_plot(p_boxes, "02_group_parameter_boxplots", 16, 10)

# 4. DISTORTION CURVES, transforming PAIRED posterior draws
# Sample evidence: logit(pi_subjective) = alpha logit(pi_true) + beta
# Prior channel: logit(V_subjective) = alpha0 logit(V_B) + beta0
make_curve <- function(alpha_col, beta_col, values) {
  map_dfr(levels(draws$group), function(g) {
    d <- filter(draws, group==g)
    map_dfr(values, function(x) {
      yy <- 100 * plogis(d[[alpha_col]] * qlogis(x/100) + d[[beta_col]])
      tibble(group=g, x=x, mean=mean(yy), lower=unname(quantile(yy,.025)),
             upper=unname(quantile(yy,.975)))
    })
  }) |> mutate(group=factor(group, levels=names(fit_files)))
}
x_grid <- sort(unique(c(seq(.1,99.9,length.out=301),50,55,65)))
evidence_curve <- make_curve("mu_alpha", "mu_beta", x_grid)
prior_curve <- make_curve("mu_alpha0", "mu_beta0", x_grid)
plot_curve <- function(dd, title, xlabel) {
  ggplot(dd, aes(x, mean, colour=group, fill=group)) +
    geom_abline(slope=1, intercept=0, linetype="dashed", colour="black") +
    geom_ribbon(aes(ymin=lower,ymax=upper), alpha=.12, colour=NA) +
    geom_line(linewidth=1.1) +
    scale_colour_manual(values=group_cols) + scale_fill_manual(values=group_cols) +
    coord_fixed(xlim=c(0,100), ylim=c(0,100)) +
    labs(title=title, subtitle="Posterior mean and pointwise 95% credible interval",
         x=xlabel, y="Transformed probability (%)") + plot_theme
}
p_ev <- plot_curve(evidence_curve, "Evidence reliability distortion", "Presented reliability (%)") +
  geom_point(data=filter(evidence_curve, x %in% c(50,55,65)), size=2)
p_pr <- plot_curve(prior_curve, "Prior-belief distortion", "Belief V_B before decision-channel transformation (%)")
save_plot(p_ev, "03_evidence_distortion", 9, 8)
save_plot(p_pr, "04_prior_distortion", 9, 8)
write_csv(filter(evidence_curve, x %in% c(50,55,65)), file.path(table_dir,"evidence_reliability_50_55_65.csv"))
write_csv(prior_curve, file.path(table_dir,"prior_distortion_curve.csv"))

# 5. SIX-POSITION RECENCY WEIGHTS (relative to final sample)
weights <- map_dfr(levels(draws$group), function(g) {
  d <- filter(draws, group==g)
  map_dfr(1:6, function(s) {
    v <- exp(d$mu_lambda * (s-6))
    tibble(group=g, position=s, mean=mean(v), lower=unname(quantile(v,.025)),
           upper=unname(quantile(v,.975)))
  })
}) |> mutate(group=factor(group, levels=names(fit_files)))
p_weights <- ggplot(weights, aes(position, mean, colour=group, fill=group)) +
  geom_ribbon(aes(ymin=lower,ymax=upper), alpha=.12, colour=NA) +
  geom_line(linewidth=1) + geom_point(size=2) +
  scale_x_continuous(breaks=1:6) + scale_colour_manual(values=group_cols) +
  scale_fill_manual(values=group_cols) +
  labs(title="Sequential-position weighting", x="Sample position",
       y="Weight relative to sample 6") + plot_theme
save_plot(p_weights, "05_sequential_position_weights", 10, 7)
write_csv(weights, file.path(table_dir, "sequential_position_weights.csv"))

# 6. SUMMARY TABLES
summaries <- draws_long |>
  group_by(group, parameter, parameter_label) |>
  summarise(mean=mean(value), median=median(value), sd=sd(value),
            lower95=unname(quantile(value,.025)), upper95=unname(quantile(value,.975)),
            .groups="drop")
write_csv(summaries, file.path(table_dir, "group_posterior_summaries.csv"))
writeup <- summaries |>
  mutate(result=sprintf("%.3f [%.3f, %.3f]",mean,lower95,upper95)) |>
  select(group, parameter_label, result) |>
  pivot_wider(names_from=parameter_label, values_from=result)
write_csv(writeup, file.path(table_dir, "group_posterior_writeup.csv"))
cat("\nGROUP-LEVEL POSTERIOR SUMMARIES:\n")
print(summaries, n=Inf)

ref_stats <- draws_long |>
  mutate(reference=unname(reference[as.character(parameter)])) |>
  group_by(group, parameter, reference) |>
  summarise(prob_above=mean(value>reference), prob_below=mean(value<reference),
            lower95=unname(quantile(value,.025)), upper95=unname(quantile(value,.975)),
            .groups="drop")
write_csv(ref_stats, file.path(table_dir, "posterior_reference_probabilities.csv"))

# 7. INDEPENDENT-GROUP PAIRWISE CONTRASTS
# Draws are independently fitted; randomly pair draws from different groups.
set.seed(12345)
pairs <- combn(names(fit_files), 2, simplify=FALSE)
contrasts <- map_dfr(param_order, function(par) {
  map_dfr(pairs, function(pair) {
    a <- draws_long |> filter(group==pair[1], parameter==par) |> pull(value)
    b <- draws_long |> filter(group==pair[2], parameter==par) |> pull(value)
    if (!length(a) || !length(b)) return(tibble())
    n <- min(length(a),length(b))
    diff <- sample(a,n) - sample(b,n)
    tibble(parameter=par, group1=pair[1], group2=pair[2],
           mean_difference=mean(diff), lower95=unname(quantile(diff,.025)),
           upper95=unname(quantile(diff,.975)),
           p_group1_greater=mean(diff>0))
  })
})
write_csv(contrasts, file.path(table_dir, "pairwise_group_posterior_contrasts.csv"))

# 8. UNDIRECTED RHO–DELTA ASSOCIATION (group mean draws, NOT individual parameters)
u <- filter(draws, group=="Explicit Undirected")
association <- cor(u$mu_rho, u$mu_delta)
write_csv(tibble(group="Explicit Undirected", delta_fixed=.10,
                 rho_delta_posterior_correlation=association),
          file.path(table_dir, "undirected_rho_delta_association.csv"))
p_corr <- ggplot(u, aes(mu_rho,mu_delta)) +
  geom_point(alpha=.08, size=.65, colour="#E69F00") +
  geom_density_2d(colour="#9B6700", linewidth=.45) +
  labs(title="Undirected: posterior association of group-level rho and delta",
       subtitle=paste0("Delta fixed at 0.10; correlation = ", round(association,3)),
       x="Group-level rho", y="Group-level delta") + plot_theme
save_plot(p_corr, "06_undirected_rho_delta_posterior_association", 8, 6)

# 9. SAMPLER DIAGNOSTICS (group-level variables only)
diag_rows <- purrr::imap_dfr(fit_files, function(filename, group_name) {
  env <- new.env(parent=globalenv())
  load(file.path(fits_dir,filename), envir=env)
  s <- env$fit$summary(variables=param_order)
  tibble(group=group_name, parameter=s$variable, rhat=s$rhat,
         ess_bulk=s$ess_bulk, ess_tail=s$ess_tail)
})
write_csv(diag_rows, file.path(table_dir, "sampler_diagnostics_group_parameters.csv"))
cat("\nMaximum R-hat by group (check all >1.01):\n")
print(diag_rows |> group_by(group) |> summarise(max_rhat=max(rhat,na.rm=TRUE),
                                                min_bulk_ess=min(ess_bulk,na.rm=TRUE),
                                                .groups="drop"))

cat("\nFixed Delta for Exp12 = 0.10 (NOT estimated).\n")
cat("Do not interpret rho for uninstructed groups (instr=4).\n")
cat("Posterior prior distortion uses alpha0/beta0 on V_B, not on the initial prior distribution itself.\n")
cat("LOO scores from different participant groups are not directly comparable.\n")
cat("\nFinished. Figures: ",figure_dir,"\nTables: ",table_dir,"\n",sep="")

























suppressPackageStartupMessages({
  library(tidyverse)
  library(posterior)
  library(cmdstanr)
})

# ============================================================
# 1. PATHS AND FILES
# ============================================================

project_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias"

# Same folder as your original plotting script
fits_dir <- file.path(
  project_dir, "stan", "results", "fits", "exp11_unaware"
)

figure_dir <- file.path(
  project_dir, "stan", "results", "figures", "instruction_model"
)

dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

fit_files <- c(
  "Implicit Unaware" =
    "learning_leta_2channel_instr_unaware_exp11.rdata",
  "Implicit Aware" =
    "learning_leta_2channel_instr_aware_exp11.rdata",
  "Explicit Undirected" =
    "learning_leta_2channel_instr_aware_exp12.rdata",
  "Explicit True" =
    "learning_leta_2channel_instr_truthful_exp13.rdata",
  "Explicit Deceptive" =
    "learning_leta_2channel_instr_deceptive_exp13.rdata"
)

group_cols <- c(
  "Implicit Unaware"    = "#F4A3A3",
  "Implicit Aware"      = "#8FD694",
  "Explicit Undirected" = "#E69F00",
  "Explicit True"       = "#1B7837",
  "Explicit Deceptive"  = "#8B0000"
)

parameter_labels <- c(
  mu_alpha  = "alpha",
  mu_beta   = "beta",
  mu_lambda = "lambda",
  mu_delta  = "delta",
  mu_eta    = "eta",
  mu_alpha0 = "alpha[0]",
  mu_beta0  = "beta[0]",
  mu_rho    = "rho",
  Delta     = "Delta"
)

plot_theme <- theme_bw(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(size = 13, face = "bold"),
    legend.position = "bottom",
    legend.title = element_blank(),
    plot.title = element_text(face = "bold")
  )

save_plot <- function(p, filename, width = 12, height = 8) {
  print(p)
  
  for (extension in c("png", "pdf")) {
    ggsave(
      file.path(figure_dir, paste0(filename, ".", extension)),
      plot = p,
      width = width,
      height = height,
      dpi = 300
    )
  }
}

# ============================================================
# 2. LOAD GROUP-LEVEL POSTERIOR DRAWS
# ============================================================

available <- file.exists(file.path(fits_dir, fit_files))

if (any(!available)) {
  warning(
    "Skipping missing files:\n",
    paste(fit_files[!available], collapse = "\n")
  )
}

fit_files <- fit_files[available]

if (!length(fit_files)) {
  stop("No fit files found. Check fits_dir and fit_files.")
}

draws <- imap_dfr(fit_files, function(filename, group_name) {
  
  e <- new.env()
  load(file.path(fits_dir, filename), envir = e)
  
  if (!exists("fit", envir = e, inherits = FALSE)) {
    stop("No object named 'fit' in: ", filename)
  }
  
  # Use the transformed values saved by the supplied Stan model.
  # No filename-based parameter guessing or raw transformations.
  d <- e$fit$draws(
    variables = names(parameter_labels),
    format = "draws_df"
  ) %>%
    as_tibble() %>%
    select(all_of(names(parameter_labels)))
  
  if (any(!is.finite(as.matrix(d)))) {
    stop("Non-finite parameter draws in: ", filename)
  }
  
  d %>%
    mutate(
      group = group_name,
      draw_id = row_number()
    )
}) %>%
  mutate(group = factor(group, levels = names(fit_files)))

draws_long <- draws %>%
  pivot_longer(
    all_of(names(parameter_labels)),
    names_to = "parameter",
    values_to = "value"
  ) %>%
  # rho does not enter the likelihood for no-instruction groups.
  filter(
    !(parameter == "mu_rho" &
        group %in% c("Implicit Unaware", "Implicit Aware")),
    parameter != "Delta" | group == "Explicit Undirected"
  ) %>%
  mutate(
    parameter = factor(
      parameter,
      levels = names(parameter_labels)
    )
  )

parameter_labeller <- as_labeller(
  parameter_labels,
  default = label_parsed
)

# ============================================================
# 3. PARAMETER POSTERIOR DISTRIBUTIONS
# ============================================================

p_parameters <- ggplot(
  draws_long,
  aes(x = value, fill = group, colour = group)
) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 45,
    position = "identity",
    alpha = 0.30,
    linewidth = 0.2
  ) +
  facet_wrap(
    ~ parameter,
    scales = "free",
    ncol = 3,
    labeller = parameter_labeller
  ) +
  scale_fill_manual(values = group_cols) +
  scale_colour_manual(values = group_cols) +
  labs(
    title = "Group-level parameter posteriors",
    x = "Parameter value",
    y = "Density"
  ) +
  plot_theme

save_plot(p_parameters, "parameter_posteriors", 12, 10)

# ============================================================
# 4. POSTERIOR BOX PLOTS
# ============================================================

# These boxes describe posterior draws, NOT participant variation.
# Boxes = median and interquartile range.
# Whiskers use the usual 1.5 × IQR rule, not a 95% interval.

p_boxes <- ggplot(
  draws_long,
  aes(x = group, y = value, fill = group)
) +
  geom_boxplot(
    width = 0.65,
    outlier.shape = NA,
    linewidth = 0.4
  ) +
  facet_wrap(
    ~ parameter,
    scales = "free",
    ncol = 3,
    labeller = parameter_labeller
  ) +
  scale_fill_manual(values = group_cols) +
  scale_x_discrete(labels = function(x) stringr::str_wrap(x, 12)) +
  labs(
    title = "Group-level posterior box plots",
    x = NULL,
    y = "Posterior value"
  ) +
  plot_theme +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 8)
  )

save_plot(p_boxes, "parameter_boxplots", 12, 10)

# ============================================================
# 5. DISTORTION CURVES
# ============================================================

# Transform each paired posterior draw, then summarise.
# Avoid exactly 0 and 1 because their logits are infinite.

make_curve <- function(alpha_name, beta_name, x_values) {
  
  map_dfr(levels(draws$group), function(g) {
    
    d <- filter(draws, group == g)
    
    map_dfr(x_values, function(x) {
      
      y <- 100 * plogis(
        d[[alpha_name]] * qlogis(x / 100) +
          d[[beta_name]]
      )
      
      tibble(
        group = g,
        x = x,
        mean = mean(y),
        lower = unname(quantile(y, 0.025)),
        upper = unname(quantile(y, 0.975))
      )
    })
  }) %>%
    mutate(group = factor(group, levels = levels(draws$group)))
}

x_grid <- sort(unique(c(
  seq(0.1, 99.9, length.out = 301),
  50, 55, 65
)))

evidence_curve <- make_curve("mu_alpha", "mu_beta", x_grid)
prior_curve <- make_curve("mu_alpha0", "mu_beta0", x_grid)

plot_curve <- function(d, title, x_label, y_label) {
  
  ggplot(d, aes(x = x, y = mean, colour = group, fill = group)) +
    geom_abline(
      slope = 1,
      intercept = 0,
      linetype = "dashed",
      colour = "black"
    ) +
    geom_ribbon(
      aes(ymin = lower, ymax = upper),
      alpha = 0.12,
      colour = NA
    ) +
    geom_line(linewidth = 1) +
    scale_colour_manual(values = group_cols) +
    scale_fill_manual(values = group_cols) +
    scale_x_continuous(breaks = seq(0, 100, 25)) +
    scale_y_continuous(breaks = seq(0, 100, 25)) +
    coord_fixed(xlim = c(0, 100), ylim = c(0, 100)) +
    labs(
      title = title,
      subtitle = "Posterior mean and pointwise 95% credible interval",
      x = x_label,
      y = y_label
    ) +
    plot_theme +
    guides(
      colour = guide_legend(ncol = 2),
      fill = guide_legend(ncol = 2)
    )
}

p_evidence <- plot_curve(
  evidence_curve,
  "Evidence reliability distortion",
  "Presented reliability (%)",
  "Transformed reliability (%)"
) +
  geom_point(
    data = filter(evidence_curve, x %in% c(50, 55, 65)),
    size = 2
  )

p_prior <- plot_curve(
  prior_curve,
  "Prior belief distortion",
  expression("Belief before transformation: " * V[B] * " (%)"),
  "Transformed prior (%)"
)

save_plot(p_evidence, "evidence_distortion", 8, 8)
save_plot(p_prior, "prior_distortion", 8, 8)

cat("\nFigures saved to:\n", figure_dir, "\n")































# ==============================================================================
# ELPD-LOO: SEVEN DISTINCT MODEL ENTRIES, ONE GRAPH PER GROUP
# ==============================================================================
# IMPORTANT: These are SEPARATE entries, NEVER aliases of one another:
#   (6) learning_leta_2channel_instr
#   (7) learning_2channel_instr
#
# Exp12 must have all seven loaded and comparable. Otherwise stop with an
# explicit diagnostic, rather than quietly plotting six bars.
# The unsuffixed learning_2channel_instr.rdata is an Exp12 fit ONLY.
# Never reuse that file for the other groups.
#
# Requires: install.packages(c("tidyverse", "loo"))
# ============================================================================

rm(list = ls(all.names = TRUE))
suppressPackageStartupMessages({
  library(tidyverse)
  library(loo)
})

# 1. PATHS --------------------------------------------------------------------
project_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias"
stan_dir <- file.path(project_dir, "stan")
loo_dir <- file.path(stan_dir, "results", "loo", "exp11_unaware")
output_dir <- file.path(stan_dir, "results", "loo_comparison_SEVEN_models")
plot_dir <- file.path(output_dir, "figures")
table_dir <- file.path(output_dir, "tables")

STOP_IF_EXP12_NOT_SEVEN <- TRUE

if (!dir.exists(loo_dir)) stop("LOO folder does not exist: ", loo_dir)
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

# 2. GROUPS AND ALL SEVEN MODEL NAMES -----------------------------------------
suffixes <- c(
  "Implicit Unaware" = "unaware_exp11",
  "Implicit Aware" = "aware_exp11",
  "Explicit Undirected" = "aware_exp12",
  "Explicit Truthful" = "truthful_exp13",
  "Explicit Deceptive" = "deceptive_exp13"
)

models <- c(
  "learning",
  "localeta",
  "learning_localeta",
  "2channels",
  "leta_2channels",
  "learning_leta_2channel_instr",   # MODEL SIX -- retain separately
  "learning_2channel_instr"         # MODEL SEVEN -- NEW, separate fit
)
stopifnot(length(models) == 7L, !anyDuplicated(models))

axis_labels <- c(
  "learning" = "Learning",
  "localeta" = "Local\neta",
  "learning_localeta" = "Learning +\nlocal eta",
  "2channels" = "2\nchannels",
  "leta_2channels" = "Local eta +\n2 channels",
  "learning_leta_2channel_instr" = "Learning + local eta\n2 channels + instr",
  "learning_2channel_instr" = "Learning + 2 channels\n+ instr (new)"
)

group_colors <- c(
  "Implicit Unaware" = "#F4A3A3",
  "Implicit Aware" = "#8FD694",
  "Explicit Undirected" = "#E69F00",
  "Explicit Truthful" = "#1B7837",
  "Explicit Deceptive" = "#8B0000"
)

# 3. EXACT FILE MATCHING ------------------------------------------------------
# Candidates in priority order. A leading "loo_" is also tried.
# The Exp12 model 7 uses the exact new name learning_2channel_instr.rdata.
file_candidates <- function(group, model) {
  suffix <- suffixes[[group]]
  if (model == "learning_localeta") {
    return(c(paste0("loo_learning_localeta_", suffix, ".rdata"),
             paste0("learning_localeta_", suffix, ".rdata")))
  }
  if (model == "learning_leta_2channel_instr") {
    if (group == "Explicit Undirected") {
      return(c(
        paste0(model, "_", suffix, "_fixedDelta.rdata"),
        paste0(model, "_", suffix, ".rdata"),
        paste0(model, ".rdata")
      ))
    }
    return(paste0(model, "_", suffix, ".rdata"))
  }
  if (model == "learning_2channel_instr") {
    if (group == "Explicit Undirected") {
      return(c("learning_2channel_instr.rdata",
               paste0(model, "_", suffix, ".rdata"),
               "learning_2channel_instr_exp12_undirected.rdata"))
    }
    # Only group-specific copies for other groups: never reuse the Exp12 fit.
    return(paste0(model, "_", suffix, ".rdata"))
  }
  paste0(model, "_", suffix, ".rdata")
}

# Case-insensitive file matching handles .RData / .rdata on macOS.
files_in_folder <- list.files(loo_dir, full.names = FALSE)
find_file <- function(candidates) {
  more <- paste0("loo_", candidates)
  for (wanted in unique(c(candidates, more))) {
    hits <- files_in_folder[tolower(files_in_folder) == tolower(wanted)]
    if (length(hits) == 1L) return(hits)
    if (length(hits) > 1L) stop("Ambiguous filename: ", wanted)
  }
  NA_character_
}

inventory <- purrr::map_dfr(names(suffixes), function(g) {
  purrr::map_dfr(models, function(m) {
    wanted <- file_candidates(g, m)
    actual <- find_file(wanted)
    tibble(group = g, model = m,
           filename = actual,
           found = !is.na(actual),
           expected_first = wanted[[1]])
  })
})
write_csv(inventory, file.path(table_dir, "01_file_inventory.csv"))
cat("\nSEVEN MODEL SLOTS PER GROUP:\n")
print(inventory, n = Inf, width = Inf)

# 4. LOAD SAVED LOO OBJECTS ---------------------------------------------------
read_loo <- function(path) {
  e <- new.env(parent = globalenv())
  names_loaded <- load(path, envir = e)
  loo_names <- names_loaded[vapply(names_loaded, function(nm) {
    inherits(e[[nm]], "loo")
  }, logical(1))]
  
  if ("loo_result" %in% loo_names) return(e[["loo_result"]])
  if (length(loo_names) == 1L) return(e[[loo_names]])
  stop("Expected one saved 'loo' object in ", basename(path),
       "; found: ", paste(loo_names, collapse = ", "),
       ". Saved objects: ", paste(names_loaded, collapse = ", "))
}

estimate <- function(obj, row, col = "Estimate") {
  if (row %in% rownames(obj$estimates) &&
      col %in% colnames(obj$estimates)) {
    return(unname(obj$estimates[row, col]))
  }
  NA_real_
}

loos <- list()
loaded <- list()
statuses <- list()
for (i in seq_len(nrow(inventory))) {
  r <- inventory[i, ]
  id <- paste(r$group, r$model, sep = "::")
  if (!r$found) {
    statuses[[length(statuses) + 1L]] <- tibble(
      group = r$group, model = r$model, status = "MISSING",
      detail = paste0("Expected ", r$expected_first))
    next
  }
  x <- tryCatch(read_loo(file.path(loo_dir, r$filename)), error = identity)
  if (inherits(x, "error")) {
    statuses[[length(statuses) + 1L]] <- tibble(
      group = r$group, model = r$model, status = "UNREADABLE",
      detail = conditionMessage(x))
    next
  }
  if (is.null(x$pointwise) ||
      !"elpd_loo" %in% colnames(x$pointwise) ||
      any(!is.finite(x$pointwise[, "elpd_loo"]))) {
    statuses[[length(statuses) + 1L]] <- tibble(
      group = r$group, model = r$model, status = "INVALID",
      detail = "Missing/non-finite pointwise elpd_loo")
    next
  }
  loos[[id]] <- x
  k <- tryCatch(loo::pareto_k_values(x), error = function(e) numeric())
  loaded[[length(loaded) + 1L]] <- tibble(
    group = r$group, model = r$model, filename = r$filename,
    N = nrow(x$pointwise),
    elpd_loo = estimate(x, "elpd_loo"),
    elpd_se = estimate(x, "elpd_loo", "SE"),
    p_loo = estimate(x, "p_loo"),
    looic = estimate(x, "looic"),
    pareto_k_over_07 = if (length(k)) sum(k > 0.7) else NA_integer_,
    pareto_k_over_1 = if (length(k)) sum(k > 1) else NA_integer_
  )
  statuses[[length(statuses) + 1L]] <- tibble(
    group = r$group, model = r$model, status = "LOADED", detail = r$filename)
}

status <- bind_rows(statuses)
write_csv(status, file.path(table_dir, "02_loading_status.csv"))
cat("\nLOADING STATUS:\n")
print(status, n = Inf, width = Inf)

if (!length(loaded)) stop("No saved LOO objects could be loaded.")
metrics <- bind_rows(loaded)
write_csv(metrics, file.path(table_dir, "03_loo_metrics.csv"))

# Explicitly require SEVEN for the Explicit Undirected Exp12 plot.
exp12 <- metrics |> filter(group == "Explicit Undirected")
missing_exp12 <- setdiff(models, exp12$model)
if (STOP_IF_EXP12_NOT_SEVEN && length(missing_exp12)) {
  stop("\nExp12 has ", nrow(exp12), " of 7 usable LOO objects. Missing/unusable: ",
       paste(missing_exp12, collapse = ", "),
       "\nSee 02_loading_status.csv for exact filenames/errors.",
       "\nScript will not quietly present this as a six-model comparison.")
}

# Prevent the same file being assigned to TWO purportedly distinct models.
for (g in unique(metrics$group)) {
  mg <- filter(metrics, group == g)
  if (anyDuplicated(tolower(mg$filename))) {
    stop("Same file assigned to multiple models in ", g, ". Check mapping.")
  }
  if (length(unique(mg$N)) != 1L) {
    stop("Cannot compare ", g, ": model files have different observation counts: ",
         paste(paste(mg$model, mg$N, sep = "="), collapse = ", "))
  }
}

# IMPORTANT: matching counts alone do not guarantee observation identity/order.

# 5. PAIRED ELPD COMPARISONS --------------------------------------------------
comparisons <- list()
pairwise <- list()
for (g in names(suffixes)) {
  mg <- filter(metrics, group == g)
  if (!nrow(mg)) next
  obs <- setNames(lapply(mg$model, function(m) {
    loos[[paste(g, m, sep = "::")]]
  }), mg$model)
  
  if (length(obs) > 1L) {
    cmp <- as.data.frame(loo::loo_compare(x = obs))
    cmp$model <- rownames(cmp)
    cmp <- as_tibble(cmp) |> select(model, elpd_diff, se_diff)
  } else {
    cmp <- tibble(model = names(obs), elpd_diff = 0, se_diff = 0)
  }
  comparisons[[g]] <- cmp |>
    left_join(mg, by = "model") |>
    mutate(rank = row_number(), .before = 1)
  
  if (length(obs) > 1L) {
    pairwise[[g]] <- purrr::map_dfr(combn(names(obs), 2, simplify = FALSE),
                                    function(p) {
                                      diff <- obs[[p[1]]]$pointwise[, "elpd_loo"] -
                                        obs[[p[2]]]$pointwise[, "elpd_loo"]
                                      tibble(group = g, model_A = p[1], model_B = p[2],
                                             elpd_A_minus_B = sum(diff),
                                             paired_se = sqrt(length(diff) * stats::var(diff)))
                                    })
  }
}
results <- bind_rows(comparisons)
write_csv(results, file.path(table_dir, "04_ELPD_rankings_and_SE.csv"))
if (length(pairwise)) {
  pairs <- bind_rows(pairwise)
  write_csv(pairs, file.path(table_dir, "05_all_pairwise_comparisons.csv"))
  instr_pair <- pairs |>
    filter((model_A == "learning_leta_2channel_instr" &
              model_B == "learning_2channel_instr") |
             (model_B == "learning_leta_2channel_instr" &
                model_A == "learning_2channel_instr")) |>
    mutate(new_minus_existing = if_else(
      model_A == "learning_2channel_instr",
      elpd_A_minus_B, -elpd_A_minus_B))
  write_csv(instr_pair, file.path(table_dir, "06_both_instruction_models.csv"))
}

cat("\nELPD RESULTS (seven distinct models where available):\n")
print(results |> select(group, rank, model, elpd_loo, elpd_diff,
                        se_diff, p_loo, looic), n = Inf, width = Inf)

# 6. ONE GRAPH FOR EACH GROUP -------------------------------------------------
# Each group gets its own y-axis, so a poor-fitting model in another group
# cannot flatten this group's ELPD differences.
make_plot <- function(group, zoom = FALSE, floor = -60) {
  d <- filter(results, .data$group == .env$group) |>
    mutate(model = factor(model, levels = models))
  if (!nrow(d)) return(NULL)
  
  scale_span <- max(10, max(abs(d$elpd_diff - d$se_diff)))
  lower <- min(d$elpd_diff - d$se_diff) - 0.13 * scale_span
  upper <- 0.12 * scale_span
  
  if (zoom) {
    lower <- floor - 7
    upper <- 8
  }
  d <- d |>
    mutate(y_draw = if (zoom) pmax(elpd_diff, floor + 1) else elpd_diff,
           truncated = zoom & elpd_diff < floor,
           value_label = if_else(truncated,
                                 paste0(sprintf("%.1f", elpd_diff), "*"),
                                 sprintf("%.1f", elpd_diff)),
           label_y = if_else(truncated,
                             floor + 8,
                             if_else(elpd_diff == 0,
                                     upper * 0.38,
                                     y_draw - if (zoom) 2.1 else 0.035 * scale_span)))
  
  p <- ggplot(d, aes(model, y_draw)) +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
    geom_col(aes(fill = group), width = 0.72, colour = "grey30") +
    geom_errorbar(data = filter(d, !truncated),
                  aes(ymin = elpd_diff - se_diff, ymax = elpd_diff + se_diff),
                  width = 0.15, linewidth = 0.65) +
    geom_text(aes(y = label_y, label = value_label), size = 3.5,
              fontface = "bold", colour = "grey20") +
    scale_fill_manual(values = group_colors, guide = "none") +
    scale_x_discrete(labels = axis_labels, drop = TRUE) +
    coord_cartesian(ylim = c(lower, upper)) +
    labs(title = paste0(group, " (", nrow(d), " models)"),
         subtitle = if (zoom)
           "Zoomed ELPD. * = truncated bar; label is actual value. Error bars = +/- 1 paired SE"
         else "Best model at zero; lower ELPD is worse. Error bars = +/- 1 paired SE",
         x = NULL, y = "Delta ELPD-LOO relative to best model") +
    theme_classic(base_size = 13) +
    theme(plot.title = element_text(face = "bold", size = 17),
          axis.text.x = element_text(size = 9, lineheight = 0.9,
                                     hjust = 0.5, margin = margin(t = 7)),
          axis.ticks.x = element_blank(),
          panel.grid.major.y = element_line(colour = "grey92", linewidth = .3),
          plot.margin = margin(12, 15, 12, 12))
  p
}

slugs <- c(
  "Implicit Unaware" = "implicit_unaware",
  "Implicit Aware" = "implicit_aware",
  "Explicit Undirected" = "explicit_undirected",
  "Explicit Truthful" = "explicit_truthful",
  "Explicit Deceptive" = "explicit_deceptive"
)

for (g in names(suffixes)) {
  d <- filter(results, group == g)
  if (!nrow(d)) next
  
  p <- make_plot(g)
  print(p)   # shows ONE group at a time in the RStudio Plots pane
  ggsave(file.path(plot_dir, paste0("ELPD_", slugs[[g]], ".png")),
         p, width = 13.5, height = 7.2, dpi = 300, bg = "white")
  ggsave(file.path(plot_dir, paste0("ELPD_", slugs[[g]], ".pdf")),
         p, width = 13.5, height = 7.2, bg = "white")
  
  # If there is an extreme outlier, additionally save a legible zoomed version.
  if (any(d$elpd_diff < -60) && any(d$elpd_diff > -60 & d$elpd_diff < 0)) {
    p_zoom <- make_plot(g, zoom = TRUE)
    print(p_zoom)
    ggsave(file.path(plot_dir, paste0("ELPD_", slugs[[g]], "_zoom.png")),
           p_zoom, width = 13.5, height = 7.2, dpi = 300, bg = "white")
    ggsave(file.path(plot_dir, paste0("ELPD_", slugs[[g]], "_zoom.pdf")),
           p_zoom, width = 13.5, height = 7.2, bg = "white")
  }
}

cat("\nSaved figures: ", plot_dir, "\nSaved statistics: ", table_dir, "\n", sep = "")
cat("Note: identical observation counts do not prove identical observation order.\n")
cat("Check R-hat/ESS separately; good Pareto-k does not establish MCMC convergence.\n")







# =============================================================================
# R-HAT / ESS ACROSS SIX MODELS AND FIVE BASE-RATE GROUPS
# =============================================================================
# Reads SAVED FIT .rdata files (NOT LOO .rdata files).
# Does not rerun Stan or modify the fitted models.
# For a fast first pass, analyses group-level parameters and hyperparameters.
# Set INCLUDE_PARTICIPANT_PARAMETERS <- TRUE for an extended convergence audit.
# =============================================================================

rm(list = ls(all.names = TRUE))

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(readr)
  library(ggplot2)
  library(tibble)
  library(stringr)
  library(cmdstanr)
})

# 1. PATHS / OPTIONS ---------------------------------------------------------
project_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias"
fits_dir <- file.path(project_dir, "stan", "results", "fits", "exp11_unaware")
output_dir <- file.path(project_dir, "stan", "results", "rhat_all_models_5groups")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# FALSE = group-level parameters + hierarchical standard deviations.
# TRUE  = also audit the participant-level latent and transformed parameters.
INCLUDE_PARTICIPANT_PARAMETERS <- FALSE
INCLUDE_HMC_DIAGNOSTICS <- TRUE

# Conservative screening rules for FOUR CHAINS.
RHAT_LIMIT <- 1.01
ESS_LIMIT <- 400

# 2. CATALOGUE ---------------------------------------------------------------
groups <- c(
  "Implicit Unaware"    = "unaware_exp11",
  "Implicit Aware"      = "aware_exp11",
  "Explicit Undirected" = "aware_exp12",
  "Explicit True"       = "truthful_exp13",
  "Explicit Deceptive"  = "deceptive_exp13"
)

models <- c(
  "Learning"                 = "learning",
  "Local eta"                = "localeta",
  "Learning + Local eta"     = "learning_localeta",
  "2 channels"               = "2channels",
  "Local eta + 2 channels"   = "leta_2channels",
  "Instruction + 2 channels" = "learning_leta_2channel_instr"
)

if (!dir.exists(fits_dir)) stop("Fits directory not found: ", fits_dir)
all_files <- list.files(fits_dir, pattern = "\\.rdata$", full.names = TRUE,
                        ignore.case = TRUE)
if (length(all_files) == 0L) stop("No .rdata files in ", fits_dir,
                                  ". Check that you are using RESULTS/FITS, not RESULTS/LOO.")

catalogue <- expand_grid(group = names(groups), model = names(models)) |>
  mutate(group = factor(group, levels = names(groups)),
         model = factor(model, levels = names(models))) |>
  arrange(group, model)

# Match exact model prefix and experimental-group suffix.
# The learning + local eta model can have either historical filename prefix.
match_fit <- function(group, model) {
  pref <- unname(models[[as.character(model)]])
  suffix <- unname(groups[[as.character(group)]])
  prefixes <- if (identical(pref, "learning_localeta")) {
    c("learning_localeta", "loo_learning_localeta")
  } else pref
  target <- paste0(prefixes, "_", suffix, ".rdata")
  matches <- all_files[tolower(basename(all_files)) %in% tolower(target)]
  if (length(matches) == 0L) return(NA_character_)
  if (length(matches) > 1L) {
    warning("Multiple candidate fit files for ", group, " / ", model,
            ": ", paste(basename(matches), collapse = ", "),
            ". Choosing the first; verify it is correct.")
  }
  matches[1]
}

catalogue$filepath <- map2_chr(catalogue$group, catalogue$model, match_fit)
catalogue$filename <- basename(catalogue$filepath)
catalogue$filename[is.na(catalogue$filepath)] <- NA_character_
catalogue$load_status <- if_else(is.na(catalogue$filepath), "MISSING FIT FILE", "PENDING")

# 3. HELPERS -----------------------------------------------------------------
get_saved_fit <- function(file) {
  env <- new.env(parent = globalenv())
  object_names <- suppressWarnings(load(file, envir = env))
  if ("fit" %in% object_names && inherits(env$fit, "CmdStanMCMC")) {
    return(env$fit)
  }
  candidates <- Filter(function(x) inherits(x, "CmdStanMCMC"),
                       mget(object_names, envir = env, inherits = FALSE))
  if (length(candidates) != 1L) {
    stop("Expected one CmdStanMCMC fit object in ", basename(file),
         "; found ", length(candidates), ".")
  }
  candidates[[1]]
}

# Rho does not enter the likelihood for the two uninstructed groups (instr = 4).
# A good R-hat for rho in those groups does NOT imply empirical identification.
not_identified_rho <- function(v, group_name) {
  if (!group_name %in% c("Implicit Unaware", "Implicit Aware"))
    return(rep(FALSE, length(v)))
  grepl("^(mu_rho|rho|mu_pr\\[8\\]|sigma_pr\\[8\\]|param_raw\\[[0-9]+,8\\]|params\\[[0-9]+,8\\])$", v)
}

status_of_parameter <- function(rhat, bulk, tail, constant, not_identified) {
  dplyr::case_when(
    not_identified ~ "NOT IDENTIFIED",
    constant ~ "FIXED / CONSTANT",
    !is.finite(rhat) | !is.finite(bulk) | !is.finite(tail) ~ "MISSING DIAGNOSTIC",
    rhat >= 1.05 | bulk < 100 | tail < 100 ~ "SERIOUS",
    rhat >= RHAT_LIMIT | bulk < ESS_LIMIT | tail < ESS_LIMIT ~ "CHECK",
    TRUE ~ "OK"
  )
}

safe_max <- function(x) if (any(is.finite(x))) max(x[is.finite(x)]) else NA_real_
safe_min <- function(x) if (any(is.finite(x))) min(x[is.finite(x)]) else NA_real_
which_max <- function(x, variable) {
  idx <- which(is.finite(x))
  if (!length(idx)) return(NA_character_)
  variable[idx[which.max(x[idx])]]
}
which_min <- function(x, variable) {
  idx <- which(is.finite(x))
  if (!length(idx)) return(NA_character_)
  variable[idx[which.min(x[idx])]]
}

# 4. EXTRACT DIAGNOSTICS ONE FIT AT A TIME -----------------------------------
all_diagnostics <- list()
all_hmc <- list()

for (i in seq_len(nrow(catalogue))) {
  group_name <- as.character(catalogue$group[i])
  model_name <- as.character(catalogue$model[i])
  file <- catalogue$filepath[i]
  
  if (is.na(file)) {
    message("MISSING: ", group_name, " / ", model_name)
    next
  }
  
  message("Reading: ", group_name, " / ", model_name, " / ", basename(file))
  
  result <- tryCatch({
    fit <- get_saved_fit(file)
    available <- fit$metadata()$model_params
    
    # Group-level transformed parameters, group hyperparameters and raw means.
    # Avoid 6240 * 4000 log_lik draws and other huge generated quantities.
    selected <- available[grepl("^mu_", available) |
                            grepl("^sigma_", available) |
                            available %in% c("mu_pr", "sigma_pr", "Delta", "Delta_raw")]
    
    if (INCLUDE_PARTICIPANT_PARAMETERS) {
      selected <- unique(c(selected,
                           intersect(available, c("param_raw", "params"))))
    }
    
    if (!length(selected)) stop("No group parameters found in CmdStan metadata.")
    s <- as_tibble(fit$summary(variables = selected))
    required <- c("variable", "rhat", "ess_bulk", "ess_tail")
    if (!all(required %in% names(s))) {
      stop("Fit summary lacks: ", paste(setdiff(required, names(s)), collapse = ", "))
    }
    
    # Fixed Delta = .10 is not a sampled parameter and should not count as an error.
    is_constant <- if ("sd" %in% names(s)) {
      is.finite(s$sd) & s$sd < 1e-10
    } else rep(FALSE, nrow(s))
    is_prior_rho <- not_identified_rho(s$variable, group_name)
    
    s <- s |>
      mutate(group = group_name, model = model_name,
             fixed_or_constant = is_constant,
             not_identified = is_prior_rho,
             status = status_of_parameter(rhat, ess_bulk, ess_tail,
                                          fixed_or_constant, not_identified),
             .before = 1)
    
    hmc <- tibble(group = group_name, model = model_name,
                  n_divergent = NA_integer_, n_max_treedepth = NA_integer_,
                  min_ebfmi = NA_real_)
    if (INCLUDE_HMC_DIAGNOSTICS) {
      d <- tryCatch(fit$diagnostic_summary(), error = function(e) NULL)
      if (!is.null(d)) {
        hmc$n_divergent <- if (!is.null(d$num_divergent))
          as.integer(sum(d$num_divergent, na.rm = TRUE)) else NA_integer_
        hmc$n_max_treedepth <- if (!is.null(d$num_max_treedepth))
          as.integer(sum(d$num_max_treedepth, na.rm = TRUE)) else NA_integer_
        hmc$min_ebfmi <- if (!is.null(d$ebfmi)) safe_min(d$ebfmi) else NA_real_
      }
    }
    
    list(s = s, hmc = hmc)
  }, error = function(e) {
    warning("FAILED: ", group_name, " / ", model_name, ": ", conditionMessage(e))
    structure(list(message = conditionMessage(e)), class = "fit_read_error")
  })
  
  if (inherits(result, "fit_read_error")) {
    catalogue$load_status[i] <- paste0("ERROR: ", result$message)
  } else {
    all_diagnostics[[length(all_diagnostics) + 1L]] <- result$s
    all_hmc[[length(all_hmc) + 1L]] <- result$hmc
    catalogue$load_status[i] <- "LOADED"
  }
  rm(result)
  gc(verbose = FALSE)
}

write_csv(catalogue |> mutate(group = as.character(group), model = as.character(model)),
          file.path(output_dir, "01_fit_inventory.csv"))

if (!length(all_diagnostics)) {
  stop("No fits could be analysed. Inspect 01_fit_inventory.csv and fits_dir.")
}

all_diag <- bind_rows(all_diagnostics)
all_hmc <- bind_rows(all_hmc)
write_csv(all_diag, file.path(output_dir, "03_all_parameter_diagnostics.csv"))
write_csv(all_hmc, file.path(output_dir, "05_hmc_sampler_diagnostics.csv"))
write_csv(all_diag |> filter(status %in% c("CHECK", "SERIOUS", "MISSING DIAGNOSTIC")),
          file.path(output_dir, "04_parameters_needing_attention.csv"))

# 5. MODEL-BY-GROUP OVERVIEW -------------------------------------------------
model_summary <- all_diag |>
  group_by(group, model) |>
  group_modify(function(d, key) {
    valid <- d |> filter(!status %in% c("NOT IDENTIFIED", "FIXED / CONSTANT"))
    tibble(
      n_parameters_checked = nrow(valid),
      n_problem_parameters = sum(valid$status != "OK"),
      n_serious_parameters = sum(valid$status == "SERIOUS"),
      max_rhat = safe_max(valid$rhat),
      worst_rhat_parameter = which_max(valid$rhat, valid$variable),
      min_bulk_ess = safe_min(valid$ess_bulk),
      worst_bulk_ess_parameter = which_min(valid$ess_bulk, valid$variable),
      min_tail_ess = safe_min(valid$ess_tail),
      worst_tail_ess_parameter = which_min(valid$ess_tail, valid$variable)
    )
  }) |>
  ungroup() |>
  left_join(all_hmc, by = c("group", "model")) |>
  mutate(
    convergence_screen = case_when(
      n_parameters_checked == 0 ~ "NO PARAMETERS",
      n_serious_parameters > 0 ~ "SERIOUS",
      n_problem_parameters > 0 ~ "CHECK",
      TRUE ~ "MEETS GROUP-LEVEL CRITERIA"
    ),
    group = factor(group, levels = names(groups)),
    model = factor(model, levels = names(models))
  ) |>
  arrange(group, model)

write_csv(model_summary, file.path(output_dir, "02_model_group_convergence_summary.csv"))

cat("\n================ ALL MODEL × GROUP CONVERGENCE CHECKS ================\n")
print(model_summary |>
        select(group, model, n_parameters_checked, n_problem_parameters,
               max_rhat, min_bulk_ess, min_tail_ess, n_divergent,
               convergence_screen), n = Inf, width = Inf)

cat("\n================ WORST RHAT PARAMETERS ==============================\n")
print(model_summary |>
        select(group, model, worst_rhat_parameter, max_rhat,
               worst_bulk_ess_parameter, min_bulk_ess), n = Inf, width = Inf)

# 6. 5 × 6 HEATMAP -----------------------------------------------------------
heat <- catalogue |>
  select(group, model, load_status) |>
  left_join(model_summary |> select(group, model, max_rhat, n_problem_parameters),
            by = c("group", "model")) |>
  mutate(label = if_else(is.na(max_rhat), "Missing",
                         sprintf("%.2f\n(%d flagged)", max_rhat,
                                 coalesce(n_problem_parameters, 0L))),
         group = factor(group, levels = rev(names(groups))),
         model = factor(model, levels = names(models)))

p_heat <- ggplot(heat, aes(model, group, fill = max_rhat)) +
  geom_tile(colour = "white", linewidth = 1.3) +
  geom_text(aes(label = label), size = 3.1, lineheight = 1.15) +
  scale_fill_gradientn(
    colours = c("#84c2a2", "#f3d177", "#d97b68"),
    values = scales::rescale(c(1, 1.05, 1.20)),
    limits = c(1, 1.5), oob = scales::squish,
    na.value = "grey92", name = "Max R-hat"
  ) +
  scale_x_discrete(labels = function(x) stringr::str_wrap(x, width = 16)) +
  labs(title = "MCMC convergence across models and base-rate groups",
       subtitle = "Cell: highest group-level R-hat (number of parameters failing R-hat/ESS checks)",
       x = "Computational model", y = "Experimental group") +
  theme_minimal(base_size = 12) +
  theme(panel.grid = element_blank(),
        axis.text.x = element_text(angle = 25, hjust = 1),
        plot.title = element_text(face = "bold"))
print(p_heat)
ggsave(file.path(output_dir, "01_model_group_max_rhat_heatmap.png"),
       p_heat, width = 13, height = 6.5, dpi = 300)
ggsave(file.path(output_dir, "01_model_group_max_rhat_heatmap.pdf"),
       p_heat, width = 13, height = 6.5)

# 7. GROUP MEANS ACROSS MODELS -----------------------------------------------
# Do not assume different models have the same mu_pr index mapping.
# Compare only explicitly named transformed group-level parameters.
selected <- all_diag |>
  filter(variable %in% c("mu_alpha", "mu_beta", "mu_lambda", "mu_delta",
                         "mu_eta", "mu_alpha0", "mu_beta0", "mu_rho"),
         status != "NOT IDENTIFIED", status != "FIXED / CONSTANT") |>
  mutate(variable = factor(variable, levels = c("mu_alpha", "mu_beta",
                                                "mu_lambda", "mu_delta", "mu_eta", "mu_alpha0",
                                                "mu_beta0", "mu_rho")),
         group = factor(group, levels = names(groups)),
         model = factor(model, levels = names(models)))
write_csv(selected, file.path(output_dir, "06_named_group_parameter_diagnostics.csv"))

if (nrow(selected)) {
  p_parameters <- ggplot(selected, aes(variable, rhat, colour = model, group = model)) +
    geom_hline(yintercept = RHAT_LIMIT, colour = "firebrick", linetype = "dashed") +
    geom_point(position = position_dodge(width = .55), size = 2) +
    facet_wrap(~ group, ncol = 2, scales = "free_x") +
    scale_x_discrete(labels = c(mu_alpha = "alpha", mu_beta = "beta",
                                mu_lambda = "lambda", mu_delta = "delta", mu_eta = "eta",
                                mu_alpha0 = "alpha0", mu_beta0 = "beta0", mu_rho = "rho")) +
    labs(title = "R-hat of named group parameters, by model",
         subtitle = "Dashed line: 1.01 (below is preferred)",
         x = NULL, y = "R-hat", colour = "Model") +
    theme_bw(base_size = 11) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1),
          legend.position = "bottom", legend.title = element_blank())
  print(p_parameters)
  ggsave(file.path(output_dir, "02_parameter_rhat_by_model.png"),
         p_parameters, width = 14, height = 10, dpi = 300)
  ggsave(file.path(output_dir, "02_parameter_rhat_by_model.pdf"),
         p_parameters, width = 14, height = 10)
}

cat("\nFiles saved in:\n", output_dir, "\n")
cat("\nIMPORTANT: A low MAX R-hat is not proof of full convergence.\n")
cat("This run checked group-level parameters and hyperparameters",
    if (INCLUDE_PARTICIPANT_PARAMETERS) "and participant parameters." else "only.\n")
cat("Check divergent transitions and participant parameters before final inference.\n")
cat("Fixed Delta=0.10 must not count as an estimated parameter.\n")
cat("Rho is excluded as not identified in the uninstructed groups.\n")















############################################################
# LEARNING + TWO CHANNELS + INSTRUCTION
#
# No eta.
# No LOO.
# No file picker.
# No files saved.
#
# Plots: RStudio Plots pane
# Statistics: Console and named objects in Environment
############################################################

suppressPackageStartupMessages({
  library(cmdstanr)
  library(tidyverse)
  library(posterior)
})

############################################################
# 1. PATHS AND GROUPS
############################################################

project_dir <- path.expand(
  "~/Documents/GitHub/reliable_info_bias"
)

fits_dir <- file.path(
  project_dir,
  "stan", "results", "fits", "exp11_unaware"
)

group_levels <- c(
  "Implicit Unaware",
  "Implicit Aware",
  "Explicit Undirected",
  "Explicit True",
  "Explicit Deceptive"
)

group_cols <- c(
  "Implicit Unaware"    = "#F4A3A3",
  "Implicit Aware"      = "#8FD694",
  "Explicit Undirected" = "#E69F00",
  "Explicit True"       = "#1B7837",
  "Explicit Deceptive"  = "#8B0000"
)

group_axis_labels <- c(
  "Implicit Unaware"    = "Implicit\nUnaware",
  "Implicit Aware"      = "Implicit\nAware",
  "Explicit Undirected" = "Explicit\nUndirected",
  "Explicit True"       = "Explicit\nTrue",
  "Explicit Deceptive"  = "Explicit\nDeceptive"
)

fit_files <- c(
  "Implicit Unaware" =
    "learning_2channel_instr_unaware_exp11.rdata",
  "Implicit Aware" =
    "learning_2channel_instr_aware_exp11.rdata",
  "Explicit Undirected" =
    "learning_2channel_instr_aware_exp12.rdata",
  "Explicit True" =
    "learning_2channel_instr_truthful_exp13.rdata",
  "Explicit Deceptive" =
    "learning_2channel_instr_deceptive_exp13.rdata"
)

paths <- file.path(fits_dir, unname(fit_files))

if (any(!file.exists(paths))) {
  stop(
    "Missing files:\n",
    paste(paths[!file.exists(paths)], collapse = "\n")
  )
}

SEED <- 12345L

############################################################
# 2. PARAMETERS
############################################################

parameter_names <- c(
  "mu_alpha",
  "mu_beta",
  "mu_lambda",
  "mu_delta",
  "mu_alpha0",
  "mu_beta0",
  "mu_rho",
  "Delta"
)

parameter_labels <- c(
  mu_alpha  = "alpha",
  mu_beta   = "beta",
  mu_lambda = "lambda",
  mu_delta  = "delta",
  mu_alpha0 = "alpha[0]",
  mu_beta0  = "beta[0]",
  mu_rho    = "rho",
  Delta     = "Delta"
)

# Reference lines are descriptive benchmarks.
# A posterior interval excluding a boundary such as zero
# is not, by itself, a test against a point-null model.
reference_values <- c(
  mu_alpha  = 1,
  mu_beta   = 0,
  mu_lambda = 0,
  mu_delta  = 1,
  mu_alpha0 = 1,
  mu_beta0  = 0
)

parameters_for_group <- function(group_name) {
  
  pars <- parameter_names
  
  # Instruction parameters are not interpreted in
  # groups that received no instruction.
  if (group_name %in% c(
    "Implicit Unaware",
    "Implicit Aware"
  )) {
    pars <- setdiff(pars, c("mu_rho", "Delta"))
  }
  
  # Delta applies only to the undirected instruction.
  if (group_name != "Explicit Undirected") {
    pars <- setdiff(pars, "Delta")
  }
  
  pars
}

############################################################
# 3. LOAD FITS
############################################################

load_fit <- function(filename) {
  
  path <- file.path(fits_dir, filename)
  env <- new.env()
  
  loaded <- load(path, envir = env)
  
  fit_names <- loaded[
    vapply(
      loaded,
      function(nm) inherits(env[[nm]], "CmdStanMCMC"),
      logical(1)
    )
  ]
  
  if (length(fit_names) != 1L) {
    stop(
      "Expected exactly one CmdStanMCMC fit in:\n",
      path
    )
  }
  
  env[[fit_names]]
}

fits <- setNames(
  lapply(unname(fit_files), load_fit),
  names(fit_files)
)

############################################################
# 4. CHECK AND EXTRACT NAMED PARAMETERS
############################################################

# Use named transformed outputs directly.
# Do not guess mu_pr indices or parameter transformations.

extract_group_draws <- function(fit, group_name) {
  
  available <- posterior::variables(fit$draws())
  required <- parameters_for_group(group_name)
  
  if ("mu_eta" %in% available) {
    stop(
      group_name,
      ": this fit contains mu_eta. ",
      "Check that it is the model WITHOUT eta."
    )
  }
  
  missing <- setdiff(required, available)
  
  if (length(missing)) {
    stop(
      "Missing named parameters for ", group_name, ":\n",
      paste(missing, collapse = ", "),
      "\nCheck the Stan model's saved parameter names."
    )
  }
  
  d <- as.data.frame(
    posterior::as_draws_matrix(
      fit$draws(variables = required)
    )
  )
  
  d <- d[, required, drop = FALSE]
  
  if (!nrow(d) || any(!is.finite(as.matrix(d)))) {
    stop("Missing or non-finite draws for ", group_name)
  }
  
  as_tibble(d) %>%
    mutate(
      group = group_name,
      draw_id = row_number(),
      .before = 1
    )
}

draws_wide <- imap_dfr(
  fits,
  extract_group_draws
) %>%
  mutate(
    group = factor(group, levels = group_levels)
  )

draws_long <- draws_wide %>%
  pivot_longer(
    all_of(parameter_names),
    names_to = "parameter",
    values_to = "value"
  ) %>%
  filter(is.finite(value)) %>%
  mutate(
    parameter = factor(
      parameter,
      levels = parameter_names
    ),
    parameter_clean = factor(
      unname(parameter_labels[as.character(parameter)]),
      levels = unname(parameter_labels)
    )
  )

cat("\nPosterior draws per group:\n")
print(count(draws_wide, group))

############################################################
# 5. PARAMETER DIAGNOSTICS
############################################################

parameter_diagnostics <- imap_dfr(
  fits,
  function(fit, group_name) {
    
    fit$summary(
      variables = parameters_for_group(group_name)
    ) %>%
      as_tibble() %>%
      mutate(group = group_name, .before = 1)
  }
)

cat("\nDIAGNOSTICS FOR THE PLOTTED PARAMETERS\n")
print(parameter_diagnostics, n = Inf, width = Inf)

if (
  any(parameter_diagnostics$rhat > 1.01, na.rm = TRUE)
) {
  warning(
    "Some plotted parameters have R-hat > 1.01. ",
    "Check convergence before interpreting comparisons."
  )
}

# Attempt to display sampler diagnostics too.
# Some saved fits may not retain the necessary sampler data.
sampler_diagnostics <- imap(
  fits,
  function(fit, group_name) {
    
    tryCatch(
      fit$diagnostic_summary(),
      error = function(e) {
        message(
          group_name,
          ": sampler diagnostics unavailable: ",
          conditionMessage(e)
        )
        NULL
      }
    )
  }
)

cat("\nSAMPLER DIAGNOSTICS\n")
print(sampler_diagnostics)

############################################################
# 6. POSTERIOR SUMMARIES
############################################################

posterior_summary <- draws_long %>%
  group_by(group, parameter, parameter_clean) %>%
  summarise(
    mean = mean(value),
    median = median(value),
    sd = sd(value),
    l95 = unname(quantile(value, 0.025)),
    u95 = unname(quantile(value, 0.975)),
    .groups = "drop"
  )

writeup_table <- posterior_summary %>%
  mutate(
    result = sprintf(
      "M = %.3f, 95%% CrI [%.3f, %.3f]",
      mean, l95, u95
    )
  ) %>%
  select(group, parameter_clean, result) %>%
  pivot_wider(
    names_from = parameter_clean,
    values_from = result
  )

cat("\nGROUP POSTERIOR SUMMARIES\n")
print(posterior_summary, n = Inf, width = Inf)

cat("\nWRITE-UP TABLE\n")
print(writeup_table, n = Inf, width = Inf)

############################################################
# 7. DESCRIPTIVE PROBABILITIES AGAINST BENCHMARKS
############################################################

reference_stats <- draws_long %>%
  filter(
    as.character(parameter) %in% names(reference_values)
  ) %>%
  mutate(
    reference = unname(
      reference_values[as.character(parameter)]
    )
  ) %>%
  group_by(group, parameter, reference) %>%
  summarise(
    probability_above = mean(value > reference),
    probability_below = mean(value < reference),
    l95 = unname(quantile(value, 0.025)),
    u95 = unname(quantile(value, 0.975)),
    .groups = "drop"
  )

cat("\nDESCRIPTIVE POSTERIOR PROBABILITIES\n")
cat(
  "These are not frequentist p-values or Bayes factors.\n",
  "Boundary comparisons can be determined by parameter bounds.\n"
)

print(reference_stats, n = Inf, width = Inf)

############################################################
# 8. BETWEEN-GROUP POSTERIOR DIFFERENCES
############################################################

# Assumes groups were fitted independently using
# the same parameter definitions and comparable priors.
#
# Randomly pair draws across independent group posteriors.
# Positive difference means group 1 > group 2.
#
# These are marginal 95% credible intervals.
# They do not provide simultaneous/multiplicity-adjusted coverage.

posterior_contrast <- function(p, group_1, group_2) {
  
  x1 <- draws_long %>%
    filter(
      as.character(parameter) == p,
      as.character(group) == group_1
    ) %>%
    pull(value)
  
  x2 <- draws_long %>%
    filter(
      as.character(parameter) == p,
      as.character(group) == group_2
    ) %>%
    pull(value)
  
  n <- min(length(x1), length(x2))
  
  difference <- x1[sample.int(length(x1), n)] -
    x2[sample.int(length(x2), n)]
  
  lower <- unname(quantile(difference, 0.025))
  upper <- unname(quantile(difference, 0.975))
  
  tibble(
    parameter = p,
    group_1 = group_1,
    group_2 = group_2,
    mean_difference = mean(difference),
    median_difference = median(difference),
    l95 = lower,
    u95 = upper,
    probability_group1_greater = mean(difference > 0),
    probability_group1_lower = mean(difference < 0),
    interval_excludes_zero = lower > 0 | upper < 0
  )
}

set.seed(SEED)

posterior_pairwise <- map_dfr(
  parameter_names,
  function(p) {
    
    eligible <- group_levels[
      vapply(
        group_levels,
        function(g) p %in% parameters_for_group(g),
        logical(1)
      )
    ]
    
    # Delta has only one eligible group.
    if (length(eligible) < 2L) {
      return(tibble())
    }
    
    pairs <- combn(
      eligible,
      2,
      simplify = FALSE
    )
    
    map_dfr(
      pairs,
      function(pair) {
        posterior_contrast(p, pair[1], pair[2])
      }
    )
  }
)

cat("\nALL BETWEEN-GROUP POSTERIOR DIFFERENCES\n")
print(posterior_pairwise, n = Inf, width = Inf)

cat("\nDIFFERENCES WITH 95% CrI EXCLUDING ZERO\n")
print(
  filter(posterior_pairwise, interval_excludes_zero),
  n = Inf,
  width = Inf
)

############################################################
# 9. PLOT STYLE
############################################################

plot_theme <- theme_bw(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(face = "bold"),
    legend.position = "bottom",
    legend.title = element_blank()
  )

reference_lines <- tibble(
  parameter_clean = factor(
    unname(parameter_labels[names(reference_values)]),
    levels = unname(parameter_labels)
  ),
  reference = unname(reference_values)
)

############################################################
# 10. POSTERIOR DISTRIBUTIONS
############################################################

p_distributions <- ggplot(
  draws_long,
  aes(value, fill = group, colour = group)
) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 60,
    position = "identity",
    alpha = 0.30,
    linewidth = 0.20
  ) +
  facet_wrap(
    ~parameter_clean,
    scales = "free",
    ncol = 4,
    labeller = label_parsed
  ) +
  scale_fill_manual(values = group_cols) +
  scale_colour_manual(values = group_cols) +
  labs(
    title = "Learning + two channels + instruction",
    subtitle = "Group-level posterior distributions; no eta",
    x = "Parameter value",
    y = "Density"
  ) +
  plot_theme

############################################################
# 11. ALPHA, BETA, ALPHA0 AND BETA0
############################################################

evidence_prior_parameters <- c(
  "mu_alpha", "mu_beta", "mu_alpha0", "mu_beta0"
)

p_alpha_beta <- ggplot(
  filter(
    draws_long,
    as.character(parameter) %in% evidence_prior_parameters
  ),
  aes(value, fill = group, colour = group)
) +
  geom_density(alpha = 0.15, linewidth = 0.8) +
  geom_vline(
    data = filter(
      reference_lines,
      as.character(parameter_clean) %in%
        unname(parameter_labels[evidence_prior_parameters])
    ),
    aes(xintercept = reference),
    inherit.aes = FALSE,
    linetype = "dashed"
  ) +
  facet_wrap(
    ~parameter_clean,
    scales = "free",
    ncol = 2,
    labeller = label_parsed
  ) +
  scale_fill_manual(values = group_cols) +
  scale_colour_manual(values = group_cols) +
  labs(
    title = "Evidence and prior transformation parameters",
    x = "Posterior value",
    y = "Density"
  ) +
  plot_theme

############################################################
# 12. POSTERIOR BOXPLOTS
############################################################

# These boxes summarise posterior draws of group parameters.
# They do not represent the distribution of participants.

p_boxplots <- ggplot(
  draws_long,
  aes(group, value, fill = group)
) +
  geom_boxplot(
    width = 0.6,
    outlier.shape = NA,
    linewidth = 0.35
  ) +
  geom_hline(
    data = reference_lines,
    aes(yintercept = reference),
    inherit.aes = FALSE,
    linetype = "dashed",
    colour = "grey35"
  ) +
  facet_wrap(
    ~parameter_clean,
    scales = "free",
    ncol = 2,
    labeller = label_parsed
  ) +
  scale_fill_manual(values = group_cols, guide = "none") +
  scale_x_discrete(labels = group_axis_labels) +
  labs(
    title = "Group-level posterior boxplots",
    subtitle = "Boxes show posterior uncertainty, not participant variability",
    x = NULL,
    y = "Posterior value"
  ) +
  plot_theme +
  theme(
    axis.text.x = element_text(size = 8),
    legend.position = "none"
  )

############################################################
# 13. EVIDENCE AND PRIOR TRANSFORMATION CURVES
############################################################

# Evidence channel:
#   plogis(alpha * qlogis(reliability) + beta)
#
# Prior channel:
#   plogis(alpha0 * qlogis(prior) + beta0)
#
# These are channel transformations, not the full
# six-sample decision prediction or learning trajectory.

summarise_values <- function(values) {
  tibble(
    mean = mean(values),
    l95 = unname(quantile(values, 0.025)),
    u95 = unname(quantile(values, 0.975))
  )
}

channel_curve <- function(d, x_vals, alpha_name, beta_name) {
  
  map_dfr(x_vals, function(x) {
    
    p <- pmin(pmax(x / 100, 1e-6), 1 - 1e-6)
    
    values <- 100 * plogis(
      d[[alpha_name]] * qlogis(p) + d[[beta_name]]
    )
    
    summarise_values(values) %>%
      mutate(x = x, .before = 1)
  })
}

curve_data <- map_dfr(
  group_levels,
  function(g) {
    
    d <- filter(draws_wide, as.character(group) == g)
    
    bind_rows(
      channel_curve(
        d, seq(0, 100, by = 0.5),
        "mu_alpha", "mu_beta"
      ) %>%
        mutate(channel = "Evidence reliability"),
      
      channel_curve(
        d, seq(0, 100, by = 0.5),
        "mu_alpha0", "mu_beta0"
      ) %>%
        mutate(channel = "Prior belief")
    ) %>%
      mutate(group = g)
  }
) %>%
  mutate(group = factor(group, levels = group_levels))

reliability_dots <- map_dfr(
  group_levels,
  function(g) {
    
    d <- filter(draws_wide, as.character(group) == g)
    
    channel_curve(
      d, c(50, 55, 65),
      "mu_alpha", "mu_beta"
    ) %>%
      mutate(
        group = g,
        channel = "Evidence reliability"
      )
  }
)

p_transformations <- ggplot(
  curve_data,
  aes(x, mean, colour = group, fill = group)
) +
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    colour = "grey40"
  ) +
  geom_ribbon(
    aes(ymin = l95, ymax = u95),
    alpha = 0.10,
    colour = NA
  ) +
  geom_line(linewidth = 0.9) +
  geom_point(
    data = reliability_dots,
    size = 2
  ) +
  facet_wrap(~channel, nrow = 1) +
  scale_colour_manual(values = group_cols) +
  scale_fill_manual(values = group_cols) +
  coord_cartesian(
    xlim = c(0, 100),
    ylim = c(0, 100)
  ) +
  labs(
    title = "Evidence and prior transformations",
    subtitle = "Posterior mean curves with pointwise 95% credible intervals",
    x = "Input probability (%)",
    y = "Transformed probability (%)"
  ) +
  plot_theme

############################################################
# 14. SEQUENTIAL-POSITION WEIGHTS
############################################################

position_weights <- map_dfr(
  group_levels,
  function(g) {
    
    d <- filter(draws_wide, as.character(group) == g)
    
    map_dfr(1:6, function(position) {
      
      values <- exp(d$mu_lambda * (position - 6))
      
      summarise_values(values) %>%
        mutate(position = position, .before = 1)
    }) %>%
      mutate(group = g)
  }
) %>%
  mutate(group = factor(group, levels = group_levels))

p_position_weights <- ggplot(
  position_weights,
  aes(position, mean, colour = group, fill = group)
) +
  geom_hline(
    yintercept = 1,
    linetype = "dashed",
    colour = "grey40"
  ) +
  geom_ribbon(
    aes(ymin = l95, ymax = u95),
    alpha = 0.10,
    colour = NA
  ) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = 1:6) +
  scale_colour_manual(values = group_cols) +
  scale_fill_manual(values = group_cols) +
  labs(
    title = "Sequential evidence weights",
    subtitle = "Weight relative to the sixth sample",
    x = "Sample position",
    y = "Relative weight"
  ) +
  plot_theme

############################################################
# 15. BETWEEN-GROUP DIFFERENCE PLOT
############################################################

contrast_plot_data <- posterior_pairwise %>%
  mutate(
    comparison = paste(group_1, "-", group_2),
    parameter_clean = factor(
      unname(parameter_labels[parameter]),
      levels = unname(parameter_labels)
    )
  )

p_contrasts <- ggplot(
  contrast_plot_data,
  aes(
    x = mean_difference,
    y = comparison
  )
) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    colour = "grey40"
  ) +
  geom_segment(
    aes(
      x = l95,
      xend = u95,
      yend = comparison
    ),
    linewidth = 0.6
  ) +
  geom_point(size = 1.8) +
  facet_wrap(
    ~parameter_clean,
    scales = "free",
    ncol = 2,
    labeller = label_parsed
  ) +
  labs(
    title = "Between-group posterior differences",
    subtitle = "Group 1 minus group 2; marginal 95% credible intervals",
    x = "Parameter difference",
    y = NULL
  ) +
  plot_theme +
  theme(axis.text.y = element_text(size = 7))

############################################################
# 16. SHOW ALL FIGURES — SAVE NOTHING
############################################################

print(p_distributions)
print(p_alpha_beta)
print(p_boxplots)
print(p_transformations)
print(p_position_weights)
print(p_contrasts)

cat(
  "\nFinished. No files were saved and no LOO was run.\n",
  "Use the arrows in RStudio's Plots pane to browse figures.\n",
  "Or display one again, for example:\n",
  "print(p_transformations)\n"
)
