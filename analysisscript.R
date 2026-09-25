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

 