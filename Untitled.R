# =========================================================================
#  POSTERIOR DISTRIBUTIONS + STATS — FIVE GROUPS
#  Implicit Unaware Base Rate, Implicit Aware Base Rate, Explicit Undirected Base Rate, Explicit True Base Rate, Explicit Deceptive Base Rate
# =========================================================================

library(tidyverse)
library(posterior)
library(ggplot2)
library(grid)

# =========================================================================
# 1. PATHS
# =========================================================================

base_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias/stan"
fits_dir <- file.path(base_dir, "results/fits/exp11_unaware")
fig_dir  <- file.path(base_dir, "results/figures")

if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

# =========================================================================
# 2. LOAD FITS
# =========================================================================

load(file.path(fits_dir, "fit_trunc_boost_unaware_exp11.rdata"))
fit_unaware <- fit

load(file.path(fits_dir, "fit_trunc_boost_aware_exp11.rdata"))
fit_aware <- fit

load(file.path(fits_dir, "fit_trunc_boost_aware_exp12.rdata"))
fit_explicit <- fit

load(file.path(fits_dir, "fit_trunc_boost_truthful_exp13.rdata"))
fit_truthful <- fit

load(file.path(fits_dir, "fit_trunc_boost_deceptive_exp13.rdata"))
fit_deceptive <- fit

rm(fit)

# =========================================================================
# 3. PARAMETER NAMES
# =========================================================================

keep_params <- c(
  "mu_alpha",
  "mu_beta",
  "mu_lambda",
  "mu_delta",
  "mu_eta"
)

param_labels <- c(
  "mu_alpha"  = "mu[alpha]",
  "mu_beta"   = "mu[beta]",
  "mu_lambda" = "mu[lambda]",
  "mu_delta"  = "mu[delta]",
  "mu_eta"    = "mu[eta]"
)

param_clean_names <- c(
  "mu_alpha"  = "alpha",
  "mu_beta"   = "beta",
  "mu_lambda" = "lambda",
  "mu_delta"  = "delta",
  "mu_eta"    = "eta"
)

# Reference values:
# alpha  = 1: veridical reliability slope
# beta   = 0: no reliability/evidence offset
# lambda = 0: no sequential recency/primacy effect
# delta  = 1: normative Bayesian updating
# eta    = 0: no confirmation bias

reference_values <- c(
  "mu_alpha"  = 1,
  "mu_beta"   = 0,
  "mu_lambda" = 0,
  "mu_delta"  = 1,
  "mu_eta"    = 0
)

# =========================================================================
# 4. EXTRACT POSTERIOR DRAWS
# =========================================================================
# Handles both cases:
# 1. Fits already have transformed generated quantities:
#    mu_alpha, mu_beta, mu_lambda, mu_delta, mu_eta
#
# 2. Fits only have raw mu_pr[1:5], which need transforming:
#    alpha  = pnorm(mu_pr[1]) * 6
#    beta   = mu_pr[2]
#    lambda = pnorm(mu_pr[3])
#    delta  = pnorm(mu_pr[4]) * 2
#    eta    = mu_pr[5]

extract_mu <- function(fit, group_label, alpha_scale = 6) {
  
  d <- as_draws_df(fit$draws())
  
  # Case 1: transformed parameters already exist
  if (all(keep_params %in% colnames(d))) {
    
    mu_df <- d %>%
      select(all_of(keep_params))
    
  } else {
    
    # Case 2: fall back to raw mu_pr columns
    mu_cols_bracket <- paste0("mu_pr[", 1:5, "]")
    mu_cols_dot     <- paste0("mu_pr.", 1:5, ".")
    
    if (all(mu_cols_bracket %in% colnames(d))) {
      actual_cols <- mu_cols_bracket
    } else if (all(mu_cols_dot %in% colnames(d))) {
      actual_cols <- mu_cols_dot
    } else {
      actual_cols <- colnames(d)[grepl("^mu_pr", colnames(d))]
      actual_cols <- actual_cols[1:5]
    }
    
    if (length(actual_cols) < 5) {
      stop("Could not find 5 group-level mu parameters for: ", group_label)
    }
    
    mu_df <- d %>%
      select(all_of(actual_cols[1:5]))
    
    colnames(mu_df) <- keep_params
    
    mu_df <- mu_df %>%
      mutate(
        mu_alpha  = pnorm(mu_alpha) * alpha_scale,
        mu_lambda = pnorm(mu_lambda),
        mu_delta  = pnorm(mu_delta) * 2
        # beta and eta remain untransformed
      )
  }
  
  mu_df %>%
    mutate(group = group_label) %>%
    pivot_longer(
      cols = all_of(keep_params),
      names_to = "parameter",
      values_to = "value"
    )
}

# =========================================================================
# 5. COMBINE ALL FIVE GROUPS
# =========================================================================

group_levels <- c(
  "Implicit Unaware Base Rate",
  "Implicit Aware Base Rate",
  "Explicit Undirected Base Rate",
  "Explicit True Base Rate",
  "Explicit Deceptive Base Rate"
)

draws_all <- bind_rows(
  extract_mu(fit_unaware,   "Implicit Unaware Base Rate"),
  extract_mu(fit_aware,     "Implicit Aware Base Rate"),
  extract_mu(fit_explicit,  "Explicit Undirected Base Rate"),
  extract_mu(fit_truthful,  "Explicit True Base Rate"),
  extract_mu(fit_deceptive, "Explicit Deceptive Base Rate")
) %>%
  mutate(
    group = factor(group, levels = group_levels),
    parameter = factor(parameter, levels = keep_params)
  )

cat("\nGroups included:\n")
print(table(draws_all$group))

cat("\nParameters included:\n")
print(table(draws_all$parameter))

# =========================================================================
# 6. GROUP COLOURS
# =========================================================================

group_cols <- c(
  "Implicit Unaware Base Rate"      = "#F4A3A3",  # light red
  "Implicit Aware Base Rate"        = "#8FD694",  # light green
  "Explicit Undirected Base Rate"   = "#E69F00",  # orange
  "Explicit True Base Rate"         = "#1B7837", # dark green
  "Explicit Deceptive Base Rate"    = "#8B0000"   # dark red
)
# =========================================================================
# 7. POSTERIOR DISTRIBUTION PLOT — ALL FIVE PARAMETERS
# =========================================================================

p_distributions <- ggplot(
  draws_all,
  aes(x = value, fill = group, colour = group)
) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 60,
    position = "identity",
    alpha = 0.42,
    linewidth = 0.25
  ) +
  facet_wrap(
    ~parameter,
    scales = "free",
    nrow = 1,
    labeller = as_labeller(param_labels, default = label_parsed)
  ) +
  scale_fill_manual(
    values = group_cols,
    guide = guide_legend(ncol = 1)
  ) +
  scale_colour_manual(
    values = group_cols,
    guide = guide_legend(ncol = 1)
  ) +
  labs(
    title = "Transformed Posterior Parameter Distributions",
    subtitle = "Five groups, five-parameter model",
    x = "Parameter value",
    y = "Density",
    fill = NULL,
    colour = NULL
  ) +
  theme_bw(base_size = 15) +
  theme(
    panel.grid.major = element_line(colour = "grey92"),
    panel.grid.minor = element_blank(),
    strip.background = element_rect(fill = "grey88", colour = "grey70"),
    strip.text = element_text(face = "bold", size = 14),
    plot.title = element_text(face = "bold", size = 17),
    plot.subtitle = element_text(size = 12, colour = "grey35"),
    axis.title = element_text(face = "bold", size = 14),
    axis.text = element_text(size = 11),
    legend.position = "bottom",
    legend.text = element_text(size = 12, face = "bold"),
    legend.key.size = unit(0.9, "lines")
  )

print(p_distributions)

ggsave(
  filename = file.path(fig_dir, "posterior_distributions_5groups_base_rate_colours.png"),
  plot = p_distributions,
  width = 16,
  height = 5,
  dpi = 300
)

ggsave(
  filename = file.path(fig_dir, "posterior_distributions_5groups_base_rate_colours.pdf"),
  plot = p_distributions,
  width = 16,
  height = 5
)

# =========================================================================
# 8. CREATE WIDE DRAWS TABLE
# =========================================================================
# Important fix:
# Need a draw_id before pivot_wider().
# Otherwise pivot_wider creates list-columns, which causes:
# non-numeric argument to binary operator

draws_wide <- draws_all %>%
  group_by(group, parameter) %>%
  mutate(draw_id = row_number()) %>%
  ungroup() %>%
  mutate(parameter = as.character(parameter)) %>%
  pivot_wider(
    id_cols = c(group, draw_id),
    names_from = parameter,
    values_from = value
  ) %>%
  mutate(
    mu_alpha  = as.numeric(mu_alpha),
    mu_beta   = as.numeric(mu_beta),
    mu_lambda = as.numeric(mu_lambda),
    mu_delta  = as.numeric(mu_delta),
    mu_eta    = as.numeric(mu_eta)
  )

cat("\nStructure of draws_wide:\n")
str(draws_wide)

# =========================================================================
# 9. RELIABILITY DISTORTION PLOT
# =========================================================================

distort_reliability <- function(x_percent, alpha, beta) {
  
  p <- x_percent / 100
  epsv <- 1e-6
  p <- pmin(pmax(p, epsv), 1 - epsv)
  
  logitp <- log(p / (1 - p))
  distorted_p <- 1 / (1 + exp(-(alpha * logitp + beta)))
  
  distorted_p * 100
}

big_plot_theme <- theme_bw(base_size = 12) +
  theme(
    axis.title.x = element_text(size = 15, face = "bold"),
    axis.title.y = element_text(size = 15, face = "bold"),
    axis.text = element_text(size = 11),
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12),
    legend.title = element_blank(),
    legend.text = element_text(size = 10),
    legend.background = element_blank(),
    legend.key.size = unit(0.55, "cm"),
    legend.spacing.x = unit(0.15, "cm"),
    legend.spacing.y = unit(0.05, "cm"),
    panel.grid.minor = element_blank()
  )

x_grid <- seq(0, 100, length.out = 501)
x_dots <- c(50, 55, 65)

distortion_curve_one_group <- function(df_group, x_vals) {
  
  alpha_draws <- df_group$mu_alpha
  beta_draws  <- df_group$mu_beta
  
  map_dfr(x_vals, function(xi) {
    
    y_draws <- distort_reliability(xi, alpha_draws, beta_draws)
    
    tibble(
      x = xi,
      mean = mean(y_draws, na.rm = TRUE),
      l95 = quantile(y_draws, 0.025, na.rm = TRUE),
      u95 = quantile(y_draws, 0.975, na.rm = TRUE)
    )
  })
}

distortion_df <- draws_wide %>%
  group_split(group) %>%
  map_dfr(function(df_g) {
    
    gname <- as.character(df_g$group[1])
    
    distortion_curve_one_group(df_g, x_grid) %>%
      mutate(group = gname)
  }) %>%
  mutate(group = factor(group, levels = group_levels))

distortion_dots_df <- draws_wide %>%
  group_split(group) %>%
  map_dfr(function(df_g) {
    
    gname <- as.character(df_g$group[1])
    
    distortion_curve_one_group(df_g, x_dots) %>%
      mutate(group = gname)
  }) %>%
  mutate(group = factor(group, levels = group_levels))

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
    aes(x = x, ymin = l95, ymax = u95, fill = group),
    alpha = 0.12,
    colour = NA
  ) +
  geom_line(
    data = distortion_df,
    aes(x = x, y = mean, color = group),
    linewidth = 1.4
  ) +
  geom_errorbar(
    data = distortion_dots_df,
    aes(x = x, ymin = l95, ymax = u95, color = group),
    width = 0,
    linewidth = 0.7
  ) +
  geom_point(
    data = distortion_dots_df,
    aes(x = x, y = mean, fill = group),
    shape = 21,
    size = 3.2,
    color = "black",
    stroke = 0.5
  ) +
  scale_color_manual(values = group_cols) +
  scale_fill_manual(values = group_cols) +
  coord_fixed() +
  labs(
    title = "Reliability distortion",
    x = "True reliability (%)",
    y = "Distorted reliability (%)"
  ) +
  xlim(0, 100) +
  ylim(0, 100) +
  big_plot_theme +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal"
  )

print(p_distortion)

ggsave(
  filename = file.path(fig_dir, "reliability_distortion_5groups_base_rate_colours.png"),
  plot = p_distortion,
  width = 7,
  height = 6,
  dpi = 300
)

# =========================================================================
# 10. SEQUENTIAL WEIGHTING PLOT
# =========================================================================
# Plot all six sample positions.
# Position 6 is the reference position and is fixed to a relative weight of 1.

Smax <- 6
positions <- 1:Smax

weights_one_group <- function(df_group, positions, Smax = 6) {
  
  lambda_draws <- df_group$mu_lambda
  
  map_dfr(positions, function(pos) {
    
    # Exponential sequential weight:
    # position 6 = exp(lambda * (6 - 6)) = 1
    w_draws <- exp(lambda_draws * (pos - Smax))
    
    tibble(
      position = pos,
      mean = mean(w_draws, na.rm = TRUE),
      l95 = quantile(w_draws, 0.025, na.rm = TRUE),
      u95 = quantile(w_draws, 0.975, na.rm = TRUE)
    )
  })
}

weights_df <- draws_wide %>%
  group_split(group) %>%
  map_dfr(function(df_g) {
    
    gname <- as.character(df_g$group[1])
    
    weights_one_group(
      df_group = df_g,
      positions = positions,
      Smax = Smax
    ) %>%
      mutate(group = gname)
  }) %>%
  mutate(
    group = factor(group, levels = group_levels)
  )

p_weights <- ggplot(
  weights_df,
  aes(
    x = position,
    y = mean,
    color = group,
    group = group
  )
) +
  geom_line(linewidth = 1.4) +
  geom_errorbar(
    aes(ymin = l95, ymax = u95),
    width = 0.15,
    linewidth = 0.7
  ) +
  geom_point(
    aes(fill = group),
    shape = 21,
    size = 3.2,
    color = "black",
    stroke = 0.5
  ) +
  geom_hline(
    yintercept = 1,
    linetype = "dashed",
    linewidth = 0.9,
    color = "grey40"
  ) +
  scale_color_manual(values = group_cols) +
  scale_fill_manual(values = group_cols) +
  scale_x_continuous(
    breaks = 1:6,
    limits = c(0.5, 6.5)
  ) +
  labs(
    title = "Sequential weighting",
    x = "Sequence position",
    y = "Relative sequential weight"
  ) +
  big_plot_theme +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    aspect.ratio = 1
  )

print(p_weights)

ggsave(
  filename = file.path(
    fig_dir,
    "sequential_weighting_5groups_base_rate_colours.png"
  ),
  plot = p_weights,
  width = 7,
  height = 6,
  dpi = 300
)

# =========================================================================
# PAIRWISE POSTERIOR CONTRASTS FOR LAMBDA
# =========================================================================

lambda_pairwise <- pairwise_stats %>%
  filter(parameter == "lambda") %>%
  mutate(
    evidence_for_difference = case_when(
      l95_difference > 0 | u95_difference < 0 ~ "95% CrI excludes 0",
      p_group_1_gt_group_2 > .95 | p_group_1_lt_group_2 > .95 ~ "Strong directional posterior probability",
      TRUE ~ "No strong evidence for a reliable difference"
    )
  )

cat("\n=========================================================\n")
cat("PAIRWISE POSTERIOR CONTRASTS FOR LAMBDA\n")
cat("=========================================================\n")

print(lambda_pairwise, n = Inf, width = Inf)

# =========================================================================
# 11. POSTERIOR STATS — ALL FIVE PARAMETERS × ALL FIVE GROUPS
# =========================================================================

all_param_stats <- draws_all %>%
  group_by(group, parameter) %>%
  summarise(
    mean   = mean(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    l95    = quantile(value, 0.025, na.rm = TRUE),
    u95    = quantile(value, 0.975, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    parameter = recode(
      as.character(parameter),
      "mu_alpha"  = "alpha",
      "mu_beta"   = "beta",
      "mu_lambda" = "lambda",
      "mu_delta"  = "delta",
      "mu_eta"    = "eta"
    )
  ) %>%
  arrange(group, parameter)

cat("\n=========================================================\n")
cat("POSTERIOR MEANS AND 95% CREDIBLE INTERVALS\n")
cat("ALL FIVE PARAMETERS × ALL FIVE GROUPS\n")
cat("=========================================================\n")

print(all_param_stats, n = Inf)

# =========================================================================
# 12. REPORTING TABLE: MEAN [95% CrI]
# =========================================================================

report_table <- all_param_stats %>%
  mutate(
    summary = sprintf("%.3f [%.3f, %.3f]", mean, l95, u95)
  ) %>%
  select(group, parameter, summary) %>%
  pivot_wider(names_from = parameter, values_from = summary)

cat("\n=========================================================\n")
cat("REPORTING TABLE: MEAN [95% CrI]\n")
cat("=========================================================\n")

print(report_table, n = Inf)

# =========================================================================
# 13. POSTERIOR PROBABILITIES AGAINST REFERENCE VALUES
# =========================================================================

reference_stats <- draws_all %>%
  mutate(
    parameter_label = recode(
      as.character(parameter),
      "mu_alpha"  = "alpha",
      "mu_beta"   = "beta",
      "mu_lambda" = "lambda",
      "mu_delta"  = "delta",
      "mu_eta"    = "eta"
    ),
    reference = case_when(
      parameter == "mu_alpha"  ~ 1,
      parameter == "mu_beta"   ~ 0,
      parameter == "mu_lambda" ~ 0,
      parameter == "mu_delta"  ~ 1,
      parameter == "mu_eta"    ~ 0,
      TRUE ~ NA_real_
    )
  ) %>%
  group_by(group, parameter_label, reference) %>%
  summarise(
    mean = mean(value, na.rm = TRUE),
    l95  = quantile(value, 0.025, na.rm = TRUE),
    u95  = quantile(value, 0.975, na.rm = TRUE),
    p_gt_reference = mean(value > reference, na.rm = TRUE),
    p_lt_reference = mean(value < reference, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(group, parameter_label)

cat("\n=========================================================\n")
cat("POSTERIOR PROBABILITIES AGAINST REFERENCE VALUES\n")
cat("=========================================================\n")

print(reference_stats, n = Inf)

# =========================================================================
# 14. CHECK TABLE
# =========================================================================
# This should show 1 for each parameter in each group.

check_table <- all_param_stats %>%
  count(group, parameter) %>%
  pivot_wider(
    names_from = parameter,
    values_from = n,
    values_fill = 0
  )

cat("\n=========================================================\n")
cat("CHECK: SHOULD SHOW 1 FOR EACH PARAMETER IN EACH GROUP\n")
cat("=========================================================\n")

print(check_table, n = Inf)

## =========================================================================
# 15. PAIRWISE POSTERIOR CONTRASTS
# =========================================================================

pairwise_exceedance <- function(draws_df, param_name) {
  
  df_param <- draws_df %>%
    filter(
      parameter == param_name,
      is.finite(value)
    ) %>%
    mutate(
      group = as.character(group)
    ) %>%
    select(group, value)
  
  # Keep groups in the intended reporting order
  group_order <- c(
    "Implicit Unaware Base Rate",
    "Implicit Aware Base Rate",
    "Explicit Undirected Base Rate",
    "Explicit True Base Rate",
    "Explicit Deceptive Base Rate"
  )
  
  group_names <- group_order[group_order %in% unique(df_param$group)]
  
  if (length(group_names) < 2) {
    stop(
      paste0(
        "Fewer than two groups found for parameter: ",
        param_name
      )
    )
  }
  
  # All unique pairwise group comparisons
  group_pairs <- combn(group_names, 2, simplify = FALSE)
  
  out <- vector("list", length(group_pairs))
  
  for (k in seq_along(group_pairs)) {
    
    g1 <- group_pairs[[k]][1]
    g2 <- group_pairs[[k]][2]
    
    x1 <- df_param %>%
      filter(group == g1) %>%
      pull(value)
    
    x2 <- df_param %>%
      filter(group == g2) %>%
      pull(value)
    
    if (length(x1) == 0 || length(x2) == 0) {
      stop(
        paste0(
          "No posterior draws found for comparison: ",
          g1, " vs ", g2,
          " for ", param_name
        )
      )
    }
    
    # Use the same number of posterior draws from both groups
    n_draws <- min(length(x1), length(x2))
    
    # The group fits are separate, so independently permute the
    # posterior draws before constructing the difference distribution
    x1 <- sample(x1, size = n_draws, replace = FALSE)
    x2 <- sample(x2, size = n_draws, replace = FALSE)
    
    diff_draws <- x1 - x2
    
    ci <- quantile(
      diff_draws,
      probs = c(0.025, 0.975),
      na.rm = TRUE,
      names = FALSE
    )
    
    p_gt <- mean(diff_draws > 0, na.rm = TRUE)
    p_lt <- mean(diff_draws < 0, na.rm = TRUE)
    
    out[[k]] <- tibble(
      parameter = param_name,
      
      group_1 = g1,
      group_2 = g2,
      
      mean_group_1 = mean(x1, na.rm = TRUE),
      mean_group_2 = mean(x2, na.rm = TRUE),
      
      # Positive = group_1 > group_2
      # Negative = group_1 < group_2
      mean_difference = mean(diff_draws, na.rm = TRUE),
      
      l95_difference = ci[1],
      u95_difference = ci[2],
      
      p_group_1_gt_group_2 = p_gt,
      p_group_1_lt_group_2 = p_lt,
      
      # Highest posterior probability in either direction
      posterior_probability = pmax(p_gt, p_lt),
      
      # TRUE when the 95% credible interval excludes zero
      cri_excludes_zero =
        (ci[1] > 0) | (ci[2] < 0)
    )
  }
  
  bind_rows(out)
}


# -------------------------------------------------------------------------
# Make results reproducible
# -------------------------------------------------------------------------

set.seed(12345)


# -------------------------------------------------------------------------
# Run pairwise contrasts for all five parameters
# -------------------------------------------------------------------------

pairwise_stats <- bind_rows(
  pairwise_exceedance(draws_all, "mu_alpha"),
  pairwise_exceedance(draws_all, "mu_beta"),
  pairwise_exceedance(draws_all, "mu_lambda"),
  pairwise_exceedance(draws_all, "mu_delta"),
  pairwise_exceedance(draws_all, "mu_eta")
) %>%
  mutate(
    parameter = recode(
      parameter,
      "mu_alpha"  = "alpha",
      "mu_beta"   = "beta",
      "mu_lambda" = "lambda",
      "mu_delta"  = "delta",
      "mu_eta"    = "eta"
    )
  ) %>%
  arrange(
    factor(
      parameter,
      levels = c(
        "alpha",
        "beta",
        "lambda",
        "delta",
        "eta"
      )
    ),
    group_1,
    group_2
  )


# =========================================================================
# PRINT ALL PAIRWISE CONTRASTS
# =========================================================================

cat("\n=========================================================\n")
cat("ALL PAIRWISE POSTERIOR CONTRASTS\n")
cat("Difference = Group 1 - Group 2\n")
cat("=========================================================\n\n")

print(pairwise_stats, n = Inf, width = Inf)


# =========================================================================
# PRINT ONLY CONTRASTS WHOSE 95% CrI EXCLUDES ZERO
# =========================================================================

clear_pairwise <- pairwise_stats %>%
  filter(cri_excludes_zero) %>%
  arrange(parameter, desc(posterior_probability))

cat("\n=========================================================\n")
cat("PAIRWISE CONTRASTS WITH 95% CrI EXCLUDING ZERO\n")
cat("=========================================================\n\n")

if (nrow(clear_pairwise) == 0) {
  
  cat("None.\n")
  
} else {
  
  print(clear_pairwise, n = Inf, width = Inf)
  
}


# =========================================================================
# DELTA CONTRASTS ONLY
# =========================================================================

delta_pairwise <- pairwise_stats %>%
  filter(parameter == "delta")

cat("\n=========================================================\n")
cat("DELTA PAIRWISE POSTERIOR CONTRASTS\n")
cat("=========================================================\n\n")

print(delta_pairwise, n = Inf, width = Inf)


# =========================================================================
# EXPLICIT TRUE DELTA VS EVERY OTHER GROUP
# =========================================================================

true_delta_contrasts <- delta_pairwise %>%
  filter(
    group_1 == "Explicit True Base Rate" |
      group_2 == "Explicit True Base Rate"
  ) %>%
  mutate(
    
    # Put the contrast into the intuitive direction:
    # Explicit True - Other Group
    true_minus_other = case_when(
      group_1 == "Explicit True Base Rate" ~ mean_difference,
      group_2 == "Explicit True Base Rate" ~ -mean_difference
    ),
    
    true_minus_other_l95 = case_when(
      group_1 == "Explicit True Base Rate" ~ l95_difference,
      group_2 == "Explicit True Base Rate" ~ -u95_difference
    ),
    
    true_minus_other_u95 = case_when(
      group_1 == "Explicit True Base Rate" ~ u95_difference,
      group_2 == "Explicit True Base Rate" ~ -l95_difference
    ),
    
    p_true_gt_other = case_when(
      group_1 == "Explicit True Base Rate" ~
        p_group_1_gt_group_2,
      
      group_2 == "Explicit True Base Rate" ~
        p_group_1_lt_group_2
    ),
    
    comparison_group = case_when(
      group_1 == "Explicit True Base Rate" ~ group_2,
      group_2 == "Explicit True Base Rate" ~ group_1
    )
  ) %>%
  select(
    comparison_group,
    true_minus_other,
    true_minus_other_l95,
    true_minus_other_u95,
    p_true_gt_other
  )

cat("\n=========================================================\n")
cat("EXPLICIT TRUE DELTA VS EACH OTHER GROUP\n")
cat("Difference = Explicit True - Other Group\n")
cat("=========================================================\n\n")

print(true_delta_contrasts, n = Inf, width = Inf)


# =========================================================================
# ETA CONTRASTS ONLY
# =========================================================================

eta_pairwise <- pairwise_stats %>%
  filter(parameter == "eta")

cat("\n=========================================================\n")
cat("ETA PAIRWISE POSTERIOR CONTRASTS\n")
cat("=========================================================\n\n")

print(eta_pairwise, n = Inf, width = Inf)
# =========================================================================
# 16. PAIRWISE TESTS FOR DELTA AND ETA ONLY
# =========================================================================

delta_eta_pairwise <- pairwise_stats %>%
  filter(parameter %in% c("delta", "eta"))

cat("\n=========================================================\n")
cat("PAIRWISE TESTS FOR DELTA AND ETA ONLY\n")
cat("=========================================================\n")

print(delta_eta_pairwise, n = Inf)

# =========================================================================
# 17. END
# =========================================================================

cat("\nDone. Main stats objects created:\n")
cat("1. all_param_stats\n")
cat("2. report_table\n")
cat("3. reference_stats\n")
cat("4. pairwise_stats\n")
cat("5. delta_eta_pairwise\n")


# =========================================================================
# 12B. CLEAN TABLE FOR RESULTS WRITE-UP
# =========================================================================
# This gives the exact M and 95% CrI for each parameter in each group.
# Use this to update the computational-modelling results paragraph.

writeup_table <- all_param_stats %>%
  mutate(
    mean_round = sprintf("%.2f", mean),
    cri_round  = sprintf("[%.2f, %.2f]", l95, u95),
    result     = paste0("M = ", mean_round, ", 95% CrI ", cri_round)
  ) %>%
  select(group, parameter, result) %>%
  pivot_wider(
    names_from = parameter,
    values_from = result
  )

cat("\n=========================================================\n")
cat("WRITE-UP TABLE: M AND 95% CrI FOR EACH GROUP × PARAMETER\n")
cat("=========================================================\n")

print(writeup_table, n = Inf, width = Inf)








# =========================================================================
# 16. EXTRACT PARTICIPANT-LEVEL PARAMETER ESTIMATES
# =========================================================================
#
# The Stan model saves:
#
# params[n, 1] = alpha
# params[n, 2] = beta
# params[n, 3] = lambda
# params[n, 4] = delta
# params[n, 5] = eta
#
# These are already transformed onto the final parameter scales.
# We therefore take the posterior mean of each participant's parameter.
# =========================================================================


extract_participant_params <- function(fit, group_label) {
  
  # ---------------------------------------------------------
  # Get summaries of participant-level generated quantities
  # ---------------------------------------------------------
  
  s <- fit$summary(
    variables = "params"
  ) %>%
    as_tibble()
  
  # Expected variable names:
  # params[1,1], params[1,2], ..., params[N,5]
  
  parsed <- stringr::str_match(
    s$variable,
    "^params\\[([0-9]+),([0-9]+)\\]$"
  )
  
  participant_index <- as.integer(parsed[, 2])
  parameter_index   <- as.integer(parsed[, 3])
  
  out <- tibble(
    participant_index = participant_index,
    parameter_index   = parameter_index,
    mean              = s$mean
  ) %>%
    filter(
      !is.na(participant_index),
      !is.na(parameter_index)
    )
  
  if (nrow(out) == 0) {
    stop(
      paste0(
        "Could not find participant-level params for group: ",
        group_label,
        "\nCheck names with:\n",
        "fit$summary(variables = 'params')"
      )
    )
  }
  
  # ---------------------------------------------------------
  # Convert parameter number to parameter name
  # ---------------------------------------------------------
  
  out <- out %>%
    mutate(
      parameter = case_when(
        parameter_index == 1 ~ "alpha",
        parameter_index == 2 ~ "beta",
        parameter_index == 3 ~ "lambda",
        parameter_index == 4 ~ "delta",
        parameter_index == 5 ~ "eta",
        TRUE ~ NA_character_
      ),
      
      group = group_label,
      
      # Unique participant identifier across all groups
      ID = paste0(
        group_label,
        "_",
        participant_index
      )
    ) %>%
    filter(
      !is.na(parameter)
    ) %>%
    select(
      ID,
      participant_index,
      group,
      parameter,
      mean
    )
  
  return(out)
}


# =========================================================================
# COMBINE PARTICIPANT-LEVEL PARAMETERS FROM ALL FIVE GROUPS
# =========================================================================

participant_param_stats <- bind_rows(
  
  extract_participant_params(
    fit_unaware,
    "Implicit Unaware Base Rate"
  ),
  
  extract_participant_params(
    fit_aware,
    "Implicit Aware Base Rate"
  ),
  
  extract_participant_params(
    fit_explicit,
    "Explicit Undirected Base Rate"
  ),
  
  extract_participant_params(
    fit_truthful,
    "Explicit True Base Rate"
  ),
  
  extract_participant_params(
    fit_deceptive,
    "Explicit Deceptive Base Rate"
  )
)


# =========================================================================
# CHECK PARTICIPANT-LEVEL EXTRACTION
# =========================================================================

cat("\n=========================================================\n")
cat("PARTICIPANT-LEVEL PARAMETER EXTRACTION\n")
cat("=========================================================\n\n")

cat("Number of participants per group:\n\n")

print(
  participant_param_stats %>%
    distinct(group, ID) %>%
    count(group, name = "N")
)

cat("\nNumber of estimates per group × parameter:\n\n")

print(
  participant_param_stats %>%
    count(group, parameter),
  n = Inf
)

cat("\nFirst few participant estimates:\n\n")

print(
  participant_param_stats %>%
    arrange(group, participant_index, parameter) %>%
    head(20)
)


# =========================================================================
# 17. RANDOMISATION TEST FOR ONE PAIR OF GROUPS
# =========================================================================

randomisation_test <- function(
    data,
    param_name,
    group_1,
    group_2,
    n_perm = 10000
) {
  
  dat <- data %>%
    filter(
      parameter == param_name,
      group %in% c(group_1, group_2),
      is.finite(mean)
    ) %>%
    mutate(
      group = as.character(group)
    )
  
  # Check both groups are present
  if (!all(c(group_1, group_2) %in% unique(dat$group))) {
    stop(
      paste0(
        "One or both groups missing for ",
        param_name,
        ": ",
        group_1,
        " vs ",
        group_2
      )
    )
  }
  
  # ---------------------------------------------------------
  # Observed difference
  # Group 1 - Group 2
  # ---------------------------------------------------------
  
  observed_diff <-
    mean(
      dat$mean[dat$group == group_1],
      na.rm = TRUE
    ) -
    mean(
      dat$mean[dat$group == group_2],
      na.rm = TRUE
    )
  
  mean_g1 <- mean(
    dat$mean[dat$group == group_1],
    na.rm = TRUE
  )
  
  mean_g2 <- mean(
    dat$mean[dat$group == group_2],
    na.rm = TRUE
  )
  
  # ---------------------------------------------------------
  # Permutation/randomisation distribution
  #
  # Parameter estimates stay fixed.
  # Group labels are randomly shuffled.
  # Group sample sizes therefore remain unchanged.
  # ---------------------------------------------------------
  
  perm_diffs <- replicate(
    n_perm,
    {
      
      shuffled_group <- sample(
        dat$group,
        replace = FALSE
      )
      
      mean(
        dat$mean[shuffled_group == group_1],
        na.rm = TRUE
      ) -
        mean(
          dat$mean[shuffled_group == group_2],
          na.rm = TRUE
        )
    }
  )
  
  # ---------------------------------------------------------
  # Two-sided randomisation p-value
  # ---------------------------------------------------------
  
  p_value <-
    (
      sum(
        abs(perm_diffs) >= abs(observed_diff)
      ) + 1
    ) /
    (n_perm + 1)
  
  tibble(
    parameter = param_name,
    group_1 = group_1,
    group_2 = group_2,
    mean_group_1 = mean_g1,
    mean_group_2 = mean_g2,
    observed_difference = observed_diff,
    randomisation_p = p_value
  )
}


# =========================================================================
# 18. RUN ALL 10 PAIRWISE COMPARISONS FOR ONE PARAMETER
# =========================================================================

all_pairwise_randomisation <- function(
    data,
    param_name,
    group_order,
    n_perm = 10000
) {
  
  pairs <- combn(
    group_order,
    2,
    simplify = FALSE
  )
  
  results <- lapply(
    pairs,
    function(pair) {
      
      randomisation_test(
        data = data,
        param_name = param_name,
        group_1 = pair[1],
        group_2 = pair[2],
        n_perm = n_perm
      )
    }
  )
  
  bind_rows(results)
}


# =========================================================================
# 19. GROUP ORDER
# =========================================================================

group_order <- c(
  "Implicit Unaware Base Rate",
  "Implicit Aware Base Rate",
  "Explicit Undirected Base Rate",
  "Explicit True Base Rate",
  "Explicit Deceptive Base Rate"
)


# =========================================================================
# 20. RUN ALL FIVE PARAMETERS
# =========================================================================

set.seed(12345)

randomisation_stats <- bind_rows(
  
  all_pairwise_randomisation(
    participant_param_stats,
    "alpha",
    group_order,
    n_perm = 10000
  ),
  
  all_pairwise_randomisation(
    participant_param_stats,
    "beta",
    group_order,
    n_perm = 10000
  ),
  
  all_pairwise_randomisation(
    participant_param_stats,
    "lambda",
    group_order,
    n_perm = 10000
  ),
  
  all_pairwise_randomisation(
    participant_param_stats,
    "delta",
    group_order,
    n_perm = 10000
  ),
  
  all_pairwise_randomisation(
    participant_param_stats,
    "eta",
    group_order,
    n_perm = 10000
  )
  
) %>%
  
  # Holm correction across the 10 pairwise comparisons
  # separately for each parameter
  group_by(parameter) %>%
  
  mutate(
    p_holm = p.adjust(
      randomisation_p,
      method = "holm"
    )
  ) %>%
  
  ungroup()


# =========================================================================
# 21. PRINT ALL 50 RANDOMISATION TESTS
# =========================================================================

cat("\n=========================================================\n")
cat("ALL PAIRWISE RANDOMISATION TESTS\n")
cat("10 comparisons per parameter\n")
cat("Holm correction applied within parameter\n")
cat("=========================================================\n\n")

print(
  randomisation_stats,
  n = Inf,
  width = Inf
)


# =========================================================================
# 22. PRINT RAW p < .05
# =========================================================================

cat("\n=========================================================\n")
cat("PAIRWISE RANDOMISATION TESTS WITH RAW p < .05\n")
cat("=========================================================\n\n")

print(
  randomisation_stats %>%
    filter(randomisation_p < .05) %>%
    arrange(parameter, randomisation_p),
  n = Inf,
  width = Inf
)


# =========================================================================
# 23. PRINT HOLM-CORRECTED p < .05
# =========================================================================

cat("\n=========================================================\n")
cat("PAIRWISE RANDOMISATION TESTS WITH HOLM p < .05\n")
cat("=========================================================\n\n")

holm_significant <- randomisation_stats %>%
  filter(p_holm < .05) %>%
  arrange(parameter, p_holm)

if (nrow(holm_significant) == 0) {
  
  cat("None.\n")
  
} else {
  
  print(
    holm_significant,
    n = Inf,
    width = Inf
  )
}


# =========================================================================
# 24. SAVE FULL RANDOMISATION TABLE
# =========================================================================

write_csv(
  randomisation_stats,
  "results/pairwise_parameter_randomisation_tests.csv"
)

cat("\nSaved full table to:\n")
cat("results/pairwise_parameter_randomisation_tests.csv\n")















# =========================================================================
#  FIGURE 5 — DELTA AND ETA BOXPLOTS
#  Five groups:
#  Implicit Unaware Base Rate
#  Implicit Aware Base Rate
#  Explicit Undirected Base Rate
#  Explicit True Base Rate
#  Explicit Deceptive Base Rate
# =========================================================================

library(tidyverse)
library(posterior)
library(ggplot2)
library(grid)

# =========================================================
# 1. Helper function
# =========================================================

extract_delta_eta_wide <- function(fit) {
  
  d <- as_draws_df(fit$draws())
  
  # Case 1: transformed parameters already exist
  if (all(c("mu_delta", "mu_eta") %in% colnames(d))) {
    
    out <- d %>%
      select(mu_delta, mu_eta)
    
  } else {
    
    # Case 2: fall back to raw mu_pr columns
    mu_cols <- colnames(d)[grepl("^mu_pr(\\.|\\[)", colnames(d))]
    
    if (length(mu_cols) < 5) {
      stop("Could not find enough mu_pr columns to extract mu_delta and mu_eta.")
    }
    
    out <- d %>%
      select(all_of(mu_cols[c(4, 5)]))
    
    colnames(out) <- c("mu_delta", "mu_eta")
    
    out <- out %>%
      mutate(
        mu_delta = pnorm(mu_delta) * 2.0
        # eta remains untransformed
      )
  }
  
  as_tibble(out)
}

# =========================================================
# 2. Paths
# =========================================================

base_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias"
fits_dir <- file.path(base_dir, "stan/results/fits/exp11_unaware")
fig_dir  <- file.path(base_dir, "results/figures")

if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

# =========================================================
# 3. Load fits
# =========================================================

load(file.path(fits_dir, "fit_trunc_global_eta_unaware_exp11.rdata"))
fit_unaware <- fit

load(file.path(fits_dir, "fit_trunc_global_eta_aware_exp11.rdata"))
fit_aware <- fit

load(file.path(fits_dir, "fit_trunc_global_eta_aware_exp12.rdata"))
fit_explicit <- fit

load(file.path(fits_dir, "fit_trunc_global_eta_truthful_exp13.rdata"))
fit_true_direction <- fit

load(file.path(fits_dir, "fit_trunc_global_eta_deceptive_exp13.rdata"))
fit_deceptive_direction <- fit

rm(fit)

# =========================================================
# 4. Extract delta and eta posterior draws
# =========================================================

d_unaware   <- extract_delta_eta_wide(fit_unaware)
d_aware     <- extract_delta_eta_wide(fit_aware)
d_explicit  <- extract_delta_eta_wide(fit_explicit)
d_true      <- extract_delta_eta_wide(fit_true_direction)
d_deceptive <- extract_delta_eta_wide(fit_deceptive_direction)

# =========================================================
# 5. Combine all five groups
# =========================================================

group_levels <- c(
  "Implicit Unaware Base Rate",
  "Implicit Aware Base Rate",
  "Explicit Undirected Base Rate",
  "Explicit True Base Rate",
  "Explicit Deceptive Base Rate"
)

plot_data <- bind_rows(
  d_unaware   %>% mutate(group = "Implicit Unaware Base Rate"),
  d_aware     %>% mutate(group = "Implicit Aware Base Rate"),
  d_explicit  %>% mutate(group = "Explicit Undirected Base Rate"),
  d_true      %>% mutate(group = "Explicit True Base Rate"),
  d_deceptive %>% mutate(group = "Explicit Deceptive Base Rate")
) %>%
  mutate(group = factor(group, levels = group_levels)) %>%
  pivot_longer(
    cols = c(mu_delta, mu_eta),
    names_to = "parameter",
    values_to = "value"
  ) %>%
  mutate(
    parameter = recode(
      parameter,
      "mu_delta" = "Delta",
      "mu_eta"   = "Eta"
    )
  )

cat("\nGroups included:\n")
print(table(plot_data$group))

cat("\nParameters included:\n")
print(table(plot_data$parameter))

# =========================================================
# 6. Colours and axis labels
# =========================================================

group_cols <- c(
  "Implicit Unaware Base Rate"    = "#F2A6A6",  # light red
  "Implicit Aware Base Rate"      = "#9AD68A",  # light green
  "Explicit Undirected Base Rate" = "#E3AF2D",  # orange
  "Explicit True Base Rate"       = "#1F7A3A",  # dark green
  "Explicit Deceptive Base Rate"  = "#8B0000"   # dark red
)

group_axis_labels <- c(
  "Implicit Unaware Base Rate"    = "Implicit\nUnaware\nBase Rate",
  "Implicit Aware Base Rate"      = "Implicit\nAware\nBase Rate",
  "Explicit Undirected Base Rate" = "Explicit\nUndirected\nBase Rate",
  "Explicit True Base Rate"       = "Explicit\nTrue\nBase Rate",
  "Explicit Deceptive Base Rate"  = "Explicit\nDeceptive\nBase Rate"
)

# =========================================================
# 7. Reference lines and labels
# =========================================================
# Keep dashed reference lines in both panels,
# but only show the "Optimal Bayesian observer" label once.

ref_lines <- tibble(
  parameter = c("Delta", "Eta"),
  ref = c(1, 0)
)

# =========================================================
# 8. Plot
# =========================================================

p_delta_eta <- ggplot(
  plot_data,
  aes(x = group, y = value)
) +
  geom_boxplot(
    aes(fill = group),
    width = 0.60,
    colour = "black",
    linewidth = 0.40,
    outlier.shape = NA
  ) +
  geom_hline(
    data = ref_lines,
    aes(yintercept = ref),
    inherit.aes = FALSE,
    colour = "red",
    linetype = "dashed",
    linewidth = 0.35
  ) +
  facet_wrap(
    ~parameter,
    scales = "free_y",
    nrow = 1
  ) +
  scale_fill_manual(
    values = group_cols,
    guide = guide_legend(
      ncol = 1,
      override.aes = list(
        colour = "black",
        linewidth = 0.35
      )
    )
  ) +
  scale_x_discrete(labels = group_axis_labels) +
  labs(
    x = NULL,
    y = "Posterior value",
    fill = NULL
  ) +
  guides(
    fill = guide_legend(ncol = 1),
    colour = "none",
    linetype = "none",
    shape = "none",
    alpha = "none"
  ) +
  theme_bw(base_size = 16) +
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
    
    legend.position = "right",
    legend.box = "vertical",
    legend.text = element_text(size = 10),
    legend.key.size = unit(0.55, "cm"),
    legend.background = element_blank(),
    
    plot.margin = margin(10, 12, 10, 10)
  )

print(p_delta_eta)

# =========================================================
# 9. Save figure
# =========================================================

ggsave(
  filename = file.path(fig_dir, "figure5_delta_eta_boxplots_5groups_base_rate_colours.png"),
  plot = p_delta_eta,
  width = 12,
  height = 5.5,
  dpi = 300
)

ggsave(
  filename = file.path(fig_dir, "figure5_delta_eta_boxplots_5groups_base_rate_colours.pdf"),
  plot = p_delta_eta,
  width = 12,
  height = 5.5
)

# =========================================================
# 10. Optional summaries
# =========================================================

summ_delta_eta <- plot_data %>%
  group_by(group, parameter) %>%
  summarise(
    mean   = mean(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    l95    = quantile(value, 0.025, na.rm = TRUE),
    u95    = quantile(value, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nPosterior summaries:\n")
print(summ_delta_eta, n = Inf)

cat("\nDone. Figure saved to:\n")
cat(file.path(fig_dir, "figure5_delta_eta_boxplots_5groups_base_rate_colours.png"), "\n")
cat(file.path(fig_dir, "figure5_delta_eta_boxplots_5groups_base_rate_colours.pdf"), "\n")



library(tidyverse)

data_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias/data"

files <- c(
  "DATA_Aware_Exp11.csv",
  "DATA_Unaware_Exp11.csv",
  "DATA_Aware_Exp12.csv",
  "data_priorbelief_truthful_exp13.csv",
  "data_priorbelief_deceptive_exp13.csv"
)

check_button_order <- function(file) {
  
  path <- file.path(data_dir, file)
  data <- read.csv(path)
  
  cat("\n====================================\n")
  cat("FILE:", file, "\n")
  cat("====================================\n")
  
  if (!"ResponseButtonOrder" %in% names(data) &&
      "Manipulation_ResponseButtonOrder" %in% names(data)) {
    data <- data %>%
      rename(ResponseButtonOrder = Manipulation_ResponseButtonOrder)
  }
  
  if (!"ResponseButtonOrder" %in% names(data)) {
    cat("No ResponseButtonOrder column found.\n")
    return(NULL)
  }
  
  if (!"randomiser_475f" %in% names(data)) {
    cat("No randomiser_475f column found.\n")
    return(NULL)
  }
  
  print(table(data$randomiser_475f, data$ResponseButtonOrder, useNA = "ifany"))
  
  cat("\nExpected:\n")
  cat("Button0Blue -> ResponseButtonOrder = 0\n")
  cat("Button0Red  -> ResponseButtonOrder = 1\n")
}

for (f in files) {
  check_button_order(f)
}


######################################
## Truthful Exp13 boost model
## Delta and eta posterior distributions
######################################

rm(list = ls(all = TRUE))

library(tidyverse)
library(posterior)
library(ggplot2)

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

base_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias"
fits_dir <- file.path(base_dir, "results/fits/Exp12")
fig_dir  <- file.path(base_dir, "results/figures")

if (!dir.exists(fig_dir)) {
  dir.create(fig_dir, recursive = TRUE)
}

# -------------------------------------------------------------------------
# Load truthful boost model
# -------------------------------------------------------------------------

load(file.path(fits_dir, "fit_trunc_boost_truthful_exp13.rdata"))

# -------------------------------------------------------------------------
# Extract posterior draws for delta and eta
# -------------------------------------------------------------------------

draws <- as_draws_df(fit$draws())

# If transformed parameters exist, use them.
# Otherwise extract from mu_pr:
#   mu_pr[4] = delta, transformed with pnorm(.) * 2
#   mu_pr[5] = eta, untransformed

if (all(c("mu_delta", "mu_eta") %in% colnames(draws))) {
  
  plot_data <- draws %>%
    select(mu_delta, mu_eta)
  
} else {
  
  mu_cols <- colnames(draws)[grepl("^mu_pr(\\[|\\.)", colnames(draws))]
  
  if (length(mu_cols) < 5) {
    stop("Could not find mu_delta/mu_eta or enough mu_pr columns.")
  }
  
  plot_data <- draws %>%
    select(all_of(mu_cols[c(4, 5)]))
  
  colnames(plot_data) <- c("mu_delta", "mu_eta")
  
  plot_data <- plot_data %>%
    mutate(
      mu_delta = pnorm(mu_delta) * 2
      # eta stays untransformed
    )
}

plot_data <- plot_data %>%
  pivot_longer(
    cols = c(mu_delta, mu_eta),
    names_to = "parameter",
    values_to = "value"
  ) %>%
  mutate(
    parameter = recode(
      parameter,
      "mu_delta" = "delta",
      "mu_eta"   = "eta"
    )
  )

# -------------------------------------------------------------------------
# Plot distributions
# -------------------------------------------------------------------------

ref_lines <- tibble(
  parameter = c("delta", "eta"),
  ref = c(1, 0)
)

p <- ggplot(plot_data, aes(x = value)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 60,
    fill = "#1B7837",
    colour = "black",
    alpha = 0.55,
    linewidth = 0.2
  ) +
  geom_vline(
    data = ref_lines,
    aes(xintercept = ref),
    linetype = "dashed",
    colour = "red",
    linewidth = 0.7
  ) +
  facet_wrap(~parameter, scales = "free", nrow = 1) +
  labs(
    title = "Truthful Exp13 boost model",
    subtitle = "Posterior distributions of delta and eta",
    x = "Posterior value",
    y = "Density"
  ) +
  theme_bw(base_size = 14) +
  theme(
    strip.text = element_text(face = "bold", size = 14),
    plot.title = element_text(face = "bold"),
    axis.title = element_text(face = "bold")
  )

print(p)

ggsave(
  file.path(fig_dir, "truthful_exp13_boost_delta_eta_distributions.png"),
  p,
  width = 8,
  height = 4,
  dpi = 300
)

# -------------------------------------------------------------------------
# Print summaries
# -------------------------------------------------------------------------

summary_table <- plot_data %>%
  group_by(parameter) %>%
  summarise(
    mean = mean(value, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    l95 = quantile(value, 0.025, na.rm = TRUE),
    u95 = quantile(value, 0.975, na.rm = TRUE),
    .groups = "drop"
  )

print(summary_table)






library(tidyverse)
library(posterior)
library(ggplot2)

draws <- as_draws_df(fit$draws())

# Extract transformed parameters if available
if (all(c("mu_alpha", "mu_beta", "mu_delta", "mu_eta") %in% colnames(draws))) {
  
  pars <- draws %>%
    select(mu_alpha, mu_beta, mu_delta, mu_eta)
  
} else {
  
  mu_cols <- colnames(draws)[grepl("^mu_pr(\\[|\\.)", colnames(draws))]
  
  pars <- draws %>%
    select(all_of(mu_cols[1:5]))
  
  colnames(pars) <- c("mu_alpha", "mu_beta", "mu_lambda", "mu_delta", "mu_eta")
  
  pars <- pars %>%
    mutate(
      mu_alpha = pnorm(mu_alpha) * 6,
      mu_delta = pnorm(mu_delta) * 2
    ) %>%
    select(mu_alpha, mu_beta, mu_delta, mu_eta)
}

# Correlation matrix
print(cor(pars, use = "complete.obs"))

# Pairwise plots against eta
pars_long <- pars %>%
  pivot_longer(
    cols = c(mu_alpha, mu_beta, mu_delta),
    names_to = "parameter",
    values_to = "value"
  )

ggplot(pars_long, aes(x = value, y = mu_eta)) +
  geom_point(alpha = 0.15, size = 0.5) +
  geom_smooth(method = "lm", se = TRUE) +
  facet_wrap(~parameter, scales = "free_x") +
  theme_bw() +
  labs(
    x = "Other parameter value",
    y = "mu_eta",
    title = "Posterior trade-offs with eta"
  )






library(tidyverse)
library(posterior)
library(ggplot2)

##################################################
## EXTRACT GROUP-LEVEL POSTERIOR PARAMETERS
##################################################

extract_pars <- function(fit, group_name) {
  
  draws <- as_draws_df(fit$draws())
  
  # -----------------------------------------------
  # Case 1: transformed parameters already available
  # -----------------------------------------------
  
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
        mu_lambda,
        mu_delta,
        mu_eta
      )
    
  } else {
    
    # -----------------------------------------------
    # Case 2: transform raw mu_pr parameters
    # -----------------------------------------------
    
    mu_cols_bracket <- paste0("mu_pr[", 1:5, "]")
    mu_cols_dot     <- paste0("mu_pr.", 1:5, ".")
    
    if (all(mu_cols_bracket %in% colnames(draws))) {
      
      mu_cols <- mu_cols_bracket
      
    } else if (all(mu_cols_dot %in% colnames(draws))) {
      
      mu_cols <- mu_cols_dot
      
    } else {
      
      mu_cols <- colnames(draws)[
        grepl("^mu_pr(\\[|\\.)", colnames(draws))
      ]
      
      mu_cols <- mu_cols[1:5]
    }
    
    if (length(mu_cols) < 5) {
      stop("Could not find all five mu_pr parameters for: ", group_name)
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
      )
  }
  
  pars %>%
    mutate(group = group_name)
}

##################################################
## EXTRACT ALL FIVE GROUPS
##################################################

pars_all <- bind_rows(
  
  extract_pars(
    fit_unaware,
    "Implicit Unaware"
  ),
  
  extract_pars(
    fit_aware,
    "Implicit Aware"
  ),
  
  extract_pars(
    fit_explicit,
    "Explicit Undirected"
  ),
  
  extract_pars(
    fit_truthful,
    "Explicit Truthful"
  )
  
 # extract_pars(
    fit_deceptive,
    "Explicit Deceptive"
  )
  
)

##################################################
## SET GROUP ORDER
##################################################

pars_all <- pars_all %>%
  mutate(
    group = factor(
      group,
      levels = c(
        "Implicit Unaware",
        "Implicit Aware",
        "Explicit Undirected",
        "Explicit Truthful",
        "Explicit Deceptive"
      )
    )
  )

##################################################
## CORRELATION MATRICES FOR EACH GROUP
##################################################

cor_matrices <- pars_all %>%
  split(.$group) %>%
  map(
    ~ cor(
      .x %>%
        select(
          mu_alpha,
          mu_beta,
          mu_lambda,
          mu_delta,
          mu_eta
        ),
      use = "complete.obs"
    )
  )

cat("\n==================================================\n")
cat("POSTERIOR CORRELATION MATRICES\n")
cat("==================================================\n")

walk2(
  cor_matrices,
  names(cor_matrices),
  function(mat, group_name) {
    
    cat("\n----------------------------------------\n")
    cat(group_name, "\n")
    cat("----------------------------------------\n")
    
    print(round(mat, 3))
  }
)

##################################################
## LONG FORMAT:
## ALPHA, BETA, LAMBDA, DELTA AGAINST ETA
##################################################

pars_long <- pars_all %>%
  pivot_longer(
    cols = c(
      mu_alpha,
      mu_beta,
      mu_lambda,
      mu_delta
    ),
    names_to = "parameter",
    values_to = "value"
  ) %>%
  mutate(
    parameter = factor(
      parameter,
      levels = c(
        "mu_alpha",
        "mu_beta",
        "mu_lambda",
        "mu_delta"
      ),
      labels = c(
        "alpha",
        "beta",
        "lambda",
        "delta"
      )
    )
  )

##################################################
## PLOT — ALL GROUPS
##################################################

p_tradeoffs <- ggplot(
  pars_long,
  aes(
    x = value,
    y = mu_eta,
    colour = group
  )
) +
  
  geom_point(
    alpha = 0.10,
    size = 0.5
  ) +
  
  geom_smooth(
    method = "lm",
    se = FALSE,
    linewidth = 1
  ) +
  
  facet_wrap(
    ~ parameter,
    scales = "free_x",
    nrow = 1,
    labeller = label_parsed
  ) +
  
  theme_bw(base_size = 13) +
  
  theme(
    legend.position = "bottom",
    strip.text = element_text(
      face = "bold",
      size = 13
    ),
    axis.title = element_text(
      face = "bold"
    ),
    legend.title = element_blank()
  ) +
  
  labs(
    x = "Parameter value",
    y = expression(mu[eta]),
    title = "Posterior parameter trade-offs with eta"
  )

print(p_tradeoffs)





library(tidyverse)

load("/Users/bty615/Documents/GitHub/reliable_info_bias/data/data_priorbelief_truthful_exp13.rdata")

data_check <- data %>%
  mutate(
    choice = case_when(
      ResponseButtonOrder == 0 & Response == 0 ~ 1, # blue
      ResponseButtonOrder == 0 & Response == 1 ~ 2, # red
      ResponseButtonOrder == 1 & Response == 0 ~ 2, # red
      ResponseButtonOrder == 1 & Response == 1 ~ 1, # blue
      TRUE ~ NA_real_
    )
  ) %>%
  mutate(
    across(
      starts_with("color"),
      ~ case_when(
        . == "blue" ~ 1,
        . == "red"  ~ 2,
        TRUE ~ NA_real_
      )
    )
  )

# Evidence aligned to true prior:
# positive = evidence favours true prior
# negative = evidence favours opposite colour

data_check <- data_check %>%
  rowwise() %>%
  mutate(
    signed_evidence_prior = sum(c_across(starts_with("proba_")) / 100 *
                                  ifelse(c_across(starts_with("color_")) == Prior_Belief, 1, -1),
                                na.rm = TRUE),
    choose_prior = as.numeric(choice == Prior_Belief)
  ) %>%
  ungroup()

# Simpler reliability-weighted version using log-odds:
data_check <- data_check %>%
  rowwise() %>%
  mutate(
    logodds_evidence_prior = sum(
      qlogis(c_across(starts_with("proba_")) / 100) *
        ifelse(c_across(starts_with("color_")) == Prior_Belief, 1, -1),
      na.rm = TRUE
    )
  ) %>%
  ungroup()

# Plot
ggplot(data_check, aes(x = logodds_evidence_prior, y = choose_prior)) +
  stat_summary_bin(fun = mean, bins = 15, geom = "point") +
  stat_summary_bin(fun.data = mean_se, bins = 15, geom = "errorbar", width = 0.1) +
  geom_smooth(method = "glm", method.args = list(family = binomial), se = TRUE) +
  theme_bw() +
  labs(
    x = "Evidence favouring true prior",
    y = "P(choose true-prior colour)",
    title = "Truthful Exp13: choices as a function of prior-aligned evidence"
  )

# Logistic check
m <- glm(
  choose_prior ~ logodds_evidence_prior,
  data = data_check,
  family = binomial()
)

summary(m)




















library(tidyverse)
library(posterior)

# ------------------------------------------------------------
# helper
# ------------------------------------------------------------
extract_delta_eta_wide <- function(fit) {
  d <- as_draws_df(fit$draws())
  
  if (all(c("mu_delta", "mu_eta") %in% colnames(d))) {
    out <- d %>%
      select(mu_delta, mu_eta)
  } else {
    mu_cols <- colnames(d)[grepl("^mu_pr(\\.|\\[)", colnames(d))]
    if (length(mu_cols) < 5) {
      stop("Could not find enough mu_pr columns to extract mu_delta and mu_eta.")
    }
    
    out <- d %>%
      select(all_of(mu_cols[c(4, 5)]))
    
    colnames(out) <- c("mu_delta", "mu_eta")
    
    out <- out %>%
      mutate(mu_delta = pnorm(mu_delta) * 2.0)
  }
  
  as_tibble(out)
}

# ------------------------------------------------------------
# load fits
# ------------------------------------------------------------
load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_boost_model_aware_exp12.rdata")
fit_explicit <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_boost_model_aware_exp11.rdata")
fit_aware <- fit

load("/Users/bty615/Documents/GitHub/reliable_info_bias/results/fits/Exp12/fit_trunc_boost_model_unaware_exp11.rdata")
fit_unaware <- fit

# ------------------------------------------------------------
# extract draws separately
# ------------------------------------------------------------
d_explicit <- extract_delta_eta_wide(fit_explicit)
d_aware    <- extract_delta_eta_wide(fit_aware)
d_unaware  <- extract_delta_eta_wide(fit_unaware)

# ------------------------------------------------------------
# summaries by group
# ------------------------------------------------------------
summ_delta_eta <- bind_rows(
  d_explicit %>% mutate(group = "Explicit Aware"),
  d_aware    %>% mutate(group = "Implicit Aware"),
  d_unaware  %>% mutate(group = "Implicit Unaware")
) %>%
  pivot_longer(cols = c(mu_delta, mu_eta), names_to = "param", values_to = "value") %>%
  group_by(group, param) %>%
  summarise(
    mean   = mean(value),
    median = median(value),
    l95    = quantile(value, 0.025),
    u95    = quantile(value, 0.975),
    .groups = "drop"
  )

print(summ_delta_eta)

# ------------------------------------------------------------
# delta contrasts
# ------------------------------------------------------------
delta_contrasts <- tibble(
  contrast = c(
    "Implicit Aware - Implicit Unaware",
    "Explicit Aware - Implicit Unaware",
    "Explicit Aware - Implicit Aware"
  ),
  mean_diff = c(
    mean(d_aware$mu_delta - d_unaware$mu_delta),
    mean(d_explicit$mu_delta - d_unaware$mu_delta),
    mean(d_explicit$mu_delta - d_aware$mu_delta)
  ),
  l95 = c(
    quantile(d_aware$mu_delta - d_unaware$mu_delta, 0.025),
    quantile(d_explicit$mu_delta - d_unaware$mu_delta, 0.025),
    quantile(d_explicit$mu_delta - d_aware$mu_delta, 0.025)
  ),
  u95 = c(
    quantile(d_aware$mu_delta - d_unaware$mu_delta, 0.975),
    quantile(d_explicit$mu_delta - d_unaware$mu_delta, 0.975),
    quantile(d_explicit$mu_delta - d_aware$mu_delta, 0.975)
  ),
  pd_gt_0 = c(
    mean((d_aware$mu_delta - d_unaware$mu_delta) > 0),
    mean((d_explicit$mu_delta - d_unaware$mu_delta) > 0),
    mean((d_explicit$mu_delta - d_aware$mu_delta) > 0)
  )
)

print(delta_contrasts)

# ------------------------------------------------------------
# eta relative to zero
# ------------------------------------------------------------
eta_stats <- tibble(
  group = c("Implicit Unaware", "Implicit Aware", "Explicit Aware"),
  mean_eta = c(mean(d_unaware$mu_eta), mean(d_aware$mu_eta), mean(d_explicit$mu_eta)),
  l95 = c(
    quantile(d_unaware$mu_eta, 0.025),
    quantile(d_aware$mu_eta, 0.025),
    quantile(d_explicit$mu_eta, 0.025)
  ),
  u95 = c(
    quantile(d_unaware$mu_eta, 0.975),
    quantile(d_aware$mu_eta, 0.975),
    quantile(d_explicit$mu_eta, 0.975)
  ),
  p_gt_0 = c(
    mean(d_unaware$mu_eta > 0),
    mean(d_aware$mu_eta > 0),
    mean(d_explicit$mu_eta > 0)
  )
)

print(eta_stats)
print(delta_contrasts)







# ===============================================================
# LOO COMPARISON
# 5 MODELS x 5 GROUPS
#
# Models:
#   1. Simple
#   2. Learning
#   3. Local eta
#   4. Learning + Local eta
#   5. Learning + Global eta
#
# Groups:
#   1. Implicit Unaware
#   2. Implicit Aware
#   3. Explicit Undirected
#   4. Explicit Truthful
#   5. Explicit Deceptive
# ===============================================================

rm(list = ls(all = TRUE))

library(loo)
library(tidyverse)
library(ggplot2)


##################################################
## SET PATH
##################################################

loo_dir <- "/Users/bty615/Documents/GitHub/reliable_info_bias/stan/results/loo/exp11_unaware"


cat("\n====================================================\n")
cat("LOO DIRECTORY\n")
cat("====================================================\n\n")

cat(loo_dir, "\n\n")


if (!dir.exists(loo_dir)) {
  
  stop(
    paste0(
      "Cannot find LOO directory:\n",
      loo_dir
    )
  )
}


cat("Files found in LOO directory:\n\n")

loo_files_found <- list.files(
  loo_dir,
  pattern = "\\.rdata$",
  ignore.case = TRUE
)

print(loo_files_found)


##################################################
## SAFE DATA-FRAME PRINT FUNCTION
##################################################

# IMPORTANT:
# Do not use print(x, n = Inf) because if x is a normal
# data.frame, R can interpret n as na.print and produce:
#
# invalid 'na.print' specification

print_df <- function(x) {
  
  x <- as.data.frame(x)
  
  print.data.frame(
    x,
    row.names = FALSE
  )
}


##################################################
## LOAD LOO OBJECT
##################################################

load_loo <- function(path) {
  
  if (!file.exists(path)) {
    
    stop(
      paste0(
        "File not found:\n",
        path
      )
    )
  }
  
  
  e <- new.env()
  
  loaded_objects <- load(
    path,
    envir = e
  )
  
  object_names <- ls(e)
  
  
  # ------------------------------------------------
  # Case 1:
  # object is called loo
  # ------------------------------------------------
  
  if (exists("loo", envir = e)) {
    
    obj <- get(
      "loo",
      envir = e
    )
    
    if (inherits(obj, "loo")) {
      
      return(obj)
    }
  }
  
  
  # ------------------------------------------------
  # Case 2:
  # global eta models save it as loo_result
  # ------------------------------------------------
  
  if (exists("loo_result", envir = e)) {
    
    obj <- get(
      "loo_result",
      envir = e
    )
    
    if (inherits(obj, "loo")) {
      
      return(obj)
    }
  }
  
  
  # ------------------------------------------------
  # Case 3:
  # search for any object with class loo
  # ------------------------------------------------
  
  valid_loo_objects <- object_names[
    
    sapply(
      object_names,
      function(object_name) {
        
        inherits(
          get(
            object_name,
            envir = e
          ),
          "loo"
        )
      }
    )
  ]
  
  
  if (length(valid_loo_objects) == 1) {
    
    return(
      get(
        valid_loo_objects[1],
        envir = e
      )
    )
  }
  
  
  if (length(valid_loo_objects) > 1) {
    
    cat("\nMore than one LOO object found in:\n")
    cat(path, "\n\n")
    
    print(valid_loo_objects)
    
    cat("\nUsing first valid LOO object.\n")
    
    return(
      get(
        valid_loo_objects[1],
        envir = e
      )
    )
  }
  
  
  cat("\nNo valid LOO object found.\n")
  
  cat("\nObjects in file:\n")
  print(object_names)
  
  cat("\nObjects returned by load():\n")
  print(loaded_objects)
  
  
  stop(
    paste0(
      "No valid LOO object found in:\n",
      path
    )
  )
}


##################################################
## GET NUMBER OF OBSERVATIONS
##################################################

get_loo_n <- function(x) {
  
  if (!is.null(x$pointwise)) {
    
    return(
      nrow(
        as.data.frame(
          x$pointwise
        )
      )
    )
  }
  
  return(NA_integer_)
}


##################################################
## SAFE PARETO-K PRINT
##################################################

safe_pareto_print <- function(x, model_name) {
  
  cat("\n----------------------------------------\n")
  cat(model_name, "\n")
  cat("----------------------------------------\n")
  
  out <- tryCatch(
    
    pareto_k_table(x),
    
    error = function(e) {
      
      cat(
        "Pareto-k table could not be obtained.\n"
      )
      
      return(NULL)
    }
  )
  
  
  if (!is.null(out)) {
    
    print(out)
  }
}


##################################################
## RUN LOO COMPARISON FOR ONE GROUP
##################################################

run_loo_comparison <- function(
    group_name,
    paths
) {
  
  
  cat("\n\n")
  cat("====================================================\n")
  cat(group_name, "\n")
  cat("====================================================\n")
  
  
  ################################################
  ## CHECK FILES
  ################################################
  
  cat("\nFiles being used:\n\n")
  
  for (i in seq_along(paths)) {
    
    cat(
      names(paths)[i],
      ":\n",
      paths[i],
      "\n\n"
    )
  }
  
  
  file_exists <- file.exists(paths)
  
  
  file_check <- data.frame(
    model = names(paths),
    file = basename(paths),
    exists = file_exists,
    stringsAsFactors = FALSE
  )
  
  
  cat("\nFile check:\n\n")
  
  print_df(file_check)
  
  
  if (any(!file_exists)) {
    
    cat("\nMISSING FILES:\n\n")
    
    print_df(
      file_check[
        !file_check$exists,
        ,
        drop = FALSE
      ]
    )
    
    stop(
      "Some LOO files are missing."
    )
  }
  
  
  ################################################
  ## LOAD LOO OBJECTS
  ################################################
  
  loos <- lapply(
    paths,
    load_loo
  )
  
  names(loos) <- names(paths)
  
  
  ################################################
  ## CHECK CLASSES
  ################################################
  
  loo_classes <- data.frame(
    
    model = names(loos),
    
    class = sapply(
      loos,
      function(x) {
        paste(
          class(x),
          collapse = ", "
        )
      }
    ),
    
    stringsAsFactors = FALSE
  )
  
  
  cat("\nLOO object classes:\n\n")
  
  print_df(
    loo_classes
  )
  
  
  ################################################
  ## CHECK NUMBER OF OBSERVATIONS
  ################################################
  
  n_obs <- sapply(
    loos,
    get_loo_n
  )
  
  
  observation_check <- data.frame(
    model = names(n_obs),
    n_observations = as.integer(n_obs),
    stringsAsFactors = FALSE
  )
  
  
  cat("\nNumber of pointwise observations:\n\n")
  
  print_df(
    observation_check
  )
  
  
  if (any(is.na(n_obs))) {
    
    stop(
      paste0(
        "At least one model does not contain ",
        "pointwise LOO values."
      )
    )
  }
  
  
  if (length(unique(n_obs)) != 1) {
    
    cat("\n")
    cat("IMPORTANT\n")
    cat("---------\n")
    
    cat(
      paste0(
        "The models do not contain the same number ",
        "of observations.\n\n"
      )
    )
    
    print_df(
      observation_check
    )
    
    
    stop(
      paste0(
        "LOO comparison stopped because ",
        "observation counts differ."
      )
    )
  }
  
  
  ################################################
  ## PARETO-K
  ################################################
  
  cat("\nPareto-k diagnostics:\n")
  
  for (model_name in names(loos)) {
    
    safe_pareto_print(
      loos[[model_name]],
      model_name
    )
  }
  
  
  ################################################
  ## LOO COMPARE
  ################################################
  
  comp <- loo_compare(
    loos
  )
  
  
  # Convert immediately to an ordinary data frame.
  # This avoids version-specific print methods.
  
  comp_df <- as.data.frame(
    comp
  )
  
  comp_df$model <- rownames(
    comp_df
  )
  
  rownames(
    comp_df
  ) <- NULL
  
  
  # Put model first
  
  comp_df <- comp_df %>%
    select(
      model,
      everything()
    )
  
  
  cat("\n")
  cat("====================================================\n")
  cat("LOO COMPARISON\n")
  cat("====================================================\n\n")
  
  cat(
    "Models are ordered from best to worst.\n"
  )
  
  cat(
    "Higher ELPD-LOO indicates better out-of-sample prediction.\n"
  )
  
  cat(
    "The best model has elpd_diff = 0.\n"
  )
  
  cat(
    "Worse models have negative elpd_diff values.\n\n"
  )
  
  
  print_df(
    comp_df
  )
  
  
  return(
    list(
      loos = loos,
      comp = comp,
      comp_df = comp_df,
      n_obs = n_obs
    )
  )
}


##################################################
## DEFINE MODEL FILES
##################################################


# ===============================================================
# EXP11 UNAWARE
# ===============================================================

paths_exp11_unaware <- c(
  
  simple_model =
    file.path(
      loo_dir,
      "loo_trunc_simple_model_unaware_exp11.rdata"
    ),
  
  learning_model =
    file.path(
      loo_dir,
      "loo_trunc_learning_model_unaware_exp11.rdata"
    ),
  
  eta_model =
    file.path(
      loo_dir,
      "loo_trunc_eta_model_unaware_exp11.rdata"
    ),
  
  learning_boost =
    file.path(
      loo_dir,
      "loo_trunc_boost_model_unaware_exp11.rdata"
    ),
  
  global_eta =
    file.path(
      loo_dir,
      "loo_trunc_global_eta_unaware_exp11.rdata"
    )
)


# ===============================================================
# EXP11 AWARE
# ===============================================================

paths_exp11_aware <- c(
  
  simple_model =
    file.path(
      loo_dir,
      "loo_trunc_simple_model_aware_exp11.rdata"
    ),
  
  learning_model =
    file.path(
      loo_dir,
      "loo_trunc_learning_model_aware_exp11.rdata"
    ),
  
  eta_model =
    file.path(
      loo_dir,
      "loo_trunc_eta_model_aware_exp11.rdata"
    ),
  
  learning_boost =
    file.path(
      loo_dir,
      "loo_trunc_boost_model_aware_exp11.rdata"
    ),
  
  global_eta =
    file.path(
      loo_dir,
      "loo_trunc_global_eta_aware_exp11.rdata"
    )
)


# ===============================================================
# EXP12 EXPLICIT UNDIRECTED
# ===============================================================

paths_exp12_aware <- c(
  
  simple_model =
    file.path(
      loo_dir,
      "loo_trunc_simple_model_aware_exp12.rdata"
    ),
  
  learning_model =
    file.path(
      loo_dir,
      "loo_trunc_learning_model_aware_exp12.rdata"
    ),
  
  eta_model =
    file.path(
      loo_dir,
      "loo_trunc_eta_model_aware_exp12.rdata"
    ),
  
  learning_boost =
    file.path(
      loo_dir,
      "loo_trunc_boost_model_aware_exp12.rdata"
    ),
  
  global_eta =
    file.path(
      loo_dir,
      "loo_trunc_global_eta_aware_exp12.rdata"
    )
)


# ===============================================================
# EXP13 TRUTHFUL
# ===============================================================

paths_exp13_truthful <- c(
  
  simple_model =
    file.path(
      loo_dir,
      "loo_trunc_simple_truthful_exp13.rdata"
    ),
  
  learning_model =
    file.path(
      loo_dir,
      "loo_trunc_learning_truthful_exp13.rdata"
    ),
  
  eta_model =
    file.path(
      loo_dir,
      "loo_trunc_eta_truthful_exp13.rdata"
    ),
  
  learning_boost =
    file.path(
      loo_dir,
      "loo_trunc_boost_truthful_exp13.rdata"
    ),
  
  global_eta =
    file.path(
      loo_dir,
      "loo_trunc_global_eta_truthful_exp13.rdata"
    )
)


# ===============================================================
# EXP13 DECEPTIVE
# ===============================================================

paths_exp13_deceptive <- c(
  
  simple_model =
    file.path(
      loo_dir,
      "loo_trunc_simple_deceptive_exp13.rdata"
    ),
  
  learning_model =
    file.path(
      loo_dir,
      "loo_trunc_learning_deceptive_exp13.rdata"
    ),
  
  eta_model =
    file.path(
      loo_dir,
      "loo_trunc_eta_deceptive_exp13.rdata"
    ),
  
  learning_boost =
    file.path(
      loo_dir,
      "loo_trunc_boost_deceptive_exp13.rdata"
    ),
  
  global_eta =
    file.path(
      loo_dir,
      "loo_trunc_global_eta_deceptive_exp13.rdata"
    )
)


##################################################
## CHECK ALL 25 FILES
##################################################

all_paths <- c(
  paths_exp11_unaware,
  paths_exp11_aware,
  paths_exp12_aware,
  paths_exp13_truthful,
  paths_exp13_deceptive
)


all_file_check <- data.frame(
  file = basename(all_paths),
  exists = file.exists(all_paths),
  stringsAsFactors = FALSE
)


cat("\n\n")
cat("====================================================\n")
cat("CHECKING ALL 25 LOO FILES\n")
cat("====================================================\n\n")


print_df(
  all_file_check
)


if (any(!all_file_check$exists)) {
  
  cat("\n")
  cat("====================================================\n")
  cat("MISSING FILES\n")
  cat("====================================================\n\n")
  
  
  print_df(
    all_file_check[
      !all_file_check$exists,
      ,
      drop = FALSE
    ]
  )
  
  
  stop(
    "At least one required LOO file is missing."
  )
}


cat("\nAll 25 LOO files found.\n")


##################################################
## RUN LOO COMPARISONS
##################################################


# ===============================================================
# EXP11 UNAWARE
# ===============================================================

res_exp11_unaware <- run_loo_comparison(
  
  group_name = "Exp11 Unaware",
  
  paths = paths_exp11_unaware
)


# ===============================================================
# EXP11 AWARE
# ===============================================================

res_exp11_aware <- run_loo_comparison(
  
  group_name = "Exp11 Aware",
  
  paths = paths_exp11_aware
)


# ===============================================================
# EXP12 EXPLICIT UNDIRECTED
# ===============================================================

res_exp12_aware <- run_loo_comparison(
  
  group_name = "Exp12 Aware",
  
  paths = paths_exp12_aware
)


# ===============================================================
# EXP13 TRUTHFUL
# ===============================================================

res_exp13_truthful <- run_loo_comparison(
  
  group_name = "Exp13 Truthful",
  
  paths = paths_exp13_truthful
)


# ===============================================================
# EXP13 DECEPTIVE
# ===============================================================

res_exp13_deceptive <- run_loo_comparison(
  
  group_name = "Exp13 Deceptive",
  
  paths = paths_exp13_deceptive
)


##################################################
## CREATE COMBINED SUMMARY
##################################################

make_summary_table <- function(
    result,
    group_name
) {
  
  d <- result$comp_df
  
  d$group <- group_name
  
  
  d <- d %>%
    select(
      group,
      model,
      elpd_diff,
      se_diff,
      elpd_loo,
      se_elpd_loo,
      p_loo,
      se_p_loo,
      looic,
      se_looic
    )
  
  
  return(d)
}


loo_summary <- bind_rows(
  
  make_summary_table(
    res_exp11_unaware,
    "Exp11 Unaware"
  ),
  
  make_summary_table(
    res_exp11_aware,
    "Exp11 Aware"
  ),
  
  make_summary_table(
    res_exp12_aware,
    "Exp12 Aware"
  ),
  
  make_summary_table(
    res_exp13_truthful,
    "Exp13 Truthful"
  ),
  
  make_summary_table(
    res_exp13_deceptive,
    "Exp13 Deceptive"
  )
)


cat("\n\n")
cat("====================================================\n")
cat("COMBINED LOO SUMMARY\n")
cat("====================================================\n\n")


print_df(
  loo_summary
)


##################################################
## SAVE COMBINED TABLE
##################################################

comparison_file <- file.path(
  loo_dir,
  "loo_comparison_5models_5groups.csv"
)


write.csv(
  loo_summary,
  file = comparison_file,
  row.names = FALSE
)


cat("\nSaved comparison table:\n")
cat(comparison_file, "\n")


##################################################
## PREPARE PLOTTING TABLE
##################################################

loo_plot <- loo_summary %>%
  
  mutate(
    
    group_label = case_when(
      
      group == "Exp11 Unaware" ~
        "Implicit Unaware Base Rate",
      
      group == "Exp11 Aware" ~
        "Implicit Aware Base Rate",
      
      group == "Exp12 Aware" ~
        "Explicit Undirected Base Rate",
      
      group == "Exp13 Truthful" ~
        "Explicit True Base Rate",
      
      group == "Exp13 Deceptive" ~
        "Explicit Deceptive Base Rate",
      
      TRUE ~ group
    ),
    
    
    model_label = case_when(
      
      model == "simple_model" ~
        "Simple",
      
      model == "learning_model" ~
        "Learning",
      
      model == "eta_model" ~
        "Local eta",
      
      model == "learning_boost" ~
        "Learning + Local eta",
      
      model == "global_eta" ~
        "Learning + Global eta",
      
      TRUE ~ model
    )
  ) %>%
  
  mutate(
    
    group_label = factor(
      
      group_label,
      
      levels = c(
        "Implicit Unaware Base Rate",
        "Implicit Aware Base Rate",
        "Explicit Undirected Base Rate",
        "Explicit True Base Rate",
        "Explicit Deceptive Base Rate"
      )
    ),
    
    
    model_label = factor(
      
      model_label,
      
      levels = c(
        "Simple",
        "Learning",
        "Local eta",
        "Learning + Local eta",
        "Learning + Global eta"
      )
    )
  )


cat("\n")
cat("====================================================\n")
cat("PLOTTING TABLE\n")
cat("====================================================\n\n")


plot_table_print <- loo_plot %>%
  select(
    group_label,
    model_label,
    elpd_loo,
    se_elpd_loo,
    elpd_diff,
    se_diff
  )


print_df(
  plot_table_print
)


##################################################
## GROUP COLOURS
##################################################

group_cols <- c(
  
  "Implicit Unaware Base Rate" =
    rgb(1.00, 0.65, 0.65),
  
  "Implicit Aware Base Rate" =
    rgb(0.56, 0.93, 0.56),
  
  "Explicit Undirected Base Rate" =
    rgb(1.00, 0.65, 0.00),
  
  "Explicit True Base Rate" =
    rgb(0.00, 0.50, 0.00),
  
  "Explicit Deceptive Base Rate" =
    rgb(0.70, 0.00, 0.00)
)


##################################################
## PLOT 1:
## DELTA ELPD RELATIVE TO BEST MODEL
##################################################

p_elpd_diff <- ggplot(
  
  loo_plot,
  
  aes(
    x = model_label,
    y = elpd_diff,
    fill = group_label
  )
  
) +
  
  geom_col(
    width = 0.75,
    colour = "black",
    linewidth = 0.25
  ) +
  
  geom_errorbar(
    
    aes(
      ymin = elpd_diff - se_diff,
      ymax = elpd_diff + se_diff
    ),
    
    width = 0.18,
    linewidth = 0.7
  ) +
  
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.7
  ) +
  
  facet_wrap(
    ~group_label,
    nrow = 1
  ) +
  
  scale_fill_manual(
    values = group_cols
  ) +
  
  labs(
    
    x = "Model",
    
    y = expression(
      Delta * ELPD ~ "(vs best model)"
    ),
    
    title = "LOO Model Comparison"
  ) +
  
  theme_classic(
    base_size = 14
  ) +
  
  theme(
    
    legend.position = "none",
    
    strip.background = element_blank(),
    
    strip.text = element_text(
      size = 11,
      face = "bold"
    ),
    
    axis.text.x = element_text(
      angle = 35,
      hjust = 1
    ),
    
    plot.title = element_text(
      face = "bold",
      hjust = 0.5
    )
  )


print(
  p_elpd_diff
)


##################################################
## SAVE DELTA ELPD PLOT
##################################################

elpd_diff_png <- file.path(
  loo_dir,
  "loo_elpd_diff_5models_5groups.png"
)

elpd_diff_pdf <- file.path(
  loo_dir,
  "loo_elpd_diff_5models_5groups.pdf"
)


ggsave(
  filename = elpd_diff_png,
  plot = p_elpd_diff,
  width = 17,
  height = 5,
  dpi = 300
)


ggsave(
  filename = elpd_diff_pdf,
  plot = p_elpd_diff,
  width = 17,
  height = 5
)


##################################################
## PLOT 2:
## RAW ELPD-LOO
##################################################

p_elpd_raw <- ggplot(
  
  loo_plot,
  
  aes(
    x = model_label,
    y = elpd_loo,
    fill = group_label
  )
  
) +
  
  geom_col(
    width = 0.75,
    colour = "black",
    linewidth = 0.25
  ) +
  
  geom_errorbar(
    
    aes(
      ymin = elpd_loo - se_elpd_loo,
      ymax = elpd_loo + se_elpd_loo
    ),
    
    width = 0.18,
    linewidth = 0.7
  ) +
  
  facet_wrap(
    ~group_label,
    nrow = 1,
    scales = "free_y"
  ) +
  
  scale_fill_manual(
    values = group_cols
  ) +
  
  labs(
    
    x = "Model",
    
    y = "ELPD-LOO",
    
    title = "Raw ELPD-LOO By Model"
  ) +
  
  theme_classic(
    base_size = 14
  ) +
  
  theme(
    
    legend.position = "none",
    
    strip.background = element_blank(),
    
    strip.text = element_text(
      size = 11,
      face = "bold"
    ),
    
    axis.text.x = element_text(
      angle = 35,
      hjust = 1
    ),
    
    plot.title = element_text(
      face = "bold",
      hjust = 0.5
    )
  )


print(
  p_elpd_raw
)


##################################################
## SAVE RAW ELPD PLOT
##################################################

elpd_raw_png <- file.path(
  loo_dir,
  "loo_elpd_raw_5models_5groups.png"
)

elpd_raw_pdf <- file.path(
  loo_dir,
  "loo_elpd_raw_5models_5groups.pdf"
)


ggsave(
  filename = elpd_raw_png,
  plot = p_elpd_raw,
  width = 17,
  height = 5,
  dpi = 300
)


ggsave(
  filename = elpd_raw_pdf,
  plot = p_elpd_raw,
  width = 17,
  height = 5
)


##################################################
## BEST MODEL IN EACH GROUP
##################################################

best_models <- loo_summary %>%
  
  group_by(
    group
  ) %>%
  
  slice_max(
    order_by = elpd_loo,
    n = 1,
    with_ties = FALSE
  ) %>%
  
  ungroup() %>%
  
  select(
    group,
    model,
    elpd_loo,
    se_elpd_loo
  )


cat("\n\n")
cat("====================================================\n")
cat("BEST MODEL IN EACH GROUP\n")
cat("====================================================\n\n")


print_df(
  best_models
)


##################################################
## SAVE BEST MODELS
##################################################

best_model_file <- file.path(
  loo_dir,
  "loo_best_models_5groups.csv"
)


write.csv(
  best_models,
  file = best_model_file,
  row.names = FALSE
)


##################################################
## FINISHED
##################################################

cat("\n\n")
cat("====================================================\n")
cat("LOO COMPARISON COMPLETE\n")
cat("====================================================\n\n")

cat("Saved files:\n\n")

cat(
  comparison_file,
  "\n"
)

cat(
  elpd_diff_png,
  "\n"
)

cat(
  elpd_diff_pdf,
  "\n"
)

cat(
  elpd_raw_png,
  "\n"
)

cat(
  elpd_raw_pdf,
  "\n"
)

cat(
  best_model_file,
  "\n"
)








 







############################################################
# SIMPLE Vb TRAJECTORY:
# BLUE PRIOR vs RED PRIOR WITHIN EACH GROUP
############################################################

library(tidyverse)
library(ggplot2)

# TrueDirection:
# 1 = Blue prior
# 2 = Red prior

plot_data <- Vb_all %>%
  mutate(
    PriorColour = case_when(
      TrueDirection == 1 ~ "Blue prior",
      TrueDirection == 2 ~ "Red prior",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(!is.na(PriorColour))

############################################################
# MEAN Vb AT EACH TRIAL
############################################################

plot_summary <- plot_data %>%
  group_by(
    group,
    PriorColour,
    TrialNumber
  ) %>%
  summarise(
    mean_Vb = mean(Vb_blue, na.rm = TRUE),
    .groups = "drop"
  )

############################################################
# PLOT
############################################################

p <- ggplot(
  plot_summary,
  aes(
    x = TrialNumber,
    y = mean_Vb,
    colour = PriorColour
  )
) +
  
  geom_hline(
    yintercept = 0.5,
    linetype = "dashed",
    linewidth = 0.8
  ) +
  
  geom_line(
    linewidth = 1.2
  ) +
  
  facet_wrap(
    ~ group,
    ncol = 2
  ) +
  
  scale_colour_manual(
    values = c(
      "Blue prior" = "blue",
      "Red prior"  = "red"
    )
  ) +
  
  coord_cartesian(
    ylim = c(0.2, 0.8)
  ) +
  
  labs(
    title = "Modelled Vb trajectory by prior colour",
    subtitle = "Vb = modelled probability that Blue is more likely",
    x = "Trial",
    y = expression(V[b]),
    colour = NULL
  ) +
  
  theme_classic(
    base_size = 14
  ) +
  
  theme(
    legend.position = "bottom",
    strip.text = element_text(
      face = "bold",
      size = 12
    )
  )

print(p)

