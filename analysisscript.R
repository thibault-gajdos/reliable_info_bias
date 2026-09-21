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
RUN_PARTICIPANT_TESTS <- TRUE
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
  "Implicit Unaware Base Rate" = "localeta_unaware_exp11.rdata",
  "Implicit Aware Base Rate" = "localeta_aware_exp11.rdata",
  "Explicit Undirected Base Rate" = "localeta_aware_exp12.rdata",
  "Explicit True Base Rate" = "localeta_truthful_exp13.rdata",
  "Explicit Deceptive Base Rate" = "localeta_deceptive_exp13.rdata"
)

parameter_names <- c("mu_alpha", "mu_beta", "mu_lambda", "mu_delta", "mu_eta")
parameter_labels <- c(
  "mu_alpha" = "alpha", "mu_beta" = "beta", "mu_lambda" = "lambda",
  "mu_delta" = "delta", "mu_eta" = "eta"
)
reference_values <- c(
  "mu_alpha" = 1, "mu_beta" = 0, "mu_lambda" = 0,
  "mu_delta" = 1, "mu_eta" = 0
)

# ==============================================================================
# 2. GENERAL HELPERS
# ==============================================================================

save_plot_both <- function(plot, stem, width, height) {
  ggsave(file.path(figure_dir, paste0(stem, ".png")), plot,
         width = width, height = height, dpi = 300)
  ggsave(file.path(figure_dir, paste0(stem, ".pdf")), plot,
         width = width, height = height)
}

load_single_object <- function(path, preferred_name = NULL) {
  if (!file.exists(path)) stop("File not found: ", path)
  e <- new.env(parent = emptyenv())
  loaded <- load(path, envir = e)
  if (!is.null(preferred_name) && preferred_name %in% loaded) {
    return(e[[preferred_name]])
  }
  if (length(loaded) == 1L) return(e[[loaded]])
  stop("File contains several objects and '", preferred_name,
       "' was not found: ", path)
}

load_fit <- function(filename) {
  fit <- load_single_object(file.path(fits_dir, filename), "fit")
  if (!inherits(fit, c("CmdStanMCMC", "CmdStanFit"))) {
    stop("Object is not a CmdStan fit: ", filename)
  }
  fit
}

fits <- set_names(map(unname(fit_files), load_fit), names(fit_files))
cat("Loaded Learning + Local eta fits for all five groups.\n")

# ==============================================================================
# 3. GROUP-LEVEL POSTERIOR DRAWS
# ==============================================================================

extract_group_draws <- function(fit, group_name, alpha_scale = 6, delta_scale = 2) {
  d <- as_draws_df(fit$draws())
  
  if (all(parameter_names %in% names(d))) {
    out <- d %>% select(all_of(parameter_names))
  } else {
    bracket <- paste0("mu_pr[", 1:5, "]")
    dotted  <- paste0("mu_pr.", 1:5, ".")
    if (all(bracket %in% names(d))) {
      raw_names <- bracket
    } else if (all(dotted %in% names(d))) {
      raw_names <- dotted
    } else {
      raw_names <- names(d)[str_detect(names(d), "^mu_pr(\\[|\\.)")]
      raw_names <- raw_names[seq_len(min(5, length(raw_names)))]
    }
    if (length(raw_names) < 5L) {
      stop("Could not find five group-level parameters for ", group_name)
    }
    out <- d %>% select(all_of(raw_names[1:5]))
    names(out) <- parameter_names
    out <- out %>% mutate(
      mu_alpha  = pnorm(mu_alpha) * alpha_scale,
      mu_lambda = pnorm(mu_lambda),
      mu_delta  = pnorm(mu_delta) * delta_scale
    )
  }
  
  out %>%
    mutate(draw_id = row_number(), group = group_name) %>%
    relocate(group, draw_id)
}

draws_wide <- imap_dfr(fits, extract_group_draws) %>%
  mutate(group = factor(group, levels = group_levels))

draws_long <- draws_wide %>%
  pivot_longer(all_of(parameter_names), names_to = "parameter", values_to = "value") %>%
  mutate(
    parameter = factor(parameter, levels = parameter_names),
    parameter_clean = recode(as.character(parameter), !!!parameter_labels)
  )

stopifnot(
  all(is.finite(draws_long$value)),
  n_distinct(draws_long$group) == 5L,
  n_distinct(draws_long$parameter) == 5L
)

# ==============================================================================
# 4. PREPARE RELIABILITY DISTORTION AND SEQUENTIAL WEIGHTS
# ==============================================================================

distort_reliability <- function(p, alpha, beta) {
  p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  plogis(alpha * qlogis(p) + beta)
}

summarise_distortion <- function(df, probabilities) {
  map_dfr(probabilities, function(p) {
    values <- distort_reliability(p, df$mu_alpha, df$mu_beta)
    tibble(objective = p, mean = mean(values),
           l95 = quantile(values, .025), u95 = quantile(values, .975))
  })
}

distortion_grid <- seq(.001, .999, length.out = 501)
distortion_curves <- draws_wide %>%
  group_split(group) %>%
  map_dfr(~ summarise_distortion(.x, distortion_grid) %>%
            mutate(group = as.character(.x$group[1]))) %>%
  mutate(group = factor(group, levels = group_levels))

distortion_points <- draws_wide %>%
  group_split(group) %>%
  map_dfr(~ summarise_distortion(.x, c(.50, .55, .65)) %>%
            mutate(group = as.character(.x$group[1]))) %>%
  mutate(group = factor(group, levels = group_levels))

# ------------------------------------------------------------------------------
# Sequential-position weights from lambda
# ------------------------------------------------------------------------------

positions <- 1:6
position_weights <- draws_wide %>%
  group_split(group) %>%
  map_dfr(function(df) {
    map_dfr(positions, function(position) {
      values <- exp(df$mu_lambda * (position - 6))
      tibble(position = position, mean = mean(values),
             l95 = quantile(values, .025), u95 = quantile(values, .975))
    }) %>% mutate(group = as.character(df$group[1]))
  }) %>%
  mutate(group = factor(group, levels = group_levels))

# ==============================================================================
# 5. ALL FIGURES
# ==============================================================================

# ------------------------------------------------------------------------------
# 5A. Posterior-distribution figure
# ------------------------------------------------------------------------------

p_distributions <- ggplot(draws_long, aes(value, fill = group, colour = group)) +
  geom_histogram(
    aes(y = after_stat(density)),
    bins = 60,
    position = "identity",
    alpha = 0.42,
    linewidth = 0.25
  ) +
  facet_wrap(~parameter_clean, scales = "free", nrow = 1,
             labeller = label_parsed) +
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
    subtitle = "Five groups, Learning + Local eta model",
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

save_plot_both(p_distributions, "posterior_distributions_five_groups", 16, 5)

# ------------------------------------------------------------------------------
# 5B. Reliability-distortion figure
# ------------------------------------------------------------------------------

p_distortion <- ggplot(distortion_curves,
                       aes(objective * 100, mean * 100, colour = group, fill = group)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", colour = "grey45") +
  geom_ribbon(aes(ymin = l95 * 100, ymax = u95 * 100), colour = NA, alpha = .10) +
  geom_line(linewidth = 1) +
  geom_point(data = distortion_points, size = 2) +
  scale_colour_manual(values = group_cols) +
  scale_fill_manual(values = group_cols) +
  coord_equal(xlim = c(45, 70), ylim = c(45, 100)) +
  labs(x = "Objective reliability (%)", y = "Modelled reliability (%)",
       colour = NULL, fill = NULL) +
  theme_bw(base_size = 13) +
  theme(legend.position = "bottom")

save_plot_both(p_distortion, "reliability_distortion_five_groups", 10, 7)
write_csv(distortion_points, file.path(table_dir, "distorted_reliability_50_55_65.csv"))

# ------------------------------------------------------------------------------
# 5C. Sequential-position weighting figure
# ------------------------------------------------------------------------------

p_weights <- ggplot(position_weights,
                    aes(position, mean, colour = group, fill = group)) +
  geom_ribbon(aes(ymin = l95, ymax = u95), colour = NA, alpha = .10) +
  geom_line(linewidth = 1) + geom_point(size = 2) +
  scale_x_continuous(breaks = positions) +
  scale_colour_manual(values = group_cols) + scale_fill_manual(values = group_cols) +
  labs(x = "Sample position", y = "Weight relative to sample 6",
       colour = NULL, fill = NULL) +
  theme_bw(base_size = 13) + theme(legend.position = "bottom")

save_plot_both(p_weights, "sequential_weights_five_groups", 10, 7)
write_csv(position_weights, file.path(table_dir, "sequential_position_weights.csv"))

# ------------------------------------------------------------------------------
# 5D. Delta and eta posterior figure
# ------------------------------------------------------------------------------
# These are group-level posterior distributions, not participant distributions.

delta_eta <- draws_long %>%
  filter(parameter %in% c("mu_delta", "mu_eta")) %>%
  mutate(parameter_clean = factor(parameter_clean, levels = c("delta", "eta")))

reference_lines <- tibble(parameter_clean = factor(c("delta", "eta"),
                                                   levels = c("delta", "eta")),
                          reference = c(1, 0))

p_delta_eta <- ggplot(delta_eta, aes(group, value, fill = group)) +
  geom_boxplot(
    width = 0.60,
    colour = "black",
    linewidth = 0.40,
    outlier.shape = NA
  ) +
  geom_hline(data = reference_lines, aes(yintercept = reference),
             inherit.aes = FALSE, linetype = "dashed", colour = "red") +
  facet_wrap(~parameter_clean, scales = "free_y", nrow = 1,
             labeller = label_parsed) +
  scale_fill_manual(values = group_cols) +
  scale_x_discrete(labels = group_axis_labels) +
  labs(x = NULL, y = "Posterior value", fill = NULL) +
  theme_bw(base_size = 16) +
  theme(
    panel.grid.major = element_line(colour = "grey88", linewidth = 0.40),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "grey40", linewidth = 0.45),
    strip.background = element_rect(
      fill = "grey82",
      colour = "grey40",
      linewidth = 0.45
    ),
    strip.text = element_text(face = "bold", size = 17),
    axis.title.y = element_text(face = "bold", size = 17),
    axis.text.x = element_text(face = "bold", size = 9.5),
    axis.text.y = element_text(face = "bold", size = 10),
    axis.ticks = element_line(linewidth = 0.35),
    legend.position = "none",
    plot.margin = margin(10, 12, 10, 10)
  )

save_plot_both(p_delta_eta, "delta_eta_posteriors_five_groups", 12, 5.5)

# ==============================================================================
# 6. POSTERIOR SUMMARIES AND REFERENCE-VALUE PROBABILITIES
# ==============================================================================

posterior_summary <- draws_long %>%
  group_by(group, parameter_clean) %>%
  summarise(
    mean = mean(value), median = median(value), sd = sd(value),
    l95 = quantile(value, .025), u95 = quantile(value, .975),
    .groups = "drop"
  )

writeup_table <- posterior_summary %>%
  mutate(result = sprintf("M = %.3f, 95%% CrI [%.3f, %.3f]", mean, l95, u95)) %>%
  select(group, parameter_clean, result) %>%
  pivot_wider(names_from = parameter_clean, values_from = result)

reference_stats <- draws_long %>%
  mutate(reference = unname(reference_values[as.character(parameter)])) %>%
  group_by(group, parameter_clean, reference) %>%
  summarise(
    probability_above = mean(value > reference),
    probability_below = mean(value < reference),
    l95 = quantile(value, .025), u95 = quantile(value, .975),
    credible_interval_excludes_reference = l95 > reference | u95 < reference,
    .groups = "drop"
  )

write_csv(posterior_summary, file.path(table_dir, "group_posterior_summaries.csv"))
write_csv(writeup_table, file.path(table_dir, "group_posterior_writeup_table.csv"))
write_csv(reference_stats, file.path(table_dir, "posterior_reference_value_tests.csv"))

cat("\n=========================================================\n")
cat("GROUP POSTERIOR SUMMARIES\n")
cat("=========================================================\n\n")
print(posterior_summary, n = Inf, width = Inf)

cat("\n=========================================================\n")
cat("WRITE-UP TABLE: M AND 95% CrI FOR EACH GROUP × PARAMETER\n")
cat("=========================================================\n\n")
print(writeup_table, n = Inf, width = Inf)

cat("\n=========================================================\n")
cat("POSTERIOR PROBABILITIES AGAINST REFERENCE VALUES\n")
cat("=========================================================\n\n")
print(reference_stats, n = Inf, width = Inf)

# ==============================================================================
# 7. BAYESIAN PAIRWISE POSTERIOR CONTRASTS
# ==============================================================================
# Groups were fitted independently. Equal numbers of draws are paired only to
# generate Monte Carlo draws from the difference distribution.

posterior_contrast <- function(parameter_name, group_1, group_2) {
  x1 <- draws_long %>%
    filter(parameter == parameter_name, group == group_1) %>% pull(value)
  x2 <- draws_long %>%
    filter(parameter == parameter_name, group == group_2) %>% pull(value)
  n <- min(length(x1), length(x2))
  difference <- sample(x1, n) - sample(x2, n)
  tibble(
    parameter = unname(parameter_labels[parameter_name]),
    group_1 = group_1, group_2 = group_2,
    mean_difference = mean(difference),
    l95 = quantile(difference, .025), u95 = quantile(difference, .975),
    probability_group1_greater = mean(difference > 0),
    probability_group1_lower = mean(difference < 0),
    credible_interval_excludes_zero = l95 > 0 | u95 < 0
  )
}

set.seed(SEED)
group_pairs <- combn(group_levels, 2, simplify = FALSE)
posterior_pairwise <- map_dfr(parameter_names, function(parameter_name) {
  map_dfr(group_pairs, ~posterior_contrast(parameter_name, .x[1], .x[2]))
})

write_csv(posterior_pairwise, file.path(table_dir, "bayesian_pairwise_parameter_contrasts.csv"))

cat("\n=========================================================\n")
cat("ALL BAYESIAN PAIRWISE PARAMETER CONTRASTS\n")
cat("=========================================================\n\n")
print(posterior_pairwise, n = Inf, width = Inf)

cat("\n=========================================================\n")
cat("CONTRASTS WHOSE 95% CrI EXCLUDES ZERO\n")
cat("=========================================================\n\n")
print(
  posterior_pairwise %>%
    filter(credible_interval_excludes_zero),
  n = Inf,
  width = Inf
)

# ==============================================================================
# 8. PARTICIPANT-LEVEL ESTIMATES AND RANDOMISATION TESTS
# ==============================================================================

extract_participant_params <- function(fit, group_name) {
  s <- fit$summary(variables = "params") %>% as_tibble()
  parsed <- str_match(s$variable, "^params\\[([0-9]+),([0-9]+)\\]$")
  out <- tibble(
    participant_index = as.integer(parsed[, 2]),
    parameter_index = as.integer(parsed[, 3]),
    estimate = s$mean
  ) %>%
    filter(!is.na(participant_index), parameter_index %in% 1:5) %>%
    mutate(
      parameter = c("alpha", "beta", "lambda", "delta", "eta")[parameter_index],
      group = group_name,
      participant_id = paste(group_name, participant_index, sep = "__")
    ) %>%
    select(participant_id, participant_index, group, parameter, estimate)
  if (nrow(out) == 0) stop("No participant-level params found for ", group_name)
  out
}

randomisation_test <- function(data, parameter_name, group_1, group_2,
                               n_perm = N_PERM) {
  d <- data %>%
    filter(parameter == parameter_name, group %in% c(group_1, group_2),
           is.finite(estimate))
  observed <- mean(d$estimate[d$group == group_1]) -
    mean(d$estimate[d$group == group_2])
  permuted <- replicate(n_perm, {
    shuffled <- sample(d$group, replace = FALSE)
    mean(d$estimate[shuffled == group_1]) - mean(d$estimate[shuffled == group_2])
  })
  tibble(
    parameter = parameter_name, group_1 = group_1, group_2 = group_2,
    n_group_1 = sum(d$group == group_1), n_group_2 = sum(d$group == group_2),
    mean_group_1 = mean(d$estimate[d$group == group_1]),
    mean_group_2 = mean(d$estimate[d$group == group_2]),
    observed_difference = observed,
    randomisation_p = (sum(abs(permuted) >= abs(observed)) + 1) / (n_perm + 1)
  )
}

if (RUN_PARTICIPANT_TESTS) {
  participant_parameters <- imap_dfr(fits, extract_participant_params)
  participant_counts <- participant_parameters %>%
    distinct(group, participant_id) %>% count(group, name = "N")
  print(participant_counts)
  
  set.seed(SEED)
  randomisation_stats <- map_dfr(c("alpha", "beta", "lambda", "delta", "eta"),
                                 function(parameter_name) {
                                   map_dfr(group_pairs, ~randomisation_test(
                                     participant_parameters, parameter_name, .x[1], .x[2]
                                   ))
                                 }) %>%
    group_by(parameter) %>%
    mutate(p_holm = p.adjust(randomisation_p, method = "holm")) %>%
    ungroup()
  
  write_csv(participant_parameters,
            file.path(table_dir, "participant_parameter_posterior_means.csv"))
  write_csv(randomisation_stats,
            file.path(table_dir, "pairwise_parameter_randomisation_tests.csv"))
  
  cat("\n=========================================================\n")
  cat("ALL PARTICIPANT-LEVEL RANDOMISATION TESTS\n")
  cat("=========================================================\n\n")
  print(randomisation_stats, n = Inf, width = Inf)
  
  cat("\n=========================================================\n")
  cat("HOLM-CORRECTED RANDOMISATION TESTS WITH p < .05\n")
  cat("=========================================================\n\n")
  print(
    randomisation_stats %>%
      filter(p_holm < .05) %>%
      arrange(parameter, p_holm),
    n = Inf,
    width = Inf
  )
}

# ==============================================================================
# 9. LOO MODEL COMPARISON — FIVE MODELS × FIVE GROUPS
# ==============================================================================

loo_file_map <- list(
  "Implicit Unaware Base Rate" = c(
    simple = "loo_trunc_simple_model_unaware_exp11.rdata",
    learning = "loo_trunc_learning_model_unaware_exp11.rdata",
    local_eta = "loo_trunc_eta_model_unaware_exp11.rdata",
    learning_local_eta = "loo_trunc_boost_model_unaware_exp11.rdata",
    learning_global_eta = "loo_trunc_global_eta_unaware_exp11.rdata"),
  "Implicit Aware Base Rate" = c(
    simple = "loo_trunc_simple_model_aware_exp11.rdata",
    learning = "loo_trunc_learning_model_aware_exp11.rdata",
    local_eta = "loo_trunc_eta_model_aware_exp11.rdata",
    learning_local_eta = "loo_trunc_boost_model_aware_exp11.rdata",
    learning_global_eta = "loo_trunc_global_eta_aware_exp11.rdata"),
  "Explicit Undirected Base Rate" = c(
    simple = "loo_trunc_simple_model_aware_exp12.rdata",
    learning = "loo_trunc_learning_model_aware_exp12.rdata",
    local_eta = "loo_trunc_eta_model_aware_exp12.rdata",
    learning_local_eta = "loo_trunc_boost_model_aware_exp12.rdata",
    learning_global_eta = "loo_trunc_global_eta_aware_exp12.rdata"),
  "Explicit True Base Rate" = c(
    simple = "loo_trunc_simple_truthful_exp13.rdata",
    learning = "loo_trunc_learning_truthful_exp13.rdata",
    local_eta = "loo_trunc_eta_truthful_exp13.rdata",
    learning_local_eta = "loo_trunc_boost_truthful_exp13.rdata",
    learning_global_eta = "loo_trunc_global_eta_truthful_exp13.rdata"),
  "Explicit Deceptive Base Rate" = c(
    simple = "loo_trunc_simple_deceptive_exp13.rdata",
    learning = "loo_trunc_learning_deceptive_exp13.rdata",
    local_eta = "loo_trunc_eta_deceptive_exp13.rdata",
    learning_local_eta = "loo_trunc_boost_deceptive_exp13.rdata",
    learning_global_eta = "loo_trunc_global_eta_deceptive_exp13.rdata")
)

load_loo_object <- function(path) {
  if (!file.exists(path)) stop("LOO file not found: ", path)
  e <- new.env(parent = emptyenv())
  loaded <- load(path, envir = e)
  candidates <- loaded[map_lgl(loaded, ~inherits(e[[.x]], "loo"))]
  if (length(candidates) != 1L) {
    stop("Expected exactly one loo object in: ", path,
         "; found ", length(candidates))
  }
  e[[candidates]]
}

compare_loo_group <- function(files, group_name) {
  objects <- set_names(map(unname(files),
                           ~load_loo_object(file.path(loo_dir, .x))), names(files))
  n_obs <- map_int(objects, ~nrow(.x$pointwise))
  if (n_distinct(n_obs) != 1L) {
    stop("LOO models use different observation counts for ", group_name)
  }
  comparison <- loo_compare(objects) %>% as.data.frame() %>%
    rownames_to_column("model") %>% as_tibble()
  estimates <- imap_dfr(objects, function(x, model_name) {
    tibble(
      model = model_name,
      elpd_loo = x$estimates["elpd_loo", "Estimate"],
      se_elpd_loo = x$estimates["elpd_loo", "SE"],
      p_loo = x$estimates["p_loo", "Estimate"],
      se_p_loo = x$estimates["p_loo", "SE"],
      looic = x$estimates["looic", "Estimate"],
      se_looic = x$estimates["looic", "SE"]
    )
  })
  comparison %>%
    select(model, elpd_diff, se_diff) %>%
    left_join(estimates, by = "model") %>%
    mutate(group = group_name, n_observations = unique(n_obs), .before = 1)
}

if (RUN_LOO_COMPARISON) {
  loo_summary <- imap_dfr(loo_file_map, compare_loo_group) %>%
    mutate(
      group = factor(group, levels = group_levels),
      model_label = recode(model,
                           simple = "Simple", learning = "Learning", local_eta = "Local eta",
                           learning_local_eta = "Learning + Local eta",
                           learning_global_eta = "Learning + Global eta")
    )
  
  best_models <- loo_summary %>% group_by(group) %>%
    slice_max(elpd_loo, n = 1, with_ties = FALSE) %>% ungroup()
  
  p_loo_diff <- ggplot(loo_summary,
                       aes(model_label, elpd_diff, colour = group, group = group)) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    geom_errorbar(aes(ymin = elpd_diff - se_diff,
                      ymax = elpd_diff + se_diff), width = .15) +
    geom_point(size = 2.5) +
    facet_wrap(~group, scales = "free_y") +
    scale_colour_manual(values = group_cols, guide = "none") +
    labs(x = NULL, y = "ELPD difference from best model") +
    theme_bw(base_size = 11) +
    theme(axis.text.x = element_text(angle = 35, hjust = 1))
  
}

