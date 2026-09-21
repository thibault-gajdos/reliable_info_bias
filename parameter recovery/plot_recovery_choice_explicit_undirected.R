#!/usr/bin/env Rscript

# ============================================================================
# Plot Parameter Recovery — Learning Choice Model: Explicit Undirected
# ============================================================================
#
# This script is for the Apocrita parameter-recovery output:
#
#   results/choice_learning_explicit_undirected/recover_1.rds
#   results/choice_learning_explicit_undirected/recover_2.rds
#   ...
#   results/choice_learning_explicit_undirected/recover_243.rds
#
# It combines all recovery files, checks diagnostics, and saves recovery plots.
#
# Run from:
#   ~/parameter_recovery/reliable_info_bias
#
# Command:
#   module load R
#   Rscript plot_recovery_choice_explicit_undirected.R
#
# ============================================================================

rm(list = ls(all = TRUE))

# Use the current working directory as the project root.
# On Apocrita, this should be:
#   /data/home/bty615/parameter_recovery/reliable_info_bias
project_dir <- getwd()

# Load renv if available. This may print an "out-of-sync" warning, but that is
# not fatal if the required packages load.
if (requireNamespace("renv", quietly = TRUE)) {
  try(renv::load(project = project_dir), silent = TRUE)
}

.libPaths(c(path.expand("~/Rlibs"), .libPaths()))

library(dplyr)
library(tidyr)
library(ggplot2)
library(tibble)

results_dir <- file.path(project_dir, "results", "choice_learning_explicit_undirected")
plot_dir    <- file.path(project_dir, "figures", "choice_learning_explicit_undirected")

dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# 1. LOAD ALL RESULTS
# ============================================================================

files <- list.files(
  results_dir,
  pattern = "^recover_[0-9]+\\.rds$",
  full.names = TRUE
)

cat(sprintf("Found %d result files.\n", length(files)))

if (length(files) == 0) {
  stop("No result files found in: ", results_dir)
}

read_safe <- function(f) {
  tryCatch(
    readRDS(f),
    error = function(e) {
      warning("Could not read file: ", f, "\n", conditionMessage(e))
      NULL
    }
  )
}

res_list <- lapply(files, read_safe)
res_list <- res_list[!vapply(res_list, is.null, logical(1))]

if (length(res_list) == 0) {
  stop("No readable recovery files.")
}

ks <- vapply(res_list, function(x) x$k, numeric(1))
res_list <- res_list[order(ks)]
ks <- sort(ks)

cat(sprintf("Readable files: %d\n", length(res_list)))
cat(sprintf("k range: %d to %d\n", min(ks), max(ks)))

missing_k <- setdiff(1:243, ks)

cat("\n=== Completion check ===\n")
cat(sprintf("Completed files: %d / 243\n", length(unique(ks))))

if (length(missing_k) == 0) {
  cat("No missing k values.\n")
} else {
  cat("Missing k values:\n")
  print(missing_k)
}

# ============================================================================
# 2. PARAMETER DEFINITIONS
# ============================================================================

param_order <- c("alpha", "beta", "lambda", "delta", "eta")

param_info <- list(
  alpha  = list(sim = "Simulated alpha",  fit = "Fitted alpha"),
  beta   = list(sim = "Simulated beta",   fit = "Fitted beta"),
  lambda = list(sim = "Simulated lambda", fit = "Fitted lambda"),
  delta  = list(sim = "Simulated delta",  fit = "Fitted delta"),
  eta    = list(sim = "Simulated eta",    fit = "Fitted eta")
)

panel_letters <- LETTERS[seq_along(param_order)]

# ============================================================================
# 3. COMBINE GROUP-LEVEL RESULTS
# ============================================================================

extract_group <- function(res) {
  
  gf <- res$group_fitted
  
  true_vals <- res$group_sim
  names(true_vals) <- res$param_names
  
  gf %>%
    mutate(
      k = res$k,
      param = sub("^mu_", "", variable),
      simulated = true_vals[param],
      estimated = mean,
      q5 = q5,
      q95 = q95,
      bias = estimated - simulated,
      abs_bias = abs(bias),
      covered_90 = simulated >= q5 & simulated <= q95
    ) %>%
    select(
      k, variable, param,
      simulated, estimated, median, q5, q95,
      bias, abs_bias, covered_90,
      rhat, ess_bulk, ess_tail
    )
}

df_group <- bind_rows(lapply(res_list, extract_group)) %>%
  mutate(param = factor(param, levels = param_order))

# ============================================================================
# 4. COMBINE INDIVIDUAL-LEVEL RESULTS
# ============================================================================

extract_indiv <- function(res) {
  
  fitted <- res$indiv_fitted
  true_mat <- res$params_indiv_sim
  
  # Fitted variables are named like params[12,3].
  idx <- regexec("params\\[([0-9]+),([0-9]+)\\]", fitted$variable)
  parsed <- regmatches(fitted$variable, idx)
  
  subject_vec <- as.integer(vapply(parsed, function(x) x[2], character(1)))
  param_index <- as.integer(vapply(parsed, function(x) x[3], character(1)))
  param_vec   <- res$param_names[param_index]
  
  simulated_vec <- mapply(
    function(s, p) true_mat[s, p],
    subject_vec,
    param_index
  )
  
  fitted %>%
    mutate(
      k = res$k,
      subject = subject_vec,
      param = param_vec,
      simulated = as.numeric(simulated_vec),
      estimated = mean,
      q5 = q5,
      q95 = q95,
      bias = estimated - simulated,
      abs_bias = abs(bias),
      covered_90 = simulated >= q5 & simulated <= q95
    ) %>%
    select(
      k, variable, subject, param,
      simulated, estimated, median, q5, q95,
      bias, abs_bias, covered_90,
      rhat, ess_bulk, ess_tail
    )
}

df_indiv <- bind_rows(lapply(res_list, extract_indiv)) %>%
  mutate(param = factor(param, levels = param_order))

# ============================================================================
# 5. DIAGNOSTICS
# ============================================================================

df_diag <- bind_rows(lapply(res_list, function(res) {
  tibble(
    k = res$k,
    n_divergent = res$n_divergent,
    n_max_td = res$n_max_td,
    max_rhat = res$max_rhat,
    min_ess = res$min_ess
  )
}))

cat("\n=== Diagnostics summary ===\n")
cat(sprintf("Total runs: %d\n", nrow(df_diag)))
cat(sprintf("Runs with divergences: %d / %d\n",
            sum(df_diag$n_divergent > 0, na.rm = TRUE), nrow(df_diag)))
cat(sprintf("Runs with max Rhat > 1.05: %d / %d\n",
            sum(df_diag$max_rhat > 1.05, na.rm = TRUE), nrow(df_diag)))
cat(sprintf("Runs with min ESS < 400: %d / %d\n",
            sum(df_diag$min_ess < 400, na.rm = TRUE), nrow(df_diag)))
cat("\n")
print(summary(df_diag))

# Keep a diagnostics-filtered version for the main recovery plots.
# This is deliberately lenient. The full diagnostics table is still saved.
good_fits <- df_diag %>%
  filter(max_rhat < 1.1, n_divergent < 50)

cat(sprintf(
  "\nKeeping %d / %d fits for recovery plots using max_rhat < 1.1 and divergences < 50.\n",
  nrow(good_fits), nrow(df_diag)
))

df_group_good <- df_group %>% filter(k %in% good_fits$k)
df_indiv_good <- df_indiv %>% filter(k %in% good_fits$k)

# Save combined data tables.
write.csv(df_group, file.path(plot_dir, "group_recovery_all_runs_explicit_undirected.csv"), row.names = FALSE)
write.csv(df_indiv, file.path(plot_dir, "individual_recovery_all_runs_explicit_undirected.csv"), row.names = FALSE)
write.csv(df_diag,  file.path(plot_dir, "diagnostics_all_runs_explicit_undirected.csv"), row.names = FALSE)

# ============================================================================
# 6. SUMMARY TABLES
# ============================================================================

summarise_recovery <- function(df) {
  df %>%
    group_by(param) %>%
    summarise(
      n = n(),
      r = cor(simulated, estimated, use = "pairwise.complete.obs"),
      bias = mean(estimated - simulated, na.rm = TRUE),
      mean_abs_bias = mean(abs(estimated - simulated), na.rm = TRUE),
      rmse = sqrt(mean((estimated - simulated)^2, na.rm = TRUE)),
      coverage_90 = mean(covered_90, na.rm = TRUE),
      max_rhat = max(rhat, na.rm = TRUE),
      min_ess_bulk = min(ess_bulk, na.rm = TRUE),
      .groups = "drop"
    )
}

group_stats <- summarise_recovery(df_group_good)
indiv_stats <- summarise_recovery(df_indiv_good)

cat("\n=== Group-level recovery summary ===\n")
print(as.data.frame(group_stats), digits = 3)

cat("\n=== Individual-level recovery summary ===\n")
print(as.data.frame(indiv_stats), digits = 3)

write.csv(group_stats, file.path(plot_dir, "group_recovery_summary_explicit_undirected.csv"), row.names = FALSE)
write.csv(indiv_stats, file.path(plot_dir, "individual_recovery_summary_explicit_undirected.csv"), row.names = FALSE)

# ============================================================================
# 7. PLOTTING HELPERS
# ============================================================================

format_p <- function(p) {
  if (is.na(p)) return("italic(p) == NA")
  if (p < 2.2e-16) return("italic(p) < 2.2e-16")
  sprintf("italic(p) == %.1e", p)
}

make_panel <- function(df, par, letter, show_errorbars = FALSE, point_size = 0.8,
                       point_alpha = 0.4) {
  
  dd <- df %>% filter(param == par)
  
  if (nrow(dd) < 2) {
    stop("Not enough rows for parameter: ", par)
  }
  
  all_vals <- c(dd$simulated, dd$estimated)
  
  if (show_errorbars && all(c("q5", "q95") %in% names(dd))) {
    all_vals <- c(all_vals, dd$q5, dd$q95)
  }
  
  axis_lim <- range(all_vals, na.rm = TRUE)
  axis_lim <- axis_lim + c(-0.08, 0.08) * diff(axis_lim)
  
  ct <- cor.test(dd$simulated, dd$estimated)
  
  label_text <- sprintf(
    "italic(R) == %.2f*','~%s",
    ct$estimate,
    format_p(ct$p.value)
  )
  
  p <- ggplot(dd, aes(x = simulated, y = estimated)) +
    geom_abline(
      intercept = 0,
      slope = 1,
      linetype = "dashed",
      colour = "black"
    ) +
    geom_smooth(
      method = "lm",
      se = FALSE,
      colour = "red",
      linewidth = 0.8
    )
  
  if (show_errorbars && all(c("q5", "q95") %in% names(dd))) {
    p <- p +
      geom_linerange(
        aes(ymin = q5, ymax = q95),
        linewidth = 0.25,
        colour = "black",
        alpha = 0.45
      )
  }
  
  p <- p +
    geom_point(
      size = point_size,
      alpha = point_alpha,
      colour = "black"
    ) +
    annotate(
      "text",
      x = -Inf,
      y = Inf,
      hjust = -0.08,
      vjust = 1.5,
      label = label_text,
      parse = TRUE,
      size = 3.2
    ) +
    scale_x_continuous(limits = axis_lim, expand = c(0, 0)) +
    scale_y_continuous(limits = axis_lim, expand = c(0, 0)) +
    coord_fixed() +
    labs(
      x = param_info[[par]]$sim,
      y = param_info[[par]]$fit,
      tag = letter
    ) +
    theme_classic(base_size = 11) +
    theme(
      plot.tag = element_text(face = "bold", size = 13),
      axis.title = element_text(size = 10),
      plot.margin = margin(5, 10, 5, 5)
    )
  
  return(p)
}

save_plot_grid <- function(panels, filename_base, width = 12, height = 7) {
  
  # Arrange manually without patchwork, so this script only depends on ggplot2.
  # The saved output is a list of panels in one PDF/PNG using gridExtra if present;
  # otherwise it saves each panel separately.
  
  if (requireNamespace("patchwork", quietly = TRUE)) {
    p <- (panels[[1]] | panels[[2]] | panels[[3]]) /
      (panels[[4]] | panels[[5]] | patchwork::plot_spacer())
    
    ggsave(file.path(plot_dir, paste0(filename_base, ".pdf")), p, width = width, height = height)
    ggsave(file.path(plot_dir, paste0(filename_base, ".png")), p, width = width, height = height, dpi = 300)
    
  } else {
    warning("Package 'patchwork' is not installed. Saving panels separately.")
    
    for (i in seq_along(panels)) {
      ggsave(
        file.path(plot_dir, sprintf("%s_panel_%s.pdf", filename_base, panel_letters[i])),
        panels[[i]],
        width = 4,
        height = 4
      )
      ggsave(
        file.path(plot_dir, sprintf("%s_panel_%s.png", filename_base, panel_letters[i])),
        panels[[i]],
        width = 4,
        height = 4,
        dpi = 300
      )
    }
  }
}

# ============================================================================
# 8. GROUP-LEVEL RECOVERY PLOT
# ============================================================================

panels_group <- lapply(seq_along(param_order), function(i) {
  make_panel(
    df = df_group_good,
    par = param_order[i],
    letter = panel_letters[i],
    show_errorbars = TRUE,
    point_size = 1.5,
    point_alpha = 0.75
  )
})

save_plot_grid(
  panels = panels_group,
  filename_base = "recovery_group_learning_explicit_undirected",
  width = 12,
  height = 7
)

cat("\nGroup-level recovery plot saved.\n")

# ============================================================================
# 9. INDIVIDUAL-LEVEL RECOVERY PLOT
# ============================================================================

panels_indiv <- lapply(seq_along(param_order), function(i) {
  make_panel(
    df = df_indiv_good,
    par = param_order[i],
    letter = panel_letters[i],
    show_errorbars = FALSE,
    point_size = 0.45,
    point_alpha = 0.18
  )
})

save_plot_grid(
  panels = panels_indiv,
  filename_base = "recovery_individual_learning_explicit_undirected",
  width = 12,
  height = 7
)

cat("Individual-level recovery plot saved.\n")

# ============================================================================
# 10. BIAS PLOTS
# ============================================================================

p_group_bias <- ggplot(df_group_good, aes(x = param, y = bias)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.5) +
  geom_boxplot(outlier.shape = NA, width = 0.55) +
  geom_jitter(width = 0.12, height = 0, size = 1.1, alpha = 0.35) +
  labs(
    title = "Group-level recovery bias across grid",
    subtitle = "Bias = recovered posterior mean minus true generating value.",
    x = "Parameter",
    y = "Recovery bias"
  ) +
  theme_classic(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(plot_dir, "recovery_group_bias_learning_explicit_undirected.pdf"), p_group_bias, width = 8, height = 5)
ggsave(file.path(plot_dir, "recovery_group_bias_learning_explicit_undirected.png"), p_group_bias, width = 8, height = 5, dpi = 300)

p_indiv_bias <- ggplot(df_indiv_good, aes(x = param, y = bias)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.5) +
  geom_boxplot(outlier.shape = NA, width = 0.55) +
  labs(
    title = "Individual-level recovery bias across grid",
    subtitle = "Bias = recovered posterior mean minus true simulated value.",
    x = "Parameter",
    y = "Recovery bias"
  ) +
  theme_classic(base_size = 13) +
  theme(plot.title = element_text(face = "bold"))

ggsave(file.path(plot_dir, "recovery_individual_bias_learning_explicit_undirected.pdf"), p_indiv_bias, width = 8, height = 5)
ggsave(file.path(plot_dir, "recovery_individual_bias_learning_explicit_undirected.png"), p_indiv_bias, width = 8, height = 5, dpi = 300)

cat("Bias plots saved.\n")

# ============================================================================
# 11. DIAGNOSTICS PLOTS
# ============================================================================

p_rhat <- ggplot(df_diag, aes(x = max_rhat)) +
  geom_histogram(bins = 50, fill = "grey30") +
  geom_vline(xintercept = 1.05, linetype = "dashed", colour = "red") +
  labs(
    title = "Maximum Rhat across recovery runs",
    x = "Max Rhat",
    y = "Count"
  ) +
  theme_classic(base_size = 11)

ggsave(file.path(plot_dir, "diagnostics_max_rhat_learning_explicit_undirected.pdf"), p_rhat, width = 7, height = 4)
ggsave(file.path(plot_dir, "diagnostics_max_rhat_learning_explicit_undirected.png"), p_rhat, width = 7, height = 4, dpi = 300)

p_ess <- ggplot(df_diag, aes(x = min_ess)) +
  geom_histogram(bins = 50, fill = "grey30") +
  geom_vline(xintercept = 400, linetype = "dashed", colour = "red") +
  labs(
    title = "Minimum ESS across recovery runs",
    x = "Minimum bulk ESS",
    y = "Count"
  ) +
  theme_classic(base_size = 11)

ggsave(file.path(plot_dir, "diagnostics_min_ess_learning_explicit_undirected.pdf"), p_ess, width = 7, height = 4)
ggsave(file.path(plot_dir, "diagnostics_min_ess_learning_explicit_undirected.png"), p_ess, width = 7, height = 4, dpi = 300)

p_div <- ggplot(df_diag, aes(x = n_divergent)) +
  geom_histogram(bins = 50, fill = "grey30") +
  labs(
    title = "Divergences across recovery runs",
    x = "Number of divergent transitions",
    y = "Count"
  ) +
  theme_classic(base_size = 11)

ggsave(file.path(plot_dir, "diagnostics_divergences_learning_explicit_undirected.pdf"), p_div, width = 7, height = 4)
ggsave(file.path(plot_dir, "diagnostics_divergences_learning_explicit_undirected.png"), p_div, width = 7, height = 4, dpi = 300)

cat("Diagnostics plots saved.\n")

# ============================================================================
# 12. PARAMETER CORRELATION PLOT
# ============================================================================

cat("Building parameter correlation plot...\n")

df_indiv_wide <- df_indiv_good %>%
  select(k, subject, param, estimated) %>%
  pivot_wider(names_from = param, values_from = estimated)

all_k <- sort(unique(df_indiv_wide$k))
n_params <- length(param_order)

cor_mats <- array(NA_real_, dim = c(n_params, n_params, length(all_k)))

for (i in seq_along(all_k)) {
  dd <- df_indiv_wide %>% filter(k == all_k[i])
  mat <- as.matrix(dd[, param_order])
  cor_mats[, , i] <- cor(mat, use = "pairwise.complete.obs")
}

mean_cor <- apply(cor_mats, c(1, 2), mean, na.rm = TRUE)

rownames(mean_cor) <- param_order
colnames(mean_cor) <- param_order

cor_long <- expand.grid(
  x = param_order,
  y = param_order,
  stringsAsFactors = FALSE
)

cor_long$r <- as.vector(mean_cor)
cor_long$x <- factor(cor_long$x, levels = param_order)
cor_long$y <- factor(cor_long$y, levels = rev(param_order))

p_corr <- ggplot(cor_long, aes(x = x, y = y, fill = r)) +
  geom_tile(colour = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", r)), size = 3) +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "#B2182B",
    midpoint = 0,
    limits = c(-1, 1),
    name = "Mean r"
  ) +
  scale_x_discrete(labels = param_order) +
  scale_y_discrete(labels = rev(param_order)) +
  labs(
    title = sprintf("Mean recovered parameter correlations across %d fits", length(all_k))
  ) +
  coord_fixed() +
  theme_minimal(base_size = 11) +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  )

ggsave(file.path(plot_dir, "param_correlations_learning_explicit_undirected.pdf"), p_corr, width = 7, height = 6)
ggsave(file.path(plot_dir, "param_correlations_learning_explicit_undirected.png"), p_corr, width = 7, height = 6, dpi = 300)

cat("\nMean within-fit correlation matrix:\n")
print(round(mean_cor, 3))
cat("\n")

cat("Parameter correlation plot saved.\n")

cat("\nAll figures and tables saved to:\n")
cat(plot_dir, "\n")