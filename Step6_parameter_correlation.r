
rm(list=ls(all=TRUE))  ## efface les données
setwd(/Users/bty615/Documents/GitHub/)
source('utils.r')

library(rstan)
library(tidyverse)

exp <- 12

fit.all <- readRDS('../results/fits/Exp11/log_seq_basic_prior_exp11.rds')
#fit.aware <- readRDS('../results/fits/Exp11/log_seq_basic_prior_aware_exp11.rds')

###################################################
### create a matrix of MCMC output plots 
### A matrix of group (global) parameters##########
###################################################

# all participants
pairs(fit.all, pars = c('mu_bias','mu_alpha', 'mu_beta'))
pairs(fit.all, pars = c('mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5'))

# aware subgroup only
file.name <- paste('../results/fits/Exp11/corrplot_aware','.png', sep='')
png(file.name, width=1600,height=1600, res=300)
pairs(fit.aware, pars = c('mu_bias','mu_alpha', 'mu_beta'))
dev.off()

file.name <- paste('../results/fits/Exp11/corrplot_weight_aware','.png', sep='')
png(file.name, width=2000,height=2000, res=300)
pairs(fit.aware, pars = c('mu_w1', 'mu_w2', 'mu_w3', 'mu_w4', 'mu_w5'))
dev.off()



library(tidyverse)
library(cmdstanr)
library(posterior)
library(bayesplot)

###################################################
### Helper: load first object from .rdata
###################################################

load_fit <- function(file_path) {
  obj_name <- load(file_path)
  get(obj_name[1])
}

###################################################
### Load three fit objects
###################################################

fit_unaware_exp11 <- load_fit("results/fits/Exp12/fit_trunc_boost_model_unaware_exp11.rdata")
fit_aware_exp11   <- load_fit("results/fits/Exp12/fit_trunc_boost_model_aware_exp11.rdata")
fit_aware_exp12   <- load_fit("results/fits/Exp12/fit_trunc_boost_model_aware_exp12.rdata")

fit_list <- list(
  unaware_exp11 = fit_unaware_exp11,
  aware_exp11   = fit_aware_exp11,
  aware_exp12   = fit_aware_exp12
)

###################################################
### Output folder
###################################################

out_dir <- "results/fits/Exp12/"

###################################################
### Parameters to plot
###################################################

pars_to_plot <- c("mu_alpha", "mu_beta", "mu_delta", "mu_lambda", "mu_eta")

###################################################
### Function to save pairs plots for cmdstanr fits
###################################################

save_pairs_plot <- function(fit_obj, fit_name, out_dir, pars) {
  
  draws_df <- as_draws_df(fit_obj$draws(variables = pars))
  
  file_name <- paste0(out_dir, "corrplot_", fit_name, ".png")
  
  png(file_name, width = 2200, height = 2200, res = 300)
  print(mcmc_pairs(draws_df, pars = pars))
  dev.off()
  
  message("Saved: ", file_name)
}

###################################################
### Loop through fits and save plots
###################################################

for (fit_name in names(fit_list)) {
  save_pairs_plot(fit_list[[fit_name]], fit_name, out_dir, pars_to_plot)
}












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