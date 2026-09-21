# =====================================================================
# EXPORT y_pred FOR BEHAVIOURAL POSTERIOR PREDICTIVE CHECKS
# =====================================================================

rm(list = ls(all.names = TRUE))

library(cmdstanr)
library(posterior)
library(tidyverse)


# ---------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------

project_root <- path.expand(
  "~/Documents/GitHub/reliable_info_bias"
)

fit_root <- file.path(
  project_root,
  "stan/results/fits/exp11_unaware"
)

output_root <- file.path(
  project_root,
  "stan/results/predictions/behavioural_ppc"
)

n_ppc_draws <- 200

if (!dir.exists(output_root)) {
  dir.create(
    output_root,
    recursive = TRUE
  )
}


# ---------------------------------------------------------------------
# 2. Files for the five groups
# ---------------------------------------------------------------------

group_files <- tribble(
  ~group, ~safe_name, ~data_file, ~fit_file,
  
  "Implicit Unaware",
  "implicit_unaware",
  file.path(project_root, "data/DATA_Unaware_Exp11.csv"),
  file.path(fit_root, "fit_trunc_boost_model_unaware_exp11.rdata"),
  
  "Implicit Aware",
  "implicit_aware",
  file.path(project_root, "data/DATA_Aware_Exp11.csv"),
  file.path(fit_root, "fit_trunc_boost_model_aware_exp11.rdata"),
  
  "Explicit Undirected",
  "explicit_undirected",
  file.path(project_root, "data/DATA_Aware_Exp12.csv"),
  file.path(fit_root, "fit_trunc_boost_model_aware_exp12.rdata"),
  
  "Explicit True",
  "explicit_true",
  file.path(project_root, "data/data_priorbelief_truthful_exp13.csv"),
  file.path(fit_root, "fit_trunc_boost_truthful_exp13.rdata"),
  
  "Explicit Deceptive",
  "explicit_deceptive",
  file.path(project_root, "data/data_priorbelief_deceptive_exp13.csv"),
  file.path(fit_root, "fit_trunc_boost_deceptive_exp13.rdata")
)


# ---------------------------------------------------------------------
# 3. Validate required files
# ---------------------------------------------------------------------

missing_files <- c(
  group_files$data_file[!file.exists(group_files$data_file)],
  group_files$fit_file[!file.exists(group_files$fit_file)]
)

if (length(missing_files) > 0) {
  stop(
    paste(
      "The following files could not be found:",
      paste(missing_files, collapse = "\n"),
      sep = "\n"
    )
  )
}


# ---------------------------------------------------------------------
# 4. Export one group
# ---------------------------------------------------------------------

export_group <- function(
    group_name,
    safe_name,
    data_file,
    fit_file,
    n_requested_draws
) {
  
  cat("\n============================================================\n")
  cat("GROUP: ", group_name, "\n", sep = "")
  cat("============================================================\n")
  
  # Read the behavioural data.
  behavioural_data <- readr::read_csv(
    data_file,
    show_col_types = FALSE,
    progress = FALSE
  )
  
  required_columns <- c(
    "ParticipantPrivateID",
    "TrialNumber"
  )
  
  missing_columns <- setdiff(
    required_columns,
    names(behavioural_data)
  )
  
  if (length(missing_columns) > 0) {
    stop(
      paste0(
        "Missing columns in ",
        data_file,
        ":\n",
        paste(missing_columns, collapse = "\n")
      )
    )
  }
  
  # Preserve the original participant order used to prepare the model.
  participant_order <- unique(
    behavioural_data$ParticipantPrivateID
  )
  
  manifest <- behavioural_data %>%
    mutate(
      subject_index = match(
        ParticipantPrivateID,
        participant_order
      ),
      trial = as.integer(TrialNumber)
    ) %>%
    arrange(
      subject_index,
      trial
    ) %>%
    transmute(
      column_position = row_number(),
      group = group_name,
      subject_index,
      participant_id = as.character(
        ParticipantPrivateID
      ),
      trial
    )
  
  if (anyDuplicated(
    manifest[c("subject_index", "trial")]
  )) {
    stop(
      paste0(
        "Duplicate participant/trial combinations found for ",
        group_name,
        "."
      )
    )
  }
  
  rows_per_subject <- manifest %>%
    count(
      subject_index,
      name = "n_trials"
    )
  
  if (any(rows_per_subject$n_trials != 240)) {
    stop(
      paste0(
        "At least one participant in ",
        group_name,
        " does not have exactly 240 trials."
      )
    )
  }
  
  # Load the CmdStanR fit.
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
        "No object named 'fit' was found for ",
        group_name,
        "."
      )
    )
  }
  
  current_fit <- fit_environment$fit
  
  if (!inherits(current_fit, "CmdStanMCMC")) {
    stop(
      paste0(
        "The saved fit for ",
        group_name,
        " is not a CmdStanMCMC object."
      )
    )
  }
  
  # Extract all posterior predictive choices.
  y_pred <- current_fit$draws(
    variables = "y_pred",
    format = "draws_matrix"
  )
  
  if (!all(
    unique(as.vector(y_pred)) %in% c(1, 2)
  )) {
    stop(
      paste0(
        "y_pred for ",
        group_name,
        " contains values other than 1 and 2."
      )
    )
  }
  
  # Match y_pred columns to the participant/trial manifest.
  expected_names <- paste0(
    "y_pred[",
    manifest$subject_index,
    ",",
    manifest$trial,
    "]"
  )
  
  column_match <- match(
    expected_names,
    colnames(y_pred)
  )
  
  if (anyNA(column_match)) {
    stop(
      paste0(
        "Not all y_pred columns could be matched for ",
        group_name,
        ". First missing variable: ",
        expected_names[which(is.na(column_match))[1]]
      )
    )
  }
  
  y_pred <- y_pred[
    ,
    column_match,
    drop = FALSE
  ]
  
  # Select evenly spaced draws across the full posterior.
  n_available_draws <- nrow(y_pred)
  
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
  
  selected_y_pred <- y_pred[
    selected_draw_rows,
    ,
    drop = FALSE
  ]
  
  # Save only the numeric prediction matrix.
  #
  # Rows = posterior draws
  # Columns = observations listed in the manifest
  prediction_file <- file.path(
    output_root,
    paste0(
      "y_pred_",
      safe_name,
      "_200_draws.csv"
    )
  )
  
  readr::write_csv(
    as.data.frame(selected_y_pred),
    prediction_file
  )
  
  # Save the manifest that defines every prediction column.
  manifest_file <- file.path(
    output_root,
    paste0(
      "manifest_",
      safe_name,
      ".csv"
    )
  )
  
  readr::write_csv(
    manifest,
    manifest_file
  )
  
  # Save which rows of the full posterior were retained.
  draw_file <- file.path(
    output_root,
    paste0(
      "selected_draws_",
      safe_name,
      ".csv"
    )
  )
  
  draw_manifest <- tibble(
    exported_draw = seq_along(
      selected_draw_rows
    ),
    full_posterior_draw_row =
      selected_draw_rows
  )
  
  readr::write_csv(
    draw_manifest,
    draw_file
  )
  
  cat("Participants: ", length(participant_order), "\n", sep = "")
  cat("Trials per participant: 240\n")
  cat("Available posterior draws: ", n_available_draws, "\n", sep = "")
  cat("Exported posterior draws: ", nrow(selected_y_pred), "\n", sep = "")
  cat("Prediction columns: ", ncol(selected_y_pred), "\n", sep = "")
  
  cat("\nSaved predictions:\n")
  cat(prediction_file, "\n")
  
  cat("\nSaved manifest:\n")
  cat(manifest_file, "\n")
  
  invisible(
    tibble(
      group = group_name,
      participants = length(participant_order),
      trials_per_participant = 240L,
      observations = ncol(selected_y_pred),
      available_draws = n_available_draws,
      exported_draws = nrow(selected_y_pred),
      prediction_file,
      manifest_file
    )
  )
}


# ---------------------------------------------------------------------
# 5. Export all five groups
# ---------------------------------------------------------------------

export_summary <- vector(
  mode = "list",
  length = nrow(group_files)
)

for (i in seq_len(nrow(group_files))) {
  
  export_summary[[i]] <- export_group(
    group_name = group_files$group[i],
    safe_name = group_files$safe_name[i],
    data_file = group_files$data_file[i],
    fit_file = group_files$fit_file[i],
    n_requested_draws = n_ppc_draws
  )
}

export_summary <- bind_rows(
  export_summary
)


# ---------------------------------------------------------------------
# 6. Save and print the export summary
# ---------------------------------------------------------------------

readr::write_csv(
  export_summary,
  file.path(
    output_root,
    "behavioural_ppc_export_summary.csv"
  )
)

cat("\n============================================================\n")
cat("BEHAVIOURAL PPC EXPORT COMPLETED\n")
cat("============================================================\n\n")

print(
  export_summary,
  n = Inf
)

cat("\nFiles saved in:\n")
cat(output_root, "\n")]








# SIMPLE BEHAVIOURAL POSTERIOR PREDICTIVE CHECKS
# Produces: accuracy, base-rate-aligned choice over trials, and evidence curves.

rm(list = ls(all.names = TRUE))

library(tidyverse)

project_root <- path.expand("~/Documents/GitHub/reliable_info_bias")
prediction_root <- file.path(project_root, "stan/results/predictions/behavioural_ppc")
output_root <- file.path(prediction_root, "simple_ppc_plots")
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)

MIN_TRIALS_PER_BIN <- 3
ROLLING_WINDOW <- 50

groups <- tribble(
  ~group, ~safe, ~data_file, ~order_col, ~prior_col,
  "Implicit Unaware", "implicit_unaware", "data/DATA_Unaware_Exp11.csv", "ResponseButtonOrder", "Prior_Belief",
  "Implicit Aware", "implicit_aware", "data/DATA_Aware_Exp11.csv", "ResponseButtonOrder", "Prior_Belief",
  "Explicit Undirected", "explicit_undirected", "data/DATA_Aware_Exp12.csv", "Manipulation_ResponseButtonOrder", "Prior_Belief",
  "Explicit True", "explicit_true", "data/data_priorbelief_truthful_exp13.csv", "ResponseButtonOrder", "TruePrior",
  "Explicit Deceptive", "explicit_deceptive", "data/data_priorbelief_deceptive_exp13.csv", "ResponseButtonOrder", "TruePrior"
) %>% mutate(data_file = file.path(project_root, data_file))

group_levels <- groups$group
group_cols <- c(
  "Implicit Unaware" = "#F28E8E", "Implicit Aware" = "#8FD18F",
  "Explicit Undirected" = "#E58C22", "Explicit True" = "#087A3E",
  "Explicit Deceptive" = "#A51F1F"
)
rel_cols <- c(`50%` = "#0072B2", `55%` = "#D55E00", `65%` = "#7B3294")

normalise_prior <- function(x) {
  z <- tolower(as.character(x))
  case_when(
    z %in% c("1", "blue", "b", "65_blue", "blueprior") ~ 1L,
    z %in% c("2", "red", "r", "65_red", "redprior") ~ 2L,
    str_detect(z, "blue") ~ 1L,
    str_detect(z, "red") ~ 2L,
    TRUE ~ suppressWarnings(as.integer(z))
  )
}

# Extract six sample colours and reliabilities. Newer files contain color_1...
# and proba_1...; older files contain string representations in Sample_*.
extract_samples <- function(d) {
  n <- nrow(d)
  cols <- matrix(NA_character_, n, 6)
  rels <- matrix(NA_real_, n, 6)
  if (all(paste0("color_", 1:6) %in% names(d)) &&
      all(paste0("proba_", 1:6) %in% names(d))) {
    cols <- as.matrix(d[paste0("color_", 1:6)])
    rels <- apply(d[paste0("proba_", 1:6)], 2, as.numeric)
    # Exp13 files may store reliability as 50/55/65 rather than
    # 0.50/0.55/0.65.
    rels[rels > 1] <- rels[rels > 1] / 100
  } else {
    for (i in seq_len(n)) {
      cs <- str_extract_all(tolower(as.character(d$Sample_Color[i])), "blue|red")[[1]]
      rs <- suppressWarnings(as.numeric(str_extract_all(
        as.character(d$Sample_Reliability[i]), "[0-9]+(?:\\.[0-9]+)?")[[1]]))
      rs[rs > 1] <- rs[rs > 1] / 100
      if (length(cs)) cols[i, seq_len(min(6, length(cs)))] <- cs[seq_len(min(6, length(cs)))]
      if (length(rs)) rels[i, seq_len(min(6, length(rs)))] <- rs[seq_len(min(6, length(rs)))]
    }
  }
  list(colour = tolower(cols), reliability = rels)
}

add_evidence <- function(d) {
  s <- extract_samples(d)
  # Neutral Bayesian probability that Blue is correct, using a 0.5/0.5
  # prior and the stated reliability of every sample.
  rb <- ifelse(s$colour == "blue", s$reliability, 1 - s$reliability)
  rr <- ifelse(s$colour == "red",  s$reliability, 1 - s$reliability)
  rb[is.na(rb)] <- 0.5
  rr[is.na(rr)] <- 0.5
  rb <- pmin(pmax(rb, 1e-8), 1 - 1e-8)
  rr <- pmin(pmax(rr, 1e-8), 1 - 1e-8)
  d$neutral_blue_probability <- plogis(rowSums(log(rb / rr)))
  for (r in c(.50, .55, .65)) {
    keep <- abs(s$reliability - r) < 1e-6
    blue <- rowSums(keep & s$colour == "blue", na.rm = TRUE)
    red  <- rowSums(keep & s$colour == "red", na.rm = TRUE)
    d[[paste0("x", as.integer(r * 100))]] <- blue - red
  }
  d
}

prepare_group <- function(info) {
  d <- read_csv(info$data_file, show_col_types = FALSE, progress = FALSE)
  id_order <- unique(d$ParticipantPrivateID)
  d <- d %>% mutate(
    subject_index = match(ParticipantPrivateID, id_order),
    trial = as.integer(TrialNumber),
    order = as.integer(.data[[info$order_col]]),
    observed_choice = if_else(as.integer(Response) == order, 1L, 2L),
    correct_category = if_else(as.integer(CorrectResponse) == order, 1L, 2L),
    prior = normalise_prior(.data[[info$prior_col]])
  ) %>% arrange(subject_index, trial)
  d <- add_evidence(d) %>% mutate(
    observed_prior_choice = as.integer(observed_choice == prior),
    prior_colour = if_else(prior == 1, "Blue prior", "Red prior"),
    x50_prior = if_else(prior == 1, x50, -x50),
    x55_prior = if_else(prior == 1, x55, -x55),
    x65_prior = if_else(prior == 1, x65, -x65)
  )
  pred_file <- file.path(prediction_root, paste0("y_pred_", info$safe, "_200_draws.csv"))
  manifest_file <- file.path(prediction_root, paste0("manifest_", info$safe, ".csv"))
  stopifnot(file.exists(pred_file), file.exists(manifest_file))
  manifest <- read_csv(manifest_file, show_col_types = FALSE)
  pred <- as.matrix(read_csv(pred_file, show_col_types = FALSE, progress = FALSE))
  key_d <- paste(d$subject_index, d$trial)
  key_m <- paste(manifest$subject_index, manifest$trial)
  ord <- match(key_d, key_m)
  if (anyNA(ord)) stop("Prediction manifest failed to match for ", info$group)
  pred <- pred[, ord, drop = FALSE]
  if (!all(pred %in% c(1, 2))) stop("Predictions are not coded 1/2 for ", info$group)
  list(info = info, data = d, pred = pred)
}

objects <- lapply(seq_len(nrow(groups)), function(i) prepare_group(groups[i, ]))

# 1. ACCURACY -----------------------------------------------------------
observed_accuracy <- map_dfr(objects, function(o) o$data %>%
                               group_by(subject_index) %>% summarise(value = mean(observed_choice == correct_category), .groups="drop") %>%
                               mutate(group = o$info$group))

pred_accuracy <- map_dfr(objects, function(o) {
  correct <- o$data$correct_category
  vals <- vapply(seq_len(nrow(o$pred)), function(k) mean(o$pred[k, ] == correct), numeric(1))
  tibble(group=o$info$group, draw=seq_along(vals), value=vals)
})

acc_intervals <- pred_accuracy %>% group_by(group) %>% summarise(
  mean=mean(value), lo=quantile(value,.025), hi=quantile(value,.975), .groups="drop")

p_accuracy <- ggplot(observed_accuracy, aes(group, 100*value, colour=group)) +
  geom_jitter(width=.13, height=0, alpha=.75, size=2) +
  geom_errorbar(
    data=acc_intervals,
    aes(x=group, ymin=100*lo, ymax=100*hi, colour=group),
    inherit.aes=FALSE,
    width=.16,
    linewidth=1.1
  ) +
  geom_point(
    data=acc_intervals,
    aes(x=group, y=100*mean, colour=group),
    inherit.aes=FALSE,
    shape=21,
    fill="white",
    size=3.5,
    stroke=1.1
  ) +
  geom_hline(yintercept=50, linetype="dashed", colour="grey50") +
  scale_colour_manual(values=group_cols) + coord_cartesian(ylim=c(40,80)) +
  labs(x=NULL,y="Accuracy (%)",title="Observed accuracy and posterior-predicted group means",
       subtitle="Dots are observed participants; open points and bars are predicted means and 95% intervals") +
  theme_classic(base_size=12) + theme(legend.position="none",axis.text.x=element_text(angle=25,hjust=1))

# 2. BASE-RATE-ALIGNED CHOICE OVER TRIALS -------------------------------
roll_mean <- function(x, width=50) as.numeric(stats::filter(x, rep(1/width,width), sides=1))

observed_time <- map_dfr(objects, function(o) o$data %>% group_by(subject_index,prior_colour) %>%
                           arrange(trial,.by_group=TRUE) %>% mutate(value=roll_mean(observed_prior_choice,ROLLING_WINDOW)) %>%
                           filter(!is.na(value)) %>% ungroup() %>% group_by(prior_colour,trial) %>%
                           summarise(mean=mean(value),.groups="drop") %>%
                           mutate(group=o$info$group))

pred_time <- map_dfr(objects, function(o) {
  d <- o$data; keep_trials <- seq(50,240,by=10)
  map_dfr(seq_len(nrow(o$pred)), function(k) {
    z <- d %>% mutate(pc=as.integer(o$pred[k, ] == prior)) %>% group_by(subject_index,prior_colour) %>%
      arrange(trial,.by_group=TRUE) %>% mutate(value=roll_mean(pc,ROLLING_WINDOW)) %>% ungroup() %>%
      filter(trial %in% keep_trials) %>% group_by(prior_colour,trial) %>%
      summarise(value=mean(value,na.rm=TRUE),.groups="drop")
    z$draw <- k; z
  }) %>% mutate(group=o$info$group)
})

pred_time_band <- pred_time %>% group_by(group,prior_colour,trial) %>% summarise(
  mean=mean(value),lo=quantile(value,.025),hi=quantile(value,.975),.groups="drop")

p_time <- ggplot() +
  geom_ribbon(data=pred_time_band,aes(trial,ymin=lo,ymax=hi,fill=prior_colour),alpha=.18) +
  geom_line(data=pred_time_band,aes(trial,mean,colour=prior_colour),linetype="dashed",linewidth=.9) +
  geom_line(data=observed_time,aes(trial,mean,colour=prior_colour),linewidth=1) +
  facet_wrap(~factor(group,levels=group_levels),nrow=1) +
  geom_hline(yintercept=.5,linetype="dashed",colour="grey55") +
  scale_colour_manual(values=c("Blue prior"="#0072B2","Red prior"="#D94F16")) +
  scale_fill_manual(values=c("Blue prior"="#0072B2","Red prior"="#D94F16")) +
  coord_cartesian(ylim=c(.35,.75)) +
  labs(x="Trial",y="Proportion choosing true base-rate colour",
       title="Base-rate-aligned choices over trials",
       subtitle="Solid = observed; dashed and shaded = predicted mean and 95% interval") +
  theme_classic(base_size=11) + theme(legend.position="bottom")

# 3. RELIABILITY-SPECIFIC EVIDENCE CURVES -------------------------------

evidence_one <- function(d, choice, group_name, draw=NA_integer_) {
  map_dfr(c("50","55","65"), function(rr) {
    xcol <- paste0("x",rr,"_prior")
    tibble(subject=d$subject_index,x=d[[xcol]],y=as.integer(choice==d$prior)) %>%
      group_by(subject,x) %>% summarise(n=n(),p=mean(y),.groups="drop") %>%
      filter(n>=MIN_TRIALS_PER_BIN) %>% group_by(x) %>%
      summarise(value=mean(p),se=sd(p)/sqrt(n()),.groups="drop") %>%
      mutate(reliability=paste0(rr,"%"))
  }) %>% mutate(group=group_name,draw=draw)
}

observed_ev <- map_dfr(objects,function(o) evidence_one(o$data,o$data$observed_choice,o$info$group))
pred_ev <- map_dfr(objects,function(o) map_dfr(seq_len(nrow(o$pred)),
                                               function(k) evidence_one(o$data,o$pred[k,],o$info$group,k)))
pred_ev_band <- pred_ev %>% group_by(group,reliability,x) %>% summarise(
  mean=mean(value),lo=quantile(value,.025),hi=quantile(value,.975),.groups="drop")

observed_ev <- observed_ev %>%
  mutate(
    plot_row="Observed data (solid)",
    group_panel=factor(group,levels=group_levels)
  )

pred_ev_band <- pred_ev_band %>%
  mutate(
    plot_row="Model prediction (dashed)",
    group_panel=factor(group,levels=group_levels)
  )

p_evidence <- ggplot() +
  geom_ribbon(
    data=pred_ev_band,
    aes(x,ymin=lo,ymax=hi,fill=reliability,group=reliability),
    alpha=.18
  ) +
  geom_line(
    data=pred_ev_band,
    aes(x,mean,colour=reliability,group=reliability),
    linetype="dashed",linewidth=1
  ) +
  geom_errorbar(
    data=observed_ev,
    aes(x,ymin=value-se,ymax=value+se,colour=reliability),
    width=.08
  ) +
  geom_line(
    data=observed_ev,
    aes(x,value,colour=reliability,group=reliability),
    linewidth=1
  ) +
  geom_point(
    data=observed_ev,
    aes(x,value,colour=reliability),
    fill="white",shape=21,size=2.3
  ) +
  facet_grid(
    rows=vars(plot_row),
    cols=vars(group_panel)
  ) +
  geom_vline(xintercept=0,linetype="dashed",colour="grey55") +
  geom_hline(yintercept=.5,linetype="dashed",colour="grey55") +
  scale_colour_manual(values=rel_cols) + scale_fill_manual(values=rel_cols) +
  coord_cartesian(ylim=c(0,1)) +
  labs(x="Net samples favouring true base-rate colour",
       y="Proportion choosing true base-rate colour",
       title="Reliability-specific evidence integration",
       subtitle="Top: observed participant data. Bottom: model predictions and 95% intervals") +
  theme_classic(base_size=11) +
  theme(
    legend.position="bottom",
    strip.text=element_text(face="bold"),
    strip.text.y=element_text(angle=90)
  )

# 4. OVERALL RAW PSE ----------------------------------------------------
# One probit psychometric fit per participant across all 240 trials.
# This is the raw PSE: Blue-prior shifts can be below 0.5 and Red-prior
# shifts can be above 0.5. It is not recoded into positive aligned bias.

fit_raw_pse <- function(x, blue_choice) {
  ok <- is.finite(x) & is.finite(blue_choice)
  x <- x[ok]
  blue_choice <- blue_choice[ok]
  if (length(x) < 12 || length(unique(blue_choice)) < 2 ||
      length(unique(x)) < 2) return(NA_real_)
  fit <- tryCatch(
    suppressWarnings(glm(blue_choice ~ x, family=binomial(link="probit"))),
    error=function(e) NULL
  )
  if (is.null(fit) || length(coef(fit)) < 2 ||
      any(!is.finite(coef(fit))) || abs(coef(fit)[2]) < 1e-10) return(NA_real_)
  pmin(pmax(-coef(fit)[1] / coef(fit)[2], 0), 1)
}

observed_pse <- map_dfr(objects, function(o) {
  split_rows <- split(seq_len(nrow(o$data)), o$data$subject_index)
  map_dfr(names(split_rows), function(ss) {
    ii <- split_rows[[ss]]
    tibble(
      subject_index=as.integer(ss),
      prior_colour=first(o$data$prior_colour[ii]),
      raw_pse=fit_raw_pse(
        o$data$neutral_blue_probability[ii],
        as.integer(o$data$observed_choice[ii] == 1)
      )
    )
  }) %>% mutate(group=o$info$group)
})

predicted_pse <- map_dfr(objects, function(o) {
  split_rows <- split(seq_len(nrow(o$data)), o$data$subject_index)
  map_dfr(seq_len(nrow(o$pred)), function(k) {
    map_dfr(names(split_rows), function(ss) {
      ii <- split_rows[[ss]]
      tibble(
        draw=k,
        subject_index=as.integer(ss),
        prior_colour=first(o$data$prior_colour[ii]),
        raw_pse=fit_raw_pse(
          o$data$neutral_blue_probability[ii],
          as.integer(o$pred[k,ii] == 1)
        )
      )
    })
  }) %>% mutate(group=o$info$group)
})

predicted_pse_group <- predicted_pse %>%
  group_by(group,prior_colour,draw) %>%
  summarise(raw_pse=mean(raw_pse,na.rm=TRUE),.groups="drop")

predicted_pse_interval <- predicted_pse_group %>%
  group_by(group,prior_colour) %>%
  summarise(
    mean=mean(raw_pse,na.rm=TRUE),
    lo=quantile(raw_pse,.025,na.rm=TRUE),
    hi=quantile(raw_pse,.975,na.rm=TRUE),
    .groups="drop"
  )

observed_pse <- observed_pse %>%
  mutate(
    plot_row="Observed participants",
    group_panel=factor(group,levels=group_levels)
  )

observed_pse_interval <- observed_pse %>%
  group_by(group,group_panel,plot_row,prior_colour) %>%
  summarise(
    n=sum(is.finite(raw_pse)),
    mean=mean(raw_pse,na.rm=TRUE),
    se=sd(raw_pse,na.rm=TRUE)/sqrt(n),
    lo=mean-qt(.975,pmax(n-1,1))*se,
    hi=mean+qt(.975,pmax(n-1,1))*se,
    .groups="drop"
  )

predicted_pse_interval <- predicted_pse_interval %>%
  mutate(
    plot_row="Model prediction",
    group_panel=factor(group,levels=group_levels)
  )

p_raw_pse <- ggplot() +
  geom_jitter(
    data=observed_pse,
    aes(prior_colour,raw_pse,colour=prior_colour),
    width=.12,height=0,alpha=.7,size=2
  ) +
  geom_errorbar(
    data=observed_pse_interval,
    aes(x=prior_colour,ymin=lo,ymax=hi,colour=prior_colour),
    inherit.aes=FALSE,
    width=.14,linewidth=1
  ) +
  geom_point(
    data=observed_pse_interval,
    aes(x=prior_colour,y=mean,colour=prior_colour),
    inherit.aes=FALSE,
    shape=18,size=3.2
  ) +
  geom_errorbar(
    data=predicted_pse_interval,
    aes(x=prior_colour,ymin=lo,ymax=hi,colour=prior_colour),
    inherit.aes=FALSE,
    width=.14,linewidth=1
  ) +
  geom_point(
    data=predicted_pse_interval,
    aes(x=prior_colour,y=mean,colour=prior_colour),
    inherit.aes=FALSE,
    shape=21,fill="white",size=3.2,stroke=1
  ) +
  facet_grid(rows=vars(plot_row),cols=vars(group_panel)) +
  geom_hline(yintercept=.5,linetype="dashed",colour="grey50") +
  scale_colour_manual(values=c("Blue prior"="#0072B2","Red prior"="#D94F16")) +
  coord_cartesian(ylim=c(.2,.8)) +
  labs(
    x=NULL,y="Raw PSE",
    title="Observed and posterior-predicted raw PSE",
    subtitle="Top: observed means and 95% confidence intervals. Bottom: model means and 95% posterior-predictive intervals"
  ) +
  theme_classic(base_size=12) +
  theme(
    axis.text.x=element_text(angle=25,hjust=1),
    legend.position="bottom",
    strip.text=element_text(face="bold"),
    strip.text.y=element_text(angle=90)
  )

ggsave(file.path(output_root,"PPC_1_accuracy.png"),p_accuracy,width=11,height=6,dpi=300)
ggsave(file.path(output_root,"PPC_2_base_rate_choices_over_trials.png"),p_time,width=15,height=5.5,dpi=300)
ggsave(file.path(output_root,"PPC_3_evidence_integration.png"),p_evidence,width=15,height=9,dpi=300)
ggsave(file.path(output_root,"PPC_4_raw_PSE.png"),p_raw_pse,width=15,height=8,dpi=300)
ggsave(file.path(output_root,"PPC_1_accuracy.pdf"),p_accuracy,width=11,height=6)
ggsave(file.path(output_root,"PPC_2_base_rate_choices_over_trials.pdf"),p_time,width=15,height=5.5)
ggsave(file.path(output_root,"PPC_3_evidence_integration.pdf"),p_evidence,width=15,height=9)
ggsave(file.path(output_root,"PPC_4_raw_PSE.pdf"),p_raw_pse,width=15,height=8)

print(p_accuracy); print(p_time); print(p_evidence); print(p_raw_pse)
cat("\nCompleted. Plots saved in:\n",output_root,"\n")
