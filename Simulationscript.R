rm(list = ls(all = TRUE))

library(tidyverse)

##################################################
## SETTINGS
##################################################

project_folder <- "/Users/bty615/Documents/GitHub/reliable_info_bias"

parameter_path <- file.path(
  project_folder,
  "stan/results/summary/exp11_unaware",
  "param_individual_leta_2channels_truthful_exp13.csv"
)

output_folder <- file.path(
  project_folder,
  "stan/results/simulations/vb_only_truthful_exp13"
)

starting_values <- c(0.20, 0.35, 0.50, 0.65, 0.80)

n_trials <- 240L
blue_base_rate <- 0.65

# Initial blue + red counts.
initial_count <- 2

set.seed(4321)

dir.create(
  output_folder,
  recursive = TRUE,
  showWarnings = FALSE
)

##################################################
## LOAD FITTED DELTA ESTIMATES
##################################################

if (!file.exists(parameter_path)) {
  stop("Parameter file not found: ", parameter_path)
}

parameter_table <- read_csv(
  parameter_path,
  show_col_types = FALSE
)

if (!all(c("participant", "parameter", "mean") %in%
         names(parameter_table))) {
  stop("CSV must contain participant, parameter and mean columns.")
}

delta_table <- parameter_table %>%
  filter(parameter == "delta") %>%
  transmute(
    participant = participant,
    delta = mean
  ) %>%
  arrange(participant)

if (
  nrow(delta_table) == 0L ||
  anyNA(delta_table$participant) ||
  anyDuplicated(delta_table$participant) > 0L ||
  any(!is.finite(delta_table$delta)) ||
  any(delta_table$delta <= 0 | delta_table$delta >= 1)
) {
  stop("Expected one fitted delta between 0 and 1 per participant.")
}

N <- nrow(delta_table)

cat("\nParticipants:", N, "\n")
cat("Trials per simulation:", n_trials, "\n")
cat("Fitted delta range:", range(delta_table$delta), "\n")

##################################################
## GENERATE FEEDBACK
##
## 1 = blue correct
## 0 = red correct
##
## Each participant receives a randomly ordered
## sequence containing exactly 65% blue outcomes.
## Their sequence is reused across starting values.
##################################################

n_blue <- as.integer(round(n_trials * blue_base_rate))
n_red <- n_trials - n_blue

feedback_sequences <- lapply(
  seq_len(N),
  function(i) {
    sample(
      c(rep(1L, n_blue), rep(0L, n_red)),
      size = n_trials,
      replace = FALSE
    )
  }
)

feedback_table <- map_dfr(
  seq_len(N),
  function(i) {
    tibble(
      participant = delta_table$participant[i],
      trial = seq_len(n_trials),
      feedback = feedback_sequences[[i]]
    )
  }
)

##################################################
## SIMULATE VB
##
## Uses the exact feedback update from your model.
## Records the initial belief and the belief after
## every feedback outcome.
##################################################

simulate_vb <- function(i, starting_Vb) {
  
  delta <- delta_table$delta[i]
  feedback <- feedback_sequences[[i]]
  
  blue_count <- initial_count * starting_Vb
  red_count <- initial_count * (1 - starting_Vb)
  
  vb <- numeric(n_trials + 1L)
  vb[1] <- blue_count / (blue_count + red_count)
  
  for (t in seq_len(n_trials)) {
    
    x <- feedback[t]
    
    blue_count <- delta * (blue_count - 1) + 1 + x
    red_count <- delta * (red_count - 1) + 1 + (1 - x)
    
    vb[t + 1L] <- blue_count / (blue_count + red_count)
  }
  
  tibble(
    participant = delta_table$participant[i],
    delta = delta,
    starting_Vb = starting_Vb,
    feedbacks_seen = 0:n_trials,
    V_b = vb
  )
}

simulation_results <- map_dfr(
  starting_values,
  function(v) {
    map_dfr(
      seq_len(N),
      function(i) simulate_vb(i, v)
    )
  }
)

##################################################
## CHECK RESULTS
##################################################

initial_rows <- simulation_results %>%
  filter(feedbacks_seen == 0L)

stopifnot(
  nrow(simulation_results) ==
    N * length(starting_values) * (n_trials + 1L),
  all(abs(initial_rows$V_b - initial_rows$starting_Vb) < 1e-12),
  all(is.finite(simulation_results$V_b)),
  all(simulation_results$V_b > 0),
  all(simulation_results$V_b < 1)
)

##################################################
## COMPARE WITH STARTING VB = 0.50
##################################################

neutral_results <- simulation_results %>%
  filter(starting_Vb == 0.50) %>%
  select(
    participant,
    feedbacks_seen,
    neutral_V_b = V_b
  )

simulation_results <- simulation_results %>%
  left_join(
    neutral_results,
    by = c("participant", "feedbacks_seen")
  ) %>%
  mutate(
    difference_from_neutral = V_b - neutral_V_b
  )

##################################################
## GROUP SUMMARY
##################################################

group_trajectory <- simulation_results %>%
  group_by(starting_Vb, feedbacks_seen) %>%
  summarise(
    participants = n(),
    mean_V_b = mean(V_b),
    sd_V_b = sd(V_b),
    mean_absolute_difference = mean(abs(difference_from_neutral)),
    .groups = "drop"
  )

##################################################
## SAVE CSVs
##################################################

write_csv(
  delta_table,
  file.path(output_folder, "fitted_delta_values.csv")
)

write_csv(
  feedback_table,
  file.path(output_folder, "generated_feedback.csv")
)

write_csv(
  simulation_results,
  file.path(output_folder, "individual_vb_trajectories.csv")
)

write_csv(
  group_trajectory,
  file.path(output_folder, "group_vb_trajectories.csv")
)

##################################################
## PLOT SETTINGS
##################################################

start_colours <- c(
  "0.2" = "#B2182B",
  "0.35" = "#EF8A62",
  "0.5" = "#666666",
  "0.65" = "#67A9CF",
  "0.8" = "#2166AC"
)

##################################################
## PLOT 1: FULL TRAJECTORY
##################################################

belief_plot <- ggplot(
  group_trajectory,
  aes(
    x = feedbacks_seen,
    y = mean_V_b,
    colour = factor(starting_Vb)
  )
) +
  geom_hline(
    yintercept = blue_base_rate,
    linetype = "dashed",
    colour = "grey50"
  ) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = start_colours) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    x = "Number of feedback outcomes seen",
    y = "Mean Vb: belief that blue is correct",
    colour = "Starting Vb",
    title = "Starting-belief simulation: truthful group",
    subtitle = "Fitted delta; generated feedback; dashed line = 65% base rate"
  ) +
  theme_bw()

##################################################
## PLOT 2: FIRST 30 FEEDBACK OUTCOMES
##################################################

early_plot <- belief_plot +
  coord_cartesian(
    xlim = c(0, 30),
    ylim = c(0, 1)
  ) +
  labs(
    title = "Starting-belief simulation: early trials"
  )

##################################################
## PLOT 3: EFFECT RELATIVE TO NEUTRAL START
##################################################

difference_plot <- ggplot(
  group_trajectory,
  aes(
    x = feedbacks_seen,
    y = mean_absolute_difference,
    colour = factor(starting_Vb)
  )
) +
  geom_line(linewidth = 0.8) +
  scale_colour_manual(values = start_colours) +
  labs(
    x = "Number of feedback outcomes seen",
    y = "Mean absolute Vb difference from neutral start",
    colour = "Starting Vb",
    title = "How long does the starting belief matter?"
  ) +
  theme_bw()

print(belief_plot)
print(early_plot)
print(difference_plot)

ggsave(
  file.path(output_folder, "vb_full_trajectory.png"),
  plot = belief_plot,
  width = 9, height = 5, dpi = 300
)

ggsave(
  file.path(output_folder, "vb_first_30_trials.png"),
  plot = early_plot,
  width = 9, height = 5, dpi = 300
)

ggsave(
  file.path(output_folder, "vb_difference_from_neutral.png"),
  plot = difference_plot,
  width = 9, height = 5, dpi = 300
)

cat("\nSimulation checks passed.\n")
cat("\nResults saved to:\n", output_folder, "\n")





##################################################
## INDIVIDUAL VB TRAJECTORIES
##################################################

individual_plot <- ggplot(
  simulation_results,
  aes(
    x = feedbacks_seen,
    y = V_b,
    colour = factor(starting_Vb),
    group = factor(starting_Vb)
  )
) +
  geom_hline(
    yintercept = blue_base_rate,
    linetype = "dashed",
    colour = "grey60"
  ) +
  geom_line(linewidth = 0.5, alpha = 0.85) +
  facet_wrap(
    ~participant,
    ncol = 4,
    labeller = label_both
  ) +
  scale_colour_manual(values = start_colours) +
  coord_cartesian(ylim = c(0, 1)) +
  labs(
    x = "Number of feedback outcomes seen",
    y = "Vb: belief that blue is correct",
    colour = "Starting Vb",
    title = "Individual simulated belief trajectories",
    subtitle = "Each panel uses that participant's fitted delta"
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )

print(individual_plot)

ggsave(
  filename = file.path(
    output_folder,
    "individual_vb_trajectories.png"
  ),
  plot = individual_plot,
  width = 14,
  height = 3 * ceiling(n_distinct(simulation_results$participant) / 4),
  dpi = 300,
  limitsize = FALSE
)

##################################################
## ZOOM INTO FIRST 30 FEEDBACK OUTCOMES
##################################################

individual_early_plot <- individual_plot +
  coord_cartesian(
    xlim = c(0, 30),
    ylim = c(0, 1)
  ) +
  labs(
    title = "Individual belief trajectories: first 30 feedback outcomes"
  )

print(individual_early_plot)

ggsave(
  filename = file.path(
    output_folder,
    "individual_vb_first_30_trials.png"
  ),
  plot = individual_early_plot,
  width = 14,
  height = 3 * ceiling(n_distinct(simulation_results$participant) / 4),
  dpi = 300,
  limitsize = FALSE
)







rm(list = ls(all = TRUE))

library(tidyverse)

##################################################
## SETTINGS
##################################################

project_folder <- "/Users/bty615/Documents/GitHub/reliable_info_bias"

output_folder <- file.path(
  project_folder,
  "stan/results/simulations/alpha0_beta0_curves"
)

dir.create(
  output_folder,
  recursive = TRUE,
  showWarnings = FALSE
)

# Same range of learned beliefs for every curve.
vb_values <- sort(unique(c(
  seq(0.01, 0.99, length.out = 301),
  0.50,
  0.65
)))

alpha0_values <- c(0.25, 0.50, 1, 2)
beta0_values <- c(-1, -0.50, 0, 0.50, 1)

fixed_beta0 <- 0
fixed_alpha0 <- 1

##################################################
## PRIOR TRANSFORMATION FROM YOUR MODEL
##
## P = alpha0 * logit(Vb) + beta0
##
## plogis(P) expresses this prior contribution
## on a probability scale.
## It does not change the underlying Vb.
##################################################

prior_probability <- function(Vb, alpha0, beta0) {
  
  Vb_safe <- pmin(pmax(Vb, 0.001), 0.999)
  
  plogis(
    alpha0 * qlogis(Vb_safe) + beta0
  )
}

##################################################
## VARY ALPHA0
##################################################

alpha0_data <- expand_grid(
  Vb = vb_values,
  alpha0 = alpha0_values
) %>%
  mutate(
    beta0 = fixed_beta0,
    prior_probability = prior_probability(Vb, alpha0, beta0),
    curve = factor(
      alpha0,
      levels = alpha0_values,
      labels = c("0.25", "0.50", "1.00", "2.00")
    )
  )

##################################################
## VARY BETA0
##################################################

beta0_data <- expand_grid(
  Vb = vb_values,
  beta0 = beta0_values
) %>%
  mutate(
    alpha0 = fixed_alpha0,
    prior_probability = prior_probability(Vb, alpha0, beta0),
    curve = factor(
      beta0,
      levels = beta0_values,
      labels = c("-1.00", "-0.50", "0.00", "0.50", "1.00")
    )
  )

##################################################
## CHECK CALCULATIONS
##################################################

stopifnot(
  all(is.finite(alpha0_data$prior_probability)),
  all(is.finite(beta0_data$prior_probability)),
  
  # Identity transformation: alpha0 = 1, beta0 = 0.
  max(abs(
    prior_probability(vb_values, 1, 0) - vb_values
  )) < 1e-12,
  
  # With beta0 = 0, a neutral belief remains neutral.
  all(abs(
    prior_probability(0.5, alpha0_values, 0) - 0.5
  ) < 1e-12)
)

##################################################
## PLOT 1: ALPHA0
##################################################

alpha0_plot <- ggplot(
  alpha0_data,
  aes(
    x = Vb,
    y = prior_probability,
    colour = curve
  )
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    colour = "grey65"
  ) +
  geom_vline(
    xintercept = 0.5,
    linetype = "dotted",
    colour = "grey70"
  ) +
  geom_hline(
    yintercept = 0.5,
    linetype = "dotted",
    colour = "grey70"
  ) +
  geom_line(linewidth = 1.2) +
  scale_colour_manual(
    values = c(
      "0.25" = "#E69F00",
      "0.50" = "#009E73",
      "1.00" = "#666666",
      "2.00" = "#0072B2"
    )
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25)
  ) +
  coord_equal() +
  labs(
    x = "Learned belief: Vb",
    y = "Prior contribution on a probability scale",
    colour = "alpha0",
    title = "Changing alpha0",
    subtitle = "beta0 fixed at 0",
    caption = "Diagonal = unchanged belief. Below 0.50 favours red; above 0.50 favours blue."
  ) +
  theme_bw(base_size = 14) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 10)
  )

##################################################
## PLOT 2: BETA0
##################################################

beta0_plot <- ggplot(
  beta0_data,
  aes(
    x = Vb,
    y = prior_probability,
    colour = curve
  )
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    colour = "grey65"
  ) +
  geom_vline(
    xintercept = 0.5,
    linetype = "dotted",
    colour = "grey70"
  ) +
  geom_hline(
    yintercept = 0.5,
    linetype = "dotted",
    colour = "grey70"
  ) +
  geom_line(linewidth = 1.2) +
  scale_colour_manual(
    values = c(
      "-1.00" = "#B2182B",
      "-0.50" = "#EF8A62",
      "0.00" = "#666666",
      "0.50" = "#67A9CF",
      "1.00" = "#2166AC"
    )
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25)
  ) +
  coord_equal() +
  labs(
    x = "Learned belief: Vb",
    y = "Prior contribution on a probability scale",
    colour = "beta0",
    title = "Changing beta0",
    subtitle = "alpha0 fixed at 1",
    caption = "Diagonal = unchanged belief. Below 0.50 favours red; above 0.50 favours blue."
  ) +
  theme_bw(base_size = 14) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    plot.caption = element_text(size = 10)
  )

##################################################
## DISPLAY PLOTS
##################################################

print(alpha0_plot)
print(beta0_plot)

##################################################
## SAVE PLOTS
##################################################

ggsave(
  filename = file.path(output_folder, "vary_alpha0.png"),
  plot = alpha0_plot,
  width = 8,
  height = 7,
  dpi = 300
)

ggsave(
  filename = file.path(output_folder, "vary_beta0.png"),
  plot = beta0_plot,
  width = 8,
  height = 7,
  dpi = 300
)

# Two-page PDF containing both plots.
pdf(
  file = file.path(output_folder, "alpha0_beta0_curves.pdf"),
  width = 8,
  height = 7
)

print(alpha0_plot)
print(beta0_plot)

dev.off()

##################################################
## SAVE VALUES
##################################################

write_csv(
  alpha0_data %>% select(-curve),
  file.path(output_folder, "vary_alpha0.csv")
)

write_csv(
  beta0_data %>% select(-curve),
  file.path(output_folder, "vary_beta0.csv")
)

##################################################
## EXAMPLE: HOLD VB AT 0.65
##################################################

cat("\nChanging alpha0 with Vb = 0.65 and beta0 = 0:\n")

print(
  alpha0_data %>%
    filter(Vb == 0.65) %>%
    select(Vb, alpha0, beta0, prior_probability)
)

cat("\nChanging beta0 with Vb = 0.65 and alpha0 = 1:\n")

print(
  beta0_data %>%
    filter(Vb == 0.65) %>%
    select(Vb, alpha0, beta0, prior_probability)
)

cat("\nFiles saved to:\n", output_folder, "\n")