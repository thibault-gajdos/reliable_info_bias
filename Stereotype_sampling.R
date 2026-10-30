# ============================================================
# SHARED 240 ITEMS: PROBABILITY BALANCE + SPATIAL COVERAGE
# Base R only. Run the whole script; choose the ORIGINAL CSV.
# Optional command line: Rscript this_script.R input.csv output_folder
# ============================================================
# HARD RULES:
# 24 items per 5-percentage-point probability range, 25-75% inclusive.
# Adjective <=10; occupation <=10; pair <=4 appearances.
# Each pair appears at most once in each low/high combination.
# LOW = score <50; HIGH = score >=50. Different repeats use different adjectives.
# SOFT SPATIAL GOAL:
# Within each probability range, favour least-used eligible 10x10 score cells.
# This encourages spatial coverage; it does NOT guarantee equal quadrant counts,
# all four combinations for a pair, or every dot size in every location.
# Existing selections are verified and reused, never silently redrawn.

seed <- 20261029L
max_adjective_appearances <- 10L
max_occupation_appearances <- 10L
max_pair_appearances <- 4L
max_sampling_attempts <- 10000L
items_per_range <- 24L
low_high_cutoff <- 50
grid_width <- 10
algorithm_version <- "spatial_cells_v1"
excluded_groups <- c("sixteen year old", "hippy", "environmental activist")
args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 2L) args[2] else file.path(
  path.expand("~"), "Desktop", "stereotype_spatial_25_75_pairmax4_v1")
experiment_file <- file.path(output_dir, "experiment_items_240.csv")
record_file <- file.path(output_dir, "original_selection_record.rds")
probability_breaks <- seq(25, 75, by = 5) / 100
range_labels <- c("25-<30%", "30-<35%", "35-<40%", "40-<45%", "45-<50%",
                  "50-<55%", "55-<60%", "60-<65%", "65-<70%", "70-75%")
region_labels <- c("A low / B low", "A low / B high",
                   "A high / B low", "A high / B high")
n_items <- items_per_range * length(range_labels)
stopifnot(n_items == 240L, 100 %% grid_width == 0,
          low_high_cutoff > 0, low_high_cutoff < 100)
n_axis <- as.integer(100 / grid_width)
n_cells <- n_axis^2L
pick_one <- function(x) x[sample.int(length(x), 1L)]

get_range <- function(p) {
  # Reject outside limits BEFORE rounding; rounding only stabilises bin edges.
  valid <- is.finite(p) & p >= .25 & p <= .75
  bin <- findInterval(round(p, 12), probability_breaks, rightmost.closed = TRUE)
  bin[!valid] <- NA_integer_
  factor(bin, levels = seq_along(range_labels), labels = range_labels)
}
score_cell <- function(x) pmin(n_axis, floor(x / grid_width) + 1L)

# ============================================================
# 1. READ, CLEAN AND IDENTIFY ITEMS
# ============================================================
message("Choose the ORIGINAL base_rate_database CSV.")
source_file <- if (length(args)) args[1] else file.choose()
raw <- read.csv(source_file, stringsAsFactors = FALSE, check.names = FALSE)
required <- c("group1", "group2", "adjective", "score1_gpt4", "score2_gpt4")
if (!all(required %in% names(raw))) stop("Missing columns: ",
                                         paste(setdiff(required, names(raw)), collapse = ", "))
raw$database_row <- seq_len(nrow(raw))
for (column in c("group1", "group2", "adjective"))
  raw[[column]] <- trimws(as.character(raw[[column]]))
for (column in c("score1_gpt4", "score2_gpt4"))
  raw[[column]] <- suppressWarnings(as.numeric(raw[[column]]))
valid <- with(raw, !is.na(group1) & nzchar(group1) & !is.na(group2) & nzchar(group2) &
                !is.na(adjective) & nzchar(adjective) & group1 != group2 &
                is.finite(score1_gpt4) & is.finite(score2_gpt4) &
                score1_gpt4 > 0 & score1_gpt4 <= 100 & score2_gpt4 > 0 & score2_gpt4 <= 100)
valid[is.na(valid)] <- FALSE
excluded <- tolower(raw$group1) %in% excluded_groups |
  tolower(raw$group2) %in% excluded_groups
database <- raw[valid & !excluded, ]
if (!nrow(database)) stop("No eligible source rows.")
occupation_order <- sort(unique(c(database$group1, database$group2)), method = "radix")
first_is_A <- match(database$group1, occupation_order) <
  match(database$group2, occupation_order)
database$occupation_A <- ifelse(first_is_A, database$group1, database$group2)
database$occupation_B <- ifelse(first_is_A, database$group2, database$group1)
database$score_A <- ifelse(first_is_A, database$score1_gpt4, database$score2_gpt4)
database$score_B <- ifelse(first_is_A, database$score2_gpt4, database$score1_gpt4)
database$occupation_pair <- paste(database$occupation_A, database$occupation_B, sep = " / ")
item_columns <- c("occupation_A", "occupation_B", "adjective")
if (anyDuplicated(database[item_columns])) stop("Duplicate pair-adjective items in source.")
database$population_base_rate_A <- .50
database$population_base_rate_B <- .50
database$probability_A <- with(database, score_A / (score_A + score_B))
database$probability_B <- 1 - database$probability_A
database$probability_favoured <- pmax(database$probability_A, database$probability_B)
database$typicality_difference <- with(database, abs(score_A - score_B))
database$stereotype_strength <- with(database, abs(log(score_A / score_B)))
database$probability_range <- get_range(database$probability_A)
database$typicality_region <- paste(
  ifelse(database$score_A < low_high_cutoff, "A low", "A high"),
  ifelse(database$score_B < low_high_cutoff, "B low", "B high"), sep = " / ")
database$grid_A <- score_cell(database$score_A)
database$grid_B <- score_cell(database$score_B)
database$grid_cell <- database$grid_A + n_axis * (database$grid_B - 1L)
pool <- database[!is.na(database$probability_range), ]
available_counts <- table(pool$probability_range)
if (any(available_counts < items_per_range)) {
  print(available_counts); stop("A probability range has fewer than 24 items.")
}
filter_summary <- data.frame(stage = c("Original database", "Invalid rows excluded",
                                       "Valid excluded labels", "Eligible inventory", "Outside 25-75%", "Sampling pool"),
                             rows = c(nrow(raw), sum(!valid), sum(valid & excluded), nrow(database),
                                      sum(is.na(database$probability_range)), nrow(pool)))

# Stable integer identities; do not rely on concatenated names for constraints.
oa <- match(pool$occupation_A, occupation_order)
ob <- match(pool$occupation_B, occupation_order)
adj <- match(pool$adjective, unique(pool$adjective))
pair_key <- paste(oa, ob, sep = ":")
pair <- match(pair_key, unique(pair_key))
region <- match(pool$typicality_region, region_labels)
pair_region <- (pair - 1L) * 4L + region
range_id <- as.integer(pool$probability_range)
cell_range <- pool$grid_cell + n_cells * (range_id - 1L)
range_indices <- lapply(seq_along(range_labels), function(k) which(range_id == k))
capacity <- function(ids, cap) sum(pmin(table(ids), cap))
if (capacity(adj, max_adjective_appearances) < n_items)
  stop("Insufficient adjective capacity.")
if (capacity(c(oa, ob), max_occupation_appearances) < 2L * n_items)
  stop("Insufficient occupation capacity: need 480 appearances (at least 48 occupations).")
pair_region_capacity <- table(pair[!duplicated(pair_region)])
if (sum(pmin(pair_region_capacity, max_pair_appearances)) < n_items)
  stop("Insufficient distinct pair-region capacity under the pair cap.")
for (ii in range_indices) {
  if (length(unique(pair_region[ii])) < items_per_range ||
      capacity(adj[ii], max_adjective_appearances) < items_per_range ||
      capacity(c(oa[ii], ob[ii]), max_occupation_appearances) < 2L * items_per_range)
    stop("A range fails a necessary capacity check.")
}

# ============================================================
# 2. SELECT ONCE OR REUSE, KEEPING THE ORIGINAL DRAW RECORD
# ============================================================
export_columns <- c("item_id", "database_row", "occupation_pair", "occupation_A",
                    "occupation_B", "adjective", "score_A", "score_B", "typicality_difference",
                    "stereotype_strength", "population_base_rate_A", "population_base_rate_B",
                    "probability_A", "probability_B", "probability_favoured", "probability_range",
                    "typicality_region", "grid_A", "grid_B", "grid_cell")
settings <- list(algorithm = algorithm_version, seed = seed, items_per_range = items_per_range,
                 adjective_cap = max_adjective_appearances, occupation_cap = max_occupation_appearances,
                 pair_cap = max_pair_appearances, cutoff = low_high_cutoff, grid_width = grid_width,
                 excluded_groups = excluded_groups, breaks = probability_breaks,
                 max_sampling_attempts = max_sampling_attempts)
source_md5 <- unname(tools::md5sum(source_file))
reuse_existing <- file.exists(experiment_file)
if (reuse_existing) {
  if (!file.exists(record_file)) stop("Missing original selection record; use a new output folder.")
  record <- readRDS(record_file)
  if (!identical(record$settings, settings) || !identical(record$source_md5, source_md5))
    stop("Source or settings changed. Use a NEW output folder; existing selection preserved.")
  selected_items <- read.csv(experiment_file, stringsAsFactors = FALSE, check.names = FALSE)
  if (!all(export_columns %in% names(selected_items))) stop("Saved CSV has missing columns.")
  message("Reusing and verifying the existing selection; no new draw.")
} else {
  if (file.exists(record_file)) stop("Draw record exists without CSV. Use a new output folder.")
  RNGkind("Mersenne-Twister", "Inversion", "Rejection")
  set.seed(seed)
  slots <- rep(seq_along(range_labels), each = items_per_range)
  accepted <- FALSE
  for (sampling_attempt in seq_len(max_sampling_attempts)) {
    adj_use <- integer(max(adj)); occ_use <- integer(length(occupation_order))
    pair_use <- integer(max(pair)); region_used <- logical(max(pair_region))
    cell_use <- integer(n_cells * length(range_labels))
    used <- logical(nrow(pool)); selected_indices <- integer(n_items)
    slot_order <- slots[sample.int(length(slots))]
    completed <- TRUE
    for (position in seq_along(slot_order)) {
      ii <- range_indices[[slot_order[position]]]
      eligible <- ii[!used[ii] & !region_used[pair_region[ii]] &
                       adj_use[adj[ii]] < max_adjective_appearances &
                       occ_use[oa[ii]] < max_occupation_appearances &
                       occ_use[ob[ii]] < max_occupation_appearances &
                       pair_use[pair[ii]] < max_pair_appearances]
      if (!length(eligible)) { completed <- FALSE; break }
      # First draw a CELL uniformly among least-used eligible cells in this range.
      # Then draw an ITEM uniformly within that cell. Dense cells get no extra votes.
      cells <- unique(cell_range[eligible])
      cells <- cells[cell_use[cells] == min(cell_use[cells])]
      chosen_cell <- pick_one(cells)
      chosen <- pick_one(eligible[cell_range[eligible] == chosen_cell])
      selected_indices[position] <- chosen
      used[chosen] <- TRUE
      region_used[pair_region[chosen]] <- TRUE
      adj_use[adj[chosen]] <- adj_use[adj[chosen]] + 1L
      occ_use[oa[chosen]] <- occ_use[oa[chosen]] + 1L
      occ_use[ob[chosen]] <- occ_use[ob[chosen]] + 1L
      pair_use[pair[chosen]] <- pair_use[pair[chosen]] + 1L
      cell_use[chosen_cell] <- cell_use[chosen_cell] + 1L
    }
    if (completed) { accepted <- TRUE; break }
    if (sampling_attempt %% 100L == 0L) message("Attempts: ", sampling_attempt)
  }
  if (!accepted) stop("No valid set found within attempt limit. Nothing saved. ",
                      "This does not prove infeasibility; no constraints have been relaxed.")
  selected_items <- pool[selected_indices, ]
  selected_items <- selected_items[order(selected_items$probability_A,
                                         selected_items$database_row), ]
  selected_items$item_id <- sprintf("S%03d", seq_len(nrow(selected_items)))
  selected_items <- selected_items[export_columns]
  rownames(selected_items) <- NULL
  record <- list(settings = settings, source_md5 = source_md5,
                 accepted_attempt = sampling_attempt, R_version = R.version.string,
                 RNG = RNGkind(), source_rows = selected_items$database_row)
}

# ============================================================
# 3. VERIFY FRESH AND REUSED SELECTIONS BEFORE WRITING ANYTHING
# ============================================================
stopifnot(nrow(selected_items) == n_items, !anyNA(selected_items),
          !anyDuplicated(selected_items$item_id), all(nzchar(selected_items$item_id)),
          !anyDuplicated(selected_items$database_row), !anyDuplicated(selected_items[item_columns]))
source_index <- match(selected_items$database_row, database$database_row)
if (anyNA(source_index)) stop("Selected rows absent from source.")
source_match <- database[source_index, ]
for (column in setdiff(export_columns, "item_id")) {
  a <- selected_items[[column]]; b <- source_match[[column]]
  correct <- if (is.numeric(b)) is.numeric(a) && all(is.finite(a)) &&
    all(abs(a - b) < 1e-10) else all(as.character(a) == as.character(b))
  if (!isTRUE(correct)) stop("Selection/source mismatch: ", column)
}
if (!identical(as.integer(selected_items$database_row), as.integer(record$source_rows)))
  stop("CSV differs from original draw record.")
selected_items$probability_range <- get_range(selected_items$probability_A)
selected_counts <- table(selected_items$probability_range)
selected_pairs <- selected_items[c("occupation_A", "occupation_B")]
stopifnot(!anyNA(selected_items$probability_range), all(selected_counts == items_per_range),
          max(table(selected_items$adjective)) <= max_adjective_appearances,
          max(table(c(selected_items$occupation_A, selected_items$occupation_B))) <= max_occupation_appearances,
          !anyDuplicated(selected_items[c("occupation_A", "occupation_B", "typicality_region")]))
# Pair identity from two separate fields, not the display label.
selected_pair_ids <- match(paste(match(selected_items$occupation_A, occupation_order),
                                 match(selected_items$occupation_B, occupation_order), sep = ":"), unique(pair_key))
stopifnot(max(table(selected_pair_ids)) <= max_pair_appearances)

# ============================================================
# 4. DIAGNOSTICS: AVAILABILITY VERSUS ACHIEVED COVERAGE
# ============================================================
selection_summary <- data.frame(probability_range = range_labels,
                                available_items = as.integer(available_counts), selected_items = as.integer(selected_counts))
probability_summary <- do.call(rbind, lapply(range_labels, function(label) {
  p <- selected_items$probability_A[selected_items$probability_range == label]
  data.frame(probability_range = label, n = length(p), minimum = min(p),
             median = median(p), maximum = max(p))
}))
spatial_counts <- function(d) table(
  grid_cell = factor(d$grid_cell, levels = seq_len(n_cells)),
  probability_range = factor(d$probability_range, levels = range_labels))
available_spatial <- spatial_counts(pool)
selected_spatial <- spatial_counts(selected_items)
spatial_summary <- expand.grid(grid_cell = seq_len(n_cells),
                               probability_range = range_labels, stringsAsFactors = FALSE)
spatial_summary$grid_A <- (spatial_summary$grid_cell - 1L) %% n_axis + 1L
spatial_summary$grid_B <- (spatial_summary$grid_cell - 1L) %/% n_axis + 1L
spatial_summary$score_A_lower <- (spatial_summary$grid_A - 1L) * grid_width
spatial_summary$score_B_lower <- (spatial_summary$grid_B - 1L) * grid_width
spatial_summary$available_items <- as.integer(available_spatial)
spatial_summary$selected_items <- as.integer(selected_spatial)
coverage_summary <- data.frame(probability_range = range_labels,
                               available_cells = colSums(available_spatial > 0),
                               selected_cells = colSums(selected_spatial > 0), row.names = NULL)
region_table <- function(d) table(
  region = factor(d$typicality_region, levels = region_labels),
  probability_range = factor(d$probability_range, levels = range_labels))
pair_summary <- do.call(rbind, lapply(sort(unique(selected_pair_ids)), function(k) {
  z <- selected_items[selected_pair_ids == k, ]
  data.frame(occupation_A = z$occupation_A[1], occupation_B = z$occupation_B[1],
             appearances = nrow(z), regions_present = length(unique(z$typicality_region)),
             both_low = sum(z$typicality_region == region_labels[1]),
             A_low_B_high = sum(z$typicality_region == region_labels[2]),
             A_high_B_low = sum(z$typicality_region == region_labels[3]),
             both_high = sum(z$typicality_region == region_labels[4]))
}))
pair_summary <- pair_summary[order(-pair_summary$appearances), ]
occupation_details <- rbind(
  data.frame(item_id = selected_items$item_id, occupation = selected_items$occupation_A,
             other_occupation = selected_items$occupation_B, adjective = selected_items$adjective,
             probability = selected_items$probability_A),
  data.frame(item_id = selected_items$item_id, occupation = selected_items$occupation_B,
             other_occupation = selected_items$occupation_A, adjective = selected_items$adjective,
             probability = selected_items$probability_B))
occupation_details$probability_range <- get_range(occupation_details$probability)
occupation_coverage <- table(occupation = factor(occupation_details$occupation,
                                                 levels = occupation_order), probability_range = occupation_details$probability_range)
occupation_summary <- data.frame(occupation = occupation_order,
                                 appearances = as.integer(rowSums(occupation_coverage)),
                                 ranges_present = as.integer(rowSums(occupation_coverage > 0)))
occupation_summary$minimum_probability <- vapply(occupation_order, function(x) {
  p <- occupation_details$probability[occupation_details$occupation == x]
  if (length(p)) min(p) else NA_real_ }, numeric(1))
occupation_summary$maximum_probability <- vapply(occupation_order, function(x) {
  p <- occupation_details$probability[occupation_details$occupation == x]
  if (length(p)) max(p) else NA_real_ }, numeric(1))
occupation_summary <- occupation_summary[order(-occupation_summary$appearances), ]
adjective_counts <- sort(table(factor(selected_items$adjective,
                                      levels = sort(unique(database$adjective)))), decreasing = TRUE)
adjective_summary <- data.frame(adjective = names(adjective_counts),
                                appearances = as.integer(adjective_counts))
stopifnot(sum(occupation_summary$appearances) == 480)

# Detailed review tables are derived AFTER selection. They never change the draw.
# Keep the experiment CSV schema stable so existing saved sets can still be reused.
chosen_item_details <- selected_items
chosen_item_details$source_group1 <- source_match$group1
chosen_item_details$source_group2 <- source_match$group2
chosen_item_details$source_score1_gpt4 <- source_match$score1_gpt4
chosen_item_details$source_score2_gpt4 <- source_match$score2_gpt4
chosen_item_details$typicality_A_0_1 <- selected_items$score_A / 100
chosen_item_details$typicality_B_0_1 <- selected_items$score_B / 100
chosen_item_details$probability_A_percent <- 100 * selected_items$probability_A
chosen_item_details$probability_B_percent <- 100 * selected_items$probability_B
chosen_item_details$probability_favoured_percent <- 100 * selected_items$probability_favoured
chosen_item_details$score_ratio_A_over_B <- with(selected_items, score_A / score_B)
chosen_item_details$odds_A_over_B <- with(selected_items, probability_A / probability_B)
chosen_item_details$signed_log_odds_A <- with(selected_items, log(score_A / score_B))
chosen_item_details$signed_log_odds_B <- -chosen_item_details$signed_log_odds_A
chosen_item_details$signed_typicality_difference_A_minus_B <- with(selected_items, score_A - score_B)
chosen_item_details$mean_typicality <- with(selected_items, (score_A + score_B) / 2)
chosen_item_details$favoured_occupation <- with(selected_items,
                                                ifelse(score_A == score_B, "Neither (equal scores)",
                                                       ifelse(score_A > score_B, occupation_A, occupation_B)))
chosen_item_details$typicality_level_A <- ifelse(selected_items$score_A < low_high_cutoff, "Low", "High")
chosen_item_details$typicality_level_B <- ifelse(selected_items$score_B < low_high_cutoff, "Low", "High")
chosen_item_details$pair_appearances <- as.integer(table(selected_pair_ids)[as.character(selected_pair_ids)])
chosen_item_details$adjective_appearances <- as.integer(table(selected_items$adjective)[selected_items$adjective])
occ_counts <- table(c(selected_items$occupation_A, selected_items$occupation_B))
chosen_item_details$occupation_A_appearances <- as.integer(occ_counts[selected_items$occupation_A])
chosen_item_details$occupation_B_appearances <- as.integer(occ_counts[selected_items$occupation_B])
stopifnot(!anyNA(chosen_item_details),
          all(abs(chosen_item_details$signed_log_odds_A - qlogis(selected_items$probability_A)) < 1e-10),
          all(abs(abs(chosen_item_details$signed_log_odds_A) - selected_items$stereotype_strength) < 1e-10))
chosen_pairs_details <- chosen_item_details[order(chosen_item_details$occupation_A,
                                                  chosen_item_details$occupation_B, match(chosen_item_details$typicality_region, region_labels),
                                                  chosen_item_details$adjective), ]

# Long format: two rows per item, each expressed for the occupation named in that row.
# This makes repeated occupations easy to compare across partners and adjectives.
detail_index <- match(occupation_details$item_id, selected_items$item_id)
is_A <- occupation_details$occupation == selected_items$occupation_A[detail_index]
occupation_details$typicality_score <- ifelse(is_A,
                                              selected_items$score_A[detail_index], selected_items$score_B[detail_index])
occupation_details$other_occupation_typicality_score <- ifelse(is_A,
                                                               selected_items$score_B[detail_index], selected_items$score_A[detail_index])
occupation_details$probability_percent <- 100 * occupation_details$probability
occupation_details$odds <- occupation_details$probability / (1 - occupation_details$probability)
occupation_details$signed_log_odds <- qlogis(occupation_details$probability)
occupation_details$stereotype_strength <- abs(occupation_details$signed_log_odds)
occupation_details$favoured_status <- ifelse(occupation_details$probability == .5, "Equal scores",
                                             ifelse(occupation_details$probability > .5, "Favoured", "Not favoured"))
occupation_details <- occupation_details[order(occupation_details$occupation,
                                               occupation_details$probability, occupation_details$item_id), ]

# ============================================================
# 5. PLOTS (ACTUAL SCORE POSITIONS; NO JITTER OR GRID ROUNDING)
# ============================================================
dot_size <- function(p) .45 + 1.55 * (p - .50) / .25
scatter_panel <- function(background, selected, title) {
  plot(background$score_A / 100, background$score_B / 100,
       pch = 16, cex = .35, col = adjustcolor("grey45", .10),
       xlim = c(0, 1), ylim = c(0, 1), asp = 1, xaxs = "i", yaxs = "i",
       xlab = "Typicality: Job A", ylab = "Typicality: Job B", main = title)
  abline(v = seq(0, 1, by = grid_width / 100),
         h = seq(0, 1, by = grid_width / 100), col = "grey90", lty = 3)
  abline(a = 0, b = 1, lty = 2, col = "grey50")
  # 25-75% probability wedge: B = 3*A and B = A/3.
  abline(a = 0, b = 3, col = "orange3", lty = 3)
  abline(a = 0, b = 1/3, col = "orange3", lty = 3)
  abline(v = low_high_cutoff / 100, h = low_high_cutoff / 100,
         col = "grey40", lty = 2)
  if (nrow(selected)) {
    ord <- order(selected$probability_favoured)
    points(selected$score_A[ord] / 100, selected$score_B[ord] / 100, pch = 16,
           cex = dot_size(selected$probability_favoured[ord]),
           col = adjustcolor("darkcyan", .65))
  }
}
plot_scatter <- function() {
  old <- par(no.readonly = TRUE); on.exit(par(old))
  par(mfrow = c(1, 1), mar = c(4, 4, 4, 1))
  scatter_panel(database, selected_items, "Selected 240 over full eligible inventory")
  legend("topleft", title = "Probability of favoured job", legend = c("50%", "60%", "75%"),
         pch = 16, pt.cex = dot_size(c(.50, .60, .75)), col = "darkcyan", bty = "n", cex = .8)
}
plot_distributions <- function() {
  old <- par(no.readonly = TRUE); on.exit(par(old))
  par(mfrow = c(1, 2), mar = c(6, 4, 3, 1))
  hist(database$probability_A, breaks = seq(0, 1, .05), right = FALSE,
       col = "steelblue", border = "white", main = "Eligible inventory",
       xlab = "Derived probability of A", xlim = c(0, 1))
  abline(v = c(.25, .75), lty = 2)
  barplot(selected_counts, las = 2, cex.names = .75, col = "darkcyan",
          ylim = c(0, 30), main = "Selected: 24 per range", ylab = "Items")
}
split_pages <- function(n, size = 22L) split(seq_len(n), ceiling(seq_len(n) / size))
plot_count_pages <- function(labels, counts, title) {
  for (ii in split_pages(length(labels))) {
    par(mfrow = c(1, 1), mar = c(4, 13, 3, 1))
    barplot(rev(counts[ii]), names.arg = rev(labels[ii]), horiz = TRUE,
            las = 1, cex.names = .75, col = "steelblue", xlim = c(0, max(1, counts) + 1),
            xlab = "Appearances", main = title)
  }
}
plot_all_checks <- function() {
  plot_distributions(); plot_scatter()
  # Ten panels let you inspect small and large dots separately by probability.
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
  for (label in range_labels) scatter_panel(
    pool[pool$probability_range == label, ],
    selected_items[selected_items$probability_range == label, ],
    paste("Probability A:", label))
  par(mfrow = c(1, 1), mar = c(5, 4, 3, 1))
  barplot(t(as.matrix(pair_summary[c("both_low", "A_low_B_high", "A_high_B_low", "both_high")])),
          col = c("grey65", "skyblue", "orange", "darkcyan"), border = NA,
          xlab = "Selected occupation pairs", ylab = "Appearances", ylim = c(0, 5),
          main = "Each pair: at most one adjective per typicality combination",
          legend.text = region_labels, args.legend = list(x = "topright", cex = .7, bty = "n"))
  plot_count_pages(occupation_summary$occupation, occupation_summary$appearances, "Occupation repetition")
  plot_count_pages(adjective_summary$adjective, adjective_summary$appearances, "Adjective repetition")
  words <- sort(unique(selected_items$adjective))
  for (ii in split_pages(length(words))) {
    labels <- rev(words[ii]); z <- selected_items[selected_items$adjective %in% labels, ]
    par(mar = c(4, 12, 3, 1))
    plot(z$stereotype_strength, match(z$adjective, labels), pch = 16,
         col = adjustcolor("darkcyan", .6), yaxt = "n", ylim = c(.5, length(labels) + .5),
         xlim = c(0, log(3) * 1.05), xlab = "Absolute log score ratio", ylab = "",
         main = "Adjective-by-strength: one dot per item")
    axis(2, seq_along(labels), labels, las = 1, cex.axis = .75)
  }
  for (ii in split_pages(nrow(occupation_coverage))) {
    m <- occupation_coverage[rev(ii), , drop = FALSE]
    par(mar = c(6, 13, 3, 1))
    # heatmap with explicit rectangles also works for a one-row final page.
    plot(NA, xlim = c(.5, 10.5), ylim = c(.5, nrow(m) + .5), axes = FALSE,
         xlab = "Probability range of the named occupation", ylab = "", main = "Occupation coverage")
    cols <- colorRampPalette(c("white", "steelblue"))(max(occupation_coverage) + 1L)
    for (i in seq_len(nrow(m))) for (j in seq_len(ncol(m))) {
      rect(j-.5, i-.5, j+.5, i+.5, col = cols[m[i,j]+1L], border = "grey90")
      text(j, i, m[i,j], cex = .7)
    }
    axis(1, seq_along(range_labels), range_labels, las = 2, cex.axis = .65)
    axis(2, seq_len(nrow(m)), rownames(m), las = 1, cex.axis = .75)
  }
}

# ============================================================
# 6. SAVE, DOCUMENT AND DISPLAY
# ============================================================
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
save_csv <- function(x, name) write.csv(x, file.path(output_dir, name),
                                        row.names = FALSE, fileEncoding = "UTF-8")
if (!reuse_existing) {
  save_csv(selected_items[export_columns], "experiment_items_240.csv")
  saveRDS(record, record_file)
  writeLines(capture.output(str(record)), file.path(output_dir, "original_selection_record.txt"))
}
save_csv(filter_summary, "database_filters.csv")
save_csv(selection_summary, "selection_by_range.csv")
save_csv(probability_summary, "selected_probability_summary.csv")
save_csv(spatial_summary, "spatial_cell_availability_and_selection.csv")
save_csv(coverage_summary, "spatial_coverage_by_probability.csv")
save_csv(as.data.frame(region_table(pool)), "region_availability_by_probability.csv")
save_csv(as.data.frame(region_table(selected_items)), "selected_regions_by_probability.csv")
save_csv(occupation_summary, "occupation_summary.csv")
save_csv(occupation_details, "occupation_item_details.csv")
save_csv(as.data.frame(occupation_coverage), "occupation_probability_coverage.csv")
save_csv(adjective_summary, "adjective_counts.csv")
save_csv(pair_summary, "pair_region_coverage.csv")
save_csv(chosen_item_details, "chosen_item_details_240.csv")
save_csv(chosen_pairs_details, "chosen_items_grouped_by_pair.csv")
detail_notes <- c(
  "CHOSEN ITEM DETAILS: COLUMN GUIDE", "",
  "chosen_item_details_240.csv: one row per selected item, in experiment CSV order.",
  "chosen_items_grouped_by_pair.csv: identical information, sorted by pair and low/high combination.",
  "occupation_item_details.csv: two rows per item; values refer to the occupation named in each row.",
  "These review tables are calculated after selection and do not change which items were chosen.",
  "A/B follow the alphabetical order used by the selection script; source_group1/source_group2 retain original order.",
  "score_A, score_B: original GPT typicality scores on the 0-100 scale, aligned to A and B.",
  "typicality_A_0_1, typicality_B_0_1: those scores divided by 100.",
  "probability_A, probability_B: derived probabilities on the 0-1 scale; _percent columns multiply by 100.",
  "probability_A = score_A/(score_A+score_B), using equal population base rates.",
  "score_ratio_A_over_B = score_A/score_B; odds_A_over_B = probability_A/probability_B.",
  "With 50/50 base rates these two ratios are equal. Neither ratio is itself a probability.",
  "signed_log_odds_A = natural log(score_A/score_B) = log(probability_A/probability_B).",
  "Positive signed_log_odds_A favours A; negative favours B; zero means equal scores.",
  "signed_log_odds_B has the opposite sign. All logarithms are natural logarithms.",
  "stereotype_strength = absolute signed log-odds: strength without direction.",
  "typicality_difference = abs(score_A-score_B); signed_typicality_difference_A_minus_B retains the direction.",
  "mean_typicality = (score_A+score_B)/2; this distinguishes low/low from high/high with similar ratios.",
  "favoured_occupation names the occupation with the higher score, or Neither for equal scores.",
  paste("typicality_level_A/B: Low below", low_high_cutoff, "and High at or above it."),
  "All appearance counts refer to the selected 240, not the full inventory.",
  "In the long occupation table, signed_log_odds is positive when the NAMED occupation is favoured.",
  "Derived probabilities assume scores can serve as likelihoods. They are not measured participant beliefs.")
writeLines(detail_notes, file.path(output_dir, "chosen_item_column_guide.txt"))
pdf(file.path(output_dir, "sampling_checks.pdf"), width = 13, height = 9, onefile = TRUE)
tryCatch(plot_all_checks(), finally = dev.off())
png(file.path(output_dir, "selected_typicality_scatter.png"), width = 1600, height = 1600, res = 170)
tryCatch(plot_scatter(), finally = dev.off())
png(file.path(output_dir, "probability_distributions.png"), width = 1800, height = 800, res = 150)
tryCatch(plot_distributions(), finally = dev.off())
methods_note <- c(
  "SHARED 240-ITEM SET: PROBABILITY BALANCE AND SPATIAL COVERAGE", "",
  "Each item is an occupation pair plus an adjective. A is alphabetically first (radix order).",
  "Population base rates are 50/50. Derived probability A = score_A/(score_A+score_B).",
  "These probabilities assume typicality scores can act as likelihoods; they are not measured participant beliefs.",
  "Only probabilities in [0.25,0.75] are eligible. Exactly 24 items per 5-percentage-point range.",
  "Ranges include their lower bound, exclude the upper bound; the last includes 0.75.",
  "Exactly 0.50 belongs to 50-<55%. Round to 12 decimals for bin assignment only.",
  paste("Low typicality is below", low_high_cutoff, "; high is at or above it (scores 0-100)."),
  "The low/high split is an operational design choice, not an empirically validated threshold.",
  paste("Each score axis uses cells of width", grid_width, "; 100 belongs to the last cell."),
  "Grid cells classify positions for sampling; actual scores and plotted positions are unchanged.",
  "Randomise the 240 probability-range slots. At each slot, remove items violating any hard cap.",
  "Among remaining cells in that probability range, choose uniformly among cells used least so far.",
  "Then choose uniformly among eligible items in that cell. Restart the entire draw at a dead end.",
  "Retain the first complete valid set. No constraints are silently relaxed.",
  "This is spatially balanced sequential selection, not uniform random sampling over items or valid sets.",
  "Inclusion probabilities are unequal and not estimated here; 24/pool size is not a valid inclusion probability.",
  "Spatial spread is encouraged, not guaranteed; inspect the range-specific plots and coverage tables.",
  "Small dots lie near equal scores; larger dots require unequal ratios and cannot occur everywhere.",
  "The probability cutoff restricts eligible points to A/3 <= B <= 3*A. Largest eligible favoured probability is 75%.",
  paste("Each adjective <=", max_adjective_appearances, "; each occupation <=", max_occupation_appearances, "."),
  paste("Each pair <=", max_pair_appearances, "appearances, at most one per low/high combination."),
  "Pair repeats use different adjectives. Repeats are allowed, not forced; four-region coverage is not guaranteed.",
  "No equal quadrant quotas or equal counts of pairs repeated 1/2/3/4 times are imposed.",
  paste("Excluded labels (case-insensitive):", paste(excluded_groups, collapse = ", ")),
  "Use the same experiment CSV for everyone; randomise presentation separately and enforce adjacency rules there.",
  "CSV order is for review only. Adviser reports, true occupations and left/right assignment are not generated.",
  "If using four blocks, assign six items per probability range to each block separately.",
  "Source and settings must match the original record for reuse. Use a new output folder to change the design.",
  "Content review and piloting are still needed.",
  paste("Source MD5:", source_md5), paste("Current R:", R.version.string),
  paste("Existing selection reused:", reuse_existing))
writeLines(methods_note, file.path(output_dir, "sampling_method.txt"))
cat("\nDATABASE FILTERS\n"); print(filter_summary, row.names = FALSE)
cat("\nPROBABILITY COUNTS\n"); print(selection_summary, row.names = FALSE)
cat("\nSPATIAL COVERAGE\n"); print(coverage_summary, row.names = FALSE)
cat("\nSELECTED TYPICALITY COMBINATIONS\n"); print(region_table(selected_items))
cat("\nPAIR REPETITIONS (number of pairs at each repetition count)\n")
print(table(factor(pair_summary$appearances, levels = 1:4)))
cat("\nMaximum occupation count:", max(occupation_summary$appearances),
    "\nMaximum adjective count:", max(adjective_counts),
    "\nPairs with all four combinations:", sum(pair_summary$regions_present == 4),
    "\nExactly neutral items:", sum(abs(selected_items$probability_A - .5) < 1e-12),
    "\n\nDONE. Experiment CSV:\n", experiment_file,
    "\nChecks, figures and method note:\n", output_dir, "\n")
if (interactive()) {
  View(chosen_item_details); View(chosen_pairs_details); View(pair_summary)
  if (identical(Sys.info()[["sysname"]], "Darwin") && capabilities("aqua")) {
    tryCatch({ quartz(width = 9, height = 9); plot_scatter() },
             error = function(e) message("Open the saved scatter PNG or sampling_checks.pdf."))
  }
}




































# ============================================================
# FULL STEREOTYPE DATABASE: TWO GRAPHS
# No sampling or simulation. Uses original GPT-4 scores.
# Requires base R only.
# ============================================================

# Select base_rate_database(9).csv when asked.
d <- read.csv(file.choose(), stringsAsFactors = FALSE)

needed <- c(
  "group1", "group2", "adjective",
  "score1_gpt4", "score2_gpt4"
)

if (!all(needed %in% names(d))) {
  stop("Choose the original base_rate_database CSV.")
}

s1 <- as.numeric(d$score1_gpt4)
s2 <- as.numeric(d$score2_gpt4)

if (any(!is.finite(s1) | !is.finite(s2) |
        s1 <= 0 | s2 <= 0 |
        s1 > 100 | s2 > 100)) {
  stop("Invalid typicality scores in CSV.")
}

if (any(s1 < s2)) {
  stop("This script expects the original database with score1 >= score2.")
}

# Population base rate BEFORE the adjective = 50/50.
# Probability of group 1 AFTER the adjective:
p_after <- s1 / (s1 + s2)

# Stereotype strength in log-odds space:
log_odds <- log(s1 / s2)

# Average of the two original typicality scores:
mean_score <- (s1 + s2) / 2


# ============================================================
# SAVE LOCATION
# ============================================================

out <- file.path(
  path.expand("~"),
  "Desktop",
  "stereotype_full_database_graphs"
)

dir.create(
  out,
  recursive = TRUE,
  showWarnings = FALSE
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

# Count occupation-pair + adjective combinations
# in each heat-map cell.

cell_count <- function(x, y, bx, by) {
  
  nx <- length(bx) - 1L
  ny <- length(by) - 1L
  
  ix <- findInterval(
    x, bx,
    rightmost.closed = TRUE
  )
  
  iy <- findInterval(
    y, by,
    rightmost.closed = TRUE
  )
  
  ok <- (
    ix >= 1 & ix <= nx &
      iy >= 1 & iy <= ny
  )
  
  matrix(
    tabulate(
      ix[ok] + nx * (iy[ok] - 1L),
      nbins = nx * ny
    ),
    nx,
    ny
  )
}


# Colour palette
cols <- colorRampPalette(
  c(
    "white",
    "#d6e8f3",
    "#77acd0",
    "#104b81"
  )
)(100)


# Graph layout:
# Main graph, colour scale and explanation below.

plot_start <- function() {
  
  layout(
    matrix(c(1, 2, 3, 3), nrow = 2, byrow = TRUE),
    widths = c(5.6, 0.65),
    heights = c(4.9, 1.0)
  )
  
  par(
    mar = c(5.0, 5.3, 6.8, 1.0),
    mgp = c(2.8, 0.8, 0)
  )
}


# Colour legend showing the number of combinations.

colour_legend <- function(max_count) {
  
  par(mar = c(5.0, 0.5, 6.8, 3.5))
  
  yy <- seq(
    0,
    max_count,
    length.out = 100
  )
  
  image(
    c(0, 1),
    yy,
    rbind(sqrt(yy), sqrt(yy)),
    col = cols,
    zlim = c(0, sqrt(max_count)),
    axes = FALSE,
    xlab = "",
    ylab = ""
  )
  
  axis(
    4,
    at = pretty(c(0, max_count), 5),
    las = 1,
    cex.axis = 0.77
  )
  
  mtext(
    "Weighted number of combinations",
    side = 4,
    line = 2.7,
    cex = 0.85
  )
}


# Plain-English explanations below each graph.

plot_notes <- function(lines) {
  
  par(mar = c(0, 0, 0, 0))
  plot.new()
  
  yy <- seq(
    0.80,
    0.18,
    length.out = length(lines)
  )
  
  for (i in seq_along(lines)) {
    
    text(
      0.5,
      yy[i],
      lines[i],
      cex = 0.92,
      font = if (i == 1) 2 else 1
    )
  }
}


# ============================================================
# GRAPH 1: TYPICALITY OF BOTH OCCUPATIONS
# ============================================================

# X-axis = adjective typicality for one occupation.
# Y-axis = adjective typicality for the other.
#
# The diagonal means the two scores are equal.
# Therefore P(group 1 | adjective) = 50%.
#
# Bottom-left = both occupations match weakly.
# Top-right = both occupations match strongly.
#
# The original CSV orders score1 >= score2.
# Reflect the distribution across the diagonal
# to display both possible orientations.

bs <- seq(0, 100, by = 5)

mid <- head(bs, -1) + 2.5

m <- cell_count(
  s1, s2,
  bs, bs
)

m <- (m + t(m)) / 2

stopifnot(
  abs(sum(m) - nrow(d)) < 1e-8
)


plot_one <- function() {
  
  plot_start()
  
  par(
    mar = c(5.0, 5.3, 6.8, 1.0)
  )
  
  image(
    mid,
    mid,
    sqrt(m),
    
    col = cols,
    zlim = c(0, sqrt(max(m))),
    
    xlim = c(0, 100),
    ylim = c(0, 100),
    
    axes = FALSE,
    asp = 1,
    
    xlab = "Typicality score: one occupation (0-100)",
    
    ylab = "Typicality score: other occupation (0-100)"
  )
  
  axis(
    1,
    at = seq(0, 100, 20)
  )
  
  axis(
    2,
    at = seq(0, 100, 20)
  )
  
  box()
  
  # Red dashed line = equal typicality scores.
  
  abline(
    0, 1,
    lty = 2,
    lwd = 2,
    col = "#b5413c"
  )
  
  title(
    "1. How strongly does the adjective match each occupation?",
    line = 4.6
  )
  
  
  # Two examples present in the original database.
  #
  # 10/10: both occupations have low scores.
  # 91/91: both occupations have high scores.
  #
  # Both produce 50/50 after the adjective.
  
  for (k in seq_along(c(10, 91))) {
    
    v <- c(10, 91)[k]
    
    if (any(s1 == v & s2 == v)) {
      
      points(
        v, v,
        pch = 21,
        bg = "white",
        col = "#b5413c",
        cex = 2.1,
        lwd = 2
      )
      
      text(
        v, v,
        k,
        font = 2,
        cex = 0.75
      )
    }
  }
  
  
  colour_legend(max(m))
  
  
  plot_notes(c(
    "Darker = more combinations. Dashed line = 50/50 probability AFTER the adjective.",
    
    "1: scores 10/10 (both weak).  2: scores 91/91 (both strong). Both give 50/50.",
    
    "Both score orientations are displayed because the original CSV puts the larger score first."
  ))
}


# Save graph 1.

png(
  file.path(
    out,
    "01_typicality_pairs.png"
  ),
  width = 1850,
  height = 1500,
  res = 160
)

tryCatch(
  plot_one(),
  finally = dev.off()
)


# ============================================================
# GRAPH 2: PROBABILITY AFTER THE ADJECTIVE
# ============================================================

# Bottom x-axis:
# Probability of group 1 AFTER the adjective.
#
# Top x-axis:
# Equivalent stereotype strength in log-odds.
#
# Y-axis:
# Average typicality score of the two occupations.
#
# The aim is to see how different original
# typicality scores can produce the same probability.
#
# Example:
# 80/20 -> probability = 80%, mean score = 50
# 40/10 -> probability = 80%, mean score = 25
#
# These have the same probability and log-odds,
# but different original typicality scores.

bp <- seq(
  0.50,
  0.93,
  by = 0.01
)

bm <- seq(
  0,
  100,
  by = 5
)

m2 <- cell_count(
  p_after,
  mean_score,
  bp,
  bm
)

stopifnot(
  sum(m2) == nrow(d)
)


# Find real examples in the database.

example <- function(a, b, adj) {
  
  ii <- which(
    d$group1 == a &
      d$group2 == b &
      d$adjective == adj
  )
  
  if (length(ii) != 1L) {
    stop("An annotated example was not found in this CSV.")
  }
  
  ii
}


e1 <- example(
  "police officer",
  "fireman",
  "dishonest"
)

e2 <- example(
  "paramedic",
  "computer programmer",
  "brave"
)


plot_two <- function() {
  
  plot_start()
  
  par(
    mar = c(5.0, 5.3, 6.8, 1.0)
  )
  
  image(
    head(bp, -1) + 0.005,
    head(bm, -1) + 2.5,
    sqrt(m2),
    
    col = cols,
    zlim = c(0, sqrt(max(m2))),
    
    xlim = c(0.50, 0.93),
    ylim = c(0, 100),
    
    axes = FALSE,
    
    xlab = "Probability of group 1 AFTER the adjective",
    
    ylab = "Average of the two typicality scores (0-100)"
  )
  
  
  # Bottom axis: probability after adjective.
  
  axis(
    1,
    at = seq(0.5, 0.9, 0.1),
    labels = paste0(seq(50, 90, 10), "%")
  )
  
  axis(
    2,
    at = seq(0, 100, 20)
  )
  
  
  # Top axis: equivalent log-odds strength.
  
  axis(
    3,
    at = plogis(c(0, 0.5, 1, 1.5, 2, 2.3)),
    labels = c("0", "0.5", "1.0", "1.5", "2.0", "2.3")
  )
  
  mtext(
    "Equivalent log-odds strength",
    side = 3,
    line = 2.1,
    cex = 0.9
  )
  
  title(
    "2. Same probability, different original scores",
    line = 5.0
  )
  
  box()
  
  
  # Orange circles = two actual combinations
  # from the original database.
  
  points(
    p_after[c(e1, e2)],
    mean_score[c(e1, e2)],
    
    pch = 21,
    bg = "#fff5dd",
    col = "#b65e1d",
    cex = 2.3,
    lwd = 2
  )
  
  text(
    p_after[c(e1, e2)],
    mean_score[c(e1, e2)],
    
    c("1", "2"),
    font = 2,
    cex = 0.77
  )
  
  
  colour_legend(max(m2))
  
  
  plot_notes(c(
    "Across = probability AFTER the adjective. Up = how high the two scores are overall.",
    
    "1: dishonest: police officer 40.28 / fireman 10.08. Probability = 80%, average = 25.18.",
    
    "2: brave: paramedic 90.12 / computer programmer 22.60. Probability = 80%, average = 56.36."
  ))
}


# Save graph 2.

png(
  file.path(
    out,
    "02_probability_after_adjective.png"
  ),
  width = 1850,
  height = 1500,
  res = 160
)

tryCatch(
  plot_two(),
  finally = dev.off()
)


# ============================================================
# FINISHED
# ============================================================

cat(
  "\nOriginal rows analysed:",
  nrow(d),
  "\n"
)

cat(
  "P(group 1 | adjective) range:",
  range(p_after),
  "\n"
)

cat(
  "Images saved to:",
  out,
  "\n"
)














# =====================================================
# STEREOTYPE ASSOCIATIONS AND RESULTING PROBABILITY
# One plotted point per original association
# Regular grid, like supervisor's example
# =====================================================

# 1. Load original database

d <- read.csv(file.choose(), stringsAsFactors = FALSE)

s1 <- as.numeric(d$score1_gpt4) / 100
s2 <- as.numeric(d$score2_gpt4) / 100

stopifnot(
  all(is.finite(s1), is.finite(s2)),
  all(s1 > 0, s2 > 0),
  all(s1 <= 1, s2 <= 1),
  all(!is.na(d$group1), !is.na(d$group2))
)


# 2. Define jobs A and B alphabetically

a_first <- d$group1 < d$group2

A <- ifelse(a_first, s1, s2)
B <- ifelse(a_first, s2, s1)


# 3. Calculate probability AFTER adjective

p_A <- A / (A + B)
p_B <- 1 - p_A

# Probability of the favoured job

p_favoured <- pmax(p_A, p_B)


# 4. Round positions to a 0.05 grid
# This changes their displayed positions only.

x <- round(A / 0.05) * 0.05
y <- round(B / 0.05) * 0.05


# 5. Bubble sizes
# Larger dot = stronger resulting probability

size <- 0.5 + 2 * (p_favoured - 0.5) / 0.5

# Draw smaller dots first
ord <- order(size)


# 6. Save graph to Desktop

output <- path.expand(
  "~/Desktop/stereotype_scatter_grid.png"
)

png(
  output,
  width = 1500,
  height = 1500,
  res = 160
)

par(mar = c(5, 5, 3, 2))

plot(
  x[ord],
  y[ord],

  pch = 16,
  cex = size[ord],
  col = "black",

  xlim = c(-0.03, 1.03),
  ylim = c(-0.03, 1.03),

  asp = 1,
  axes = FALSE,
  xaxs = "i",
  yaxs = "i",

  xlab = "Typicality score: Job A",
  ylab = "Typicality score: Job B",

  main = "Stereotype associations and resulting probability"
)

axis(1, at = seq(0, 1, 0.1))
axis(2, at = seq(0, 1, 0.1))
box()

dev.off()

cat("Saved to:", output, "\n")
cat("Associations plotted:", nrow(d), "\n")
cat(
  "Distinct visible grid positions:",
  nrow(unique(data.frame(x, y))),
  "\n"
)









# ============================================================
# FULL INVENTORY SCATTER PLOT
# Correct version: one point per original association
# Bigger dots, exact positions, no grid rounding
# ============================================================

# 1. Load original CSV
d <- read.csv(file.choose(), stringsAsFactors = FALSE)

required <- c(
  "group1", "group2", "adjective",
  "score1_gpt4", "score2_gpt4"
)

if (!all(required %in% names(d))) {
  stop("Please choose the original base_rate_database CSV.")
}

s1 <- as.numeric(d$score1_gpt4)
s2 <- as.numeric(d$score2_gpt4)

stopifnot(
  all(is.finite(s1), is.finite(s2)),
  all(s1 > 0, s2 > 0),
  all(s1 <= 100, s2 <= 100),
  !anyNA(d$group1),
  !anyNA(d$group2),
  all(nzchar(trimws(d$group1))),
  all(nzchar(trimws(d$group2))),
  all(d$group1 != d$group2)
)

# 2. Define Job A and Job B alphabetically
a_first <- d$group1 < d$group2

A <- ifelse(a_first, s1, s2) / 100
B <- ifelse(a_first, s2, s1) / 100

# 3. Calculate probability AFTER adjective
p_A <- A / (A + B)
p_B <- 1 - p_A

# Probability of the favoured occupation
p_favoured <- pmax(p_A, p_B)

# 4. Bigger dot size function
dot_size <- function(p) {
  0.25 + 1.00 * (p - 0.50) / 0.50
}

# Draw smaller dots first
ord <- order(p_favoured)

# 5. Save location
output_file <- path.expand("~/Desktop/stereotype_scatter_small_correct.png")

# 6. Create PNG
png(
  filename = output_file,
  width = 1700,
  height = 1600,
  res = 170,
  bg = "white"
)

par(mar = c(5, 5, 3, 2))

plot(
  A[ord],
  B[ord],
  pch = 16,
  cex = dot_size(p_favoured[ord]),
  col = adjustcolor("black", alpha.f = 0.05),
  
  xlim = c(-0.03, 1.03),
  ylim = c(-0.03, 1.03),
  
  asp = 1,
  xaxs = "i",
  yaxs = "i",
  
  xlab = "Typicality score: Job A (0-1)",
  ylab = "Typicality score: Job B (0-1)",
  main = "Stereotype associations and resulting probability"
)

abline(0, 1, lty = 2, col = "grey55")

legend(
  "topright",
  title = "Probability after adjective",
  legend = c("50%", "60%", "75%", "90%"),
  pch = 16,
  pt.cex = dot_size(c(0.50, 0.60, 0.75, 0.90)),
  col = "black",
  bty = "n",
  cex = 0.85
)

dev.off()

cat("Saved to:\n", output_file, "\n")
cat("Number of original associations plotted:", nrow(d), "\n")





