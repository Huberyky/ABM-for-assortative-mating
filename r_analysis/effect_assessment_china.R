#!/usr/bin/env Rscript

# China analogue of Table 3.  Run means are first formed within each Chinese
# birth cohort; the displayed mean and SD then describe heterogeneity across
# the six cohorts (one country, unlike the paper's 12-country table).
if (.Platform$OS.type == "windows" && identical(Sys.getlocale("LC_CTYPE"), "C")) {
  invisible(suppressWarnings(try(
    Sys.setlocale("LC_CTYPE", "Chinese (Simplified)_China.utf8"), silent = TRUE
  )))
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript effect_assessment_china.R <ablation_runs.csv> <targets.csv> <output_dir>")
}
runs <- read.csv(args[[1]], stringsAsFactors = FALSE, check.names = FALSE)
targets <- read.csv(args[[2]], stringsAsFactors = FALSE, check.names = FALSE)
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
source_map <- c(`1` = "full", `2` = "no_s", `3` = "no_delta", `4` = "no_a", `5` = "no_y")
required <- c("country", "condition_no", "run_no", "ten_cohort",
              "perc_hyper_sim", "perc_homo_sim", "perc_hypo_sim", "a_diff_sim")
missing <- setdiff(required, names(runs))
if (length(missing)) stop("ablation input lacks columns: ", paste(missing, collapse = ", "))
if (any(duplicated(runs[c("condition_no", "run_no", "ten_cohort")]))) stop("duplicate run/cohort rows")
run_counts <- aggregate(run_no ~ condition_no, runs, function(x) length(unique(x)))
if (nrow(run_counts) != 5 || any(run_counts$run_no != 50)) stop("five variants x 50 runs are required")
runs$source <- unname(source_map[as.character(runs$condition_no)])
if (any(is.na(runs$source))) stop("unexpected condition number in ablation input")

simulation <- aggregate(
  cbind(perc_hyper_sim, perc_homo_sim, perc_hypo_sim, a_diff_sim) ~ country + ten_cohort + source,
  data = runs, mean
)
observed <- targets[c("country", "ten_cohort", "perc_hyper_obs", "perc_homo_obs", "perc_hypo_obs", "a_diff_obs")]
names(observed)[3:6] <- c("perc_hyper_sim", "perc_homo_sim", "perc_hypo_sim", "a_diff_sim")
observed$source <- "CGSS"
country_cohort <- rbind(observed, simulation)
source_levels <- c("CGSS", "full", "no_s", "no_delta", "no_a", "no_y")
country_cohort$source <- factor(country_cohort$source, levels = source_levels)

metric_map <- c(
  perc_hyper_sim = "hypergamic",
  perc_homo_sim = "homogamic",
  perc_hypo_sim = "hypogamic"
)
long <- do.call(rbind, lapply(names(metric_map), function(metric) data.frame(
  country = country_cohort$country,
  ten_cohort = country_cohort$ten_cohort,
  source = country_cohort$source,
  couple_type = metric_map[[metric]],
  share = country_cohort[[metric]]
)))
long$source <- factor(long$source, levels = source_levels)
long$couple_type <- factor(long$couple_type,
  levels = c("hypergamic", "homogamic", "hypogamic"))
means <- aggregate(share ~ source + couple_type, long, mean)
sds <- aggregate(share ~ source + couple_type, long, sd)
names(means)[3] <- "mean"
names(sds)[3] <- "sd"
table_long <- merge(means, sds, by = c("source", "couple_type"), sort = FALSE)
table_long <- table_long[order(table_long$source, table_long$couple_type), ]

full <- subset(country_cohort, source == "full")
effect_rows <- list()
for (variant in c("no_s", "no_delta", "no_a", "no_y")) {
  ablated <- subset(country_cohort, source == variant)
  paired <- merge(full, ablated, by = c("country", "ten_cohort"), suffixes = c("_full", "_ablated"))
  for (metric in names(metric_map)) {
    difference <- paired[[paste0(metric, "_ablated")]] - paired[[paste0(metric, "_full")]]
    effect_rows[[length(effect_rows) + 1]] <- data.frame(
      source = variant,
      couple_type = metric_map[[metric]],
      difference_vs_full = mean(difference),
      difference_percentage_points = 100 * mean(difference),
      sd_difference_across_cohorts = sd(difference),
      cohorts = length(difference)
    )
  }
}
effects <- do.call(rbind, effect_rows)
full_parameters <- runs[runs$condition_no == 1, c("ws_f", "ws_m", "wy_f", "wy_m", "wa_f", "wa_m", "delta")]
full_parameters <- full_parameters[1, ]
active_map <- c(
  no_s = any(abs(c(full_parameters$ws_f, full_parameters$ws_m)) > 1e-12),
  no_delta = abs(full_parameters$delta - 0.5) > 1e-12,
  no_a = any(abs(c(full_parameters$wa_f, full_parameters$wa_m)) > 1e-12),
  no_y = any(abs(c(full_parameters$wy_f, full_parameters$wy_m)) > 1e-12)
)
effects$mechanism_active_in_full <- unname(active_map[effects$source])
effects$mechanism_effect_percentage_points <- ifelse(
  effects$mechanism_active_in_full, effects$difference_percentage_points, 0
)
effects$interpretation <- ifelse(
  effects$mechanism_active_in_full,
  "ablation contrast",
  "mechanism already zero in calibrated full model; raw contrast is independent-run Monte Carlo noise"
)

fit <- merge(full, targets, by = c("country", "ten_cohort"))
fit_metrics <- data.frame(
  metric = c("hypergamic", "homogamic", "hypogamic", "age_difference"),
  MAE = c(
    mean(abs(fit$perc_hyper_sim - fit$perc_hyper_obs)),
    mean(abs(fit$perc_homo_sim - fit$perc_homo_obs)),
    mean(abs(fit$perc_hypo_sim - fit$perc_hypo_obs)),
    mean(abs(fit$a_diff_sim - fit$a_diff_obs))
  ),
  RMSE = c(
    sqrt(mean((fit$perc_hyper_sim - fit$perc_hyper_obs)^2)),
    sqrt(mean((fit$perc_homo_sim - fit$perc_homo_obs)^2)),
    sqrt(mean((fit$perc_hypo_sim - fit$perc_hypo_obs)^2)),
    sqrt(mean((fit$a_diff_sim - fit$a_diff_obs)^2))
  )
)

write.csv(country_cohort, file.path(output_dir, "china_table3_country_cohort.csv"), row.names = FALSE)
write.csv(table_long, file.path(output_dir, "china_table3_long.csv"), row.names = FALSE)
write.csv(effects, file.path(output_dir, "china_ablation_effects_vs_full.csv"), row.names = FALSE)
write.csv(fit_metrics, file.path(output_dir, "china_full_fit_metrics.csv"), row.names = FALSE)
save(country_cohort, table_long, effects, fit_metrics,
     file = file.path(output_dir, "china_ablation_results.RData"))
cat(sprintf("Saved China Table-3 analogue: %d cells and %d effects\n", nrow(table_long), nrow(effects)))
