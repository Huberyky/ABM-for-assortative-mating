#!/usr/bin/env Rscript

# Reproduce the aggregation behind Grow & Van Bavel (2015), Table 3, from
# run-level Mesa summaries.  Simulation remains in Mesa; the paper's original
# R aggregation semantics are retained here.
if (.Platform$OS.type == "windows" && identical(Sys.getlocale("LC_CTYPE"), "C")) {
  invisible(suppressWarnings(try(
    Sys.setlocale("LC_CTYPE", "Chinese (Simplified)_China.utf8"), silent = TRUE
  )))
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7) {
  stop(paste(
    "Usage: Rscript effect_assessment_mesa.R",
    "<full.csv> <no_s.csv> <no_delta.csv> <no_a.csv> <no_y.csv>",
    "<ESS_results.RData> <output_dir>"
  ))
}

paths <- args[1:5]
labels <- c("full", "no_s", "no_delta", "no_a", "no_y")
ess_path <- normalizePath(args[[6]], winslash = "/", mustWork = TRUE)
output_dir <- args[[7]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

required <- c(
  "country", "condition_no", "run_no", "ten_cohort",
  "perc_hyper_sim", "perc_homo_sim", "perc_hypo_sim"
)

prepare_variant <- function(path, label) {
  dat <- read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
  missing <- setdiff(required, names(dat))
  if (length(missing) > 0) {
    stop(label, " input lacks columns: ", paste(missing, collapse = ", "))
  }
  if (any(duplicated(dat[c("condition_no", "run_no", "ten_cohort")]))) {
    stop(label, " contains duplicate condition/run/cohort rows")
  }
  run_counts <- aggregate(run_no ~ country, dat, function(x) length(unique(x)))
  if (length(unique(run_counts$run_no)) != 1) {
    stop(label, " has unequal run counts across countries")
  }
  out <- aggregate(
    cbind(perc_hyper_sim, perc_homo_sim, perc_hypo_sim) ~ country + ten_cohort,
    data = dat, mean
  )
  out$source <- label
  out$runs_per_country <- unique(run_counts$run_no)
  out
}

simulation <- do.call(rbind, Map(prepare_variant, paths, labels))
simulation$source <- factor(simulation$source, levels = labels)

load(ess_path)
if (!exists("ESS_results")) stop("ESS_results object not found in ", ess_path)
countries <- unique(simulation$country)
cohorts <- unique(simulation$ten_cohort)
ess <- subset(ESS_results, country %in% countries & ten_cohort %in% cohorts)
ess <- ess[c(
  "country", "ten_cohort", "perc_hyper_obs", "perc_homo_obs", "perc_hypo_obs"
)]
names(ess)[3:5] <- c("perc_hyper_sim", "perc_homo_sim", "perc_hypo_sim")
ess$source <- factor("ESS", levels = c("ESS", labels))
ess$runs_per_country <- NA_integer_

country_cohort <- rbind(
  ess[c("country", "ten_cohort", "perc_hyper_sim", "perc_homo_sim",
        "perc_hypo_sim", "source", "runs_per_country")],
  simulation[c("country", "ten_cohort", "perc_hyper_sim", "perc_homo_sim",
               "perc_hypo_sim", "source", "runs_per_country")]
)
country_cohort$source <- factor(
  as.character(country_cohort$source), levels = c("ESS", labels)
)

metric_map <- c(
  perc_hyper_sim = "hypergamic",
  perc_homo_sim = "homogamic",
  perc_hypo_sim = "hypogamic"
)
long <- do.call(rbind, lapply(names(metric_map), function(metric) {
  data.frame(
    country = country_cohort$country,
    ten_cohort = country_cohort$ten_cohort,
    source = country_cohort$source,
    couple_type = metric_map[[metric]],
    share = country_cohort[[metric]],
    stringsAsFactors = FALSE
  )
}))
long$source <- factor(long$source, levels = c("ESS", labels))
long$couple_type <- factor(
  long$couple_type, levels = c("hypergamic", "homogamic", "hypogamic")
)

means <- aggregate(share ~ source + couple_type, long, mean)
sds <- aggregate(share ~ source + couple_type, long, sd)
names(means)[3] <- "mean"
names(sds)[3] <- "sd"
table_long <- merge(means, sds, by = c("source", "couple_type"), sort = FALSE)
table_long$source <- factor(table_long$source, levels = c("ESS", labels))
table_long$couple_type <- factor(
  table_long$couple_type, levels = c("hypergamic", "homogamic", "hypogamic")
)
table_long <- table_long[order(table_long$source, table_long$couple_type), ]

table_wide <- reshape(
  table_long, idvar = "source", timevar = "couple_type", direction = "wide"
)
table_wide <- table_wide[match(c("ESS", labels), table_wide$source), ]

write.csv(
  country_cohort,
  file.path(output_dir, "mesa_table3_country_cohort.csv"), row.names = FALSE
)
write.csv(
  table_long,
  file.path(output_dir, "mesa_table3_long.csv"), row.names = FALSE
)
write.csv(
  table_wide,
  file.path(output_dir, "mesa_table3_wide.csv"), row.names = FALSE
)
save(country_cohort, table_long, table_wide,
     file = file.path(output_dir, "mesa_table3_results.RData"))

cat(sprintf(
  "Saved Table 3: %d source/type rows; %d country/cohort rows; %d countries\n",
  nrow(table_long), nrow(country_cohort), length(countries)
))
