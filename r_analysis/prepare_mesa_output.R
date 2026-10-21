#!/usr/bin/env Rscript

# Parameterized replacement for original scripts 3_1--3_5.  The aggregation
# logic is intentionally unchanged; only hard-coded paths are removed.
# Some Windows R installations start in the C locale after failing to find
# C.UTF-8.  Set a UTF-8 CTYPE fallback so Chinese project paths remain visible.
if (.Platform$OS.type == "windows" && identical(Sys.getlocale("LC_CTYPE"), "C")) {
  invisible(suppressWarnings(try(
    Sys.setlocale("LC_CTYPE", "Chinese (Simplified)_China.utf8"),
    silent = TRUE
  )))
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop("Usage: Rscript prepare_mesa_output.R <census_dir> <output.RData> <object_name> [pattern] [output.csv]")
}

census_dir <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
output_rdata <- args[[2]]
object_name <- args[[3]]
pattern <- if (length(args) >= 4) args[[4]] else "\\.txt$"
output_csv <- if (length(args) >= 5) args[[5]] else NA_character_

files <- list.files(census_dir, pattern = pattern, full.names = TRUE)
if (length(files) == 0) stop("No Mesa census files matched pattern: ", pattern)

read_run <- function(path) {
  read.table(path, skip = 39, header = TRUE, sep = "", stringsAsFactors = FALSE,
             na.strings = "NA", check.names = FALSE)
}
dat <- do.call(rbind, lapply(files, read_run))

dat_married_female <- subset(
  dat, l_i == "married" & g_i == "female" &
    year_of_birth_i >= 1941 & year_of_birth_i <= 1980.9
)
dat_married_male <- subset(
  dat, l_i == "married" & g_i == "male" &
    year_of_birth_i >= 1941 & year_of_birth_i <= 1980.9
)
dat_married_female$a_f <- dat_married_female$a_i
dat_married_female$a_m <- dat_married_female$a_j
dat_married_male$a_f <- dat_married_male$a_j
dat_married_male$a_m <- dat_married_male$a_i
dat_married <- rbind(dat_married_female, dat_married_male)
if (nrow(dat_married) == 0) stop("No married focal-cohort records found")

dat_married$year_of_birth_i_floor <- floor(as.numeric(dat_married$year_of_birth_i))
dat_married$age_i <- floor(as.numeric(dat_married$a_i) / 10)
dat_married$year_of_birth_j_floor <- floor(as.numeric(dat_married$year_of_birth_j))
dat_married$age_j <- floor(as.numeric(dat_married$a_j) / 10)
dat_married$age_f <- floor(as.numeric(dat_married$a_f) / 10)
dat_married$age_m <- floor(as.numeric(dat_married$a_m) / 10)

dat_married$s_i_reduced <- NA_integer_
dat_married$s_j_reduced <- NA_integer_
dat_married$s_i_reduced[dat_married$s_i >= 1 & dat_married$s_i <= 2] <- 1
dat_married$s_i_reduced[dat_married$s_i == 3] <- 2
dat_married$s_i_reduced[dat_married$s_i == 4] <- 3
dat_married$s_j_reduced[dat_married$s_j >= 1 & dat_married$s_j <= 2] <- 1
dat_married$s_j_reduced[dat_married$s_j == 3] <- 2
dat_married$s_j_reduced[dat_married$s_j == 4] <- 3
dat_married$ten_cohort <- factor(cut(
  dat_married$year_of_birth_i_floor,
  breaks = seq(1900, 2020, by = 10), dig.lab = 20
))

dat_married$hyper <- 0L
dat_married$homo <- 0L
dat_married$hypo <- 0L
dat_married$hyper[dat_married$s_i_reduced > dat_married$s_j_reduced & dat_married$g_i == "male"] <- 1L
dat_married$hyper[dat_married$s_i_reduced < dat_married$s_j_reduced & dat_married$g_i == "female"] <- 1L
dat_married$homo[dat_married$s_i_reduced == dat_married$s_j_reduced] <- 1L
dat_married$hypo[dat_married$s_i_reduced < dat_married$s_j_reduced & dat_married$g_i == "male"] <- 1L
dat_married$hypo[dat_married$s_i_reduced > dat_married$s_j_reduced & dat_married$g_i == "female"] <- 1L

run_formula <- cbind(hyper, homo, hypo) ~
  country + condition_no + run_no + ten_cohort + delta +
  ws_f + ws_m + wy_f + wy_m + wa_f + wa_m
couple_types_run_level <- aggregate(run_formula, data = dat_married, sum)
names(couple_types_run_level)[names(couple_types_run_level) == "hyper"] <- "count_hyper_sim"
names(couple_types_run_level)[names(couple_types_run_level) == "homo"] <- "count_homo_sim"
names(couple_types_run_level)[names(couple_types_run_level) == "hypo"] <- "count_hypo_sim"
denominator <- with(couple_types_run_level, count_hyper_sim + count_homo_sim + count_hypo_sim)
couple_types_run_level$perc_hyper_sim <- couple_types_run_level$count_hyper_sim / denominator
couple_types_run_level$perc_homo_sim <- couple_types_run_level$count_homo_sim / denominator
couple_types_run_level$perc_hypo_sim <- couple_types_run_level$count_hypo_sim / denominator

condition_formula <- cbind(perc_hyper_sim, perc_homo_sim, perc_hypo_sim) ~
  condition_no + country + ten_cohort + delta + ws_f + ws_m + wy_f + wy_m + wa_f + wa_m
couple_types_condition_level <- aggregate(condition_formula, data = couple_types_run_level, mean)

dat_married$a_diff_sim <- dat_married$age_m - dat_married$age_f
age_run <- aggregate(
  a_diff_sim ~ country + condition_no + run_no + ten_cohort + delta +
    ws_f + ws_m + wy_f + wy_m + wa_f + wa_m,
  data = dat_married, mean
)
age_condition <- aggregate(
  a_diff_sim ~ country + condition_no + ten_cohort + delta +
    ws_f + ws_m + wy_f + wy_m + wa_f + wa_m,
  data = age_run, mean
)

simulation_results <- merge(
  couple_types_condition_level, age_condition,
  by = intersect(names(couple_types_condition_level), names(age_condition))
)
assign(object_name, simulation_results, envir = .GlobalEnv)
dir.create(dirname(output_rdata), recursive = TRUE, showWarnings = FALSE)
save(list = object_name, file = output_rdata)
if (!is.na(output_csv)) {
  dir.create(dirname(output_csv), recursive = TRUE, showWarnings = FALSE)
  write.csv(simulation_results, output_csv, row.names = FALSE)
}
cat(sprintf("Saved %s: %d condition/cohort rows from %d census files\n",
            object_name, nrow(simulation_results), length(files)))
