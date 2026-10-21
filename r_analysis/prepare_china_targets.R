#!/usr/bin/env Rscript

# China analogue of the paper's empirical-target preparation.  The respondent
# remains the focal person, as in the ESS analysis; spouse schooling and birth
# year are used to orient husband/wife outcomes.
if (.Platform$OS.type == "windows" && identical(Sys.getlocale("LC_CTYPE"), "C")) {
  invisible(suppressWarnings(try(
    Sys.setlocale("LC_CTYPE", "Chinese (Simplified)_China.utf8"), silent = TRUE
  )))
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript prepare_china_targets.R <CGSS2021.dta> <output_dir>")
}
if (!requireNamespace("haven", quietly = TRUE)) stop("R package 'haven' is required")

input_path <- normalizePath(args[[1]], winslash = "/", mustWork = TRUE)
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

raw <- haven::read_dta(input_path)[c("A2", "A3_1", "A71a", "A69", "A7a", "A72", "weight")]
audit <- data.frame(stage = "raw", rows = nrow(raw))
dat <- subset(raw, A2 %in% c(1, 2) & A69 %in% c(3, 4))
audit <- rbind(audit, data.frame(stage = "currently_married_with_spouse", rows = nrow(dat)))

recode_education <- function(x) {
  out <- rep(NA_integer_, length(x))
  out[x >= 1 & x <= 3] <- 1L
  out[x >= 4 & x <= 8] <- 2L
  out[x >= 9 & x <= 13] <- 3L
  out
}
dat$self_edu <- recode_education(as.numeric(dat$A7a))
dat$partner_edu <- recode_education(as.numeric(dat$A72))
valid <- complete.cases(dat[c("A3_1", "A71a", "self_edu", "partner_edu", "weight")]) &
  dat$A3_1 >= 1900 & dat$A3_1 <= 2005 & dat$A71a >= 1900 & dat$A71a <= 2005 &
  dat$weight > 0
dat <- dat[valid, ]
audit <- rbind(audit, data.frame(stage = "valid_birth_education_weight", rows = nrow(dat)))

male_respondent <- dat$A2 == 1
dat$male_edu <- ifelse(male_respondent, dat$self_edu, dat$partner_edu)
dat$female_edu <- ifelse(male_respondent, dat$partner_edu, dat$self_edu)
dat$male_birth <- ifelse(male_respondent, dat$A3_1, dat$A71a)
dat$female_birth <- ifelse(male_respondent, dat$A71a, dat$A3_1)
dat$ten_cohort <- cut(floor(dat$A3_1), breaks = seq(1940, 2000, by = 10), dig.lab = 20)
dat$hyper <- as.integer(dat$male_edu > dat$female_edu)
dat$homo <- as.integer(dat$male_edu == dat$female_edu)
dat$hypo <- as.integer(dat$male_edu < dat$female_edu)
dat$a_diff_obs <- dat$female_birth - dat$male_birth
dat <- dat[!is.na(dat$ten_cohort), ]
audit <- rbind(audit, data.frame(stage = "within_requested_respondent_cohorts", rows = nrow(dat)))

weighted_cohort <- function(group) {
  n_eff <- sum(group$weight)^2 / sum(group$weight^2)
  hyper <- weighted.mean(group$hyper, group$weight)
  homo <- weighted.mean(group$homo, group$weight)
  hypo <- weighted.mean(group$hypo, group$weight)
  age <- weighted.mean(group$a_diff_obs, group$weight)
  age_variance <- weighted.mean((group$a_diff_obs - age)^2, group$weight)
  data.frame(
    perc_hyper_obs = hyper,
    perc_homo_obs = homo,
    perc_hypo_obs = hypo,
    a_diff_obs = age,
    perc_hyper_obs_se_kish = sqrt(hyper * (1 - hyper) / n_eff),
    perc_homo_obs_se_kish = sqrt(homo * (1 - homo) / n_eff),
    perc_hypo_obs_se_kish = sqrt(hypo * (1 - hypo) / n_eff),
    a_diff_obs_se_kish = sqrt(age_variance / n_eff),
    n_unweighted = nrow(group),
    weight_sum = sum(group$weight),
    kish_effective_n = n_eff
  )
}
parts <- split(dat, dat$ten_cohort, drop = TRUE)
targets <- do.call(rbind, lapply(parts, weighted_cohort))
targets$ten_cohort <- rownames(targets)
rownames(targets) <- NULL
targets$country <- "CN"
targets <- targets[c("country", "ten_cohort", "perc_hyper_obs", "perc_homo_obs",
                     "perc_hypo_obs", "a_diff_obs", "perc_hyper_obs_se_kish",
                     "perc_homo_obs_se_kish", "perc_hypo_obs_se_kish",
                     "a_diff_obs_se_kish", "n_unweighted", "weight_sum", "kish_effective_n")]
if (nrow(targets) != 6) stop("CGSS target extraction did not produce all six cohorts")
if (max(abs(rowSums(targets[3:5]) - 1)) > 1e-10) stop("target shares do not sum to one")

write.csv(targets, file.path(output_dir, "cgss2021_targets_R.csv"), row.names = FALSE)
write.csv(audit, file.path(output_dir, "cgss2021_target_audit_R.csv"), row.names = FALSE)
save(targets, audit, file = file.path(output_dir, "cgss2021_targets.RData"))
cat(sprintf("Saved %d weighted China cohort targets from %d eligible respondents\n",
            nrow(targets), nrow(dat)))
