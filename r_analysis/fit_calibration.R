#!/usr/bin/env Rscript

# Continue the paper's original R response-surface analysis from Mesa-prepared
# RData files.  This keeps the original estimator (rsm::SO, OLS then WLS).
if (.Platform$OS.type == "windows" && identical(Sys.getlocale("LC_CTYPE"), "C")) {
  invisible(suppressWarnings(try(
    Sys.setlocale("LC_CTYPE", "Chinese (Simplified)_China.utf8"),
    silent = TRUE
  )))
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop(paste(
    "Usage: Rscript fit_calibration.R <ESS_results.RData> <output_dir>",
    "<simulation_1.RData> <simulation_2.RData> ..."
  ))
}
if (!requireNamespace("rsm", quietly = TRUE)) {
  stop("R package 'rsm' is required to reproduce the original response-surface estimator")
}

ess_path <- args[[1]]
output_dir <- args[[2]]
simulation_paths <- args[3:length(args)]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

simulation_frames <- lapply(simulation_paths, function(path) {
  env <- new.env(parent = emptyenv())
  names_loaded <- load(path, envir = env)
  candidates <- names_loaded[vapply(names_loaded, function(name) is.data.frame(env[[name]]), logical(1))]
  if (length(candidates) != 1) stop("Expected exactly one data frame in ", path)
  env[[candidates[[1]]]]
})
simulation_results <- do.call(rbind, simulation_frames)

ess_env <- new.env(parent = emptyenv())
load(ess_path, envir = ess_env)
if (!exists("ESS_results", envir = ess_env)) stop("ESS_results object not found in ", ess_path)
ESS_results <- ess_env$ESS_results
integrated_cohort <- merge(
  simulation_results, ESS_results,
  by = intersect(names(simulation_results), names(ESS_results))
)

integrated_condition <- aggregate(
  cbind(
    abs(integrated_cohort$perc_hyper_obs - integrated_cohort$perc_hyper_sim),
    abs(integrated_cohort$perc_homo_obs - integrated_cohort$perc_homo_sim),
    abs(integrated_cohort$perc_hypo_obs - integrated_cohort$perc_hypo_sim),
    abs(integrated_cohort$a_diff_obs - integrated_cohort$a_diff_sim)
  ) ~ condition_no + country + delta + ws_f + ws_m + wy_f + wy_m + wa_f + wa_m,
  data = integrated_cohort, sum
)
names(integrated_condition)[(ncol(integrated_condition) - 3):ncol(integrated_condition)] <-
  c("DEV_HYPER", "DEV_HOMO", "DEV_HYPO", "DEV_AGE")
integrated_condition$deviation <- with(
  integrated_condition, (DEV_HYPER * DEV_HOMO * DEV_HYPO * DEV_AGE) ^ 0.25
)
integrated_condition$fit <- -log(integrated_condition$deviation)

model_OLS <- rsm::rsm(
  fit ~ rsm::SO(ws_m, ws_f, delta, wa_m, wa_f, wy_m, wy_f),
  data = integrated_condition
)
e_sq <- residuals(model_OLS) ^ 2
pred_y <- predict(model_OLS)
variance_model <- lm(e_sq ~ pred_y)
v <- predict(variance_model)
model_WLS <- rsm::rsm(
  fit ~ rsm::SO(ws_m, ws_f, delta, wa_m, wa_f, wy_m, wy_f),
  weights = 1 / v, data = integrated_condition
)

write.csv(integrated_condition, file.path(output_dir, "calibration_condition_loss.csv"), row.names = FALSE)
capture.output(summary(model_OLS), file = file.path(output_dir, "response_surface_OLS.txt"))
capture.output(summary(model_WLS), file = file.path(output_dir, "response_surface_WLS.txt"))
save(integrated_cohort, integrated_condition, model_OLS, variance_model, model_WLS,
     file = file.path(output_dir, "mesa_calibration_models.RData"))
cat("Original R OLS/WLS response-surface analysis completed in: ", output_dir, "\n", sep = "")
