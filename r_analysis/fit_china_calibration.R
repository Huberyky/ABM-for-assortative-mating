#!/usr/bin/env Rscript

# Fixed-delta China continuation of Grow & Van Bavel's original calibration:
# run means -> cohort deviations -> geometric loss -> second-order OLS/WLS RSM.
if (.Platform$OS.type == "windows" && identical(Sys.getlocale("LC_CTYPE"), "C")) {
  invisible(suppressWarnings(try(
    Sys.setlocale("LC_CTYPE", "Chinese (Simplified)_China.utf8"), silent = TRUE
  )))
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: Rscript fit_china_calibration.R <runs.csv> <targets.csv> <output_dir>")
}
if (!requireNamespace("rsm", quietly = TRUE)) stop("R package 'rsm' is required")

runs <- read.csv(args[[1]], stringsAsFactors = FALSE, check.names = FALSE)
targets <- read.csv(args[[2]], stringsAsFactors = FALSE, check.names = FALSE)
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
parameters <- c("ws_m", "ws_f", "wa_m", "wa_f", "wy_m", "wy_f")
metrics <- c("perc_hyper_sim", "perc_homo_sim", "perc_hypo_sim", "a_diff_sim")
required <- c("country", "condition_no", "run_no", "ten_cohort", "delta", parameters, metrics)
missing <- setdiff(required, names(runs))
if (length(missing)) stop("calibration runs lack columns: ", paste(missing, collapse = ", "))
if (any(duplicated(runs[c("condition_no", "run_no", "ten_cohort")]))) {
  stop("duplicate condition/run/cohort rows in calibration input")
}
run_counts <- aggregate(run_no ~ condition_no, runs, function(x) length(unique(x)))
if (any(run_counts$run_no != 50)) stop("formal calibration requires exactly 50 runs per condition")
if (length(unique(runs$delta)) != 1 || unique(runs$delta) != 0.9) {
  stop("China calibration requires delta fixed at 0.9")
}

formula_condition <- as.formula(paste(
  "cbind(", paste(metrics, collapse = ","), ") ~ condition_no + country + ten_cohort + delta +",
  paste(parameters, collapse = "+")
))
simulation <- aggregate(formula_condition, data = runs, mean)
integrated_cohort <- merge(simulation, targets, by = c("country", "ten_cohort"))
if (nrow(integrated_cohort) != length(unique(runs$condition_no)) * nrow(targets)) {
  stop("simulation-target merge is incomplete")
}

integrated_cohort$DEV_HYPER <- abs(integrated_cohort$perc_hyper_obs - integrated_cohort$perc_hyper_sim)
integrated_cohort$DEV_HOMO <- abs(integrated_cohort$perc_homo_obs - integrated_cohort$perc_homo_sim)
integrated_cohort$DEV_HYPO <- abs(integrated_cohort$perc_hypo_obs - integrated_cohort$perc_hypo_sim)
integrated_cohort$DEV_AGE <- abs(integrated_cohort$a_diff_obs - integrated_cohort$a_diff_sim)
loss_formula <- as.formula(paste(
  "cbind(DEV_HYPER,DEV_HOMO,DEV_HYPO,DEV_AGE) ~ condition_no + country + delta +",
  paste(parameters, collapse = "+")
))
integrated_condition <- aggregate(loss_formula, data = integrated_cohort, sum)
integrated_condition$deviation <- with(
  integrated_condition, (DEV_HYPER * DEV_HOMO * DEV_HYPO * DEV_AGE) ^ 0.25
)
if (any(!is.finite(integrated_condition$deviation)) || any(integrated_condition$deviation <= 0)) {
  stop("non-positive or non-finite calibration deviation")
}
integrated_condition$fit <- -log(integrated_condition$deviation)

model_OLS <- rsm::rsm(
  fit ~ rsm::SO(ws_m, ws_f, wa_m, wa_f, wy_m, wy_f),
  data = integrated_condition
)
e_sq <- residuals(model_OLS)^2
pred_y <- predict(model_OLS)
variance_model <- lm(e_sq ~ pred_y)
v <- as.numeric(predict(variance_model))
if (any(!is.finite(v)) || any(v <= 0)) {
  stop("the paper's 1/v WLS rule produced non-positive fitted variances")
}
model_WLS <- rsm::rsm(
  fit ~ rsm::SO(ws_m, ws_f, wa_m, wa_f, wy_m, wy_f),
  weights = 1 / v, data = integrated_condition
)

# The source script selects a point on canonical.path.  Search a dense,
# prespecified distance grid and retain the best in-bound point, then require a
# fresh 50-run validation outside this script.
path <- tryCatch(
  as.data.frame(rsm::canonical.path(model_WLS, dist = seq(-2, 2, by = 0.001))),
  error = function(e) data.frame()
)
if (nrow(path) && length(intersect(parameters, names(path))) == length(parameters)) {
  within <- with(path,
    ws_m >= 0 & ws_m <= 2 & ws_f >= 0 & ws_f <= 2 &
    wy_m >= 0 & wy_m <= 2 & wy_f >= 0 & wy_f <= 2 &
    wa_m >= 0 & wa_m <= 20 & wa_f >= 0 & wa_f <= 20
  )
  path <- path[within, ]
}
lower <- c(ws_m = 0, ws_f = 0, wa_m = 0, wa_f = 0, wy_m = 0, wy_f = 0)
upper <- c(ws_m = 2, ws_f = 2, wa_m = 20, wa_f = 20, wy_m = 2, wy_f = 2)
if (nrow(path)) {
  path$predicted_fit <- as.numeric(predict(model_WLS, newdata = path))
  candidate <- path[which.max(path$predicted_fit), c(parameters, "predicted_fit")]
  selection_method <- "WLS canonical.path dense grid"
} else {
  # China data can place the estimated stationary point/ridge wholly outside
  # the experimental region.  In that case canonical.path has no admissible
  # point.  Maximize the same WLS response surface inside the prespecified CCD
  # bounds; no simulator result or bound is changed after inspection.
  objective <- function(theta) {
    newdata <- as.data.frame(as.list(setNames(theta, parameters)))
    -as.numeric(predict(model_WLS, newdata = newdata))
  }
  starts <- rbind(
    c(ws_m = 1, ws_f = 1, wa_m = 10, wa_f = 10, wy_m = 1, wy_f = 1),
    as.matrix(head(integrated_condition[order(-integrated_condition$fit), parameters], 10))
  )
  solutions <- lapply(seq_len(nrow(starts)), function(i) optim(
    par = starts[i, ], fn = objective, method = "L-BFGS-B", lower = lower, upper = upper
  ))
  objective_values <- vapply(solutions, function(x) x$value, numeric(1))
  best <- solutions[[which.min(objective_values)]]
  candidate <- as.data.frame(as.list(setNames(best$par, parameters)))
  candidate$predicted_fit <- -best$value
  selection_method <- "bounded maximization of the same WLS surface; canonical.path had no in-bound point"
}
candidate$delta <- 0.9
candidate$selection_method <- paste0(selection_method, "; independent validation required")
# L-BFGS-B may return a boundary as -1e-16 or 2+1e-16.  Snap only this
# floating-point tolerance back to the already prespecified experimental box.
for (name in parameters) {
  candidate[[name]] <- pmin(upper[[name]], pmax(lower[[name]], candidate[[name]]))
}
candidate <- candidate[c("ws_f", "ws_m", "wy_f", "wy_m", "wa_f", "wa_m", "delta",
                         "predicted_fit", "selection_method")]

write.csv(integrated_cohort, file.path(output_dir, "china_calibration_cohort_loss.csv"), row.names = FALSE)
write.csv(integrated_condition, file.path(output_dir, "china_calibration_condition_loss.csv"), row.names = FALSE)
write.csv(path, file.path(output_dir, "china_calibration_canonical_path.csv"), row.names = FALSE)
write.csv(candidate, file.path(output_dir, "china_calibrated_parameters.csv"), row.names = FALSE)
capture.output(summary(model_OLS), file = file.path(output_dir, "china_response_surface_OLS.txt"))
capture.output(summary(model_WLS), file = file.path(output_dir, "china_response_surface_WLS.txt"))
save(integrated_cohort, integrated_condition, model_OLS, variance_model, model_WLS, path, candidate,
     file = file.path(output_dir, "china_calibration_models.RData"))
cat("China fixed-delta response-surface calibration complete\n")
print(candidate)
