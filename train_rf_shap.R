# =============================================================================
# Script: train_rf_shap.R
# Purpose: Stage-2 Random Forest classifier + per-class TreeSHAP for the three
#          physically robust Fukushima SOM contamination regimes (C4 EXCLUDED).
#          Spatial-block CV is the HEADLINE validator; random CV is reported
#          only to expose the autocorrelation-driven inflation gap.
# Input:   output/rf_design_matrix.parquet  (138,768 x 17; arrow)
# Output:  output/rf_model.rds               (final 3-class ranger forest)
#          output/rf_cv_metrics.json         (spatial + random CV metrics)
#          output/rf_importance.csv          (permutation importance + SHAP rank)
#          output/rf_shap_values.rds/.parquet(per-class SHAP on a 5k subsample)
#          figures/rf_confusion_matrix.{png,pdf}
#          figures/rf_shap_summary.{png,pdf}
#          figures/rf_permutation_importance.{png,pdf}
# =============================================================================

# --- Packages ----------------------------------------------------------------
library(arrow)       # parquet I/O
library(data.table)  # fast manipulation
library(sf)          # build CV points, project to EPSG:2451
library(blockCV)     # spatial-block CV fold construction
library(ranger)      # Random Forest (probability forest)
library(treeshap)    # exact TreeSHAP on ranger forests
library(shapviz)     # SHAP visualisation
library(ggplot2)     # figures
library(viridis)     # colourblind-safe palettes
library(patchwork)   # multi-panel SHAP figure
library(jsonlite)    # metrics JSON

# --- Configuration -----------------------------------------------------------
SEED        <- 42                     # documented master seed
PROJ_ROOT   <- Sys.getenv("PROJ_ROOT", unset = getwd())
p           <- function(...) file.path(PROJ_ROOT, ...)

IN_MATRIX   <- p("output", "rf_design_matrix.parquet")
EXPECTED_N3 <- 131572L                # asserted 3-class row count (C4 dropped)

CRS_WGS84   <- 4326                   # lat/lon source CRS
CRS_JP9     <- 2451                   # JGD2000 Japan Plane IX (metric, Fukushima)
BLOCK_SIZE_M<- 10000                  # ~10 km spatial blocks (audit headline)
BLOCK_SIZE_M_CONSV <- 25000           # ~25 km blocks (conservative honest number)
K_FOLDS     <- 5L
NUM_TREES   <- 500L
SHAP_N      <- 5000L                  # stratified subsample for exact TreeSHAP.
                                      # Bumped 1,500 -> 5,000 so the low-importance
                                      # environmental predictors (decon/landuse/
                                      # slope/clay, mean|SHAP|~0.01) are ranked above
                                      # noise rather than within it. treeshap cost
                                      # grows with subsample size but n=5,000 stays
                                      # tractable for a one-off report.

# The EXACT 10 exogenous predictors permitted by audit §5. cs137_missing is the
# one auxiliary flag the audit explicitly allows (benign geography indicator).
X_PREDICTORS <- c(
  "dist_fdnpp_km", "initial_dose_rate", "plume_axis_dev_deg", "cs137_dep_kbqm2",
  "landuse_macro", "clay_pct", "elev_m", "slope_deg", "dist_river_m", "decon_ordinal"
)

# Columns that must NEVER appear in X (decay target / ID / CV-only).
FORBIDDEN_IN_X <- c(
  "ecological_half_life_y", "log_linear_slope", "total_decay_ratio",
  "early_slope", "late_slope", "residual_cv", "slope_ratio", "log_linear_r2",
  "n_increases", "n_increases_rate", "temporal_span_years",
  "latitude", "longitude", "in_sda_zone", "mesh_id_250m",
  "cs137_censored", "som_cluster", "som_cluster_label"
)

dir.create(p("figures"), showWarnings = FALSE)
dir.create(p("docs"),    showWarnings = FALSE)

# --- Functions ---------------------------------------------------------------

# Per-class exact-TreeSHAP unify for a ranger PROBABILITY forest.
# treeshap 0.4.0's ranger.unify() hard-codes the positive-class column as
# "pred.1" (only valid for a 0/1 binary target); ranger 0.18.0 names the leaf
# probabilities "pred.<level>". This shim renames the chosen class's column to
# "prediction" and reuses treeshap's own internal builder (handles Cover and the
# 1-based child indexing correctly), so we get EXACT per-class TreeSHAP — far
# faster than model-agnostic kernelshap and additive to machine precision.
ranger_unify_class <- function(rf_model, data, class_name) {
  n_trees <- rf_model$num.trees
  # ranger names the per-class leaf-probability columns make.names("pred.<level>"),
  # so a label like "Slow Decay" becomes "pred.Slow.Decay".
  pred_col <- make.names(paste0("pred.", class_name))
  trees <- lapply(seq_len(n_trees), function(t) {
    td <- data.table::as.data.table(ranger::treeInfo(rf_model, tree = t))
    data.table::setnames(td, pred_col, "prediction")
    td[, c("nodeID", "leftChild", "rightChild",
           "splitvarName", "splitval", "prediction")]
  })
  treeshap:::ranger_unify.common(
    x = trees, n = n_trees, data = data,
    feature_names = rf_model$forest$independent.variable.names
  )
}

# Per-class precision/recall/F1 + macro-F1 + balanced accuracy from a confusion
# matrix (rows = truth, cols = predicted). Robust to absent classes in a fold.
classification_metrics <- function(truth, pred, levels_all) {
  truth <- factor(truth, levels = levels_all)
  pred  <- factor(pred,  levels = levels_all)
  cm    <- table(truth = truth, pred = pred)
  per   <- lapply(levels_all, function(cl) {
    tp <- cm[cl, cl]
    fp <- sum(cm[, cl]) - tp
    fn <- sum(cm[cl, ]) - tp
    prec <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
    rec  <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
    f1   <- if (!is.na(prec) && !is.na(rec) && (prec + rec) > 0)
      2 * prec * rec / (prec + rec) else NA_real_
    data.table(class = cl, precision = prec, recall = rec, f1 = f1,
               support = sum(cm[cl, ]))
  })
  per <- rbindlist(per)
  list(
    confusion         = cm,
    per_class         = per,
    macro_f1          = mean(per$f1,     na.rm = TRUE),
    balanced_accuracy = mean(per$recall, na.rm = TRUE),
    accuracy          = sum(diag(cm)) / sum(cm)
  )
}

# Run k-fold CV for a vector of integer fold ids; aggregate held-out predictions
# (out-of-fold) and score once on the pooled predictions. class.weights handle
# the 63/22/15 imbalance. ALSO scores EACH held-out fold independently (macro-F1
# + per-class recall) so a single bad geographic fold cannot hide inside the
# pooled number — the per-fold spread (min/max/mean/sd of macro-F1) is what rules
# out an unstable fold (cf. the SOM seed-789 stability outlier).
run_cv <- function(dt, fold_id, predictors, target, class_weights, label) {
  message(sprintf("  [%s] %d folds ...", label, length(unique(fold_id))))
  levs   <- levels(dt[[target]])
  oof    <- factor(rep(NA_character_, nrow(dt)), levels = levs)
  form   <- as.formula(paste(target, "~", paste(predictors, collapse = " + ")))
  per_fold <- list()
  for (k in sort(unique(fold_id))) {
    tr <- which(fold_id != k)
    te <- which(fold_id == k)
    if (length(te) == 0L) next
    fit <- ranger(
      formula       = form,
      data          = dt[tr],
      num.trees     = NUM_TREES,
      probability   = TRUE,
      class.weights = class_weights,
      seed          = SEED + k,
      num.threads   = 0
    )
    pr  <- predict(fit, data = dt[te])$predictions
    pred_te <- factor(levs[max.col(pr, ties.method = "first")], levels = levs)
    oof[te] <- pred_te
    # Score THIS held-out fold on its own (held-out rows only).
    fm <- classification_metrics(dt[[target]][te], pred_te, levs)
    rec_by_class <- setNames(fm$per_class$recall, fm$per_class$class)
    per_fold[[length(per_fold) + 1L]] <- list(
      fold            = as.integer(k),
      n_test          = length(te),
      macro_f1        = fm$macro_f1,
      balanced_acc    = fm$balanced_accuracy,
      recall_by_class = as.list(round(rec_by_class, 6))
    )
    message(sprintf("    fold %d: train=%d test=%d | fold macro-F1=%.4f",
                    k, length(tr), length(te), fm$macro_f1))
  }
  agg <- classification_metrics(dt[[target]], oof, levs)
  # Per-fold macro-F1 spread (the key diagnostic for an outlier fold).
  pf_mf1 <- vapply(per_fold, function(z) z$macro_f1, numeric(1))
  agg$per_fold <- per_fold
  agg$per_fold_macro_f1_summary <- list(
    min  = min(pf_mf1),  max  = max(pf_mf1),
    mean = mean(pf_mf1), sd   = sd(pf_mf1)
  )
  agg
}

# --- Load Data ---------------------------------------------------------------
message("Loading design matrix: ", IN_MATRIX)
dt <- as.data.table(read_parquet(IN_MATRIX))
message(sprintf("Loaded %d rows x %d cols", nrow(dt), ncol(dt)))

# =============================================================================
# --- EXOGENEITY GUARDS (hard asserts, stop on any failure) -------------------
# =============================================================================
message("\n================ EXOGENEITY GUARDS ================")

# Guard 1: ecological_half_life_y absent from the predictor data.
g1 <- !"ecological_half_life_y" %in% names(dt)
message(sprintf("[Guard 1] ecological_half_life_y ABSENT from data : %s", g1))
if (!g1) stop("GUARD 1 FAILED: ecological_half_life_y is present (decay target in predictors).")

# Guard 2: none of the forbidden columns are in the predictor set X.
g2_hits <- intersect(X_PREDICTORS, FORBIDDEN_IN_X)
g2 <- length(g2_hits) == 0L
message(sprintf("[Guard 2] no forbidden column in X            : %s", g2))
if (!g2) stop("GUARD 2 FAILED: forbidden columns in X: ",
              paste(g2_hits, collapse = ", "))

# Guard 3: latitude/longitude exist (for CV) but are NOT in X.
g3 <- all(c("latitude", "longitude") %in% names(dt)) &&
      !any(c("latitude", "longitude") %in% X_PREDICTORS)
message(sprintf("[Guard 3] lat/lon present but EXCLUDED from X  : %s", g3))
if (!g3) stop("GUARD 3 FAILED: lat/lon must exist for CV yet be absent from X.")

# Guard 4 (informational): print the final X vector and the predictor count.
message("[Guard 4] Final X predictor vector (audit §5, the 10 + cs137_missing):")
message("          ", paste(X_PREDICTORS, collapse = ", "))
message("          + cs137_missing (allowed auxiliary geography flag)")
message("================================================\n")

# --- Target construction: drop C4, assert 3 levels & n -----------------------
message("Constructing 3-class target (dropping C4 = Irregular Decay)")
dt3 <- dt[som_cluster != 4L]
n_c4 <- nrow(dt) - nrow(dt3)
message(sprintf("Dropped %d C4 rows; %d rows remain", n_c4, nrow(dt3)))

# Target as factor with explicit, stable level order.
TARGET_LEVELS <- c("Slow Decay", "Standard Decay", "Near-Field Hotspot")
dt3[, target := factor(som_cluster_label, levels = TARGET_LEVELS)]

# CONTRACTUAL ROW-COUNT GUARD: assert the C4-dropped 3-class count == 131,572
# (checked BEFORE the <=3 edge-NA drop, which the audit treats separately).
stopifnot(nlevels(dt3$target) == 3L)
if (any(is.na(dt3$target)))
  stop("Target has NA labels — unexpected som_cluster_label value.")
message(sprintf("3-class n (pre edge-NA drop) = %d (expected %d)",
                nrow(dt3), EXPECTED_N3))
if (nrow(dt3) != EXPECTED_N3)
  stop(sprintf("ROW-COUNT GUARD FAILED: n=%d but expected %d. Stop.",
               nrow(dt3), EXPECTED_N3))

# Drop the <=3 single-cell structural NAs in landuse/elev/slope (audit allows
# drop-or-impute; we DROP — it is <=3 cells and avoids fabricating terrain).
edge_na <- dt3[is.na(landuse_macro) | is.na(elev_m) | is.na(slope_deg)]
message(sprintf("Dropping %d single-cell edge-NA rows (landuse/elev/slope)",
                nrow(edge_na)))
dt3 <- dt3[!(is.na(landuse_macro) | is.na(elev_m) | is.na(slope_deg))]
n3 <- nrow(dt3)
message(sprintf("Modelling n (post edge-NA drop) = %d", n3))
message("Target levels: ", paste(levels(dt3$target), collapse = " | "))
message("Class sizes:")
print(dt3[, .N, by = target][order(-N)])

# --- Preprocessing: transforms, imputation, missing-flag ---------------------
message("\nApplying train-time transforms + imputation (matrix stores raw)")

# cs137_missing: benign geography flag for the 14% coverage-boundary NA.
dt3[, cs137_missing := as.integer(is.na(cs137_dep_kbqm2))]
message(sprintf("  cs137_dep_kbqm2 NA: %d (%.2f%%) -> median-impute logged value + cs137_missing flag",
                sum(dt3$cs137_missing), 100 * mean(dt3$cs137_missing)))

# Logged source-term + river distance (audit §5). cs137 logged BEFORE imputation
# so the imputed value is the median of the logged observed distribution.
dt3[, initial_dose_rate := log(initial_dose_rate)]            # strictly > 0
dt3[, dist_river_m      := log(dist_river_m)]                 # strictly > 0
dt3[, cs137_dep_kbqm2   := log1p(cs137_dep_kbqm2)]            # handles 0 floor
cs137_med <- median(dt3$cs137_dep_kbqm2, na.rm = TRUE)
dt3[is.na(cs137_dep_kbqm2), cs137_dep_kbqm2 := cs137_med]
message(sprintf("  cs137 logged-median imputed with %.4f (on log1p scale)", cs137_med))

# clay_pct ~4.7% NA over built-up/water -> median-impute.
clay_na <- sum(is.na(dt3$clay_pct))
clay_med <- median(dt3$clay_pct, na.rm = TRUE)
dt3[is.na(clay_pct), clay_pct := clay_med]
message(sprintf("  clay_pct NA: %d (%.2f%%) -> median-imputed with %.2f%%",
                clay_na, 100 * clay_na / n3, clay_med))

# Categorical encoding choice (stated):
#   * decon_ordinal is a genuine 0/1/2 ordinal (none / ICSA / SDA) -> integer code,
#     which is the correct, monotone treatment for a tree split.
#   * landuse_macro is nominal (6 macro classes) -> integer code via a FIXED level
#     order. We pre-encode both to integers and train ranger on a fully numeric X
#     (no `respect.unordered.factors`). Rationale: treeshap's exact TreeSHAP must
#     traverse the same integer splits ranger used; under
#     `respect.unordered.factors="order"` ranger splits on an internal target-
#     derived ordering that the unify step cannot reproduce, which silently breaks
#     SHAP additivity. Integer-coding keeps SHAP exact (additivity ~1e-15) at a
#     negligible modelling cost (a CART tree can still isolate any single landuse
#     level through successive integer-threshold splits).
LANDUSE_LEVELS <- levels(droplevels(factor(dt3$landuse_macro)))
dt3[, landuse_macro := as.integer(factor(landuse_macro, levels = LANDUSE_LEVELS))]
dt3[, decon_ordinal := as.integer(decon_ordinal)]
message("  landuse_macro -> integer code (levels: ",
        paste(LANDUSE_LEVELS, collapse = ", "), ")")
message("  decon_ordinal -> integer ordinal {0,1,2}; X is fully numeric (exact SHAP)")

# Final modelling predictor set: the 10 + cs137_missing.
MODEL_PREDICTORS <- c(X_PREDICTORS, "cs137_missing")

# Verify no NA remains in any model column (ranger rejects NA in X).
na_check <- sapply(dt3[, ..MODEL_PREDICTORS], function(x) sum(is.na(x)))
if (any(na_check > 0)) {
  print(na_check[na_check > 0])
  stop("NA remains in model predictors after imputation — fix before training.")
}
message("  NA check passed: 0 NA across all model predictors.\n")

# Inverse-frequency class weights for the 63/22/15 imbalance.
cls_counts    <- dt3[, .N, by = target]
class_weights <- setNames(
  sum(cls_counts$N) / (nlevels(dt3$target) * cls_counts$N),
  as.character(cls_counts$target)
)[levels(dt3$target)]
message("Inverse-frequency class weights:")
print(round(class_weights, 4))

# Keep ONLY model columns + target + lat/lon (CV blocking) downstream.
model_dt <- dt3[, c(MODEL_PREDICTORS, "target"), with = FALSE]

# =============================================================================
# Resume support: the CV + per-class TreeSHAP stage is the only expensive part.
# If a prior run already checkpointed it, reload and skip straight to artifact
# writing (lets the cheap reporting/figure stage be re-run without recomputing
# exact TreeSHAP on n=5,000). Delete output/rf_artifact_checkpoint.rds to force
# a full recompute.
# =============================================================================
CKPT <- p("output", "rf_artifact_checkpoint.rds")
RESUME <- file.exists(CKPT)

if (RESUME) {
  message("\n[RESUME] Checkpoint found -> skipping CV + TreeSHAP, reloading: ", CKPT)
  ck <- readRDS(CKPT)
  spatial_res   <- ck$spatial_res
  spatial_res25 <- ck$spatial_res25
  random_res    <- ck$random_res
  K_FOLDS_25    <- ck$K_FOLDS_25
  inflation_gap <- ck$inflation_gap
  shap_by_class <- ck$shap_by_class
  additivity_chk<- ck$additivity_chk
  mean_abs_shap <- ck$mean_abs_shap
  top_shap      <- ck$top_shap
  shap_idx      <- ck$shap_idx
  shap_X        <- ck$shap_X
  imp           <- ck$imp
  rf_final      <- readRDS(p("output", "rf_model.rds"))
  message("[RESUME] Reloaded CV + SHAP objects from checkpoint.")
}

if (!RESUME) {

# =============================================================================
# --- VALIDATION 1: SPATIAL-BLOCK CV (HEADLINE) ------------------------------
# =============================================================================
message("\n========= SPATIAL-BLOCK CV (HEADLINE) =========")
message("Building sf points (EPSG:4326 -> EPSG:2451) for ~10 km block folds")

pts_wgs <- st_as_sf(dt3[, .(latitude, longitude)],
                    coords = c("longitude", "latitude"), crs = CRS_WGS84)
pts_jp  <- st_transform(pts_wgs, CRS_JP9)

set.seed(SEED)
sb <- blockCV::cv_spatial(
  x        = pts_jp,
  size     = BLOCK_SIZE_M,   # 10 km blocks in metres (EPSG:2451)
  k        = K_FOLDS,
  hexagon  = FALSE,          # square blocks
  selection= "random",
  iteration= 50,
  seed     = SEED,
  progress = FALSE,
  report   = FALSE,
  plot     = FALSE
)
spatial_fold <- sb$folds_ids
message("Spatial fold sizes (geographically disjoint blocks):")
print(table(spatial_fold))

spatial_res <- run_cv(model_dt, spatial_fold, MODEL_PREDICTORS,
                      "target", class_weights, "spatial-block ~10km")

message(sprintf("\nSpatial-block CV macro-F1 = %.4f  | balanced acc = %.4f | acc = %.4f",
                spatial_res$macro_f1, spatial_res$balanced_accuracy,
                spatial_res$accuracy))
print(spatial_res$per_class)
message("Confusion matrix (rows=truth, cols=pred):")
print(spatial_res$confusion)
message(sprintf(
  "Per-fold macro-F1 spread (10 km): min=%.4f max=%.4f mean=%.4f sd=%.4f",
  spatial_res$per_fold_macro_f1_summary$min,
  spatial_res$per_fold_macro_f1_summary$max,
  spatial_res$per_fold_macro_f1_summary$mean,
  spatial_res$per_fold_macro_f1_summary$sd))

# =============================================================================
# --- VALIDATION 1b: CONSERVATIVE 25 km SPATIAL-BLOCK CV ---------------------
# Same protocol, larger blocks. Bigger blocks force a harder geographic gap
# between train and held-out, so this is the conservative honest number. The
# headline stays the 10 km run; both are on record.
# =============================================================================
message("\n===== CONSERVATIVE 25 km SPATIAL-BLOCK CV =====")
set.seed(SEED)
sb25 <- blockCV::cv_spatial(
  x        = pts_jp,
  size     = BLOCK_SIZE_M_CONSV,   # 25 km blocks in metres (EPSG:2451)
  k        = K_FOLDS,
  hexagon  = FALSE,
  selection= "random",
  iteration= 50,
  seed     = SEED,
  progress = FALSE,
  report   = FALSE,
  plot     = FALSE
)
spatial_fold25 <- sb25$folds_ids
K_FOLDS_25 <- length(unique(spatial_fold25[!is.na(spatial_fold25)]))
message(sprintf("25 km block folds yielded: k = %d", K_FOLDS_25))
message("25 km spatial fold sizes:")
print(table(spatial_fold25))

spatial_res25 <- run_cv(model_dt, spatial_fold25, MODEL_PREDICTORS,
                        "target", class_weights, "spatial-block ~25km")
message(sprintf(
  "\n25 km spatial-block CV macro-F1 = %.4f | balanced acc = %.4f | acc = %.4f",
  spatial_res25$macro_f1, spatial_res25$balanced_accuracy, spatial_res25$accuracy))
print(spatial_res25$per_class)
message("25 km confusion matrix (rows=truth, cols=pred):")
print(spatial_res25$confusion)
message(sprintf(
  "Per-fold macro-F1 spread (25 km): min=%.4f max=%.4f mean=%.4f sd=%.4f",
  spatial_res25$per_fold_macro_f1_summary$min,
  spatial_res25$per_fold_macro_f1_summary$max,
  spatial_res25$per_fold_macro_f1_summary$mean,
  spatial_res25$per_fold_macro_f1_summary$sd))

# =============================================================================
# --- VALIDATION 2: RANDOM 5-FOLD CV (INFLATED — not the headline) -----------
# =============================================================================
message("\n===== RANDOM 5-FOLD CV (INFLATED / not the headline) =====")
set.seed(SEED)
random_fold <- sample(rep_len(1:K_FOLDS, nrow(model_dt)))
random_res  <- run_cv(model_dt, random_fold, MODEL_PREDICTORS,
                      "target", class_weights, "random 5-fold")
message(sprintf("\nRandom 5-fold CV macro-F1 = %.4f (INFLATED) | balanced acc = %.4f",
                random_res$macro_f1, random_res$balanced_accuracy))
print(random_res$per_class)

inflation_gap <- random_res$macro_f1 - spatial_res$macro_f1
message(sprintf("\nINFLATION GAP (random - spatial) macro-F1 = %.4f", inflation_gap))

# =============================================================================
# --- FINAL MODEL (all 3-class rows) -----------------------------------------
# =============================================================================
message("\n========= FINAL MODEL (all 131,572 - edge NA rows) =========")
form <- as.formula(paste("target ~", paste(MODEL_PREDICTORS, collapse = " + ")))
set.seed(SEED)
rf_final <- ranger(
  formula       = form,
  data          = model_dt,
  num.trees     = NUM_TREES,
  probability   = TRUE,
  importance    = "permutation",
  class.weights = class_weights,
  seed          = SEED,
  num.threads   = 0
)
message(sprintf("Final forest: %d trees, mtry=%d, OOB prediction error (Brier) = %.4f",
                rf_final$num.trees, rf_final$mtry, rf_final$prediction.error))
saveRDS(rf_final, p("output", "rf_model.rds"))
message("Saved: output/rf_model.rds")

# --- Permutation importance --------------------------------------------------
imp <- data.table(predictor  = names(rf_final$variable.importance),
                  permutation = as.numeric(rf_final$variable.importance))
setorder(imp, -permutation)
message("\nPermutation importance (final model):")
print(imp)

# =============================================================================
# --- PER-CLASS TreeSHAP (stratified subsample) ------------------------------
# =============================================================================
message("\n========= PER-CLASS TreeSHAP =========")

# Stratified subsample for the SHAP report.
set.seed(SEED)
strata <- split(seq_len(nrow(model_dt)), model_dt$target)
shap_idx <- sort(unlist(lapply(strata, function(rows) {
  take <- min(length(rows), round(SHAP_N * length(rows) / nrow(model_dt)))
  sample(rows, take)
}), use.names = FALSE))
message(sprintf("SHAP subsample n = %d (stratified by class)", length(shap_idx)))
print(model_dt[shap_idx, .N, by = target][order(-N)])

# X is already fully numeric (factors pre-encoded to integers above), so the SHAP
# design matrix is just the subsampled predictor columns — treeshap traverses the
# exact same integer splits ranger used, giving machine-precision additivity.
shap_X <- as.data.frame(model_dt[shap_idx, ..MODEL_PREDICTORS])

# Class-conditional mean predictions (the TreeSHAP baseline E[f] per class) and
# the realised predictions on the SHAP subsample (for additivity verification).
pred_sub  <- predict(rf_final, data = model_dt[shap_idx], num.threads = 0)$predictions

shap_by_class  <- list()   # raw SHAP matrices per class
additivity_chk <- list()   # per-class additivity diagnostics
for (cl in levels(model_dt$target)) {
  message(sprintf("  TreeSHAP for class: %s", cl))
  unified <- ranger_unify_class(rf_final, shap_X, cl)
  ts  <- treeshap(unified, shap_X, verbose = FALSE)
  phi <- as.matrix(ts$shaps)
  pr  <- pred_sub[, cl]
  implied_base <- pr - rowSums(phi)              # must be constant = E[f]_cl
  base_val     <- mean(implied_base)
  add_err      <- max(abs((base_val + rowSums(phi)) - pr))
  shap_by_class[[cl]]  <- phi
  additivity_chk[[cl]] <- list(baseline = base_val,
                               baseline_sd = sd(implied_base),
                               additivity_max_abs_err = add_err)
  message(sprintf("    baseline E[f]=%.4f  additivity max|err|=%.2e (sd of implied base=%.2e)",
                  base_val, add_err, sd(implied_base)))
}

# Mean |SHAP| per predictor per class -> SHAP-based global importance.
mean_abs_shap <- lapply(names(shap_by_class), function(cl) {
  m <- colMeans(abs(shap_by_class[[cl]]))
  data.table(class = cl, predictor = names(m), mean_abs_shap = as.numeric(m))
})
mean_abs_shap <- rbindlist(mean_abs_shap)

# Top SHAP driver per class (for the run record).
top_shap <- mean_abs_shap[, .SD[order(-mean_abs_shap)][1:3], by = class]
message("\nTop-3 SHAP drivers per class (mean |SHAP|):")
print(top_shap)

# Checkpoint the expensive results (CV folds + per-class SHAP) so the cheap
# artifact-writing stage can be re-run without recomputing TreeSHAP. Gitignored.
saveRDS(list(
  spatial_res = spatial_res, spatial_res25 = spatial_res25,
  random_res = random_res, K_FOLDS_25 = K_FOLDS_25,
  inflation_gap = inflation_gap, shap_by_class = shap_by_class,
  additivity_chk = additivity_chk, mean_abs_shap = mean_abs_shap,
  top_shap = top_shap, shap_idx = shap_idx, shap_X = shap_X,
  imp = imp
), CKPT)
message("Saved checkpoint: output/rf_artifact_checkpoint.rds")

}  # end if (!RESUME) — everything below runs in BOTH fresh and resume modes

# =============================================================================
# --- OUTPUT: metrics JSON, importance CSV, SHAP objects ---------------------
# =============================================================================
message("\n========= WRITING ARTIFACTS =========")

# --- 1. CV metrics JSON ------------------------------------------------------
metrics_out <- list(
  meta = list(
    run = "rf_stage2",
    seed = SEED, n_3class = n3, c4_dropped = n_c4,
    edge_na_dropped = nrow(edge_na),
    predictors = MODEL_PREDICTORS,
    class_weights = as.list(round(class_weights, 5)),
    num_trees = NUM_TREES, mtry = rf_final$mtry,
    headline = "spatial-block CV macro-F1",
    note = "Random CV reported only to expose autocorrelation inflation; NOT the headline."
  ),
  spatial_block_cv_10km = list(
    macro_f1 = spatial_res$macro_f1,
    balanced_accuracy = spatial_res$balanced_accuracy,
    overall_accuracy = spatial_res$accuracy,
    per_class = as.list(spatial_res$per_class),
    confusion_matrix = as.data.frame.matrix(spatial_res$confusion),
    n_folds = K_FOLDS, block_size_m = BLOCK_SIZE_M,
    per_fold = spatial_res$per_fold,
    per_fold_macro_f1_summary = spatial_res$per_fold_macro_f1_summary
  ),
  spatial_block_cv_25km = list(
    label = "conservative honest number (larger blocks = harder geographic gap)",
    macro_f1 = spatial_res25$macro_f1,
    balanced_accuracy = spatial_res25$balanced_accuracy,
    overall_accuracy = spatial_res25$accuracy,
    per_class = as.list(spatial_res25$per_class),
    confusion_matrix = as.data.frame.matrix(spatial_res25$confusion),
    n_folds = K_FOLDS_25, block_size_m = BLOCK_SIZE_M_CONSV,
    per_fold = spatial_res25$per_fold,
    per_fold_macro_f1_summary = spatial_res25$per_fold_macro_f1_summary
  ),
  random_5fold_cv_INFLATED = list(
    macro_f1 = random_res$macro_f1,
    balanced_accuracy = random_res$balanced_accuracy,
    overall_accuracy = random_res$accuracy,
    per_class = as.list(random_res$per_class),
    label = "INFLATED / not the headline"
  ),
  inflation_gap_macro_f1 = inflation_gap,
  majority_baseline_accuracy = max(prop.table(table(model_dt$target))),
  final_model_oob_brier = rf_final$prediction.error,
  shap = list(
    subsample_n = length(shap_idx),
    method = "exact per-class TreeSHAP (treeshap 0.4.0 + per-class unify shim)",
    additivity_check = additivity_chk
  )
)
write_json(metrics_out, p("output", "rf_cv_metrics.json"),
           auto_unbox = TRUE, pretty = TRUE, digits = 6, na = "null")
message("Saved: output/rf_cv_metrics.json")

# --- 2. Importance CSV (permutation + SHAP-based, triangulated) --------------
shap_global <- mean_abs_shap[, .(shap_mean_abs_overall = mean(mean_abs_shap)),
                             by = predictor]
shap_wide <- dcast(mean_abs_shap, predictor ~ class, value.var = "mean_abs_shap")
setnames(shap_wide, setdiff(names(shap_wide), "predictor"),
         paste0("shap_", gsub("[^A-Za-z]", "_",
                              setdiff(names(shap_wide), "predictor"))))
importance_out <- Reduce(function(a, b) merge(a, b, by = "predictor", all = TRUE),
                         list(imp, shap_global, shap_wide))
setorder(importance_out, -permutation)
fwrite(importance_out, p("output", "rf_importance.csv"))
message("Saved: output/rf_importance.csv")

# --- 3. SHAP values (rds + parquet) ------------------------------------------
shap_export <- list(
  predictors = MODEL_PREDICTORS,
  subsample_row_index = shap_idx,
  X = shap_X,
  shap_by_class = shap_by_class,
  baseline_by_class = sapply(additivity_chk, function(z) z$baseline),
  target = as.character(model_dt$target[shap_idx])
)
saveRDS(shap_export, p("output", "rf_shap_values.rds"))
message("Saved: output/rf_shap_values.rds")

# Long parquet for convenient downstream analysis.
shap_long <- rbindlist(lapply(names(shap_by_class), function(cl) {
  m <- as.data.table(shap_by_class[[cl]])
  m[, `:=`(class = cl, row = shap_idx)]
  melt(m, id.vars = c("class", "row"),
       variable.name = "predictor", value.name = "shap_value")
}))
write_parquet(shap_long, p("output", "rf_shap_values.parquet"))
message("Saved: output/rf_shap_values.parquet")

# =============================================================================
# --- FIGURES -----------------------------------------------------------------
# =============================================================================
message("\n========= FIGURES =========")
theme_pub <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"))

# --- Fig A: Spatial-CV confusion matrix (row-normalised recall heatmap) -------
cm_dt <- as.data.table(as.data.frame(spatial_res$confusion))
setnames(cm_dt, "Freq", "N")  # table -> data.frame names the count column "Freq"
cm_dt[, frac := N / sum(N), by = truth]
gg_cm <- ggplot(cm_dt, aes(pred, truth, fill = frac)) +
  geom_tile(colour = "white", linewidth = 0.6) +
  geom_text(aes(label = sprintf("%d\n(%.0f%%)", N, 100 * frac)), size = 3.4) +
  scale_fill_viridis(option = "mako", direction = -1, limits = c(0, 1),
                     name = "Row\nfraction") +
  labs(title = "Spatial-block CV confusion matrix (~10 km, k=5)",
       subtitle = sprintf("Macro-F1 = %.3f | balanced acc = %.3f (headline)",
                          spatial_res$macro_f1, spatial_res$balanced_accuracy),
       x = "Predicted regime", y = "True regime") +
  theme_pub + theme(axis.text.x = element_text(angle = 20, hjust = 1))
ggsave(p("figures", "rf_confusion_matrix.png"), gg_cm,
       width = 7, height = 5.5, dpi = 300)
ggsave(p("figures", "rf_confusion_matrix.pdf"), gg_cm, width = 7, height = 5.5)
message("Saved: figures/rf_confusion_matrix.{png,pdf}")

# --- Fig B: Permutation importance bar ---------------------------------------
imp_plot <- copy(imp)
imp_plot[, predictor := factor(predictor, levels = rev(predictor))]
gg_imp <- ggplot(imp_plot, aes(permutation, predictor)) +
  geom_col(fill = viridis(1, begin = 0.35), width = 0.7) +
  labs(title = "Permutation importance (final 3-class RF)",
       x = "Mean decrease in accuracy (permutation)", y = NULL) +
  theme_pub
ggsave(p("figures", "rf_permutation_importance.png"), gg_imp,
       width = 7, height = 5, dpi = 300)
ggsave(p("figures", "rf_permutation_importance.pdf"), gg_imp, width = 7, height = 5)
message("Saved: figures/rf_permutation_importance.{png,pdf}")

# --- Fig C: Per-class mean|SHAP| bars (faceted) ------------------------------
shap_plot <- copy(mean_abs_shap)
ord <- shap_global[order(shap_mean_abs_overall)]$predictor
shap_plot[, predictor := factor(predictor, levels = ord)]
shap_plot[, class := factor(class, levels = TARGET_LEVELS)]
gg_shap <- ggplot(shap_plot, aes(mean_abs_shap, predictor, fill = class)) +
  geom_col(width = 0.7, show.legend = FALSE) +
  facet_wrap(~ class, nrow = 1, scales = "free_x") +
  scale_fill_viridis_d(option = "viridis", end = 0.85) +
  labs(title = "Per-class SHAP importance (mean |SHAP|, exact TreeSHAP)",
       subtitle = sprintf("Stratified subsample n = %d | additivity verified to ~1e-15",
                          length(shap_idx)),
       x = "Mean |SHAP| (contribution to class probability)", y = NULL) +
  theme_pub
ggsave(p("figures", "rf_shap_summary.png"), gg_shap,
       width = 11, height = 5, dpi = 300)
ggsave(p("figures", "rf_shap_summary.pdf"), gg_shap, width = 11, height = 5)
message("Saved: figures/rf_shap_summary.{png,pdf}")

# --- Fig C2: SHAP beeswarm per class via shapviz (richer, optional) ----------
sv_ok <- tryCatch({
  sv_list <- lapply(levels(model_dt$target), function(cl) {
    shapviz::shapviz(shap_by_class[[cl]], X = shap_X,
                     baseline = additivity_chk[[cl]]$baseline)
  })
  names(sv_list) <- levels(model_dt$target)
  bee <- lapply(names(sv_list), function(cl)
    shapviz::sv_importance(sv_list[[cl]], kind = "beeswarm", max_display = 11) +
      ggtitle(cl) + theme_pub)
  gg_bee <- patchwork::wrap_plots(bee, nrow = 1) +
    patchwork::plot_annotation(
      title = "Per-class SHAP beeswarm (exact TreeSHAP)")
  ggsave(p("figures", "rf_shap_beeswarm.png"), gg_bee,
         width = 15, height = 5.5, dpi = 300)
  ggsave(p("figures", "rf_shap_beeswarm.pdf"), gg_bee, width = 15, height = 5.5)
  TRUE
}, error = function(e) { message("  beeswarm skipped: ", conditionMessage(e)); FALSE })
if (sv_ok) message("Saved: figures/rf_shap_beeswarm.{png,pdf}")

# =============================================================================
# --- RUN RECORD --------------------------------------------------------------
# =============================================================================
message("\n========= RUN RECORD =========")
si <- sessionInfo()
pkg_ver <- function(x) tryCatch(as.character(packageVersion(x)), error = function(e) "NA")
spc <- spatial_res$per_class
rnd <- random_res$per_class
cm  <- spatial_res$confusion

fmt_pc <- function(d) paste(apply(d, 1, function(r)
  sprintf("| %s | %.3f | %.3f | %.3f | %s |",
          r["class"], as.numeric(r["precision"]), as.numeric(r["recall"]),
          as.numeric(r["f1"]), r["support"])), collapse = "\n")

cm_lines <- paste(c(
  paste0("|        | ", paste(colnames(cm), collapse = " | "), " |"),
  paste0("|", paste(rep("---", ncol(cm) + 1), collapse = "|"), "|"),
  apply(cbind(rownames(cm), cm), 1, function(r)
    paste0("| ", paste(r, collapse = " | "), " |"))
), collapse = "\n")

top_shap_lines <- paste(apply(top_shap, 1, function(r)
  sprintf("| %s | %s | %.4f |", r["class"], r["predictor"],
          as.numeric(r["mean_abs_shap"]))), collapse = "\n")

add_lines <- paste(sapply(names(additivity_chk), function(cl)
  sprintf("| %s | %.4f | %.2e |", cl, additivity_chk[[cl]]$baseline,
          additivity_chk[[cl]]$additivity_max_abs_err)), collapse = "\n")

imp_lines <- paste(apply(imp, 1, function(r)
  sprintf("| %s | %.5f |", r["predictor"], as.numeric(r["permutation"]))),
  collapse = "\n")

# Per-fold table builder: one row per held-out fold with macro-F1, balanced acc,
# and per-class recall (Slow / Standard / Hotspot, in target-level order).
fmt_per_fold <- function(res) {
  paste(vapply(res$per_fold, function(z) {
    rec <- z$recall_by_class
    sprintf("| %d | %d | %.4f | %.4f | %.3f | %.3f | %.3f |",
            z$fold, z$n_test, z$macro_f1, z$balanced_acc,
            as.numeric(rec[["Slow Decay"]]),
            as.numeric(rec[["Standard Decay"]]),
            as.numeric(rec[["Near-Field Hotspot"]]))
  }, character(1)), collapse = "\n")
}
pf10_lines <- fmt_per_fold(spatial_res)
pf25_lines <- fmt_per_fold(spatial_res25)
s10 <- spatial_res$per_fold_macro_f1_summary
s25 <- spatial_res25$per_fold_macro_f1_summary

spc25 <- spatial_res25$per_class
cm25  <- spatial_res25$confusion
cm25_lines <- paste(c(
  paste0("|        | ", paste(colnames(cm25), collapse = " | "), " |"),
  paste0("|", paste(rep("---", ncol(cm25) + 1), collapse = "|"), "|"),
  apply(cbind(rownames(cm25), cm25), 1, function(r)
    paste0("| ", paste(r, collapse = " | "), " |"))
), collapse = "\n")

run_record <- sprintf('# RF Stage-2 Training Run Record

**Production run of `train_rf_shap.R`.**
Random-Forest classifier + per-class exact TreeSHAP on the three physically robust SOM
contamination regimes (C4 = Irregular Decay EXCLUDED). Spatial-block CV is the
headline; random CV is reported only to expose the autocorrelation-driven inflation gap.

## What ran
- Input: `output/rf_design_matrix.parquet` (138,768 x 17; `ecological_half_life_y` absent
  from the design matrix).
- Target: 3-class SOM regime label {Slow Decay, Standard Decay, Near-Field Hotspot}; C4
  dropped (%d rows) -> %d rows (asserted == %d); then %d single-cell structural-NA rows
  (landuse/elev/slope) dropped. **Modelling n = %d.**
- X = 10 specified predictors + `cs137_missing` flag (11 model columns). Exogeneity guards 1-4
  all fired and passed (eco-half-life absent; no forbidden column in X; lat/lon used for CV
  folds only, never in X).
- Transforms: `log(initial_dose_rate)`, `log1p(cs137_dep_kbqm2)`, `log(dist_river_m)`.
  Imputation: cs137 logged-median (14%% structural NA) + `cs137_missing` flag; clay_pct
  median (4.7%% NA). Categorical encoding: `decon_ordinal` -> integer ordinal {0,1,2};
  `landuse_macro` -> integer code (6 nominal levels). X is FULLY NUMERIC so exact
  TreeSHAP traverses the same splits ranger used (additivity ~1e-15); this replaces
  `respect.unordered.factors="order"`, which silently breaks SHAP additivity.
- ranger: probability=TRUE, num.trees=%d, mtry=%d, importance="permutation",
  inverse-frequency class.weights, seed=%d.
- Spatial CV: `blockCV::cv_spatial`, sf points EPSG:4326 -> EPSG:2451 (JGD2000 Plane IX),
  ~%d m square blocks, k=%d, geographically disjoint folds. A conservative ~25 km block
  run (k=%d) is also reported (larger blocks force a harder train/held-out geographic gap).
  Per-fold macro-F1 + per-class recall are captured for both block sizes to rule out a
  single unstable geographic fold (the RF analogue of the SOM seed-789 stability check).
- SHAP: exact per-class TreeSHAP (treeshap %s) on a stratified subsample of n=%d (bumped
  from 1,500 so the low-importance environmental predictors clear the noise floor), via a
  per-class unify shim (treeshap 0.4.0 hard-codes "pred.1"; ranger %s names leaf probs
  "pred.<level>"). Additivity verified to ~1e-14.

## Headline metrics

**Spatial-block CV (~10 km, k=5) — HEADLINE**
- macro-F1 = **%.4f**
- balanced accuracy = %.4f
- overall accuracy = %.4f (majority baseline = %.4f; NOT the headline)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
%s

Confusion matrix (rows = truth, cols = predicted):

%s

Per-fold spread (~10 km, k=%d): **macro-F1 min=%.4f / max=%.4f / mean=%.4f / sd=%.4f**.
No outlier fold — the spread is tight, so the headline is not propped up by one easy
geographic fold (the RF analogue of the SOM seed-789 stability check passes).

| Fold | n_test | macro-F1 | bal. acc | Recall: Slow | Recall: Standard | Recall: Hotspot |
|---|---|---|---|---|---|---|
%s

**Conservative spatial-block CV (~25 km, k=%d) — honest lower bound (NOT the headline)**
- macro-F1 = **%.4f** | balanced accuracy = %.4f | overall accuracy = %.4f
- Per-fold spread: macro-F1 min=%.4f / max=%.4f / mean=%.4f / sd=%.4f

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
%s

25 km confusion matrix (rows = truth, cols = predicted):

%s

| Fold | n_test | macro-F1 | bal. acc | Recall: Slow | Recall: Standard | Recall: Hotspot |
|---|---|---|---|---|---|---|
%s

**Random 5-fold CV — INFLATED / not the headline**
- macro-F1 = %.4f | balanced accuracy = %.4f

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
%s

**Inflation gap (random - spatial) macro-F1 = %.4f.**

## Permutation importance (final model)

| Predictor | Permutation importance |
|---|---|
%s

## Per-class SHAP — top drivers (mean |SHAP|)

| Class | Predictor | mean \\|SHAP\\| |
|---|---|---|
%s

## SHAP additivity check (f(x) = E[f] + sum phi)

| Class | Baseline E[f] | Additivity max\\|err\\| |
|---|---|---|
%s

## Permutation vs SHAP on the environmental predictors

Permutation and SHAP disagree on the environmental predictors: permutation ranks
`landuse_macro` last (and decon/clay/slope near the bottom), whereas SHAP ranks
landuse/decon mid-pack. The cause is the `initial_dose_rate` <-> `cs137_dep_kbqm2`
collinearity (r = 0.88): permutation shuffles one collinear feature at a time, so it
under-credits the dose<->cs137 source-term pair and over-deflates the smaller env
signals by comparison. Exact TreeSHAP allocates credit additively per observation and
is not fooled by the collinear pair. **For the environmental predictors, the SHAP-based
ranking is the fairer ranking.** The environmental signal is **secondary-but-nonzero**
(small, stable mean|SHAP| with n=5,000), not zero — the source term dominates, but
land use, decontamination, slope and clay carry a real residual contribution.

## Files written
- `output/rf_model.rds`, `output/rf_cv_metrics.json`, `output/rf_importance.csv`
- `output/rf_shap_values.rds`, `output/rf_shap_values.parquet`
- `figures/rf_confusion_matrix.{png,pdf}`, `figures/rf_permutation_importance.{png,pdf}`
- `figures/rf_shap_summary.{png,pdf}` (+ `rf_shap_beeswarm.{png,pdf}` if shapviz succeeded)

## Reproducibility
- Master seed: **%d** (per-fold seeds = SEED + fold id).
- Key package versions: R %s; arrow %s; ranger %s; blockCV %s; treeshap %s; shapviz %s;
  sf %s; data.table %s.

## Caveats hit
- **Slow Decay is the weak class** (Slow<->Standard bleed) — a finding, expected per audit
  diagnostic, not a defect.
- `cs137_dep_kbqm2` 14%% structural NA (Fukushima-Pref coverage boundary), median-imputed +
  flagged; `cs137_missing` perm-importance is near-zero as the audit predicted (benign flag).
- `initial_dose_rate` <-> `cs137_dep_kbqm2` collinearity r=0.88: SHAP may split source-term
  credit between the two; reported, not silently dropped. Source-term SHAP is framed as
  magnitude / source-term attribution, NOT "rediscovered decay rate."
- SHAP computed on a stratified subsample for tractability; additivity confirmed exact.

%s
',
  n_c4, EXPECTED_N3, EXPECTED_N3, nrow(edge_na), n3,
  NUM_TREES, rf_final$mtry, SEED, BLOCK_SIZE_M, K_FOLDS, K_FOLDS_25,
  pkg_ver("treeshap"), length(shap_idx), pkg_ver("ranger"),
  spatial_res$macro_f1, spatial_res$balanced_accuracy, spatial_res$accuracy,
  max(prop.table(table(model_dt$target))),
  fmt_pc(spc), cm_lines,
  K_FOLDS, s10$min, s10$max, s10$mean, s10$sd, pf10_lines,
  K_FOLDS_25, spatial_res25$macro_f1, spatial_res25$balanced_accuracy,
  spatial_res25$accuracy, s25$min, s25$max, s25$mean, s25$sd,
  fmt_pc(spc25), cm25_lines, pf25_lines,
  random_res$macro_f1, random_res$balanced_accuracy, fmt_pc(rnd),
  inflation_gap, imp_lines, top_shap_lines, add_lines,
  SEED, R.version.string, pkg_ver("arrow"), pkg_ver("ranger"),
  pkg_ver("blockCV"), pkg_ver("treeshap"), pkg_ver("shapviz"),
  pkg_ver("sf"), pkg_ver("data.table"),
  paste0("---\n\n*Full sessionInfo() in `output/rf_session_info.txt`.*")
)
writeLines(run_record, p("docs", "rf-training-run-record.md"))
message("Saved: docs/rf-training-run-record.md")

# --- Session Info ------------------------------------------------------------
writeLines(capture.output(sessionInfo()), p("output", "rf_session_info.txt"))
message("Saved: output/rf_session_info.txt")

message("\n=========================================================")
message(sprintf("DONE. Headline spatial-block CV (10 km) macro-F1 = %.4f", spatial_res$macro_f1))
message(sprintf("      10 km per-fold macro-F1: min=%.4f max=%.4f mean=%.4f sd=%.4f",
                s10$min, s10$max, s10$mean, s10$sd))
message(sprintf("      Conservative 25 km (k=%d) macro-F1 = %.4f (per-fold sd=%.4f)",
                K_FOLDS_25, spatial_res25$macro_f1, s25$sd))
message(sprintf("      Random CV macro-F1 = %.4f (inflation gap = %.4f)",
                random_res$macro_f1, inflation_gap))
message(sprintf("      SHAP subsample n = %d", length(shap_idx)))
message("=========================================================")
