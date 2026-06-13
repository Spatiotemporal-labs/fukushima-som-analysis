# =============================================================================
# Script: som_clustering.R
# Purpose: Train the 8-feature Self-Organizing Map. Fits candidate maps across
#          two hex-grid sizes and five seeds, selects the best map on quantization
#          error and seed stability, and writes the trained model, cell-to-neuron
#          assignments, cluster summary, and metrics. No figure suite (see
#          som_finalize_k4.R for the k=4 cut and figures).
#          The 8 features are 7 robust-IQR z-scored decay descriptors plus a
#          derived n_increases_rate_z = n_increases / n_measurement_dates.
# Input:   output/feature_matrix.parquet, output/feature_summary.json
# Output:  output/som_clusters_sensitivity.parquet,
#          output/som_metrics_sensitivity.json,
#          output/som_cluster_summary_sensitivity.csv,
#          output/som_model_sensitivity.rds
#          (checkpoint: output/som_all_models_checkpoint_sensitivity.rds)
# =============================================================================

# --- Packages ----------------------------------------------------------------
suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(kohonen)
  library(cluster)
  library(mclust)
  library(jsonlite)
})

# --- Configuration -----------------------------------------------------------
SEEDS        <- c(42L, 123L, 456L, 789L, 2024L)
RLEN         <- 2000L
ALPHA        <- c(0.05, 0.01)
MIN_CLUSTER_PCT <- 0.01  # 1% minimum cluster size
K_RANGE      <- 3:8      # acceptable range for final k
FDNPP_LAT    <- 37.4211
FDNPP_LON    <- 141.0328
GAP_B        <- 100L     # bootstrap samples for gap statistic

# 8-feature set: 7 robust-IQR z-scored decay descriptors plus the derived
# n_increases_rate_z (upticks per measurement opportunity).
SOM_FEATURES <- c(
  "total_decay_ratio_z", "log_linear_slope_z", "log_linear_r2_z",
  "early_slope_z", "late_slope_z", "residual_cv_z",
  "slope_ratio_z", "n_increases_rate_z"
)

# Human-readable labels in canonical order
FEATURE_LABELS <- c(
  "Total Decay Ratio", "Log-Linear Slope", "Log-Linear R²",
  "Early Phase Slope", "Late Phase Slope", "Residual CV",
  "Early/Late Slope Ratio", "Number of Increases (rate)"
)
names(FEATURE_LABELS) <- SOM_FEATURES

dir.create("output",  showWarnings = FALSE)

t_start <- Sys.time()

log_progress <- function(msg) {
  elapsed <- round(difftime(Sys.time(), t_start, units = "mins"), 1)
  message(sprintf("[%s min] %s", elapsed, msg))
}

# --- Helper functions (VERBATIM from som_clustering.R) -----------------------

compute_qe <- function(model) {
  # kohonen stores per-observation distances to BMU as a vector (single layer)
  mean(model$distances)
}

check_convergence <- function(model, run_id) {
  changes <- model$changes[, 1]
  late_mean  <- mean(changes[1901:2000])
  mid_mean   <- mean(changes[1400:1500])
  rel_change <- abs(late_mean - mid_mean) / mid_mean
  converged  <- rel_change < 0.01

  message(sprintf("  %s: final_change=%.6f, rel_delta=%.4f, converged=%s",
                  run_id, late_mean, rel_change, converged))

  if (!converged) {
    warning(sprintf("Model %s may not have converged (rel_change=%.4f > 0.01)",
                    run_id, rel_change))
  }
  list(converged = converged, rel_change = rel_change, final_change = late_mean)
}

build_adjacency_set <- function(grid) {
  # Derive adjacency directly from kohonen's hex coordinate positions.
  pts <- grid$pts
  n_neurons <- nrow(pts)
  adj <- vector("list", n_neurons)

  for (i in 1:n_neurons) {
    dists <- sqrt((pts[, 1] - pts[i, 1])^2 + (pts[, 2] - pts[i, 2])^2)
    adj[[i]] <- which(dists > 0 & dists < 1.1)
  }
  adj
}

validate_adjacency <- function() {
  grid <- somgrid(xdim = 5, ydim = 5, topo = "hexagonal")
  adj  <- build_adjacency_set(grid)

  message(sprintf("  Neuron 1 neighbors: [%s] (expect 3)",
                  paste(adj[[1]], collapse = ", ")))
  stopifnot(length(adj[[1]]) == 3)

  message(sprintf("  Neuron 13 (center) neighbors: [%s] (expect 6)",
                  paste(adj[[13]], collapse = ", ")))
  stopifnot(length(adj[[13]]) == 6)

  message(sprintf("  Neuron 5 (corner) neighbors: [%s] (expect 2)",
                  paste(adj[[5]], collapse = ", ")))
  stopifnot(length(adj[[5]]) == 2)

  message("  Adjacency validation passed.")
}

compute_te <- function(model) {
  codebook  <- model$codes[[1]]
  n_neurons <- nrow(codebook)
  data_mat  <- model$data[[1]]
  n_obs     <- nrow(data_mat)

  adj <- build_adjacency_set(model$grid)

  data_sq  <- rowSums(data_mat^2)
  code_sq  <- rowSums(codebook^2)
  cross    <- data_mat %*% t(codebook)
  dist_matrix <- sqrt(
    pmax(outer(data_sq, rep(1, n_neurons)) - 2 * cross +
         outer(rep(1, n_obs), code_sq), 0)
  )

  bmu1 <- max.col(-dist_matrix, ties.method = "first")

  idx_mat <- cbind(1:n_obs, bmu1)
  dist_matrix[idx_mat] <- Inf
  bmu2 <- max.col(-dist_matrix, ties.method = "first")

  is_adjacent <- vapply(1:n_obs, function(i) bmu2[i] %in% adj[[bmu1[i]]],
                        logical(1))
  sum(!is_adjacent) / n_obs
}

compute_seed_stability <- function(results, grid_name, seeds) {
  assignments <- lapply(seeds, function(s) {
    run_id <- paste0(grid_name, "_seed", s)
    results[[run_id]]$model$unit.classif
  })

  n_seeds <- length(seeds)
  ari_matrix <- matrix(1, nrow = n_seeds, ncol = n_seeds,
                       dimnames = list(seeds, seeds))

  for (i in 1:(n_seeds - 1)) {
    for (j in (i + 1):n_seeds) {
      ari_val <- mclust::adjustedRandIndex(assignments[[i]], assignments[[j]])
      ari_matrix[i, j] <- ari_val
      ari_matrix[j, i] <- ari_val
    }
  }

  mean_ari <- mean(ari_matrix[upper.tri(ari_matrix)])
  min_ari  <- min(ari_matrix[upper.tri(ari_matrix)])

  message(sprintf("  %s: mean_ARI=%.4f, min_ARI=%.4f, target=0.80",
                  grid_name, mean_ari, min_ari))

  list(ari_matrix = ari_matrix, mean_ari = mean_ari,
       min_ari = min_ari, pass = mean_ari > 0.80)
}

compute_cluster_centroids <- function(som_matrix, obs_clusters, norm_params,
                                       som_features) {
  k <- length(unique(obs_clusters))
  cluster_ids <- sort(unique(obs_clusters))

  centroids_z <- t(sapply(cluster_ids, function(cl) {
    colMeans(som_matrix[obs_clusters == cl, , drop = FALSE])
  }))
  colnames(centroids_z) <- som_features
  rownames(centroids_z) <- paste0("Cluster_", cluster_ids)

  # Back-transform to original units
  centroids_orig <- centroids_z
  for (j in seq_along(som_features)) {
    feat_z   <- som_features[j]
    feat_raw <- sub("_z$", "", feat_z)
    med <- norm_params[[feat_raw]]$median
    iqr <- norm_params[[feat_raw]]$iqr
    centroids_orig[, j] <- centroids_z[, j] * iqr + med
  }
  colnames(centroids_orig) <- sub("_z$", "", som_features)

  list(z_scored = centroids_z, original = centroids_orig)
}

auto_label_clusters <- function(centroids_z) {
  k      <- nrow(centroids_z)
  labels <- rep(NA_character_, k)
  used   <- rep(FALSE, k)

  # Rule 1: Rapid Decay -- most negative early_slope + low residual_cv
  early_rank <- order(centroids_z[, "early_slope_z"])
  for (idx in early_rank) {
    if (!used[idx] &&
        centroids_z[idx, "residual_cv_z"] < median(centroids_z[, "residual_cv_z"])) {
      labels[idx] <- "Rapid Decay"
      used[idx]   <- TRUE
      break
    }
  }

  # Rule 2: Slow Retention -- highest total_decay_ratio + flattest slope
  if (any(!used)) {
    tdr_rank <- order(centroids_z[, "total_decay_ratio_z"], decreasing = TRUE)
    for (idx in tdr_rank) {
      if (!used[idx]) {
        labels[idx] <- "Slow Retention"
        used[idx]   <- TRUE
        break
      }
    }
  }

  # Rule 3: Recontamination -- highest n_increases_rate
  if (any(!used)) {
    ninc_rank <- order(centroids_z[, "n_increases_rate_z"], decreasing = TRUE)
    for (idx in ninc_rank) {
      if (!used[idx]) {
        labels[idx] <- "Recontamination/Anomalous"
        used[idx]   <- TRUE
        break
      }
    }
  }

  # Rule 4: Consistent Exponential -- highest R^2
  if (any(!used)) {
    r2_rank <- order(centroids_z[, "log_linear_r2_z"], decreasing = TRUE)
    for (idx in r2_rank) {
      if (!used[idx]) {
        labels[idx] <- "Consistent Exponential Decay"
        used[idx]   <- TRUE
        break
      }
    }
  }

  # Rule 5: Fallback -- detect near-zero "standard/modal" clusters first
  if (any(!used)) {
    for (idx in which(!used)) {
      deviations <- abs(centroids_z[idx, ])
      if (max(deviations) < 0.5) {
        labels[idx] <- "Standard Decay"
        used[idx]   <- TRUE
        next
      }
      max_feat   <- names(which.max(deviations))
      max_val    <- centroids_z[idx, max_feat]

      label <- switch(max_feat,
        "late_slope_z"          = if (max_val < 0) "Accelerating Late Decay"
                                  else "Stagnant Late Phase",
        "slope_ratio_z"         = if (max_val > 0) "Front-Loaded Decay"
                                  else "Delayed Decay Onset",
        "temporal_span_years_z" = if (max_val < 0) "Short Monitoring Record"
                                  else "Extended Monitoring",
        "total_decay_ratio_z"   = if (max_val > 0) "Elevated Retention"
                                  else "Enhanced Decay",
        "log_linear_slope_z"    = if (max_val < 0) "Steep Overall Decay"
                                  else "Shallow Overall Decay",
        "early_slope_z"         = if (max_val < 0) "Fast Early Decay"
                                  else "Slow Early Phase",
        "residual_cv_z"         = if (max_val > 0) "High Variability"
                                  else "Low Variability",
        "n_increases_rate_z"    = if (max_val > 0) "Frequent Increases"
                                  else "Steady Decline",
        "log_linear_r2_z"       = if (max_val > 0) "Good Exponential Fit"
                                  else "Poor Exponential Fit",
        paste0("Regime ", idx)
      )
      labels[idx] <- label
    }
  }

  # Ensure uniqueness — append cluster number to any duplicates
  dup_labels <- duplicated(labels) | duplicated(labels, fromLast = TRUE)
  if (any(dup_labels)) {
    for (idx in which(dup_labels)) {
      labels[idx] <- paste0(labels[idx], " ", idx)
    }
  }

  labels
}

compute_umatrix <- function(model, adj) {
  codebook  <- model$codes[[1]]
  n_neurons <- nrow(codebook)
  u_values  <- numeric(n_neurons)

  for (i in 1:n_neurons) {
    neighbors <- adj[[i]]
    if (length(neighbors) > 0) {
      dists <- vapply(neighbors, function(j) {
        sqrt(sum((codebook[i, ] - codebook[j, ])^2))
      }, numeric(1))
      u_values[i] <- mean(dists)
    }
  }
  u_values
}

# =============================================================================
# --- Load Data ---------------------------------------------------------------
# =============================================================================

log_progress("Loading data...")
dt <- as.data.table(read_parquet("output/feature_matrix.parquet"))

# --- Derive n_increases_rate and robust-IQR z-score (engineer_features.py:821-833)
log_progress("Deriving n_increases_rate_z (robust IQR scaling)...")

stopifnot(min(dt$n_measurement_dates) >= 1)

n_increases_rate <- dt$n_increases / dt$n_measurement_dates

rate_median <- as.numeric(median(n_increases_rate))
rate_q25    <- as.numeric(quantile(n_increases_rate, 0.25))
rate_q75    <- as.numeric(quantile(n_increases_rate, 0.75))
rate_iqr    <- rate_q75 - rate_q25

if (rate_iqr < 1e-12) {
  n_increases_rate_z <- rep(0.0, length(n_increases_rate))
  rate_iqr_used <- NA_real_
} else {
  n_increases_rate_z <- pmin(pmax((n_increases_rate - rate_median) / rate_iqr, -3.0), 3.0)
  rate_iqr_used <- rate_iqr
}
rate_clipped_frac <- mean(abs(n_increases_rate_z) >= 3.0)

message(sprintf("  n_increases_rate: median=%.6f IQR=%.6f clipped_frac=%.4f%%",
                rate_median,
                if (is.finite(rate_iqr_used)) rate_iqr_used else 0.0,
                100.0 * rate_clipped_frac))

# Append derived columns to dt for parquet output
dt[, n_increases_rate   := n_increases_rate]
dt[, n_increases_rate_z := n_increases_rate_z]

# Build the 8-feature SOM input matrix
som_matrix <- as.matrix(dt[, ..SOM_FEATURES])

# Load normalization params for back-transformation (7 z-scored features + derived rate)
params      <- fromJSON("output/feature_summary.json")
norm_params <- params$normalization_params
# Inject derived params so back-transform works for n_increases_rate
norm_params[["n_increases_rate"]] <- list(median = rate_median, iqr = rate_iqr_used)

# --- Pre-training sanity checks ----------------------------------------------
log_progress("Running pre-training sanity checks...")

stopifnot(nrow(som_matrix) == 138768)
stopifnot(ncol(som_matrix) == 8)
stopifnot(!anyNA(som_matrix))
stopifnot(all(apply(som_matrix, 2, sd) > 0))
stopifnot(all(som_matrix >= -3 & som_matrix <= 3))

message("Feature correlation matrix:")
print(round(cor(som_matrix), 3))

message("All sanity checks passed.")

# =============================================================================
# --- SOM Training Loop -------------------------------------------------------
# =============================================================================

grid_sizes <- list(
  "5x5" = somgrid(xdim = 5, ydim = 5, topo = "hexagonal"),
  "6x6" = somgrid(xdim = 6, ydim = 6, topo = "hexagonal")
)

checkpoint_file <- "output/som_all_models_checkpoint_sensitivity.rds"

if (file.exists(checkpoint_file)) {
  log_progress("Checkpoint found. Loading previously trained models...")
  results <- readRDS(checkpoint_file)
  message(sprintf("Loaded %d models from checkpoint.", length(results)))
} else {
  log_progress("Starting SOM training loop (10 models)...")
  results <- list()

  for (grid_name in names(grid_sizes)) {
    grid <- grid_sizes[[grid_name]]
    for (seed in SEEDS) {
      run_id <- paste0(grid_name, "_seed", seed)
      message(sprintf("Training %s ...", run_id))
      t_run <- Sys.time()

      set.seed(seed)
      som_model <- som(
        som_matrix,
        grid     = grid,
        rlen     = RLEN,
        alpha    = ALPHA,
        dist.fcts = "euclidean",
        keep.data = TRUE
      )

      elapsed_run <- round(difftime(Sys.time(), t_run, units = "mins"), 1)
      message(sprintf("  Completed in %s min", elapsed_run))

      results[[run_id]] <- list(
        grid_name = grid_name,
        seed      = seed,
        model     = som_model
      )
    }
  }

  log_progress("SOM training complete. Saving checkpoint...")
  saveRDS(results, checkpoint_file)
}

# =============================================================================
# --- Validate kohonen grid layout --------------------------------------------
# =============================================================================
log_progress("Validating grid coordinate system...")

test_model <- results[["5x5_seed42"]]$model
pts <- test_model$grid$pts
message(sprintf("  Neuron 1 coords: x=%.1f, y=%.1f", pts[1, 1], pts[1, 2]))
message(sprintf("  Neuron 2 coords: x=%.1f, y=%.1f", pts[2, 1], pts[2, 2]))
message(sprintf("  Neuron 6 coords: x=%.1f, y=%.1f", pts[6, 1], pts[6, 2]))

validate_adjacency()

# =============================================================================
# --- Quality Metrics (QE, TE, Convergence) -----------------------------------
# =============================================================================

log_progress("Computing quality metrics for all 10 models...")

metrics_list <- list()

for (run_id in names(results)) {
  model     <- results[[run_id]]$model
  grid_name <- results[[run_id]]$grid_name
  seed      <- results[[run_id]]$seed

  qe  <- compute_qe(model)
  conv <- check_convergence(model, run_id)

  message(sprintf("  Computing TE for %s...", run_id))
  te <- compute_te(model)
  message(sprintf("  %s: QE=%.4f, TE=%.4f", run_id, qe, te))

  metrics_list[[run_id]] <- list(
    grid_name  = grid_name,
    seed       = seed,
    qe         = qe,
    te         = te,
    converged  = conv$converged,
    rel_change = conv$rel_change,
    final_change = conv$final_change
  )
}

log_progress("All quality metrics computed.")

# =============================================================================
# --- Seed Stability (ARI) ---------------------------------------------------
# =============================================================================

log_progress("Computing seed stability (ARI)...")

stability <- list()
for (grid_name in names(grid_sizes)) {
  stability[[grid_name]] <- compute_seed_stability(results, grid_name, SEEDS)
}

# =============================================================================
# --- Model Selection ---------------------------------------------------------
# =============================================================================

log_progress("Selecting best model...")

grid_summary <- list()
for (gn in names(grid_sizes)) {
  runs <- metrics_list[grep(paste0("^", gn, "_"), names(metrics_list))]
  grid_summary[[gn]] <- list(
    mean_qe  = mean(sapply(runs, `[[`, "qe")),
    mean_te  = mean(sapply(runs, `[[`, "te")),
    mean_ari = stability[[gn]]$mean_ari,
    min_ari  = stability[[gn]]$min_ari
  )
}

message("\n=== GRID COMPARISON ===")
for (gn in names(grid_summary)) {
  gs <- grid_summary[[gn]]
  message(sprintf("  %s: mean_QE=%.4f, mean_TE=%.4f, mean_ARI=%.4f",
                  gn, gs$mean_qe, gs$mean_te, gs$mean_ari))
}

# Stage 1: Select grid size
eligible_grids <- names(grid_summary)[sapply(grid_summary, function(g) {
  g$mean_te < 0.10 && g$mean_ari > 0.80
})]

if (length(eligible_grids) == 0) {
  selected_grid <- names(which.min(sapply(grid_summary, `[[`, "mean_te")))
  message("  WARNING: No grid met both TE<0.10 and ARI>0.80. Selecting by lowest TE.")
} else if (length(eligible_grids) == 1) {
  selected_grid <- eligible_grids
} else {
  selected_grid <- eligible_grids[which.min(
    sapply(eligible_grids, function(g) grid_summary[[g]]$mean_qe)
  )]
}

# Stage 2: Within selected grid, pick seed with lowest QE
grid_runs <- metrics_list[grep(paste0("^", selected_grid, "_"), names(metrics_list))]
best_run_id <- names(which.min(sapply(grid_runs, `[[`, "qe")))
selected_seed <- results[[best_run_id]]$seed
selected_model <- results[[best_run_id]]$model
selected_qe <- metrics_list[[best_run_id]]$qe
selected_te <- metrics_list[[best_run_id]]$te

message(sprintf("\n=== MODEL SELECTION ==="))
message(sprintf("Grid selected: %s (mean_QE=%.4f, mean_TE=%.4f, mean_ARI=%.4f)",
                selected_grid, grid_summary[[selected_grid]]$mean_qe,
                grid_summary[[selected_grid]]$mean_te,
                grid_summary[[selected_grid]]$mean_ari))
message(sprintf("Seed selected: %d (QE=%.4f, TE=%.4f)", selected_seed,
                selected_qe, selected_te))

# =============================================================================
# --- Hierarchical Clustering on Codebook Vectors -----------------------------
# =============================================================================

log_progress("Performing hierarchical clustering on codebook vectors...")

codebook      <- selected_model$codes[[1]]
codebook_dist <- dist(codebook, method = "euclidean")
hc            <- hclust(codebook_dist, method = "ward.D2")

# Primary: Silhouette width (k = 2 to 10)
sil_widths <- sapply(2:10, function(k) {
  labels <- cutree(hc, k = k)
  sil    <- silhouette(labels, codebook_dist)
  mean(sil[, "sil_width"])
})
names(sil_widths) <- 2:10
k_sil <- as.integer(names(which.max(sil_widths)))

# Secondary: Gap statistic (B = 100)
set.seed(42L)
gap_stat <- clusGap(codebook, FUN = function(x, k) {
  list(cluster = cutree(hclust(dist(x), method = "ward.D2"), k = k))
}, K.max = 10, B = GAP_B)
k_gap <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"],
                method = "Tibs2001SEmax")

# Visual: Within-cluster SS (for elbow plot)
wss <- sapply(2:10, function(k) {
  labels <- cutree(hc, k = k)
  sum(sapply(unique(labels), function(cl) {
    members <- codebook[labels == cl, , drop = FALSE]
    sum(scale(members, scale = FALSE)^2)
  }))
})
names(wss) <- 2:10

message(sprintf("  Silhouette optimal k: %d (width=%.4f)", k_sil, max(sil_widths)))
message(sprintf("  Gap statistic optimal k: %d", k_gap))

# Decision rule: use silhouette (primary), clamp to [3, 8]
selected_k <- k_sil
if (selected_k < 3) selected_k <- 3L
if (selected_k > 8) selected_k <- 8L

neuron_clusters <- cutree(hc, k = selected_k)
bmu_assignments <- selected_model$unit.classif
obs_clusters    <- neuron_clusters[bmu_assignments]

# Minimum cluster size check -- decrement k if any cluster < 1%
min_threshold <- ceiling(MIN_CLUSTER_PCT * nrow(som_matrix))

while (selected_k > 2) {
  cluster_sizes_tbl <- table(obs_clusters)
  if (min(cluster_sizes_tbl) >= min_threshold) break

  violating <- names(which(cluster_sizes_tbl < min_threshold))
  message(sprintf("  WARNING: Cluster(s) %s below 1%% threshold. Decrementing k from %d to %d.",
                  paste(violating, collapse = ", "), selected_k, selected_k - 1))
  selected_k <- selected_k - 1L
  neuron_clusters <- cutree(hc, k = selected_k)
  obs_clusters    <- neuron_clusters[bmu_assignments]
}

cluster_sizes_tbl <- table(obs_clusters)
message(sprintf("\n  Final k = %d", selected_k))
message(sprintf("  Cluster sizes: %s",
                paste(sprintf("%d (%.1f%%)", cluster_sizes_tbl,
                              100 * cluster_sizes_tbl / sum(cluster_sizes_tbl)),
                      collapse = ", ")))

# =============================================================================
# --- Cluster Interpretation --------------------------------------------------
# =============================================================================

log_progress("Computing cluster centroids and auto-labeling...")

centroids <- compute_cluster_centroids(som_matrix, obs_clusters, norm_params,
                                        SOM_FEATURES)

cluster_labels <- auto_label_clusters(centroids$z_scored)

message("\n=== AUTO-GENERATED LABELS (review before manuscript) ===")
for (i in 1:selected_k) {
  sz <- cluster_sizes_tbl[as.character(i)]
  message(sprintf("Cluster %d -> \"%s\" (n=%s, %.1f%%)",
                  i, cluster_labels[i], format(sz, big.mark = ","),
                  100 * sz / sum(cluster_sizes_tbl)))
  message(sprintf("  Key z-scores: early_slope=%.2f, total_decay=%.2f, n_increases_rate=%.2f, R2=%.2f",
                  centroids$z_scored[i, "early_slope_z"],
                  centroids$z_scored[i, "total_decay_ratio_z"],
                  centroids$z_scored[i, "n_increases_rate_z"],
                  centroids$z_scored[i, "log_linear_r2_z"]))
}

# =============================================================================
# --- Output File Writing (FOUR *_sensitivity artifacts) ----------------------
# =============================================================================

log_progress("Writing sensitivity output files...")

# 1. som_model_sensitivity.rds
saveRDS(selected_model, "output/som_model_sensitivity.rds")
message("  Saved output/som_model_sensitivity.rds")

# 2. som_clusters_sensitivity.parquet
dt[, som_bmu           := bmu_assignments]
dt[, som_cluster       := obs_clusters]
dt[, som_cluster_label := cluster_labels[obs_clusters]]
write_parquet(dt, "output/som_clusters_sensitivity.parquet")
message("  Saved output/som_clusters_sensitivity.parquet")

# 3. som_cluster_summary_sensitivity.csv (Table 1)
summary_dt <- data.table(
  cluster_id    = 1:selected_k,
  cluster_label = cluster_labels,
  n_observations = as.integer(cluster_sizes_tbl),
  pct_of_total  = round(100 * as.numeric(cluster_sizes_tbl) / sum(cluster_sizes_tbl), 2)
)

for (j in 1:ncol(centroids$original)) {
  col_name <- colnames(centroids$original)[j]
  summary_dt[, (col_name) := signif(centroids$original[, j], 4)]
}

for (i in 1:selected_k) {
  mask <- obs_clusters == i
  summary_dt[i, mean_ecological_half_life_y := round(mean(dt$ecological_half_life_y[mask], na.rm = TRUE), 2)]
  summary_dt[i, mean_distance_from_fdnpp_km := round(mean(dt$distance_from_fdnpp_km[mask], na.rm = TRUE), 2)]
  summary_dt[i, mean_initial_dose_rate := signif(mean(dt$initial_dose_rate[mask], na.rm = TRUE), 4)]
}

fwrite(summary_dt, "output/som_cluster_summary_sensitivity.csv")
message("  Saved output/som_cluster_summary_sensitivity.csv")

# 4. som_metrics_sensitivity.json
all_runs_json <- lapply(names(metrics_list), function(run_id) {
  m <- metrics_list[[run_id]]
  list(
    grid      = m$grid_name,
    seed      = m$seed,
    qe        = round(m$qe, 6),
    te        = round(m$te, 6),
    converged = m$converged,
    rel_change = round(m$rel_change, 6)
  )
})

grid_comp_json <- lapply(names(grid_summary), function(gn) {
  gs <- grid_summary[[gn]]
  list(
    mean_qe    = round(gs$mean_qe, 6),
    mean_te    = round(gs$mean_te, 6),
    mean_ari   = round(gs$mean_ari, 4),
    min_ari    = round(gs$min_ari, 4),
    ari_matrix = round(stability[[gn]]$ari_matrix, 4)
  )
})
names(grid_comp_json) <- names(grid_summary)

clusters_json <- lapply(1:selected_k, function(i) {
  list(
    cluster_id      = i,
    label           = cluster_labels[i],
    n_observations  = as.integer(cluster_sizes_tbl[as.character(i)]),
    pct_of_total    = round(100 * as.numeric(cluster_sizes_tbl[as.character(i)]) / sum(cluster_sizes_tbl), 2),
    centroid_z      = as.list(round(centroids$z_scored[i, ], 4)),
    centroid_original = as.list(signif(centroids$original[i, ], 4)),
    neurons_in_cluster = which(neuron_clusters == i)
  )
})

selected_grid_clears_bars <-
  (grid_summary[[selected_grid]]$mean_te < 0.10) &&
  (grid_summary[[selected_grid]]$mean_ari > 0.80)

metrics_json <- list(
  generated_at   = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  pipeline_phase = "4_som_clustering",
  run_type       = "som_clustering",
  input_file     = "output/feature_matrix.parquet",
  n_observations = nrow(som_matrix),
  n_features     = ncol(som_matrix),
  feature_names  = SOM_FEATURES,
  n_increases_rate = list(
    definition   = "n_increases / n_measurement_dates, robust-IQR z-scored then clipped to [-3,3]",
    median       = rate_median,
    q25          = rate_q25,
    q75          = rate_q75,
    iqr          = if (is.finite(rate_iqr_used)) rate_iqr_used else NA,
    clipped_frac = rate_clipped_frac
  ),
  training_config = list(
    rlen      = RLEN,
    alpha     = ALPHA,
    dist_fcts = "euclidean",
    mode      = "online",
    seeds     = SEEDS,
    grid_sizes_tested = names(grid_sizes)
  ),
  all_runs          = all_runs_json,
  grid_comparison   = grid_comp_json,
  selected_model    = list(
    grid             = selected_grid,
    seed             = selected_seed,
    qe               = round(selected_qe, 6),
    te               = round(selected_te, 6),
    n_neurons        = nrow(codebook),
    mean_te          = round(grid_summary[[selected_grid]]$mean_te, 6),
    mean_ari         = round(grid_summary[[selected_grid]]$mean_ari, 4),
    min_ari          = round(grid_summary[[selected_grid]]$min_ari, 4),
    clears_acceptance_bars = selected_grid_clears_bars,
    selection_reason = if (selected_grid_clears_bars) {
      sprintf("Grid %s selected (TE<0.10 and ARI>0.80); seed %d had lowest QE",
              selected_grid, selected_seed)
    } else {
      sprintf("WARNING: Grid %s selected by fallback (lowest mean TE); it does NOT clear both TE<0.10 AND ARI>0.80. seed %d had lowest QE",
              selected_grid, selected_seed)
    }
  ),
  hierarchical_clustering = list(
    method           = "ward.D2",
    distance         = "euclidean",
    k_selected       = selected_k,
    k_silhouette     = k_sil,
    k_gap            = k_gap,
    silhouette_widths = as.list(round(sil_widths, 4)),
    gap_values       = as.list(round(gap_stat$Tab[, "gap"], 4)),
    wss_values       = as.list(round(wss, 4))
  ),
  clusters = clusters_json
)

write_json(metrics_json, "output/som_metrics_sensitivity.json", pretty = TRUE, auto_unbox = TRUE)
message("  Saved output/som_metrics_sensitivity.json")

# =============================================================================
# --- Final Summary -----------------------------------------------------------
# =============================================================================

log_progress("Sensitivity pipeline complete. Summary:")

message("\n=== FINAL RESULTS (SENSITIVITY) ===")
message(sprintf("Selected model: %s, seed %d", selected_grid, selected_seed))
message(sprintf("QE = %.4f, TE = %.4f", selected_qe, selected_te))
message(sprintf("Selected grid mean_ARI = %.4f, min_ARI = %.4f",
                grid_summary[[selected_grid]]$mean_ari,
                grid_summary[[selected_grid]]$min_ari))
message(sprintf("Acceptance bars (TE<0.10 AND mean_ARI>0.80) cleared: %s",
                selected_grid_clears_bars))
message(sprintf("Number of regimes: %d", selected_k))

message("\nCluster summary:")
for (i in 1:selected_k) {
  sz  <- cluster_sizes_tbl[as.character(i)]
  pct <- 100 * sz / sum(cluster_sizes_tbl)
  message(sprintf("  %d. %-35s  n=%s (%5.1f%%)  n_increases_rate_z=%.3f",
                  i, cluster_labels[i], format(sz, big.mark = ","), pct,
                  centroids$z_scored[i, "n_increases_rate_z"]))
}

message("\nOutput files:")
message("  output/som_model_sensitivity.rds")
message("  output/som_clusters_sensitivity.parquet")
message("  output/som_metrics_sensitivity.json")
message("  output/som_cluster_summary_sensitivity.csv")

total_time <- round(difftime(Sys.time(), t_start, units = "mins"), 1)
message(sprintf("\nTotal runtime: %s minutes", total_time))

# --- Session Info ------------------------------------------------------------
writeLines(capture.output(sessionInfo()), "output/som_session_info_sensitivity.txt")
message("Session info saved to output/som_session_info_sensitivity.txt")

message("SENSITIVITY_RUN_COMPLETE")
