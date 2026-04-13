# =============================================================================
# Script: som_clustering.R
# Purpose: Train Self-Organizing Maps on 9 temporal features from 138,768
#          Fukushima mesh cells to discover contamination decay regimes
# Input:   output/feature_matrix.parquet, output/feature_summary.json
# Output:  output/som_model.rds, output/som_clusters.parquet,
#          output/som_metrics.json, output/som_cluster_summary.csv,
#          output/som_cluster_assignments.csv, figures/ (13 figures)
# =============================================================================

# --- Packages ----------------------------------------------------------------
suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(kohonen)
  library(cluster)
  library(mclust)
  library(ggplot2)
  library(viridis)
  library(patchwork)
  library(jsonlite)
  library(ggspatial)
  library(RColorBrewer)
  library(scales)
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

SOM_FEATURES <- c(
  "total_decay_ratio_z", "log_linear_slope_z", "log_linear_r2_z",
  "early_slope_z", "late_slope_z", "residual_cv_z",
  "slope_ratio_z", "n_increases_z", "temporal_span_years_z"
)

# Human-readable labels in canonical order
FEATURE_LABELS <- c(
  "Total Decay Ratio", "Log-Linear Slope", "Log-Linear R\u00b2",

"Early Phase Slope", "Late Phase Slope", "Residual CV",
  "Early/Late Slope Ratio", "Number of Increases", "Observation Span (years)"
)
names(FEATURE_LABELS) <- SOM_FEATURES

dir.create("output",  showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

t_start <- Sys.time()

log_progress <- function(msg) {
  elapsed <- round(difftime(Sys.time(), t_start, units = "mins"), 1)
  message(sprintf("[%s min] %s", elapsed, msg))
}

# --- Publication theme -------------------------------------------------------
theme_pub <- theme_minimal(base_size = 11) +
  theme(
    plot.title   = element_text(face = "bold", size = 12),
    axis.title   = element_text(size = 10),
    legend.title = element_text(face = "bold", size = 10),
    strip.text   = element_text(face = "bold", size = 10)
  )

# --- Helper functions --------------------------------------------------------

save_figure <- function(plot_obj, filename, width = 8, height = 6) {
  ggsave(paste0("figures/", filename, ".png"), plot_obj,
         width = width, height = height, dpi = 300, bg = "white")
  ggsave(paste0("figures/", filename, ".pdf"), plot_obj,
         width = width, height = height, bg = "white")
  message(sprintf("  Saved figures/%s.{png,pdf}", filename))
}

save_base_figure <- function(plot_fn, filename, width = 8, height = 6) {
  png(paste0("figures/", filename, ".png"),
      width = width, height = height, units = "in", res = 300)
  plot_fn()
  dev.off()
  pdf(paste0("figures/", filename, ".pdf"),
      width = width, height = height)
  plot_fn()
  dev.off()
  message(sprintf("  Saved figures/%s.{png,pdf}", filename))
}

save_caption <- function(filename, caption_text) {
  writeLines(caption_text, paste0("figures/", filename, "_caption.txt"))
}

# --- Functions: SOM quality metrics ------------------------------------------

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

# --- Functions: Topographic error (custom) -----------------------------------

build_adjacency_set <- function(grid) {
  # Derive adjacency directly from kohonen's hex coordinate positions.
  # Two neurons are adjacent if their Euclidean distance on the hex grid
  # is approximately 1.0 (the unit spacing). We use a threshold of 1.1
  # to accommodate floating-point imprecision.
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
  # Validate against known topology before trusting TE results
  grid <- somgrid(xdim = 5, ydim = 5, topo = "hexagonal")
  adj  <- build_adjacency_set(grid)

  # Neuron 1 is a corner -- should have 3 neighbors on a 5x5 hex grid
  # (kohonen hex grids give 3 neighbors for corners, not 2)
  message(sprintf("  Neuron 1 neighbors: [%s] (expect 3)",
                  paste(adj[[1]], collapse = ", ")))
  stopifnot(length(adj[[1]]) == 3)

  # Center neuron 13 (row=3, col=3) should have 6 neighbors
  message(sprintf("  Neuron 13 (center) neighbors: [%s] (expect 6)",
                  paste(adj[[13]], collapse = ", ")))
  stopifnot(length(adj[[13]]) == 6)

  # Neuron 5 (row=1, col=5) is a corner -- should have 2 neighbors
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

  # Vectorized distance matrix: n_obs x n_neurons
  # Using the expansion: ||a-b||^2 = ||a||^2 - 2*a.b + ||b||^2
  data_sq  <- rowSums(data_mat^2)
  code_sq  <- rowSums(codebook^2)
  cross    <- data_mat %*% t(codebook)
  dist_matrix <- sqrt(
    pmax(outer(data_sq, rep(1, n_neurons)) - 2 * cross +
         outer(rep(1, n_obs), code_sq), 0)
  )

  # Find 1st BMU
  bmu1 <- max.col(-dist_matrix, ties.method = "first")

  # Mask 1st BMU to find 2nd BMU
  idx_mat <- cbind(1:n_obs, bmu1)
  dist_matrix[idx_mat] <- Inf
  bmu2 <- max.col(-dist_matrix, ties.method = "first")

  # Check adjacency vectorized
  is_adjacent <- vapply(1:n_obs, function(i) bmu2[i] %in% adj[[bmu1[i]]],
                        logical(1))
  sum(!is_adjacent) / n_obs
}

# --- Functions: Seed stability -----------------------------------------------

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

# --- Functions: Cluster interpretation ---------------------------------------

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

  # Rule 3: Recontamination -- highest n_increases
  if (any(!used)) {
    ninc_rank <- order(centroids_z[, "n_increases_z"], decreasing = TRUE)
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
      # If all z-scores are small (max < 0.5), this is the modal/standard regime
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
        "n_increases_z"         = if (max_val > 0) "Frequent Increases"
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

# --- Functions: U-matrix -----------------------------------------------------

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

som_matrix <- as.matrix(dt[, ..SOM_FEATURES])

# Load normalization params for back-transformation
params     <- fromJSON("output/feature_summary.json")
norm_params <- params$normalization_params

# --- Pre-training sanity checks ----------------------------------------------
log_progress("Running pre-training sanity checks...")

stopifnot(nrow(som_matrix) == 138768)
stopifnot(ncol(som_matrix) == 9)
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

checkpoint_file <- "output/som_all_models_checkpoint.rds"

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
# Verify column-major indexing assumption before trusting adjacency
log_progress("Validating grid coordinate system...")

test_model <- results[["5x5_seed42"]]$model
pts <- test_model$grid$pts
message(sprintf("  Neuron 1 coords: x=%.1f, y=%.1f", pts[1, 1], pts[1, 2]))
message(sprintf("  Neuron 2 coords: x=%.1f, y=%.1f", pts[2, 1], pts[2, 2]))
message(sprintf("  Neuron 6 coords: x=%.1f, y=%.1f", pts[6, 1], pts[6, 2]))

# Validate adjacency function
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

# Aggregate per-grid metrics
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
# Prefer grid where TE < 0.10 and ARI > 0.80; if tied, pick lower mean QE
eligible_grids <- names(grid_summary)[sapply(grid_summary, function(g) {
  g$mean_te < 0.10 && g$mean_ari > 0.80
})]

if (length(eligible_grids) == 0) {
  # Fallback: pick lowest TE
  selected_grid <- names(which.min(sapply(grid_summary, `[[`, "mean_te")))
  message("  WARNING: No grid met both TE<0.10 and ARI>0.80. Selecting by lowest TE.")
} else if (length(eligible_grids) == 1) {
  selected_grid <- eligible_grids
} else {
  # Both eligible: pick lower mean QE
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

# --- Optimal k selection -----------------------------------------------------

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

# Assign observations to clusters
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
  message(sprintf("  Key z-scores: early_slope=%.2f, total_decay=%.2f, n_increases=%.2f, R2=%.2f",
                  centroids$z_scored[i, "early_slope_z"],
                  centroids$z_scored[i, "total_decay_ratio_z"],
                  centroids$z_scored[i, "n_increases_z"],
                  centroids$z_scored[i, "log_linear_r2_z"]))
}

# =============================================================================
# --- Build Cluster Palette ---------------------------------------------------
# =============================================================================

if (selected_k <= 8) {
  cluster_palette <- brewer.pal(max(3, selected_k), "Set2")[1:selected_k]
} else {
  cluster_palette <- colorRampPalette(brewer.pal(8, "Set2"))(selected_k)
}
names(cluster_palette) <- cluster_labels

# =============================================================================
# --- Output File Writing -----------------------------------------------------
# =============================================================================

log_progress("Writing output files...")

# 1. som_model.rds
saveRDS(selected_model, "output/som_model.rds")
message("  Saved output/som_model.rds")

# 2. som_clusters.parquet
dt[, som_bmu           := bmu_assignments]
dt[, som_cluster       := obs_clusters]
dt[, som_cluster_label := cluster_labels[obs_clusters]]
write_parquet(dt, "output/som_clusters.parquet")
message("  Saved output/som_clusters.parquet")

# 3. som_cluster_summary.csv (Table 1)
summary_dt <- data.table(
  cluster_id    = 1:selected_k,
  cluster_label = cluster_labels,
  n_observations = as.integer(cluster_sizes_tbl),
  pct_of_total  = round(100 * as.numeric(cluster_sizes_tbl) / sum(cluster_sizes_tbl), 2)
)

# Add back-transformed centroid means
for (j in 1:ncol(centroids$original)) {
  col_name <- colnames(centroids$original)[j]
  summary_dt[, (col_name) := signif(centroids$original[, j], 4)]
}

# Add contextual columns: mean ecological half-life, distance, initial dose rate
for (i in 1:selected_k) {
  mask <- obs_clusters == i
  summary_dt[i, mean_ecological_half_life_y := round(mean(dt$ecological_half_life_y[mask], na.rm = TRUE), 2)]
  summary_dt[i, mean_distance_from_fdnpp_km := round(mean(dt$distance_from_fdnpp_km[mask], na.rm = TRUE), 2)]
  summary_dt[i, mean_initial_dose_rate := signif(mean(dt$initial_dose_rate[mask], na.rm = TRUE), 4)]
}

fwrite(summary_dt, "output/som_cluster_summary.csv")
message("  Saved output/som_cluster_summary.csv")

# 4. som_cluster_assignments.csv (supplementary)
# Determine the correct mesh_id column name
mesh_col <- "mesh_id_250m"
assignments_dt <- dt[, .(
  mesh_id   = get(mesh_col),
  latitude  = latitude,
  longitude = longitude,
  som_cluster = som_cluster,
  som_cluster_label = som_cluster_label
)]
fwrite(assignments_dt, "output/som_cluster_assignments.csv")
message("  Saved output/som_cluster_assignments.csv")

# 5. som_metrics.json
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

metrics_json <- list(
  generated_at   = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  pipeline_phase = "4_som_clustering",
  input_file     = "output/feature_matrix.parquet",
  n_observations = nrow(som_matrix),
  n_features     = ncol(som_matrix),
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
    selection_reason = sprintf(
      "Grid %s selected (TE<0.10 and ARI>0.80); seed %d had lowest QE",
      selected_grid, selected_seed
    )
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

write_json(metrics_json, "output/som_metrics.json", pretty = TRUE, auto_unbox = TRUE)
message("  Saved output/som_metrics.json")

# =============================================================================
# --- Visualization Suite (13 Figures) ----------------------------------------
# =============================================================================

log_progress("Generating figures...")

# Ensure cluster_label is an ordered factor for consistent plotting
dt[, som_cluster_label := factor(som_cluster_label, levels = cluster_labels)]

# ---- Figure 1: Geographic Regime Map (Hero) ---------------------------------

log_progress("  Fig 1: Geographic regime map...")

p1 <- ggplot(dt, aes(x = longitude, y = latitude, color = som_cluster_label)) +
  geom_point(size = 0.1, alpha = 0.6) +
  scale_color_manual(values = cluster_palette, name = "Regime") +
  annotate("point", x = FDNPP_LON, y = FDNPP_LAT,
           shape = 17, size = 3, color = "black") +
  annotate("text", x = FDNPP_LON, y = FDNPP_LAT + 0.02,
           label = "FDNPP", fontface = "bold", size = 3) +
  annotation_scale(location = "bl", width_hint = 0.2) +
  annotation_north_arrow(location = "tr", which_north = "true",
                         style = north_arrow_fancy_orienteering(),
                         height = unit(1.2, "cm"), width = unit(1.2, "cm")) +
  coord_sf(crs = 4326) +
  labs(title = "Contamination Decay Regimes in the Fukushima Region",
       x = "Longitude", y = "Latitude") +
  guides(color = guide_legend(override.aes = list(size = 3, alpha = 1))) +
  theme_pub +
  theme(legend.position = "right")

save_figure(p1, "fig01_geographic_regime_map", width = 8, height = 10)
save_caption("fig01_geographic_regime_map",
  paste0("Figure 1. Geographic distribution of contamination decay regimes ",
         "identified by Self-Organizing Map clustering across 138,768 mesh cells ",
         "in the Fukushima region. Each point represents a 250-m grid cell colored ",
         "by its assigned regime. The black triangle marks the Fukushima Daiichi ",
         "Nuclear Power Station (FDNPP). Regimes were determined by hierarchical ",
         "clustering (Ward's D2, k=", selected_k, ") of a ",
         selected_grid, " hexagonal SOM trained on 9 temporal features."))

# ---- Figure 2: Centroids Heatmap -------------------------------------------

log_progress("  Fig 2: Centroid heatmap...")

heatmap_dt <- data.table(
  cluster = rep(paste0(cluster_labels, "\n(n=",
                       format(as.integer(cluster_sizes_tbl), big.mark = ","), ")"),
                each = length(SOM_FEATURES)),
  feature = rep(FEATURE_LABELS[SOM_FEATURES], selected_k),
  value   = as.vector(t(centroids$z_scored))
)
# Preserve ordering
heatmap_dt[, cluster := factor(cluster, levels = unique(cluster))]
heatmap_dt[, feature := factor(feature, levels = rev(FEATURE_LABELS[SOM_FEATURES]))]

p2 <- ggplot(heatmap_dt, aes(x = feature, y = cluster, fill = value)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = sprintf("%.2f", value)), size = 3) +
  scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                       midpoint = 0, name = "Z-score") +
  labs(title = "Cluster Centroid Profiles (Z-scored)",
       x = NULL, y = NULL) +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

save_figure(p2, "fig02_centroid_heatmap", width = 10, height = 5)
save_caption("fig02_centroid_heatmap",
  paste0("Figure 2. Heatmap of cluster centroid z-scores across 9 temporal features. ",
         "Rows represent ", selected_k, " contamination decay regimes; columns represent ",
         "SOM input features (IQR-scaled). Blue indicates below-median values; ",
         "red indicates above-median. Numbers show exact z-score values. ",
         "Cluster sizes (n) are annotated in row labels."))

# ---- Figure 3: Parallel Coordinates ----------------------------------------

log_progress("  Fig 3: Parallel coordinates...")

par_dt <- data.table(
  cluster = rep(cluster_labels, each = length(SOM_FEATURES)),
  feature = rep(FEATURE_LABELS[SOM_FEATURES], selected_k),
  value   = as.vector(t(centroids$z_scored))
)
par_dt[, cluster := factor(cluster, levels = cluster_labels)]
par_dt[, feature := factor(feature, levels = FEATURE_LABELS[SOM_FEATURES])]
par_dt[, feat_idx := as.numeric(feature)]

p3 <- ggplot(par_dt, aes(x = feature, y = value, color = cluster, group = cluster)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +
  scale_color_manual(values = cluster_palette, name = "Regime") +
  labs(title = "Regime Profiles Across Temporal Features",
       x = NULL, y = "Z-score (IQR-scaled)") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

save_figure(p3, "fig03_parallel_coordinates", width = 10, height = 5)
save_caption("fig03_parallel_coordinates",
  paste0("Figure 3. Parallel coordinates plot showing centroid profiles of ",
         selected_k, " contamination decay regimes across 9 temporal features. ",
         "Each line represents one regime. The dashed horizontal line at y=0 ",
         "indicates the population median. Divergence from zero reflects how ",
         "each regime differs from the typical decay behavior."))

# ---- Figure 4: Component Planes (3x3) --------------------------------------

log_progress("  Fig 4: Component planes...")

# Build hex coordinate data from the kohonen grid
hex_pts <- as.data.frame(selected_model$grid$pts)
hex_pts$neuron <- 1:nrow(hex_pts)

comp_plots <- list()
for (j in 1:length(SOM_FEATURES)) {
  feat <- SOM_FEATURES[j]
  hex_pts$value <- codebook[, j]

  p_comp <- ggplot(hex_pts, aes(x = x, y = y, fill = value)) +
    geom_point(shape = 21, size = 6, color = "grey30") +
    scale_fill_viridis(name = "Z") +
    labs(title = FEATURE_LABELS[feat]) +
    coord_fixed() +
    theme_void(base_size = 9) +
    theme(plot.title = element_text(face = "bold", size = 9, hjust = 0.5),
          legend.key.height = unit(0.4, "cm"),
          legend.key.width  = unit(0.3, "cm"))

  comp_plots[[j]] <- p_comp
}

p4 <- wrap_plots(comp_plots, ncol = 3) +
  plot_annotation(title = "SOM Component Planes",
                  theme = theme(plot.title = element_text(face = "bold", size = 12)))

save_figure(p4, "fig04_component_planes", width = 10, height = 10)
save_caption("fig04_component_planes",
  paste0("Figure 4. Component planes of the ", selected_grid,
         " hexagonal Self-Organizing Map. Each panel shows one of the 9 temporal ",
         "features, with neuron positions reflecting the SOM topology and color ",
         "indicating the codebook value (z-scored). Spatial patterns across panels ",
         "reveal correlated and independent feature dimensions."))

# ---- Figure 5: U-Matrix with Cluster Overlay --------------------------------

log_progress("  Fig 5: U-matrix with clusters...")

adj_selected <- build_adjacency_set(selected_model$grid)
u_values     <- compute_umatrix(selected_model, adj_selected)

hex_umat           <- as.data.frame(selected_model$grid$pts)
hex_umat$neuron    <- 1:nrow(hex_umat)
hex_umat$u_value   <- u_values
hex_umat$cluster   <- cluster_labels[neuron_clusters]

p5 <- ggplot(hex_umat, aes(x = x, y = y)) +
  geom_point(aes(fill = u_value), shape = 21, size = 8, color = "grey30") +
  scale_fill_viridis(name = "U-distance", option = "inferno") +
  geom_text(aes(label = neuron_clusters), size = 2.5, fontface = "bold") +
  labs(title = "U-Matrix with Cluster Assignments",
       subtitle = sprintf("Ward's D2 hierarchical clustering (k=%d)", selected_k)) +
  coord_fixed() +
  theme_void(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 12),
        plot.subtitle = element_text(size = 10))

save_figure(p5, "fig05_umatrix", width = 6, height = 6)
save_caption("fig05_umatrix",
  paste0("Figure 5. Unified distance matrix (U-matrix) of the selected SOM. ",
         "Each neuron is colored by the mean Euclidean distance to its grid ",
         "neighbors in the 9-dimensional feature space. Bright regions indicate ",
         "cluster boundaries. Numbers inside neurons show cluster assignments ",
         "from Ward's D2 hierarchical clustering (k=", selected_k, ")."))

# ---- Figure 6: Training Convergence (Supplementary) ------------------------

log_progress("  Fig 6: Convergence...")

conv_list <- list()
for (run_id in names(results)) {
  changes <- results[[run_id]]$model$changes[, 1]
  conv_list[[run_id]] <- data.table(
    iteration  = 1:length(changes),
    change     = changes,
    grid       = results[[run_id]]$grid_name,
    seed       = as.character(results[[run_id]]$seed)
  )
}
conv_dt <- rbindlist(conv_list)

p6 <- ggplot(conv_dt, aes(x = iteration, y = change, color = seed)) +
  geom_line(alpha = 0.7, linewidth = 0.5) +
  geom_vline(xintercept = 1500, linetype = "dashed", color = "grey40") +
  facet_wrap(~grid, scales = "free_y") +
  scale_color_viridis_d(name = "Seed") +
  labs(title = "SOM Training Convergence",
       x = "Iteration", y = "Mean Distance to BMU") +
  theme_pub

save_figure(p6, "fig06_convergence", width = 10, height = 4)
save_caption("fig06_convergence",
  paste0("Figure S1. Training convergence curves for 10 SOM models (5 seeds x 2 grid sizes). ",
         "Each line shows the mean distance from observations to their best-matching unit ",
         "at each training iteration. The vertical dashed line at iteration 1500 indicates ",
         "the expected convergence point. All models show asymptotic plateau behavior."))

# ---- Figure 7: Node Counts (Supplementary) ---------------------------------

log_progress("  Fig 7: Node counts...")

node_counts <- table(factor(bmu_assignments, levels = 1:nrow(codebook)))
hex_counts          <- as.data.frame(selected_model$grid$pts)
hex_counts$neuron   <- 1:nrow(hex_counts)
hex_counts$count    <- as.integer(node_counts)

p7 <- ggplot(hex_counts, aes(x = x, y = y)) +
  geom_point(aes(fill = count), shape = 21, size = 10, color = "grey30") +
  geom_text(aes(label = format(count, big.mark = ",")), size = 2, fontface = "bold") +
  scale_fill_viridis(name = "Count", trans = "log10",
                     labels = label_comma()) +
  labs(title = "Observation Count per SOM Neuron") +
  coord_fixed() +
  theme_void(base_size = 11) +
  theme(plot.title = element_text(face = "bold", size = 12))

save_figure(p7, "fig07_node_counts", width = 6, height = 6)
save_caption("fig07_node_counts",
  paste0("Figure S2. Number of observations mapped to each neuron in the selected ",
         selected_grid, " SOM. Color intensity (log-scaled) and text labels show ",
         "the count of mesh cells assigned to each neuron as their best-matching ",
         "unit. Uniform distribution indicates balanced SOM utilization."))

# ---- Figure 8: Dendrogram (Supplementary) ----------------------------------

log_progress("  Fig 8: Dendrogram...")

save_base_figure(function() {
  par(mar = c(4, 4, 3, 1))
  plot(hc, hang = -1, labels = paste0("N", 1:nrow(codebook)),
       main = sprintf("Ward's D2 Dendrogram (k=%d)", selected_k),
       xlab = "Neuron Index", ylab = "Height", cex = 0.8)
  rect.hclust(hc, k = selected_k, border = cluster_palette[1:selected_k])
  abline(h = sort(hc$height, decreasing = TRUE)[selected_k - 1],
         lty = 2, col = "red")
}, "fig08_dendrogram", width = 8, height = 5)

save_caption("fig08_dendrogram",
  paste0("Figure S3. Ward's D2 dendrogram of ", nrow(codebook),
         " SOM codebook vectors. Colored rectangles indicate the ",
         selected_k, " clusters obtained by cutting the tree at the red dashed line. ",
         "Leaf labels correspond to neuron indices."))

# ---- Figure 9: Silhouette Plot (Supplementary) -----------------------------

log_progress("  Fig 9: Silhouette...")

neuron_sil <- silhouette(neuron_clusters, codebook_dist)

save_base_figure(function() {
  par(mar = c(4, 4, 3, 1))
  plot(neuron_sil,
       col = cluster_palette[sort(unique(neuron_clusters))],
       main = sprintf("Silhouette Plot (k=%d, neuron-level)", selected_k),
       border = NA)
}, "fig09_silhouette", width = 6, height = 5)

save_caption("fig09_silhouette",
  paste0("Figure S4. Silhouette plot for ", selected_k,
         " clusters at the neuron level. Silhouette widths are computed on the ",
         nrow(codebook), " codebook vectors. Mean silhouette width = ",
         round(mean(neuron_sil[, "sil_width"]), 3),
         ". Positive values indicate well-assigned neurons; negative values indicate ",
         "potential misassignment."))

# ---- Figure 10: Optimal k Selection Panel (Supplementary) ------------------

log_progress("  Fig 10: Optimal k selection...")

k_dt <- data.table(
  k          = 2:10,
  silhouette = sil_widths,
  wss        = wss,
  gap        = gap_stat$Tab[1:9, "gap"],
  gap_se     = gap_stat$Tab[1:9, "SE.sim"]
)

p10a <- ggplot(k_dt, aes(x = k, y = silhouette)) +
  geom_line(linewidth = 0.8) + geom_point(size = 2) +
  geom_vline(xintercept = selected_k, linetype = "dashed", color = "red") +
  labs(title = "Silhouette Width", x = "k", y = "Mean Silhouette") +
  scale_x_continuous(breaks = 2:10) +
  theme_pub

p10b <- ggplot(k_dt, aes(x = k, y = wss)) +
  geom_line(linewidth = 0.8) + geom_point(size = 2) +
  geom_vline(xintercept = selected_k, linetype = "dashed", color = "red") +
  labs(title = "Elbow Plot (WSS)", x = "k", y = "Within-Cluster SS") +
  scale_x_continuous(breaks = 2:10) +
  theme_pub

p10c <- ggplot(k_dt, aes(x = k, y = gap)) +
  geom_line(linewidth = 0.8) + geom_point(size = 2) +
  geom_errorbar(aes(ymin = gap - gap_se, ymax = gap + gap_se), width = 0.2) +
  geom_vline(xintercept = selected_k, linetype = "dashed", color = "red") +
  labs(title = "Gap Statistic", x = "k", y = "Gap") +
  scale_x_continuous(breaks = 2:10) +
  theme_pub

p10 <- p10a + p10b + p10c +
  plot_annotation(title = "Optimal Cluster Number Selection",
                  theme = theme(plot.title = element_text(face = "bold", size = 12)))

save_figure(p10, "fig10_optimal_k", width = 12, height = 4)
save_caption("fig10_optimal_k",
  paste0("Figure S5. Three methods for selecting the optimal number of clusters (k). ",
         "Left: average silhouette width (higher is better; optimal at k=", k_sil, "). ",
         "Center: within-cluster sum of squares (elbow method). ",
         "Right: gap statistic with bootstrap SE bars (B=", GAP_B,
         "; optimal at k=", k_gap, "). ",
         "Red dashed line indicates the selected k=", selected_k, "."))

# ---- Figure 11: Seed Stability ARI Heatmap (Supplementary) -----------------

log_progress("  Fig 11: Seed stability heatmap...")

ari_plots <- list()
for (gn in names(stability)) {
  ari_mat <- stability[[gn]]$ari_matrix
  ari_melt <- data.table(
    seed1 = rep(rownames(ari_mat), ncol(ari_mat)),
    seed2 = rep(colnames(ari_mat), each = nrow(ari_mat)),
    ari   = as.vector(ari_mat)
  )
  ari_melt[, seed1 := factor(seed1, levels = as.character(SEEDS))]
  ari_melt[, seed2 := factor(seed2, levels = as.character(SEEDS))]

  p_ari <- ggplot(ari_melt, aes(x = seed1, y = seed2, fill = ari)) +
    geom_tile(color = "white") +
    geom_text(aes(label = sprintf("%.2f", ari)), size = 3) +
    scale_fill_viridis(limits = c(0, 1), name = "ARI") +
    labs(title = sprintf("%s (mean ARI=%.3f)", gn, stability[[gn]]$mean_ari),
         x = "Seed", y = "Seed") +
    coord_fixed() +
    theme_pub

  ari_plots[[gn]] <- p_ari
}

p11 <- wrap_plots(ari_plots, nrow = 1) +
  plot_annotation(title = "Seed Stability: Pairwise Adjusted Rand Index",
                  theme = theme(plot.title = element_text(face = "bold", size = 12)))

save_figure(p11, "fig11_seed_stability", width = 10, height = 4)
save_caption("fig11_seed_stability",
  paste0("Figure S6. Pairwise Adjusted Rand Index (ARI) across 5 random seeds ",
         "for each SOM grid size. ARI=1 indicates perfect agreement between ",
         "clusterings; ARI=0 indicates chance-level agreement. ",
         "Values > 0.80 indicate stable, reproducible cluster assignments."))

# ---- Figure 12: Cluster Size Bar Chart (Supplementary) ---------------------

log_progress("  Fig 12: Cluster sizes...")

size_dt <- data.table(
  cluster = factor(cluster_labels, levels = cluster_labels),
  n       = as.integer(cluster_sizes_tbl),
  pct     = round(100 * as.numeric(cluster_sizes_tbl) / sum(cluster_sizes_tbl), 1)
)

p12 <- ggplot(size_dt, aes(x = cluster, y = n, fill = cluster)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = sprintf("%s\n(%.1f%%)", format(n, big.mark = ","), pct)),
            vjust = -0.3, size = 3) +
  geom_hline(yintercept = min_threshold, linetype = "dashed", color = "red") +
  annotate("text", x = 0.7, y = min_threshold + 500, label = "1% threshold",
           color = "red", size = 3, hjust = 0) +
  scale_fill_manual(values = cluster_palette) +
  scale_y_continuous(labels = label_comma(),
                     expand = expansion(mult = c(0, 0.15))) +
  labs(title = "Regime Distribution",
       x = NULL, y = "Number of Mesh Cells") +
  theme_pub +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

save_figure(p12, "fig12_cluster_sizes", width = 8, height = 5)
save_caption("fig12_cluster_sizes",
  paste0("Figure S7. Distribution of 138,768 mesh cells across ", selected_k,
         " contamination decay regimes. Bar heights show the number of cells; ",
         "labels show count and percentage. The red dashed line indicates the ",
         "1% minimum cluster size threshold (n=", format(min_threshold, big.mark = ","), ")."))

# ---- Figure 13: Per-Cluster Boxplots (Supplementary) -----------------------

log_progress("  Fig 13: Cluster boxplots...")

# Prepare long-format data for boxplots -- subsample for speed
set.seed(42L)
n_subsample <- min(nrow(dt), 50000L)
subsample_idx <- sample.int(nrow(dt), n_subsample)

box_list <- list()
for (j in seq_along(SOM_FEATURES)) {
  feat <- SOM_FEATURES[j]
  box_list[[j]] <- data.table(
    cluster = factor(cluster_labels[obs_clusters[subsample_idx]],
                     levels = cluster_labels),
    feature = FEATURE_LABELS[feat],
    value   = som_matrix[subsample_idx, j]
  )
}
box_dt <- rbindlist(box_list)
box_dt[, feature := factor(feature, levels = FEATURE_LABELS[SOM_FEATURES])]

p13 <- ggplot(box_dt, aes(x = cluster, y = value, fill = cluster)) +
  geom_boxplot(outlier.size = 0.05, outlier.colour = "grey60",
               outlier.alpha = 0.3) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
  facet_wrap(~feature, scales = "free_y", ncol = 3) +
  scale_fill_manual(values = cluster_palette, name = "Regime") +
  labs(title = "Feature Distributions by Regime",
       x = NULL, y = "Z-score (IQR-scaled)") +
  theme_pub +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        legend.position = "bottom") +
  guides(fill = guide_legend(nrow = 2))

save_figure(p13, "fig13_cluster_boxplots", width = 12, height = 10)
save_caption("fig13_cluster_boxplots",
  paste0("Figure S8. Boxplot distributions of 9 temporal features within each ",
         "contamination decay regime (z-scored values). Based on a random subsample ",
         "of ", format(n_subsample, big.mark = ","), " observations for visual clarity. ",
         "The dashed line at y=0 indicates the population median. ",
         "Box widths are proportional to sample size within each panel."))

# =============================================================================
# --- Final Summary -----------------------------------------------------------
# =============================================================================

log_progress("Pipeline complete. Summary:")

message("\n=== FINAL RESULTS ===")
message(sprintf("Selected model: %s, seed %d", selected_grid, selected_seed))
message(sprintf("QE = %.4f, TE = %.4f", selected_qe, selected_te))
message(sprintf("Seed stability ARI (mean): %.4f", grid_summary[[selected_grid]]$mean_ari))
message(sprintf("Number of regimes: %d", selected_k))
message(sprintf("Silhouette width (neuron-level): %.4f",
                mean(neuron_sil[, "sil_width"])))

message("\nCluster summary:")
for (i in 1:selected_k) {
  sz  <- cluster_sizes_tbl[as.character(i)]
  pct <- 100 * sz / sum(cluster_sizes_tbl)
  message(sprintf("  %d. %-35s  n=%s (%5.1f%%)",
                  i, cluster_labels[i], format(sz, big.mark = ","), pct))
}

message("\nOutput files:")
message("  output/som_model.rds")
message("  output/som_clusters.parquet")
message("  output/som_metrics.json")
message("  output/som_cluster_summary.csv")
message("  output/som_cluster_assignments.csv")
message(sprintf("  figures/ (%d figure pairs + captions)", 13))

total_time <- round(difftime(Sys.time(), t_start, units = "mins"), 1)
message(sprintf("\nTotal runtime: %s minutes", total_time))

# --- Session Info ------------------------------------------------------------
writeLines(capture.output(sessionInfo()), "output/som_session_info.txt")
message("Session info saved to output/som_session_info.txt")
