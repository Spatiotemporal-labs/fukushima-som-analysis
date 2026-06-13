# =============================================================================
# Script: som_finalize_k4.R
# Purpose: Finalize the 8-feature SOM at k=4. Re-cut ONLY (no retraining): load
#          the trained model, Ward.D2 cut at k=4, apply the manual regime labels,
#          write the primary artifacts, and regenerate the SOM-derived figures.
# Input:   output/som_model_sensitivity.rds
#          output/som_clusters_sensitivity.parquet
#          output/feature_summary.json
#          output/som_metrics_sensitivity.json  (carry over k_selection)
# Output:  PRIMARY:  output/som_clusters.parquet, output/som_cluster_summary.csv,
#                     output/som_model.rds, output/som_metrics.json
#          UPDATED:  output/som_clusters_sensitivity.parquet,
#                     output/som_cluster_summary_sensitivity.csv,
#                     output/som_metrics_sensitivity.json (clusters -> 4 only)
#          FIGURES:  figures/fig02..fig14 SOM-derived (PNG 300dpi + PDF + captions)
# NOTE:    Does NOT retrain.
# =============================================================================

# --- Packages ----------------------------------------------------------------
suppressPackageStartupMessages({
  library(arrow)
  library(data.table)
  library(kohonen)
  library(cluster)
  library(ggplot2)
  library(viridis)
  library(patchwork)
  library(jsonlite)
  library(ggspatial)
  library(RColorBrewer)
  library(scales)
})

# --- Configuration -----------------------------------------------------------
K_FINAL      <- 4L
MIN_CLUSTER_PCT <- 0.01
FDNPP_LAT    <- 37.4211
FDNPP_LON    <- 141.0328
GAP_B        <- 100L

# 8-feature set: 7 z-scored decay descriptors plus derived n_increases_rate
SOM_FEATURES <- c(
  "total_decay_ratio_z", "log_linear_slope_z", "log_linear_r2_z",
  "early_slope_z", "late_slope_z", "residual_cv_z",
  "slope_ratio_z", "n_increases_rate_z"
)

FEATURE_LABELS <- c(
  "Total Decay Ratio", "Log-Linear Slope", "Log-Linear R²",
  "Early Phase Slope", "Late Phase Slope", "Residual CV",
  "Early/Late Slope Ratio", "Number of Increases (rate)"
)
names(FEATURE_LABELS) <- SOM_FEATURES

# LOCKED manual labels in cluster_id order (Slow, Standard, NearField, Anomalous).
# cluster_id is assigned by the cutree IDs which (validated) already align 1:1.
CLUSTER_LABELS <- c(
  "Slow Decay",                # cluster_id 1
  "Standard Decay",            # cluster_id 2
  "Near-Field Hotspot",        # cluster_id 3
  "Irregular Decay"            # cluster_id 4
)

# Expected sizes for the +/-0.2% assertion (from locked spec)
EXPECTED_N   <- c(28737L, 82994L, 19841L, 7196L)
EXPECTED_PCT <- c(20.7, 59.8, 14.3, 5.2)
SIZE_TOL_PCT <- 0.2  # absolute percentage-point tolerance on pct-of-total

# 4-colour palette (Set2[1:4]) as required.
cluster_palette <- RColorBrewer::brewer.pal(8, "Set2")[1:4]

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

# --- Helper functions (figure I/O) -------------------------------------------
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
  plot_fn(); dev.off()
  pdf(paste0("figures/", filename, ".pdf"), width = width, height = height)
  plot_fn(); dev.off()
  message(sprintf("  Saved figures/%s.{png,pdf}", filename))
}

save_caption <- function(filename, caption_text) {
  writeLines(caption_text, paste0("figures/", filename, "_caption.txt"))
}

# --- Helper functions (SOM, VERBATIM from som_clustering.R) -------------------
build_adjacency_set <- function(grid) {
  pts <- grid$pts
  n_neurons <- nrow(pts)
  adj <- vector("list", n_neurons)
  for (i in 1:n_neurons) {
    dists <- sqrt((pts[, 1] - pts[i, 1])^2 + (pts[, 2] - pts[i, 2])^2)
    adj[[i]] <- which(dists > 0 & dists < 1.1)
  }
  adj
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

# Back-transform z-centroids to original units (same handling as sensitivity script)
compute_cluster_centroids <- function(som_matrix, obs_clusters, norm_params, som_features) {
  cluster_ids <- sort(unique(obs_clusters))
  centroids_z <- t(sapply(cluster_ids, function(cl) {
    colMeans(som_matrix[obs_clusters == cl, , drop = FALSE])
  }))
  colnames(centroids_z) <- som_features
  rownames(centroids_z) <- paste0("Cluster_", cluster_ids)

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

# =============================================================================
# --- Step 1: Load trained map, re-cut at k=4, apply LOCKED labels ------------
# =============================================================================
log_progress("Loading trained sensitivity model and cell assignments...")

selected_model <- tryCatch(
  readRDS("output/som_model_sensitivity.rds"),
  error = function(e) stop("Failed to read som_model_sensitivity.rds: ", conditionMessage(e))
)
dt <- tryCatch(
  as.data.table(read_parquet("output/som_clusters_sensitivity.parquet")),
  error = function(e) stop("Failed to read som_clusters_sensitivity.parquet: ", conditionMessage(e))
)

stopifnot(nrow(dt) == 138768L)
stopifnot(all(SOM_FEATURES %in% names(dt)))
stopifnot("som_bmu" %in% names(dt))
stopifnot(all(c("n_increases_rate", "n_increases_rate_z") %in% names(dt)))

som_matrix <- as.matrix(dt[, ..SOM_FEATURES])

# Ward.D2 on codebook, Euclidean distance; identical to the validated recut.
log_progress("Ward.D2 hierarchical clustering on codebook; cutting at k=4...")
codebook      <- selected_model$codes[[1]]
codebook_dist <- dist(codebook, method = "euclidean")
hc            <- hclust(codebook_dist, method = "ward.D2")

neuron_clusters <- cutree(hc, k = K_FINAL)
bmu_assignments <- dt$som_bmu
obs_clusters    <- neuron_clusters[bmu_assignments]

# Diagnostics for k-selection figures (silhouette/wss/gap over k=2..10).
sil_widths <- sapply(2:10, function(k) {
  labels <- cutree(hc, k = k)
  mean(silhouette(labels, codebook_dist)[, "sil_width"])
})
names(sil_widths) <- 2:10
k_sil <- as.integer(names(which.max(sil_widths)))

set.seed(42L)
gap_stat <- clusGap(codebook, FUN = function(x, k) {
  list(cluster = cutree(hclust(dist(x), method = "ward.D2"), k = k))
}, K.max = 10, B = GAP_B)
k_gap <- maxSE(gap_stat$Tab[, "gap"], gap_stat$Tab[, "SE.sim"], method = "Tibs2001SEmax")

wss <- sapply(2:10, function(k) {
  labels <- cutree(hc, k = k)
  sum(sapply(unique(labels), function(cl) {
    members <- codebook[labels == cl, , drop = FALSE]
    sum(scale(members, scale = FALSE)^2)
  }))
})
names(wss) <- 2:10

# --- Verify cutree id ordering matches the locked signatures -----------------
log_progress("Verifying cluster signatures against locked spec...")

cluster_sizes_tbl <- table(factor(obs_clusters, levels = 1:K_FINAL))
sig_z <- t(sapply(1:K_FINAL, function(cl) {
  colMeans(som_matrix[obs_clusters == cl, , drop = FALSE])
}))
colnames(sig_z) <- SOM_FEATURES

mean_dose <- sapply(1:K_FINAL, function(cl) mean(dt$initial_dose_rate[obs_clusters == cl]))
mean_dist <- sapply(1:K_FINAL, function(cl) mean(dt$distance_from_fdnpp_km[obs_clusters == cl]))

# Signature assertions (locked spec). cutree id order is expected to be
# 1=Slow, 2=Standard, 3=NearField, 4=Anomalous.
which_slow      <- which.max(sig_z[, "total_decay_ratio_z"])
which_nearfield <- which.max(sig_z[, "residual_cv_z"])
which_anom      <- which.min(sig_z[, "log_linear_r2_z"])
which_standard  <- which.min(rowSums(abs(sig_z)))  # modal: all features near 0

message(sprintf("  Highest total_decay_ratio_z (Slow)      -> cutree id %d", which_slow))
message(sprintf("  Highest residual_cv_z      (Near-Field) -> cutree id %d", which_nearfield))
message(sprintf("  Lowest  log_linear_r2_z    (Anomalous)  -> cutree id %d", which_anom))
message(sprintf("  Modal (min sum|z|)         (Standard)   -> cutree id %d", which_standard))

stopifnot(which_slow      == 1L)
stopifnot(which_standard  == 2L)
stopifnot(which_nearfield == 3L)
stopifnot(which_anom      == 4L)
# Near-Field corroborating signatures: highest dose, closest to FDNPP.
stopifnot(which.max(mean_dose) == 3L)
stopifnot(which.min(mean_dist) == 3L)
# Anomalous corroborating signature: highest slope_ratio_z.
stopifnot(which.max(sig_z[, "slope_ratio_z"]) == 4L)

message("  Signature checks passed: cutree ids align 1=Slow 2=Standard 3=NearField 4=Anomalous.")

# --- Apply LOCKED manual labels (cluster_id == cutree id) --------------------
cluster_labels <- CLUSTER_LABELS  # already in cluster_id order 1..4

# --- ASSERT sizes within +/-0.2 percentage points of locked spec -------------
obs_n   <- as.integer(cluster_sizes_tbl[as.character(1:K_FINAL)])
obs_pct <- 100 * obs_n / sum(obs_n)
message("\n=== k=4 cluster sizes (cluster_id order) ===")
for (i in 1:K_FINAL) {
  message(sprintf("  %d. %-26s n=%s (%.2f%%)  [expected n=%s (%.1f%%)]",
                  i, cluster_labels[i], format(obs_n[i], big.mark = ","),
                  obs_pct[i], format(EXPECTED_N[i], big.mark = ","), EXPECTED_PCT[i]))
}
pct_dev <- abs(obs_pct - EXPECTED_PCT)
if (any(pct_dev > SIZE_TOL_PCT)) {
  stop(sprintf("SIZE ASSERTION FAILED: pct deviation %s exceeds +/-%.1f pp tolerance.",
               paste(sprintf("%.3f", pct_dev), collapse = ", "), SIZE_TOL_PCT))
}
message(sprintf("SIZE ASSERTION PASSED: max pct deviation = %.3f pp (tol +/-%.1f pp).",
                max(pct_dev), SIZE_TOL_PCT))

# =============================================================================
# --- Back-transformed centroids (reuse normalization handling) ---------------
# =============================================================================
log_progress("Computing back-transformed centroids...")

params      <- fromJSON("output/feature_summary.json")
norm_params <- params$normalization_params
# Inject derived n_increases_rate params EXACTLY as som_clustering_sensitivity.R does.
norm_params[["n_increases_rate"]] <- list(median = 0.2381, iqr = 0.1310)

centroids <- compute_cluster_centroids(som_matrix, obs_clusters, norm_params, SOM_FEATURES)

message("\n=== Back-transformed centroids (key features) ===")
for (i in 1:K_FINAL) {
  message(sprintf("  %d. %-26s tdr=%.4f resid_cv=%.4f slope_ratio=%.4f r2=%.4f ninc_rate=%.4f",
                  i, cluster_labels[i],
                  centroids$original[i, "total_decay_ratio"],
                  centroids$original[i, "residual_cv"],
                  centroids$original[i, "slope_ratio"],
                  centroids$original[i, "log_linear_r2"],
                  centroids$original[i, "n_increases_rate"]))
}

# =============================================================================
# --- Step 2: Write PRIMARY + updated _sensitivity artifacts -------------------
# =============================================================================
log_progress("Writing PRIMARY and updated _sensitivity artifacts...")

# Assign final columns to dt (drop any stale sensitivity assignment cols first).
dt[, som_cluster       := obs_clusters]
dt[, som_cluster_label := cluster_labels[obs_clusters]]

# Build the cluster summary data.table (shared schema for primary + sensitivity).
build_summary_dt <- function() {
  s <- data.table(
    cluster_id     = 1:K_FINAL,
    cluster_label  = cluster_labels,
    n_observations = obs_n,
    pct_of_total   = round(obs_pct, 2)
  )
  for (j in 1:ncol(centroids$original)) {
    col_name <- colnames(centroids$original)[j]
    s[, (col_name) := signif(centroids$original[, j], 4)]
  }
  for (i in 1:K_FINAL) {
    mask <- obs_clusters == i
    s[i, mean_ecological_half_life_y := round(mean(dt$ecological_half_life_y[mask], na.rm = TRUE), 2)]
    s[i, mean_distance_from_fdnpp_km := round(mean(dt$distance_from_fdnpp_km[mask], na.rm = TRUE), 2)]
    s[i, mean_initial_dose_rate := signif(mean(dt$initial_dose_rate[mask], na.rm = TRUE), 4)]
  }
  s
}
summary_dt <- build_summary_dt()

# --- PRIMARY: som_model.rds (copy of sensitivity model) ----------------------
saveRDS(selected_model, "output/som_model.rds")
message("  Wrote output/som_model.rds")

# --- PRIMARY + updated _sensitivity: clusters parquet (same dt) --------------
write_parquet(dt, "output/som_clusters.parquet")
message("  Wrote output/som_clusters.parquet")
write_parquet(dt, "output/som_clusters_sensitivity.parquet")
message("  Wrote output/som_clusters_sensitivity.parquet")

# --- PRIMARY + updated _sensitivity: cluster summary csv ---------------------
fwrite(summary_dt, "output/som_cluster_summary.csv")
message("  Wrote output/som_cluster_summary.csv")
fwrite(summary_dt, "output/som_cluster_summary_sensitivity.csv")
message("  Wrote output/som_cluster_summary_sensitivity.csv")

# --- Build the 4-cluster JSON array (shared) ---------------------------------
clusters_json <- lapply(1:K_FINAL, function(i) {
  list(
    cluster_id        = i,
    label             = cluster_labels[i],
    n_observations    = obs_n[i],
    pct_of_total      = round(obs_pct[i], 2),
    centroid_z        = as.list(round(centroids$z_scored[i, ], 4)),
    centroid_original = as.list(signif(centroids$original[i, ], 4)),
    neurons_in_cluster = which(neuron_clusters == i)
  )
})

# --- Read existing sensitivity metrics to (a) carry blocks into primary,
#     and (b) update its clusters array in place. ------------------------------
sens_metrics <- fromJSON("output/som_metrics_sensitivity.json", simplifyVector = FALSE)
k_selection_block  <- sens_metrics$k_selection
stopifnot(!is.null(k_selection_block))

# --- Update _sensitivity metrics: replace clusters array (->4), set k_selected=4,
#     keep k_selection intact. ------------------------------------------------
sens_metrics$generated_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
sens_metrics$clusters <- clusters_json
if (!is.null(sens_metrics$hierarchical_clustering)) {
  sens_metrics$hierarchical_clustering$k_selected <- K_FINAL
}
write_json(sens_metrics, "output/som_metrics_sensitivity.json",
           pretty = TRUE, auto_unbox = TRUE)
message("  Wrote output/som_metrics_sensitivity.json (clusters -> 4; k_selection preserved)")

# --- PRIMARY: som_metrics.json -----------------------------------------------
# n_features=8, 8 feature names, training_config, selected-model block,
# hierarchical_clustering with k_selected=4, clusters array(4),
# carry over k_selection verbatim from the training metrics.
primary_metrics <- list(
  generated_at   = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
  pipeline_phase = "4_som_clustering",
  run_type       = "som_finalize_k4",
  input_file     = "output/feature_matrix.parquet",
  n_observations = nrow(dt),
  n_features     = length(SOM_FEATURES),
  feature_names  = SOM_FEATURES,
  training_config = sens_metrics$training_config,
  selected_model  = sens_metrics$selected_model,
  hierarchical_clustering = list(
    method            = "ward.D2",
    distance          = "euclidean",
    k_selected        = K_FINAL,
    k_silhouette      = k_sil,
    k_gap             = as.integer(k_gap),
    silhouette_widths = as.list(round(sil_widths, 4)),
    gap_values        = as.list(round(gap_stat$Tab[, "gap"], 4)),
    wss_values        = as.list(round(wss, 4))
  ),
  clusters     = clusters_json,
  k_selection  = k_selection_block
)
write_json(primary_metrics, "output/som_metrics.json",
           pretty = TRUE, auto_unbox = TRUE)
message("  Wrote output/som_metrics.json (n_features=8, k=4, k_selection carried over)")

# =============================================================================
# --- Step 3: Regenerate SOM-derived figures at k=4 ---------------------------
#     Current manuscript naming scheme (fig01 = workflow, NOT regenerated):
#       fig02_geographic_regime_map  (HERO)   <- som_clustering.R fig01 logic
#       fig03_centroid_heatmap                <- fig02 logic
#       fig04_parallel_coordinates            <- fig03 logic
#       fig05_component_planes                <- fig04 logic
#       fig06_umatrix                         <- fig05 logic
#       fig07_convergence                     <- fig06 logic
#       fig08_node_counts                     <- fig07 logic
#       fig09_dendrogram                      <- fig08 logic
#       fig10_silhouette                      <- fig09 logic
#       fig11_optimal_k                       <- fig10 logic
#       fig12_seed_stability                  <- fig11 logic
#       fig13_cluster_sizes                   <- fig12 logic
#       fig14_cluster_boxplots                <- fig13 logic
# =============================================================================
log_progress("Regenerating SOM-derived figures at k=4...")

selected_grid <- "5x5"
min_threshold <- ceiling(MIN_CLUSTER_PCT * nrow(dt))

# Ordered factor for consistent plotting/colour mapping.
dt[, som_cluster_label := factor(som_cluster_label, levels = cluster_labels)]

failed_figs <- character(0)
try_fig <- function(name, expr) {
  tryCatch(expr, error = function(e) {
    message(sprintf("  [FAIL] %s: %s", name, conditionMessage(e)))
    failed_figs[[length(failed_figs) + 1]] <<- name
  })
}

# ---- fig02_geographic_regime_map (HERO) -------------------------------------
try_fig("fig02_geographic_regime_map", {
  log_progress("  fig02: Geographic regime map (HERO)...")
  # Initial 20 km evacuation radius (March 2011), drawn as a geodesic ring.
  ring_t  <- seq(0, 2 * pi, length.out = 240)
  ring_df <- data.frame(
    longitude = FDNPP_LON + (20 / (111.32 * cos(FDNPP_LAT * pi / 180))) * cos(ring_t),
    latitude  = FDNPP_LAT + (20 / 111.32) * sin(ring_t)
  )
  p1 <- ggplot(dt, aes(x = longitude, y = latitude, color = som_cluster_label)) +
    geom_point(size = 0.1, alpha = 0.6) +
    scale_color_manual(values = cluster_palette, name = "Regime") +
    geom_path(data = ring_df, aes(x = longitude, y = latitude),
              inherit.aes = FALSE, linetype = "dashed",
              color = "black", linewidth = 0.4) +
    annotate("point", x = FDNPP_LON, y = FDNPP_LAT,
             shape = 17, size = 3, color = "black") +
    annotate("text", x = FDNPP_LON, y = FDNPP_LAT + 0.02,
             label = "FDNPP", fontface = "bold", size = 3) +
    annotate("text", x = FDNPP_LON, y = FDNPP_LAT + (20 / 111.32) + 0.018,
             label = "20 km", size = 2.5) +
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
  save_figure(p1, "fig02_geographic_regime_map", width = 8, height = 10)
  save_caption("fig02_geographic_regime_map",
    paste0("Figure 2. Geographic distribution of contamination decay regimes ",
           "identified by Self-Organizing Map clustering across 138,768 mesh cells ",
           "in the Fukushima region. Each point represents a 250-m grid cell colored ",
           "by its assigned regime. The black triangle marks the Fukushima Daiichi ",
           "Nuclear Power Station (FDNPP), and the dashed circle the initial 20 km ",
           "evacuation radius. Regimes were determined by hierarchical ",
           "clustering (Ward's D2, k=", K_FINAL, ") of a ", selected_grid,
           " hexagonal SOM trained on 8 temporal features."))
})

# ---- fig03_centroid_heatmap -------------------------------------------------
try_fig("fig03_centroid_heatmap", {
  log_progress("  fig03: Centroid heatmap...")
  heatmap_dt <- data.table(
    cluster = rep(paste0(cluster_labels, "\n(n=",
                         format(obs_n, big.mark = ","), ")"),
                  each = length(SOM_FEATURES)),
    feature = rep(FEATURE_LABELS[SOM_FEATURES], K_FINAL),
    value   = as.vector(t(centroids$z_scored))
  )
  heatmap_dt[, cluster := factor(cluster, levels = unique(cluster))]
  heatmap_dt[, feature := factor(feature, levels = rev(FEATURE_LABELS[SOM_FEATURES]))]
  p2 <- ggplot(heatmap_dt, aes(x = feature, y = cluster, fill = value)) +
    geom_tile(color = "white", linewidth = 0.5) +
    geom_text(aes(label = sprintf("%.2f", value)), size = 3) +
    scale_fill_gradient2(low = "#2166AC", mid = "white", high = "#B2182B",
                         midpoint = 0, name = "Z-score") +
    labs(title = "Cluster Centroid Profiles (Z-scored)", x = NULL, y = NULL) +
    theme_pub +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  save_figure(p2, "fig03_centroid_heatmap", width = 10, height = 5)
  save_caption("fig03_centroid_heatmap",
    paste0("Figure 3. Heatmap of cluster centroid z-scores across 8 temporal features. ",
           "Rows represent ", K_FINAL, " contamination decay regimes; columns represent ",
           "SOM input features (IQR-scaled). Blue indicates below-median values; ",
           "red indicates above-median. Numbers show exact z-score values. ",
           "Cluster sizes (n) are annotated in row labels."))
})

# ---- fig04_parallel_coordinates ---------------------------------------------
try_fig("fig04_parallel_coordinates", {
  log_progress("  fig04: Parallel coordinates...")
  par_dt <- data.table(
    cluster = rep(cluster_labels, each = length(SOM_FEATURES)),
    feature = rep(FEATURE_LABELS[SOM_FEATURES], K_FINAL),
    value   = as.vector(t(centroids$z_scored))
  )
  par_dt[, cluster := factor(cluster, levels = cluster_labels)]
  par_dt[, feature := factor(feature, levels = FEATURE_LABELS[SOM_FEATURES])]
  p3 <- ggplot(par_dt, aes(x = feature, y = value, color = cluster, group = cluster)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    geom_line(linewidth = 1.2) +
    geom_point(size = 2.5) +
    scale_color_manual(values = cluster_palette, name = "Regime") +
    labs(title = "Regime Profiles Across Temporal Features",
         x = NULL, y = "Z-score (IQR-scaled)") +
    theme_pub +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  save_figure(p3, "fig04_parallel_coordinates", width = 10, height = 5)
  save_caption("fig04_parallel_coordinates",
    paste0("Figure 4. Parallel coordinates plot showing centroid profiles of ",
           K_FINAL, " contamination decay regimes across 8 temporal features. ",
           "Each line represents one regime. The dashed horizontal line at y=0 ",
           "indicates the population median. Divergence from zero reflects how ",
           "each regime differs from the typical decay behavior."))
})

# ---- fig05_component_planes -------------------------------------------------
try_fig("fig05_component_planes", {
  log_progress("  fig05: Component planes...")
  hex_pts <- as.data.frame(selected_model$grid$pts)
  hex_pts$neuron <- 1:nrow(hex_pts)
  comp_plots <- list()
  for (j in seq_along(SOM_FEATURES)) {
    hex_pts$value <- codebook[, j]
    comp_plots[[j]] <- ggplot(hex_pts, aes(x = x, y = y, fill = value)) +
      geom_point(shape = 21, size = 6, color = "grey30") +
      scale_fill_viridis(name = "Z") +
      labs(title = FEATURE_LABELS[SOM_FEATURES[j]]) +
      coord_fixed() +
      theme_void(base_size = 9) +
      theme(plot.title = element_text(face = "bold", size = 9, hjust = 0.5),
            legend.key.height = unit(0.4, "cm"),
            legend.key.width  = unit(0.3, "cm"))
  }
  p4 <- wrap_plots(comp_plots, ncol = 3) +
    plot_annotation(title = "SOM Component Planes",
                    theme = theme(plot.title = element_text(face = "bold", size = 12)))
  save_figure(p4, "fig05_component_planes", width = 10, height = 10)
  save_caption("fig05_component_planes",
    paste0("Figure 5. Component planes of the ", selected_grid,
           " hexagonal Self-Organizing Map. Each panel shows one of the 8 temporal ",
           "features, with neuron positions reflecting the SOM topology and color ",
           "indicating the codebook value (z-scored). Spatial patterns across panels ",
           "reveal correlated and independent feature dimensions."))
})

# ---- fig06_umatrix ----------------------------------------------------------
try_fig("fig06_umatrix", {
  log_progress("  fig06: U-matrix with clusters...")
  adj_selected <- build_adjacency_set(selected_model$grid)
  u_values     <- compute_umatrix(selected_model, adj_selected)
  hex_umat <- as.data.frame(selected_model$grid$pts)
  hex_umat$neuron  <- 1:nrow(hex_umat)
  hex_umat$u_value <- u_values
  hex_umat$cluster <- cluster_labels[neuron_clusters]
  p5 <- ggplot(hex_umat, aes(x = x, y = y)) +
    geom_point(aes(fill = u_value), shape = 21, size = 8, color = "grey30") +
    scale_fill_viridis(name = "U-distance", option = "inferno") +
    geom_text(aes(label = neuron_clusters), size = 2.5, fontface = "bold") +
    labs(title = "U-Matrix with Cluster Assignments",
         subtitle = sprintf("Ward's D2 hierarchical clustering (k=%d)", K_FINAL)) +
    coord_fixed() +
    theme_void(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 12),
          plot.subtitle = element_text(size = 10))
  save_figure(p5, "fig06_umatrix", width = 6, height = 6)
  save_caption("fig06_umatrix",
    paste0("Figure 6. Unified distance matrix (U-matrix) of the selected SOM. ",
           "Each neuron is colored by the mean Euclidean distance to its grid ",
           "neighbors in the 8-dimensional feature space. Bright regions indicate ",
           "cluster boundaries. Numbers inside neurons show cluster assignments ",
           "from Ward's D2 hierarchical clustering (k=", K_FINAL, ")."))
})

# ---- fig07_convergence ------------------------------------------------------
try_fig("fig07_convergence", {
  log_progress("  fig07: Convergence...")
  changes <- selected_model$changes[, 1]
  conv_dt <- data.table(iteration = seq_along(changes), change = changes)
  p6 <- ggplot(conv_dt, aes(x = iteration, y = change)) +
    geom_line(alpha = 0.85, linewidth = 0.6, color = "#1b9e77") +
    geom_vline(xintercept = 1500, linetype = "dashed", color = "grey40") +
    labs(title = "SOM Training Convergence (Selected Model)",
         x = "Iteration", y = "Mean Distance to BMU") +
    theme_pub
  save_figure(p6, "fig07_convergence", width = 10, height = 4)
  save_caption("fig07_convergence",
    paste0("Figure S1. Training convergence curve for the selected 5x5 SOM ",
           "(8-feature set). The line shows the mean distance ",
           "from observations to their best-matching unit at each training iteration. ",
           "The vertical dashed line at iteration 1500 indicates the expected ",
           "convergence point; the curve shows asymptotic plateau behavior."))
})

# ---- fig08_node_counts ------------------------------------------------------
try_fig("fig08_node_counts", {
  log_progress("  fig08: Node counts...")
  node_counts <- table(factor(bmu_assignments, levels = 1:nrow(codebook)))
  hex_counts <- as.data.frame(selected_model$grid$pts)
  hex_counts$neuron <- 1:nrow(hex_counts)
  hex_counts$count  <- as.integer(node_counts)
  p7 <- ggplot(hex_counts, aes(x = x, y = y)) +
    geom_point(aes(fill = count), shape = 21, size = 10, color = "grey30") +
    geom_text(aes(label = format(count, big.mark = ",")), size = 2, fontface = "bold") +
    scale_fill_viridis(name = "Count", trans = "log10", labels = label_comma()) +
    labs(title = "Observation Count per SOM Neuron") +
    coord_fixed() +
    theme_void(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 12))
  save_figure(p7, "fig08_node_counts", width = 6, height = 6)
  save_caption("fig08_node_counts",
    paste0("Figure S2. Number of observations mapped to each neuron in the selected ",
           selected_grid, " SOM. Color intensity (log-scaled) and text labels show ",
           "the count of mesh cells assigned to each neuron as their best-matching ",
           "unit. Uniform distribution indicates balanced SOM utilization."))
})

# ---- fig09_dendrogram (mark k=4 cut) ----------------------------------------
try_fig("fig09_dendrogram", {
  log_progress("  fig09: Dendrogram (k=4 cut marked)...")
  save_base_figure(function() {
    par(mar = c(4, 4, 3, 1))
    plot(hc, hang = -1, labels = paste0("N", 1:nrow(codebook)),
         main = sprintf("Ward's D2 Dendrogram (k=%d)", K_FINAL),
         xlab = "Neuron Index", ylab = "Height", cex = 0.8)
    rect.hclust(hc, k = K_FINAL, border = cluster_palette[1:K_FINAL])
    abline(h = sort(hc$height, decreasing = TRUE)[K_FINAL - 1], lty = 2, col = "red")
  }, "fig09_dendrogram", width = 8, height = 5)
  save_caption("fig09_dendrogram",
    paste0("Figure S3. Ward's D2 dendrogram of ", nrow(codebook),
           " SOM codebook vectors. Colored rectangles indicate the ", K_FINAL,
           " clusters obtained by cutting the tree at the red dashed line (k=", K_FINAL,
           "). Leaf labels correspond to neuron indices."))
})

# ---- fig10_silhouette (mark k=4) --------------------------------------------
try_fig("fig10_silhouette", {
  log_progress("  fig10: Silhouette (k=4)...")
  neuron_sil <- silhouette(neuron_clusters, codebook_dist)
  mean_sw <- round(mean(neuron_sil[, "sil_width"]), 3)
  save_base_figure(function() {
    par(mar = c(4, 4, 3, 1))
    plot(neuron_sil,
         col = cluster_palette[sort(unique(neuron_clusters))],
         main = sprintf("Silhouette Plot (k=%d, neuron-level)", K_FINAL),
         border = NA)
  }, "fig10_silhouette", width = 6, height = 5)
  save_caption("fig10_silhouette",
    paste0("Figure S4. Silhouette plot for k=", K_FINAL,
           " clusters at the neuron level. Silhouette widths are computed on the ",
           nrow(codebook), " codebook vectors. Mean silhouette width = ", mean_sw,
           ". Positive values indicate well-assigned neurons; negative values indicate ",
           "potential misassignment."))
})

# ---- fig11_optimal_k (mark k=4) ---------------------------------------------
try_fig("fig11_optimal_k", {
  log_progress("  fig11: Optimal k selection (k=4 marked)...")
  k_dt <- data.table(
    k          = 2:10,
    silhouette = as.numeric(sil_widths),
    wss        = as.numeric(wss),
    gap        = gap_stat$Tab[1:9, "gap"],
    gap_se     = gap_stat$Tab[1:9, "SE.sim"]
  )
  p10a <- ggplot(k_dt, aes(x = k, y = silhouette)) +
    geom_line(linewidth = 0.8) + geom_point(size = 2) +
    geom_vline(xintercept = K_FINAL, linetype = "dashed", color = "red") +
    labs(title = "Silhouette Width", x = "k", y = "Mean Silhouette") +
    scale_x_continuous(breaks = 2:10) + theme_pub
  p10b <- ggplot(k_dt, aes(x = k, y = wss)) +
    geom_line(linewidth = 0.8) + geom_point(size = 2) +
    geom_vline(xintercept = K_FINAL, linetype = "dashed", color = "red") +
    labs(title = "Elbow Plot (WSS)", x = "k", y = "Within-Cluster SS") +
    scale_x_continuous(breaks = 2:10) + theme_pub
  p10c <- ggplot(k_dt, aes(x = k, y = gap)) +
    geom_line(linewidth = 0.8) + geom_point(size = 2) +
    geom_errorbar(aes(ymin = gap - gap_se, ymax = gap + gap_se), width = 0.2) +
    geom_vline(xintercept = K_FINAL, linetype = "dashed", color = "red") +
    labs(title = "Gap Statistic", x = "k", y = "Gap") +
    scale_x_continuous(breaks = 2:10) + theme_pub
  p10 <- p10a + p10b + p10c +
    plot_annotation(title = "Optimal Cluster Number Selection",
                    theme = theme(plot.title = element_text(face = "bold", size = 12)))
  save_figure(p10, "fig11_optimal_k", width = 12, height = 4)
  save_caption("fig11_optimal_k",
    paste0("Figure S5. Three methods for selecting the number of clusters (k). ",
           "Left: average silhouette width (statistical optimum at k=", k_sil, "). ",
           "Center: within-cluster sum of squares (elbow method). ",
           "Right: gap statistic with bootstrap SE bars (B=", GAP_B,
           "; optimum at k=", as.integer(k_gap), "). ",
           "The red dashed line marks the selected k=", K_FINAL,
           ", chosen as the smallest k that resolves the near-field high-dose ",
           "regime at negligible silhouette cost (see Results)."))
})

# ---- fig12_seed_stability ---------------------------------------------------
try_fig("fig12_seed_stability", {
  log_progress("  fig12: Seed stability heatmap...")
  sens <- fromJSON("output/som_metrics_sensitivity.json", simplifyVector = TRUE)
  gc_block <- sens$grid_comparison
  ari_plots <- list()
  for (gn in names(gc_block)) {
    ari_mat <- gc_block[[gn]]$ari_matrix
    seeds_lbl <- as.character(sens$training_config$seeds)
    rownames(ari_mat) <- seeds_lbl
    colnames(ari_mat) <- seeds_lbl
    ari_melt <- data.table(
      seed1 = rep(rownames(ari_mat), ncol(ari_mat)),
      seed2 = rep(colnames(ari_mat), each = nrow(ari_mat)),
      ari   = as.vector(ari_mat)
    )
    ari_melt[, seed1 := factor(seed1, levels = seeds_lbl)]
    ari_melt[, seed2 := factor(seed2, levels = seeds_lbl)]
    ari_plots[[gn]] <- ggplot(ari_melt, aes(x = seed1, y = seed2, fill = ari)) +
      geom_tile(color = "white") +
      geom_text(aes(label = sprintf("%.2f", ari)), size = 3) +
      scale_fill_viridis(limits = c(0, 1), name = "ARI") +
      labs(title = sprintf("%s (mean ARI=%.3f)", gn, gc_block[[gn]]$mean_ari),
           x = "Seed", y = "Seed") +
      coord_fixed() + theme_pub
  }
  p11 <- wrap_plots(ari_plots, nrow = 1) +
    plot_annotation(title = "Seed Stability: Pairwise Adjusted Rand Index",
                    theme = theme(plot.title = element_text(face = "bold", size = 12)))
  save_figure(p11, "fig12_seed_stability", width = 10, height = 4)
  save_caption("fig12_seed_stability",
    paste0("Figure S6. Pairwise Adjusted Rand Index (ARI) across 5 random seeds ",
           "for each SOM grid size (8-feature set). ARI=1 indicates ",
           "perfect agreement between clusterings; ARI=0 indicates chance-level ",
           "agreement. Mean ARI values denote substantial seed-to-seed stability."))
})

# ---- fig13_cluster_sizes ----------------------------------------------------
try_fig("fig13_cluster_sizes", {
  log_progress("  fig13: Cluster sizes...")
  size_dt <- data.table(
    cluster = factor(cluster_labels, levels = cluster_labels),
    n       = obs_n,
    pct     = round(obs_pct, 1)
  )
  p12 <- ggplot(size_dt, aes(x = cluster, y = n, fill = cluster)) +
    geom_col(show.legend = FALSE) +
    geom_text(aes(label = sprintf("%s\n(%.1f%%)", format(n, big.mark = ","), pct)),
              vjust = -0.3, size = 3) +
    geom_hline(yintercept = min_threshold, linetype = "dashed", color = "red") +
    annotate("text", x = 0.7, y = min_threshold + 500, label = "1% threshold",
             color = "red", size = 3, hjust = 0) +
    scale_fill_manual(values = cluster_palette) +
    scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.15))) +
    labs(title = "Regime Distribution", x = NULL, y = "Number of Mesh Cells") +
    theme_pub +
    theme(axis.text.x = element_text(angle = 30, hjust = 1))
  save_figure(p12, "fig13_cluster_sizes", width = 8, height = 5)
  save_caption("fig13_cluster_sizes",
    paste0("Figure S7. Distribution of 138,768 mesh cells across ", K_FINAL,
           " contamination decay regimes. Bar heights show the number of cells; ",
           "labels show count and percentage. The red dashed line indicates the ",
           "1% minimum cluster size threshold (n=", format(min_threshold, big.mark = ","), ")."))
})

# ---- fig14_cluster_boxplots -------------------------------------------------
try_fig("fig14_cluster_boxplots", {
  log_progress("  fig14: Cluster boxplots...")
  set.seed(42L)
  n_subsample <- min(nrow(dt), 50000L)
  subsample_idx <- sample.int(nrow(dt), n_subsample)
  box_list <- list()
  for (j in seq_along(SOM_FEATURES)) {
    box_list[[j]] <- data.table(
      cluster = factor(cluster_labels[obs_clusters[subsample_idx]], levels = cluster_labels),
      feature = FEATURE_LABELS[SOM_FEATURES[j]],
      value   = som_matrix[subsample_idx, j]
    )
  }
  box_dt <- rbindlist(box_list)
  box_dt[, feature := factor(feature, levels = FEATURE_LABELS[SOM_FEATURES])]
  p13 <- ggplot(box_dt, aes(x = cluster, y = value, fill = cluster)) +
    geom_boxplot(outlier.size = 0.05, outlier.colour = "grey60", outlier.alpha = 0.3) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "grey50") +
    facet_wrap(~feature, scales = "free_y", ncol = 3) +
    scale_fill_manual(values = cluster_palette, name = "Regime") +
    labs(title = "Feature Distributions by Regime",
         x = NULL, y = "Z-score (IQR-scaled)") +
    theme_pub +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank(),
          legend.position = "bottom") +
    guides(fill = guide_legend(nrow = 2))
  save_figure(p13, "fig14_cluster_boxplots", width = 12, height = 10)
  save_caption("fig14_cluster_boxplots",
    paste0("Figure S8. Boxplot distributions of 8 temporal features within each ",
           "contamination decay regime (z-scored values). Based on a random subsample ",
           "of ", format(n_subsample, big.mark = ","), " observations for visual clarity. ",
           "The dashed line at y=0 indicates the population median."))
})

# =============================================================================
# --- Final Summary -----------------------------------------------------------
# =============================================================================
log_progress("Finalization complete. Summary:")

message("\n=== FINAL RESULTS (k=4 PRIMARY) ===")
message(sprintf("Model: %s (copied from sensitivity); k = %d", selected_grid, K_FINAL))
for (i in 1:K_FINAL) {
  message(sprintf("  %d. %-26s n=%s (%5.2f%%)",
                  i, cluster_labels[i], format(obs_n[i], big.mark = ","), obs_pct[i]))
}

if (length(failed_figs) == 0) {
  message("\nAll figures rendered successfully.")
} else {
  message(sprintf("\nFAILED FIGURES: %s", paste(failed_figs, collapse = ", ")))
}

# --- Session Info ------------------------------------------------------------
writeLines(capture.output(sessionInfo()), "output/som_session_info.txt")
message("Session info saved to output/som_session_info.txt")

message("FINALIZE_K4_COMPLETE")
