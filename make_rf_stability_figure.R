#!/usr/bin/env Rscript
# =============================================================================
# make_rf_stability_figure.R
# Stand-alone: builds the RF spatial-CV STABILITY + block-size ROBUSTNESS figure
# from the already-saved output/rf_cv_metrics.json (NO retraining).
# Shows: per-fold macro-F1 for 10 km (headline) and 25 km (conservative) blocks,
# fold mean +/- sd, majority baseline, and the inflated random-CV reference -
# the visual evidence for "no outlier fold + robust to block size".
# Output: figures/rf_cv_stability.{png,pdf}
# =============================================================================
suppressMessages({
  library(jsonlite)
  library(ggplot2)
  library(viridisLite)
})

p <- function(...) file.path(...)
d <- fromJSON(p("output", "rf_cv_metrics.json"))

mk <- function(block, label) {
  pf <- d[[block]]$per_fold
  data.frame(block = label, fold = as.integer(pf$fold),
             macro_f1 = as.numeric(pf$macro_f1))
}
df <- rbind(
  mk("spatial_block_cv_10km", "10 km\n(headline)"),
  mk("spatial_block_cv_25km", "25 km\n(conservative)")
)
df$xpos <- ifelse(grepl("10 km", df$block), 1, 2)
BLK_LABS <- c("10 km\n(headline)", "25 km\n(conservative)")

s10 <- d$spatial_block_cv_10km$per_fold_macro_f1_summary
s25 <- d$spatial_block_cv_25km$per_fold_macro_f1_summary
agg <- data.frame(
  block = c("10 km\n(headline)", "25 km\n(conservative)"),
  xpos  = c(1, 2),
  mean  = c(s10$mean, s25$mean),
  sd    = c(s10$sd,   s25$sd),
  aggF1 = c(d$spatial_block_cv_10km$macro_f1, d$spatial_block_cv_25km$macro_f1),
  lo    = c(s10$min,  s25$min),
  hi    = c(s10$max,  s25$max)
)
base <- d$majority_baseline_accuracy
rand <- d$random_5fold_cv_INFLATED$macro_f1

theme_pub <- theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"),
        legend.position = "none")

gg <- ggplot(df, aes(xpos, macro_f1, colour = block)) +
  # reference lines
  geom_hline(yintercept = rand, linetype = "dotted", colour = "grey60") +
  annotate("text", x = 0.5, y = rand + 0.006,
           label = sprintf("random 5-fold CV %.3f (autocorrelation-inflated)", rand),
           hjust = 0, size = 3, colour = "grey45") +
  geom_hline(yintercept = base, linetype = "dashed", colour = "grey55") +
  annotate("text", x = 0.5, y = base + 0.007,
           label = sprintf("majority-class baseline %.3f", base),
           hjust = 0, size = 3, colour = "grey40") +
  # fold mean +/- sd
  geom_errorbar(data = agg, inherit.aes = FALSE,
                aes(x = xpos, ymin = mean - sd, ymax = mean + sd),
                width = 0.12, colour = "grey30", linewidth = 0.5) +
  # individual folds
  geom_jitter(width = 0.07, height = 0, size = 2.8, alpha = 0.85) +
  # aggregate (pooled out-of-fold) macro-F1 as a black diamond
  geom_point(data = agg, inherit.aes = FALSE, aes(xpos, aggF1),
             shape = 18, size = 4.2, colour = "black") +
  # annotate each block with aggregate F1 + spread
  geom_text(data = agg, inherit.aes = FALSE,
            aes(xpos, hi + 0.012,
                label = sprintf("macro-F1 %.3f\nfolds %.3f-%.3f (sd %.3f)",
                                aggF1, lo, hi, sd)),
            size = 3.1, lineheight = 0.95, fontface = "bold") +
  scale_colour_viridis_d(begin = 0.35, end = 0.70) +
  scale_x_continuous(breaks = c(1, 2), labels = BLK_LABS, limits = c(0.45, 2.6)) +
  coord_cartesian(ylim = c(0.60, 0.95)) +
  labs(
    title = "RF spatial-CV stability and block-size robustness (3-class regimes)",
    subtitle = paste0(
      "Each point = one held-out spatial fold (k=5); black diamond = pooled out-of-fold macro-F1.\n",
      "No outlier fold (sd 0.013 / 0.020); the 25 km block barely lowers it - stable + robust to block size."),
    x = "Spatial-block size (geographically disjoint folds)",
    y = "Macro-F1 (held-out)"
  ) +
  theme_pub

ggsave(p("figures", "rf_cv_stability.png"), gg, width = 9.5, height = 5.5, dpi = 300)
ggsave(p("figures", "rf_cv_stability.pdf"), gg, width = 9.5, height = 5.5)
cat("Saved: figures/rf_cv_stability.{png,pdf}\n")
cat(sprintf("10 km: agg %.4f, folds [%.4f, %.4f] sd %.4f\n", agg$aggF1[1], agg$lo[1], agg$hi[1], agg$sd[1]))
cat(sprintf("25 km: agg %.4f, folds [%.4f, %.4f] sd %.4f\n", agg$aggF1[2], agg$lo[2], agg$hi[2], agg$sd[2]))
