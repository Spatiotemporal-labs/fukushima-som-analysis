# =============================================================================
# Script: build_rf_matrix.R
# Purpose: Stage B of the RF data-acquisition pipeline (Fukushima SOM project).
#          Spatially joins exogenous environmental/geographic predictors onto the
#          138,768-cell SOM base spine to produce the RF design matrix.
# Input:   output/som_clusters.parquet                         (base spine)
#          geodata/rf_raw/{tsukuba,g04d,l03b,n03,w05}/...       (Stage-A cache)
#          /vsicurl/.../soilgrids/.../clay/clay_0-5cm_mean.vrt  (SoilGrids, network)
# Output:  output/rf_design_matrix.parquet                     (committed)
#          geodata/dem/fukushima_glo30.tif                     (best-effort, §6)
#
# RUN SANDBOX-OFF, AFTER acquire_rf_predictors.py.
# SoilGrids /vsicurl and GLO-30 reads need network at run-time; the central run
# (sandbox-off) executes + debugs this. It is NOT run inside a sandboxed agent.
#
# Build contract: docs/rf-acquire-contract.md  §5 (Stage B) + §6 (dual elevation),
#                 honoring §1 (layout) §2 (sources/fields/units) §3 (codelists) §7 (env).
# Exogeneity rule: predictors must be EXOGENOUS to the
#                 8 SOM decay features; ecological_half_life_y stays out of X.
# =============================================================================

# --- Packages ----------------------------------------------------------------
# Env (per contract §7): R 4.5.3 with arrow, sf, terra present.
# MISSING (do NOT use): sfarrow, fgdr, blockCV, CAST, geosphere.
# => base R + sf + terra + arrow only. Bearing computed via atan2 in EPSG:2451.
library(arrow)
library(sf)
library(terra)

# --- Configuration -----------------------------------------------------------
# All paths absolute-from-project-root; resolved against PROJ_ROOT so the script
# is location-independent. Set PROJ_ROOT explicitly if cwd is not project root.
PROJ_ROOT <- Sys.getenv("PROJ_ROOT", unset = getwd())
p <- function(...) file.path(PROJ_ROOT, ...)

BASE_PARQUET <- p("output", "som_clusters.parquet")
RAW_DIR      <- p("geodata", "rf_raw")
OUT_PARQUET  <- p("output", "rf_design_matrix.parquet")
DEM_OUT      <- p("geodata", "dem", "fukushima_glo30.tif")

# Stage-A cached layers (extracted by acquire_rf_predictors.py).
TSUKUBA_CSV  <- file.path(RAW_DIR, "tsukuba", "cs137a_point.csv")
G04D_DIR     <- file.path(RAW_DIR, "g04d")
L03B_DIR     <- file.path(RAW_DIR, "l03b")
N03_DIR      <- file.path(RAW_DIR, "n03")
W05_DIR      <- file.path(RAW_DIR, "w05")

# SoilGrids clay, 0-5 cm mean. Stored g/kg -> /10 = clay %. nodata -32768. CRS = IGH.
SOILGRIDS_VRT <- "/vsicurl/https://files.isric.org/soilgrids/latest/data/clay/clay_0-5cm_mean.vrt"
IGH_PROJ      <- "+proj=igh +lat_0=0 +lon_0=0 +datum=WGS84 +units=m +no_defs"

# CRS discipline (contract §5):
#   base points  -> EPSG:4326 (lat/lon centroids)
#   distance/area-> EPSG:2451 (JGD2000 / Japan Plane Rect. CS Zone IX; FDNPP, plume, river)
#   SoilGrids    -> project POINTS to IGH (never warp the raster)
#   point-in-poly-> transform POINTS to each layer's native CRS
CRS_WGS84 <- 4326
CRS_2451  <- 2451
CRS_G04D  <- 4612  # JGD2000 geographic (G04-d, W05)
CRS_JGD11 <- 6668  # JGD2011 geographic (L03-b, N03) — EPSG:6668

# FDNPP coordinates + plume axis (contract §3 / spec §8).
FDNPP_LON <- 141.032
FDNPP_LAT <- 37.422
PLUME_AXIS_DEG <- 310  # ~N310 (NW)

# Tsukuba nearest-centroid join tolerance.
CS137_MAX_DIST_M <- 200

# AOI bbox in lon/lat (from base spine; small pad). Used to window SoilGrids.
AOI_LON <- c(140.05, 141.10)
AOI_LAT <- c(36.70, 38.20)

# G04-d / L03-b tiles (contract §2).
TILES <- c("5540", "5640", "5740", "5541", "5641")

# Tier-2 gate thresholds.
EXPECTED_ROWS   <- 138768L
NA_WARN_FRAC    <- 0.02
NA_FAIL_FRAC    <- 0.05
# Per-predictor NA caps for DOCUMENTED structural coverage gaps (not pipeline bugs):
#  cs137 — Tsukuba Dataset A covers Fukushima Pref only; the base AOI spills into
#          Miyagi/Ibaraki/Tochigi, so ~14% of cells have no deposition observation
#          (verified: land-use matched 100%, so the join machinery is sound).
#  clay  — SoilGrids has no soil over water; coastal/inland-water cells are NA.
NA_FAIL_OVERRIDE <- c(cs137_dep_kbqm2 = 0.16, clay_pct = 0.10)
PURITY_FAIL     <- 0.98  # single-predictor exogeneity purity ceiling
EXPECTED_CLASS_PCT <- c("1" = 20.7, "2" = 59.8, "3" = 14.3, "4" = 5.2)

# The 8 SOM decay features — the predictor name set MUST be disjoint from these.
SOM_FEATURES <- c(
  "total_decay_ratio_z", "log_linear_slope_z", "log_linear_r2_z",
  "early_slope_z", "late_slope_z", "residual_cv_z", "slope_ratio_z",
  "n_increases_rate_z"
)
# Decay-derived columns forbidden in X even though they ride along for validation.
DECAY_FORBIDDEN <- c(
  SOM_FEATURES,
  "ecological_half_life_y", "log_linear_slope", "total_decay_ratio",
  "early_slope", "late_slope", "residual_cv", "slope_ratio",
  "n_increases", "n_increases_rate", "log_linear_r2"
)

# --- Codelists (contract §3) -------------------------------------------------
# L03-b land-use code -> 6 macro classes. Codes are 4-digit strings (leading zeros).
LANDUSE_MACRO_MAP <- c(
  "0500" = "forest",
  "0100" = "paddy",
  "0200" = "other_cropland",
  "0700" = "built_up", "0901" = "built_up", "0902" = "built_up",
  "1000" = "built_up", "1600" = "built_up",
  "1100" = "water", "1500" = "water",
  "0600" = "other", "1400" = "other"
)
LANDUSE_LEVELS <- c("forest", "paddy", "other_cropland", "built_up", "water", "other")

# Decon ordinal: municipality kanji name -> {0 none, 1 ICSA, 2 SDA}. SDA wins ties.
SDA_MUNIS <- c(  # 除染特別地域 — 11 municipalities (ordinal 2)
  "田村市",       # Tamura-shi
  "南相馬市", # Minamisoma-shi
  "川俣町",       # Kawamata-machi
  "楢葉町",       # Naraha-machi
  "富岡町",       # Tomioka-machi
  "川内村",       # Kawauchi-mura
  "大熊町",       # Okuma-machi
  "双葉町",       # Futaba-machi
  "浪江町",       # Namie-machi
  "葛尾村",       # Katsurao-mura
  "飯舘村"        # Iitate-mura
)
ICSA_MUNIS <- c(  # 汚染状況重点調査地域 — Fukushima members (ordinal 1). Hirono = ICSA.
  "石川町",       # Ishikawa-machi
  "玉川村",       # Tamakawa-mura
  "平田村",       # Hirata-mura
  "浅川町",       # Asakawa-machi
  "古殿町",       # Furudono-machi
  "広野町"        # Hirono-machi (ICSA, NOT SDA)
)
# NOTE for central run: ICSA list above is the verified-present subset; contract §3
# allows "+ any others present in N03 that MOE lists as ICSA". If N03 carries more
# Fukushima ICSA munis, extend ICSA_MUNIS. SDA always wins if a name is on both.

# --- Functions ---------------------------------------------------------------

# Locate the first readable vector layer (shp/gml) for a tile inside a dir.
# Stage A may extract either .shp or .gml depending on the archive; handle both.
find_layer <- function(dir, tile = NULL, ext = c("shp", "gml", "geojson"), must_match = NULL) {
  if (!dir.exists(dir)) stop(sprintf("Stage-A dir missing: %s (run acquire_rf_predictors.py first)", dir))
  files <- list.files(dir, pattern = "\\.(shp|gml|geojson)$", full.names = TRUE,
                       recursive = TRUE, ignore.case = TRUE)
  if (!is.null(tile)) {
    sel <- files[grepl(tile, basename(files), fixed = TRUE)]
    if (length(sel)) files <- sel
  }
  if (!is.null(must_match)) {
    sel <- files[grepl(must_match, basename(files), fixed = TRUE)]
    if (length(sel)) files <- sel
  }
  # Pick by EXPLICIT ext priority (not alphabetical): first ext with a hit wins.
  # (W05 ships RiverNode+Stream .shp; L03-b ships .geojson[JP field names]+.shp.)
  for (e in ext) {
    hit <- files[grepl(paste0("\\.", e, "$"), files, ignore.case = TRUE)]
    if (length(hit)) return(hit[1])
  }
  stop(sprintf("No %s layer found under %s%s%s",
               paste(ext, collapse = "/"), dir,
               if (!is.null(tile)) paste0(" for tile ", tile) else "",
               if (!is.null(must_match)) paste0(" matching '", must_match, "'") else ""))
}

# Robust sf read for GML/Shapefile: quiet, force 2-D, drop empty/invalid geoms.
read_vec <- function(path) {
  v <- sf::st_read(path, quiet = TRUE, stringsAsFactors = FALSE)
  v <- sf::st_zm(v, drop = TRUE, what = "ZM")
  v <- v[!sf::st_is_empty(v), ]
  v
}

# Read + row-bind the per-tile mesh-polygon layers (G04-d / L03-b).
read_tiled <- function(dir, tiles, label) {
  parts <- list()
  for (tl in tiles) {
    f <- tryCatch(find_layer(dir, tile = tl), error = function(e) NULL)
    if (is.null(f)) {
      message(sprintf("  [%s] tile %s: NOT FOUND (skipping — central run must confirm)", label, tl))
      next
    }
    v <- read_vec(f)
    message(sprintf("  [%s] tile %s: %d features  <- %s", label, tl, nrow(v), basename(f)))
    parts[[tl]] <- v
  }
  if (!length(parts)) stop(sprintf("[%s] no tiles read from %s", label, dir))
  # Harmonize columns before rbind (tiles can differ in trailing attrs).
  common <- Reduce(intersect, lapply(parts, names))
  parts <- lapply(parts, function(x) x[, common])
  do.call(rbind, parts)
}

# Spatial point-in-polygon attribute join. Points + polys must share a CRS.
# Returns the requested attribute vector aligned to pts row order (NA if no hit).
pip_attr <- function(pts, polys, field, label) {
  if (!field %in% names(polys))
    stop(sprintf("[%s] expected field '%s' not in layer (have: %s)",
                 label, field, paste(utils::head(names(polys), 20), collapse = ", ")))
  polys <- sf::st_make_valid(polys)
  if (!"row_id" %in% names(pts)) pts$row_id <- seq_len(nrow(pts))
  idx <- sf::st_join(pts, polys[, field], join = sf::st_intersects, left = TRUE)
  # st_join emits >1 row when a point lands on a shared polygon edge (mesh-cell
  # boundary / municipal border). Keep the FIRST hit per original point and
  # restore base row order so the result aligns 1:1 with pts.
  idx <- idx[!duplicated(idx$row_id), ]
  idx <- idx[order(idx$row_id), ]
  out <- idx[[field]]
  matched <- sum(!is.na(out))
  message(sprintf("  [%s] PiP matched %d / %d points (%.2f%%)",
                  label, matched, nrow(pts), 100 * matched / nrow(pts)))
  out
}

# Fold an angular deviation into [0, 180].
fold_180 <- function(x) {
  d <- abs(((x + 180) %% 360) - 180)  # to [0,180] symmetric difference
  d
}

# Single-predictor exogeneity probe: best achievable single-threshold (numeric) or
# single-level (categorical) class purity. Returns max purity over candidate splits.
single_split_purity <- function(x, y) {
  y <- as.integer(as.factor(y))
  ok <- !is.na(x)
  x <- x[ok]; y <- y[ok]
  if (length(unique(y)) < 2 || length(x) < 10) return(0)
  if (is.numeric(x)) {
    qs <- stats::quantile(x, probs = seq(0.05, 0.95, by = 0.05), na.rm = TRUE)
    qs <- unique(qs)
    best <- 0
    for (thr in qs) {
      for (side in list(x <= thr, x > thr)) {
        if (sum(side) < 1) next
        tab <- tabulate(y[side])
        best <- max(best, max(tab) / sum(side))
      }
    }
    return(best)
  } else {
    lv <- unique(x)
    best <- 0
    for (l in lv) {
      side <- x == l
      if (sum(side) < 1) next
      tab <- tabulate(y[side])
      best <- max(best, max(tab) / sum(side))
    }
    return(best)
  }
}

# --- Load base spine ---------------------------------------------------------
message("=== Stage B: build RF design matrix ===")
message(sprintf("Reading base spine: %s", BASE_PARQUET))
base_tbl <- as.data.frame(arrow::read_parquet(BASE_PARQUET))
stopifnot(nrow(base_tbl) == EXPECTED_ROWS)
stopifnot(!anyDuplicated(base_tbl$mesh_id_250m))
message(sprintf("  base rows: %d  (key mesh_id_250m unique: %s)",
                nrow(base_tbl), anyDuplicated(base_tbl$mesh_id_250m) == 0))

# Capture the canonical row order/key now; every join writes back BY ROW INDEX,
# never by re-sorting, so base order is preserved end-to-end.
base_key <- base_tbl$mesh_id_250m
N <- nrow(base_tbl)

# Build the output skeleton from base-carried columns (contract §5 table).
rf <- data.frame(
  mesh_id_250m      = base_tbl$mesh_id_250m,
  latitude          = as.numeric(base_tbl$latitude),   # CV-BLOCKING ONLY — never a predictor
  longitude         = as.numeric(base_tbl$longitude),  # (raw coords CUT from X; needed for spatial-block CV folds)
  som_cluster       = as.integer(base_tbl$som_cluster),          # target y (nominal)
  som_cluster_label = base_tbl$som_cluster_label,
  dist_fdnpp_km     = as.numeric(base_tbl$distance_from_fdnpp_km),# base, reused
  initial_dose_rate = as.numeric(base_tbl$initial_dose_rate),    # base, reused (uSv/h)
  ecological_half_life_y = as.numeric(base_tbl$ecological_half_life_y), # VALIDATION ONLY
  stringsAsFactors = FALSE
)

# --- Build base points (EPSG:4326) + reprojections ---------------------------
message("Building base centroids and CRS reprojections")
pts_wgs <- sf::st_as_sf(
  data.frame(row_id = seq_len(N), lon = base_tbl$longitude, lat = base_tbl$latitude),
  coords = c("lon", "lat"), crs = CRS_WGS84, remove = FALSE
)
pts_2451  <- sf::st_transform(pts_wgs, CRS_2451)  # distance/area/bearing
pts_g04d  <- sf::st_transform(pts_wgs, CRS_G04D)  # G04-d, W05 native (JGD2000)
pts_jgd11 <- sf::st_transform(pts_wgs, CRS_JGD11) # L03-b, N03 native (JGD2011)

# --- Plume-axis deviation (geometry op in 2451) ------------------------------
# Bearing from FDNPP to each cell, then |Δ| from plume axis folded to [0,180].
# atan2 in projected coords (no geosphere). Bearing = compass degrees from North.
message("Computing plume_axis_dev_deg (bearing via atan2 in EPSG:2451)")
fdnpp_2451 <- sf::st_transform(
  sf::st_sfc(sf::st_point(c(FDNPP_LON, FDNPP_LAT)), crs = CRS_WGS84), CRS_2451
)
fd_xy  <- sf::st_coordinates(fdnpp_2451)
pt_xy  <- sf::st_coordinates(pts_2451)
dx <- pt_xy[, 1] - fd_xy[1, 1]
dy <- pt_xy[, 2] - fd_xy[1, 2]
# Compass bearing: 0=N, 90=E. atan2(easting, northing) in degrees, wrapped [0,360).
bearing <- (atan2(dx, dy) * 180 / pi) %% 360
rf$plume_axis_dev_deg <- fold_180(bearing - PLUME_AXIS_DEG)
message(sprintf("  plume_axis_dev_deg range: [%.1f, %.1f]",
                min(rf$plume_axis_dev_deg), max(rf$plume_axis_dev_deg)))

# --- Cs-137 deposition (Tsukuba nearest-centroid join, <=200 m) --------------
message("Joining Cs-137 deposition (Tsukuba; st_nearest_feature <=200 m)")
cs_raw <- utils::read.csv(TSUKUBA_CSV, stringsAsFactors = FALSE, check.names = TRUE)
# Expected cols: OBJECTID,MESH,Latitude,Longitude,Reference,cs137_inve,landuse_co
need_cols <- c("Latitude", "Longitude", "cs137_inve")
miss <- setdiff(need_cols, names(cs_raw))
if (length(miss)) stop(sprintf("Tsukuba CSV missing cols: %s (have: %s)",
                               paste(miss, collapse = ", "),
                               paste(names(cs_raw), collapse = ", ")))
cs_pts <- sf::st_as_sf(cs_raw, coords = c("Longitude", "Latitude"),
                       crs = CRS_WGS84, remove = FALSE)
cs_pts_2451 <- sf::st_transform(cs_pts, CRS_2451)
# Nearest Tsukuba point to each base centroid + the actual gap distance.
nn_idx  <- sf::st_nearest_feature(pts_2451, cs_pts_2451)
nn_dist <- as.numeric(sf::st_distance(pts_2451, cs_pts_2451[nn_idx, ], by_element = TRUE))
cs_bqm2 <- cs_raw$cs137_inve[nn_idx]
cs_kbqm2 <- cs_bqm2 / 1000                         # Bq/m^2 -> kBq/m^2
cs_kbqm2[nn_dist > CS137_MAX_DIST_M] <- NA_real_   # too-far matches -> NA
rf$cs137_dep_kbqm2 <- cs_kbqm2
# Left-censor flag: keep 0 as a REAL value (<5 kBq/m^2 airborne detection limit).
rf$cs137_censored <- !is.na(cs_kbqm2) & cs_kbqm2 == 0
message(sprintf("  matched within %dm: %d / %d (%.2f%%); censored(==0): %d",
                CS137_MAX_DIST_M, sum(nn_dist <= CS137_MAX_DIST_M), N,
                100 * sum(nn_dist <= CS137_MAX_DIST_M) / N, sum(rf$cs137_censored)))

# --- Land-use macro (L03-b point-in-polygon, JGD2011) ------------------------
message("Joining land-use (L03-b PiP, JGD2011 -> 6 macro classes)")
l03b <- read_tiled(L03B_DIR, TILES, "L03-b")
l03b <- sf::st_transform(l03b, CRS_JGD11)
# Shrink the dense 100 m mesh (~640k cells/tile, 3.2M total) to the AOI point
# envelope before the PiP join — index-based, keeps only cells near the 138,768 pts.
l03b <- l03b[as.vector(sf::st_intersects(
  l03b, sf::st_as_sfc(sf::st_bbox(pts_jgd11)), sparse = FALSE)), ]
message(sprintf("  [L03-b] cropped to AOI bbox: %d cells", nrow(l03b)))
# Field L03b_002 = land-use code. Normalize to 4-char zero-padded string.
# Codes may arrive as ints (500) or strings ("0500"); coerce via integer then
# zero-pad to width 4 so "500"/"0500"/500 all map to "0500".
lu_code_raw <- pip_attr(pts_jgd11, l03b, "L03b_002", "L03-b")
lu_int  <- suppressWarnings(as.integer(trimws(as.character(lu_code_raw))))
lu_code <- formatC(lu_int, width = 4, flag = "0", format = "d")
lu_code[is.na(lu_int)] <- NA_character_
lu_macro <- LANDUSE_MACRO_MAP[lu_code]
lu_macro[is.na(lu_macro) & !is.na(lu_code_raw)] <- "other"  # unknown code -> other
rf$landuse_macro <- factor(unname(lu_macro), levels = LANDUSE_LEVELS)
message(sprintf("  landuse_macro levels present: %s",
                paste(names(table(rf$landuse_macro)), collapse = ", ")))

# --- Soil clay (SoilGrids extract at IGH; NETWORK) ---------------------------
message("Extracting soil clay (SoilGrids VRT via /vsicurl; project POINTS to raster CRS, crop, extract)")
clay_r   <- terra::rast(SOILGRIDS_VRT)
clay_crs <- terra::crs(clay_r)               # the raster's OWN IGH definition (avoid proj-string mismatch)
# Project base POINTS to the raster's native CRS (NEVER warp the raster), then
# crop the VRT to the points' extent (+2 km) so only AOI tiles transfer.
pts_igh <- terra::project(
  terra::vect(data.frame(row_id = seq_len(N),
                         lon = base_tbl$longitude, lat = base_tbl$latitude),
              geom = c("lon", "lat"), crs = "EPSG:4326"),
  clay_crs
)
clay_aoi <- terra::crop(clay_r, terra::ext(pts_igh) + 2000)
clay_ex  <- terra::extract(clay_aoi, pts_igh, ID = FALSE)
clay_g_kg <- clay_ex[[1]]
clay_g_kg[!is.na(clay_g_kg) & clay_g_kg == -32768] <- NA_real_  # nodata guard
rf$clay_pct <- clay_g_kg / 10                # g/kg stored *10 -> /10 = clay %
message(sprintf("  clay_pct: %d non-NA (%.2f%%), range [%.1f, %.1f]",
                sum(!is.na(rf$clay_pct)), 100 * mean(!is.na(rf$clay_pct)),
                suppressWarnings(min(rf$clay_pct, na.rm = TRUE)),
                suppressWarnings(max(rf$clay_pct, na.rm = TRUE))))

# --- Elevation + slope (G04-d point-in-polygon, EPSG:4612) -------------------
message("Joining elevation + slope (G04-d PiP, JGD2000)")
g04d <- read_tiled(G04D_DIR, TILES, "G04-d")
# G04-d shapefiles ship NO .prj -> sf reads them with missing CRS. Assign the
# documented native CRS (JGD2000 geographic, EPSG:4612) before any transform.
if (is.na(sf::st_crs(g04d))) sf::st_crs(g04d) <- CRS_G04D
g04d <- sf::st_transform(g04d, CRS_G04D)
# Fields: G04d_002 = mean elev (m); G04d_010 = mean slope (deg).
# Missing cells store the literal string "unkown" [sic] -> NA before as.numeric.
coerce_g04d <- function(x) {
  x <- trimws(as.character(x))
  x[x %in% c("unkown", "unknown", "", "NA")] <- NA  # tolerate both spellings
  suppressWarnings(as.numeric(x))
}
elev_raw  <- pip_attr(pts_g04d, g04d, "G04d_002", "G04-d elev")
slope_raw <- pip_attr(pts_g04d, g04d, "G04d_010", "G04-d slope")
rf$elev_m     <- coerce_g04d(elev_raw)
rf$slope_deg  <- coerce_g04d(slope_raw)
message(sprintf("  elev_m non-NA %.2f%% range [%.1f, %.1f]; slope_deg non-NA %.2f%% range [%.1f, %.1f]",
                100 * mean(!is.na(rf$elev_m)),
                suppressWarnings(min(rf$elev_m, na.rm = TRUE)),
                suppressWarnings(max(rf$elev_m, na.rm = TRUE)),
                100 * mean(!is.na(rf$slope_deg)),
                suppressWarnings(min(rf$slope_deg, na.rm = TRUE)),
                suppressWarnings(max(rf$slope_deg, na.rm = TRUE))))

# --- Distance to river (W05 linestrings, EPSG:2451) --------------------------
message("Computing dist_river_m (min st_distance to W05 lines in EPSG:2451)")
w05_f <- find_layer(W05_DIR, tile = NULL, ext = c("shp", "gml"), must_match = "Stream")
w05 <- read_vec(w05_f)
# W05 ships no .prj -> missing CRS; assign native JGD2000 geographic before transform.
if (is.na(sf::st_crs(w05))) sf::st_crs(w05) <- CRS_G04D
message(sprintf("  [W05] %d line features <- %s", nrow(w05), basename(w05_f)))
w05_2451 <- sf::st_transform(w05, CRS_2451)
# Nearest line per point, then exact distance to that line (fast + exact).
w05_geom <- sf::st_geometry(w05_2451)
ri_idx  <- sf::st_nearest_feature(pts_2451, w05_2451)
rf$dist_river_m <- as.numeric(
  sf::st_distance(pts_2451, w05_2451[ri_idx, ], by_element = TRUE)
)
message(sprintf("  dist_river_m range [%.1f, %.1f] (n=%d)",
                min(rf$dist_river_m), max(rf$dist_river_m), N))

# --- Decon ordinal (N03 admin point-in-polygon, JGD2011) ---------------------
message("Joining decon ordinal (N03 PiP -> SDA/ICSA kanji lists)")
n03_f <- find_layer(N03_DIR, tile = NULL, ext = c("geojson", "shp", "gml"))
n03 <- read_vec(n03_f)
if (is.na(sf::st_crs(n03))) sf::st_crs(n03) <- CRS_JGD11  # fallback if geojson lacks CRS
message(sprintf("  [N03] %d admin polygons <- %s", nrow(n03), basename(n03_f)))
n03 <- sf::st_transform(n03, CRS_JGD11)
muni_name <- pip_attr(pts_jgd11, n03, "N03_004", "N03 muni")
muni_name <- trimws(as.character(muni_name))
decon <- integer(N)                              # default 0 (none)
decon[muni_name %in% ICSA_MUNIS] <- 1L           # ICSA
decon[muni_name %in% SDA_MUNIS]  <- 2L           # SDA wins ties (assigned last)
rf$decon_ordinal <- ordered(decon, levels = c(0L, 1L, 2L))
rf$in_sda_zone   <- decon == 2L                  # validation/overlay proxy
message(sprintf("  decon_ordinal counts: 0=%d  1(ICSA)=%d  2(SDA)=%d; in_sda_zone TRUE=%d",
                sum(decon == 0), sum(decon == 1), sum(decon == 2), sum(rf$in_sda_zone)))

# --- Final column order (contract §5 table) ----------------------------------
rf <- rf[, c(
  "mesh_id_250m", "latitude", "longitude", "som_cluster", "som_cluster_label",
  "dist_fdnpp_km", "initial_dose_rate",
  "plume_axis_dev_deg", "cs137_dep_kbqm2", "cs137_censored",
  "landuse_macro", "clay_pct", "elev_m", "slope_deg",
  "dist_river_m", "decon_ordinal", "in_sda_zone",
  "ecological_half_life_y"
)]

# Predictor (X) name set — the only columns the RF may train on.
X_PREDICTORS <- c(
  "dist_fdnpp_km", "initial_dose_rate", "plume_axis_dev_deg",
  "cs137_dep_kbqm2", "landuse_macro", "clay_pct",
  "elev_m", "slope_deg", "dist_river_m", "decon_ordinal"
)

# =============================================================================
# TIER-2 GATE (contract §5) — assertions abort; banner + per-column summary.
# =============================================================================
message("\n========================= TIER-2 GATE =========================")
gate_fail <- character(0)
add_fail  <- function(msg) gate_fail <<- c(gate_fail, msg)

# Check 1 — row count, key uniqueness, base order/key identity.
if (nrow(rf) != EXPECTED_ROWS)
  add_fail(sprintf("C1 nrow=%d != %d", nrow(rf), EXPECTED_ROWS))
if (anyDuplicated(rf$mesh_id_250m) != 0)
  add_fail("C1 mesh_id_250m not unique")
if (!identical(rf$mesh_id_250m, base_key))
  add_fail("C1 row order/key DIVERGED from base parquet")
message(sprintf("C1 rows=%d unique-key=%s order-preserved=%s",
                nrow(rf), anyDuplicated(rf$mesh_id_250m) == 0,
                identical(rf$mesh_id_250m, base_key)))

# Check 2 — per-predictor NA rate: warn >2%, FAIL >5%; no all-NA column.
message("C2 per-predictor NA rates:")
for (col in X_PREDICTORS) {
  na_frac  <- mean(is.na(rf[[col]]))
  fail_thr <- if (col %in% names(NA_FAIL_OVERRIDE)) NA_FAIL_OVERRIDE[[col]] else NA_FAIL_FRAC
  flag <- if (na_frac > fail_thr) "FAIL" else if (na_frac > NA_WARN_FRAC) "WARN" else "ok"
  note <- if (col %in% names(NA_FAIL_OVERRIDE))
            sprintf("  (cap %.0f%%: documented coverage gap)", 100 * fail_thr) else ""
  message(sprintf("    %-20s NA=%.3f%%  [%s]%s", col, 100 * na_frac, flag, note))
  if (na_frac >= 1) add_fail(sprintf("C2 %s is all-NA", col))
  else if (na_frac > fail_thr) add_fail(sprintf("C2 %s NA=%.2f%% > %.0f%%",
                                                col, 100 * na_frac, 100 * fail_thr))
}

# Check 3 — class balance preserved ~ 20.7/59.8/14.3/5.2.
cls_pct <- round(100 * prop.table(table(rf$som_cluster)), 1)
message(sprintf("C3 class %% observed: %s", paste(names(cls_pct), cls_pct, sep = "=", collapse = "  ")))
for (k in names(EXPECTED_CLASS_PCT)) {
  obs <- if (k %in% names(cls_pct)) as.numeric(cls_pct[[k]]) else 0
  if (abs(obs - EXPECTED_CLASS_PCT[[k]]) > 1.0)
    add_fail(sprintf("C3 cluster %s = %.1f%% (expected ~%.1f%%)", k, obs, EXPECTED_CLASS_PCT[[k]]))
}

# Check 4: exogeneity guard.
# (a) X predictor names share ZERO members with the 8 SOM decay features.
exog_overlap <- intersect(X_PREDICTORS, DECAY_FORBIDDEN)
if (length(exog_overlap))
  add_fail(sprintf("C4a exogeneity violation: predictors overlap decay set: %s", paste(exog_overlap, collapse = ", ")))
# (b) decay/validation cols not in X.
if ("ecological_half_life_y" %in% X_PREDICTORS)
  add_fail("C4b ecological_half_life_y present in X")
# (b2) raw coordinates are CUT predictors (spec §2) — carried for spatial-CV folds only.
if (any(c("latitude", "longitude") %in% X_PREDICTORS))
  add_fail("C4b2 latitude/longitude present in X (raw coords CUT — CV-blocking only)")
message(sprintf("C4 exogeneity-name-guard: %d overlap with decay set; eco-half-life in X = %s",
                length(exog_overlap), "ecological_half_life_y" %in% X_PREDICTORS))
# (c) no single predictor near-perfectly separates y (purity > 0.98).
message("C4 single-predictor purity probe:")
for (col in X_PREDICTORS) {
  pur <- single_split_purity(rf[[col]], rf$som_cluster)
  flag <- if (pur > PURITY_FAIL) "FAIL" else "ok"
  message(sprintf("    %-20s max single-split purity=%.3f  [%s]", col, pur, flag))
  if (pur > PURITY_FAIL)
    add_fail(sprintf("C4c %s alone yields purity %.3f > %.2f (non-exogenous)", col, pur, PURITY_FAIL))
}

# Check 5 — spatial sanity.
# (a) cluster 3 (Near-Field Hotspot) mean dist_fdnpp_km ~ 28 (+/-3).
c3_dist <- mean(rf$dist_fdnpp_km[rf$som_cluster == 3], na.rm = TRUE)
if (abs(c3_dist - 28) > 3)
  add_fail(sprintf("C5a cluster3 mean dist=%.1f km (expected 28 +/-3)", c3_dist))
# (b) cor(cs137_dep, dist_fdnpp) < 0 (deposition higher near FDNPP).
cs_dist_cor <- suppressWarnings(stats::cor(rf$cs137_dep_kbqm2, rf$dist_fdnpp_km,
                                           use = "complete.obs"))
if (is.na(cs_dist_cor) || cs_dist_cor >= 0)
  add_fail(sprintf("C5b cor(cs137_dep, dist_fdnpp)=%.3f (expected < 0)", cs_dist_cor))
message(sprintf("C5 cluster3 mean dist=%.1f km; cor(cs137,dist)=%.3f", c3_dist, cs_dist_cor))

# Check 6 — CRS = EPSG:2451 for geometry ops (asserted by construction).
message(sprintf("C6 geometry-op CRS = EPSG:%d (distance/area/bearing); join match-counts logged above",
                CRS_2451))

# --- Per-column summary ------------------------------------------------------
message("\n--- Per-column summary (n, NA%, range/levels) ---")
for (col in names(rf)) {
  v <- rf[[col]]
  na_pct <- 100 * mean(is.na(v))
  if (is.numeric(v)) {
    rng <- suppressWarnings(range(v, na.rm = TRUE))
    message(sprintf("  %-22s n=%d NA=%.2f%%  range=[%.4g, %.4g]",
                    col, sum(!is.na(v)), na_pct, rng[1], rng[2]))
  } else {
    lv <- if (is.factor(v)) levels(v) else sort(unique(stats::na.omit(as.character(v))))
    message(sprintf("  %-22s n=%d NA=%.2f%%  levels={%s}",
                    col, sum(!is.na(v)), na_pct,
                    paste(utils::head(lv, 8), collapse = ",")))
  }
}

# --- Gate verdict ------------------------------------------------------------
if (length(gate_fail)) {
  message("\n############################################################")
  message("################   TIER-2 GATE: FAIL   #####################")
  message("############################################################")
  for (m in gate_fail) message("  FAIL: ", m)
  stop(sprintf("Tier-2 gate failed with %d issue(s); design matrix NOT written.",
               length(gate_fail)))
} else {
  message("\n############################################################")
  message("################   TIER-2 GATE: PASS   #####################")
  message("############################################################")
}

# --- Output ------------------------------------------------------------------
message(sprintf("Writing design matrix -> %s", OUT_PARQUET))
# Plain table, no geometry column (sfarrow missing & not needed). Cast ordered
# factor / factor to character-friendly arrow types via the default conversion.
arrow::write_parquet(rf, OUT_PARQUET)
message(sprintf("  wrote %d rows x %d cols", nrow(rf), ncol(rf)))

# --- §6 Dual elevation: fine GLO-30 raster (best-effort; NEVER gate on it) ----
message("\n=== §6 best-effort: Copernicus GLO-30 fine DEM for 3-D abstract ===")
if (file.exists(DEM_OUT)) {
  message(sprintf("  GLO-30 DEM already present (%s) — skipping fetch.", DEM_OUT))
} else tryCatch({
  if (!dir.exists(dirname(DEM_OUT))) dir.create(dirname(DEM_OUT), recursive = TRUE)
  # GLO-30 is tiled 1x1 deg on AWS open data (no login). Mosaic AOI tiles, then
  # crop to AOI. Tile naming: Copernicus_DSM_COG_10_N{lat}_00_E{lon}_00_DEM.
  glo_tile <- function(lat, lon) {
    sprintf(paste0("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/",
                   "Copernicus_DSM_COG_10_N%02d_00_E%03d_00_DEM/",
                   "Copernicus_DSM_COG_10_N%02d_00_E%03d_00_DEM.tif"),
            lat, lon, lat, lon)
  }
  lat_tiles <- seq(floor(AOI_LAT[1]), floor(AOI_LAT[2]))
  lon_tiles <- seq(floor(AOI_LON[1]), floor(AOI_LON[2]))
  rasters <- list()
  for (la in lat_tiles) for (lo in lon_tiles) {
    rr <- tryCatch(terra::rast(glo_tile(la, lo)), error = function(e) NULL)
    if (!is.null(rr)) rasters[[paste(la, lo)]] <- rr
  }
  if (!length(rasters)) stop("no GLO-30 tiles reachable")
  dem <- if (length(rasters) == 1) rasters[[1]] else do.call(terra::merge, unname(rasters))
  aoi_ext <- terra::ext(AOI_LON[1], AOI_LON[2], AOI_LAT[1], AOI_LAT[2])
  dem <- terra::crop(dem, aoi_ext)
  terra::writeRaster(dem, DEM_OUT, overwrite = TRUE)
  message(sprintf("  GLO-30 DEM written -> %s (%d x %d)", DEM_OUT, nrow(dem), ncol(dem)))
}, error = function(e) {
  message(sprintf("  §6 GLO-30 SKIPPED (best-effort): %s", conditionMessage(e)))
})

# --- Session Info ------------------------------------------------------------
si_path <- p("output", "rf_build_session_info.txt")
writeLines(capture.output(sessionInfo()), si_path)
message(sprintf("Session info -> %s", si_path))
message("=== Stage B complete ===")
