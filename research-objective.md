<!-- Phase relevance guide:
     Phase 4 (SOM): "Stage 1 — Unsupervised Discovery" approach section
     Phase 5 (RF): "Datasets" section (12 predictor profiles), "Stage 2" approach
     Phase 6 (SHAP): "Expected Results" for interpretation framing
     Manuscript: "Literature Landscape" for drafting guidance per paper -->

# Research Objective

## Title
Discovery of Behavioral Contamination Regimes in the Fukushima Exclusion Zone Using a Self-Organizing Map and Random Forest Pipeline

## Goal
Develop a two-stage unsupervised-to-supervised machine learning pipeline (SOM → RF) that first discovers distinct dose-rate decay regimes across the Fukushima exclusion zone without imposing prior categories, then identifies which environmental factors predict regime membership — shifting the analytical framing from continuous regression to discrete behavioral classification.

---

## Literature Landscape

This section outlines the key research papers that inform the pipeline design, framing, and manuscript positioning. Each paper shapes a specific decision in this research — they are not passive citations but active reference points that must be kept in mind during drafting.

### Papers That Define the Competitive Landscape (Fukushima ML)

#### Sun et al. (2022) — *The primary comparator*

**Reference:** Sun, C. et al. (2022). Prediction of ambient dose equivalent rates around Fukushima by integrating Random Forest with Kalman filter. *Journal of Environmental Radioactivity*, 252, 106949.
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0265931X22001370)

**What they did:** Built a Random Forest to predict ambient dose equivalent rates across the exclusion zone, integrating an Ensemble Kalman Filter (EnKF) for temporal smoothing. Used JAEA KURAMA car-survey data with only 4 predictors: initial dose rate, land-use type, soil type, and spatial coordinates — all from EMDB. Achieved R² ~ 0.9.

**Why it matters when drafting this research:**
- Every reviewer will compare this project directly against Sun et al. The manuscript must make the epistemological difference explicit: Sun et al. treated decay as a continuous regression problem (predicting dose-rate values); this project treats decay heterogeneity as a classification problem (discovering discrete behavioral regimes, then explaining membership).
- Their 4-predictor set from EMDB is the baseline this project builds on. The expansion to 8-12 predictors must be justified as providing richer environmental context for regime explanation — not just chasing better accuracy.
- Their data pipeline confirms EMDB as the one-stop shop and validates the acquisition strategy.
- They did not use any clustering, SOM, or XAI. This is the gap.

**Key differentiator:** Sun et al. modeled the continuous slope of log-linear decay across all sites simultaneously. This project models discrete behavioral classes with qualitatively different dynamics and maps them onto identifiable landscape units.

---

#### Shuryak (2022) — *RF on biological contamination, not dose rates*

**Reference:** Shuryak, I. (2022). Machine learning analysis of Cs-137 contamination in terrestrial plants after the Fukushima accident. *Journal of Environmental Radioactivity*, 243, 106806.
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0265931X21002447)

**What they did:** Applied RF to predict Cs-137 concentration in terrestrial plant samples (not ambient dose rates).

**Why it matters when drafting this research:**
- Shows RF is established in Fukushima radiation science — but on biological matrices, not dose-rate time series. This project's use of RF on dose-rate regime labels is distinct.
- Demonstrates that RF alone is no longer novel for Fukushima data. An RF-only submission would be incremental. The SOM stage is what elevates the pipeline.
- Cite to show breadth of RF adoption, then differentiate by target variable (dose-rate behavioral class vs. plant contamination level) and the unsupervised discovery stage.

---

#### Shuryak (2023) — *Causal forests — the sophistication ceiling*

**Reference:** Shuryak, I. (2023). Application of causal forests for estimating effects of environmental factors on Cs-137 contamination in trees after Fukushima. *Journal of Environmental Radioactivity*, 261, 107130.
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0265931X2300098X)

**What they did:** Used causal forests (causal inference extension of RF) to estimate heterogeneous treatment effects of environmental variables on Cs-137 in trees. Goes beyond prediction to causal estimation.

**Why it matters when drafting this research:**
- Sets the sophistication ceiling. A reviewer who has read this will expect methodological rigor. Position the SOM-RF pipeline as complementary: Shuryak asks "what causes contamination variation?"; this project asks "what distinct behavioral regimes exist, and what predicts membership in each?"
- Uses biological samples, not dose rates — same distinction as Shuryak 2022.
- Does not use SOM or any clustering. The unsupervised discovery framing remains novel even against this more advanced paper.

---

#### Wainwright et al. (2018) — *Bayesian geostatistics benchmark*

**Reference:** Wainwright, H.M. et al. (2018). Bayesian hierarchical geostatistical analysis of dose rates in Fukushima. *Journal of Environmental Radioactivity*, 189, 120-131.
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0265931X17310032)

**What they did:** Applied Bayesian hierarchical models with geostatistical kriging to interpolate dose rates across the exclusion zone using JAEA airborne and ground survey data.

**Why it matters when drafting this research:**
- Represents the geostatistical tradition this project departs from. Wainwright et al. model spatial correlation to interpolate; this project models temporal behavior to classify. The manuscript should acknowledge geostatistics as the dominant paradigm and position SOM-RF as an alternative that reveals structure geostatistics cannot — discrete regimes with different decay dynamics.
- Does not address temporal heterogeneity. Produces spatial maps at single time snapshots, not behavioral profiles over time. The SOM's temporal feature extraction is the key differentiator.
- Cite in the introduction when establishing what has been done for spatial modeling, before pivoting to the temporal behavioral gap.

---

### The Physical Motivation Paper

#### Andoh et al. (2020) — *Empirical proof that decay rates vary by land-use*

**Reference:** Andoh, M. et al. (2020). Measurement of air dose rates over a wide area around the Fukushima Daiichi Nuclear Power Station through a series of car-borne surveys. *Journal of Nuclear Science and Technology*, 57(12), 1352-1363.
[Tandfonline](https://www.tandfonline.com/doi/full/10.1080/00223131.2020.1789008)

**What they did:** Conducted systematic KURAMA-II car-borne surveys and calculated ecological half-lives of ambient dose rates. Found that half-lives vary significantly by location and land-use type — urban, agricultural, and forest surfaces decay at qualitatively different rates.

**Why it matters when drafting this research:**
- This is the physical motivation for the entire SOM stage. Andoh et al. empirically demonstrated that dose-rate decay is not uniform — it clusters by land-use type. The SOM is designed to rediscover this structure (and potentially finer structure) from the data without imposing land-use categories a priori.
- Provides ground-truth expectations. If SOM clusters roughly align with known land-use-driven half-life differences, that validates the method. If the SOM finds additional structure (sub-regimes within forests, anomalous re-contamination clusters), that is a novel discovery.
- Cite in the Methods section: "Andoh et al. (2020) demonstrated that ecological half-lives vary by land-use, suggesting monitoring locations may exhibit distinct behavioral regimes — a hypothesis the SOM is designed to test without supervised labels."

---

### SOM Methodology Papers

#### Licen et al. (2023) — *SOM review — validates the method, confirms the gap*

**Reference:** Licen, S. et al. (2023). Self-organizing maps as a tool for environmental pollutant pattern recognition. *Science of the Total Environment*, 878, 163084.
[ScienceDirect](https://doi.org/10.1016/j.scitotenv.2023.163084)

**What they did:** Comprehensive review of SOM applications in environmental pollution monitoring across air, water, soil, and sediment. Catalogued methodological patterns, grid sizing, validation approaches, and integration with other ML methods.

**Why it matters when drafting this research:**
- Confirms SOM has never been applied to radioactive contamination. Licen et al. covered environmental pollutants broadly and found no nuclear/radiological SOM application. Directly supports novelty claims 1 and 2.
- Provides methodological authority for grid size (5x5 or 6x6), validation metrics (quantization error, topographic error), and the SOM → supervised learning pipeline pattern.
- Published in Science of the Total Environment (one of the target journals), signaling the journal is receptive to SOM-based environmental methodology.

---

### Structural Analogues (SOM + RF in Other Domains)

These papers validate the two-stage pipeline architecture in peer-reviewed contexts outside radiation science. They are the precedent that makes the SOM → RF combination defensible.

#### Fried et al. (2019) — SOM + RF for hydrocarbon source signatures

**Reference:** Fried, R. et al. (2019). *Environmental Monitoring and Assessment*, 191, 429.
[Springer](https://link.springer.com/article/10.1007/s10661-019-7429-9)

Direct structural analogue — same two-stage architecture (SOM clusters → RF explains membership) applied to hydrocarbon contamination. Validates the pipeline is methodologically sound and peer-reviewable. Cite to establish precedent: "The SOM → RF pipeline has been validated for hydrocarbon source identification (Fried et al. 2019) and groundwater mapping (Appukuttan & Reghunath 2025); this study extends it to radioactive contamination for the first time."

#### Appukuttan & Reghunath (2025) — SOM + RF for groundwater potential mapping

**Reference:** Appukuttan, A. & Reghunath, R. (2025). *Environmental Monitoring and Assessment*, 197, 14779.
[Springer](https://link.springer.com/article/10.1007/s10661-025-14779-9)

Second structural analogue — SOM clustering of hydrogeological variables, then RF for zone prediction. Most recent (2025), showing the approach is current and actively published. Demonstrates to reviewers this is a live methodological trend.

#### Jin et al. (2024) — PARAFAC + SOM + RF for river DOM analysis

**Reference:** Jin, X. et al. (2024). *International Journal of Environmental Science and Technology*.
[Springer](https://link.springer.com/article/10.1007/s41742-024-00574-w)

Three-stage analogue using PARAFAC for feature extraction, SOM for clustering, RF for classification in river chemistry. Shows the multi-stage environmental ML pipeline is a recognized pattern — useful for positioning SOM → RF as methodologically mainstream even though it is novel in the radiation domain.

#### Mia et al. (2023) — SOM + SHAP/LIME for water quality drivers

**Reference:** Mia, M.B. et al. (2023). *Science of the Total Environment*, 905, 166927.
[ScienceDirect](https://doi.org/10.1016/j.scitotenv.2023.166927)

Validates the SOM + XAI combination. If SHAP is included in this project's RF stage, Mia et al. provides precedent for pairing SOM-based clustering with formal explainability methods. Also published in Science of the Total Environment, further confirming the target journal's receptiveness.

---

### Data Decision Papers (Safecast Exclusion)

These papers justify the decision to use JAEA data alone and exclude Safecast.

#### KURAMA vs. Safecast Comparison (2025)

**Reference:** (2025). *Journal of Environmental Radioactivity*.
[ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0265931X25001766)

Systematic comparison: R² = 0.8034 but with systematic bias (Safecast overestimates at low dose rates, underestimates at high). Primary justification for the JAEA-only decision if a reviewer asks "why not use Safecast?"

#### Cervone & Hultquist (2018) — Time-dependent Safecast calibration

**Reference:** Cervone, G. & Hultquist, C. (2018). *Journal of Environmental Radioactivity*, 188, 73-81.
[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0265931X17306884)

Developed a time-dependent calibration model for Safecast to account for Cs-134/Cs-137 isotope ratio drift. The fact that a dedicated paper was needed just for the calibration model reinforces that data fusion is a project unto itself — cite if the manuscript must explicitly justify why fusion was not attempted.

#### Coletti et al. (2017) — Safecast vs. DOE aerial survey validation

**Reference:** Coletti, J. et al. (2017). [OSTI](https://www.osti.gov/biblio/1393887)

Safecast validated against DOE aerial survey — further documents the systematic biases and spatial sampling limitations.

#### Anomaly Detection for Safecast VGI (2021)

**Reference:** (2021). [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/13658816.2021.1981333)

Used KURAMA as ground truth for anomaly detection in Safecast volunteered geographic information. Reinforces JAEA's role as the authoritative reference dataset.

---

## Datasets

### Primary Radiation Data
**JAEA EMDB (Environmental Monitoring Database)** — official government monitoring dose-rate time series from walk surveys, car surveys (KURAMA), and airborne surveys. This is the sole radiation data source; Safecast citizen-science data is excluded due to documented systematic biases (see Data Decision Papers above).

### RF Predictor Datasets — Detailed Profiles

The Random Forest stage requires environmental and geographic predictor layers to explain SOM-discovered regime membership. All datasets are freely available. The profiles below cover what each predictor measures physically, why it matters for Cs-137 dynamics, and where to get it.

#### Core Predictors (EMDB — Sun et al. 2022 baseline)

##### 1. Initial Dose Rate

| Attribute | Detail |
|---|---|
| **What it measures** | First recorded ambient dose equivalent rate (uSv/h) at each monitoring location from JAEA car-borne surveys (KURAMA system) |
| **Physical relevance** | The starting contamination level is the single strongest predictor of current dose rate. Locations with higher initial deposition retain higher absolute levels even after identical fractional decay. |
| **Source** | JAEA EMDB — car-survey (KURAMA) time series |
| **Portal** | https://emdb.jaea.go.jp/emdb/ (no registration) |
| **Format** | CSV with columns for location ID, coordinates, date, dose rate |
| **Resolution** | Point measurements along survey routes, typically 100m intervals |
| **Notes** | Extract the earliest available measurement per location. Some locations have data starting from June 2011; others begin later. Document the reference date used. |

##### 2. Land-Use / Land-Cover Type

| Attribute | Detail |
|---|---|
| **What it measures** | Categorical classification of surface cover: forest, cropland, urban, water, grassland, bare soil, etc. (9 categories in JAXA HRLULC) |
| **Physical relevance** | Land-use is the primary driver of ecological half-life variation (Andoh et al. 2020). Forests retain Cs-137 in leaf litter and organic soil layers, slowing washoff. Urban impervious surfaces shed contamination via rain runoff, accelerating decay. Agricultural land depends on tillage practices. |
| **Source** | EMDB (serves JAXA HRLULC at 100m) or JAXA directly at 10-30m |
| **Portal (EMDB)** | https://emdb.jaea.go.jp/emdb/ |
| **Portal (JAXA)** | https://www.eorc.jaxa.jp/ALOS/en/lulc/lulc_index_v2103.htm |
| **Format** | CSV (via EMDB) or GeoTIFF (via JAXA) |
| **Resolution** | 100m (EMDB) or 10-30m (JAXA direct) |
| **Notes** | Use the EMDB version for consistency with Sun et al. Encode as a categorical factor in R, not numeric. The 9 JAXA categories: water, urban, rice paddy, crop, grass, deciduous broadleaf, deciduous conifer, evergreen broadleaf, evergreen conifer. |

##### 3. Soil Type

| Attribute | Detail |
|---|---|
| **What it measures** | Geological/pedological classification of surface soil from MLIT's 1:200,000 geological map |
| **Physical relevance** | Soil mineralogy controls Cs-137 fixation. Clay-rich soils (especially those with illite and vermiculite) bind cesium tightly in mineral interlayer sites, reducing mobility and biotic uptake. Sandy soils allow greater vertical migration and potential re-mobilization. |
| **Source** | EMDB (serves MLIT 1:200k map) or SoilGrids (250m global) |
| **Portal (EMDB)** | https://emdb.jaea.go.jp/emdb/ |
| **Portal (SoilGrids)** | https://www.isric.org/explore/soilgrids |
| **Format** | CSV (via EMDB) or GeoTIFF (via SoilGrids) |
| **Resolution** | Variable (1:200,000 map scale via EMDB); 250m (SoilGrids) |
| **Notes** | EMDB provides categorical soil types. SoilGrids provides continuous properties (clay %, sand %, organic carbon). Clay fraction from SoilGrids serves as a proxy for soil permeability — clay content inversely correlates with permeability and directly affects Cs retention. Both categorical and continuous representations are valid; choose based on RF cross-validation performance. |

##### 4. Spatial Coordinates

| Attribute | Detail |
|---|---|
| **What it measures** | Geographic position of each monitoring location (easting/northing or latitude/longitude) |
| **Physical relevance** | Captures spatial autocorrelation and large-scale geographic gradients in deposition patterns (e.g., the NW plume direction from FDNPP). |
| **Source** | Included in EMDB radiation data downloads |
| **Format** | Embedded in CSV columns |
| **Notes** | Sun et al. used Japan Plane Rectangular Coordinate System (EPSG:2451 for Zone IX). Use the same CRS for consistency. Also the basis for computing derived spatial predictors (distance-to-FDNPP, distance-to-rivers). |

---

#### Extended Predictors (expanding beyond Sun et al. 2022)

##### 5. Elevation (DEM)

| Attribute | Detail |
|---|---|
| **What it measures** | Height above sea level (meters) |
| **Physical relevance** | Elevation influences precipitation patterns, runoff velocity, soil moisture retention, and vegetation type — all affecting Cs-137 redistribution. Higher elevations in the Abukuma Highlands received different deposition patterns than coastal lowlands. Correlates with land-use transitions (forest at higher elevation, agriculture/urban at lower). |
| **Source** | EMDB (GSI 10m DEM) or GSI portal directly (5m DEM) |
| **Portal (EMDB)** | https://emdb.jaea.go.jp/emdb/ |
| **Portal (GSI)** | https://fgd.gsi.go.jp/download/menu.php (free account required) |
| **Format** | CSV (EMDB) or GeoTIFF (GSI) |
| **Resolution** | 10m (EMDB) or 5m (GSI direct) |
| **Notes** | 10m from EMDB is sufficient. The 5m GSI DEM is overkill given monitoring point spacing. Also used to derive slope and aspect. |

##### 6. Slope

| Attribute | Detail |
|---|---|
| **What it measures** | Steepness of terrain (degrees or percent) |
| **Physical relevance** | Controls surface runoff velocity. Steep slopes promote rapid washoff of particulate-bound Cs-137, leading to faster apparent dose-rate decay. Flat areas accumulate runoff-transported contamination in depressions. Affects erosion potential and soil redistribution. |
| **Source** | Derived from DEM in R using `terra::terrain()` |
| **Format** | Computed raster |
| **Notes** | Derive after obtaining the DEM. Use degrees for interpretability in SHAP plots. |

##### 7. Aspect

| Attribute | Detail |
|---|---|
| **What it measures** | Compass direction a slope faces (0-360 degrees) |
| **Physical relevance** | Affects solar exposure, evapotranspiration, snow melt timing, and wind-driven deposition. NW-facing slopes may have received different fallout loads due to the prevailing plume direction during the March 2011 accident. |
| **Source** | Derived from DEM in R using `terra::terrain()` |
| **Format** | Computed raster |
| **Notes** | Circular variable — consider transforming to sin(aspect) and cos(aspect) for the RF, or use as-is since RF handles non-linear splits natively. |

##### 8. Distance from FDNPP

| Attribute | Detail |
|---|---|
| **What it measures** | Euclidean distance (km) from each monitoring point to the Fukushima Daiichi Nuclear Power Station |
| **Physical relevance** | Distance from the source is a first-order control on initial deposition intensity. The relationship is not monotonic due to the directional NW plume — some locations close to FDNPP received less fallout than locations further away in the plume direction. Captures the radial component of the spatial deposition pattern. |
| **Source** | Computed from monitoring point coordinates and FDNPP reactor coordinates (141.032E, 37.422N) |
| **Format** | Computed numeric column |
| **Notes** | Use geodesic distance (Vincenty or Haversine) if coordinates are in lat/lon. If projected to a planar CRS (e.g., EPSG:2451), Euclidean is fine. |

##### 9. Distance to Nearest River

| Attribute | Detail |
|---|---|
| **What it measures** | Distance (meters) from each monitoring point to the nearest river channel |
| **Physical relevance** | Rivers act as sinks and transport corridors for Cs-137. Locations near rivers may show faster dose-rate decay due to hydrological washoff into the channel. Floodplains can also accumulate redeposited contamination during flood events, creating anomalous hot spots. |
| **Source** | MERIT Hydro (global 90m river network) or MLIT river shapefiles |
| **Portal (MERIT Hydro)** | https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/ |
| **Portal (MLIT)** | http://nlftp.mlit.go.jp/ksj-e/index.html |
| **Format** | Raster (MERIT) or Shapefile (MLIT) |
| **Resolution** | 90m (MERIT Hydro); vector (MLIT) |
| **Notes** | Compute distance using `sf::st_distance()` or `terra::distance()`. MERIT Hydro is globally consistent; MLIT provides official Japanese river data with higher local accuracy. |

##### 10. Initial Cs-137 Deposition

| Attribute | Detail |
|---|---|
| **What it measures** | Total Cs-137 deposited on the ground surface (kBq/m2) as of a reference date (July 2, 2011 for the Tsukuba map) |
| **Physical relevance** | The total inventory of Cs-137 at each location is the fundamental source term. Unlike initial dose rate (which depends on measurement geometry and detector height), deposition represents the actual contamination burden. Captures the spatial heterogeneity of the fallout plume. |
| **Source** | U. Tsukuba fallout map (250m mesh, decay-corrected to July 2, 2011) or EMDB airborne survey data |
| **Portal** | https://www.ied.tsukuba.ac.jp/~fukushimafallout/ |
| **Format** | CSV or Shapefile (250m mesh cells) |
| **Resolution** | 250m |
| **Notes** | The Tsukuba map combines airborne survey, in-situ measurement, and soil sampling data — the most widely cited deposition reference map. Extract values at monitoring points using `terra::extract()`. Deposition (kBq/m2) and dose rate (uSv/h) are related but not identical due to shielding, detector height, and surface geometry. Including both provides complementary information. |

##### 11. Precipitation

| Attribute | Detail |
|---|---|
| **What it measures** | Mean annual precipitation (mm/year) or monthly precipitation climatology |
| **Physical relevance** | Rainfall drives the primary mechanism of Cs-137 removal from impervious surfaces (washoff) and vertical migration in soils (percolation). High-precipitation areas on the eastern slopes of the Abukuma Highlands are expected to show faster dose-rate decay on urban surfaces but potentially slower decay in forested soils (rainfall drives Cs deeper into the organic layer without removing it). |
| **Source** | WorldClim 2.1 |
| **Portal** | https://www.worldclim.org/data/index.html |
| **Format** | GeoTIFF |
| **Resolution** | ~1km (30 arc-seconds) |
| **Notes** | Download "bio" variables (BIO12 = annual precipitation) or monthly data. 1km resolution is coarse relative to other predictors but sufficient for regional precipitation gradients. Extract values at monitoring points. |

##### 12. NDVI (Vegetation Density)

| Attribute | Detail |
|---|---|
| **What it measures** | Normalized Difference Vegetation Index — vegetation greenness and density (range: -1 to +1, typically 0.2-0.8 for vegetated areas) |
| **Physical relevance** | Dense vegetation canopies intercepted fallout during the initial deposition event and continue to cycle Cs-137 through leaf litter. Forest canopy NDVI correlates with organic layer thickness and Cs retention capacity. NDVI also captures defoliation/remediation effects — areas cleared for decontamination show NDVI drops. |
| **Source** | MODIS MOD13Q1 (250m, 16-day composites) or Sentinel-2 (10m) |
| **Portal (MODIS)** | https://lpdaac.usgs.gov/products/mod13q1v006/ |
| **Format** | GeoTIFF (HDF for MODIS raw) |
| **Resolution** | 250m (MODIS) or 10m (Sentinel-2) |
| **Notes** | MODIS 250m recommended for simplicity and temporal consistency. Use a summer composite (July-August) for peak vegetation. Consider averaging across multiple years (2012-2020) to smooth interannual variability. Optional free EarthData account may be needed for bulk downloads. |

---

### Predictor Summary Table

| # | Predictor | Type | Physical Role in Cs-137 Dynamics | Source |
|---|---|---|---|---|
| 1 | Initial dose rate | Continuous | Source term magnitude | EMDB |
| 2 | Land-use type | Categorical (9 classes) | Controls ecological half-life via surface properties | EMDB (JAXA) |
| 3 | Soil type | Categorical / Continuous | Cs-137 fixation capacity and vertical migration | EMDB (MLIT) / SoilGrids |
| 4 | Easting, Northing | Continuous | Spatial autocorrelation, plume geometry | EMDB |
| 5 | Elevation | Continuous | Precipitation, runoff, vegetation gradients | EMDB (GSI) |
| 6 | Slope | Continuous | Runoff velocity, erosion potential | Derived from DEM |
| 7 | Aspect | Circular / Continuous | Solar exposure, plume-facing orientation | Derived from DEM |
| 8 | Distance from FDNPP | Continuous | Radial deposition gradient | Computed |
| 9 | Distance to river | Continuous | Hydrological washoff and redeposition | MERIT Hydro / MLIT |
| 10 | Initial Cs-137 deposition | Continuous | Total fallout inventory (kBq/m2) | U. Tsukuba |
| 11 | Precipitation | Continuous | Washoff and percolation driver | WorldClim |
| 12 | NDVI | Continuous | Canopy interception, organic layer cycling | MODIS |

### Data Portals

| Portal | URL | What It Provides |
|---|---|---|
| JAEA EMDB (legacy) | https://emdb.jaea.go.jp/emdb_old/en/ | Radiation data + DEM + LULC + soil map (no registration) |
| JAEA EMDB (current) | https://emdb.jaea.go.jp/emdb/ | Same as above, migrated interface |
| U. Tsukuba Fallout Map | https://www.ied.tsukuba.ac.jp/~fukushimafallout/ | Initial Cs-137 deposition (July 2, 2011 reference date) |
| GSI DEM Portal | https://fgd.gsi.go.jp/download/menu.php | 5m/10m DEM (free account required) |
| JAXA LULC | https://www.eorc.jaxa.jp/ALOS/en/lulc/lulc_index_v2103.htm | High-resolution land cover |
| SoilGrids | https://www.isric.org/explore/soilgrids | Global 250m soil properties (clay, sand, organic carbon) |
| WorldClim | https://www.worldclim.org/data/index.html | Global climate data (~1km) |
| MODIS NDVI | https://lpdaac.usgs.gov/products/mod13q1v006/ | 250m 16-day vegetation index |
| MERIT Hydro | https://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/ | Global river network |
| MLIT NLNI | http://nlftp.mlit.go.jp/ksj-e/index.html | Japanese administrative, land-use, river shapefiles |

---

## Approach

### Stage 1 — Unsupervised Discovery (SOM)
Train a Self-Organizing Map on multivariate dose-rate temporal features (decay slopes, half-lives, temporal variance) extracted from JAEA monitoring points. The SOM clusters monitoring locations into distinct behavioral regimes — e.g., fast-decaying urban surfaces, slow-decaying forest floors, anomalous re-contamination zones — without imposing these categories a priori.

- Grid size: 5x5 or 6x6
- Stability tested across 3-5 random seeds
- Quality assessed via quantization error and topographic error
- Methodological grounding: Licen et al. (2023) review of SOM in environmental science

### Stage 2 — Supervised Explanation (RF)
Train a Random Forest classifier using SOM cluster labels as the target variable and the 12 environmental/geographic predictors as features. The RF answers: *what landscape and environmental factors determine which behavioral regime a location belongs to?*

- Start with Sun et al.'s 4 EMDB predictors, expand to 8-12
- Feature importance via Gini importance, permutation importance, and SHAP values
- SHAP provides the first formal XAI application on a Fukushima ML model
- Structural precedent: Fried et al. (2019), Appukuttan & Reghunath (2025)

### Integration
Map SOM-discovered regimes geographically and overlay them against Japan's official evacuation/restriction zones to reveal where data-driven behavioral boundaries align with or diverge from policy boundaries.

---

## Expected Results

After completing this pipeline, the following results are expected:

1. **Distinct contamination regimes identified** — The SOM will reveal 4-8 behaviorally distinct clusters of monitoring locations, each with a characteristic dose-rate decay profile (e.g., rapid urban washoff, slow forest retention, plateau/re-contamination patterns). These should broadly align with Andoh et al.'s (2020) land-use-driven half-life variation, with potential for finer-grained structure.

2. **Environmental drivers of regime membership explained** — The RF will rank which predictors most strongly determine regime membership, with SHAP values providing per-feature, per-location explanations. Land-use and initial deposition are expected to dominate, with elevation, precipitation, and soil type contributing secondary explanatory power.

3. **Geographic regime map** — A spatial map showing SOM-discovered regimes across the exclusion zone, directly comparable to official evacuation zone boundaries. This is the hero figure.

4. **Divergence between data-driven and policy-driven boundaries** — Locations where SOM regimes disagree with official zone classifications highlight areas where contamination behavior deviates from policy assumptions — actionable insight for remediation planning.

5. **Publication-ready manuscript** with four novelty claims:
   - First SOM application to Fukushima dose-rate data
   - First SOM application to any radioactive contamination dataset
   - First SOM → RF pipeline in radiation science
   - First SHAP/XAI analysis on a Fukushima ML model

---

## Important Notes

- **No data fusion**: Safecast data is not used. JAEA is the sole data source, consistent with all prior high-quality Fukushima ML studies. Justified by KURAMA vs. Safecast (2025) and Cervone & Hultquist (2018).
- **Value is in the framing, not the metrics**: The paper's contribution is the interpretive framework (discovery before explanation), not chasing maximum R2 or accuracy scores.
- **Scope discipline**: Do not over-optimize SOM grids or expand beyond 8-12 RF predictors. Keep the pipeline lean and interpretable.
- **Tools**: R (kohonen, randomForest/ranger, sf, terra, SHAP packages) as the primary environment.
- **Target journals**: Science of the Total Environment (IF ~8.2), Environmental Pollution (IF ~7.6), J. Environmental Radioactivity (IF ~2.6).
