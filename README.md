# Fukushima Contamination Regime Discovery

**Self-Organizing Map Classification of Radioactive Contamination Decay Regimes in the Fukushima Exclusion Zone**

This repository contains the code, figures, and key outputs for an analysis of spatiotemporal contamination decay patterns across the Fukushima Daiichi exclusion zone. A 5x5 Self-Organizing Map trained on **8 temporal dose-rate features** discovers **4 contamination decay regimes** across 138,768 mesh cells, without imposing prior categories. A supervised random forest with per-class SHAP attribution, trained on independent environmental and geographic predictors, externally validates the three physically robust regimes.

## Pipeline Overview

| Phase | Script | Description |
|-------|--------|-------------|
| 1. Data acquisition | `download_emdb.py` | Downloads JAEA EMDB air dose rate datasets |
| 2. Data filtration | `filter_emdb.py` | Filter chain: 14.5M records to 138,768 mesh cells |
| 3. Feature engineering | `engineer_features.py` | Extracts the temporal decay features per mesh cell |
| 4. SOM training | `som_clustering.R` | Trains 5x5 and 6x6 hexagonal SOMs on the 8 features, five seeds each, and selects the map |
| 5. Cluster finalization | `som_finalize_k4.R` | Ward's D2 clustering of the codebook, k=4 selection, regime labels, figures, and outputs |
| 6. External validation | `acquire_geodata.py`, `acquire_rf_predictors.py`, `build_rf_matrix.R`, `train_rf_shap.R` | Assembles 10 exogenous predictors and trains a random forest with per-class TreeSHAP under spatial-block cross-validation |

`som_clustering.R` writes intermediate candidate artifacts; `som_finalize_k4.R` reads them, cuts the codebook at k=4, applies the regime labels, and writes the primary outputs and the SOM-derived figures.

## Data Source

Air dose rate measurements are from Japan's [Extension Site of Distribution Map for Radioactive Substances](https://ramap.jmc.or.jp/map/eng/) (EMDB), maintained by the Japan Atomic Energy Agency (JAEA) and the Nuclear Regulation Authority (NRA). The raw EMDB archive (~534 MB) is not included here. The random-forest predictors are drawn from public sources (MLIT National Land Numerical Information, a University of Tsukuba airborne Cs-137 deposition map, and the MLIT 10 m digital elevation model). To reproduce from scratch:

```bash
# Phases 1-3: acquire, filter, engineer features (Python)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 download_emdb.py
python3 filter_emdb.py
python3 engineer_features.py

# Phases 4-5: SOM training and k=4 finalization (R, requires the kohonen package)
Rscript som_clustering.R
Rscript som_finalize_k4.R

# Phase 6: external validation (R, requires ranger, blockCV, treeshap)
python3 acquire_geodata.py
python3 acquire_rf_predictors.py
Rscript build_rf_matrix.R
Rscript train_rf_shap.R
```

Run any Python script with `--help` for usage options, or `--dry-run` to preview without processing.

## Key Results

Four contamination decay regimes, from a 5x5 SOM (seed 456) with Ward's D2 clustering at k=4 (quantization error 1.01, topographic error 0.29, mean pairwise seed Adjusted Rand Index 0.77). Half-life, distance, and initial dose rate are reported as median [Q1, Q3]:

| Regime | Share | Half-life (y) | Distance to FDNPP (km) | Initial dose (uSv/h) | Characterization |
|--------|-------|---------------|------------------------|----------------------|------------------|
| Slow Decay | 20.7% | 7.92 [6.74, 10.44] | 66.5 [51.5, 75.1] | 0.18 [0.13, 0.24] | Far-field forest retention; long half-life, low dose |
| Standard Decay | 59.8% | 5.05 [4.47, 5.77] | 56.8 [43.5, 67.9] | 0.50 [0.34, 0.81] | Modal background; features near the population median |
| Near-Field Hotspot | 14.3% | 3.81 [3.54, 4.12] | 27.6 [17.1, 37.9] | 3.9 [2.3, 7.7] | Near-source, highest dose; the true recontamination zone (flag 2.7x the population rate) |
| Irregular Decay | 5.2% | 7.89 [6.22, 10.55] | 54.7 [40.4, 66.3] | 0.33 [0.24, 0.53] | Poor exponential fit; frequent minor upticks; isolated as a survey-cadence acquisition artifact |

A bootstrap clusterwise-stability check places all four regimes in the 0.6 to 0.75 band, and a random forest on exogenous predictors recovers the three physical regimes (Slow, Standard, Near-Field) under spatial-block cross-validation at macro-F1 0.85 to 0.87. SHAP attribution shows that contamination magnitude and proximity to the source, rather than land use, set regime membership.

## Repository Structure

```
.
├── download_emdb.py                # Phase 1: data acquisition
├── filter_emdb.py                  # Phase 2: filtration
├── engineer_features.py            # Phase 3: temporal feature extraction
├── som_clustering.R                # Phase 4: 8-feature SOM training
├── som_finalize_k4.R               # Phase 5: Ward's D2 clustering + k=4 finalization + figures
├── acquire_geodata.py              # Phase 6: geographic/environmental layer acquisition
├── acquire_rf_predictors.py        # Phase 6: predictor assembly
├── build_rf_matrix.R               # Phase 6: design-matrix build + exogeneity checks
├── train_rf_shap.R                 # Phase 6: random forest + per-class TreeSHAP
├── make_rf_stability_figure.R      # Spatial-CV stability figure
├── requirements.txt                # Python dependencies
├── figures/                        # publication figures (PNG 300 dpi + PDF vector + captions)
└── output/
    ├── som_clusters.parquet            # 4-cluster assignments for 138,768 mesh cells
    ├── som_cluster_summary.csv         # centroids and statistics
    ├── som_model.rds                   # trained SOM model object
    ├── som_metrics.json                # validation metrics and k-selection
    ├── som_stability_clusterboot.json  # bootstrap clusterwise-stability diagnostics
    ├── rf_design_matrix.parquet        # random-forest predictor matrix
    ├── rf_cv_metrics.json              # spatial-block and random CV metrics
    ├── rf_importance.csv               # permutation importance + SHAP rank
    └── rf_shap_values.parquet          # per-class SHAP values (5k subsample)
```

## Citation

If you use this code or data, please cite:

> Alsyed, E., Fatani, O., Quraby, W., & Abulhasan, W. (2026). Discovery of Contamination Decay Regimes in the Fukushima Exclusion Zone Using Self-Organizing Maps. Manuscript in preparation.

## License

MIT License. See [LICENSE](LICENSE) for details.
