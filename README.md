# Fukushima Contamination Regime Discovery

**Self-Organizing Map Classification of Radioactive Contamination Decay Regimes in the Fukushima Exclusion Zone**

This repository contains the code, figures, and key outputs for the analysis of spatiotemporal contamination decay patterns across the Fukushima Daiichi exclusion zone. A 5×5 Self-Organizing Map trained on 9 temporal dose-rate features discovers 6 distinct contamination decay regimes across 138,768 mesh cells, without imposing prior categories.

## Pipeline Overview

| Phase | Script | Description |
|-------|--------|-------------|
| 1. Data Acquisition | `download_emdb.py` | Downloads JAEA EMDB air dose rate datasets |
| 2. Data Filtration | `filter_emdb.py` | Six-stage filter chain: 14.5M → 9.2M rows (138,768 mesh cells) |
| 3. Feature Engineering | `engineer_features.py` | Extracts 9 temporal features per mesh cell |
| 4. SOM Clustering | `som_clustering.R` | Trains 5×5 SOM, discovers 6 regimes via Ward's D2 hierarchical clustering |

## Data Source

Air dose rate measurements are from Japan's [Extension Site of Distribution Map for Radioactive Substances](https://ramap.jmc.or.jp/map/eng/) (EMDB), maintained by the Japan Atomic Energy Agency (JAEA) and the Nuclear Regulation Authority (NRA). The raw data (~534 MB) is not included in this repository. To reproduce from scratch:

```bash
# Phase 1: Download raw EMDB data
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 download_emdb.py

# Phase 2: Filter
python3 filter_emdb.py

# Phase 3: Engineer temporal features
python3 engineer_features.py

# Phase 4: SOM clustering (requires R with kohonen package)
Rscript som_clustering.R
```

Run any Python script with `--help` for usage options, or `--dry-run` to preview without processing.

## Key Results

Six contamination decay regimes identified:

| Regime | Proportion | Characterization |
|--------|-----------|------------------|
| 1. Slow Retention | 9.9% | High dose retention in forested/mountainous terrain, half-life ~13.8 yr |
| 2. Rapid Decay | 6.2% | Biphasic decay with fast initial washoff, half-life ~9.3 yr |
| 3. Standard Decay | 55.3% | Modal behavior, near-average characteristics across the exclusion zone |
| 4. Recontamination | 7.3% | Frequent dose increases and reversals, closest to FDNPP |
| 5. Consistent Exponential | 8.3% | Clean exponential fit (R²=0.84), fewest reversals, half-life ~4.7 yr |
| 6. High Variability | 13.1% | Erratic behavior near reactor, highest initial dose (7.66 µSv/h) |

## Repository Structure

```
├── download_emdb.py          # Phase 1: data acquisition
├── filter_emdb.py            # Phase 2: six-stage filtration
├── engineer_features.py      # Phase 3: temporal feature extraction
├── som_clustering.R          # Phase 4: SOM training and clustering
├── requirements.txt          # Python dependencies
├── research-objective.md     # Research framing and literature context
├── figures/                  # 13 publication-ready figures (PNG 300dpi + PDF vector)
│   ├── fig01_geographic_regime_map.*
│   ├── fig02_centroid_heatmap.*
│   ├── ...
│   └── fig13_cluster_boxplots.*
└── output/
    ├── som_clusters.parquet       # 6-cluster assignments for 138,768 mesh cells
    ├── som_cluster_summary.csv    # Centroids and statistics (Table 1)
    └── som_metrics.json           # Validation metrics and ARI matrices
```

## Citation

If you use this code or data, please cite:

> Alsayed, E., Fatani, O., Quraby, W., Abulhasan, W., & Mubarki, S. (2026). Discovery of Contamination Decay Regimes in the Fukushima Exclusion Zone Using Self-Organizing Maps. *[Journal TBD]*.

## License

MIT License. See [LICENSE](LICENSE) for details.
