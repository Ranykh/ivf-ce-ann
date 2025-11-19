# IVF-CE Approximate Nearest Neighbor Search on SIFT1M

**Course:** Data Analysis and Presentation – Final Project  
**Team:** Tameer Milhem · Luay Marzok · Rany Khirbawi · Weaam Mulla

This project implements and analyzes an Inverted File Index with Cross-Cluster Exploration (IVF-CE) for Approximate Nearest Neighbor (ANN) search on the SIFT1M dataset.

The work is divided into three stages:

1. Stage 1 — Hyperparameter Analysis  
Compare three IVF-CE presets (A/B/C) and select the final configuration.
2. Stage 2 — Budget & Baseline Comparison  
Compare IVF-CE vs standard IVF and brute force across search budgets.
3. Stage 3 — Cross-Cluster Link Behavior  
Study how cross-cluster links are actually used during search and how much recall they contribute.

The repository contains all indexes, logs, and notebooks required to reproduce the evaluation.

⸻

## Repository Structure

```text
ivf-ce-ann/
│
├── PROJECT_CONTEXT.md
├── README.md
├── config/                     # Default settings and preset configs
│   ├── default_config.yaml
│   └── presets/
│       ├── A.yaml
│       ├── B.yaml
│       └── C.yaml
│
├── data/                       # Dataset root (SIFT1M expected here)
│   ├── download_datasets.py
│   └── raw/sift1m/             # Downloaded TexMex SIFT1M data
│
├── evaluation/
│   ├── evaluator.py            # Recall + latency measurement
│   └── metrics.py
│
├── experiments/
│   ├── run_ivfce_workflow.py   # CLI for building indexes and running sweeps
│   ├── run_baseline_comparison.py
│   ├── run_ablation_study.py
│   └── utils.py
│
├── notebooks/
│   ├── 01_stage1_hyperparam_analysis.ipynb
│   ├── 02_stage2_budget_comparison.ipynb
│   └── 03_stage3_link_analysis.ipynb
│
├── results/
│   ├── indexes/                # Saved IVF-CE / IVF indexes
│   └── logs/                   # Stage 1 + Stage 2 logs
│
├── runs/
│   └── stage3_link_analysis.jsonl   # Stage 3 per-query diagnostics
│
├── src/
│   ├── baselines/
│   │   ├── bruteforce.py       # Exact search
│   │   └── faiss_baselines.py
│   ├── clustering/
│   │   └── kmeans.py
│   ├── index/
│   │   ├── base_index.py
│   │   ├── cross_links.py
│   │   ├── ivf_index.py        # Standard IVF index
│   │   ├── ivf_ce_index.py     # IVF-CE index (with cross-cluster links)
│   │   ├── metadata.py
│   │   └── presets.py
│   ├── search/
│   │   ├── base_searcher.py
│   │   ├── ivf_searcher.py
│   │   ├── ivf_ce_searcher.py
│   │   ├── ivf_ce_link_analysis.py  # Stage 3 diagnostics
│   │   └── utils.py
│   └── utils/
│       ├── distance.py
│       └── logger.py
│
├── tests/
│   ├── test_clustering.py
│   ├── test_data.py
│   ├── test_experiments_utils.py
│   ├── test_full_pipeline.py
│   ├── test_index.py
│   ├── test_index_pipeline.py
│   ├── test_presets.py
│   └── test_search.py
│
└── requirements.txt
```

⸻

## 1️⃣ Environment Setup

```bash
cd ivf-ce-ann
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

This installs FAISS, NumPy, SciPy, Jupyter, and all dependencies.  
The active environment allows notebooks to automatically import project modules.

## 2️⃣ Download the SIFT1M Dataset

```bash
python data/download_datasets.py
```

This downloads and extracts the TexMex SIFT1M dataset to:  
`data/raw/sift1m/`

## 3️⃣ Stage 1 — Build IVF-CE Indexes + Run Sweeps

Build all IVF-CE presets (A/B/C)

```bash
for preset in A B C; do
  python -m experiments.run_ivfce_workflow \
    --config config/default_config.yaml \
    build-index \
    --preset "$preset" \
    --output-dir results/indexes
done
```

This produces:

```text
results/indexes/
  ivfce_sift1m_a_*.idx
  ivfce_sift1m_b_*.idx
  ivfce_sift1m_c_*.idx
```

Run search sweeps for each preset

```bash
python -m experiments.run_ivfce_workflow \
  run-search --preset A \
  --index-path results/indexes/ivfce_sift1m_a_*.idx \
  --log-path results/logs/search_runs.jsonl

python -m experiments.run_ivfce_workflow \
  run-search --preset B \
  --index-path results/indexes/ivfce_sift1m_b_*.idx \
  --log-path results/logs/search_runs.jsonl

python -m experiments.run_ivfce_workflow \
  run-search --preset C \
  --index-path results/indexes/ivfce_sift1m_c_*.idx \
  --log-path results/logs/search_runs.jsonl
```

All results go into:  
`results/logs/search_runs.jsonl`

Analyze Stage 1

Open:  
`notebooks/01_stage1_hyperparam_analysis.ipynb`

This notebook:

- Loads all Stage 1 logs
- Compares presets A/B/C
- Chooses Config B as the best compromise of recall vs latency

## 4️⃣ Stage 2 — IVF-CE vs IVF vs Brute Force

Open:  
`notebooks/02_stage2_budget_comparison.ipynb`

Before running:

- Set RUN_EVALUATIONS = True.

Then run all cells. This will:

- Load the Config B IVF-CE index
- Build IVF baseline
- Build brute force
- Evaluate B ∈ {2, 3, 4, 5, 6, 8}
- Log results to:

  `results/logs/stage2_results.jsonl`

The notebook generates:

- Recall–latency scatter plots
- Recall vs budget curves
- Indexing time comparison
- Effect of redistributing the budget between n1 and n2

## 5️⃣ Stage 3 — Cross-Cluster Link Diagnostics

Open:  
`notebooks/03_stage3_link_analysis.ipynb`

Before running:

- Set RUN_EVALUATIONS = True.

The notebook will:

1. Run per-query diagnostics for every (B, n1, n2).
2. Save logs to:

   `runs/stage3_link_analysis.jsonl`

3. Produce:

   - Link vs fallback usage plots
   - Recall gain from n2 = 1
   - Overlap between link-recommended clusters and IVF next-centroids

This isolates the true contribution of cross-cluster links.

⸻

## 6️⃣ Reproducing All Figures

Once Stage 1–3 logs are generated, simply open each notebook and click:  
Run → Run All

All plots and tables used in the report will regenerate automatically.
