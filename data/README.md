# Data Directory

This folder manages datasets used throughout the IVF-CE project. We keep raw downloads and cached NumPy conversions so experiments can run reproducibly without re-downloading or re-parsing large binary files.

## Layout

```
data/
├── README.md                 # This document
├── download_datasets.py      # CLI helper to fetch datasets
├── dataset_loader.py         # Load datasets into NumPy arrays
├── utils.py                  # Parsers for .fvecs/.ivecs/.bvecs files
├── raw/                      # Raw downloads (as provided by dataset authors)
│   └── sift1m/               # Example: SIFT1M tarball and extracted files
└── processed/                # Cached .npy/.npz files for fast reloads
    └── sift1m/
```

## SIFT1M

The SIFT1M dataset is hosted by the `TEXMEX` corpus:

- URL: `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz`
- Contents after extraction:
  - `sift_base.fvecs` – 1,000,000 base vectors (float32, 128-dim)
  - `sift_query.fvecs` – 10,000 query vectors (float32, 128-dim)
  - `sift_groundtruth.ivecs` – Top-100 ground-truth neighbors per query (int32)
  - `sift_learn.fvecs` – Optional training set (used for coarse quantizer training)

### Downloading

To download and extract SIFT1M into `data/raw/sift1m/`:

```bash
python -m data.download_datasets
```

This script saves the compressed tarball and extracts its contents. If the archive already exists, rerun with `overwrite=True` when calling `download_sift1m`.

### Loading

Experiments should rely on the loader functions:

```python
from data.dataset_loader import load_dataset

base, queries, ground_truth = load_dataset("sift1m")
```

The loader expects the raw `.fvecs`/`.ivecs` files to exist under `data/raw/sift1m/sift/`.

## Adding New Datasets

1. Add a download helper to `data/download_datasets.py`.
2. Implement a loader in `data/dataset_loader.py` and register it in `load_dataset`.
3. Document the dataset here.

## Testing

Use small synthetic arrays directly in tests when large datasets are unavailable.
