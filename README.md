# METLN Article Processing and Classification

This project partners with METLN to classify news articles into six User Needs categories using supervised and unsupervised machine learning, enabling the newsroom to retroactively tag historical content and analyze their content strategy across years of coverage.

> Final project for Unsupervised Machine Learning, Spring 2026.

---

## Project Structure

```
UML-FINAL-PROJECT/
├── notebooks/
│   ├── article_classification_EDA.ipynb      #exploratory data analysis
│   ├── data_prep-bv.ipynb                    #data cleaning & preprocessing
│   ├── Supervised Learning(Part -1).ipynb    #supervised classifiers
│   ├── #here
│   └── article_classification.ipynb          #full classification pipeline
├── src/
│   ├── config.py                             #all hyperparameters and paths
│   └── data.py                               #data loading and processing utilities
├── data/                                     #gitignored, CSV files go here
├── requirements.txt
└── README.md
```

---

## Dataset

Data is hosted on Google Drive:

[Download from Google Drive](https://drive.google.com/drive/folders/1qEl6-kTLLMMlakKNZrRNlJGNTqQjS9XV?usp=sharing)

After downloading, place files in the `data/` folder.

---

## Environment Setup

### Option A - pip (recommended for most users)

```bash
pip install -r requirements.txt
```

> **PyTorch note:** For GPU/CUDA support, install PyTorch separately from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above.

### Option B - conda

```bash
conda create -n METLN-classification python=3.10
conda activate METLN-classification
pip install -r requirements.txt
```

---

## Quick Start

Run notebooks in this order:

1. `article_classification_EDA.ipynb` - understand the data
2. `data_prep-bv.ipynb` - clean and prepare features
3. `article_classification.ipynb` - full pipeline with embeddings and deep models
4. `Supervised Learning(Part -1).ipynb` - train and evaluate ML classifiers
5. `#here`

---

## Configuration

All hyperparameters and paths are centralized in `src/config.py`. Key settings can be overridden via environment variables:

**Linux/macOS (bash):**
```bash
export DATA_DIR=/path/to/data
export MODEL_DIR=/path/to/models
export OUTPUT_DIR=/path/to/outputs
```

**Windows (PowerShell):**
```powershell
$env:DATA_DIR = "C:\path\to\data"
$env:MODEL_DIR = "C:\path\to\models"
$env:OUTPUT_DIR = "C:\path\to\outputs"
```

Notable config values in `src/config.py`:

| Setting | Default | Description |
|---|---|---|
| `SETFIT_BASE_MODEL` | `all-MiniLM-L6-v2` | SetFit base embedding model |
| `LONGFORMER_MODEL_ID` | `allenai/longformer-base-4096` | Longformer model |
| `TEST_SIZE` | `0.2` | Train/test split ratio |
| `CV_N_SPLITS` | `5` | Cross-validation folds |
| `MAX_CHARS` | `2000` | Article text truncation limit |

---

## User Need Labels

The 6 classification labels used throughout the project:

| Label | Description |
|---|---|
| `update-me` | Breaking news / status updates |
| `educate-me` | Explanatory / deep-dive content |
| `give-me-perspective` | Opinion / analysis |
| `help-me` | Practical how-to guidance |
| `connect-me` | Community / people stories |
| `inspire-me` | Uplifting / motivational content |
