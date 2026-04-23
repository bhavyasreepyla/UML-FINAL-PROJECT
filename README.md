# METLN Article Processing and Classification

## Environment and Dependencies

```bash
# 1. Install dependencies (pick one)
bash scripts/setup.sh --pip       # Install via pip
bash scripts/setup.sh --conda     # Create conda env
bash scripts/setup.sh --check     # Just check what's installed

# 2. Activate environment (if using conda)
conda activate METLN-classification
```

## Dataset Setup

Data from Google Drive
https://drive.google.com/drive/folders/1qEl6-kTLLMMlakKNZrRNlJGNTqQjS9XV?usp=sharing

```bash
# Place your raw data
cp posts-export-*.csv data/
# or, if you already have the cleaned dataset:
cp EDA_data-FULL.csv data/
```

## Configuration

All hyperparameters and paths are centralized in `src/config.py`. Key settings can be overridden via environment variables:

```bash
export DATA_DIR=/path/to/data
export MODEL_DIR=/path/to/models
export OUTPUT_DIR=/path/to/outputs
```
