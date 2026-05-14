"""Persist and load trained models (sklearn, SetFit) plus an optional registry.

Library-only. Local filesystem is the default; MLflow and ONNX paths are
provided as opt-ins that fail soft when their packages aren't installed.

Inputs:
    Trained models, fitted encoders, vectorizers, or metrics dicts.

Outputs:
    Files under `models/artifacts/`:
      <name>.joblib                  — sklearn-style estimators
      label_encoder.joblib           — LabelEncoder
      tfidf_vectorizer.joblib        — TF-IDF vectorizer
      <name>/                        — SetFit save_pretrained directory
      model_registry.json            — { model_name: {path, metrics, ...} }
"""

import json
import os
import time
from typing import Any, Dict, Optional

import joblib

from config import LOG_DIR, MODEL_DIR


def save_sklearn_model(
    model,
    name: str,
    metadata: Optional[Dict] = None,
    model_dir: str = MODEL_DIR,
) -> str:
    """Serialize an sklearn-style model with optional metadata sidecar.

    In:  fitted model; filename stem; optional metadata dict; target dir.
    Out: path to the written `.joblib` file; also writes `<name>_meta.json`
         when metadata is provided.
    """
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{name}.joblib")
    joblib.dump(model, model_path)

    if metadata:
        meta_path = os.path.join(model_dir, f"{name}_meta.json")
        metadata["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"Saved {name} to {model_path}")
    return model_path


def load_sklearn_model(name: str, model_dir: str = MODEL_DIR):
    """Load an sklearn-style model previously written by `save_sklearn_model`.

    In:  filename stem; directory.
    Out: deserialized model object.
    """
    model_path = os.path.join(model_dir, f"{name}.joblib")
    model = joblib.load(model_path)
    print(f"Loaded {name} from {model_path}")
    return model


def save_label_encoder(le, model_dir: str = MODEL_DIR) -> str:
    """Persist a LabelEncoder under the standard `label_encoder.joblib` name."""
    return save_sklearn_model(le, "label_encoder", model_dir=model_dir)


def load_label_encoder(model_dir: str = MODEL_DIR):
    """Load the persisted LabelEncoder (mirror of `save_label_encoder`)."""
    return load_sklearn_model("label_encoder", model_dir=model_dir)


def save_tfidf_vectorizer(vec, model_dir: str = MODEL_DIR) -> str:
    """Persist a TF-IDF vectorizer under `tfidf_vectorizer.joblib`."""
    return save_sklearn_model(vec, "tfidf_vectorizer", model_dir=model_dir)


def load_tfidf_vectorizer(model_dir: str = MODEL_DIR):
    """Load the persisted TF-IDF vectorizer (mirror of `save_tfidf_vectorizer`)."""
    return load_sklearn_model("tfidf_vectorizer", model_dir=model_dir)


def save_setfit_model(
    model,
    name: str = "setfit_best",
    model_dir: str = MODEL_DIR,
) -> str:
    """Save a SetFit model via its native `save_pretrained` API.

    In:  trained SetFitModel; directory stem; parent dir.
    Out: path to the directory the SetFit model was written into.
    """
    path = os.path.join(model_dir, name)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    print(f"SetFit model saved to {path}")
    return path


def load_setfit_model(name: str = "setfit_best", model_dir: str = MODEL_DIR):
    """Reload a SetFit model saved by `save_setfit_model`."""
    from setfit import SetFitModel

    path = os.path.join(model_dir, name)
    model = SetFitModel.from_pretrained(path)
    print(f"SetFit model loaded from {path}")
    return model


def save_model_registry(registry: Dict[str, Dict], model_dir: str = MODEL_DIR) -> None:
    """Write a JSON map of model_name → {path, metrics, ...} for run reproducibility.

    In:  dict keyed by model name; target dir.
    Out: writes `model_registry.json` (overwrites if present).
    """
    path = os.path.join(model_dir, "model_registry.json")
    with open(path, "w") as f:
        json.dump(registry, f, indent=2, default=str)
    print(f"Model registry saved to {path}")


def load_model_registry(model_dir: str = MODEL_DIR) -> Dict:
    """Load the model registry JSON, returning an empty dict if missing."""
    path = os.path.join(model_dir, "model_registry.json")
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def log_to_mlflow(
    model,
    model_name: str,
    metrics: Dict,
    params: Optional[Dict] = None,
    experiment_name: str = "article_classification",
) -> None:
    """Log a run to MLflow if the `mlflow` package is installed; otherwise no-op.

    In:  model + name + metrics dict; optional params; experiment name.
    Out: writes to the configured MLflow tracking URI (env var `MLFLOW_TRACKING_URI`).
    """
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        print("MLflow not installed — skipping. pip install mlflow")
        return

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=model_name):
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"MLflow: logged {model_name}")


def export_setfit_to_onnx(
    model_path: str,
    output_path: str = "models/artifacts/setfit_onnx",
) -> Optional[str]:
    """Export a SetFit model's sentence-transformer body to ONNX for fast inference.

    In:  saved SetFit dir; ONNX output dir.
    Out: output path on success, None when `optimum[onnxruntime]` is missing.
    """
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
    except ImportError:
        print(
            "optimum[onnxruntime] not installed. "
            "pip install optimum[onnxruntime]"
        )
        return None

    os.makedirs(output_path, exist_ok=True)
    ort_model = ORTModelForFeatureExtraction.from_pretrained(model_path, export=True)
    ort_model.save_pretrained(output_path)
    print(f"ONNX model exported to {output_path}")
    return output_path
