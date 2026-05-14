"""Sklearn-compatible classifiers: registry, training, evaluation, comparison.

Library-only. Covers the full classical-supervised surface used in the project:
  - Calibrated linear models: LR, LinearSVC (calibrated)
  - Tree ensembles:           GBM, RandomForest
  - Generative / baselines:   GaussianNB, Dummy(most_frequent)
  - Optional GBDTs:           CatBoost, XGBoost, LightGBM (fail soft on import)

Inputs:
    Feature matrices (TF-IDF or dense embeddings) and target labels passed in
    from `scripts/run_supervised.py` or notebooks. No file I/O.

Outputs:
    Fitted estimators and pandas results DataFrames (acc / F1-macro / F1-weighted).
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import (
    CATBOOST_PARAMS,
    CV_N_SPLITS,
    GBM_LEARNING_RATE,
    GBM_MAX_DEPTH,
    GBM_N_ESTIMATORS,
    LIGHTGBM_PARAMS,
    LR_C,
    LR_MAX_ITER,
    RANDOM_STATE,
    RF_N_ESTIMATORS,
    SVC_MAX_ITER,
    TFIDF_CLF_MAX_FEATURES,
    TFIDF_CLF_MIN_DF,
    XGBOOST_PARAMS,
)


def _balanced_kwargs() -> Dict[str, Any]:
    """Shared `class_weight='balanced'` + seed kwargs for sklearn classifiers."""
    return dict(class_weight="balanced", random_state=RANDOM_STATE)


def _make_lr():
    """Construct a class-balanced LogisticRegression with project defaults."""
    return LogisticRegression(max_iter=LR_MAX_ITER, C=LR_C, **_balanced_kwargs())


def _make_svc():
    """Construct a CalibratedClassifierCV(LinearSVC) so we get `predict_proba`."""
    return CalibratedClassifierCV(
        LinearSVC(C=LR_C, max_iter=SVC_MAX_ITER, **_balanced_kwargs()), cv=5
    )


def _make_gbm():
    """Construct sklearn's GradientBoostingClassifier with project defaults."""
    return GradientBoostingClassifier(
        n_estimators=GBM_N_ESTIMATORS,
        max_depth=GBM_MAX_DEPTH,
        learning_rate=GBM_LEARNING_RATE,
        random_state=RANDOM_STATE,
    )


def _make_rf():
    """Construct a class-balanced RandomForestClassifier."""
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=None, **_balanced_kwargs()
    )


def _make_nb():
    """Construct GaussianNB — only valid on DENSE inputs (e.g. embeddings)."""
    return GaussianNB()


def _make_dummy():
    """Construct the majority-class baseline (`strategy='most_frequent'`)."""
    return DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)


def _make_catboost():
    """Construct CatBoostClassifier from CATBOOST_PARAMS (raises if not installed)."""
    from catboost import CatBoostClassifier

    return CatBoostClassifier(**CATBOOST_PARAMS)


def _make_xgboost():
    """Construct XGBClassifier from XGBOOST_PARAMS (raises if not installed)."""
    import xgboost as xgb

    return xgb.XGBClassifier(**XGBOOST_PARAMS)


def _make_lightgbm():
    """Construct LGBMClassifier from LIGHTGBM_PARAMS (raises if not installed)."""
    import lightgbm as lgb

    return lgb.LGBMClassifier(**LIGHTGBM_PARAMS)


# Always-available models vs optional GBDTs (gated on package import).
_CORE_FACTORIES: Dict[str, Callable[[], Any]] = {
    "LR": _make_lr,
    "SVC": _make_svc,
    "GBM": _make_gbm,
    "RF": _make_rf,
    "NB": _make_nb,
    "Dummy": _make_dummy,
}

_OPTIONAL_FACTORIES: Dict[str, Callable[[], Any]] = {
    "CatBoost": _make_catboost,
    "XGBoost": _make_xgboost,
    "LightGBM": _make_lightgbm,
}


def get_model_registry(include_optional: bool = True) -> Dict[str, Any]:
    """Return mapping of model_name → fresh estimator instance.

    In:  `include_optional` toggles CatBoost/XGBoost/LightGBM (skipped if uninstalled).
    Out: dict of name → instantiated sklearn-compatible estimator.
    """
    registry = {name: factory() for name, factory in _CORE_FACTORIES.items()}
    if include_optional:
        for name, factory in _OPTIONAL_FACTORIES.items():
            try:
                registry[name] = factory()
            except ImportError:
                continue
    return registry


def available_optional_models() -> List[str]:
    """Return the optional models whose package is currently importable.

    In:  none.
    Out: list of names (subset of CatBoost/XGBoost/LightGBM).
    """
    available = []
    for name, factory in _OPTIONAL_FACTORIES.items():
        try:
            factory()
            available.append(name)
        except ImportError:
            continue
    return available


def train_and_evaluate(model, X_train, X_test, y_train, y_test) -> Tuple[Dict[str, float], np.ndarray]:
    """Fit a model and score it on the held-out set.

    In:  estimator; train/test feature matrices; train/test label vectors.
    Out: (metrics dict with acc/f1_macro/f1_weighted, y_pred ndarray).
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "acc": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }
    return metrics, y_pred


def evaluate_predictions(y_test, y_pred, label_names) -> Tuple[str, np.ndarray]:
    """Build a classification report + confusion matrix from predictions.

    In:  true labels; predicted labels; class-name list.
    Out: (report text, confusion-matrix ndarray).
    """
    return (
        classification_report(y_test, y_pred, target_names=label_names),
        confusion_matrix(y_test, y_pred),
    )


def compare_models(
    model_configs: List[Tuple[str, Any, np.ndarray, np.ndarray]],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[Any, np.ndarray]]]:
    """Train each (name, model, X_train, X_test) tuple and rank by macro-F1.

    In:  list of model configs; shared train/test label vectors.
    Out: (results_df sorted by f1_macro, dict of name → (fitted_model, X_test)).
    """
    results, trained = {}, {}
    for name, model, X_tr, X_te in model_configs:
        metrics, _ = train_and_evaluate(model, X_tr, X_te, y_train, y_test)
        results[name] = metrics
        trained[name] = (model, X_te)
        print(
            f"  {name}: acc={metrics['acc']:.4f}  "
            f"F1m={metrics['f1_macro']:.4f}  F1w={metrics['f1_weighted']:.4f}"
        )
    results_df = pd.DataFrame(results).T.sort_values("f1_macro", ascending=False)
    return results_df, trained


def _tfidf_vectorizer() -> TfidfVectorizer:
    """Construct the project's TF-IDF vectorizer with consistent settings."""
    return TfidfVectorizer(
        max_features=TFIDF_CLF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words="english",
        min_df=TFIDF_CLF_MIN_DF,
    )


def build_tfidf_features(X_train_text, X_test_text) -> Tuple[TfidfVectorizer, Any, Any]:
    """Fit TF-IDF on training text and transform both train and test splits.

    In:  training and test text arrays (already preprocessed).
    Out: (vectorizer, X_train_tfidf sparse matrix, X_test_tfidf sparse matrix).
    """
    vec = _tfidf_vectorizer()
    return vec, vec.fit_transform(X_train_text), vec.transform(X_test_text)


def cross_validate_pipeline(name, pipeline, X, y, n_splits: int = CV_N_SPLITS) -> np.ndarray:
    """Run stratified K-fold CV on a pipeline and print the macro-F1 summary.

    In:  display name; sklearn Pipeline; features X; labels y; fold count.
    Out: array of per-fold macro-F1 scores.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    print(f"{name}: F1-macro = {scores.mean():.4f} (+/- {scores.std():.4f})")
    return scores


def _build_tfidf_pipeline(clf_factory: Callable[[], Any]) -> Pipeline:
    """Helper: TF-IDF + arbitrary classifier as a single Pipeline."""
    return Pipeline([("tfidf", _tfidf_vectorizer()), ("clf", clf_factory())])


def build_tfidf_lr_pipeline() -> Pipeline:
    """TF-IDF → LogisticRegression Pipeline (used for cross-validation)."""
    return _build_tfidf_pipeline(_make_lr)


def build_tfidf_svc_pipeline() -> Pipeline:
    """TF-IDF → CalibratedSVC Pipeline (used for cross-validation)."""
    return _build_tfidf_pipeline(_make_svc)
