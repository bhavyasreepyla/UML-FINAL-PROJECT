"""SetFit few-shot / full training with checkpoint save-resume and JSONL logging.

Library-only.

Inputs:
    Training text + label arrays (and optional eval splits). Reads from
    `models/checkpoints/<run_name>/` when resuming an interrupted run.

Outputs:
    `models/checkpoints/<run_name>/`     — per-epoch checkpoints
    `logs/<run_name>.jsonl`              — per-run metrics line
    Returned (model, metrics) for downstream evaluation / prediction.
"""

import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    RANDOM_STATE,
    SETFIT_BASE_MODEL,
    SETFIT_BATCH_SIZE_FEW,
    SETFIT_FEW_SHOT_PER_CLASS,
    SETFIT_NUM_EPOCHS,
    SETFIT_NUM_ITERATIONS,
)


def sample_few_shot(
    train_df: pd.DataFrame,
    label_col: str = "label",
    shots_per_class: int = SETFIT_FEW_SHOT_PER_CLASS,
) -> pd.DataFrame:
    """Stratified subsample of up to N examples per label for few-shot training.

    In:  training DataFrame with a label column; N per class.
    Out: shuffled subsample DataFrame with reset index.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    indices = []
    for label in sorted(train_df[label_col].unique()):
        class_idx = train_df[train_df[label_col] == label].index.tolist()
        n = min(shots_per_class, len(class_idx))
        indices.extend(rng.choice(class_idx, n, replace=False))
    return train_df.loc[indices].reset_index(drop=True)


def train_setfit(
    train_texts,
    train_labels,
    eval_texts=None,
    eval_labels=None,
    base_model: str = SETFIT_BASE_MODEL,
    batch_size: int = SETFIT_BATCH_SIZE_FEW,
    num_epochs: int = SETFIT_NUM_EPOCHS,
    num_iterations: int = SETFIT_NUM_ITERATIONS,
    run_name: str = "setfit",
    output_dir: Optional[str] = None,
):
    """Fine-tune a SetFit model, resuming from `output_dir` if a checkpoint exists.

    In:  train texts + labels; optional eval splits; base model id; hyperparams.
    Out: (trained SetFitModel, eval metrics dict); appends a JSONL log entry.
    """
    from setfit import SetFitModel, Trainer, TrainingArguments
    from datasets import Dataset

    if output_dir is None:
        output_dir = os.path.join(CHECKPOINT_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f"{run_name}.jsonl")

    train_ds = Dataset.from_dict({"text": list(train_texts),
                                  "label": list(train_labels)})
    eval_ds = None
    if eval_texts is not None and eval_labels is not None:
        eval_ds = Dataset.from_dict({"text": list(eval_texts),
                                     "label": list(eval_labels)})

    # Resume from `output_dir` if it already contains a SetFit head.
    checkpoint_marker = os.path.join(output_dir, "model_head.pkl")
    if os.path.isfile(checkpoint_marker):
        print(f"Resuming from checkpoint: {output_dir}")
        model = SetFitModel.from_pretrained(output_dir)
    else:
        print(f"Starting fresh from {base_model}")
        model = SetFitModel.from_pretrained(base_model)

    args = TrainingArguments(
        output_dir=output_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        num_iterations=num_iterations,
        seed=RANDOM_STATE,
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True if eval_ds else False,
        eval_strategy="epoch" if eval_ds else "no",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    start = time.time()
    trainer.train()
    elapsed = time.time() - start

    metrics = {}
    if eval_ds:
        metrics = trainer.evaluate(eval_ds)

    log_entry = {
        "run_name": run_name,
        "base_model": base_model,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "num_iterations": num_iterations,
        "train_size": len(train_texts),
        "elapsed_seconds": round(elapsed, 2),
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in metrics.items()},
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"Training log appended to {log_path}")

    return model, metrics


def predict_setfit(model, texts) -> np.ndarray:
    """Run inference with a trained SetFit model.

    In:  fitted SetFitModel; iterable of texts.
    Out: (N,) ndarray of predicted class indices.
    """
    preds = model.predict(list(texts))
    return np.array(preds)
