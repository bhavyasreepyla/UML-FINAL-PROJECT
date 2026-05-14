"""RoBERTa fine-tuning for sequence classification via HuggingFace Trainer.

Library-only. Supports class-weighted cross-entropy, optional
RandomOverSampler-based balancing, checkpoint save-resume, TensorBoard +
JSONL logging, and dynamic padding (DataCollatorWithPadding).

Inputs:
    Train + optional eval text/label lists. Reads from
    `models/checkpoints/<run_name>/` when resuming.

Outputs:
    `models/checkpoints/<run_name>/`     — Trainer checkpoints
    `models/artifacts/<run_name>/`       — best model + tokenizer
    `logs/<run_name>.jsonl`              — per-run metrics
    `logs/tensorboard/<run_name>/`       — TensorBoard event files
"""

import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    ROBERTA_EVAL_BATCH_SIZE,
    ROBERTA_FP16,
    ROBERTA_GRADIENT_ACCUMULATION_STEPS,
    ROBERTA_LEARNING_RATE,
    ROBERTA_LOGGING_STEPS,
    ROBERTA_LR_SCHEDULER,
    ROBERTA_MAX_LENGTH,
    ROBERTA_MODEL_ID,
    ROBERTA_NUM_EPOCHS,
    ROBERTA_SAVE_TOTAL_LIMIT,
    ROBERTA_TRAIN_BATCH_SIZE,
    ROBERTA_WARMUP_STEPS,
    ROBERTA_WEIGHT_DECAY,
)


def _compute_metrics(eval_pred):
    """HF Trainer metric callback: accuracy + macro/weighted F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def _build_weighted_trainer(class_weights: torch.Tensor):
    """Build a Trainer subclass that uses weighted CE with `class_weights`."""
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.CrossEntropyLoss(weight=class_weights)(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def _oversample(texts: List[str], labels: List[int]) -> Tuple[List[str], List[int]]:
    """Balance class counts via RandomOverSampler (requires imbalanced-learn)."""
    from imblearn.over_sampling import RandomOverSampler

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(
        np.asarray(texts).reshape(-1, 1), np.asarray(labels)
    )
    return X_res.flatten().tolist(), y_res.tolist()


def _resume_checkpoint(output_dir: str) -> Optional[str]:
    """Return path to the latest `checkpoint-N` subdir if one exists, else None."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(output_dir, latest)


def train_roberta(
    train_texts: List[str],
    train_labels: List[int],
    eval_texts: Optional[List[str]] = None,
    eval_labels: Optional[List[int]] = None,
    *,
    num_labels: int = 6,
    run_name: str = "roberta_clf",
    model_id: str = ROBERTA_MODEL_ID,
    use_class_weights: bool = True,
    oversample: bool = False,
    early_stopping_patience: Optional[int] = None,
    resume_from_checkpoint: bool = True,
):
    """Fine-tune RoBERTa for sequence classification.

    In:  train + optional eval text/label lists; num_labels; model id;
         class-weight / oversampling toggles; early-stopping patience.
    Out: (model, tokenizer, eval metrics). Writes checkpoints, the best model
         dir, a JSONL log line, and TensorBoard events.
    """
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    output_dir = os.path.join(CHECKPOINT_DIR, run_name)
    best_model_dir = os.path.join(MODEL_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{run_name}.jsonl")

    if oversample:
        before = len(train_texts)
        train_texts, train_labels = _oversample(train_texts, train_labels)
        print(f"  Oversampled training set: {before} → {len(train_texts)}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=ROBERTA_MAX_LENGTH
        )

    train_ds = Dataset.from_dict(
        {"text": list(train_texts), "label": list(map(int, train_labels))}
    )
    train_ds = train_ds.map(_tokenize, batched=True, remove_columns=["text"])

    eval_ds = None
    if eval_texts is not None and eval_labels is not None:
        eval_ds = Dataset.from_dict(
            {"text": list(eval_texts), "label": list(map(int, eval_labels))}
        )
        eval_ds = eval_ds.map(_tokenize, batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels
    )

    trainer_cls = Trainer
    if use_class_weights:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = compute_class_weight(
            "balanced", classes=np.unique(train_labels), y=np.asarray(train_labels)
        )
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        trainer_cls = _build_weighted_trainer(class_weights)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=ROBERTA_NUM_EPOCHS,
        per_device_train_batch_size=ROBERTA_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=ROBERTA_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=ROBERTA_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=ROBERTA_LEARNING_RATE,
        weight_decay=ROBERTA_WEIGHT_DECAY,
        warmup_steps=ROBERTA_WARMUP_STEPS,
        lr_scheduler_type=ROBERTA_LR_SCHEDULER,
        save_strategy="epoch",
        save_total_limit=ROBERTA_SAVE_TOTAL_LIMIT,
        eval_strategy="epoch" if eval_ds is not None else "no",
        load_best_model_at_end=eval_ds is not None,
        metric_for_best_model="f1_macro" if eval_ds is not None else None,
        greater_is_better=True,
        logging_dir=os.path.join(LOG_DIR, "tensorboard", run_name),
        logging_steps=ROBERTA_LOGGING_STEPS,
        report_to=["tensorboard"],
        fp16=ROBERTA_FP16 and torch.cuda.is_available(),
        seed=RANDOM_STATE,
        dataloader_num_workers=2,
    )

    callbacks = []
    if eval_ds is not None and early_stopping_patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=_compute_metrics if eval_ds is not None else None,
        data_collator=DataCollatorWithPadding(tokenizer),
        callbacks=callbacks,
    )

    checkpoint = _resume_checkpoint(output_dir) if resume_from_checkpoint else None
    if checkpoint:
        print(f"Resuming from checkpoint: {checkpoint}")

    start = time.time()
    trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.time() - start

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Best model saved to {best_model_dir}")

    metrics = trainer.evaluate() if eval_ds is not None else {}

    log_entry = {
        "run_name": run_name,
        "model_id": model_id,
        "num_epochs": ROBERTA_NUM_EPOCHS,
        "train_size": len(train_labels),
        "use_class_weights": use_class_weights,
        "oversample": oversample,
        "elapsed_seconds": round(elapsed, 2),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return model, tokenizer, metrics


def predict_roberta(
    model_dir: str,
    texts: List[str],
    *,
    max_length: int = ROBERTA_MAX_LENGTH,
    batch_size: int = ROBERTA_EVAL_BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a saved RoBERTa classifier and run batched inference.

    In:  best-model dir from `train_roberta`; texts; optional batch knobs.
    Out: (preds (N,), proba (N, C)) — proba is softmax over the logits.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    all_logits = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**enc)
        all_logits.append(outputs.logits.cpu().numpy())

    logits = np.vstack(all_logits)
    proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    return preds, proba
