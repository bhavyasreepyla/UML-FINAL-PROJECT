"""Longformer fine-tuning for sequence classification via HuggingFace Trainer.

Library-only. Provides checkpoint save-resume, TensorBoard + JSONL logging,
gradient accumulation for limited-memory GPUs, and a paired inference helper.

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
from typing import List, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from config import (
    CHECKPOINT_DIR,
    LOG_DIR,
    LONGFORMER_EVAL_BATCH_SIZE,
    LONGFORMER_EVAL_STEPS,
    LONGFORMER_FP16,
    LONGFORMER_GRADIENT_ACCUMULATION_STEPS,
    LONGFORMER_LEARNING_RATE,
    LONGFORMER_LOGGING_STEPS,
    LONGFORMER_MAX_LENGTH,
    LONGFORMER_MODEL_ID,
    LONGFORMER_NUM_EPOCHS,
    LONGFORMER_SAVE_STEPS,
    LONGFORMER_TRAIN_BATCH_SIZE,
    LONGFORMER_WARMUP_RATIO,
    LONGFORMER_WEIGHT_DECAY,
    MODEL_DIR,
    RANDOM_STATE,
)


class ArticleDataset(torch.utils.data.Dataset):
    """Tokenized HF dataset wrapper: per-example encoding dict + label tensor."""

    def __init__(self, encodings, labels):
        """In: tokenizer output (dict of tensors) + integer labels list."""
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        """Number of examples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Return one example: encoding fields + `labels` tensor at index `idx`."""
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    """HF Trainer metric callback: accuracy + macro/weighted F1.

    In:  EvalPrediction-like (logits, labels) tuple.
    Out: dict with keys 'accuracy', 'f1_macro', 'f1_weighted'.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def train_longformer(
    train_texts: List[str],
    train_labels: List[int],
    eval_texts: Optional[List[str]] = None,
    eval_labels: Optional[List[int]] = None,
    num_labels: int = 6,
    run_name: str = "longformer_clf",
    resume_from_checkpoint: bool = True,
):
    """Fine-tune Longformer for sequence classification with HF Trainer.

    In:  train + optional eval text/label lists; num_labels; run name; resume flag.
    Out: (model, tokenizer, eval metrics). Side effects: writes checkpoints,
         the best model dir, a JSONL log line, and TensorBoard events.
    """
    from transformers import (
        EarlyStoppingCallback,
        LongformerForSequenceClassification,
        LongformerTokenizer,
        Trainer,
        TrainingArguments,
    )

    output_dir = os.path.join(CHECKPOINT_DIR, run_name)
    best_model_dir = os.path.join(MODEL_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f"{run_name}.jsonl")

    tokenizer = LongformerTokenizer.from_pretrained(LONGFORMER_MODEL_ID)
    train_enc = tokenizer(
        train_texts, padding=True, truncation=True,
        max_length=LONGFORMER_MAX_LENGTH, return_tensors="pt",
    )
    train_dataset = ArticleDataset(train_enc, train_labels)

    eval_dataset = None
    if eval_texts is not None:
        eval_enc = tokenizer(
            eval_texts, padding=True, truncation=True,
            max_length=LONGFORMER_MAX_LENGTH, return_tensors="pt",
        )
        eval_dataset = ArticleDataset(eval_enc, eval_labels)

    model = LongformerForSequenceClassification.from_pretrained(
        LONGFORMER_MODEL_ID, num_labels=num_labels,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=LONGFORMER_NUM_EPOCHS,
        per_device_train_batch_size=LONGFORMER_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=LONGFORMER_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=LONGFORMER_GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LONGFORMER_LEARNING_RATE,
        warmup_ratio=LONGFORMER_WARMUP_RATIO,
        weight_decay=LONGFORMER_WEIGHT_DECAY,
        fp16=LONGFORMER_FP16 and torch.cuda.is_available(),
        save_strategy="steps",
        save_steps=LONGFORMER_SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=LONGFORMER_EVAL_STEPS if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="f1_macro" if eval_dataset else None,
        greater_is_better=True,
        logging_dir=os.path.join(LOG_DIR, "tensorboard", run_name),
        logging_steps=LONGFORMER_LOGGING_STEPS,
        report_to=["tensorboard"],
        seed=RANDOM_STATE,
        dataloader_num_workers=2,
    )

    callbacks = []
    if eval_dataset:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    checkpoint = None
    if resume_from_checkpoint and os.path.isdir(output_dir):
        checkpoints = [
            d for d in os.listdir(output_dir) if d.startswith("checkpoint-")
        ]
        if checkpoints:
            latest = sorted(checkpoints,
                            key=lambda x: int(x.split("-")[1]))[-1]
            checkpoint = os.path.join(output_dir, latest)
            print(f"Resuming from checkpoint: {checkpoint}")

    start = time.time()
    trainer.train(resume_from_checkpoint=checkpoint)
    elapsed = time.time() - start

    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    print(f"Best model saved to {best_model_dir}")

    metrics = {}
    if eval_dataset:
        metrics = trainer.evaluate()

    log_entry = {
        "run_name": run_name,
        "model_id": LONGFORMER_MODEL_ID,
        "num_epochs": LONGFORMER_NUM_EPOCHS,
        "train_size": len(train_labels),
        "elapsed_seconds": round(elapsed, 2),
        **{k: round(v, 4) if isinstance(v, float) else v
           for k, v in metrics.items()},
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return model, tokenizer, metrics


def predict_longformer(model_dir: str, texts: List[str],
                       max_length: int = LONGFORMER_MAX_LENGTH,
                       batch_size: int = LONGFORMER_EVAL_BATCH_SIZE):
    """Load a saved Longformer classifier and run batched inference.

    In:  best-model dir from `train_longformer`; texts; optional batch knobs.
    Out: (preds (N,), proba (N, C)) — proba is softmax over the logits.
    """
    from transformers import (
        LongformerForSequenceClassification,
        LongformerTokenizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = LongformerTokenizer.from_pretrained(model_dir)
    model = LongformerForSequenceClassification.from_pretrained(
        model_dir).to(device)
    model.eval()

    all_logits = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            outputs = model(**enc)
        all_logits.append(outputs.logits.cpu().numpy())

    logits = np.vstack(all_logits)
    proba = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(logits, axis=-1)
    return preds, proba
