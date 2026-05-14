"""PyTorch classifiers (FFNN + RNN ensemble) on top of dense embeddings.

Library-only. Two model families share one training loop and one inference
helper to keep training/inference DRY:
  - FFNNClassifier            : configurable MLP head
  - LSTM / GRU / BiLSTM       : reshape (N, D) → (N, seq_len, feat_dim) sequences
  - train_rnn_ensemble + majority_vote: train all three and combine via mode

Inputs:
    Dense embedding matrices (e.g. Gemma classification 768-d) plus integer
    labels. No file I/O on input.

Outputs:
    Best-by-val-accuracy checkpoints in `models/artifacts/<run_name>*.pth`
    when `save_path` is provided. Predict helpers return `(preds, proba)`.
"""

import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from config import (
    FFNN_BATCH_SIZE,
    FFNN_DROPOUT,
    FFNN_HIDDEN_DIMS,
    FFNN_LR,
    FFNN_NUM_EPOCHS,
    FFNN_WEIGHT_DECAY,
    MODEL_DIR,
    RNN_BATCH_SIZE,
    RNN_DROPOUT,
    RNN_FEAT_DIM,
    RNN_HIDDEN,
    RNN_LR,
    RNN_NUM_EPOCHS,
    RNN_NUM_LAYERS,
    RNN_SEQ_LEN,
)


def _resolve_device(device: Optional[str]) -> str:
    """Return the requested device or CUDA-if-available else CPU."""
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class FFNNClassifier(nn.Module):
    """Configurable feed-forward MLP: stacked Linear → ReLU → Dropout, logits head."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Iterable[int] = FFNN_HIDDEN_DIMS,
        dropout: float = FFNN_DROPOUT,
    ):
        """In: input dim, num classes, hidden-layer sizes, dropout. Out: ready model."""
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """In: (B, D) tensor. Out: (B, num_classes) logits."""
        return self.net(x)


class _RNNClassifier(nn.Module):
    """Shared scaffolding for LSTM / GRU / BiLSTM subclasses."""

    def __init__(
        self,
        rnn: nn.Module,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool,
    ):
        """Hold the underlying RNN + a logits head sized to the (bi)directional output."""
        super().__init__()
        self.rnn = rnn
        self.dropout = nn.Dropout(dropout)
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, num_classes)
        self._bidirectional = bidirectional

    def _pool_hidden(self, h: torch.Tensor) -> torch.Tensor:
        """Concat last forward/backward hidden states for bidirectional, else last only."""
        if self._bidirectional:
            return torch.cat((h[-2], h[-1]), dim=1)
        return h[-1]


class LSTMClassifier(_RNNClassifier):
    """Unidirectional multi-layer LSTM with a final linear logits head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = RNN_NUM_LAYERS,
        dropout: float = RNN_DROPOUT,
    ):
        """In: feat_dim per step, hidden size, num classes, depth, dropout."""
        rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        super().__init__(rnn, hidden_dim, num_classes, dropout, bidirectional=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """In: (B, seq_len, feat_dim). Out: (B, num_classes) logits."""
        _, (h, _) = self.rnn(x)
        return self.fc(self.dropout(self._pool_hidden(h)))


class GRUClassifier(_RNNClassifier):
    """Unidirectional multi-layer GRU with a final linear logits head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = RNN_NUM_LAYERS,
        dropout: float = RNN_DROPOUT,
    ):
        """In: feat_dim per step, hidden size, num classes, depth, dropout."""
        rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        super().__init__(rnn, hidden_dim, num_classes, dropout, bidirectional=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """In: (B, seq_len, feat_dim). Out: (B, num_classes) logits."""
        _, h = self.rnn(x)
        return self.fc(self.dropout(self._pool_hidden(h)))


class BiLSTMClassifier(_RNNClassifier):
    """Bidirectional multi-layer LSTM (concatenates forward + backward final hidden)."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = RNN_NUM_LAYERS,
        dropout: float = RNN_DROPOUT,
    ):
        """In: feat_dim per step, hidden size (per direction), classes, depth, dropout."""
        rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        super().__init__(rnn, hidden_dim, num_classes, dropout, bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """In: (B, seq_len, feat_dim). Out: (B, num_classes) logits."""
        _, (h, _) = self.rnn(x)
        return self.fc(self.dropout(self._pool_hidden(h)))


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap arrays into a TensorDataset DataLoader (float32 X, int64 y)."""
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def _class_weights(y: np.ndarray, device: str) -> torch.Tensor:
    """Compute sklearn-style balanced class weights on `device`."""
    weights = compute_class_weight("balanced", classes=np.unique(y), y=y)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _eval_accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Evaluate top-1 accuracy on a DataLoader. Returns 0 when loader is empty."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for bx, by in loader:
            preds = torch.argmax(model(bx.to(device)), dim=1).cpu()
            correct += (preds == by).sum().item()
            total += by.size(0)
    return correct / max(total, 1)


def train_torch_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    num_epochs: int = 30,
    batch_size: int = 64,
    use_class_weights: bool = True,
    device: Optional[str] = None,
    save_path: Optional[str] = None,
    log_every: int = 5,
    verbose: bool = True,
) -> Tuple[nn.Module, float]:
    """Generic training loop for any logits-returning torch classifier.

    In:  model + train arrays + optional val arrays + standard hyperparameters.
    Out: (trained model, best validation accuracy). When `save_path` is given
         and val data exists, the best-by-val checkpoint is written to disk.
    """
    device = _resolve_device(device)
    model = model.to(device)
    train_loader = _make_loader(X_train, y_train, batch_size, shuffle=True)
    val_loader = (
        _make_loader(X_val, y_val, batch_size * 2, shuffle=False)
        if X_val is not None and y_val is not None
        else None
    )

    weights = _class_weights(np.asarray(y_train), device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            acc = _eval_accuracy(model, val_loader, device)
            if acc > best_acc:
                best_acc = acc
                if save_path:
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    torch.save(model.state_dict(), save_path)
            if verbose and (epoch % log_every == 0 or epoch == num_epochs - 1):
                print(f"  epoch {epoch + 1}/{num_epochs}  val_acc={acc:.4f}")

    return model, best_acc


def predict_torch_classifier(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a torch classifier and return both preds and probabilities.

    In:  model + (N, D) features.
    Out: (preds (N,), proba (N, C)) — proba is softmax over the logits.
    """
    device = _resolve_device(device)
    model = model.to(device).eval()
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)

    chunks = []
    with torch.no_grad():
        for start in range(0, len(X_t), batch_size):
            logits = model(X_t[start : start + batch_size].to(device))
            chunks.append(logits.cpu().numpy())
    logits = np.vstack(chunks)
    proba = torch.softmax(torch.as_tensor(logits), dim=-1).numpy()
    return np.argmax(logits, axis=1), proba


def train_ffnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    *,
    num_classes: int,
    run_name: str = "ffnn",
    hidden_dims: Iterable[int] = FFNN_HIDDEN_DIMS,
    dropout: float = FFNN_DROPOUT,
    lr: float = FFNN_LR,
    weight_decay: float = FFNN_WEIGHT_DECAY,
    num_epochs: int = FFNN_NUM_EPOCHS,
    batch_size: int = FFNN_BATCH_SIZE,
    use_class_weights: bool = True,
    device: Optional[str] = None,
) -> Tuple[FFNNClassifier, float]:
    """Construct an FFNN sized to the input embedding dim and train it.

    In:  train + optional val arrays; num_classes; standard hyperparameters.
    Out: (trained FFNN, best val accuracy); best checkpoint at
         `models/artifacts/<run_name>.pth`.
    """
    input_dim = int(np.asarray(X_train).shape[1])
    model = FFNNClassifier(input_dim, num_classes, hidden_dims=hidden_dims, dropout=dropout)
    save_path = os.path.join(MODEL_DIR, f"{run_name}.pth")
    return train_torch_classifier(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        batch_size=batch_size,
        use_class_weights=use_class_weights,
        device=device,
        save_path=save_path,
    )


_RNN_FACTORIES: Dict[str, type] = {
    "LSTM": LSTMClassifier,
    "GRU": GRUClassifier,
    "BiLSTM": BiLSTMClassifier,
}


def _reshape_for_rnn(X: np.ndarray, seq_len: int, feat_dim: int) -> np.ndarray:
    """Reshape (N, D) embeddings into (N, seq_len, feat_dim) pseudo-sequences."""
    X = np.asarray(X)
    if X.shape[1] != seq_len * feat_dim:
        raise ValueError(
            f"Embedding dim {X.shape[1]} != seq_len({seq_len}) * feat_dim({feat_dim})"
        )
    return X.reshape(-1, seq_len, feat_dim)


def train_rnn_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    *,
    num_classes: int,
    seq_len: int = RNN_SEQ_LEN,
    feat_dim: int = RNN_FEAT_DIM,
    hidden_dim: int = RNN_HIDDEN,
    num_layers: int = RNN_NUM_LAYERS,
    dropout: float = RNN_DROPOUT,
    lr: float = RNN_LR,
    num_epochs: int = RNN_NUM_EPOCHS,
    batch_size: int = RNN_BATCH_SIZE,
    use_class_weights: bool = True,
    device: Optional[str] = None,
    run_name: str = "rnn_ensemble",
) -> Tuple[Dict[str, nn.Module], Dict[str, float]]:
    """Train LSTM + GRU + BiLSTM on the same data and report per-model val accuracy.

    In:  train + optional val arrays; standard hyperparameters; seq_len * feat_dim
         must equal embedding dim.
    Out: ({name: trained_model}, {name: best_val_acc}); per-model checkpoints
         at `models/artifacts/<run_name>_<lower>.pth`.
    """
    X_tr = _reshape_for_rnn(X_train, seq_len, feat_dim)
    X_va = _reshape_for_rnn(X_val, seq_len, feat_dim) if X_val is not None else None

    models: Dict[str, nn.Module] = {}
    accs: Dict[str, float] = {}
    for name, cls in _RNN_FACTORIES.items():
        print(f"\ntraining {name}...")
        model = cls(
            feat_dim, hidden_dim, num_classes, num_layers=num_layers, dropout=dropout
        )
        save_path = os.path.join(MODEL_DIR, f"{run_name}_{name.lower()}.pth")
        models[name], accs[name] = train_torch_classifier(
            model,
            X_tr,
            y_train,
            X_va,
            y_val,
            lr=lr,
            weight_decay=0.0,
            num_epochs=num_epochs,
            batch_size=batch_size,
            use_class_weights=use_class_weights,
            device=device,
            save_path=save_path,
        )
    return models, accs


def majority_vote(predictions: List[np.ndarray]) -> np.ndarray:
    """Combine multiple prediction arrays via mode (ties → smallest label).

    In:  list of (N,) prediction arrays from different models.
    Out: (N,) ensemble prediction.
    """
    stacked = np.stack(predictions, axis=1)
    try:
        from scipy import stats

        return stats.mode(stacked, axis=1, keepdims=False)[0]
    except ImportError:
        from collections import Counter

        return np.asarray([Counter(row).most_common(1)[0][0] for row in stacked])


def predict_rnn_ensemble(
    models: Dict[str, nn.Module],
    X: np.ndarray,
    *,
    seq_len: int = RNN_SEQ_LEN,
    feat_dim: int = RNN_FEAT_DIM,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Run each ensemble member on X and combine.

    In:  dict of trained {name: model}; (N, D) features.
    Out: (ensemble_preds, {name: proba (N, C)}, mean_proba (N, C)).
    """
    X_seq = _reshape_for_rnn(X, seq_len, feat_dim)
    per_model_preds: Dict[str, np.ndarray] = {}
    per_model_proba: Dict[str, np.ndarray] = {}
    for name, model in models.items():
        preds, proba = predict_torch_classifier(
            model, X_seq, batch_size=batch_size, device=device
        )
        per_model_preds[name] = preds
        per_model_proba[name] = proba
    ensemble_preds = majority_vote(list(per_model_preds.values()))
    mean_proba = np.mean(list(per_model_proba.values()), axis=0)
    return ensemble_preds, per_model_proba, mean_proba
