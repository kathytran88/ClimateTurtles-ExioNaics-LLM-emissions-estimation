"""
SBERT + classifier head for NAICS sector prediction.

NOTICE - If you don't set Batch_size, things will crash!!!

Pipeline
--------
1. Load ExioNAICS.csv
2. Extract company descriptions and NAICS labels (2-, 3-, or 6-digit)
3. Embed descriptions with a frozen SentenceTransformer
4. Train a logistic regression or MLP head
5. Evaluate: accuracy, macro-F1, weighted-F1, classification report, confusion matrix
6. Save predictions to CSV

Usage
-----
    python sbert_logistic_classifier.py                            # logistic, MiniLM, NAICS2
    python sbert_logistic_classifier.py --head mlp
    python sbert_logistic_classifier.py --naics_level 3
    python sbert_logistic_classifier.py --backbone sentence-transformers/all-mpnet-base-v2
    python sbert_logistic_classifier.py --head mlp --hidden_dims 512 256 128

its advised to use --batch_size 32 and --cache_embeddings

current best performing

python sbert_logistic_classifier.py --backbone BAAI/bge-large-en-v1.5 --head mlp --hidden_dims 512 256 128 --batch_size 32 --cache_embeddings

============================================================
  EXTENDED METRICS
============================================================
  Top-1 accuracy : 0.6052
  Top-3 accuracy : 0.8252
  Top-5 accuracy : 0.9021
  Hier acc @2    : 0.6052  (2-digit prefix match)
  Hier acc @3    : 0.6052  (3-digit prefix match)

python sbert_logistic_classifier.py --head mlp --hidden_dims 512 256 128 --batch_size 32 --cache_embeddings

============================================================
  EXTENDED METRICS
============================================================
  Top-1 accuracy : 0.5852
  Top-3 accuracy : 0.8067
  Top-5 accuracy : 0.8914
  Hier acc @2    : 0.5852  (2-digit prefix match)
  Hier acc @3    : 0.5852  (3-digit prefix match)

python sbert_logistic_classifier.py --head mlp --hidden_dims 1024 512 256 --batch_size 32 --cache_embeddings

  EXTENDED METRICS
============================================================
  Top-1 accuracy : 0.5813
  Top-3 accuracy : 0.8150
  Top-5 accuracy : 0.8914
  Hier acc @2    : 0.5813  (2-digit prefix match)
  Hier acc @3    : 0.5813  (3-digit prefix match)

python sbert_logistic_classifier.py --head torch_mlp --hidden_dims 512 256 128 --backbone BAAI/bge-large-en-v1.5 --batch_size 32 --cache_embeddings

============================================================
  EXTENDED METRICS
============================================================
  Top-1 accuracy : 0.4971
  Top-3 accuracy : 0.7736
  Top-5 accuracy : 0.8710
  Hier acc @2    : 0.4971  (2-digit prefix match)
  Hier acc @3    : 0.4971  (3-digit prefix match)
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Extended metrics (top-k, hierarchical accuracy)
from naics_metrics import compute_naics_metrics, print_metrics as print_naics_metrics

# pretty-print the confusion matrix
try:
    import matplotlib
    matplotlib.use("Agg") # non-interactive backend (safe on headless servers)
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# ── PyTorch MLP (frozen-backbone mode) ───────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: tuple, num_classes: int, dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TorchMLPClassifier:
    """sklearn-compatible PyTorch MLP trained on pre-computed embeddings."""

    def __init__(self, in_dim, num_classes, hidden_dims=(256, 128),
                 batch_size=64, epochs=50, lr=1e-3, dropout=0.3, seed=0):
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        torch.manual_seed(self.seed)
        self.model = _MLP(self.in_dim, self.hidden_dims, self.num_classes, self.dropout).to(self.device)
        loader = DataLoader(
            TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)),
            batch_size=self.batch_size, shuffle=True,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(xb), yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        loader = DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32)), batch_size=self.batch_size)
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                probs.append(torch.softmax(self.model(xb.to(self.device)), dim=-1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ── PyTorch fine-tuning (end-to-end backbone + MLP) ──────────────────────────

class _SBERTWithMLP(nn.Module):
    def __init__(self, backbone_name, hidden_dims, num_classes, freeze_layers=0, dropout=0.3):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(backbone_name)
        if freeze_layers > 0:
            for param in self.encoder.embeddings.parameters():
                param.requires_grad = False
            for layer in self.encoder.encoder.layer[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False
        self.head = _MLP(self.encoder.config.hidden_size, hidden_dims, num_classes, dropout)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = nn.functional.normalize(pooled, dim=-1)
        return self.head(pooled)


class TorchFinetuneClassifier:
    """sklearn-compatible end-to-end fine-tuning wrapper; accepts raw texts."""

    def __init__(self, backbone_name, num_classes, hidden_dims=(256, 128),
                 batch_size=16, epochs=10, lr=2e-5, dropout=0.3,
                 freeze_layers=0, max_length=128, seed=0):
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.freeze_layers = freeze_layers
        self.max_length = max_length
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None

    def _tokenize(self, texts):
        from transformers import AutoTokenizer
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)
        return self.tokenizer(texts, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt")

    def fit(self, texts, y: np.ndarray):
        torch.manual_seed(self.seed)
        self.model = _SBERTWithMLP(
            self.backbone_name, self.hidden_dims, self.num_classes,
            self.freeze_layers, self.dropout,
        ).to(self.device)
        enc = self._tokenize(texts)
        loader = DataLoader(
            TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(y, dtype=torch.long)),
            batch_size=self.batch_size, shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr, weight_decay=1e-2,
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.1, total_iters=len(loader) * self.epochs,
        )
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for ids, mask, yb in loader:
                ids, mask, yb = ids.to(self.device), mask.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(ids, mask), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            print(f"  epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.4f}")
        return self

    def predict_proba(self, texts) -> np.ndarray:
        self.model.eval()
        enc = self._tokenize(texts)
        loader = DataLoader(
            TensorDataset(enc["input_ids"], enc["attention_mask"]),
            batch_size=self.batch_size,
        )
        probs = []
        with torch.no_grad():
            for ids, mask in loader:
                logits = self.model(ids.to(self.device), mask.to(self.device))
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict(self, texts) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=1)


# CLI arguments

def parse_args():
    parser = argparse.ArgumentParser(description="SBERT + Logistic Regression NAICS classifier")
    parser.add_argument(
        "--data_path",
        default="data/ExioNAICS_combined.csv",
        help="Path to ExioNAICS_combined.csv (relative to this script)",
    )
    parser.add_argument(
        "--naics_level",
        type=int,
        choices=[2, 3, 6],
        default=2,
        help="NAICS granularity to predict: 2, 3, or 6 digits",
    )
    parser.add_argument(
        "--backbone",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name or local path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Encoding batch size (reduce if you hit OOM)",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Max iterations for logistic regression solver",
    )
    parser.add_argument(
        "--head",
        choices=["logistic", "mlp", "torch_mlp"],
        default="logistic",
        help="Classifier head: logistic regression, sklearn MLP, or PyTorch MLP",
    )
    parser.add_argument(
        "--finetune",
        action="store_true",
        help="Fine-tune the SBERT backbone end-to-end (only for torch_mlp)",
    )
    parser.add_argument(
        "--freeze_layers",
        type=int,
        default=0,
        help="Number of transformer encoder layers to freeze during fine-tuning (torch_mlp --finetune only)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for torch_mlp (frozen: 50 default; finetune: use 10)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Max token length for tokenizer (torch_mlp --finetune only)",
    )
    parser.add_argument(
        "--hidden_dims",
        nargs="+",
        type=int,
        default=[256, 128],
        help="MLP hidden layer sizes, e.g. --hidden_dims 512 256 128 (ignored for logistic)",
    )
    parser.add_argument(
        "--mlp_batch_size",
        type=int,
        default=64,
        help="Mini-batch size for MLP training (ignored for logistic). Reduce to lower memory usage.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=None,
        help="Logistic regression inverse regularisation strength. Omit to auto-tune (ignored for mlp)",
    )
    parser.add_argument(
        "--cache_embeddings",
        action="store_true",
        help="Cache embeddings to disk so re-runs skip encoding",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to write predictions CSV and plots",
    )
    parser.add_argument(
        "--ood_path",
        default=None,
        help="Path to OOD CSV (ood_dataset_official.csv) for out-of-distribution evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=192,
    )
    return parser.parse_args()


# Data loading and label preparation

NAICS_LEVEL_COL = {
    2: "NAICS_2 Code",
    3: "NAICS_3 Code",
    6: "NAICS Code", # full 6-digit code already in the dataset
}

def load_data(data_path: str, naics_level: int) -> pd.DataFrame:
    """Load CSV and return a clean dataframe with 'text' and 'label' columns."""
    df = pd.read_csv(data_path, dtype=str, low_memory=False)

    text_col  = "Company Description"
    label_col = NAICS_LEVEL_COL[naics_level]

    # Keep text, label, company name, and NAICS_2 Title (used for stratification)
    extra_cols = [] if label_col == "NAICS_2 Title" else ["NAICS_2 Title"]
    df = df[[text_col, label_col, "Company Name"] + extra_cols].copy()
    df = df.rename(columns={text_col: "text", label_col: "label", "Company Name": "company_name"})

    # Drop rows missing either text or label
    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df["text"]  = df["text"].str.strip()
    df["label"] = df["label"].str.strip()
    df = df[df["text"] != ""]
    df = df[df["label"] != ""]
    print(f"Loaded {len(df):,} rows  (dropped {before - len(df):,} with missing text/label)")

    # For NAICS6 derive the 2-digit prefix anyway (useful for diagnostics)
    if naics_level == 6:
        # Normalise: zero-pad to 6 digits if numeric
        df["label"] = df["label"].str.zfill(6)
    else:
        # Pre-computed columns are already the right length, but ensure str
        df["label"] = df["label"].astype(str)

    # Drop classes with too few samples to survive a stratified 80/10/10 split.
    counts = df["label"].value_counts()
    rare = counts[counts < 3].index
    if len(rare):
        df = df[~df["label"].isin(rare)]
        print(f"Dropped {len(rare)} class(es) with <3 samples: {sorted(rare.tolist())}")

    print(f"Unique NAICS{naics_level} labels: {df['label'].nunique()}")
    return df.reset_index(drop=True)


# Train / val / test split (stratified)

def stratified_split(df: pd.DataFrame, seed: int):
    """
    80/10/10 stratified split using NAICS_2 Title to preserve class distribution.

    Stage 1: split full df → train+val (90%) / test (10%)
    Stage 2: split train+val → train (80% of full) / val (10% of full)
             using test_size=1/9 so that val ends up as 1/9 of 90% = 10% of total.

    Returns (train_df, val_df, test_df).
    """
    df = df.drop(columns=["NAICS2"], errors="ignore")

    train_andval, test_df = train_test_split(
        df,
        test_size=0.1,
        stratify=df["NAICS_2 Title"],
        random_state=seed,
    )
    train_df, val_df = train_test_split(
        train_andval,
        test_size=1 / 9,
        stratify=train_andval["NAICS_2 Title"],
        random_state=seed,
    )

    print(
        f"Split sizes — train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}"
    )

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        dist = split["NAICS_2 Title"].value_counts(normalize=True).sort_index()
        print(f"\n{name} class distribution (NAICS_2 Title):")
        for cls, frac in dist.items():
            print(f"  {cls:<55} {frac:.4f}")

    return train_df, val_df, test_df


# SBERT embedding

def embed_texts(
    texts: list[str],
    backbone: str,
    batch_size: int,
    cache_path: str | None = None,
) -> np.ndarray:
    """
    Return (N, D) float32 array of L2-normalised SBERT embeddings.
    If cache_path is given, load from disk when present, else encode and save.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    print(f"Encoding {len(texts):,} texts with '{backbone}'  (batch_size={batch_size})")
    t0 = time.time()
    model = SentenceTransformer(backbone)
    # normalize_embeddings=True → unit-norm vectors; cosine sim == dot product
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"Encoding done in {time.time() - t0:.1f}s  shape={embeddings.shape}")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"Embeddings cached to {cache_path}")

    return embeddings


# Classifier head training

def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    head: str,
    max_iter: int,
    seed: int,
    C: float = 1.0,
    hidden_dims: tuple = (256, 128),
    mlp_batch_size: int = 64,
    epochs: int = 50,
):
    t0 = time.time()
    if head == "logistic":
        print(f"Training logistic regression  (C={C}, max_iter={max_iter}) ...")
        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
            verbose=0,
        )
        clf.fit(X_train, y_train)
    elif head == "mlp":
        print(f"Training sklearn MLP  (hidden_dims={hidden_dims}, max_iter={max_iter}) ...")
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_dims),
            activation="relu",
            max_iter=max_iter,
            batch_size=mlp_batch_size,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
            verbose=False,
        )
        clf.fit(X_train, y_train)
    else:  # torch_mlp
        num_classes = len(np.unique(y_train))
        in_dim = X_train.shape[1]
        print(f"Training PyTorch MLP  (hidden_dims={hidden_dims}, epochs={epochs}, device={'cuda' if torch.cuda.is_available() else 'cpu'}) ...")
        clf = TorchMLPClassifier(
            in_dim=in_dim,
            num_classes=num_classes,
            hidden_dims=tuple(hidden_dims),
            batch_size=mlp_batch_size,
            epochs=epochs,
            seed=seed,
        )
        clf.fit(X_train, y_train)
    print(f"Training done in {time.time() - t0:.1f}s")
    return clf


# C hyper-parameter search (validation-set macro-F1)

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def tune_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_iter: int,
    seed: int,
) -> float:
    """
    Train one logistic regression per C value, score each on validation
    macro-F1, and return the best C.  Final model is trained separately
    so the validation set is never used for fitting.
    """
    print(f"\nTuning C over {C_GRID} (scored by validation macro-F1) ...")
    results = []
    for C in C_GRID:
        clf = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
            verbose=0,
        )
        clf.fit(X_train, y_train)
        score = f1_score(y_val, clf.predict(X_val), average="macro", zero_division=0)
        results.append((C, score))
        print(f"  C={C:<8}  val macro-F1={score:.4f}")

    best_C, best_score = max(results, key=lambda x: x[1])
    print(f"\n→ Best C = {best_C}  (val macro-F1 = {best_score:.4f})\n")
    return best_C


# Evaluation

def evaluate(
    clf,
    X: np.ndarray,
    y_true: np.ndarray,
    label_encoder: LabelEncoder,
    split_name: str,
    output_dir: str,
    naics_level: int,
) -> dict:
    """Compute and print all metrics; optionally save a confusion-matrix plot."""
    y_pred     = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)  # (N, num_classes)

    acc        = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    class_names = label_encoder.classes_

    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SET — NAICS{naics_level} classification")
    print(f"{'='*60}")
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  Macro-F1      : {macro_f1:.4f}   ← primary metric (imbalanced)")
    print(f"  Weighted-F1   : {weighted_f1:.4f}")
    print()
    # Only report classes that actually appear in this split
    present_labels = sorted(set(y_true) | set(y_pred))
    present_names  = label_encoder.inverse_transform(present_labels)
    print(classification_report(y_true, y_pred, labels=present_labels, target_names=present_names, zero_division=0))

    # Confusion matrix (only practical for NAICS2/3; skip for NAICS6)
    if naics_level <= 3 and PLOT_AVAILABLE:
        cm = confusion_matrix(y_true, y_pred, labels=label_encoder.transform(class_names))
        fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cmap="Blues",
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{split_name} confusion matrix — NAICS{naics_level}")
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"confusion_matrix_{split_name}_naics{naics_level}.png")
        fig.savefig(plot_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"Confusion matrix saved → {plot_path}")

    return {
        "split": split_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
    }

# Save predictions

def save_predictions(
    df_subset: pd.DataFrame,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    label_encoder: LabelEncoder,
    output_path: str,
):
    """
    Write a CSV with one row per sample:
      company_name | text | true_label | pred_label | confidence (max class prob)
    """
    pred_labels = label_encoder.inverse_transform(y_pred)
    confidence  = y_pred_proba.max(axis=1)

    out = df_subset[["company_name", "text", "label"]].copy().reset_index(drop=True)
    out["pred_label"]  = pred_labels
    out["confidence"]  = confidence.round(4)
    out["correct"]     = (out["label"] == out["pred_label"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Predictions saved → {output_path}  ({len(out):,} rows)")

# Main

def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Resolve paths relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(script_dir, args.data_path)
    output_dir = os.path.join(script_dir, args.output_dir)

    # Load and clean data
    df = load_data(data_path, args.naics_level)

    # Integer-encode labels (sklearn classifiers need numeric targets)
    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    # Stratified split (80/10/10) using NAICS_2 Title
    train_df, val_df, test_df = stratified_split(df, args.seed)
    train_idx = train_df.index.to_numpy()
    val_idx   = val_df.index.to_numpy()
    test_idx  = test_df.index.to_numpy()

    y_train = df["label_enc"].values[train_idx]
    y_val   = df["label_enc"].values[val_idx]
    y_test  = df["label_enc"].values[test_idx]

    is_finetune = args.head == "torch_mlp" and args.finetune

    if is_finetune:
        # End-to-end fine-tuning: skip embedding, work with raw texts
        print(f"\nFine-tuning backbone '{args.backbone}' + PyTorch MLP head ...")
        clf = TorchFinetuneClassifier(
            backbone_name=args.backbone,
            num_classes=num_classes,
            hidden_dims=tuple(args.hidden_dims),
            batch_size=args.mlp_batch_size,
            epochs=args.epochs,
            freeze_layers=args.freeze_layers,
            max_length=args.max_length,
            seed=args.seed,
        )
        clf.fit(train_df["text"].tolist(), y_train)
        X_val_eval  = val_df["text"].tolist()
        X_test_eval = test_df["text"].tolist()
    else:
        # Frozen-backbone path: embed once, then train classifier head
        cache_path = None
        if args.cache_embeddings:
            backbone_slug = args.backbone.replace("/", "_")
            cache_path = os.path.join(output_dir, f"embeddings_{backbone_slug}.npy")

        all_embeddings = embed_texts(
            df["text"].tolist(),
            backbone=args.backbone,
            batch_size=args.batch_size,
            cache_path=cache_path,
        )

        X_train = all_embeddings[train_idx]
        X_val   = all_embeddings[val_idx]
        X_test  = all_embeddings[test_idx]

        if args.head == "logistic":
            if args.C is None:
                best_C = tune_C(X_train, y_train, X_val, y_val, args.max_iter, args.seed)
            else:
                best_C = args.C
                print(f"Skipping C search — using fixed C={best_C}")
        else:
            best_C = 1.0

        clf = train_classifier(
            X_train, y_train,
            head=args.head,
            max_iter=args.max_iter,
            seed=args.seed,
            C=best_C,
            hidden_dims=args.hidden_dims,
            mlp_batch_size=args.mlp_batch_size,
            epochs=args.epochs,
        )
        X_val_eval  = X_val
        X_test_eval = X_test

    # Evaluate on validation set
    val_results = evaluate(
        clf, X_val_eval, y_val, le,
        split_name="validation",
        output_dir=output_dir,
        naics_level=args.naics_level,
    )

    # Evaluate on held-out test set (report this number)
    test_results = evaluate(
        clf, X_test_eval, y_test, le,
        split_name="test",
        output_dir=output_dir,
        naics_level=args.naics_level,
    )

    # Save test-set predictions
    pred_csv = os.path.join(
        output_dir, f"predictions_naics{args.naics_level}_test.csv"
    )
    save_predictions(
        test_df,
        test_results["y_pred"],
        test_results["y_pred_proba"],
        le,
        pred_csv,
    )

    # Extended metrics: top-k and hierarchical accuracy (test set)
    # y_true must be the original NAICS label strings (not encoded ints) so
    # that prefix truncation ([:2], [:3]) works correctly on the code digits.
    ext_metrics = compute_naics_metrics(
        y_true       = test_df["label"].tolist(),
        y_scores     = test_results["y_pred_proba"],
        class_labels = le.classes_.tolist(),
    )
    print_naics_metrics(ext_metrics)

    # Summary table
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    for r in [val_results, test_results]:
        print(
            f"  {r['split']:10s}  acc={r['accuracy']:.4f}  "
            f"macro-F1={r['macro_f1']:.4f}  weighted-F1={r['weighted_f1']:.4f}"
        )
    print()

    # ── OOD evaluation ────────────────────────────────────────────────────────
    if args.ood_path:
        ood_path = os.path.join(script_dir, args.ood_path)
        ood_df = pd.read_csv(ood_path, dtype=str).dropna(subset=["model_input", "naics2_code"])
        ood_df["model_input"] = ood_df["model_input"].str.strip()
        ood_df["naics2_code"] = ood_df["naics2_code"].str.strip()

        n_total = len(ood_df)
        ood_df  = ood_df[ood_df["naics2_code"].isin(set(le.classes_))].reset_index(drop=True)
        print(f"\nOOD rows: {n_total:,} total → {len(ood_df):,} with a known NAICS-2 class")

        y_ood_true = le.transform(ood_df["naics2_code"].tolist())

        if is_finetune:
            X_ood_eval = ood_df["model_input"].tolist()
        else:
            X_ood_eval = embed_texts(ood_df["model_input"].tolist(), backbone=args.backbone, batch_size=args.batch_size)

        ood_results = evaluate(clf, X_ood_eval, y_ood_true, le,
                               split_name="ood", output_dir=output_dir, naics_level=args.naics_level)

        ext_ood = compute_naics_metrics(
            y_true       = ood_df["naics2_code"].tolist(),
            y_scores     = ood_results["y_pred_proba"],
            class_labels = le.classes_.tolist(),
        )
        print_naics_metrics(ext_ood)

        ood_pred_csv = os.path.join(output_dir, f"predictions_naics{args.naics_level}_ood.csv")
        ood_save_df  = ood_df.rename(columns={"company name": "company_name", "model_input": "text", "naics2_code": "label"})
        save_predictions(ood_save_df, ood_results["y_pred"], ood_results["y_pred_proba"], le, ood_pred_csv)

    # ── Save model artifacts for OOD evaluation ───────────────────────────────
    model_dir = os.path.join(output_dir, "saved_model")
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(le, os.path.join(model_dir, "label_encoder.joblib"))

    config = {
        "backbone":    args.backbone,
        "naics_level": args.naics_level,
        "hidden_dims": args.hidden_dims,
        "head":        args.head,
        "finetune":    is_finetune,
        "max_length":  args.max_length,
        "num_classes": num_classes,
    }
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if is_finetune:
        torch.save(clf.model.state_dict(), os.path.join(model_dir, "model_state.pt"))
        clf.tokenizer.save_pretrained(model_dir)
    else:
        joblib.dump(clf, os.path.join(model_dir, "classifier.joblib"))

    print(f"Model artifacts saved → {model_dir}")


if __name__ == "__main__":
    main()
