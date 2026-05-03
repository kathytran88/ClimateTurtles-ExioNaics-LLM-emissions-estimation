"""
Fine-tuned transformer classifier for NAICS sector prediction.
Uses AutoModelForSequenceClassification for end-to-end training.

Usage
-----
    python bert_finetuned_classifier.py                              # bert-base-uncased, NAICS2
    python bert_finetuned_classifier.py --backbone roberta-base
    python bert_finetuned_classifier.py --backbone distilbert-base-uncased
    python bert_finetuned_classifier.py --naics_level 3
    python bert_finetuned_classifier.py --backbone roberta-base --naics_level 3 --epochs 8
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from naics_metrics import compute_naics_metrics, print_metrics as print_naics_metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tuned transformer NAICS classifier")
    p.add_argument("--data_path", default="data/ExioNAICS.csv")
    p.add_argument("--naics_level", type=int, choices=[2, 3, 6], default=2)
    p.add_argument(
        "--backbone",
        default="bert-base-uncased",
        help="HuggingFace model ID, e.g. roberta-base, distilbert-base-uncased",
    )
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Linear warmup steps (0 = no warmup)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--seed", type=int, default=192)
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

NAICS_LEVEL_COL = {2: "NAICS_2 Code", 3: "NAICS_3 Code", 6: "NAICS Code"}


def load_data(data_path: str, naics_level: int) -> pd.DataFrame:
    df = pd.read_csv(data_path, dtype=str, low_memory=False)
    text_col  = "Company Description"
    label_col = NAICS_LEVEL_COL[naics_level]
    extra_cols = [] if label_col == "NAICS_2 Title" else ["NAICS_2 Title"]
    df = df[[text_col, label_col, "Company Name"] + extra_cols].copy()
    df = df.rename(columns={text_col: "text", label_col: "label", "Company Name": "company_name"})

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df["text"]  = df["text"].str.strip()
    df["label"] = df["label"].str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    print(f"Loaded {len(df):,} rows  (dropped {before - len(df):,} with missing text/label)")

    df["label"] = df["label"].str.zfill(6) if naics_level == 6 else df["label"].astype(str)

    counts = df["label"].value_counts()
    rare = counts[counts < 3].index
    if len(rare):
        df = df[~df["label"].isin(rare)]
        print(f"Dropped {len(rare)} rare class(es)")
    print(f"Unique NAICS{naics_level} labels: {df['label'].nunique()}")
    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, seed: int):
    """80/10/10 split stratified on NAICS_2 Title."""
    df = df.drop(columns=["NAICS2"], errors="ignore")
    train_andval, test_df = train_test_split(
        df, test_size=0.1, stratify=df["NAICS_2 Title"], random_state=seed,
    )
    train_df, val_df = train_test_split(
        train_andval, test_size=1 / 9, stratify=train_andval["NAICS_2 Title"], random_state=seed,
    )
    print(f"Split sizes — train: {len(train_df):,} | val: {len(val_df):,} | test: {len(test_df):,}")
    return train_df, val_df, test_df


# ── Dataset ───────────────────────────────────────────────────────────────────

class NAICSDataset(Dataset):
    def __init__(self, encodings, labels: list[int]):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ── Metrics for Trainer ───────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":     float(accuracy_score(labels, preds)),
        "macro_f1":     float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "weighted_f1":  float(f1_score(labels, preds, average="weighted", zero_division=0)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path  = os.path.join(script_dir, args.data_path)
    backbone_slug = args.backbone.replace("/", "_")
    run_dir = os.path.join(
        script_dir, args.output_dir,
        f"finetune_{backbone_slug}_naics{args.naics_level}",
    )

    df = load_data(data_path, args.naics_level)

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    num_classes = len(le.classes_)
    print(f"Number of classes: {num_classes}")

    train_df, val_df, test_df = stratified_split(df, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    def tokenize(texts: list[str]):
        return tokenizer(texts, truncation=True, max_length=args.max_length)

    train_dataset = NAICSDataset(tokenize(train_df["text"].tolist()), train_df["label_enc"].tolist())
    val_dataset   = NAICSDataset(tokenize(val_df["text"].tolist()),   val_df["label_enc"].tolist())
    test_dataset  = NAICSDataset(tokenize(test_df["text"].tolist()),  test_df["label_enc"].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        args.backbone,
        num_labels=num_classes,
        id2label={i: lbl for i, lbl in enumerate(le.classes_)},
        label2id={lbl: i for i, lbl in enumerate(le.classes_)},
    )

    training_args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=args.seed,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # ── Test-set evaluation ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  TEST SET — NAICS{args.naics_level} classification ({args.backbone})")
    print(f"{'='*60}")

    test_output  = trainer.predict(test_dataset)
    y_pred       = np.argmax(test_output.predictions, axis=-1)
    y_true       = test_df["label_enc"].values
    y_pred_proba = torch.softmax(torch.tensor(test_output.predictions, dtype=torch.float32), dim=-1).numpy()

    acc         = accuracy_score(y_true, y_pred)
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Macro-F1    : {macro_f1:.4f}   ← primary metric")
    print(f"  Weighted-F1 : {weighted_f1:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))

    ext_metrics = compute_naics_metrics(
        y_true=test_df["label"].tolist(),
        y_scores=y_pred_proba,
        class_labels=le.classes_.tolist(),
    )
    print_naics_metrics(ext_metrics)

    # ── Save predictions ──────────────────────────────────────────────────────
    pred_path = os.path.join(run_dir, f"predictions_naics{args.naics_level}_test.csv")
    out = test_df[["company_name", "text", "label"]].copy().reset_index(drop=True)
    out["pred_label"] = le.inverse_transform(y_pred)
    out["confidence"] = y_pred_proba.max(axis=1).round(4)
    out["correct"]    = out["label"] == out["pred_label"]
    os.makedirs(run_dir, exist_ok=True)
    out.to_csv(pred_path, index=False)
    print(f"Predictions saved → {pred_path}  ({len(out):,} rows)")


if __name__ == "__main__":
    main()
