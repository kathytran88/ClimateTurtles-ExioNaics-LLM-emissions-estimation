"""
Embed the OOD dataset's `model_input` column with text-embedding-3-large (3072 dims)
and collapse the multi-range naics2_code values to a single code.

Code collapse: 31-33 -> 31, 44-45 -> 44, 48-49 -> 48.

Usage:
    export OPENAI_API_KEY=sk-...
    python embed_ood_large.py
"""

import time
import pandas as pd
from openai import OpenAI

# ── Config ──
INPUT_CSV = "../../Kathy/ood_dataset_official.csv"
OUTPUT_CSV = "../../.ipynb_checkpoints/ood_dataset_embeddings_large.csv"
MODEL = "text-embedding-3-large"
BATCH_SIZE = 256

NAICS2_CODE_FIX = {
    "31-33": "31",
    "44-45": "44",
    "48-49": "48",
}

client = OpenAI()

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")

# Normalize naics2_code (string to keep leading zeros / format consistent)
df["naics2_code"] = df["naics2_code"].astype(str).replace(NAICS2_CODE_FIX)
print(f"naics2_code unique after fix: {sorted(df['naics2_code'].unique().tolist())}")

texts = df["model_input"].astype(str).tolist()
print(f"Embedding {len(texts)} model_input texts...")


def embed_batched(texts):
    results = []
    total = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_times = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        start = time.time()
        response = client.embeddings.create(model=MODEL, input=batch)
        elapsed = time.time() - start
        batch_times.append(elapsed)

        results.extend([item.embedding for item in response.data])

        avg = sum(batch_times) / len(batch_times)
        remaining = (total - batch_num) * avg
        print(f"  Batch {batch_num}/{total}  ({elapsed:.1f}s)  ETA: {remaining:.0f}s", flush=True)
    return results


embeddings = embed_batched(texts)

df["embeddings"] = [str(e) for e in embeddings]

df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
print(f"Embedding dims: {len(embeddings[0])}")
print(f"Columns: {df.columns.tolist()}")
