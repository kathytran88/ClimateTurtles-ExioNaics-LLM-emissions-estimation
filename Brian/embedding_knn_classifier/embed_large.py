"""
Re-embed company data using text-embedding-3-large (3072 dims).
Produces three embedding columns:
  - embeddings:      full "Industry classification: name - description"
  - embeddings_name: company name only
  - embeddings_desc: description with company name stripped out

Usage:
    export OPENAI_API_KEY=sk-...
    python embed_large.py
"""

import re
import time
import pandas as pd
from openai import OpenAI

# ── Config ──
INPUT_CSV = "../../.ipynb_checkpoints/ExioNAICS_embeddings.csv"
OUTPUT_CSV = "../../.ipynb_checkpoints/ExioNAICS_embeddings_large.csv"
MODEL = "text-embedding-3-large"
BATCH_SIZE = 256

client = OpenAI()

# Load existing data
df = pd.read_csv(INPUT_CSV, index_col=0)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")


def strip_name_from_desc(name, desc):
    """Remove company name (and common variations) from description."""
    # Escape for regex, then replace with empty string (case-insensitive)
    pattern = re.escape(name)
    cleaned = re.sub(pattern, "", desc, flags=re.IGNORECASE).strip()
    # Clean up artifacts like leading commas, double spaces, parenthesized leftovers
    cleaned = re.sub(r"^\W+", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


# Build the three text lists
texts_full = []
texts_name = []
texts_desc = []

for name, desc in zip(df["Company Name"], df["Company Description"]):
    texts_full.append(f"Industry classification: {name} - {desc}")
    texts_name.append(str(name))
    texts_desc.append(strip_name_from_desc(str(name), str(desc)))

# All texts to embed in one flat list, then split results after
all_texts = texts_full + texts_name + texts_desc
n = len(df)
print(f"Embedding {len(all_texts)} texts ({n} x 3 columns)...")


def embed_batched(label, texts):
    """Embed a list of texts in batches, returns list of embedding lists."""
    results = []
    total = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    batch_times = []
    print(f"[{label}]")
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


emb_full = embed_batched("full", texts_full)
emb_name = embed_batched("name", texts_name)
emb_desc = embed_batched("desc", texts_desc)

# Save
df["embeddings"] = [str(e) for e in emb_full]
df["embeddings_name"] = [str(e) for e in emb_name]
df["embeddings_desc"] = [str(e) for e in emb_desc]

# Drop old embedding column if format differs
df.to_csv(OUTPUT_CSV)
print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
print(f"Embedding dims: {len(emb_full[0])}")
print(f"Columns: {df.columns.tolist()}")
