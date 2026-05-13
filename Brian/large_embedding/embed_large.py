import re
import time
import pandas as pd
from openai import OpenAI

INPUT_CSV = "../../.ipynb_checkpoints/ExioNAICS_embeddings.csv"
OUTPUT_CSV = "../../.ipynb_checkpoints/ExioNAICS_embeddings_large.csv"
MODEL = "text-embedding-3-large"
BATCH_SIZE = 256

client = OpenAI()

df = pd.read_csv(INPUT_CSV, index_col=0)
print(f"Loaded {len(df)} rows from {INPUT_CSV}")


def strip_name_from_desc(name, desc):
    cleaned = re.sub(re.escape(name), "", desc, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^\W+", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned


texts_full = []
texts_name = []
texts_desc = []

for name, desc in zip(df["Company Name"], df["Company Description"]):
    texts_full.append(f"Industry classification: {name} - {desc}")
    texts_name.append(str(name))
    texts_desc.append(strip_name_from_desc(str(name), str(desc)))

n = len(df)
print(f"Embedding {n * 3} texts ({n} x 3 columns)...")


def embed_batched(label, texts):
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

df["embeddings"] = [str(e) for e in emb_full]
df["embeddings_name"] = [str(e) for e in emb_name]
df["embeddings_desc"] = [str(e) for e in emb_desc]

df.to_csv(OUTPUT_CSV)
print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
print(f"Embedding dims: {len(emb_full[0])}")
print(f"Columns: {df.columns.tolist()}")
