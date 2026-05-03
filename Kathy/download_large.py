from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/deberta-v3-large",
    cache_dir="./hf_cache_large"
)

print("Done! Files are in ./hf_cache_large/")