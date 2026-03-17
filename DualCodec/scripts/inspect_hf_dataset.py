"""Inspect a HuggingFace Parquet dataset schema without using the datasets library."""
import sys
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

ds_name = sys.argv[1] if len(sys.argv) > 1 else "Pranavz/hinglish-casual"
print(f"Downloading first shard metadata from: {ds_name} ...")

# Download just the first parquet shard
local_path = hf_hub_download(
    repo_id=ds_name,
    filename="data/train-00000-of-00128.parquet",
    repo_type="dataset",
)

# Read schema + first row
pf = pq.ParquetFile(local_path)
print(f"\n=== Schema ({pf.metadata.num_rows} rows in this shard) ===")
print(pf.schema_arrow)

print("\n=== First row ===")
table = pf.read_row_group(0, columns=None)
row = table.slice(0, 1).to_pydict()
for k, v_list in row.items():
    v = v_list[0]
    if isinstance(v, dict):
        print(f"  {k}: dict with keys {list(v.keys())}")
        for k2, v2 in v.items():
            if hasattr(v2, '__len__') and len(v2) > 10:
                print(f"    {k2}: type={type(v2).__name__}, len={len(v2)}")
            else:
                print(f"    {k2}: {v2}")
    elif isinstance(v, bytes):
        print(f"  {k}: bytes, len={len(v)}")
    elif isinstance(v, str) and len(v) > 200:
        print(f"  {k}: str len={len(v)}, preview='{v[:100]}...'")
    else:
        print(f"  {k}: {v}")
