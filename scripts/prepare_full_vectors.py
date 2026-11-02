#!/usr/bin/env python3
"""
scripts/prepare_full_vectors.py

Full-corpus MVE1 generator for the Week 10 demonstration.

This script runs the entire CMU Movie Summary Corpus through the exact same
encoder, chunking, and mean-pooling policy as the 16-movie sample generator.

Outputs:
  1. data/full_movies.bin          - The binary MVE1 corpus (32-byte aligned rows)
  2. data/reports/rejections.csv   - Every rejected record and the reason why
  3. data/reports/manifest.json    - Provenance, counts, and SHA-256 integrity hashes

Usage:
  python scripts/prepare_full_vectors.py
"""

import csv
import hashlib
import json
import struct
import sys
import time
from pathlib import Path
from operator import itemgetter
try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Run: pip install -r scripts/requirements.txt", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
try:
    from prepare_sample_vectors import (
        encode_plot,
        TARGET_DIM,
        MAX_TOKENS,
        LONG_PLOT_POLICY,
        write_mve1_header,
        write_metadata_record,
    )
except ImportError:
    print("Warning: Could not import helper functions from prepare_sample_vectors.py.")
    print("Falling back to internal default contract implementations.")
    
    from sentence_transformers import SentenceTransformer
    
    # Fallback defaults
    TARGET_DIM = 384
    MAX_TOKENS = 256
    LONG_PLOT_POLICY = "chunk_mean_pool"
    _model = None

    def encode_plot(plot_text: str) -> np.ndarray:
        global _model
        if _model is None:
            _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # Basic chunking fallback
        tokens = plot_text.split()
        chunks = [" ".join(tokens[i:i + MAX_TOKENS]) for i in range(0, len(tokens), MAX_TOKENS)]
        embeddings = _model.encode(chunks)
        mean_emb = np.mean(embeddings, axis=0)
        # L2 Normalize
        norm = np.linalg.norm(mean_emb)
        if norm > 1e-12:
            mean_emb = mean_emb / norm
        return mean_emb.astype(np.float32)

    def write_mve1_header(f, num_movies: int, dim: int):
        magic = b"MVE1"
        f.write(magic)
        f.write(struct.pack("<II", num_movies, dim))

    def write_metadata_record(f, wiki_id: int, title: str):
        title_bytes = title.encode("utf-8")[:127]
        f.write(struct.pack("<Q", wiki_id))
        f.write(title_bytes.ljust(128, b"\x00"))


RAW_PLOTS    = Path("data/raw/MovieSummaries/plot_summaries.txt")
RAW_METADATA = Path("data/raw/MovieSummaries/movie.metadata.tsv")
OUT_BIN      = Path("data/full_movies.bin")
OUT_REJ      = Path("data/reports/rejections.csv")
OUT_MANIFEST = Path("data/reports/manifest.json")


def sha256_file(path: Path) -> str:
    """Calculates the SHA-256 hash of a file for provenance tracking."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata() -> dict:
    """Parses movie.metadata.tsv to map wiki_id -> title."""
    meta = {}
    with RAW_METADATA.open(encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3:
                wiki_id = parts[0].strip()
                title = parts[2].strip()
                meta[wiki_id] = title
    return meta


def main():
    if not RAW_PLOTS.exists() or not RAW_METADATA.exists():
        print(f"Error: Raw data not found at {RAW_PLOTS.parent}.", file=sys.stderr)
        print("Run the Dataset Download steps from README.md first.", file=sys.stderr)
        sys.exit(1)

    OUT_BIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_REJ.parent.mkdir(parents=True, exist_ok=True)

    print("Loading metadata...")
    meta = load_metadata()
    
    seen_ids = set()
    accepted = []
    rejections = []

    print(f"Processing plots from {RAW_PLOTS}...")
    
    # Read all lines to get an accurate raw count for the manifest
    with RAW_PLOTS.open(encoding="utf-8") as f:
        lines = f.readlines()

    for line_no, line in enumerate(lines, 1):
        if line_no % 10000 == 0:
            print(f"  Processed {line_no}/{len(lines)} lines...")
            
        parts = line.rstrip("\n").split("\t")
        
        # 1. Malformed line check
        if len(parts) != 2:
            rejections.append(("", "", "malformed_line"))
            continue
            
        wiki_id, plot = parts[0].strip(), parts[1].strip()

        # 2. Duplicate ID policy (First occurrence wins)
        if wiki_id in seen_ids:
            rejections.append((wiki_id, "", "duplicate_id"))
            continue
        seen_ids.add(wiki_id)

        # 3. Metadata matching
        if wiki_id not in meta:
            rejections.append((wiki_id, "", "no_metadata_match"))
            continue
        
        title = meta[wiki_id]
        if not title:
            rejections.append((wiki_id, "", "empty_title"))
            continue

        # 4. Plot cleaning
        if not plot:
            rejections.append((wiki_id, title, "empty_plot"))
            continue

        # 5. Encode
        try:
            vec = encode_plot(plot)
        except Exception as e:
            rejections.append((wiki_id, title, f"encode_error:{type(e).__name__}"))
            continue

        # 6. Numeric validation
        if vec.shape != (TARGET_DIM,):
            rejections.append((wiki_id, title, "wrong_dim"))
            continue
        if not np.all(np.isfinite(vec)):
            rejections.append((wiki_id, title, "non_finite"))
            continue
        if np.linalg.norm(vec) < 1e-6:
            rejections.append((wiki_id, title, "zero_magnitude"))
            continue

        accepted.append((int(wiki_id), title, vec))

    print(f"\nWriting {len(accepted)} accepted records to {OUT_BIN}...")
    
    # Write binary in MVE1 format
    with OUT_BIN.open("wb") as f:
        write_mve1_header(f, len(accepted), TARGET_DIM)
        for wiki_id, title, _ in accepted:
            write_metadata_record(f, wiki_id, title)
        for _, _, vec in accepted:
            # Ensure little-endian float32
            f.write(vec.astype("<f4").tobytes())

    # Write rejections CSV
    print(f"Writing {len(rejections)} rejected records to {OUT_REJ}...")
    with OUT_REJ.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wiki_id", "title", "reason"])
        writer.writerows(rejections)

    # Calculate rejection stats
    reasons = {}
    for _, _, r in rejections:
        reasons[r] = reasons.get(r, 0) + 1

    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {
            "plots":    {"path": str(RAW_PLOTS),    "sha256": sha256_file(RAW_PLOTS)},
            "metadata": {"path": str(RAW_METADATA), "sha256": sha256_file(RAW_METADATA)},
        },
        "output": {
            "path":       str(OUT_BIN),
            "sha256":     sha256_file(OUT_BIN),
            "num_movies": len(accepted),
            "vector_dim": TARGET_DIM,
        },
        "contract": {
            "magic":          "MVE1",
            "default_dim":    TARGET_DIM,
            "title_max_len":  128,
            "encoder":        "sentence-transformers/all-MiniLM-L6-v2",
            "max_tokens":     MAX_TOKENS,
            "long_plot":      LONG_PLOT_POLICY,
        },
        "counts": {
            "raw_lines":   len(lines),
            "accepted":    len(accepted),
            "rejected":    len(rejections),
            "by_reason":   reasons,
        },
    }
    
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


    print("\n" + "="*50)
    print(" PIPELINE COMPLETE")
    print("="*50)
    print(f" Raw records   : {len(lines)}")
    print(f" Accepted      : {len(accepted)}")
    print(f" Rejected      : {len(rejections)}")
    print("-" * 50)
    print(" Rejections by reason:")
    for reason, count in sorted(reasons.items(), key=itemgetter(1), reverse=True):
        print(f"   - {reason:<20}: {count}")
    print("="*50)
    print(f" Binary Output : {OUT_BIN} ({OUT_BIN.stat().st_size / (1024*1024):.2f} MB)")
    print(f" Manifest      : {OUT_MANIFEST}")
    print(f" Rejections    : {OUT_REJ}")
    print("="*50)

if __name__ == "__main__":
    main()