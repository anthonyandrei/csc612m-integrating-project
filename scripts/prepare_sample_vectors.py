#!/usr/bin/env python3
"""
Generate a reproducible sample binary movie vector dataset following the MVE1
input contract specification for Issue #2.

Features:
- Selects 16 real movies from the CMU Movie Summary Corpus (with matched metadata).
- Tests both single-chunk (< 256 tokens) and multi-chunk (> 256 tokens) plots.
- Applies the approved Chunking + Mean-Pooling long-plot policy.
- Encodes with sentence-transformers/all-MiniLM-L6-v2 (dim=384).
- Serializes to data/sample_movies.bin matching include/movie_contract.h.
"""

import os
import struct
import numpy as np
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# Model & Contract Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384
MAX_CHUNK_TOKENS = 256
MAX_TITLE_LEN = 128
MAGIC_BYTES = b"MVE1"

OUTPUT_DIR = "data"
OUTPUT_BIN = os.path.join(OUTPUT_DIR, "sample_movies.bin")
RAW_PLOTS_PATH = "data/raw/MovieSummaries/plot_summaries.txt"
RAW_META_PATH = "data/raw/MovieSummaries/movie.metadata.tsv"

# Curated list of 16 real movies covering diverse genres and plot lengths
SAMPLE_MOVIES = [
    # Multi-chunk plots (> 256 tokens)
    (30007, "The Matrix"),
    (23270459, "Inception"),
    (52371, "Titanic"),
    (4273140, "Avatar"),
    (54173, "Pulp Fiction"),
    (1009041, "Fight Club"),
    (41528, "Forrest Gump"),
    (2466773, "The Godfather"),
    (4276475, "The Dark Knight"),
    (68485, "Jurassic Park"),
    (3616797, "Gladiator"),
    (30625, "The Shawshank Redemption"),
    (65834, "Schindler's List"),
    (64394, "Goodfellas"),
    # Single-chunk plots (< 256 tokens)
    (171005, "Henry V"),
    (26067101, "Siam Sunset")
]

def load_sample_plots():
    """Extract plot summaries for the selected sample movies."""
    target_ids = {m[0] for m in SAMPLE_MOVIES}
    plots = {}
    with open(RAW_PLOTS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                wiki_id = int(parts[0])
                if wiki_id in target_ids:
                    plots[wiki_id] = parts[1]
    return plots

def encode_with_chunking(text, tokenizer, model):
    """
    Encodes plot text using Chunking with Mean Pooling:
    - If tokens <= 256: encodes directly in a single pass.
    - If tokens > 256: splits into 256-token non-overlapping windows,
      encodes each window, and takes the element-wise mean vector.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(tokens) <= MAX_CHUNK_TOKENS:
        vec = model.encode(text, normalize_embeddings=False)
        return vec.astype(np.float32), 1
    
    # Split into chunks of MAX_CHUNK_TOKENS
    chunks = []
    for i in range(0, len(tokens), MAX_CHUNK_TOKENS):
        chunk_token_ids = tokens[i : i + MAX_CHUNK_TOKENS]
        chunk_text = tokenizer.decode(chunk_token_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text)
            
    chunk_embeddings = model.encode(chunks, normalize_embeddings=False)
    mean_vec = np.mean(chunk_embeddings, axis=0)
    return mean_vec.astype(np.float32), len(chunks)

def write_mve1_binary(filepath, movies_data, vector_dim=VECTOR_DIM):
    """
    Serializes movie records to binary matching include/movie_contract.h:
    - Header: magic[4] ('MVE1'), num_movies (uint32), vector_dim (uint32)
    - Metadata: array of { wiki_id (uint64), title (char[128]) }
    - Vectors: contiguous array of float32 values (num_movies * vector_dim)
    """
    num_movies = len(movies_data)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, "wb") as f:
        # Header (12 bytes)
        f.write(struct.pack("<4sII", MAGIC_BYTES, num_movies, vector_dim))
        
        # Metadata Records (num_movies * 136 bytes)
        for wiki_id, title, _ in movies_data:
            title_bytes = title.encode("utf-8")[: MAX_TITLE_LEN - 1]
            padded_title = title_bytes.ljust(MAX_TITLE_LEN, b"\x00")
            f.write(struct.pack("<Q", wiki_id))
            f.write(padded_title)
            
        # Contiguous Vector Buffer (num_movies * vector_dim * 4 bytes)
        for _, _, vec in movies_data:
            assert len(vec) == vector_dim, f"Vector dimension mismatch: {len(vec)} != {vector_dim}"
            f.write(vec.astype(np.float32).tobytes())
            
    expected_size = 12 + (num_movies * 136) + (num_movies * vector_dim * 4)
    actual_size = os.path.getsize(filepath)
    assert actual_size == expected_size, f"File size mismatch: {actual_size} != {expected_size}"
    return actual_size

def main():
    print(f"Loading tokenizer & model: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    
    print(f"Loading plots for {len(SAMPLE_MOVIES)} sample movies...")
    plots = load_sample_plots()
    assert len(plots) == len(SAMPLE_MOVIES), f"Missing plots! Found {len(plots)}/{len(SAMPLE_MOVIES)}"
    
    movies_data = []
    print("\nEncoding sample movies:")
    print(f"{'ID':<10} | {'Title':<26} | {'Tokens':<7} | {'Chunks':<6} | {'Vector Norm':<11}")
    print("-" * 70)
    
    for wiki_id, title in SAMPLE_MOVIES:
        plot_text = plots[wiki_id]
        total_tokens = len(tokenizer.encode(plot_text, add_special_tokens=False, truncation=False))
        vec, num_chunks = encode_with_chunking(plot_text, tokenizer, model)
        norm = np.linalg.norm(vec)
        
        # Verify finite values & non-zero magnitude
        assert np.all(np.isfinite(vec)), f"Non-finite value in vector for {title}"
        assert norm > 1e-6, f"Zero magnitude vector for {title}"
        
        movies_data.append((wiki_id, title, vec))
        print(f"{wiki_id:<10} | {title:<26} | {total_tokens:<7} | {num_chunks:<6} | {norm:.4f}")
        
    print("\nWriting MVE1 binary contract dataset...")
    size = write_mve1_binary(OUTPUT_BIN, movies_data)
    print(f"Successfully generated {OUTPUT_BIN}")
    print(f"Total Movies: {len(movies_data)}")
    print(f"Vector Dim:   {VECTOR_DIM}")
    print(f"File Size:    {size} bytes ({size / 1024:.2f} KB)")

if __name__ == "__main__":
    main()
