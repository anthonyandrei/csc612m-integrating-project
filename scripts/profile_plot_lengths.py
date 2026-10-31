#!/usr/bin/env python3
"""
Profile plot lengths in the CMU Movie Summary Corpus against the Sentence-BERT
tokenizer (all-MiniLM-L6-v2, 256 token limit).
"""

import sys
import numpy as np
from transformers import AutoTokenizer

DATA_PATH = "data/raw/MovieSummaries/plot_summaries.txt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    max_seq_len = 256

    print(f"Reading plots from {DATA_PATH}...")
    plot_ids = []
    token_lengths = []
    word_lengths = []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split("\t", 1)
            if len(parts) != 2:
                continue
            wiki_id, plot_text = parts
            plot_ids.append(wiki_id)
            
            # Word count
            words = plot_text.split()
            word_lengths.append(len(words))
            
            # Subword token count
            tokens = tokenizer.encode(plot_text, add_special_tokens=True, truncation=False)
            token_lengths.append(len(tokens))

            if (line_idx + 1) % 10000 == 0:
                print(f"Processed {line_idx + 1} plots...")

    token_arr = np.array(token_lengths)
    word_arr = np.array(word_lengths)
    total_plots = len(token_arr)

    print("\n=== PLOT LENGTH PROFILING RESULTS ===")
    print(f"Total plots analyzed: {total_plots}")
    print("\n--- Word Length Distribution ---")
    print(f"Min: {word_arr.min()}, Median: {np.median(word_arr):.1f}, Mean: {word_arr.mean():.1f}, Max: {word_arr.max()}")

    print("\n--- Subword Token Distribution (all-MiniLM-L6-v2) ---")
    print(f"Min tokens:    {token_arr.min()}")
    print(f"25th percentile: {np.percentile(token_arr, 25):.1f}")
    print(f"Median (50th):   {np.median(token_arr):.1f}")
    print(f"75th percentile: {np.percentile(token_arr, 75):.1f}")
    print(f"90th percentile: {np.percentile(token_arr, 90):.1f}")
    print(f"95th percentile: {np.percentile(token_arr, 95):.1f}")
    print(f"99th percentile: {np.percentile(token_arr, 99):.1f}")
    print(f"Max tokens:    {token_arr.max()}")

    exceed_256 = np.sum(token_arr > 256)
    exceed_512 = np.sum(token_arr > 512)

    print(f"\nPlots exceeding 256 tokens: {exceed_256} ({exceed_256 / total_plots * 100:.2f}%)")
    print(f"Plots exceeding 512 tokens: {exceed_512} ({exceed_512 / total_plots * 100:.2f}%)")
    print(f"Plots within 256 tokens:    {total_plots - exceed_256} ({(total_plots - exceed_256) / total_plots * 100:.2f}%)")

if __name__ == "__main__":
    main()
