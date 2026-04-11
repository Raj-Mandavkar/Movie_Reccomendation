"""
compress_models.py — Compress pickle files with gzip for Vercel deployment.
Run this once to produce .pkl.gz files that are much smaller on disk.
"""

import pickle
import gzip
import os

def compress_file(input_path, output_path):
    """Load a pickle file and re-save it with gzip compression."""
    print(f"🔄 Loading {input_path}...")
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    print(f"💾 Compressing to {output_path}...")
    with gzip.open(output_path, "wb", compresslevel=6) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    original_size = os.path.getsize(input_path) / (1024 * 1024)
    compressed_size = os.path.getsize(output_path) / (1024 * 1024)
    ratio = (1 - compressed_size / original_size) * 100

    print(f"✅ {input_path}: {original_size:.2f} MB → {compressed_size:.2f} MB ({ratio:.1f}% reduction)")


if __name__ == "__main__":
    compress_file("movies.pkl", "movies.pkl.gz")
    compress_file("similarity.pkl", "similarity.pkl.gz")
    print("\n🎉 Done! Compressed files are ready for deployment.")
