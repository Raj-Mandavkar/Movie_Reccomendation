"""
model_builder.py — TMDB Movie Recommendation Model Builder
Loads, cleans, and processes the TMDB 5000 dataset to produce
a pickled movie DataFrame and cosine similarity matrix.
"""

import ast
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def parse_json_column(text):
    """Safely parse a JSON-like string column into a Python list."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []


def extract_names(obj_list, key="name", limit=None):
    """Extract values for a given key from a list of dicts."""
    names = [item[key] for item in obj_list if key in item]
    if limit:
        names = names[:limit]
    return names


def get_director(crew_list):
    """Extract the director's name from the crew list."""
    for member in crew_list:
        if member.get("job") == "Director":
            return member.get("name", "")
    return ""


def collapse_spaces(name_list):
    """Remove spaces from multi-word names to treat them as single tokens."""
    return [name.replace(" ", "") for name in name_list]


def build_model():
    print("📦 Loading datasets...")
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    # Merge on movie ID
    movies = movies.merge(credits, left_on="id", right_on="movie_id", how="left")

    # Keep only needed columns
    movies = movies[
        [
            "id",
            "title_x",
            "overview",
            "genres",
            "keywords",
            "cast",
            "crew",
            "vote_average",
            "runtime",
            "release_date",
            "popularity",
        ]
    ].copy()
    movies.rename(columns={"title_x": "title"}, inplace=True)

    # ── Load and append Indian movies ────────────────────────────
    import os
    if os.path.exists("indian_movies.csv"):
        print("🇮🇳 Loading Indian movies dataset...")
        indian = pd.read_csv("indian_movies.csv")
        # Rename columns to match TMDB merged format
        indian = indian[
            ["id", "title", "overview", "genres", "keywords",
             "cast", "crew", "vote_average", "runtime",
             "release_date", "popularity"]
        ].copy()
        movies = pd.concat([movies, indian], ignore_index=True)
        print(f"   Added {len(indian)} Indian movies")

    # Drop rows missing overview
    movies.dropna(subset=["overview"], inplace=True)

    # Fill any remaining NaNs
    movies["runtime"] = movies["runtime"].fillna(0)
    movies["vote_average"] = movies["vote_average"].fillna(0)
    movies["release_date"] = movies["release_date"].fillna("")
    movies["popularity"] = movies["popularity"].fillna(0)

    print("🔧 Parsing JSON columns...")
    movies["genres"] = movies["genres"].apply(parse_json_column)
    movies["keywords"] = movies["keywords"].apply(parse_json_column)
    movies["cast"] = movies["cast"].apply(parse_json_column)
    movies["crew"] = movies["crew"].apply(parse_json_column)

    # Extract names
    movies["genre_names"] = movies["genres"].apply(lambda x: extract_names(x))
    movies["keyword_names"] = movies["keywords"].apply(lambda x: extract_names(x))
    movies["top_cast"] = movies["cast"].apply(lambda x: extract_names(x, limit=3))
    movies["director"] = movies["crew"].apply(get_director)

    # Build the tags column
    print("🏷️  Building tags...")
    movies["tags"] = (
        movies["overview"].apply(lambda x: x.split())
        + movies["genre_names"].apply(collapse_spaces)
        + movies["keyword_names"].apply(collapse_spaces)
        + movies["top_cast"].apply(collapse_spaces)
        + movies["director"].apply(lambda x: [x.replace(" ", "")] if x else [])
    )
    movies["tags"] = movies["tags"].apply(lambda x: " ".join(x).lower())

    # Store genre names as comma-separated string for the API
    movies["genres_str"] = movies["genre_names"].apply(lambda x: ", ".join(x))

    print("🔢 Vectorizing tags...")
    cv = CountVectorizer(max_features=5000, stop_words="english")
    vectors = cv.fit_transform(movies["tags"])

    print("📐 Computing cosine similarity...")
    sim_matrix = cosine_similarity(vectors)

    # Prepare final DataFrame for serialization
    movies_out = movies[
        [
            "id",
            "title",
            "overview",
            "genres_str",
            "vote_average",
            "runtime",
            "release_date",
            "popularity",
            "tags",
        ]
    ].reset_index(drop=True)

    # Save
    print("💾 Saving pickled files...")
    pickle.dump(movies_out, open("movies.pkl", "wb"))
    pickle.dump(sim_matrix, open("similarity.pkl", "wb"))

    print(f"✅ Done! Processed {len(movies_out)} movies.")
    print(f"   movies.pkl   — {movies_out.shape}")
    print(f"   similarity.pkl — {sim_matrix.shape}")

    return movies_out, sim_matrix


if __name__ == "__main__":
    build_model()
