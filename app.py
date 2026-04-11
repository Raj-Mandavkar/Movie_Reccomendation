"""
app.py — Flask backend for the Movie Recommendation System
Serves the main page and provides API endpoints for recommendations.
"""

import os
import pickle
import gzip
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Lazy-loaded models (loaded on first request, not at import time) ──
_movies = None
_similarity = None
_all_genres = None


def _load_models():
    """Load pre-computed models from gzip-compressed pickle files."""
    global _movies, _similarity, _all_genres

    if _movies is not None:
        return

    print("🔄 Loading model data...")

    # Try compressed files first, fall back to uncompressed
    if os.path.exists("movies.pkl.gz"):
        with gzip.open("movies.pkl.gz", "rb") as f:
            _movies = pickle.load(f)
    else:
        _movies = pickle.load(open("movies.pkl", "rb"))

    if os.path.exists("similarity.pkl.gz"):
        with gzip.open("similarity.pkl.gz", "rb") as f:
            _similarity = pickle.load(f)
    else:
        _similarity = pickle.load(open("similarity.pkl", "rb"))

    print(f"✅ Loaded {len(_movies)} movies")

    # Precompute genre list
    genres_set = set()
    for g_str in _movies["genres_str"]:
        if g_str:
            for g in g_str.split(", "):
                g = g.strip()
                if g:
                    genres_set.add(g)
    _all_genres = sorted(genres_set)


def recommend(title, genre_filter=None, min_rating=0, max_runtime=999):
    """
    Find similar movies and apply hard filters.
    Returns a list of dicts with movie metadata.
    """
    _load_models()

    # Find the movie index
    matches = _movies[_movies["title"].str.lower() == title.lower()]
    if matches.empty:
        return []

    idx = matches.index[0]

    # Get similarity scores
    distances = list(enumerate(_similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    results = []
    for i, score in distances[1:]:  # skip self
        if len(results) >= 10:
            break

        movie = _movies.iloc[i]

        # Hard filters
        if movie["vote_average"] < min_rating:
            continue
        if movie["runtime"] > max_runtime and max_runtime > 0:
            continue
        if genre_filter and genre_filter.lower() not in movie["genres_str"].lower():
            continue

        results.append(
            {
                "title": movie["title"],
                "genres": movie["genres_str"],
                "rating": round(float(movie["vote_average"]), 1),
                "runtime": int(movie["runtime"]) if movie["runtime"] else 0,
                "overview": movie["overview"],
                "release_date": movie.get("release_date", ""),
                "popularity": round(float(movie.get("popularity", 0)), 1),
                "similarity": round(float(score), 4),
            }
        )

    return results


def browse_movies(genre_filter=None, min_rating=0, max_runtime=999):
    """
    Browse movies by filters without requiring a title.
    Returns top movies sorted by popularity matching the filters.
    """
    _load_models()

    filtered = _movies.copy()

    if genre_filter:
        filtered = filtered[filtered["genres_str"].str.lower().str.contains(genre_filter.lower(), na=False)]
    if min_rating > 0:
        filtered = filtered[filtered["vote_average"] >= min_rating]
    if max_runtime < 999:
        filtered = filtered[filtered["runtime"] <= max_runtime]

    # Sort by popularity descending
    filtered = filtered.sort_values("popularity", ascending=False).head(12)

    results = []
    for _, movie in filtered.iterrows():
        results.append(
            {
                "title": movie["title"],
                "genres": movie["genres_str"],
                "rating": round(float(movie["vote_average"]), 1),
                "runtime": int(movie["runtime"]) if movie["runtime"] else 0,
                "overview": movie["overview"],
                "release_date": movie.get("release_date", ""),
                "popularity": round(float(movie.get("popularity", 0)), 1),
                "similarity": 0,  # no similarity in browse mode
            }
        )

    return results


# ── Routes ───────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/movies")
def api_movies():
    """Return all movie titles for autocomplete."""
    _load_models()
    movie_list = _movies[["title", "genres_str", "vote_average", "runtime"]].to_dict(
        orient="records"
    )
    return jsonify(movie_list)


@app.route("/api/genres")
def api_genres():
    """Return unique genre list."""
    _load_models()
    return jsonify(_all_genres)


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Get recommendations with optional filters. Title is optional."""
    data = request.get_json()
    title = data.get("title", "").strip()
    genre_filter = data.get("genre", None)
    min_rating = float(data.get("min_rating", 0))
    max_runtime = float(data.get("max_runtime", 999))

    # Browse mode: no title, just filters
    if not title:
        results = browse_movies(genre_filter, min_rating, max_runtime)
        if not results:
            return jsonify({"error": "No movies match your filters"}), 404
        label = genre_filter if genre_filter else "Popular"
        return jsonify({"movie": None, "browse_label": label, "recommendations": results})

    # Similarity mode: title provided
    results = recommend(title, genre_filter, min_rating, max_runtime)

    if not results:
        return jsonify({"error": f"No recommendations found for '{title}'"}), 404

    return jsonify({"movie": title, "recommendations": results})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
