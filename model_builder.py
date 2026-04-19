"""
model_builder.py — Enhanced Movie Recommendation Model
Uses TF-IDF, weighted features, and improved similarity scoring.
"""

import ast
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


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


def weighted_features(genre_names, keyword_names, top_cast, director, overview):
    """Create weighted combination of features - emphasis on genres and plot."""
    features = []
    
    # Genres (highest weight - repeat 4x for strong signal)
    for g in genre_names:
        features.extend([g.lower()] * 4)
    
    # Director (3x weight - strong influence)
    if director:
        d = director.replace(" ", "").lower()
        features.extend([d] * 3)
    
    # Keywords (2x weight - thematic elements)
    for k in keyword_names:
        features.extend([k.lower()] * 2)
    
    # Top cast (1.5x weight - star power)
    for c in top_cast:
        c_clean = c.replace(" ", "").lower()
        features.extend([c_clean] * 2)
    
    # Overview plot keywords (1x weight - reduced)
    if overview:
        words = overview.lower().split()
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now', 'their', 'they', 'them', 'his', 'her', 'its', 'our', 'film', 'movie', 'story'}
        filtered = [w for w in words if w not in stopwords and len(w) > 3]
        # Only take first 50 words from overview to avoid noise
        features.extend(filtered[:50])
    
    return " ".join(features)


def build_model():
    print("Loading datasets...")
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits, left_on="id", right_on="movie_id", how="left")

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
            "vote_count",
            "runtime",
            "release_date",
            "popularity",
        ]
    ].copy()
    movies.rename(columns={"title_x": "title"}, inplace=True)

    # Load Indian movies
    import os
    if os.path.exists("indian_movies.csv"):
        print("Loading Indian movies dataset...")
        indian = pd.read_csv("indian_movies.csv")
        indian_subset = indian[
            ["id", "title", "overview", "genres", "keywords",
             "cast", "crew", "vote_average", "runtime",
             "release_date", "popularity"]
        ].copy()
        # Add vote_count for Indian movies (default to low count)
        indian_subset["vote_count"] = 0
        movies = pd.concat([movies, indian_subset], ignore_index=True)
        print(f"   Added {len(indian_subset)} Indian movies")

    # Load International movies
    if os.path.exists("international_movies.csv"):
        print("Loading International movies dataset...")
        intl = pd.read_csv("international_movies.csv")
        intl_subset = intl[
            ["id", "title", "overview", "genres", "keywords",
             "cast", "crew", "vote_average", "runtime",
             "release_date", "popularity"]
        ].copy()
        intl_subset["vote_count"] = 0
        movies = pd.concat([movies, intl_subset], ignore_index=True)
        print(f"   Added {len(intl_subset)} International movies")

    movies.dropna(subset=["overview"], inplace=True)

    movies["runtime"] = movies["runtime"].fillna(0)
    movies["vote_average"] = movies["vote_average"].fillna(0)
    movies["vote_count"] = movies["vote_count"].fillna(0)
    movies["release_date"] = movies["release_date"].fillna("")
    movies["popularity"] = movies["popularity"].fillna(0)

    # Filter out low-quality movies (optional: comment out to keep all)
    # movies = movies[(movies["vote_count"] >= 10) | (movies["vote_average"] >= 6.0)]
    # print(f"After filtering low-quality: {len(movies)} movies")

    print("Parsing JSON columns...")
    movies["genres"] = movies["genres"].apply(parse_json_column)
    movies["keywords"] = movies["keywords"].apply(parse_json_column)
    movies["cast"] = movies["cast"].apply(parse_json_column)
    movies["crew"] = movies["crew"].apply(parse_json_column)

    movies["genre_names"] = movies["genres"].apply(lambda x: extract_names(x))
    movies["keyword_names"] = movies["keywords"].apply(lambda x: extract_names(x))
    movies["top_cast"] = movies["cast"].apply(lambda x: extract_names(x, limit=5))
    movies["director"] = movies["crew"].apply(get_director)

    # Calculate weighted rating (IMDB formula)
    # WR = (v / (v + m)) * R + (m / (v + m)) * C
    # R = vote_average, v = vote_count, m = min votes required (default 50), C = mean vote
    m = 50  # minimum votes threshold
    C = movies["vote_average"].mean()
    movies["weighted_rating"] = (movies["vote_count"] / (movies["vote_count"] + m) * movies["vote_average"]) + (m / (movies["vote_count"] + m) * C)

    # Extract year
    movies["year"] = movies["release_date"].apply(lambda x: x[:4] if x and len(x) >= 4 else "")

    print("Building weighted tags...")
    movies["tags"] = movies.apply(
        lambda row: weighted_features(
            row["genre_names"],
            row["keyword_names"],
            row["top_cast"],
            row["director"],
            row["overview"]
        ),
        axis=1
    )

    movies["genres_str"] = movies["genre_names"].apply(lambda x: ", ".join(x))

    print("Vectorizing with TF-IDF...")
    tfidf = TfidfVectorizer(
        max_features=8000,
        stop_words="english",
        ngram_range=(1, 2),  # Include bigrams for better context
        min_df=2,           # Ignore very rare terms
        max_df=0.95         # Ignore very common terms
    )
    vectors = tfidf.fit_transform(movies["tags"])

    print("Computing cosine similarity...")
    sim_matrix = cosine_similarity(vectors).astype(np.float16)

    movies_out = movies[
        [
            "id",
            "title",
            "overview",
            "genres_str",
            "vote_average",
            "weighted_rating",
            "runtime",
            "release_date",
            "popularity",
            "tags",
            "genre_names",
        ]
    ].reset_index(drop=True)

    print("Saving pickled files...")
    pickle.dump(movies_out, open("movies.pkl", "wb"))
    pickle.dump(sim_matrix, open("similarity.pkl", "wb"))

    # Also save compressed versions
    import gzip
    pickle.dump(movies_out, gzip.open("movies.pkl.gz", "wb"))
    pickle.dump(sim_matrix, gzip.open("similarity.pkl.gz", "wb"))

    print(f"Done! Processed {len(movies_out)} movies.")
    print(f"   movies.pkl — {movies_out.shape}")
    print(f"   similarity.pkl — {sim_matrix.shape}")

    return movies_out, sim_matrix


if __name__ == "__main__":
    build_model()