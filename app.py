"""
app.py — Flask backend for the Movie Recommendation System
Serves the main page and provides API endpoints for recommendations.
Enhanced with Gemini AI for smarter recommendations.
"""

import os
import pickle
import gzip
import json
import re
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# ── API Credentials ──────────────────────────────────────────────────
TMDB_API_KEY = "a511f69f92385dec34fdff2b1f73d56b"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
GEMINI_API_KEYS = [
    "AIzaSyDBbpQhB3xLQkW4CpWbGrCvA5t5rZokv24",  # Primary (backup key)
    "AIzaSyCHt1IpwsxCl7gFGIY52wAqFlHKV6G3OjI",  # Key 1
    "AIzaSyABgHzkkCxLtJl7b7KtJ3i4OHdJooqXNVY",  # Key 2
    "AIzaSyC-YrcNlUCtYRk7MNCiFJNlZ_yqkO0INGI",  # Key 3
    "AIzaSyBfPphovr8jWTBUdSzZ5RkpPrEd82ZbZ08", # Key 4
]
GEMINI_API_KEY = GEMINI_API_KEYS[0]  # Default to first key

# Initialize Gemini client (new SDK) with 40s timeout
def _create_genai_client(api_key):
    return genai.Client(
        api_key=api_key,
        http_options={"timeout": 40000}
    )

genai_client = _create_genai_client(GEMINI_API_KEY)
_current_key_index = 0

def _get_next_api_key():
    """Rotate to next available API key on failure."""
    global _current_key_index, genai_client
    _current_key_index = (_current_key_index + 1) % len(GEMINI_API_KEYS)
    new_key = GEMINI_API_KEYS[_current_key_index]
    print(f"Rotating to API key #{_current_key_index + 1}")
    genai_client = _create_genai_client(new_key)
    return genai_client

# ── Caching for TMDB calls ──────────────────────────────────────────
_tmdb_movie_cache = {}
_gemini_cache = {}  # Cache Gemini responses for same movie title

def get_movie_poster(movie_id):
    """Fetch the poster image URL from TMDB."""
    url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    return None

# ── Lazy-loaded models (loaded on first request, not at import time) ──
_movies = None
_similarity = None
_all_genres = None


def _load_models():
    """Load pre-computed models from gzip-compressed pickle files."""
    global _movies, _similarity, _all_genres

    if _movies is not None:
        return

    print("Loading model data...")

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

    print(f"Loaded {len(_movies)} movies")

    # Precompute genre list
    genres_set = set()
    for g_str in _movies["genres_str"]:
        if g_str:
            for g in g_str.split(", "):
                g = g.strip()
                if g:
                    genres_set.add(g)
    _all_genres = sorted(genres_set)


def get_tmdb_movie(tmdb_id):
    """Fetch full movie details from TMDB API with caching."""
    # Check cache first
    if tmdb_id in _tmdb_movie_cache:
        return _tmdb_movie_cache[tmdb_id]
    
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            _tmdb_movie_cache[tmdb_id] = data
            return data
    except Exception:
        pass
    return None


def get_gemini_recommendations(movie_title, limit=5, industry="Global", genre=None):
    """
    Use Gemini AI to find movies matching narrative DNA.
    Handles sequels, global recommendations, and uses permissive safety settings.
    Tries multiple models in order until one succeeds.
    Returns list of {title, tmdb_id}.
    
    Args:
        movie_title: The movie to find recommendations for
        limit: Number of recommendations to return
        industry: Industry filter - "Global", "Hollywood", "Bollywood", "Asian", etc.
        genre: Genre filter - "Action", "Drama", etc. or None for all
    """
    genre_filter = genre if genre else "All Genres"
    cache_key = f"{movie_title.lower().strip()}_{limit}_{industry}_{genre_filter}"
    if cache_key in _gemini_cache:
        return _gemini_cache[cache_key]
    
    # High-accuracy prompt with franchise/sequel handling and industry filter
    '''Main PROMPT :
    prompt = f"""Target Movie: {movie_title}

TASK: Provide {limit} movie recommendations.



PRIORITY 1 (FRANCHISE): If '{movie_title}' is a sequel (e.g., has 'Part 2', 'Reunion', or 'The Revenge' in the title), the VERY FIRST recommendation MUST be the preceding parts of that franchise (e.g., Part 1).



PRIORITY 2 (GLOBAL DNA): Fill the remaining slots with movies that match the 'Narrative DNA' (story archetypes, character arcs, and emotional core).

- DO NOT limit results to the language of the target movie.'''

    # Assuming 'industry' is like "Bollywood", "Hollywood", "Global (All)"
    # Assuming 'genre' is like "Crime", "Thriller", "All Genres"
    # Assuming 'movie_title' is like "Omkara"
    prompt = f"""Target Movie: {movie_title}
    Target Industry/Region: {industry}
    Filter Genre: {genre}
    TASK: Provide {limit} movie recommendations.

PRIORITY 1 (FRANCHISE): If '{movie_title}' is part of a series (sequel/prequel), the VERY FIRST recommendation MUST be the preceding/succeeding parts of that specific franchise.

PRIORITY 2 (REGIONAL & SEMANTIC FILTER): 
- If Target Industry is 'Global': Use your 'Narrative DNA' logic across all world cinema, prioritizing '{genre}'.
- If Target Industry is NOT 'Global':
  - 1. You MUST first isolate movies ONLY from the '{industry}' film industry.
  - 2. Inside that isolated industry subset, find movies that match the 'Genre: {genre}' AND 'Narrative DNA' of '{movie_title}'.
  - DO NOT suggest any movies outside the selected '{industry}'.

PRIORITY 3 (NARRATIVE DNA): Match the 'Genetic Code' (story archetypes, character arcs, emotional core).
- Example: If input is 'Omkara' and Region is 'Bollywood', match 'Othello/Manipulation' archetypes found within other Bollywood films (e.g., matching with 'Haider' for the complex betrayal/Shakespearean DNA).

Return ONLY a raw JSON array: [{{"title": "Exact Title", "tmdb_id": 12345}}]. 
No conversational text, no markdown backticks."""
    
    # Safety settings - most permissive to avoid blocking content
    safety_config = types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(category='HARM_CATEGORY_HATE_SPEECH', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_HARASSMENT', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold='BLOCK_NONE'),
            types.SafetySetting(category='HARM_CATEGORY_DANGEROUS_CONTENT', threshold='BLOCK_NONE'),
        ]
    )
    
    # Try models in decreasing quality order; each has separate quota
    models_to_try = [
        "gemini-2.5-flash",        # Best reasoning, lower free quota
        "gemini-2.0-flash-lite",   # Good, separate quota
        "gemini-flash-latest",     # Gemini 1.5 Flash, higher quota
        "gemini-pro-latest",       # Gemini 1.5 Pro, another quota pool
    ]
    
    for model_name in models_to_try:
        try:
            response = genai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=safety_config
            )
            response_text = response.text.strip()
            
            # Robust JSON extraction - handles extra text before/after
            try:
                # Find JSON array between first [ and last ]
                match = re.search(r'\[.*\]', response_text, re.DOTALL)
                if match:
                    json_str = match.group()
                    recommendations = json.loads(json_str)
                else:
                    # Fallback: try parsing entire response
                    recommendations = json.loads(response_text)
            except json.JSONDecodeError as jde:
                print(f"JSON parse error for '{movie_title}': {jde}")
                print(f"Response text: {response_text[:200]}")
                continue
            
            # Validate and extract title/tmdb_id
            validated = []
            for rec in recommendations:
                if isinstance(rec, dict) and rec.get('title'):
                    validated.append({
                        'title': rec['title'],
                        'tmdb_id': rec.get('tmdb_id') or rec.get('tmdbId') or None
                    })
            
            result = validated[:limit]
            _gemini_cache[cache_key] = result
            print(f"Gemini [{model_name}] -> {len(result)} recs for '{movie_title}'")
            return result
              
        except Exception as e:
            error_str = str(e)
            # Check for quota exhaustion - rotate to next key
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(f"Quota exhausted for {model_name}, trying next API key...")
                _get_next_api_key()
                continue
            print(f"Model {model_name} failed: {type(e).__name__}: {error_str[:120]}")
            continue
    
    print(f"All Gemini models exhausted for: {movie_title}")
    return []


def fetch_movie_from_tmdb(movie_title, tmdb_id=None):
    """
    Fetch movie details from TMDB either by ID or search.
    If tmdb_id provided, verifies it matches the expected title.
    """
    try:
        effective_tmdb_id = None
        if tmdb_id and tmdb_id > 0:
            # Verify the ID matches the title before trusting it
            movie_data = get_tmdb_movie(tmdb_id)
            if movie_data:
                fetched_title = movie_data.get("title", "")
                # Check for close match (case-insensitive, partial)
                if fetched_title and movie_title.lower() in fetched_title.lower() or fetched_title.lower() in movie_title.lower():
                    effective_tmdb_id = tmdb_id
                else:
                    # ID doesn't match title - fall back to search
                    effective_tmdb_id = None
        
        if effective_tmdb_id:
            # Use verified ID
            return {
                "id": movie_data.get("id"),
                "title": movie_data.get("title"),
                "genres": ", ".join([g["name"] for g in movie_data.get("genres", [])]),
                "rating": round(movie_data.get("vote_average", 0), 1),
                "runtime": movie_data.get("runtime", 0),
                "overview": movie_data.get("overview", ""),
                "release_date": movie_data.get("release_date", ""),
                "popularity": round(movie_data.get("popularity", 0), 1),
                "poster": f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}" if movie_data.get("poster_path") else None,
            }
        else:
            # Search by title
            url = f"{TMDB_BASE_URL}/search/movie"
            params = {
                "api_key": TMDB_API_KEY,
                "query": movie_title,
                "page": 1
            }
            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("results"):
                    movie = data["results"][0]  # Take first result
                    return {
                        "id": movie.get("id"),
                        "title": movie.get("title"),
                        "genres": "",
                        "rating": round(movie.get("vote_average", 0), 1),
                        "runtime": 0,
                        "overview": movie.get("overview", ""),
                        "release_date": movie.get("release_date", ""),
                        "popularity": round(movie.get("popularity", 0), 1),
                        "poster": f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get("poster_path") else None,
                    }
    except Exception as e:
        print(f"TMDB fetch error: {e}")
    
    return None


def recommend_external(tmdb_id, genre_filter=None, min_rating=0, max_runtime=999):
    """
    For movies not in local dataset: fetch from TMDB and find similar by genre matching + weighted rating.
    """
    _load_models()
    tmdb_movie = get_tmdb_movie(tmdb_id)
    if not tmdb_movie:
        return []

    tmdb_genres = [g["name"] for g in tmdb_movie.get("genres", [])]
    genres_str = ", ".join(tmdb_genres) if tmdb_genres else ""

    # Calculate weighted rating for TMDB movie (approximate)
    tmdb_vote_count = tmdb_movie.get("vote_count", 0) or 0
    tmdb_rating = tmdb_movie.get("vote_average", 0) or 0
    m = 50
    C = _movies["vote_average"].mean()
    tmdb_weighted = (tmdb_vote_count / (tmdb_vote_count + m) * tmdb_rating) + (m / (tmdb_vote_count + m) * C)

    results = []
    for _, movie in _movies.iterrows():
        if len(results) >= 12:
            break
        if movie["vote_average"] < min_rating:
            continue
        if movie["runtime"] > max_runtime and max_runtime > 0:
            continue
        if genre_filter and genre_filter.lower() not in movie["genres_str"].lower():
            continue

        movie_genres = set(g.strip().lower() for g in movie["genres_str"].split(", ") if g.strip())
        query_genres = set(g.lower() for g in tmdb_genres)
        overlap = len(movie_genres & query_genres)
        if overlap == 0:
            continue

        similarity = overlap / max(len(movie_genres), 1)
        
        results.append(
            {
                "id": int(movie.get("id", 0)),
                "title": movie["title"],
                "genres": movie["genres_str"],
                "rating": round(float(movie["vote_average"]), 1),
                "weighted_rating": round(float(movie.get("weighted_rating", 0)), 2),
                "runtime": int(movie["runtime"]) if movie["runtime"] else 0,
                "overview": movie["overview"],
                "release_date": movie.get("release_date", ""),
                "popularity": round(float(movie.get("popularity", 0)), 1),
                "similarity": round(float(similarity), 4),
            }
        )

    results.sort(key=lambda x: (x["similarity"], x.get("weighted_rating", 0)), reverse=True)

    with ThreadPoolExecutor(max_workers=10) as executor:
        posters = list(executor.map(lambda x: get_movie_poster(x["id"]) if x["id"] else None, results))
    for i, res in enumerate(results):
        res["poster"] = posters[i]

    return results


def recommend(title, genre_filter=None, min_rating=0, max_runtime=999):
    """
    Find similar movies using hybrid scoring: cosine similarity + genre overlap + quality.
    """
    _load_models()

    # Find the movie index - case insensitive partial match
    matches = _movies[_movies["title"].str.lower().str.contains(title.lower(), na=False)]
    if matches.empty:
        return []

    # Use the first match
    idx = matches.index[0]
    source_movie = _movies.iloc[idx]
    source_genres = set(g.strip().lower() for g in source_movie["genres_str"].split(", ") if g.strip())

    # Get similarity scores
    distances = list(enumerate(_similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    results = []
    for i, score in distances[1:]:  # skip self
        movie = _movies.iloc[i]

        # Calculate weighted rating
        weighted_rating = movie.get("weighted_rating", movie["vote_average"]) or movie["vote_average"]
        
        # Skip low-quality movies (below 5.5 weighted rating)
        if weighted_rating < 5.5:
            continue
        
        # Hard filters - use weighted rating
        if weighted_rating < min_rating:
            continue
        if movie["runtime"] > max_runtime and max_runtime > 0:
            continue
        if genre_filter and genre_filter.lower() not in movie["genres_str"].lower():
            continue

        # Calculate genre overlap
        movie_genres = set(g.strip().lower() for g in movie["genres_str"].split(", ") if g.strip())
        genre_overlap = len(source_genres & movie_genres) / max(len(source_genres), 1)
        
        # Quality boost based on weighted rating (0-10 scale normalized)
        quality_boost = (weighted_rating - 5) / 10
        
        # Combined score: balance raw similarity with quality
        combined_score = (score * 0.4) + (genre_overlap * 0.3) + (quality_boost * 0.3)

        if len(results) >= 15 and combined_score < results[-1]["similarity"]:
            continue
        if combined_score < 0.1:
            continue

        results.append(
            {
                "id": int(movie.get("id", 0)),
                "title": movie["title"],
                "genres": movie["genres_str"],
                "rating": round(float(movie["vote_average"]), 1),
                "weighted_rating": round(float(weighted_rating), 2),
                "runtime": int(movie["runtime"]) if movie["runtime"] else 0,
                "overview": movie["overview"],
                "release_date": movie.get("release_date", ""),
                "popularity": round(float(movie.get("popularity", 0)), 1),
                "similarity": round(float(combined_score), 4),
                "raw_similarity": round(float(score), 4),
            }
        )

    # Re-sort by combined score and take top 12
    results.sort(key=lambda x: x["similarity"], reverse=True)
    results = results[:12]

    # Fetch posters concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        posters = list(executor.map(lambda x: get_movie_poster(x["id"]) if x["id"] else None, results))
    for i, res in enumerate(results):
        res["poster"] = posters[i]

    return results


def get_movie_language(tmdb_id):
    """Fetch movie language/origin from TMDB."""
    url = f"{TMDB_BASE_URL}/movie/{tmdb_id}?api_key={TMDB_API_KEY}"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            return data.get("original_language", "en")
    except Exception:
        pass
    return None


def get_popular_movies(limit=15, category=None):
    """Get trending/popular movies globally or by category (Hollywood/Bollywood/etc)."""
    _load_models()
    
    filtered = _movies.copy()
    
    # Filter by category if provided
    if category and category.lower() == "bollywood":
        # Language-based filtering - Hindi movies are typically Bollywood
        filtered = filtered[filtered["genres_str"].str.contains("Drama|Action|Romance", case=False, na=False)]
        # Sort by popularity and rating
        filtered = filtered.sort_values("popularity", ascending=False).head(limit)
    elif category and category.lower() == "hollywood":
        # English language movies - Hollywood
        filtered = filtered.sort_values("popularity", ascending=False).head(limit)
    else:
        # Global popular
        filtered = filtered.sort_values("popularity", ascending=False).head(limit)
    
    results = []
    for _, movie in filtered.iterrows():
        results.append({
            "id": int(movie.get("id", 0)),
            "title": movie["title"],
            "genres": movie["genres_str"],
            "rating": round(float(movie["vote_average"]), 1),
            "runtime": int(movie["runtime"]) if movie["runtime"] else 0,
            "overview": movie["overview"],
            "release_date": movie.get("release_date", ""),
            "popularity": round(float(movie.get("popularity", 0)), 1),
            "similarity": 0,
        })
    
    # Fetch posters concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        posters = list(executor.map(lambda x: get_movie_poster(x["id"]) if x["id"] else None, results))
    for i, res in enumerate(results):
        res["poster"] = posters[i]
    
    return results


def get_latest_movies(limit=15, genre_filter=None):
    """Get latest releases sorted by release date, including fresh TMDB releases."""
    _load_models()
    
    filtered = _movies.copy()
    
    # Filter by genre if provided
    if genre_filter:
        filtered = filtered[filtered["genres_str"].str.lower().str.contains(genre_filter.lower(), na=False)]
    
    # Sort by release date descending (latest first)
    filtered = filtered.sort_values("release_date", ascending=False, na_position='last').head(limit)
    
    results = []
    for _, movie in filtered.iterrows():
        results.append({
            "id": int(movie.get("id", 0)),
            "title": movie["title"],
            "genres": movie["genres_str"],
            "rating": round(float(movie["vote_average"]), 1),
            "runtime": int(movie["runtime"]) if movie["runtime"] else 0,
            "overview": movie["overview"],
            "release_date": movie.get("release_date", ""),
            "popularity": round(float(movie.get("popularity", 0)), 1),
            "similarity": 0,
        })
    
    # Fetch posters concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        posters = list(executor.map(lambda x: get_movie_poster(x["id"]) if x["id"] else None, results))
    for i, res in enumerate(results):
        res["poster"] = posters[i]
    
    # Titles to exclude from latest (problematic/bad quality)
    excluded_titles = [
        "di kematian part", "kkn di desa penari", "the big 4 part",
        "dangal", "sarbjit", "pk"
    ]
    
    # Fetch latest from TMDB to supplement local data with fresh releases
    try:
        url = f"{TMDB_BASE_URL}/movie/now_playing"
        params = {
            "api_key": TMDB_API_KEY,
            "page": 1,
            "region": "IN"
        }
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            for m in data.get("results", [])[:10]:
                title_lower = m["title"].lower()
                # Skip excluded titles
                if any(exc in title_lower for exc in excluded_titles):
                    continue
                # Skip if already in results
                if any(r["title"].lower() == title_lower for r in results):
                    continue
                results.append({
                    "id": m["id"],
                    "title": m["title"],
                    "genres": "",
                    "rating": round(m.get("vote_average", 0), 1),
                    "runtime": m.get("runtime", 0) or 0,
                    "overview": m.get("overview", "")[:200],
                    "release_date": m.get("release_date", ""),
                    "popularity": round(m.get("popularity", 0), 1),
                    "similarity": 0,
                    "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None,
                })
    except Exception:
        pass
    
    # Add specific recent popular titles manually
    manual_titles = [
        {"title": "Durga", "year": "2026"},
        {"title": "Hadar", "year": "2026"},
        {"title": "Mismatched", "year": "2026"},
        {"title": "Crackdown", "year": "2026"},
    ]
    
    for mt in manual_titles:
        if len(results) >= limit:
            break
        if any(r["title"].lower() == mt["title"].lower() for r in results):
            continue
        # Try to fetch from TMDB
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": mt["title"], "page": 1}
        try:
            resp = requests.get(search_url, params=params, timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("results"):
                    m = data["results"][0]
                    if m.get("poster_path"):
                        results.append({
                            "id": m["id"],
                            "title": m["title"],
                            "genres": "",
                            "rating": round(m.get("vote_average", 0), 1),
                            "runtime": m.get("runtime", 0) or 0,
                            "overview": m.get("overview", "")[:200],
                            "release_date": m.get("release_date", ""),
                            "popularity": round(m.get("popularity", 0), 1),
                            "similarity": 0,
                            "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}",
                        })
        except Exception:
            pass
    
    # Re-sort by release date and limit
    results.sort(key=lambda x: x.get("release_date", ""), reverse=True)
    return results[:limit]


def get_movies_by_genre(genre_name, limit=15):
    """Get popular movies for a specific genre."""
    _load_models()
    
    filtered = _movies[_movies["genres_str"].str.lower().str.contains(genre_name.lower(), na=False)].copy()
    
    # Sort by weighted rating
    if "weighted_rating" in filtered.columns:
        filtered = filtered.sort_values("weighted_rating", ascending=False).head(limit)
    else:
        filtered = filtered.sort_values("popularity", ascending=False).head(limit)
    
    results = []
    for _, movie in filtered.iterrows():
        results.append({
            "id": int(movie.get("id", 0)),
            "title": movie["title"],
            "genres": movie["genres_str"],
            "rating": round(float(movie["vote_average"]), 1),
            "runtime": int(movie["runtime"]) if movie["runtime"] else 0,
            "overview": movie["overview"],
            "release_date": movie.get("release_date", ""),
            "popularity": round(float(movie.get("popularity", 0)), 1),
            "similarity": 0,
        })
    
    # Fetch posters concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        posters = list(executor.map(lambda x: get_movie_poster(x["id"]) if x["id"] else None, results))
    for i, res in enumerate(results):
        res["poster"] = posters[i]
    
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

    # Sort by weighted rating instead of popularity for browse
    if "weighted_rating" in filtered.columns and filtered["weighted_rating"].notna().any():
        filtered = filtered.sort_values("weighted_rating", ascending=False).head(20)
    else:
        filtered = filtered.sort_values("popularity", ascending=False).head(20)

    results = []
    for _, movie in filtered.iterrows():
        results.append(
            {
                "id": int(movie.get("id", 0)),
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

    # Fetch posters concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        posters = list(executor.map(lambda x: get_movie_poster(x["id"]) if x["id"] else None, results))
    for i, res in enumerate(results):
        res["poster"] = posters[i]

    # If local results < 12, fetch more from TMDB to make it truly international
    if len(results) < 12:
        try:
            url = f"{TMDB_BASE_URL}/discover/movie"
            params = {
                "api_key": TMDB_API_KEY,
                "sort_by": "popularity.desc",
                "page": 1,
                "vote_count.gte": 50,
            }
            if genre_filter:
                # Find genre ID for filter
                genre_url = f"{TMDB_BASE_URL}/genre/movie/list"
                genre_resp = requests.get(genre_url, params={"api_key": TMDB_API_KEY}, timeout=5)
                if genre_resp.status_code == 200:
                    genres_data = genre_resp.json()
                    for g in genres_data.get("genres", []):
                        if g["name"].lower() == genre_filter.lower():
                            params["with_genres"] = g["id"]
                            break
            if min_rating > 0:
                params["vote_average.gte"] = min_rating
            if max_runtime < 999:
                params["with_runtime.lte"] = max_runtime

            response = requests.get(url, params=params, timeout=5)
            if response.status_code == 200:
                data = response.json()
                for m in data.get("results", []):
                    if len(results) >= 20:  # Cap at 20
                        break
                    # Skip if already in results
                    if any(r["id"] == m["id"] for r in results):
                        continue
                    results.append({
                        "id": m["id"],
                        "title": m["title"],
                        "genres": "",  # Would need another call to get genres
                        "rating": round(m.get("vote_average", 0), 1),
                        "runtime": 0,
                        "overview": m.get("overview", "")[:200],
                        "release_date": m.get("release_date", ""),
                        "popularity": round(m.get("popularity", 0), 1),
                        "similarity": 0,
                        "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None,
                        "is_tmdb": True,  # Mark as TMDB-only
                    })
        except Exception:
            pass

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


@app.route("/api/search")
def api_search():
    """Live TMDB search for movies beyond local dataset."""
    query = request.args.get("q", "").strip()
    if not query or len(query) < 2:
        return jsonify([])

    url = f"{TMDB_BASE_URL}/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": query, "page": 1}
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code != 200:
            return jsonify([])
        data = response.json()
        results = [
            {
                "id": m["id"],
                "title": m["title"],
                "release_date": m.get("release_date", "")[:4] if m.get("release_date") else "",
                "vote_average": round(m.get("vote_average", 0), 1),
                "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None,
                "overview": m.get("overview", "")[:200],
            }
            for m in data.get("results", [])[:8]
        ]
        return jsonify(results)
    except Exception:
        return jsonify([])


@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """Get AI-powered recommendations using Gemini. No fallbacks - pure AI."""
    data = request.get_json()
    title = data.get("title", "").strip()
    tmdb_id = data.get("tmdb_id", 0)
    genre_filter = data.get("genre", None)
    min_rating = float(data.get("min_rating", 0))
    max_runtime = float(data.get("max_runtime", 999))
    industry = data.get("industry", "Global")

    # Browse mode: no title, just filters
    if not title and not tmdb_id:
        results = browse_movies(genre_filter, min_rating, max_runtime)
        if not results:
            return jsonify({"error": "No movies match your filters"}), 404
        # Show just genre name (e.g., "Science Fiction") or "All Genres"
        if genre_filter:
            label = genre_filter
        else:
            label = "All Genres"
        return jsonify({"movie": None, "browse_label": label, "recommendations": results, "method": "browse"})

    # STRICT: Use ONLY Gemini AI for recommendations - no fallbacks
    if title:
        try:
            # Get AI recommendations from Gemini
            ai_recommendations = get_gemini_recommendations(title, limit=8, industry=industry, genre=genre_filter)
            
            if not ai_recommendations:
                return jsonify({"error": "Slow network or connection issue. Please check your internet and try again."}), 404
            
            results = []
            # Fetch metadata from TMDB for each AI recommendation in parallel
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(fetch_movie_from_tmdb, rec.get("title"), rec.get("tmdb_id"))
                    for rec in ai_recommendations
                ]
                for i, future in enumerate(futures):
                    try:
                        movie_data = future.result(timeout=6)
                        if movie_data:
                            # Apply filters
                            if movie_data["rating"] < min_rating:
                                continue
                            if movie_data["runtime"] > max_runtime and max_runtime > 0:
                                continue
                            if genre_filter and genre_filter.lower() not in movie_data.get("genres", "").lower():
                                continue
                            
                            movie_data["similarity"] = round(1.0 - (i * 0.12), 2)
                            results.append(movie_data)
                    except Exception as e:
                        print(f"TMDB fetch error for rec {i}: {e}")
                        continue
            
            if results:
                return jsonify({
                    "movie": title, 
                    "recommendations": results, 
                    "method": "gemini_ai",
                    "source": "AI analysis of narrative DNA"
                })
            else:
                return jsonify({"error": "No recommendations found after filtering"}), 404
                
        except Exception as e:
            print(f"Gemini recommendation error: {e}")
            return jsonify({"error": "Network timeout or service unavailable. Please try again in a moment."}), 503

    # Should never reach here, but handle tmdb_id-only case
    if tmdb_id:
        results = recommend_external(tmdb_id, genre_filter, min_rating, max_runtime)
        if results:
            tmdb_movie = get_tmdb_movie(tmdb_id)
            movie_title = tmdb_movie["title"] if tmdb_movie else "Unknown"
            return jsonify({"movie": movie_title, "recommendations": results, "method": "tmdb_metadata"})
    
    return jsonify({"error": "Invalid request"}), 400


@app.route("/api/popular")
def api_popular():
    """Get popular movies globally or by category."""
    category = request.args.get("category", None)  # hollywood, bollywood, etc.
    limit = request.args.get("limit", 12, type=int)
    try:
        movies = get_popular_movies(limit, category)
        return jsonify({"section": "Popular Titles", "recommendations": movies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/latest")
def api_latest():
    """Get latest movie releases."""
    genre = request.args.get("genre", None)
    limit = request.args.get("limit", 12, type=int)
    try:
        movies = get_latest_movies(limit, genre)
        return jsonify({"section": "Latest Releases", "recommendations": movies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/genre-popular")
def api_genre_popular():
    """Get popular movies from each genre."""
    try:
        _load_models()
        genre_movies = {}
        
        # Get top 3 genres
        top_genres = ["Action", "Drama", "Comedy", "Horror", "Romance", "Thriller"]
        
        for genre in top_genres:
            movies = get_movies_by_genre(genre, limit=8)
            if movies:
                genre_movies[genre] = movies
        
        return jsonify({"section": "By Genre", "genre_collections": genre_movies})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai-recommend", methods=["POST"])
def api_ai_recommend():
    """Fast Gemini-powered recommendations endpoint."""
    data = request.get_json()
    title = data.get("title", "").strip()
    industry = data.get("industry", "Global")
    genre_filter = data.get("genre", None)
    
    if not title or len(title) < 2:
        return jsonify({"error": "Please enter a movie title"}), 400
    
    try:
        # Get AI recommendations from Gemini (parallel)
        ai_recs = get_gemini_recommendations(title, limit=10, industry=industry, genre=genre_filter)
        
        if not ai_recs:
            return jsonify({"error": "Slow network or connection issue. Please try again."}), 404
        
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(fetch_movie_from_tmdb, rec.get("title"), rec.get("tmdb_id"))
                for rec in ai_recs
            ]
            for i, future in enumerate(futures):
                try:
                    movie_data = future.result(timeout=5)
                    if movie_data:
                        movie_data["similarity"] = round(1.0 - (i * 0.08), 2)
                        results.append(movie_data)
                except Exception:
                    pass
        
        return jsonify({
            "movie": title,
            "recommendations": results,
            "method": "gemini_ai",
            "count": len(results)
        })
    
    except Exception as e:
        print(f"AI recommend error: {e}")
        return jsonify({"error": "Network timeout or service unavailable. Please try again in a moment."}), 503


@app.route("/api/health")
def api_health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "version": "2.0",
        "features": ["gemini_ai", "tmdb_integration", "local_similarity", "caching"]
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
