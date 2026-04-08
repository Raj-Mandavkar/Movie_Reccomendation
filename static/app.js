/**
 * CineMatch — Frontend Application Logic
 * Handles autocomplete search, filter controls, and recommendation fetching.
 */

// ── State ──────────────────────────────────────────────────────────
let allMovies = [];
let debounceTimer = null;

// ── DOM References ─────────────────────────────────────────────────
const searchInput = document.getElementById('searchInput');
const autocompleteDD = document.getElementById('autocompleteDropdown');
const genreFilter = document.getElementById('genreFilter');
const ratingFilter = document.getElementById('ratingFilter');
const ratingValue = document.getElementById('ratingValue');
const runtimeFilter = document.getElementById('runtimeFilter');
const runtimeValue = document.getElementById('runtimeValue');
const loadingContainer = document.getElementById('loadingContainer');
const resultsSection = document.getElementById('resultsSection');
const resultsGrid = document.getElementById('resultsGrid');
const resultTitle = document.getElementById('resultTitle');
const resultCount = document.getElementById('resultCount');
const errorContainer = document.getElementById('errorContainer');
const errorMessage = document.getElementById('errorMessage');
const filtersBody = document.getElementById('filtersBody');
const filtersToggle = document.getElementById('filtersToggle');

// ── Init ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    await loadMovies();
    await loadGenres();
    setupEventListeners();
});

async function loadMovies() {
    try {
        const res = await fetch('/api/movies');
        allMovies = await res.json();
    } catch (e) {
        console.error('Failed to load movies:', e);
    }
}

async function loadGenres() {
    try {
        const res = await fetch('/api/genres');
        const genres = await res.json();
        genres.forEach(g => {
            const opt = document.createElement('option');
            opt.value = g;
            opt.textContent = g;
            genreFilter.appendChild(opt);
        });
    } catch (e) {
        console.error('Failed to load genres:', e);
    }
}

// ── Event Listeners ────────────────────────────────────────────────
function setupEventListeners() {
    // Search input → autocomplete
    searchInput.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(showAutocomplete, 150);
    });

    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            autocompleteDD.classList.remove('show');
            getRecommendations();
        }
        if (e.key === 'Escape') {
            autocompleteDD.classList.remove('show');
        }
    });

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.search-container')) {
            autocompleteDD.classList.remove('show');
        }
    });

    // Filter sliders
    ratingFilter.addEventListener('input', () => {
        ratingValue.textContent = ratingFilter.value;
    });

    runtimeFilter.addEventListener('input', () => {
        const val = parseInt(runtimeFilter.value);
        runtimeValue.textContent = val >= 250 ? '250+ min' : `${val} min`;
    });
}

// ── Autocomplete ───────────────────────────────────────────────────
function showAutocomplete() {
    const query = searchInput.value.trim().toLowerCase();

    if (query.length < 2) {
        autocompleteDD.classList.remove('show');
        return;
    }

    // Filter & limit
    const matches = allMovies
        .filter(m => m.title.toLowerCase().includes(query))
        .slice(0, 8);

    if (matches.length === 0) {
        autocompleteDD.classList.remove('show');
        return;
    }

    autocompleteDD.innerHTML = matches.map(m => `
        <div class="ac-item" onclick="selectMovie('${escapeHtml(m.title)}')">
            <span class="ac-item-title">${highlightMatch(m.title, query)}</span>
            <div class="ac-item-meta">
                <span>⭐ ${m.vote_average}</span>
                <span>⏱ ${m.runtime || '—'}m</span>
            </div>
        </div>
    `).join('');

    autocompleteDD.classList.add('show');
}

function selectMovie(title) {
    searchInput.value = title;
    autocompleteDD.classList.remove('show');
    getRecommendations();
}

function highlightMatch(text, query) {
    const idx = text.toLowerCase().indexOf(query);
    if (idx === -1) return escapeHtml(text);
    const before = text.substring(0, idx);
    const match = text.substring(idx, idx + query.length);
    const after = text.substring(idx + query.length);
    return `${escapeHtml(before)}<strong style="color:var(--accent-1)">${escapeHtml(match)}</strong>${escapeHtml(after)}`;
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ── Filters Toggle ─────────────────────────────────────────────────
function toggleFilters() {
    filtersBody.classList.toggle('show');
    filtersToggle.classList.toggle('open');
}

// ── Get Recommendations ────────────────────────────────────────────
async function getRecommendations() {
    const title = searchInput.value.trim();

    // Show loading
    showState('loading');

    const payload = {
        title: title || '',
        genre: genreFilter.value || null,
        min_rating: parseFloat(ratingFilter.value) || 0,
        max_runtime: parseInt(runtimeFilter.value) || 999,
    };

    try {
        const res = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await res.json();

        if (!res.ok || data.error) {
            showError(data.error || 'Something went wrong');
            return;
        }

        const headerLabel = data.movie || data.browse_label || 'Popular';
        const isBrowse = !data.movie;
        renderResults(headerLabel, data.recommendations, isBrowse);
    } catch (e) {
        showError('Network error — is the server running?');
        console.error(e);
    }
}

// ── Render Results ─────────────────────────────────────────────────
function renderResults(headerLabel, recommendations, isBrowse = false) {
    resultTitle.textContent = headerLabel;
    if (isBrowse) {
        document.querySelector('.results-header h2').innerHTML = `Top <span class="results-title" id="resultTitle">${escapeHtml(headerLabel)}</span> Movies`;
        resultCount.textContent = `${recommendations.length} movies found`;
    } else {
        document.querySelector('.results-header h2').innerHTML = `Movies similar to <span class="results-title" id="resultTitle">${escapeHtml(headerLabel)}</span>`;
        resultCount.textContent = `${recommendations.length} similar movies found`;
    }

    resultsGrid.innerHTML = recommendations.map((movie, i) => `
        <div class="movie-card" style="animation-delay: ${i * 0.06}s">
            <div class="card-top">
                <h3 class="card-title">${escapeHtml(movie.title)}</h3>
                <div class="card-rank">#${i + 1}</div>
            </div>
            <div class="card-badges">
                <span class="badge badge-rating">⭐ ${movie.rating}</span>
                <span class="badge badge-runtime">⏱ ${movie.runtime} min</span>
                <span class="badge badge-match">🎯 ${Math.round(movie.similarity * 100)}%</span>
            </div>
            <p class="card-genres">${escapeHtml(movie.genres)}</p>
            <p class="card-overview">${escapeHtml(movie.overview)}</p>
            <div class="card-footer">
                <span>${movie.release_date ? '📅 ' + movie.release_date.substring(0, 4) : ''}</span>
                <span>🔥 ${movie.popularity} popularity</span>
            </div>
        </div>
    `).join('');

    showState('results');
}

// ── State Management ───────────────────────────────────────────────
function showState(state) {
    loadingContainer.style.display = 'none';
    resultsSection.style.display = 'none';
    errorContainer.style.display = 'none';

    switch (state) {
        case 'loading':
            loadingContainer.style.display = 'block';
            break;
        case 'results':
            resultsSection.style.display = 'block';
            break;
        case 'error':
            errorContainer.style.display = 'block';
            break;
    }
}

function showError(msg) {
    errorMessage.textContent = msg;
    showState('error');
}
