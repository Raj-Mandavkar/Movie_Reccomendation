/**
 * CineMatch Frontend
 * Simple, clean JavaScript for movie recommendations
 */

// ─────────────────────────────────────────────────────────────────
// GLOBAL STATE
// ─────────────────────────────────────────────────────────────────

window.appState = {
  allMovies: [],
  debounceTimer: null,
  selectedTmdbId: null,
  currentTheme: 'dark',
  selectedIndustry: 'Global'
};

// ─────────────────────────────────────────────────────────────────
// DOM SELECTORS (lazy-loaded)
// ─────────────────────────────────────────────────────────────────

function getDOM(id) {
  return document.getElementById(id);
}

function getElement(selector) {
  return document.querySelector(selector);
}

// ─────────────────────────────────────────────────────────────────
// THEME MANAGEMENT
// ─────────────────────────────────────────────────────────────────

window.toggleTheme = function() {
  const html = document.documentElement;
  const currentTheme = html.getAttribute('data-theme');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  
  html.setAttribute('data-theme', newTheme);
  localStorage.setItem('theme', newTheme);
  appState.currentTheme = newTheme;
  
  const themeToggle = getDOM('themeToggle');
  if (themeToggle) {
    const icon = themeToggle.querySelector('.theme-icon');
    if (icon) {
      icon.textContent = newTheme === 'dark' ? '☀️' : '🌙';
    }
  }
};

window.initTheme = function() {
  const savedTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  appState.currentTheme = savedTheme;
  
  const themeToggle = getDOM('themeToggle');
  if (themeToggle) {
    const icon = themeToggle.querySelector('.theme-icon');
    if (icon) {
      icon.textContent = savedTheme === 'dark' ? '☀️' : '🌙';
    }
  }
};

// ─────────────────────────────────────────────────────────────────
// AUTOCOMPLETE & SEARCH
// ─────────────────────────────────────────────────────────────────

window.showAutocomplete = async function() {
  const searchInput = getDOM('searchInput');
  const autocompleteDD = getDOM('autocompleteDropdown');
  
  if (!searchInput || !autocompleteDD) return;
  
  const query = searchInput.value.trim().toLowerCase();
  
  if (query.length < 2) {
    autocompleteDD.classList.remove('show');
    return;
  }
  
  // Local matches
  const localMatches = appState.allMovies
    .filter(m => m.title.toLowerCase().includes(query))
    .slice(0, 5);
  
  // TMDB matches
  let tmdbMatches = [];
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    tmdbMatches = await res.json();
  } catch (e) {
    console.error('Search error:', e);
  }
  
  const allMatches = [...localMatches, ...tmdbMatches];
  
  if (allMatches.length === 0) {
    autocompleteDD.classList.remove('show');
    return;
  }
  
  let html = '';
  
  // Local matches
  html += localMatches.map(m => `
    <div class="ac-item" onclick="window.selectMovie('${escapeHtml(m.title)}')">
      <span class="ac-item-title">${m.title}</span>
      <div class="ac-item-meta">
        <span>⭐ ${m.vote_average || '—'}</span>
      </div>
    </div>
  `).join('');
  
  // TMDB matches
  if (tmdbMatches.length > 0) {
    if (html) html += '<div class="ac-divider">— More Results —</div>';
    html += tmdbMatches.map(m => `
      <div class="ac-item" onclick="window.selectMovie('${escapeHtml(m.title)}', ${m.id})">
        <span class="ac-item-title">${m.title}</span>
        <div class="ac-item-meta">
          <span>⭐ ${m.vote_average || '—'}</span>
        </div>
      </div>
    `).join('');
  }
  
  autocompleteDD.innerHTML = html;
  autocompleteDD.classList.add('show');
};

window.selectMovie = function(title, tmdbId = null) {
  const searchInput = getDOM('searchInput');
  const autocompleteDD = getDOM('autocompleteDropdown');
  
  if (searchInput) searchInput.value = title;
  appState.selectedTmdbId = tmdbId;
  
  if (autocompleteDD) autocompleteDD.classList.remove('show');
  
  window.getRecommendations();
};

window.escapeHtml = function(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
};

// ─────────────────────────────────────────────────────────────────
// RECOMMENDATIONS
// ─────────────────────────────────────────────────────────────────

window.getRecommendations = async function() {
  const searchInput = getDOM('searchInput');
  const genreFilter = getDOM('genreFilter');
  const ratingFilter = getDOM('ratingFilter');
  const runtimeFilter = getDOM('runtimeFilter');
  const industryGroup = getDOM('industryGroup');
  const loadingContainer = getDOM('loadingContainer');
  const resultsSection = getDOM('resultsSection');
  const homeSections = getDOM('homeSections');
  
  // Show loading
  if (loadingContainer) loadingContainer.style.display = 'block';
  if (resultsSection) resultsSection.style.display = 'none';
  if (homeSections) homeSections.style.display = 'none';
  
  const title = searchInput ? searchInput.value.trim() : '';
  
  // Show industry toggle only when searching a movie (not genre-only)
  if (industryGroup) {
    if (title && title.length > 0) {
      industryGroup.classList.remove('hidden');
    } else {
      industryGroup.classList.add('hidden');
      appState.selectedIndustry = 'Global';
      const industryBtns = document.querySelectorAll('.industry-btn');
      industryBtns.forEach(b => b.classList.remove('active'));
      industryBtns[0].classList.add('active');
    }
  }
  
  const payload = {
    title: title || '',
    tmdb_id: appState.selectedTmdbId || null,
    genre: genreFilter ? genreFilter.value || null : null,
    min_rating: ratingFilter ? parseFloat(ratingFilter.value) || 0 : 0,
    industry: appState.selectedIndustry,
    max_runtime: runtimeFilter ? parseInt(runtimeFilter.value) || 999 : 999,
  };
  
  try {
    const res = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    
    const data = await res.json();
    
    if (!res.ok || data.error) {
      window.showError(data.error || 'Something went wrong');
      return;
    }
    
    const headerLabel = data.movie || data.browse_label || 'Popular';
    let resultsToShow = data.recommendations || [];
    
    if (data.searched_movie) {
      resultsToShow = [data.searched_movie, ...resultsToShow];
    }
    
    window.renderResults(headerLabel, resultsToShow);
  } catch (e) {
    console.error('Recommendation error:', e);
    window.showError('Network error — is the server running?');
  }
};

window.renderResults = function(headerLabel, recommendations) {
  const resultsSection = getDOM('resultsSection');
  const resultsGrid = getDOM('resultsGrid');
  const resultTitle = getDOM('resultTitle');
  const resultCount = getDOM('resultCount');
  const loadingContainer = getDOM('loadingContainer');
  const homeSections = getDOM('homeSections');
  const resultsHeader = getElement('.results-header h2');
  
  if (loadingContainer) loadingContainer.style.display = 'none';
  
  if (resultsHeader) {
    // Always show "Movies Related to [Movie/Genre Name]"
    resultsHeader.innerHTML = `Movies Related to <span class="results-title">${window.escapeHtml(headerLabel)}</span>`;
  }
  
  if (resultCount) resultCount.textContent = `${recommendations.length} related movies found`;
  
  if (resultsGrid) {
    resultsGrid.innerHTML = recommendations.map((movie, i) => {
      return window.renderPortraitCard(movie, i);
    }).join('');
  }
  
  if (resultsSection) resultsSection.style.display = 'block';
  if (homeSections) homeSections.style.display = 'none';
};

window.renderPortraitCard = function(movie, idx) {
  const rating = movie.rating || movie.vote_average || '—';
  const year = movie.release_date ? movie.release_date.substring(0, 4) : '—';
  const poster = movie.poster || 'data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 200 300%22%3E%3Crect fill=%22%23333%22 width=%22200%22 height=%22300%22/%3E%3C/svg%3E';
  
  return `
    <div class="movie-card" style="animation-delay: ${idx * 0.05}s">
      <img src="${poster}" alt="${window.escapeHtml(movie.title)}" class="card-poster"
        onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 200 300%22%3E%3Crect fill=%22%23333%22 width=%22200%22 height=%22300%22/%3E%3C/svg%3E'">
      <div class="card-overlay">
        <div class="card-title">${window.escapeHtml(movie.title)}</div>
        <div class="card-info">
          <span>⭐ ${rating}</span>
          <span>📅 ${year}</span>
        </div>
      </div>
    </div>
  `;
};

window.showError = function(msg) {
  const errorContainer = getDOM('errorContainer');
  const errorMessage = getDOM('errorMessage');
  const loadingContainer = getDOM('loadingContainer');
  const resultsSection = getDOM('resultsSection');
  const homeSections = getDOM('homeSections');
  
  if (loadingContainer) loadingContainer.style.display = 'none';
  if (resultsSection) resultsSection.style.display = 'none';
  if (homeSections) homeSections.style.display = 'none';
  
  if (errorMessage) errorMessage.textContent = msg;
  if (errorContainer) errorContainer.style.display = 'block';
};

// ─────────────────────────────────────────────────────────────────
// FILTERS & UI
// ─────────────────────────────────────────────────────────────────

window.toggleFilters = function() {
  const filtersBody = getDOM('filtersBody');
  const filtersToggle = getDOM('filtersToggle');
  
  if (filtersBody) filtersBody.classList.toggle('show');
  if (filtersToggle) filtersToggle.classList.toggle('open');
};

// ─────────────────────────────────────────────────────────────────
// HOME PAGE SECTIONS
// ─────────────────────────────────────────────────────────────────

window.loadHomePageSections = async function() {
  try {
    const [popularRes, latestRes, genreRes] = await Promise.all([
      fetch('/api/popular?limit=12'),
      fetch('/api/latest?limit=12'),
      fetch('/api/genre-popular')
    ]);
    
    const popularData = await popularRes.json();
    const latestData = await latestRes.json();
    const genreData = await genreRes.json();
    
    window.renderHomeSections(popularData, latestData, genreData);
  } catch (e) {
    console.error('Failed to load home sections:', e);
  }
};

window.renderHomeSections = function(popularData, latestData, genreData) {
  const homeSections = getDOM('homeSections');
  if (!homeSections) return;
  
  let html = '';
  
  if (popularData.recommendations && popularData.recommendations.length > 0) {
    html += window.renderMovieSection('🔥 Popular Titles', popularData.recommendations, 'popular');
  }
  
  if (latestData.recommendations && latestData.recommendations.length > 0) {
    html += window.renderMovieSection('🆕 Latest Releases', latestData.recommendations, 'latest');
  }
  
  if (genreData.genre_collections) {
    for (const [genre, movies] of Object.entries(genreData.genre_collections)) {
      html += window.renderMovieSection(`🎬 ${genre}`, movies, genre.toLowerCase());
    }
  }
  
  homeSections.innerHTML = html;
};

window.renderMovieSection = function(title, movies, id) {
  const scrollerId = `scroller-${id}`;
  let html = `
    <section class="section-block">
      <div class="section-header">
        <h2>${title}</h2>
        <a href="#" class="view-all">View All</a>
      </div>
      <div class="horizontal-scroller" id="${scrollerId}">
  `;
  
  movies.forEach((movie, idx) => {
    html += window.renderPortraitCard(movie, idx);
  });
  
  html += `
      </div>
    </section>
  `;
  return html;
};

// ─────────────────────────────────────────────────────────────────
// INITIALIZATION
// ─────────────────────────────────────────────────────────────────

window.loadMovies = async function() {
  try {
    const res = await fetch('/api/movies');
    appState.allMovies = await res.json();
  } catch (e) {
    console.error('Failed to load movies:', e);
  }
};

window.loadGenres = async function() {
  try {
    const res = await fetch('/api/genres');
    const genres = await res.json();
    const genreFilter = getDOM('genreFilter');
    if (genreFilter) {
      genres.forEach(g => {
        const opt = document.createElement('option');
        opt.value = g;
        opt.textContent = g;
        genreFilter.appendChild(opt);
      });
    }
  } catch (e) {
    console.error('Failed to load genres:', e);
  }
};

window.setupEventListeners = function() {
  const themeToggle = getDOM('themeToggle');
  const searchInput = getDOM('searchInput');
  const filtersHeader = getElement('.filters-header');
  const ratingFilter = getDOM('ratingFilter');
  const runtimeFilter = getDOM('runtimeFilter');
  const ratingValue = getDOM('ratingValue');
  const runtimeValue = getDOM('runtimeValue');
  
  if (themeToggle) {
    themeToggle.addEventListener('click', window.toggleTheme);
  }
  
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      clearTimeout(appState.debounceTimer);
      appState.debounceTimer = setTimeout(window.showAutocomplete, 150);
    });
    
    searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        const autocompleteDD = getDOM('autocompleteDropdown');
        if (autocompleteDD) autocompleteDD.classList.remove('show');
        window.getRecommendations();
      }
    });
  }
  
  if (filtersHeader) {
    filtersHeader.addEventListener('click', window.toggleFilters);
  }
  
  if (ratingFilter) {
    ratingFilter.addEventListener('input', () => {
      if (ratingValue) ratingValue.textContent = ratingFilter.value;
    });
  }
  
  if (runtimeFilter) {
    runtimeFilter.addEventListener('input', () => {
      const val = parseInt(runtimeFilter.value);
      if (runtimeValue) {
        runtimeValue.textContent = val >= 250 ? '250+ min' : `${val} min`;
      }
    });
  }
  
  // Industry toggle buttons
  const industryBtns = document.querySelectorAll('.industry-btn');
  industryBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      industryBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      appState.selectedIndustry = btn.dataset.value;
    });
  });
};

// ─────────────────────────────────────────────────────────────────
// MAIN INITIALIZATION
// ─────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  window.initTheme();
  await window.loadMovies();
  await window.loadGenres();
  window.setupEventListeners();
  await window.loadHomePageSections();
  
  const homeSections = getDOM('homeSections');
  if (homeSections) homeSections.style.display = 'block';
});
