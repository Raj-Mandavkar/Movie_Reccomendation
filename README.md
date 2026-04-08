# 🎬 CineMatch — AI Movie Recommender

A premium, content-based movie recommendation system built with **Python, Flask, and Scikit-Learn**. Discover similar movies from a dataset of 5,000+ global and Indian titles using AI-powered similarity analysis.

## ✨ Features
- **Similarity Search**: Search any movie (global or Indian) to find 10 similar titles.
- **Filter-Only Browse**: Discover movies by genre, rating, or runtime without a search title.
- **Indian Movie Support**: Curated coverage for Bollywood, Tollywood, Kollywood, and Malayalam cinema.
- **Premium UI**: Dark-mode glassmorphism design with smooth animations and autocomplete search.

---

## 🚀 Local Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. You will also need the TMDB CSV files (`tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`) in the root directory.

### 2. Install Dependencies
```bash
pip install pandas scikit-learn flask numpy
```

### 3. Build the Recommendation Model
This processes the CSV data, extracts features, computes cosine similarity, and saves them as `.pkl` files.
```bash
# Optional: Create the Indian movies dataset first
python create_indian_movies.py

# Rebuild the model (generates movies.pkl and similarity.pkl)
python model_builder.py
```

### 4. Run the Application
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

---

## 🌐 Deployment (Cloud)

To deploy CineMatch online, follow these steps for **Render** or **Railway**:

### 1. Prepare for Deployment
- Ensure you have a `requirements.txt` file (I'll create this for you).
- The `.pkl` files must be uploaded with your code.

### 2. Deploy to Render (Recommended)
1. Push your code to a **GitHub** repository.
2. Create a new "Web Service" on [render.com](https://render.com).
3. Connect your repository.
4. Set **Build Command**: `pip install -r requirements.txt`
5. Set **Start Command**: `python app.py` (or use `gunicorn app:app` for production).
6. Render will provide a public URL for your site!

### 3. Deploy to Railway
1. Install the [Railway CLI](https://railway.app/download).
2. Run `railway up` in the project root.
3. Railway will auto-detect the Flask app and deploy it.

---

## 🛠️ Project Structure
- `app.py`: Flask backend and API endpoints.
- `model_builder.py`: ML pipeline for processing data.
- `create_indian_movies.py`: Generates the curated Indian movie dataset.
- `templates/`: HTML frontend.
- `static/`: CSS styling and JavaScript logic.
- `movies.pkl` / `similarity.pkl`: Pre-computed model data (ignored by git usually, but required for run).
