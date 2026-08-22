# 🎬 Movie Recommendation System Using Machine Learning and Streamlit

A content-based movie recommendation web application built for a Polytechnic AIML internship project. The user selects a movie, and the system recommends the top 5–10 most similar movies using TF-IDF vectorization and cosine similarity.

---

## 📌 Project Objective

Build a web-based system where a user selects/enters a movie name and receives similar movie recommendations based on movie content (genre, overview, cast, director, keywords) — not user ratings.

---

## 🧰 Technology Stack

- Python
- Pandas, NumPy
- Scikit-learn (TF-IDF, Cosine Similarity)
- Streamlit (web interface)
- Pickle (model persistence)

---

## 📂 Project Structure

```
movie-recommendation-system/
│
├── dataset/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── notebook/
│   └── movie_recommendation.ipynb   # exploratory development notebook
│
├── app.py                # Streamlit web application
├── train_model.py        # Standalone training script
├── movie_list.pkl        # Generated: processed movie data
├── similarity.pkl        # Generated: cosine similarity matrix
├── requirements.txt
├── README.md
└── screenshots/
```

**File purposes:**
| File | Purpose |
|---|---|
| `dataset/` | Raw TMDB CSV files |
| `notebook/` | Step-by-step ML development and experimentation |
| `train_model.py` | One-shot script: loads data, cleans it, trains, saves `.pkl` files |
| `app.py` | Streamlit UI that loads `.pkl` files and serves recommendations |
| `movie_list.pkl` | Cleaned movie titles + tags, saved as a dict |
| `similarity.pkl` | Precomputed cosine similarity matrix (movie x movie) |
| `requirements.txt` | Exact Python packages needed |

---

## 📊 Dataset

**TMDB 5000 Movie Dataset** (Kaggle)
- Download: search "TMDB 5000 Movie Dataset" on Kaggle (user: tmdb)
- Files needed: `tmdb_5000_movies.csv`, `tmdb_5000_credits.csv`
- Place both inside the `dataset/` folder

---

## ▶️ How to Run (VS Code)

```bash
# 1. Open project folder in VS Code terminal

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 4. Install requirements
pip install -r requirements.txt

# 5. Place tmdb_5000_movies.csv and tmdb_5000_credits.csv into dataset/

# 6. Train the model
python train_model.py

# 7. Verify movie_list.pkl and similarity.pkl were created

# 8. Run the app
streamlit run app.py

# 9. Open the local URL shown in the terminal (usually http://localhost:8501)
```

---

## 🧠 Machine Learning Approach

**Content-Based Filtering**

1. **Data Loading** — merge movies + credits on `movie_id`/`title`
2. **Preprocessing** — parse genres/cast/crew JSON strings, extract top 3 cast + director, remove spaces in names
3. **Feature Combination** — merge overview + genres + keywords + cast + crew into a single `tags` string
4. **TF-IDF Vectorization** — convert tags into numeric vectors (top 5000 words, English stopwords removed)
5. **Cosine Similarity** — compute similarity between every pair of movies
6. **Recommendation** — for a selected movie, return the top N movies with highest similarity score

---

## ⚠️ Error Handling

The app handles:
- Missing dataset files (`train_model.py` stops with a clear message)
- Missing `.pkl` model files (`app.py` shows a Streamlit error instead of crashing)
- Movie not found in database (friendly warning, no crash)
- No movie selected (warning prompt)

---

## 🚀 Future Scope

- Add collaborative filtering using user ratings
- Include movie posters via an external API
- Deploy on Streamlit Cloud for public access
- Add genre/year filters

---

## 👨‍🎓 Developer

Polytechnic AIML 3rd Year — Internship Project
