"""
train_model.py
----------------
Movie Recommendation System - Model Training Script

This script:
1. Loads the TMDB 5000 dataset (movies + credits)
2. Cleans and preprocesses the data
3. Builds a combined "tags" feature for each movie
4. Converts tags to TF-IDF vectors
5. Computes cosine similarity between all movies
6. Saves movie_list.pkl and similarity.pkl for app.py to use

Run this with:  python train_model.py
"""

import ast
import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MOVIES_CSV = os.path.join("dataset", "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join("dataset", "tmdb_5000_credits.csv")
MOVIE_LIST_OUT = "movie_list.pkl"
SIMILARITY_OUT = "similarity.pkl"


def check_files_exist():
    """Make sure the dataset files exist before doing anything else."""
    missing = []
    if not os.path.exists(MOVIES_CSV):
        missing.append(MOVIES_CSV)
    if not os.path.exists(CREDITS_CSV):
        missing.append(CREDITS_CSV)
    if missing:
        print("ERROR: The following required dataset file(s) are missing:")
        for m in missing:
            print(f"   - {m}")
        print("\nPlease download the TMDB 5000 Movie Dataset from Kaggle and place")
        print("'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' inside the 'dataset/' folder.")
        raise SystemExit(1)


def load_data():
    print("Step 1/6: Loading dataset...")
    movies = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)
    movies = movies.merge(credits, on="title")
    print(f"   Loaded and merged. Shape: {movies.shape}")
    return movies


def convert_json_list(text):
    """Convert a stringified list-of-dicts (e.g. genres) into a plain list of names."""
    try:
        return [item["name"] for item in ast.literal_eval(text)]
    except (ValueError, SyntaxError):
        return []


def convert_cast(text, top_n=3):
    """Extract only the top N cast members."""
    try:
        L = []
        for i, item in enumerate(ast.literal_eval(text)):
            if i >= top_n:
                break
            L.append(item["name"])
        return L
    except (ValueError, SyntaxError):
        return []


def fetch_director(text):
    """Extract the director's name from the crew list."""
    try:
        for item in ast.literal_eval(text):
            if item.get("job") == "Director":
                return [item["name"]]
        return []
    except (ValueError, SyntaxError):
        return []


def preprocess(movies):
    print("Step 2/6: Preprocessing data...")
    movies = movies[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
    movies = movies.dropna().copy()

    movies["genres"] = movies["genres"].apply(convert_json_list)
    movies["keywords"] = movies["keywords"].apply(convert_json_list)
    movies["cast"] = movies["cast"].apply(convert_cast)
    movies["crew"] = movies["crew"].apply(fetch_director)

    # remove spaces inside names so "Sam Worthington" -> "SamWorthington"
    for col in ["genres", "keywords", "cast", "crew"]:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    movies["overview"] = movies["overview"].apply(lambda x: x.split())

    movies["tags"] = (
        movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
    )

    new_df = movies[["movie_id", "title", "tags"]].copy()
    new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())
    new_df.reset_index(drop=True, inplace=True)

    print(f"   Preprocessing complete. Final movie count: {len(new_df)}")
    return new_df


def vectorize_and_compute_similarity(new_df):
    print("Step 3/6: Vectorizing text with TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    vectors = tfidf.fit_transform(new_df["tags"]).toarray()
    print(f"   Vector shape: {vectors.shape}")

    print("Step 4/6: Computing cosine similarity matrix...")
    similarity = cosine_similarity(vectors)
    print(f"   Similarity matrix shape: {similarity.shape}")

    return similarity


def save_outputs(new_df, similarity):
    print("Step 5/6: Saving model files...")
    pickle.dump(new_df.to_dict(), open(MOVIE_LIST_OUT, "wb"))
    pickle.dump(similarity, open(SIMILARITY_OUT, "wb"))

    size_ml = os.path.getsize(MOVIE_LIST_OUT) / (1024 * 1024)
    size_sim = os.path.getsize(SIMILARITY_OUT) / (1024 * 1024)
    print(f"   Saved '{MOVIE_LIST_OUT}' ({size_ml:.2f} MB)")
    print(f"   Saved '{SIMILARITY_OUT}' ({size_sim:.2f} MB)")


def sanity_check(new_df, similarity):
    print("Step 6/6: Running sanity check...")
    test_title = new_df.iloc[0]["title"]
    scores = sorted(list(enumerate(similarity[0])), key=lambda x: x[1], reverse=True)[1:4]
    print(f"   Test movie: '{test_title}'")
    print("   Top 3 similar movies:")
    for idx, score in scores:
        print(f"     - {new_df.iloc[idx]['title']} (score: {score:.3f})")


if __name__ == "__main__":
    print("=" * 55)
    print(" MOVIE RECOMMENDATION SYSTEM - MODEL TRAINING")
    print("=" * 55)

    check_files_exist()
    raw_movies = load_data()
    processed_df = preprocess(raw_movies)
    sim_matrix = vectorize_and_compute_similarity(processed_df)
    save_outputs(processed_df, sim_matrix)
    sanity_check(processed_df, sim_matrix)

    print("\nTRAINING COMPLETE. You can now run: streamlit run app.py")
