"""
app.py
-------
Movie Recommendation System - Streamlit Web Application

Run with:  streamlit run app.py
"""

import os
import pickle

import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

MOVIE_LIST_PATH = "movie_list.pkl"
SIMILARITY_PATH = "similarity.pkl"


# ---------------------------------------------------------
# Load model files (cached so it only loads once per session)
# ---------------------------------------------------------
@st.cache_data
def load_model():
    if not os.path.exists(MOVIE_LIST_PATH) or not os.path.exists(SIMILARITY_PATH):
        return None, None
    movie_dict = pickle.load(open(MOVIE_LIST_PATH, "rb"))
    movies_df = pd.DataFrame(movie_dict)
    similarity_matrix = pickle.load(open(SIMILARITY_PATH, "rb"))
    return movies_df, similarity_matrix


movies_df, similarity = load_model()

# ---------------------------------------------------------
# Handle missing model files gracefully
# ---------------------------------------------------------
if movies_df is None or similarity is None:
    st.error(
        "Model files not found. Please run `python train_model.py` first "
        "to generate 'movie_list.pkl' and 'similarity.pkl'."
    )
    st.stop()


# ---------------------------------------------------------
# Recommendation function
# ---------------------------------------------------------
def recommend(movie_title, top_n=5):
    titles_lower = movies_df["title"].str.lower()
    query = movie_title.lower().strip()

    if query not in titles_lower.values:
        return None

    movie_index = movies_df[titles_lower == query].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(
        list(enumerate(distances)), key=lambda x: x[1], reverse=True
    )[1: top_n + 1]

    results = []
    for idx, score in movie_list:
        results.append({"title": movies_df.iloc[idx]["title"], "score": round(float(score), 3)})
    return results


# ---------------------------------------------------------
# Sidebar - About / How it works / Developer info
# ---------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About This Project")
    st.write(
        "A content-based Movie Recommendation System built using "
        "Machine Learning (TF-IDF + Cosine Similarity) and Streamlit."
    )

    st.subheader("⚙️ How It Works")
    st.write(
        """
        1. Each movie's overview, genre, cast, and director are combined into a single text feature.
        2. This text is converted into numeric vectors using TF-IDF.
        3. Cosine similarity is calculated between all movie vectors.
        4. The movies most similar to your selected movie are recommended.
        """
    )

    st.subheader("👨‍🎓 Developer")
    st.write("Polytechnic AIML Internship Project")
    st.write(f"Total movies in database: **{len(movies_df)}**")


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🎬 Movie Recommendation System")
st.markdown(
    "##### A Machine Learning based application that recommends movies "
    "similar to the one you select, using content-based filtering."
)
st.divider()


# ---------------------------------------------------------
# Main input section
# ---------------------------------------------------------
col1, col2 = st.columns([3, 1])

with col1:
    movie_names = movies_df["title"].values
    selected_movie = st.selectbox(
        "Select or search a movie:",
        options=movie_names,
        index=None,
        placeholder="Type to search a movie...",
    )

with col2:
    num_recommendations = st.selectbox("Number of recommendations:", [5, 10], index=0)

button_col1, button_col2 = st.columns([1, 1])
with button_col1:
    recommend_clicked = st.button("🔍 Recommend Movies", type="primary", use_container_width=True)
with button_col2:
    reset_clicked = st.button("🔄 Reset", use_container_width=True)

if reset_clicked:
    st.rerun()

st.divider()

# ---------------------------------------------------------
# Handle recommendation logic + error handling
# ---------------------------------------------------------
if recommend_clicked:
    if not selected_movie:
        st.warning("⚠️ Please select a movie before clicking Recommend.")
    else:
        with st.spinner("Finding similar movies..."):
            results = recommend(selected_movie, top_n=num_recommendations)

        if results is None:
            st.error(
                f"❌ Sorry, '{selected_movie}' was not found in our database. "
                "Please try selecting a different movie."
            )
        else:
            st.subheader(f"✨ Recommended Movies for You (based on '{selected_movie}')")

            cols = st.columns(5)
            for i, movie in enumerate(results):
                with cols[i % 5]:
                    st.markdown(
                        f"""
                        <div style="
                            background-color:#1e1e2f;
                            padding:16px;
                            border-radius:12px;
                            text-align:center;
                            margin-bottom:12px;
                            min-height:130px;
                        ">
                            <h5 style="color:white; margin-bottom:8px;">{i + 1}. {movie['title']}</h5>
                            <p style="color:#9ad1ff; font-size:14px;">Similarity: {movie['score']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.divider()
            st.subheader("📋 Recommended Movies (List View)")
            for i, movie in enumerate(results, start=1):
                st.write(f"**{i}. {movie['title']}** — similarity score: `{movie['score']}`")

st.divider()
st.caption("Built with Python, Scikit-learn, and Streamlit | Content-Based Recommendation System")
