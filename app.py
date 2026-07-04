import streamlit as st
import pickle
import pandas as pd
import ast
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineSphere",
    page_icon="🎬",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #f0f0f0;
    }
    .hero {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .hero p { color: #aaa; font-size: 1.1rem; }
    .movie-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(255,210,0,0.15);
    }
    .movie-rank { font-size: 0.75rem; color: #ffd200; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.2rem; }
    .movie-title { font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 0.3rem; }
    .movie-meta { font-size: 0.82rem; color: #bbb; }
    .genre-badge {
        display: inline-block;
        background: rgba(247,151,30,0.18);
        border: 1px solid rgba(247,151,30,0.4);
        color: #f7971e;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        margin: 2px 3px 2px 0;
        font-weight: 600;
    }
    .star { color: #ffd200; }
    label { color: #ddd !important; font-size: 1rem !important; }
    div[data-baseweb="select"] { border-radius: 10px !important; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        color: #1a1a2e;
        font-weight: 700;
        font-size: 1rem;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1rem;
    }
    hr { border-color: rgba(255,255,255,0.1); }
    section[data-testid="stSidebar"] { background: rgba(15,12,41,0.9); }
</style>
""", unsafe_allow_html=True)


# ── NLTK setup ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    return set(stopwords.words("english")), WordNetLemmatizer()


# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open("indices.pkl", "rb") as f:
        indices = pickle.load(f)
    with open("df.pickle", "rb") as f:
        df = pickle.load(f)
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return tfidf_matrix, indices, df, tfidf


# ── Recommendation functions ──────────────────────────────────────────────────
def recommend(title, df, tfidf_matrix, indices, n=10):
    if title not in indices:
        return pd.DataFrame()
    idx = indices[title]
    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][1:n + 1]
    result = df.iloc[similar_idx][["title", "genres", "vote_average", "popularity", "overview", "tagline"]].copy()
    result["similarity"] = sim_scores[similar_idx]
    return result.reset_index(drop=True)


def preprocess_and_recommend(query_text, df, tfidf, tfidf_matrix, n=10):
    stop_words, lemmatizer = load_nltk()
    text = re.sub(r"[^a-zA-Z\s]", "", query_text.lower())
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    query_vec = tfidf.transform([" ".join(words)])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    similar_idx = sim_scores.argsort()[::-1][:n]
    result = df.iloc[similar_idx][["title", "genres", "vote_average", "popularity", "overview", "tagline"]].copy()
    result["similarity"] = sim_scores[similar_idx]
    return result.reset_index(drop=True)


def render_card(rank, row):
    genres_html = "".join(f'<span class="genre-badge">{g}</span>' for g in str(row["genres"]).split())
    filled = int(round(row["vote_average"] / 2))
    stars = ("&#9733;" * filled) + ("&#9734;" * (5 - filled))
    overview = str(row["overview"])[:160] + "..." if len(str(row["overview"])) > 160 else str(row["overview"])
    tagline = f'<em>"{row["tagline"]}"</em>' if row["tagline"] else ""
    sim_pct = f'{row["similarity"] * 100:.1f}% match' if row["similarity"] > 0 else ""

    st.markdown(f"""
    <div class="movie-card">
        <div class="movie-rank">#{rank} &nbsp; {sim_pct}</div>
        <div class="movie-title">{row["title"]}</div>
        <div class="movie-meta">
            <span class="star">{stars}</span> &nbsp; {row["vote_average"]}/10
            &nbsp;&middot;&nbsp; Popularity: {row["popularity"]:.1f}
        </div>
        <div style="margin: 0.5rem 0;">{genres_html}</div>
        <div class="movie-meta">{tagline}</div>
        <div class="movie-meta" style="margin-top:0.4rem; color:#999;">{overview}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    tfidf_matrix, indices, df, tfidf = load_artifacts()
    movie_list = sorted(df["title"].dropna().unique().tolist())

    st.markdown("""
    <div class="hero">
        <h1>&#127909; CineSphere</h1>
        <p>Discover films similar to what you love &mdash; powered by TF-IDF &amp; cosine similarity</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    with st.sidebar:
        st.markdown("### Settings")
        n_results = st.slider("Number of recommendations", 5, 20, 10)
        st.markdown("---")
        st.markdown("### Dataset stats")
        st.metric("Total movies", f"{len(df):,}")
        st.metric("Unique genres", df["genres"].str.split().explode().nunique())
        avg_rating = df["vote_average"][df["vote_average"] > 0].mean()
        st.metric("Avg. rating", f"{avg_rating:.2f} / 10")
        st.markdown("---")
        st.markdown("<small style='color:#888;'>TF-IDF (10k features, bigrams)<br>+ Cosine Similarity</small>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Search by Movie Title", "Search by Description"])

    with tab1:
        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.selectbox("Pick a movie you like:", options=[""] + movie_list, index=0)
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("Get Recommendations", key="btn_title")

        if search_clicked or selected:
            if not selected:
                st.warning("Please select a movie first.")
            else:
                row = df[df["title"] == selected].iloc[0]
                with st.expander(f"About: {selected}", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rating", f"{row['vote_average']}/10")
                    c2.metric("Popularity", f"{row['popularity']:.1f}")
                    c3.metric("Genres", str(row["genres"])[:30])
                    if row["overview"]:
                        st.markdown(f"<small style='color:#bbb;'>{row['overview']}</small>", unsafe_allow_html=True)

                st.markdown(f"#### Movies similar to *{selected}*")
                results = recommend(selected, df, tfidf_matrix, indices, n=n_results)
                if results.empty:
                    st.error("No recommendations found.")
                else:
                    for i, row_r in results.iterrows():
                        render_card(i + 1, row_r)

    with tab2:
        query = st.text_area("Describe what you're in the mood for:", placeholder="e.g. A thrilling space adventure with action and stunning visuals", height=100)
        if st.button("Find Movies", key="btn_desc"):
            if not query.strip():
                st.warning("Please enter a description.")
            else:
                st.markdown("#### Top picks for your mood")
                results = preprocess_and_recommend(query, df, tfidf, tfidf_matrix, n=n_results)
                if results.empty:
                    st.error("Could not find matching movies.")
                else:
                    for i, row_r in results.iterrows():
                        render_card(i + 1, row_r)


if __name__ == "__main__":
    main()
