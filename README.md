# 🎬 Movie Recommendation System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-F7931E?style=flat-square&logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A content-based movie recommendation system built with TF-IDF vectorization and cosine similarity, served through a Streamlit web UI with a dark cinema-themed design.

---

## 📸 Demo

> Search by movie title or describe the kind of film you're in the mood for — and get instant recommendations.

| Search by Title | Search by Description |
|---|---|
| Select any of 4,800+ movies | Type a mood or theme |
| View genre, rating & tagline | Matches via TF-IDF pipeline |
| See similarity % per result | Top N configurable results |

---

## 🚀 Features

- **Content-based filtering** — recommendations driven by movie overview, genres, and tagline
- **Two search modes**
  - 🎥 Pick a movie title from a dropdown of 4,803 films
  - 🔍 Describe what you want in plain text
- **Similarity scoring** — each result shows a % match based on cosine similarity
- **Rich movie cards** — genre badges, star ratings, popularity score, tagline, and overview snippet
- **Configurable results** — sidebar slider to choose 5–20 recommendations
- **Dataset stats** — total movies, unique genres, and average rating in the sidebar

---

## 🧠 How It Works

```
Raw CSV Data
    │
    ▼
Feature Engineering
  overview + genres + tagline  →  "tags" column
    │
    ▼
Text Preprocessing (NLTK)
  lowercase → remove punctuation → remove stopwords → lemmatize
    │
    ▼
TF-IDF Vectorization
  10,000 features, bigrams (1,2), English stopwords
    │
    ▼
Cosine Similarity
  Query vector vs. all 4,803 movie vectors
    │
    ▼
Top-N Recommendations
```

---

## 📁 Project Structure

```
moviesdata/
├── Movies.ipynb          # Model training notebook (run on Google Colab)
├── app.py                # Streamlit web UI
├── requirements.txt      # Python dependencies
├── moviesds.csv          # Source dataset (TMDB 5000 movies)
├── df.pickle             # Preprocessed DataFrame
├── tfidf.pkl             # Fitted TF-IDF vectorizer
├── tfidf_matrix.pkl      # TF-IDF document matrix (4803 × 10000)
└── indices.pkl           # Title → index mapping
```

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/Arjun-Ghungrud/CineSphere.git
cd CineSphere
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate the model artifacts (if not present)

Open `Movies.ipynb` in Jupyter or Google Colab and run all cells. This will produce:

```
df.pickle
tfidf.pkl
tfidf_matrix.pkl
indices.pkl
```

Place all four files in the project root alongside `app.py`.

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.35 | Web UI framework |
| `pandas` | ≥ 2.2 | Data manipulation |
| `numpy` | ≥ 1.26 | Numerical operations |
| `scikit-learn` | ≥ 1.5 | TF-IDF & cosine similarity |
| `nltk` | ≥ 3.8 | Text preprocessing |

---

## 📊 Dataset

Based on the **TMDB 5000 Movie Dataset** from Kaggle.

| Stat | Value |
|---|---|
| Total movies | 4,803 |
| Features used | `overview`, `genres`, `tagline` |
| TF-IDF features | 10,000 |
| N-gram range | (1, 2) — unigrams + bigrams |

---

## 🔬 Model Details

**Vectorization**
- `TfidfVectorizer` with `max_features=10000` and `ngram_range=(1,2)`
- English stopwords removed at both preprocessing and vectorization stages
- Additional preprocessing: lowercasing, regex cleaning, NLTK stopword removal, WordNet lemmatization

**Similarity**
- Pairwise cosine similarity between the query vector and all document vectors
- Scores range from 0 (no overlap) to 1 (identical)
- Top-N results returned by descending score, excluding the query movie itself

---

## 🗺️ Roadmap

- [ ] Add TMDB poster images via API
- [ ] Add cast & director-based filtering
- [ ] Deploy to Streamlit Community Cloud
- [ ] Add collaborative filtering layer
- [ ] Support multi-movie "blend" recommendations

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/cool-feature`)
3. Commit your changes (`git commit -m 'Add cool feature'`)
4. Push to the branch (`git push origin feature/cool-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) — Kaggle
- [Streamlit](https://streamlit.io/) — for the web framework
- [scikit-learn](https://scikit-learn.org/) — for TF-IDF and cosine similarity

---

<p align="center">Made with Python</p>
