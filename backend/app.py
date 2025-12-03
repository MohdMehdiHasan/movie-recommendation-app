from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
import json
import os
import sys

try:
    from flask_cors import CORS
    cors_available = True
except ImportError:
    cors_available = False

try:
    import faiss
    use_faiss = True
except ImportError:
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        use_faiss = False
    except ImportError:
        print("❌ ERROR: Neither FAISS nor sklearn available!")
        sys.exit(1)

app = Flask(__name__, static_folder='.')

# Enable CORS
if cors_available:
    CORS(app)
else:
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

# Safe embedding loader
def safe_load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    try:
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:
            arr = np.vstack(arr)
        return arr.astype("float32")
    except Exception as e:
        raise Exception(f"Error loading embeddings: {str(e)}")

print("\n" + "="*50)
print("Loading Embeddings & Data...")
print("="*50)

try:
    embeddings = safe_load_embeddings("movie_embeddings.npy")
    df = pd.read_pickle("movies_subset.pkl")

    with open("title_to_idx.json", 'r', encoding='utf-8') as f:
        title_to_idx = json.load(f)

    title_to_idx = {k.lower(): v for k, v in title_to_idx.items()}

    print(f"✓ Embeddings Shape: {embeddings.shape}")
    print(f"✓ Movies Loaded: {len(df)}")
    print(f"✓ Title mappings: {len(title_to_idx)}")
    print("="*50 + "\n")

except Exception as e:
    print(f"❌ ERROR loading data: {str(e)}")
    sys.exit(1)

# Normalize embeddings
def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms

embeddings = l2_normalize(embeddings).astype("float32")

# Build FAISS index / fallback to sklearn
index = None
if use_faiss:
    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        print(f"✓ FAISS Index Built: {index.ntotal} vectors")
    except Exception as e:
        print(f"⚠ FAISS failed: {str(e)}")
        use_faiss = False
        from sklearn.metrics.pairwise import cosine_similarity

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

def search_similar(movie_idx, top_k=6):
    if use_faiss and index is not None:
        query = embeddings[movie_idx].reshape(1, -1)
        scores, idxs = index.search(query, top_k + 1)
        return idxs[0][1:].tolist(), scores[0][1:].tolist()
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        query = embeddings[movie_idx].reshape(1, -1)
        sims = cosine_similarity(query, embeddings)[0]
        idxs = np.argsort(sims)[::-1][1: top_k + 1]
        return idxs.tolist(), sims[idxs].tolist()

def find_movie_index(movie_name):
    movie_name = movie_name.lower().strip()
    if movie_name in title_to_idx:
        return title_to_idx[movie_name]

    matches = [t for t in title_to_idx if movie_name in t or t in movie_name]
    if matches:
        return title_to_idx[max(matches, key=len)]

    return None

@app.route("/browse")
def browse():
    try:
        min_rating = float(request.args.get("rating", 0))
        max_runtime = int(request.args.get("runtime", 10000))
        genre_filter = request.args.get("genre", "").lower()

        filtered_df = df.copy()

        if genre_filter:
            filtered_df = filtered_df[filtered_df['genres'].astype(str).str.lower().str.contains(genre_filter, na=False)]

        if min_rating > 0:
            filtered_df = filtered_df[filtered_df['vote_average'] >= min_rating]

        if max_runtime < 10000:
            filtered_df = filtered_df[filtered_df['runtime'] <= max_runtime]

        filtered_df = filtered_df.sort_values('vote_average', ascending=False).head(24)

        results = [
            {
                "title": row["title"],
                "genres": row["genres"],
                "overview": row["overview"],
                "poster": row["poster_path"],
                "rating": float(row["vote_average"]),
                "runtime": int(row["runtime"])
            }
            for _, row in filtered_df.iterrows()
        ]

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recommend")
def recommend():
    movie = request.args.get("movie", "").strip()
    if not movie:
        return jsonify({"error": "Movie name required"}), 400

    movie_idx = find_movie_index(movie)
    if movie_idx is None:
        return jsonify({"error": f"Movie '{movie}' not found"}), 404

    idxs, scores = search_similar(movie_idx, top_k=12)

    results = []
    for idx, score in zip(idxs, scores):
        row = df.iloc[idx]
        results.append({
            "title": row["title"],
            "genres": row["genres"],
            "overview": row["overview"],
            "poster": row["poster_path"],
            "rating": float(row["vote_average"]),
            "runtime": int(row["runtime"]),
            "similarity": float(score)
        })

    return jsonify(results)
@app.route("/trending")
def trending():
    try:
        limit = int(request.args.get("limit", 24))
        min_votes = int(request.args.get("min_votes", 50))

        trending_df = df.copy()

        # Filter by minimum votes
        if "vote_count" in trending_df.columns:
            trending_df = trending_df[trending_df["vote_count"].fillna(0) >= min_votes]

        # Prefer 'popularity' if available
        if "popularity" in trending_df.columns:
            trending_df = trending_df.sort_values("popularity", ascending=False)
        else:
            # Weighted rating if popularity not available
            C = trending_df["vote_average"].mean()
            m = min_votes
            v = trending_df["vote_count"].fillna(0)
            R = trending_df["vote_average"].fillna(C)

            trending_df["score"] = (v/(v+m))*R + (m/(v+m))*C
            trending_df = trending_df.sort_values("score", ascending=False)

        trending_df = trending_df.head(limit)

        results = [
            {
                "title": row["title"],
                "genres": row["genres"],
                "overview": row["overview"],
                "poster": row["poster_path"],
                "rating": float(row["vote_average"]),
                "runtime": int(row["runtime"]),
                "popularity": float(row.get("popularity", 0))
            }
            for _, row in trending_df.iterrows()
        ]

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "movies_loaded": len(df),
        "embeddings_shape": embeddings.shape,
        "faiss_enabled": use_faiss
    })

# ✅ FINAL RENDER-COMPATIBLE SERVER START
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    print("\n" + "="*60)
    print("🚀 Starting Movie Recommendation Server on Render")
    print("="*60)
    print(f"📍 Running on: http://0.0.0.0:{port}")
    print("📡 API endpoints:")
    print("   - /recommend")
    print("   - /browse")
    print("   - /health")
    print("="*60 + "\n")

    app.run(host="0.0.0.0", port=port)
