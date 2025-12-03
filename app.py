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

if cors_available:
    CORS(app)
else:
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

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
    
except FileNotFoundError as e:
    print(f"❌ ERROR: {str(e)}")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR loading data: {str(e)}")
    sys.exit(1)

def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return x / norms

try:
    embeddings = l2_normalize(embeddings).astype("float32")
except Exception as e:
    print(f"❌ ERROR normalizing embeddings: {str(e)}")
    sys.exit(1)

index = None
if use_faiss:
    try:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        print(f"✓ FAISS Index Built: {index.ntotal} vectors")
    except Exception as e:
        print(f"⚠ FAISS index build failed: {str(e)}")
        use_faiss = False
        from sklearn.metrics.pairwise import cosine_similarity

@app.route("/")
def home():
    try:
        return send_from_directory(".", "index.html")
    except Exception as e:
        return jsonify({"error": f"Error serving frontend: {str(e)}"}), 500

def search_similar(movie_idx, top_k=6):
    try:
        if movie_idx < 0 or movie_idx >= len(embeddings):
            raise ValueError(f"Invalid movie index: {movie_idx}")
        
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
    except Exception as e:
        print(f"Error in search_similar: {str(e)}")
        raise

def find_movie_index(movie_name):
    movie_name = movie_name.lower().strip()
    
    if movie_name in title_to_idx:
        return title_to_idx[movie_name]
    
    matches = [t for t in title_to_idx.keys() if movie_name in t or t in movie_name]
    if matches:
        best_match = max(matches, key=len)
        return title_to_idx[best_match]
    
    movie_words = set(movie_name.split())
    best_score = 0
    best_match = None
    
    for title in title_to_idx.keys():
        title_words = set(title.split())
        common_words = movie_words.intersection(title_words)
        if common_words:
            score = len(common_words) / max(len(movie_words), len(title_words))
            if score > best_score:
                best_score = score
                best_match = title
    
    if best_match and best_score > 0.3:
        return title_to_idx[best_match]
    
    return None

def filter_movies(movies_df, genre_filter, min_rating, max_runtime):
    """Apply filters to dataframe"""
    filtered = movies_df.copy()
    
    # Filter by genre
    if genre_filter:
        try:
            filtered = filtered[filtered['genres'].astype(str).str.lower().str.contains(genre_filter, na=False)]
        except Exception as e:
            print(f"Error filtering by genre: {str(e)}")
    
    # Filter by rating
    if min_rating > 0:
        try:
            filtered = filtered[filtered['vote_average'].notna() & (filtered['vote_average'] >= min_rating)]
        except Exception as e:
            print(f"Error filtering by rating: {str(e)}")
    
    # Filter by runtime
    if max_runtime < 10000:
        try:
            filtered = filtered[filtered['runtime'].notna() & (filtered['runtime'] <= max_runtime)]
        except Exception as e:
            print(f"Error filtering by runtime: {str(e)}")
    
    return filtered

def format_movie_response(row, score=None):
    """Format a movie row into response dict"""
    return {
        "title": str(row.get("title", "Unknown")),
        "genres": str(row.get("genres", "Unknown")),
        "overview": str(row.get("overview", "No overview available.")),
        "poster": str(row.get("poster_path", "")),
        "rating": float(row.get("vote_average", 0)) if not pd.isna(row.get("vote_average")) else 0.0,
        "runtime": int(row.get("runtime", 0)) if not pd.isna(row.get("runtime")) else 0,
        "similarity": float(score) if score is not None else None
    }

@app.route("/browse")
def browse():
    """Browse movies with optional filters - shows trending movies sorted by rating"""
    try:
        try:
            min_rating = float(request.args.get("rating", 0))
        except (ValueError, TypeError):
            min_rating = 0.0
        
        try:
            max_runtime = int(request.args.get("runtime", 10000))
        except (ValueError, TypeError):
            max_runtime = 10000
        
        genre_filter = request.args.get("genre", "").strip().lower()

        # Apply filters
        filtered_df = filter_movies(df, genre_filter, min_rating, max_runtime)
        
        if len(filtered_df) == 0:
            return jsonify([])
        
        # Sort by rating (descending) to show trending/popular movies
        try:
            filtered_df = filtered_df.sort_values('vote_average', ascending=False, na_position='last')
        except Exception as e:
            print(f"Error sorting by rating: {str(e)}")
        
        # Get top movies (up to 24 for better display)
        num_movies = min(24, len(filtered_df))
        top_movies = filtered_df.head(num_movies)
        
        results = []
        for _, row in top_movies.iterrows():
            try:
                results.append(format_movie_response(row))
            except Exception as e:
                print(f"Error formatting movie: {str(e)}")
                continue
        
        return jsonify(results)
        
    except Exception as e:
        print(f"Error in /browse endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/recommend")
def recommend():
    try:
        movie = request.args.get("movie", "").strip()
        
        if not movie:
            return jsonify({"error": "Movie name is required"}), 400

        try:
            min_rating = float(request.args.get("rating", 0))
        except (ValueError, TypeError):
            min_rating = 0.0
        
        try:
            max_runtime = int(request.args.get("runtime", 10000))
        except (ValueError, TypeError):
            max_runtime = 10000
        
        genre_filter = request.args.get("genre", "").strip().lower()

        movie_idx = find_movie_index(movie)
        
        if movie_idx is None:
            return jsonify({
                "error": f"Movie '{movie}' not found in database",
                "suggestions": "Try searching with a different movie name"
            }), 404

        try:
            idxs, scores = search_similar(movie_idx, top_k=50)
        except Exception as e:
            return jsonify({"error": f"Error finding similar movies: {str(e)}"}), 500

        results = []
        for idx, score in zip(idxs, scores):
            try:
                if idx >= len(df):
                    continue
                    
                row = df.iloc[idx]

                # Apply filters
                vote_avg = row.get("vote_average", 0)
                if pd.isna(vote_avg) or float(vote_avg) < min_rating:
                    continue

                runtime = row.get("runtime", 0)
                if pd.isna(runtime) or int(runtime) > max_runtime:
                    continue

                if genre_filter:
                    genres = str(row.get("genres", "")).lower()
                    if genre_filter not in genres:
                        continue

                results.append(format_movie_response(row, score))
                
                if len(results) >= 12:
                    break
                    
            except Exception as e:
                print(f"Error processing movie at index {idx}: {str(e)}")
                continue

        return jsonify(results)
        
    except Exception as e:
        print(f"Error in /recommend endpoint: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "movies_loaded": len(df),
        "embeddings_shape": embeddings.shape,
        "faiss_enabled": use_faiss
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting Movie Recommendation Server...")
    print("="*50)
    print("📍 Server running at: http://127.0.0.1:5000")
    print("🎬 Frontend available at: http://127.0.0.1:5000/")
    print("📡 API endpoints:")
    print("   - /recommend - Get recommendations")
    print("   - /browse - Browse all movies")
    print("   - /health - Health check")
    print("="*50 + "\n")
    
    try:
        app.run(debug=True, host="127.0.0.1", port=5000)
    except Exception as e:
        print(f"❌ ERROR starting server: {str(e)}")
        sys.exit(1)