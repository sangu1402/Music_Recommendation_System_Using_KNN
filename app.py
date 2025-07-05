from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Union
import pandas as pd
import joblib
import numpy as np

app = FastAPI()

df = pd.read_csv("data/songs.csv")
X_scaled = np.load("data/X_scaled.npy")
knn = joblib.load("model/knn_model.pkl")

class RecommendationRequest(BaseModel):
    track_ids: Union[str, List[str]]
    top_n: int = 5

def get_recommendations(track_ids, top_n=5):
    if isinstance(track_ids, str):
        track_ids = [track_ids]

    valid_ids = [tid for tid in track_ids if tid in df['track_id'].values]
    if not valid_ids:
        return []

    indices = df[df['track_id'].isin(valid_ids)].index.tolist()
    query_vectors = X_scaled[indices]

    distances, indices_arr = knn.kneighbors(query_vectors, n_neighbors=top_n + 10)

    seen_track_ids = set(valid_ids)
    recommendations = []

    for row_dists, row_indices in zip(distances, indices_arr):
        for dist, i in zip(row_dists, row_indices):
            song = df.iloc[i]
            song_id = song['track_id']

            if song_id in seen_track_ids:
                continue

            recommendations.append({
                'track_id': song_id,
                'track_name': song['track_name'],
                'artist_name': song['artist_name'],
                'genre': song['genre'],
                'distance': round(float(dist), 4)
            })
            seen_track_ids.add(song_id)

            if len(recommendations) >= top_n:
                break
        if len(recommendations) >= top_n:
            break

    for r in recommendations:
        r['similarity'] = round((1 - r['distance']) * 100, 1)

    recommendations.sort(key=lambda x: x['distance'])
    return recommendations

app.mount("/static", StaticFiles(directory="view"), name="static")


@app.get("/")
def read_root():
    return FileResponse("view/index.html")

@app.post("/recommendations")
def recommend(request: RecommendationRequest):
    recs = get_recommendations(request.track_ids, request.top_n)
    if not recs:
        raise HTTPException(status_code=404, detail="No valid track IDs found or no recommendations available.")
    return {"recommendations": recs}

@app.get("/songs")
def get_songs(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100)):
    start = (page - 1) * limit
    end = start + limit
    total = len(df)

    songs = df.iloc[start:end][['track_id', 'track_name', 'artist_name', 'genre']].to_dict(orient='records')

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": (total + limit - 1) // limit,
        "songs": songs
    }

@app.get("/search")
def search_songs(query: str = Query(..., min_length=1)):
    query_lower = query.lower()

    def matches(row):
        return any(
            query_lower in str(row[field]).lower()
            for field in ['track_name', 'artist_name']
            if pd.notna(row[field])
        )

    results = df[df.apply(matches, axis=1)]

    if results.empty:
        return {"results": []}
    
    songs = results[['track_id', 'track_name', 'artist_name']].to_dict(orient='records')
    return {"results": songs}