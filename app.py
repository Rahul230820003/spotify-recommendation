from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import io
import base64
from functools import lru_cache

app = Flask(__name__)

# Global variables to store our model and data
model = None
data = None
features = None
df_scaled = None
df = None

def load_data():
    """Load and prepare the Spotify data"""
    global data, features, model, df_scaled, df
    
    # Load the dataset
    df = pd.read_csv("spotify dataset.csv")
    
    # Clean the data
    df.dropna(inplace=True)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset='track_name', keep='first')
    
    # Select features for recommendation
    features = df[['danceability', 'energy', 'speechiness', 
                   'acousticness', 'instrumentalness', 'liveness', 
                   'valence', 'tempo']]
    
    # Feature engineering - add more weight to important features
    feature_weights = {
        'danceability': 1.2,
        'energy': 1.5,
        'speechiness': 0.8,
        'acousticness': 1.0,
        'instrumentalness': 0.7,
        'liveness': 0.6,
        'valence': 1.3,
        'tempo': 0.9
    }
    
    # Apply weights to features
    weighted_features = features.copy()
    for feature, weight in feature_weights.items():
        weighted_features[feature] = weighted_features[feature] * weight
    
    # Scale the weighted features
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(weighted_features)
    
    # Cluster the songs
    kmeans = KMeans(n_clusters=10, random_state=42)
    df['cluster'] = kmeans.fit_predict(df_scaled)
    
    # Initialize and train the model for feature-based recommendations
    model = NearestNeighbors(n_neighbors=5, algorithm='auto', metric='cosine')  # Use cosine similarity instead of Euclidean distance
    model.fit(df_scaled)
    
    print("Data and model loaded successfully")

def recommend_songs(song_name, n_recs=5):  # Changed default to 5
    """Recommend songs based on a song name"""
    # Convert song name to lower case
    song_name = song_name.lower()

    # Find song ID
    song_id = df.loc[df['track_name'].str.lower() == song_name, 'track_id'].values
    
    # Check if song exists
    if len(song_id) == 0:
        return {"status": "error", "message": f"Song '{song_name}' not found in dataset."}
    
    # Use the first matching song ID
    song_id = song_id[0]
    
    # Get song's cluster
    song_cluster = df.loc[df['track_id'] == song_id, 'cluster'].values[0]

    # Get songs in the same cluster
    cluster_songs = df[(df['cluster'] == song_cluster) & (df['track_id'] != song_id)]

    # Check if cluster is empty
    if cluster_songs.empty:
        return {"status": "error", "message": f"Cluster for song '{song_name}' is empty."}

    # Calculate distances between songs
    # Calculate distances with popularity weighting
    popularity_weight = 0.2  # How much to consider popularity (0-1)
    
    distances = []
    song_index = df[df['track_id'] == song_id].index[0]
    for _, row in cluster_songs.iterrows():
        row_index = row.name
        if song_index < len(df_scaled) and row_index < len(df_scaled):
            # Calculate feature distance
            feature_distance = np.linalg.norm(df_scaled[song_index] - df_scaled[row_index])
            
            # Get popularity score (normalized to 0-1)
            popularity = row['track_popularity'] / 100 if not np.isnan(row['track_popularity']) else 0.5
            
            # Combine feature distance with inverse popularity (lower distance for popular songs)
            adjusted_distance = feature_distance * (1 - (popularity * popularity_weight))
            
            distances.append((row['track_id'], row['track_name'], row['track_artist'], 
                             row['track_album_name'], row['playlist_genre'], row['track_popularity'], adjusted_distance))

    # Sort songs by distance
    distances.sort(key=lambda x: x[6])

    # Format recommendations
    recommendations = []
    for rec in distances[:n_recs]:  # Limit to n_recs (now 5 by default)
        recommendations.append({
            "name": rec[1],
            "artist": rec[2],
            "album": rec[3],
            "genre": rec[4],
            "popularity": int(rec[5]) if not np.isnan(rec[5]) else None
        })
    
    return {"status": "success", "recommendations": recommendations}

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

# Cache recommendations for better performance
@lru_cache(maxsize=100)
def cached_recommend_songs(song_name, n_recs=5):
    """Cached version of recommend_songs"""
    return recommend_songs(song_name, n_recs)

@app.route('/recommend_by_song', methods=['POST'])
def recommend_by_song():
    """Get song recommendations based on a song name"""
    if request.method == 'POST':
        try:
            # Get song name from the request
            song_name = request.json.get('song_name', '')
            
            # Get recommendations using cached function
            result = cached_recommend_songs(song_name, 5)
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get song recommendations based on input parameters"""
    if request.method == 'POST':
        try:
            # Get input parameters from the request
            input_data = request.json
            
            # Create a feature vector from the input
            input_features = np.array([[
                input_data.get('danceability', 0.5),
                input_data.get('energy', 0.5),
                input_data.get('speechiness', 0.05),
                input_data.get('acousticness', 0.5),
                input_data.get('instrumentalness', 0.0),
                input_data.get('liveness', 0.1),
                input_data.get('valence', 0.5),
                input_data.get('tempo', 120)
            ]])
            
            # Use the model to find nearest neighbors (limited to 5)
            distances, indices = model.kneighbors(input_features)
            
            # Get the recommended songs
            recommendations = []
            for i in indices[0]:
                song = {
                    "name": df.iloc[i]['track_name'],
                    "artist": df.iloc[i]['track_artist'],
                    "album": df.iloc[i]['track_album_name'],
                    "popularity": int(df.iloc[i]['track_popularity']) if not np.isnan(df.iloc[i]['track_popularity']) else None,
                    "genre": df.iloc[i]['playlist_genre']
                }
                recommendations.append(song)
            
            return jsonify({
                'status': 'success',
                'recommendations': recommendations
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

@app.route('/genre_distribution')
def genre_distribution():
    """Generate and return a plot of genre distribution"""
    plt.figure(figsize=(10, 6))
    genre_counts = df['playlist_genre'].value_counts()
    sns.barplot(x=genre_counts.index, y=genre_counts.values)
    plt.title('Distribution of Songs by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Number of Songs')
    plt.xticks(rotation=45)
    
    # Save plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Convert to base64 for embedding in HTML
    image_png = buffer.getvalue()
    buffer.close()
    encoded = base64.b64encode(image_png).decode('utf-8')
    
    return jsonify({
        'status': 'success',
        'image': f'data:image/png;base64,{encoded}'
    })

@app.route('/feature_analysis', methods=['POST'])
def feature_analysis():
    """Analyze audio features for a specific genre"""
    if request.method == 'POST':
        try:
            genre = request.json.get('genre', 'pop')
            
            # Filter data by genre
            genre_data = df[df['playlist_genre'] == genre]
            
            # Calculate average features
            avg_features = genre_data[['danceability', 'energy', 'speechiness', 
                                      'acousticness', 'instrumentalness', 'liveness', 
                                      'valence']].mean().to_dict()
            
            # Create radar chart
            categories = list(avg_features.keys())
            values = list(avg_features.values())
            
            # Close any existing plots
            plt.close('all')
            
            # Create the plot
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, polar=True)
            
            # Plot the data
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            categories += categories[:1]
            
            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
            ax.set_title(f'Audio Features for {genre.capitalize()} Genre')
            
            # Save plot to a bytes buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Convert to base64 for embedding in HTML
            image_png = buffer.getvalue()
            buffer.close()
            encoded = base64.b64encode(image_png).decode('utf-8')
            
            return jsonify({
                'status': 'success',
                'avg_features': avg_features,
                'image': f'data:image/png;base64,{encoded}'
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 500
    
    return jsonify({'status': 'error', 'message': 'Invalid request method'}), 405

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Load the data and model
    load_data()
    
    # Run the Flask app
    app.run(debug=True)


def diversify_recommendations(recommendations, df_scaled, indices, diversity_factor=0.3):
    """Ensure recommendations aren't too similar to each other"""
    if len(recommendations) <= 1:
        return recommendations
    
    diverse_recs = [recommendations[0]]  # Start with the best match
    
    for rec in recommendations[1:]:
        # Find the song index in the dataset
        rec_name = rec["name"]
        rec_indices = df[df['track_name'] == rec_name].index
        
        if len(rec_indices) == 0:
            continue
            
        rec_index = rec_indices[0]
        
        # Check if this recommendation is diverse enough from existing ones
        is_diverse = True
        for existing_rec in diverse_recs:
            existing_name = existing_rec["name"]
            existing_indices = df[df['track_name'] == existing_name].index
            
            if len(existing_indices) == 0:
                continue
                
            existing_index = existing_indices[0]
            
            # Calculate similarity
            similarity = 1 - np.linalg.norm(df_scaled[rec_index] - df_scaled[existing_index])
            
            # If too similar, reject
            if similarity > (1 - diversity_factor):
                is_diverse = False
                break
        
        if is_diverse or len(diverse_recs) < 3:  # Ensure at least 3 recommendations
            diverse_recs.append(rec)
            
        if len(diverse_recs) >= 5:
            break
    
    return diverse_recs