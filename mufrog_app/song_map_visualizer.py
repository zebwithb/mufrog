"""
Song Map Visualizer - Creates 2D visualizations of music library using dimensionality reduction.
Maps high-dimensional mood data to 2D space for interactive exploration.
Includes clustering analysis to discover implicit mood patterns.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')


class SongMapVisualizer:
    """Creates interactive 2D maps of songs based on their mood profiles"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.mood_features = None
        self.song_metadata = None
        self.reduced_data = None
        self.method_used = None
        
        # Configuration dictionary for coloring options
        self.color_configs = {
            'top_mood': {
                'type': 'categorical',
                'column': 'top_mood',
                'title': 'Top Mood',
                'color_scale': None,
                'show_legend': True
            },
            'cluster': {
                'type': 'categorical',
                'column': 'cluster',
                'title': 'Discovered Clusters',
                'color_scale': None,
                'show_legend': True
            },
            'valence': {
                'type': 'continuous',
                'column': 'valence',
                'title': 'Valence (Happiness)',
                'color_scale': 'RdYlBu',
                'show_legend': False
            },
            'arousal': {
                'type': 'continuous',
                'column': 'arousal',
                'title': 'Arousal (Energy)',
                'color_scale': 'Viridis',
                'show_legend': False
            },
            'view_count': {
                'type': 'continuous',
                'column': 'view_count',
                'title': 'Popularity (Views)',
                'color_scale': 'Blues',
                'show_legend': False
            },
            'likes': {
                'type': 'continuous',
                'column': 'likes',
                'title': 'Likes',
                'color_scale': 'Reds',
                'show_legend': False
            },
            'top_mood_score': {
                'type': 'continuous',
                'column': 'top_mood_score',
                'title': 'Mood Confidence',
                'color_scale': 'Plasma',
                'show_legend': False
            }
        }
    
    def prepare_mood_matrix(self, music_data: List[Dict]) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """
        Convert song mood data into a matrix suitable for dimensionality reduction.
        
        Returns:
            mood_matrix: (n_songs, n_moods) array of mood scores
            song_metadata: List of song metadata for each row
            mood_names: List of mood names for each column
        """
        # Get all unique moods across the dataset
        all_moods = set()
        valid_songs = []
        
        for song in music_data:
            if song.get('predicted_moods') and len(song['predicted_moods']) > 0:
                valid_songs.append(song)
                for mood_data in song['predicted_moods']:
                    all_moods.add(mood_data['mood'])
        
        mood_names = sorted(list(all_moods))
        n_songs = len(valid_songs)
        n_moods = len(mood_names)
        
        print(f"Creating mood matrix: {n_songs} songs × {n_moods} moods")
        
        # Create mood matrix
        mood_matrix = np.zeros((n_songs, n_moods))
        song_metadata = []
        
        for i, song in enumerate(valid_songs):
            # Create mood score vector for this song
            mood_dict = {mood_data['mood']: mood_data['score'] 
                        for mood_data in song['predicted_moods']}
            
            for j, mood_name in enumerate(mood_names):
                mood_matrix[i, j] = mood_dict.get(mood_name, 0.0)
            
            # Store metadata
            song_metadata.append({
                'title': song.get('title', 'Unknown'),
                'artist': song.get('artist', 'Unknown'),
                'view_count': song.get('view_count', 0),
                'likes': song.get('likes', 0),
                'valence': song.get('valence'),
                'arousal': song.get('arousal'),
                'top_mood': song['predicted_moods'][0]['mood'] if song['predicted_moods'] else 'Unknown',
                'top_mood_score': song['predicted_moods'][0]['score'] if song['predicted_moods'] else 0,
                'genre': song.get('genre', 'Unknown')
            })
        
        return mood_matrix, song_metadata, mood_names
    
    def reduce_dimensions(self, mood_matrix: np.ndarray, method: str = 'tsne', 
                         random_state: int = 42) -> np.ndarray:
        """
        Reduce high-dimensional mood data to 2D using the specified method.
        
        Args:
            mood_matrix: (n_songs, n_moods) array of mood scores
            method: 'tsne', 'pca', or 'umap'
            random_state: Random seed for reproducibility
            
        Returns:
            coordinates: (n_songs, 2) array of 2D coordinates
        """        # Standardize the features
        mood_matrix_scaled = self.scaler.fit_transform(mood_matrix)
        
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=random_state)
            coordinates = reducer.fit_transform(mood_matrix_scaled)
            print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
            
        else:  # Default to t-SNE
            reducer = TSNE(n_components=2, random_state=random_state, 
                          perplexity=min(30, len(mood_matrix_scaled) - 1),
                          learning_rate='auto')
            coordinates = reducer.fit_transform(mood_matrix_scaled)
            method = 'tsne'  # Ensure we know what method was actually used
            method = 'tsne'  # Ensure we know what method was actually used
        
        self.method_used = method
        return coordinates
    
    def create_interactive_plot(self, coordinates: np.ndarray, song_metadata: List[Dict], 
                              color_by: str = 'top_mood', title: Optional[str] = None) -> go.Figure:
        """
        Create an interactive plotly scatter plot of the song map.
        
        Args:
            coordinates: (n_songs, 2) array of 2D coordinates
            song_metadata: List of song metadata
            color_by: What to color the points by ('top_mood', 'valence', 'arousal', 'view_count')
            title: Plot title
            
        Returns:
            plotly Figure object
        """
        method_name = (self.method_used or 'unknown').upper()
        
        df = pd.DataFrame({
            'x': coordinates[:, 0],
            'y': coordinates[:, 1],
            'title': [meta['title'] for meta in song_metadata],
            'artist': [meta['artist'] for meta in song_metadata],
            'top_mood': [meta['top_mood'] for meta in song_metadata],
            'top_mood_score': [meta['top_mood_score'] for meta in song_metadata],
            'valence': [meta['valence'] or 0 for meta in song_metadata],
            'arousal': [meta['arousal'] or 0 for meta in song_metadata],
            'view_count': [meta['view_count'] for meta in song_metadata],
            'likes': [meta['likes'] for meta in song_metadata],
            'cluster': [meta.get('cluster', 'Unknown') for meta in song_metadata],
        })        # Create hover text
        df['hover_text'] = df.apply(lambda row: 
            f"<b>{row['title']}</b><br>" +
            f"Artist: {row['artist']}<br>" +
            f"Top Mood: {row['top_mood']} ({row['top_mood_score']:.2f})<br>" +
            f"Cluster: {row['cluster']}<br>" +
            f"Valence: {row['valence']:.2f}<br>" +
            f"Arousal: {row['arousal']:.2f}<br>" +
            f"Views: {row['view_count']:,}<br>" +
            f"Likes: {row['likes']:,}", axis=1)
        
        # Get color configuration
        if color_by not in self.color_configs:
            color_by = 'top_mood'  # Fallback to default
        
        config = self.color_configs[color_by]
        
        # Create the scatter plot based on configuration
        plot_kwargs = {
            'data_frame': df,
            'x': 'x',
            'y': 'y',
            'color': config['column'],
            'hover_name': 'hover_text',
            'title': title or f"Song Map - Colored by {config['title']} ({method_name})",
            'size': 'view_count',  # Size by popularity
            'size_max': 15
        }
        
        # Add color scale for continuous variables
        if config['type'] == 'continuous' and config['color_scale']:
            plot_kwargs['color_continuous_scale'] = config['color_scale']
        
        fig = px.scatter(**plot_kwargs)
          # Update layout for better visualization
        fig.update_layout(
            width=800,
            height=600,
            xaxis_title=f"Dimension 1 ({method_name})",
            yaxis_title=f"Dimension 2 ({method_name})",
            showlegend=config['show_legend'],
            hovermode='closest'
        )
        
        # Make points slightly larger and add some transparency
        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='white')))
        
        return fig
    
    def generate_song_map(self, music_data: List[Dict], method: str = 'tsne',
                         color_by: str = 'top_mood', enable_clustering: bool = True) -> go.Figure:
        """
        Complete pipeline to generate a song map from music data.
        
        Args:
            music_data: List of song dictionaries with mood predictions
            method: Dimensionality reduction method ('tsne', 'pca')
            color_by: What to color points by (see color_configs keys)
            enable_clustering: Whether to perform clustering for implicit mood discovery
            
        Returns:
            plotly Figure object
        """
        # Prepare data
        mood_matrix, song_metadata, mood_names = self.prepare_mood_matrix(music_data)
        
        if len(song_metadata) < 3:
            # Create empty plot with message
            fig = go.Figure()
            fig.add_annotation(
                text="Not enough songs with mood data to create visualization<br>Need at least 3 songs",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(
                title="Song Map - Insufficient Data",
                xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
                width=800, height=600
            )
            return fig
        
        # Store for later use
        self.mood_features = mood_names
        self.song_metadata = song_metadata
        
        # Reduce dimensions
        coordinates = self.reduce_dimensions(mood_matrix, method)
        self.reduced_data = coordinates
        
        # Perform clustering if enabled
        if enable_clustering:
            song_metadata = self.perform_clustering(coordinates, song_metadata)
        
        # Create plot
        fig = self.create_interactive_plot(coordinates, song_metadata, color_by)
        
        return fig
    
    def get_available_methods(self) -> List[str]:
        """Get list of available dimensionality reduction methods"""
        return ['tsne', 'pca']
    
    def get_stats(self) -> str:
        """Get statistics about the current song map"""
        if self.song_metadata is None:
            return "No song map generated yet."
        
        stats = f"**Song Map Statistics:**\n"
        stats += f"- Total songs mapped: {len(self.song_metadata)}\n"
        stats += f"- Mood dimensions: {len(self.mood_features) if self.mood_features else 0}\n"
        stats += f"- Reduction method: {self.method_used.upper() if self.method_used else 'None'}\n"
        
        # Top moods distribution
        top_moods = [meta['top_mood'] for meta in self.song_metadata]
        mood_counts = pd.Series(top_moods).value_counts()
        
        stats += f"\n**Most common primary moods:**\n"
        for mood, count in mood_counts.head(8).items():
            percentage = (count / len(self.song_metadata)) * 100
            stats += f"- {mood}: {count} songs ({percentage:.1f}%)\n"
        
        return stats    
    def perform_clustering(self, coordinates: np.ndarray, song_metadata: List[Dict], 
                          method: str = 'kmeans', n_clusters: Optional[int] = None) -> List[Dict]:
        """
        Perform clustering on the 2D coordinates to discover implicit moods.
        
        Args:
            coordinates: (n_songs, 2) array of 2D coordinates
            song_metadata: List of song metadata
            method: 'kmeans' or 'dbscan'
            n_clusters: Number of clusters for kmeans (auto-determined if None)
            
        Returns:
            Updated song_metadata with cluster assignments
        """
        if len(coordinates) < 3:
            # Not enough data for clustering
            for i, meta in enumerate(song_metadata):
                meta['cluster'] = 'Single'
            return song_metadata
        
        if method == 'kmeans':
            if n_clusters is None:
                # Determine optimal number of clusters using silhouette score
                max_clusters = min(10, len(coordinates) // 2)
                best_score = -1
                best_k = 3
                
                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(coordinates)
                    score = silhouette_score(coordinates, cluster_labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                
                n_clusters = best_k
                print(f"Optimal clusters: {n_clusters} (silhouette score: {best_score:.3f})")
            
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(coordinates)
            
        elif method == 'dbscan':
            # DBSCAN with automatic parameter selection
            clusterer = DBSCAN(eps=0.5, min_samples=3)
            cluster_labels = clusterer.fit_predict(coordinates)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            print(f"DBSCAN found {n_clusters} clusters")
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Create cluster names
        cluster_names = []
        for label in cluster_labels:
            if label == -1:  # DBSCAN noise
                cluster_names.append('Outlier')
            else:
                cluster_names.append(f'Cluster {label + 1}')
        
        # Update metadata with cluster information
        for i, meta in enumerate(song_metadata):
            meta['cluster'] = cluster_names[i]
            meta['cluster_id'] = int(cluster_labels[i]) if cluster_labels[i] != -1 else -1
        
        return song_metadata# Global visualizer instance
_visualizer_instance = None

def get_song_map_visualizer() -> SongMapVisualizer:
    """Get the global song map visualizer instance"""
    global _visualizer_instance
    if _visualizer_instance is None:
        _visualizer_instance = SongMapVisualizer()
    return _visualizer_instance
