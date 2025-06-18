"""
Emotion Song Map Visualizer - Creates interactive 2D visualizations based on fundamental emotions.
Uses emotion recipes to discover Joy, Sadness, Fear, Serenity, Triumph, and Surprise patterns.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
from emotion_recipes import EmotionRecipeAnalyzer


class EmotionSongMapVisualizer:
    """
    Creates interactive 2D visualizations of song libraries based on fundamental emotions.
    """
    
    def __init__(self):
        self.emotion_analyzer = EmotionRecipeAnalyzer()
        self.scaler = StandardScaler()
        
        # Color configurations for different visualization options
        self.emotion_color_configs = {
            'Dominant Emotion': {
                'type': 'categorical',
                'column': 'dominant_emotion',
                'color_map': self.emotion_analyzer.get_emotion_colors(),
                'title': 'Dominant Fundamental Emotion',
                'description': 'Color by the strongest fundamental emotion detected'
            },
            'Joy/Ecstasy': {
                'type': 'continuous',
                'column': 'emotion_joy_ecstasy',
                'color_scale': 'Viridis',
                'title': 'Joy/Ecstasy Score',
                'description': 'Festival anthems, jubilant pop songs, celebratory music'
            },
            'Sadness/Grief': {
                'type': 'continuous',
                'column': 'emotion_sadness_grief',
                'color_scale': 'Blues',
                'title': 'Sadness/Grief Score',
                'description': 'Lonely melodies, funeral dirges, melancholic music'
            },
            'Fear/Tension': {
                'type': 'continuous',
                'column': 'emotion_fear_tension',
                'color_scale': 'Reds',
                'title': 'Fear/Tension Score',
                'description': 'Horror soundtracks, tense build-ups, dramatic music'
            },
            'Serenity/Contentment': {
                'type': 'continuous',
                'column': 'emotion_serenity_contentment',
                'color_scale': 'Greens',
                'title': 'Serenity/Contentment Score',
                'description': 'Meditation music, ambient sounds, peaceful scenes'
            },
            'Triumph/Heroism': {
                'type': 'continuous',
                'column': 'emotion_triumph_heroism',
                'color_scale': 'Oranges',
                'title': 'Triumph/Heroism Score',
                'description': 'Epic themes, motivational music, Olympic fanfares'
            },
            'Surprise/Shock': {
                'type': 'continuous',
                'column': 'emotion_surprise_shock',
                'color_scale': 'Purples',
                'title': 'Surprise/Shock Score',
                'description': 'Sudden stings, unexpected changes, dramatic moments'
            },
            'Emotion Confidence': {
                'type': 'continuous',
                'column': 'emotion_confidence',
                'color_scale': 'Plasma',
                'title': 'Emotion Classification Confidence',
                'description': 'How clearly the dominant emotion is expressed'
            }
        }
    
    def create_emotion_map(self, 
                          songs_data: List[Dict], 
                          method: str = 'tsne',
                          color_by: str = 'Dominant Emotion',
                          size_by_popularity: bool = True,
                          perplexity: int = 30) -> go.Figure:
        """
        Create an interactive 2D emotion map of songs.
        """
        try:
            # Analyze songs for fundamental emotions
            print("Analyzing songs for fundamental emotions...")
            analyzed_songs = self.emotion_analyzer.analyze_song_library(songs_data)
            
            if not analyzed_songs:
                return self._create_empty_plot("No songs available for emotion analysis")
            
            df = pd.DataFrame(analyzed_songs)
            
            # Extract emotion features for dimensionality reduction
            emotion_columns = [col for col in df.columns if col.startswith('emotion_') and not col.endswith('_score')]
            if not emotion_columns:
                return self._create_empty_plot("No emotion features found")
            
            # Prepare data for dimensionality reduction
            emotion_features = df[emotion_columns].fillna(0)
            
            if len(emotion_features) < 3:
                return self._create_empty_plot("Need at least 3 songs for emotion mapping")
            
            features_scaled = self.scaler.fit_transform(emotion_features)
            
            # Apply dimensionality reduction
            print(f"Applying {method} dimensionality reduction...")
            if method.lower() == 'tsne':
                reducer = TSNE(n_components=2, perplexity=min(perplexity, len(df)-1), 
                              random_state=42, n_iter=1000, learning_rate='auto')
            else:  # PCA
                reducer = PCA(n_components=2, random_state=42)
            
            coords_2d = reducer.fit_transform(features_scaled)
            
            # Add coordinates to dataframe
            df['dim_1'] = coords_2d[:, 0]
            df['dim_2'] = coords_2d[:, 1]
            
            # Create the plot
            fig = self._create_emotion_plot(df, color_by, size_by_popularity, method)
            
            return fig
            
        except Exception as e:
            print(f"Error creating emotion map: {str(e)}")
            return self._create_empty_plot(f"Error: {str(e)}")
    
    def _create_emotion_plot(self, df: pd.DataFrame, color_by: str, size_by_popularity: bool, method: str) -> go.Figure:
        """Create the main emotion plot."""
        
        config = self.emotion_color_configs.get(color_by, self.emotion_color_configs['Dominant Emotion'])
        
        # Prepare hover text
        df['hover_text'] = df.apply(self._create_hover_text, axis=1)
        
        # Prepare size data
        size_data = None
        if size_by_popularity and 'view_count' in df.columns:
            size_data = df['view_count'].fillna(0)
            size_data = np.log1p(size_data)  # Log scale for better visualization
            if size_data.max() > size_data.min():
                size_data = (size_data - size_data.min()) / (size_data.max() - size_data.min()) * 25 + 5
            else:
                size_data = 8  # Default size if all values are the same
        
        # Create scatter plot
        if config['type'] == 'categorical':
            fig = px.scatter(
                df,
                x='dim_1',
                y='dim_2',
                color=config['column'],
                color_discrete_map=config.get('color_map', {}),
                title=f"🎭 Fundamental Emotion Map ({method.upper()})",
                labels={'dim_1': f'{method.upper()} Dimension 1', 'dim_2': f'{method.upper()} Dimension 2'},
                hover_name='hover_text'
            )
        else:
            fig = px.scatter(
                df,
                x='dim_1',
                y='dim_2',
                color=config['column'],
                color_continuous_scale=config['color_scale'],
                title=f"🎭 {config['title']} Map ({method.upper()})",
                labels={'dim_1': f'{method.upper()} Dimension 1', 'dim_2': f'{method.upper()} Dimension 2'},
                hover_name='hover_text'
            )
        
        # Update traces with size and hover
        fig.update_traces(
            marker=dict(
                size=size_data if size_data is not None else 8,
                opacity=0.7,
                line=dict(width=0.5, color='white')
            )
        )
        
        # Update layout
        fig.update_layout(
            width=1000,
            height=700,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            margin=dict(r=200)
        )
        
        return fig
    
    def _create_hover_text(self, row) -> str:
        """Create hover text for a song row."""
        hover_parts = [
            f"<b>{row.get('title', 'Unknown')}</b>",
            f"Artist: {row.get('artist', 'Unknown')}",
            "",
            f"<b>🎭 Dominant:</b> {row.get('dominant_emotion', 'Unknown')} ({row.get('dominant_emotion_score', 0):.3f})",
            "",
            f"<b>📊 Emotion Scores:</b>",
            f"😄 Joy/Ecstasy: {row.get('emotion_joy_ecstasy', 0):.3f}",
            f"😢 Sadness/Grief: {row.get('emotion_sadness_grief', 0):.3f}",
            f"😨 Fear/Tension: {row.get('emotion_fear_tension', 0):.3f}",
            f"😌 Serenity: {row.get('emotion_serenity_contentment', 0):.3f}",
            f"🏆 Triumph: {row.get('emotion_triumph_heroism', 0):.3f}",
            f"😲 Surprise: {row.get('emotion_surprise_shock', 0):.3f}",
            "",
            f"🎯 Confidence: {row.get('emotion_confidence', 0):.3f}",
            f"👀 Views: {row.get('view_count', 0):,}",
        ]
        
        return "<br>".join(hover_parts)
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title="🎭 Fundamental Emotion Map",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=1000,
            height=700
        )
        return fig
    
    def get_color_options(self) -> List[str]:
        """Get available color options for the emotion map."""
        return list(self.emotion_color_configs.keys())
    
    def get_emotion_descriptions(self) -> Dict[str, str]:
        """Get emotion descriptions for UI help text."""
        return self.emotion_analyzer.get_emotion_descriptions()
    
    def calculate_emotion_statistics(self, songs_data: List[Dict]) -> Dict:
        """Calculate emotion distribution statistics."""
        analyzed_songs = self.emotion_analyzer.analyze_song_library(songs_data)
        
        if not analyzed_songs:
            return {'total_songs': 0, 'emotion_distribution': {}, 'average_scores': {}}
        
        df = pd.DataFrame(analyzed_songs)
        
        # Calculate emotion distribution
        emotion_counts = df['dominant_emotion'].value_counts().to_dict()
        total_songs = len(df)
        
        # Calculate average scores for each emotion
        emotion_columns = [col for col in df.columns if col.startswith('emotion_') and not col.endswith('_score')]
        average_scores = {}
        for col in emotion_columns:
            emotion_name = col.replace('emotion_', '').replace('_', '/').title()
            average_scores[emotion_name] = df[col].mean()
        
        return {
            'total_songs': total_songs,
            'emotion_distribution': emotion_counts,
            'average_scores': average_scores
        }


# Global visualizer instance
_emotion_visualizer_instance = None

def get_emotion_song_map_visualizer() -> EmotionSongMapVisualizer:
    """Get the global emotion song map visualizer instance"""
    global _emotion_visualizer_instance
    if _emotion_visualizer_instance is None:
        _emotion_visualizer_instance = EmotionSongMapVisualizer()
    return _emotion_visualizer_instance
