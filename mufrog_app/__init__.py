"""
MuFrog Gradio Demo Package
Music Emotion Analysis and Recommendation System
"""

__version__ = "1.0.0"
__author__ = "MuFrog Team"
__description__ = "Music Emotion Analysis and Recommendation System with Gradio Interface"

from .app import main
from .data_loader import initialize_data, load_music_data
from .mood_classifier import classify_user_prompt, MOOD_KEYS
from .recommender import find_matching_songs
from .ui_components import format_recommendations

__all__ = [
    'main',
    'initialize_data',
    'load_music_data', 
    'classify_user_prompt',
    'MOOD_KEYS',
    'find_matching_songs',
    'format_recommendations'
]
