"""
Configuration settings for MuFrog Gradio Demo
"""

# Server configuration
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860
SHARE_PUBLICLY = False

# Data paths (relative to mufrog_app directory)
DATA_PATHS = [
    "../mufrog_core/src/analyzed_output/analyzed_metadata_latest.json",
    "mufrog_core/src/analyzed_output/analyzed_metadata_latest.json",
    "../mufrog_core/src/scripts/analyzed_metadata_latest.json",
    "mufrog_core/src/scripts/analyzed_metadata_latest.json"
]

# Recommendation settings
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

# UI settings
MAX_DISPLAY_SONGS = 1000
MOOD_DISPLAY_LIMIT = 15

# Mood classification settings
MIN_MOOD_SCORE_THRESHOLD = 0.1
MOOD_NORMALIZATION_FACTOR = 1.0

# Feature toggles
ENABLE_LLM_ENHANCEMENT = False  # Future feature
ENABLE_ADVANCED_FILTERS = True
ENABLE_ANALYTICS = True

# App metadata
APP_TITLE = "MuFrog - Music Emotion Analysis Demo"
APP_DESCRIPTION = """
Welcome to MuFrog! This demo showcases our Music2Emotion analysis system that predicts emotional moods from music.

## Features:
- **Browse Database**: Explore our analyzed music collection with mood predictions
- **Smart Recommendations**: Get song recommendations based on your mood description
- **Mood Analytics**: View statistics about emotional patterns in music
"""

ABOUT_TEXT = """
---
**About MuFrog**: This demo uses machine learning to analyze music and predict emotional moods. 
The recommendation system matches your described mood with songs in our database using similarity scoring.

*Future improvements will include LLM-powered mood interpretation and advanced matching algorithms.*
"""
