"""
Data loading and processing module for MuFrog Gradio demo.
"""
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


def load_music_data() -> List[Dict]:
    """Load the analyzed metadata from the JSON file"""
    json_path = Path("../mufrog_core/src/analyzed_output/analyzed_metadata_latest.json")
    
    if not json_path.exists():
        # Fallback to other possible locations
        fallback_paths = [
            Path("mufrog_core/src/analyzed_output/analyzed_metadata_latest.json"),
            Path("../mufrog_core/src/scripts/analyzed_metadata_latest.json"),
            Path("mufrog_core/src/scripts/analyzed_metadata_latest.json")
        ]
        
        for path in fallback_paths:
            if path.exists():
                json_path = path
                break
        else:
            raise FileNotFoundError("Could not find analyzed_metadata_latest.json")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_songs_dataframe(music_data: List[Dict]) -> pd.DataFrame:
    """Convert music data to a pandas DataFrame for display"""
    if not music_data:
        return pd.DataFrame()
    
    rows = []
    for song in music_data:
        # Skip songs without mood predictions
        if not song.get('predicted_moods'):
            continue
            
        # Get top 3 moods
        top_moods = song['predicted_moods'][:3]
        mood_str = ", ".join([f"{mood['mood']} ({mood['score']:.2f})" for mood in top_moods])
        
        rows.append({
            'Title': song.get('title', 'Unknown'),
            'Artist': song.get('artist', 'Unknown'),
            'View Count': f"{song.get('view_count', 0):,}",
            'Likes': f"{song.get('likes', 0):,}",
            'Valence': f"{song.get('valence', 0):.2f}" if song.get('valence') else "N/A",
            'Arousal': f"{song.get('arousal', 0):.2f}" if song.get('arousal') else "N/A",
            'Top Moods': mood_str,
            'Release Date': song.get('release_date', 'Unknown')
        })
    
    return pd.DataFrame(rows)


def get_all_moods(music_data: List[Dict]) -> List[str]:
    """Extract all unique moods from the dataset"""
    moods = set()
    for song in music_data:
        for mood_data in song.get('predicted_moods', []):
            moods.add(mood_data['mood'])
    return sorted(list(moods))


def get_mood_statistics(music_data: List[Dict]) -> str:
    """Get statistics about moods in the dataset"""
    if not music_data:
        return "No data available"
    
    mood_counts = {}
    total_songs = 0
    
    for song in music_data:
        if song.get('predicted_moods'):
            total_songs += 1
            for mood_data in song['predicted_moods']:
                mood = mood_data['mood']
                if mood not in mood_counts:
                    mood_counts[mood] = 0
                mood_counts[mood] += 1
    
    # Sort by frequency
    sorted_moods = sorted(mood_counts.items(), key=lambda x: x[1], reverse=True)
    
    result = f"**Dataset Statistics:**\n"
    result += f"Total songs with mood analysis: {total_songs}\n"
    result += f"Unique moods detected: {len(mood_counts)}\n\n"
    result += "**Most common moods:**\n"
    
    for mood, count in sorted_moods[:15]:  # Top 15 moods
        percentage = (count / total_songs) * 100
        result += f"- {mood}: {count} songs ({percentage:.1f}%)\n"
    
    return result


# Global data loading
def initialize_data() -> List[Dict]:
    """Initialize and load music data with error handling"""
    try:
        music_data = load_music_data()
        print(f"Loaded {len(music_data)} songs from database")
        return music_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return []
