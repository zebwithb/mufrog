"""
Song recommendation engine for MuFrog Gradio demo.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from mood_classifier import classify_user_prompt


def calculate_song_similarity(user_mood_scores: Dict[str, float], song: Dict) -> float:
    """
    Calculate similarity between user mood preferences and a song's mood profile.
    
    Args:
        user_mood_scores: Dictionary of mood categories and their scores from user prompt
        song: Song dictionary with predicted_moods
        
    Returns:
        Similarity score (0-1, higher is more similar)
    """
    if not song.get('predicted_moods'):
        return 0.0
    
    # Create mood score dictionary for the song
    song_moods = {mood_data['mood']: mood_data['score'] for mood_data in song['predicted_moods']}
    
    # Calculate similarity using cosine similarity of mood vectors
    similarity = 0
    for mood, user_score in user_mood_scores.items():
        if mood in song_moods:
            similarity += user_score * song_moods[mood]
    
    # Normalize by the magnitude of user mood vector
    user_magnitude = np.sqrt(sum(score**2 for score in user_mood_scores.values()))
    if user_magnitude > 0:
        similarity /= user_magnitude
    
    return similarity


def find_matching_songs(music_data: List[Dict], user_prompt: str, top_k: int = 10) -> Tuple[List[Dict], str]:
    """
    Find songs that match the user's mood prompt using similarity matching.
    
    Args:
        music_data: List of song dictionaries
        user_prompt: User's description of desired music mood
        top_k: Number of top recommendations to return
        
    Returns:
        Tuple of (recommended songs list, explanation string)
    """
    if not music_data:
        return [], "No music data available"
    
    # Classify user prompt to get mood scores
    user_mood_scores = classify_user_prompt(user_prompt)
    
    if not user_mood_scores:
        return [], "Could not identify any moods from your prompt. Try using more descriptive emotional words."
    
    # Calculate similarity scores for each song
    song_similarities = []
    
    for song in music_data:
        if not song.get('predicted_moods'):
            continue
        
        similarity = calculate_song_similarity(user_mood_scores, song)
        song_similarities.append((similarity, song))
    
    # Sort by similarity and return top_k
    song_similarities.sort(key=lambda x: x[0], reverse=True)
    top_songs = [song for _, song in song_similarities[:top_k]]
    
    # Create explanation
    detected_moods = ", ".join(user_mood_scores.keys())
    explanation = f"Detected moods: {detected_moods}\nFound {len(top_songs)} matching songs based on mood similarity."
    
    return top_songs, explanation


def get_song_features_for_recommendation(song: Dict) -> Dict:
    """
    Extract relevant features from a song for recommendation display.
    
    Args:
        song: Song dictionary
        
    Returns:
        Dictionary with relevant song features
    """
    return {
        'title': song.get('title', 'Unknown'),
        'artist': song.get('artist', 'Unknown'),
        'valence': song.get('valence'),
        'arousal': song.get('arousal'),
        'view_count': song.get('view_count', 0),
        'likes': song.get('likes', 0),
        'top_moods': song.get('predicted_moods', [])[:3],
        'release_date': song.get('release_date', 'Unknown')
    }


def recommend_songs_with_filters(
    music_data: List[Dict], 
    user_prompt: str, 
    min_valence: Optional[float] = None,
    max_valence: Optional[float] = None,
    min_arousal: Optional[float] = None,
    max_arousal: Optional[float] = None,
    top_k: int = 10
) -> Tuple[List[Dict], str]:
    """
    Enhanced recommendation with optional valence/arousal filters.
    
    Args:
        music_data: List of song dictionaries
        user_prompt: User's description of desired music mood
        min_valence: Minimum valence score filter (0-1)
        max_valence: Maximum valence score filter (0-1)
        min_arousal: Minimum arousal score filter (0-1)
        max_arousal: Maximum arousal score filter (0-1)
        top_k: Number of top recommendations to return
        
    Returns:
        Tuple of (recommended songs list, explanation string)
    """
    # First get basic recommendations
    songs, explanation = find_matching_songs(music_data, user_prompt, top_k * 2)  # Get more to filter
    
    if not songs:
        return songs, explanation
    
    # Apply filters if specified
    filtered_songs = []
    for song in songs:
        valence = song.get('valence')
        arousal = song.get('arousal')
        
        # Skip songs without valence/arousal data if filters are applied
        if (min_valence is not None or max_valence is not None) and valence is None:
            continue
        if (min_arousal is not None or max_arousal is not None) and arousal is None:
            continue
        
        # Apply valence filters
        if min_valence is not None and valence < min_valence:
            continue
        if max_valence is not None and valence > max_valence:
            continue
        
        # Apply arousal filters
        if min_arousal is not None and arousal < min_arousal:
            continue
        if max_arousal is not None and arousal > max_arousal:
            continue
        
        filtered_songs.append(song)
        
        if len(filtered_songs) >= top_k:
            break
    
    # Update explanation if filters were applied
    if any([min_valence, max_valence, min_arousal, max_arousal]):
        filter_info = []
        if min_valence is not None:
            filter_info.append(f"valence ≥ {min_valence:.2f}")
        if max_valence is not None:
            filter_info.append(f"valence ≤ {max_valence:.2f}")
        if min_arousal is not None:
            filter_info.append(f"arousal ≥ {min_arousal:.2f}")
        if max_arousal is not None:
            filter_info.append(f"arousal ≤ {max_arousal:.2f}")
        
        explanation += f"\nApplied filters: {', '.join(filter_info)}"
        explanation += f"\nFiltered results: {len(filtered_songs)} songs"
    
    return filtered_songs[:top_k], explanation
