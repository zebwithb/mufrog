"""
Emotion Recipe Analyzer - Analyzes songs for fundamental emotions using mood key "recipes".
Each emotion is defined by combinations of high and low mood scores.
"""
import numpy as np
from typing import Dict, List, Tuple


class EmotionRecipeAnalyzer:
    """
    Analyzes songs for fundamental emotions using mood key "recipes".
    Each emotion is defined by combinations of high and low mood scores.
    """
    
    def __init__(self):
        self.emotion_recipes = {
            'Joy/Ecstasy': {
                'high_moods': ['happy', 'upbeat', 'energetic', 'positive', 'party', 'fun', 'uplifting'],
                'low_moods': ['sad', 'dark', 'melancholic', 'slow'],
                'valence_range': (0.7, 1.0),  # High Valence
                'arousal_range': (0.7, 1.0),  # High Arousal
                'description': 'A festival anthem, a jubilant pop song, the climax of a celebratory film score.',
                'color': '#FFD700',  # Gold
                'emoji': '😄'
            },
            'Sadness/Grief': {
                'high_moods': ['sad', 'melancholic', 'slow', 'emotional', 'soft'],
                'low_moods': ['happy', 'upbeat', 'energetic', 'party', 'fun'],
                'valence_range': (0.0, 0.3),  # Low Valence
                'arousal_range': (0.0, 0.3),  # Low Arousal
                'description': 'A lonely piano melody, a funeral dirge, the music when a beloved character dies.',
                'color': '#4169E1',  # Royal Blue
                'emoji': '😢'
            },
            'Fear/Tension': {
                'high_moods': ['dark', 'drama', 'dramatic', 'soundscape'],
                'low_moods': ['hopeful', 'positive', 'love', 'soft', 'melodic'],
                'valence_range': (0.0, 0.3),  # Low Valence
                'arousal_range': (0.6, 1.0),  # High Arousal
                'description': 'A horror movie soundtrack with dissonant strings, the tense build-up before a jump scare.',
                'color': '#8B0000',  # Dark Red
                'emoji': '😨'
            },
            'Serenity/Contentment': {
                'high_moods': ['meditative', 'soft', 'slow', 'dream', 'soundscape', 'hopeful'],
                'low_moods': ['powerful', 'energetic', 'fast', 'dramatic', 'dark'],
                'valence_range': (0.6, 1.0),  # High Valence
                'arousal_range': (0.0, 0.4),  # Low Arousal
                'description': 'Ambient music for meditation or sleep, a gentle pastoral scene in a movie.',
                'color': '#20B2AA',  # Light Sea Green
                'emoji': '😌'
            },
            'Triumph/Heroism': {
                'high_moods': ['powerful', 'uplifting', 'motivational', 'dramatic', 'melodic', 'positive'],
                'low_moods': ['sad', 'soft', 'funny', 'melancholic'],
                'valence_range': (0.7, 1.0),  # High Valence
                'arousal_range': (0.8, 1.0),  # Very High Arousal
                'description': 'The main theme from Star Wars or Indiana Jones, epic trailer music, Olympic fanfares.',
                'color': '#FF4500',  # Orange Red
                'emoji': '🏆'
            },
            'Surprise/Shock': {
                'high_moods': ['dramatic', 'fast', 'energetic', 'powerful'],
                'low_moods': [],  # Defined more by suddenness than low scores
                'valence_range': (0.4, 0.6),  # Neutral Valence
                'arousal_range': (0.7, 1.0),  # High Arousal
                'description': 'A sudden orchestral "sting" in a movie, an unexpected beat drop or key change.',
                'color': '#FF1493',  # Deep Pink
                'emoji': '😲'
            }
        }
    
    def calculate_emotion_scores(self, song_data: Dict) -> Dict[str, float]:
        """
        Calculate emotion scores for a song based on mood recipes.
        Returns a dictionary of emotion names and their scores (0-1).
        """
        emotion_scores = {}
        
        # Convert predicted moods to a dictionary for easier lookup
        mood_dict = {}
        if 'predicted_moods' in song_data:
            for mood_entry in song_data['predicted_moods']:
                mood_dict[mood_entry['mood']] = mood_entry['score']
        
        for emotion_name, recipe in self.emotion_recipes.items():
            score = self._calculate_single_emotion_score(mood_dict, song_data, recipe)
            emotion_scores[emotion_name] = score
            
        return emotion_scores
    
    def _calculate_single_emotion_score(self, mood_dict: Dict, song_data: Dict, recipe: Dict) -> float:
        """
        Calculate score for a single emotion based on its recipe.
        """
        high_score = 0.0
        low_score = 0.0
        
        # Calculate positive contribution from high moods
        for mood in recipe['high_moods']:
            if mood in mood_dict:
                high_score += mood_dict[mood]
        
        # Normalize by number of high moods
        if recipe['high_moods']:
            high_score = high_score / len(recipe['high_moods'])
        
        # Calculate negative contribution from low moods (inverted)
        for mood in recipe['low_moods']:
            if mood in mood_dict:
                low_score += (1.0 - mood_dict[mood])  # Invert: low scores are good
        
        # Normalize by number of low moods
        if recipe['low_moods']:
            low_score = low_score / len(recipe['low_moods'])
        else:
            low_score = 0.5  # Neutral if no low moods specified
        
        # Combine high and low contributions
        if recipe['high_moods'] and recipe['low_moods']:
            emotion_score = (high_score + low_score) / 2
        elif recipe['high_moods']:
            emotion_score = high_score
        else:
            emotion_score = low_score
        
        # Optional: Apply valence/arousal constraints if available
        if song_data.get('valence') is not None and song_data.get('arousal') is not None:
            valence_match = self._check_range_match(song_data['valence'], recipe['valence_range'])
            arousal_match = self._check_range_match(song_data['arousal'], recipe['arousal_range'])
            range_bonus = (valence_match + arousal_match) / 2
            emotion_score = emotion_score * 0.7 + range_bonus * 0.3  # Weight mood recipe more
        
        return max(0.0, min(1.0, emotion_score))  # Clamp to [0, 1]
    
    def _check_range_match(self, value: float, target_range: Tuple[float, float]) -> float:
        """
        Check how well a value matches a target range.
        Returns 1.0 for perfect match, 0.0 for complete mismatch.
        """
        min_val, max_val = target_range
        if min_val <= value <= max_val:
            return 1.0
        else:
            # Calculate distance from range
            if value < min_val:
                distance = min_val - value
            else:
                distance = value - max_val
            
            # Convert distance to similarity (max distance assumed to be 1.0)
            return max(0.0, 1.0 - distance)
    
    def get_dominant_emotion(self, emotion_scores: Dict[str, float]) -> Tuple[str, float]:
        """
        Get the dominant emotion and its score.
        """
        if not emotion_scores:
            return "Unknown", 0.0
        
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        return dominant_emotion
    
    def get_emotion_colors(self) -> Dict[str, str]:
        """
        Get color mapping for emotions.
        """
        return {emotion: recipe['color'] for emotion, recipe in self.emotion_recipes.items()}
    
    def get_emotion_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions for emotions.
        """
        return {emotion: f"{recipe['emoji']} {recipe['description']}" 
                for emotion, recipe in self.emotion_recipes.items()}
    
    def analyze_song_library(self, songs_data: List[Dict]) -> List[Dict]:
        """
        Analyze entire song library for fundamental emotions.
        Returns list of songs with emotion analysis added.
        """
        analyzed_songs = []
        
        for song in songs_data:
            song_analysis = song.copy()
            
            # Calculate emotion scores
            emotion_scores = self.calculate_emotion_scores(song)
            
            # Add emotion scores to song data
            for emotion, score in emotion_scores.items():
                emotion_key = f'emotion_{emotion.lower().replace("/", "_").replace(" ", "_")}'
                song_analysis[emotion_key] = score
            
            # Add dominant emotion
            dominant_emotion, dominant_score = self.get_dominant_emotion(emotion_scores)
            song_analysis['dominant_emotion'] = dominant_emotion
            song_analysis['dominant_emotion_score'] = dominant_score
            
            # Add emotion confidence (how clear the dominant emotion is)
            scores_list = list(emotion_scores.values())
            if len(scores_list) > 1:
                scores_sorted = sorted(scores_list, reverse=True)
                confidence = scores_sorted[0] - scores_sorted[1] if len(scores_sorted) > 1 else scores_sorted[0]
                song_analysis['emotion_confidence'] = confidence
            else:
                song_analysis['emotion_confidence'] = dominant_score
            
            analyzed_songs.append(song_analysis)
        
        return analyzed_songs
