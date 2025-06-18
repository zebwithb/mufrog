"""
Mood classification and matching logic for MuFrog Gradio demo.
"""
from typing import Dict, List


# Mood keywords based on Music2Emotion analysis
MOOD_KEYS = [
    'adventure', 'ballad', 'christmas', 'commercial', 'dark', 'deep', 'drama', 
    'dramatic', 'dream', 'emotional', 'energetic', 'fast', 'fun', 'funny', 
    'game', 'groovy', 'happy', 'holiday', 'hopeful', 'love', 'meditative', 
    'melancholic', 'melodic', 'motivational', 'party', 'positive', 'powerful', 
    'retro', 'romantic', 'sad', 'sexy', 'slow', 'soft', 'soundscape', 'space', 
    'sport', 'summer', 'travel', 'upbeat', 'uplifting'
]


def get_mood_keywords_mapping() -> Dict[str, List[str]]:
    """
    Define mood keywords mapping user language to Music2Emotion categories.
    This mapping helps classify user prompts into mood categories.
    """
    return {
        'adventure': ['adventure', 'adventurous', 'exciting', 'thrilling', 'epic', 'journey', 'exploration'],
        'ballad': ['ballad', 'slow song', 'emotional song', 'heartfelt', 'storytelling'],
        'christmas': ['christmas', 'holiday', 'festive', 'winter', 'xmas', 'seasonal'],
        'commercial': ['commercial', 'mainstream', 'popular', 'radio-friendly'],
        'dark': ['dark', 'gloomy', 'sinister', 'ominous', 'haunting', 'mysterious'],
        'deep': ['deep', 'profound', 'meaningful', 'thoughtful', 'contemplative'],
        'drama': ['drama', 'theatrical', 'intense', 'serious'],
        'dramatic': ['dramatic', 'intense', 'powerful', 'emotional', 'climactic'],
        'dream': ['dream', 'dreamy', 'ethereal', 'surreal', 'floating', 'otherworldly'],
        'emotional': ['emotional', 'moving', 'touching', 'heartfelt', 'feelings'],
        'energetic': ['energetic', 'high energy', 'vigorous', 'dynamic', 'lively'],
        'fast': ['fast', 'quick', 'rapid', 'speedy', 'tempo', 'up-tempo'],
        'fun': ['fun', 'enjoyable', 'entertaining', 'playful', 'lighthearted'],
        'funny': ['funny', 'humorous', 'comedy', 'amusing', 'witty', 'silly'],
        'game': ['game', 'gaming', 'video game', 'arcade', 'electronic'],
        'groovy': ['groovy', 'groove', 'funky', 'rhythmic', 'danceable'],
        'happy': ['happy', 'joy', 'cheerful', 'bright', 'sunny', 'glad'],
        'holiday': ['holiday', 'vacation', 'celebration', 'festive'],
        'hopeful': ['hopeful', 'optimistic', 'inspiring', 'encouraging', 'promising'],
        'love': ['love', 'romantic', 'affection', 'romance', 'passion', 'heart'],
        'meditative': ['meditative', 'meditation', 'peaceful', 'zen', 'mindful', 'tranquil'],
        'melancholic': ['melancholic', 'melancholy', 'wistful', 'bittersweet', 'nostalgic'],
        'melodic': ['melodic', 'melody', 'tuneful', 'musical', 'harmonic'],
        'motivational': ['motivational', 'inspiring', 'empowering', 'uplifting', 'encouraging'],
        'party': ['party', 'celebration', 'dance', 'club', 'festive', 'wild'],
        'positive': ['positive', 'upbeat', 'optimistic', 'good vibes', 'cheerful'],
        'powerful': ['powerful', 'strong', 'mighty', 'forceful', 'intense'],
        'retro': ['retro', 'vintage', 'old school', 'nostalgic', 'classic', '80s', '90s'],
        'romantic': ['romantic', 'love', 'intimate', 'tender', 'passionate'],
        'sad': ['sad', 'melancholy', 'depressed', 'down', 'blue', 'sorrowful'],
        'sexy': ['sexy', 'sensual', 'seductive', 'sultry', 'alluring'],
        'slow': ['slow', 'relaxed', 'leisurely', 'calm', 'gentle'],
        'soft': ['soft', 'gentle', 'quiet', 'mellow', 'tender'],
        'soundscape': ['soundscape', 'ambient', 'atmospheric', 'cinematic', 'background'],
        'space': ['space', 'cosmic', 'galaxy', 'universe', 'stars', 'futuristic'],
        'sport': ['sport', 'sports', 'athletic', 'workout', 'exercise', 'competition'],
        'summer': ['summer', 'sunny', 'beach', 'vacation', 'warm', 'tropical'],
        'travel': ['travel', 'journey', 'road trip', 'adventure', 'wanderlust'],
        'upbeat': ['upbeat', 'lively', 'energetic', 'positive', 'cheerful'],
        'uplifting': ['uplifting', 'inspiring', 'encouraging', 'motivating', 'hopeful']
    }


def classify_user_prompt(user_prompt: str) -> Dict[str, float]:
    """
    Simple keyword-based mood classification for user prompts using Music2Emotion mood categories.
    This is a basic implementation that can be enhanced with LLM integration.
    
    Args:
        user_prompt: User's description of their mood or music preference
        
    Returns:
        Dictionary mapping mood categories to confidence scores (0-1)
    """
    user_prompt = user_prompt.lower()
    mood_keywords = get_mood_keywords_mapping()
    mood_scores = {}
    
    # Score each mood based on keyword matches
    for mood, keywords in mood_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in user_prompt:
                score += 1
        
        # Normalize score
        if score > 0:
            mood_scores[mood] = min(score / len(keywords), 1.0)
    
    return mood_scores


def get_available_moods() -> List[str]:
    """Get list of available mood categories from Music2Emotion analysis"""
    return MOOD_KEYS


def format_mood_categories() -> str:
    """Format mood categories for display in the interface"""
    return ", ".join(MOOD_KEYS)


def enhance_mood_classification_with_llm(user_prompt: str) -> Dict[str, float]:
    """
    Future enhancement: Use LLM to better understand user prompts and map to mood categories.
    This is a placeholder for LLM integration.
    
    Args:
        user_prompt: User's description of their mood or music preference
        
    Returns:
        Dictionary mapping mood categories to confidence scores (0-1)
    """
    # TODO: Implement LLM-based mood classification
    # For now, fall back to keyword-based classification
    return classify_user_prompt(user_prompt)
