"""
UI components and formatting utilities for MuFrog Gradio demo.
"""
from typing import List, Dict, Optional


def format_recommendations(songs: List[Dict], explanation: str) -> str:
    """Format song recommendations as a readable string"""
    if not songs:
        return explanation
    
    result = f"{explanation}\n\n**Recommended Songs:**\n\n"
    
    for i, song in enumerate(songs, 1):
        title = song.get('title', 'Unknown')
        artist = song.get('artist', 'Unknown')
        
        # Get top 3 moods
        top_moods = song.get('predicted_moods', [])[:3]
        mood_str = ", ".join([f"{mood['mood']} ({mood['score']:.2f})" for mood in top_moods])
        
        result += f"{i}. **{title}** by {artist}\n"
        result += f"   Top moods: {mood_str}\n"
        if song.get('valence') is not None:
            result += f"   Valence: {song['valence']:.2f}, Arousal: {song['arousal']:.2f}\n"
        result += "\n"
    
    return result


def format_recommendations_enhanced(songs: List[Dict], explanation: str, user_mood_scores: Optional[Dict[str, float]] = None) -> tuple:
    """
    Format song recommendations for enhanced display with separate components.
    Returns tuple of (summary, detailed_results, mood_analysis)
    """
    if not songs:
        return explanation, "", ""
    
    # Summary section
    summary = f"🎵 **Found {len(songs)} matching songs**\n\n{explanation}"
    
    # Detailed results
    detailed_results = "## 🎶 Recommended Songs\n\n"
    
    for i, song in enumerate(songs, 1):
        title = song.get('title', 'Unknown')
        artist = song.get('artist', 'Unknown')
        
        # Get all song moods
        song_moods = song.get('predicted_moods', [])
        
        detailed_results += f"### {i}. {title}\n"
        detailed_results += f"**Artist:** {artist}\n\n"
        
        if song_moods and user_mood_scores:
            # Separate moods into matching and non-matching based on user preferences
            matching_moods = []
            non_matching_moods = []
            
            for mood_data in song_moods:
                mood_name = mood_data['mood']
                if mood_name in user_mood_scores:
                    matching_moods.append(mood_data)
                else:
                    non_matching_moods.append(mood_data)
            
            # Sort matching moods by user preference score (highest first)
            matching_moods.sort(key=lambda x: user_mood_scores.get(x['mood'], 0), reverse=True)
            
            # Display top matching moods
            if matching_moods:
                detailed_results += "**🎯 Top Matching Moods:**\n"
                for mood in matching_moods[:3]:  # Top 3 matching
                    mood_emoji = get_mood_emoji(mood['mood'])
                    user_score = user_mood_scores.get(mood['mood'], 0)
                    detailed_results += f"- {mood_emoji} {mood['mood'].title()}: {mood['score']:.2f} (Match: {user_score:.2f})\n"
                detailed_results += "\n"
            
            # Display least matching moods (but still present in song)
            if non_matching_moods:
                # Sort by song score (highest first) and take top 2
                non_matching_moods.sort(key=lambda x: x['score'], reverse=True)
                detailed_results += "**🔍 Other Notable Moods:**\n"
                for mood in non_matching_moods[:2]:  # Top 2 non-matching
                    mood_emoji = get_mood_emoji(mood['mood'])
                    detailed_results += f"- {mood_emoji} {mood['mood'].title()}: {mood['score']:.2f}\n"
                detailed_results += "\n"
        
        elif song_moods:
            # Fallback to showing top moods if no user mood scores available
            detailed_results += "**🎵 Top Moods:**\n"
            for mood in song_moods[:3]:
                mood_emoji = get_mood_emoji(mood['mood'])
                detailed_results += f"- {mood_emoji} {mood['mood'].title()}: {mood['score']:.2f}\n"
            detailed_results += "\n"
        
        if song.get('valence') is not None:
            detailed_results += f"**Emotional Profile:**\n"
            detailed_results += f"- 😊 Valence (positivity): {song['valence']:.2f}\n"
            detailed_results += f"- ⚡ Arousal (energy): {song['arousal']:.2f}\n\n"
        
        if song.get('view_count'):
            detailed_results += f"**Popularity:**\n"
            detailed_results += f"- 👀 Views: {song['view_count']:,}\n"
            if song.get('likes'):
                detailed_results += f"- ❤️ Likes: {song['likes']:,}\n"
            detailed_results += "\n"
        
        detailed_results += "---\n\n"
      # Mood analysis
    if songs:
        mood_analysis = create_mood_analysis(songs, user_mood_scores)
    else:
        mood_analysis = ""
    
    return summary, detailed_results, mood_analysis


def get_mood_emoji(mood: str) -> str:
    """Get emoji representation for moods"""
    mood_emojis = {
        'happy': '😄', 'sad': '😢', 'love': '💕', 'romantic': '💖',
        'energetic': '⚡', 'calm': '😌', 'peaceful': '🕊️', 'upbeat': '🎵',
        'dark': '🌑', 'bright': '☀️', 'powerful': '💪', 'soft': '🌸',
        'dramatic': '🎭', 'fun': '🎉', 'party': '🎊', 'groovy': '🕺',
        'melancholic': '💭', 'hopeful': '🌟', 'dreamy': '☁️', 'emotional': '💫',
        'adventure': '🚀', 'summer': '🌞', 'retro': '📻', 'space': '🌌',
        'meditative': '🧘', 'motivational': '🔥', 'positive': '✨'
    }
    return mood_emojis.get(mood, '🎵')


def create_mood_analysis(songs: List[Dict], user_mood_scores: Optional[Dict[str, float]] = None) -> str:
    """Create mood distribution analysis for recommended songs"""
    if not songs:
        return ""
    
    mood_counts = {}
    total_moods = 0
    
    for song in songs:
        for mood_data in song.get('predicted_moods', [])[:3]:  # Top 3 moods per song
            mood = mood_data['mood']
            score = mood_data['score']
            if mood not in mood_counts:
                mood_counts[mood] = {'count': 0, 'total_score': 0}
            mood_counts[mood]['count'] += 1
            mood_counts[mood]['total_score'] += score
            total_moods += 1
    
    # Sort by frequency
    sorted_moods = sorted(mood_counts.items(), key=lambda x: x[1]['count'], reverse=True)
    
    analysis = "## 📊 Mood Analysis\n\n"
    
    # Show user's requested moods vs found moods
    if user_mood_scores:
        analysis += "**🎯 Your Requested Moods vs Found Moods:**\n\n"
        
        requested_moods = sorted(user_mood_scores.items(), key=lambda x: x[1], reverse=True)
        found_moods = {mood: data for mood, data in mood_counts.items()}
        
        for mood, user_score in requested_moods:
            emoji = get_mood_emoji(mood)
            if mood in found_moods:
                count = found_moods[mood]['count']
                avg_score = found_moods[mood]['total_score'] / found_moods[mood]['count']
                percentage = (count / total_moods) * 100
                analysis += f"- {emoji} **{mood.title()}**: ✅ Found in {count} songs ({percentage:.1f}%) - Avg: {avg_score:.2f}\n"
            else:
                analysis += f"- {emoji} **{mood.title()}**: ❌ Not found in recommendations\n"
        
        analysis += "\n**🎵 Additional Moods Found:**\n\n"
        
        # Show moods found but not requested
        additional_moods = [(mood, data) for mood, data in sorted_moods if mood not in user_mood_scores]
        for mood, data in additional_moods[:5]:  # Top 5 additional moods
            percentage = (data['count'] / total_moods) * 100
            avg_score = data['total_score'] / data['count']
            emoji = get_mood_emoji(mood)
            analysis += f"- {emoji} **{mood.title()}**: {data['count']} songs ({percentage:.1f}%) - Avg: {avg_score:.2f}\n"
    
    else:
        analysis += "**Mood distribution in recommendations:**\n\n"
        
        for mood, data in sorted_moods[:8]:  # Top 8 moods
            percentage = (data['count'] / total_moods) * 100
            avg_score = data['total_score'] / data['count']
            emoji = get_mood_emoji(mood)
            analysis += f"- {emoji} **{mood.title()}**: {data['count']} songs ({percentage:.1f}%) - Avg score: {avg_score:.2f}\n"
    
    return analysis


def create_recommendation_stats(songs: List[Dict]) -> str:
    """Create statistics summary for recommendations"""
    if not songs:
        return ""
    
    # Calculate statistics
    valences = [s['valence'] for s in songs if s.get('valence') is not None]
    arousals = [s['arousal'] for s in songs if s.get('arousal') is not None]
    view_counts = [s['view_count'] for s in songs if s.get('view_count')]
    
    stats = "## 📈 Recommendation Statistics\n\n"
    
    if valences:
        avg_valence = sum(valences) / len(valences)
        stats += f"**Average Emotional Profile:**\n"
        stats += f"- 😊 Valence (positivity): {avg_valence:.2f}\n"
        
    if arousals:
        avg_arousal = sum(arousals) / len(arousals)
        stats += f"- ⚡ Arousal (energy): {avg_arousal:.2f}\n\n"
    
    if view_counts:
        avg_views = sum(view_counts) / len(view_counts)
        max_views = max(view_counts)
        stats += f"**Popularity Metrics:**\n"
        stats += f"- 📊 Average views: {avg_views:,.0f}\n"
        stats += f"- 🔥 Most popular: {max_views:,} views\n"
    
    return stats


def get_example_prompts() -> List[str]:
    """Get list of example prompts for the UI"""
    return [
        "I want energetic and upbeat music for my workout",
        "Something romantic and soft for a date night", 
        "Calm and meditative music for studying",
        "Happy and fun songs for a party",
        "Sad and melancholic music to match my mood",
        "Powerful and dramatic music for motivation",
        "Dreamy and ethereal soundscape for relaxation"
    ]


def create_app_description() -> str:
    """Create the main app description markdown"""
    return """
    # 🐸 MuFrog - Music Emotion Analysis Demo
    
    Welcome to MuFrog! This demo showcases our Music2Emotion analysis system that predicts emotional moods from music.
    
    ## Features:
    - **Browse Database**: Explore our analyzed music collection with mood predictions
    - **Smart Recommendations**: Get song recommendations based on your mood description
    - **Mood Analytics**: View statistics about emotional patterns in music
    """


def create_browse_tab_description() -> str:
    """Create description for the browse tab"""
    return "### Explore our analyzed music collection"


def create_recommendations_tab_description() -> str:
    """Create description for the recommendations tab"""
    return "### Describe your mood and get personalized song recommendations"


def create_analytics_tab_description() -> str:
    """Create description for the analytics tab"""
    return "### Dataset insights and mood distribution"


def create_mood_categories_info(mood_categories: str) -> str:
    """Create information about available mood categories"""
    return f"""
    **Available mood categories from Music2Emotion analysis:**
    
    {mood_categories}
    
    Use these keywords or describe your mood in natural language - our system will try to match your description to these emotional categories.
    """


def create_about_section() -> str:
    """Create the about section for the app"""
    return """
    ---
    **About MuFrog**: This demo uses machine learning to analyze music and predict emotional moods. 
    The recommendation system matches your described mood with songs in our database using similarity scoring.
    
    *Future improvements will include LLM-powered mood interpretation and advanced matching algorithms.*
    """


def format_error_message(error_type: str, details: str = "") -> str:
    """Format error messages for consistent display"""
    if error_type == "no_data":
        return "No music data available. Please check if the analyzed_metadata_latest.json file exists."
    elif error_type == "no_input":
        return "Please enter a description of what kind of music you're looking for."
    elif error_type == "no_moods":
        return "Could not identify any moods from your prompt. Try using more descriptive emotional words."
    else:
        return f"Error: {details}" if details else "An unexpected error occurred."


def create_input_placeholder() -> str:
    """Create placeholder text for the user input"""
    return "e.g., 'I want something upbeat and energetic for working out' or 'I need calm, peaceful music for studying'"


def format_dataset_info(total_songs: int) -> str:
    """Format dataset information for display"""
    return f"**Total songs displayed:** {total_songs}"


def create_custom_css() -> str:
    """Create custom CSS for enhanced styling"""
    return """
    <style>
    /* Custom styling for the MuFrog demo */
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    #summary_output {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    
    #summary_output * {
        color: white !important;
    }
    
    #detailed_output {
        background: #f8f9fa;
        color: #212529 !important;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    
    #detailed_output * {
        color: #212529 !important;
    }
    
    #analysis_output {
        background: #e8f4f8;
        color: #155724 !important;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    #analysis_output * {
        color: #155724 !important;
    }
    
    #stats_output {
        background: #fff3cd;
        color: #856404 !important;
        padding: 10px;
        border-radius: 6px;
        border-left: 3px solid #ffc107;
        margin-top: 10px;
    }
    
    #stats_output * {
        color: #856404 !important;
    }
    
    .recommendation-card {
        background: white;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        color: #212529 !important;
    }
    
    .recommendation-card * {
        color: #212529 !important;
    }
    
    .mood-tag {
        display: inline-block;
        background: #e3f2fd;
        color: #1976d2 !important;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 0.85em;
        margin: 2px;
    }
    
    /* Override Gradio's markdown styling in custom components */
    .gradio-markdown {
        color: inherit !important;
    }
    
    .gradio-markdown * {
        color: inherit !important;
    }
    
    /* Ensure better contrast for all text elements */
    .gradio-container .markdown p,
    .gradio-container .markdown h1,
    .gradio-container .markdown h2,
    .gradio-container .markdown h3,
    .gradio-container .markdown h4,
    .gradio-container .markdown h5,
    .gradio-container .markdown h6,
    .gradio-container .markdown span,
    .gradio-container .markdown div {
        color: inherit !important;
    }
    </style>
    """


def create_download_interface_components():
    """Create components for the download interface"""
    return {
        'description': """
### 🎵 Download Songs from YouTube Playlists

Add YouTube playlist URLs to download songs for analysis. The system will:
- Skip duplicate songs automatically
- Queue multiple playlists for batch downloading
- Show real-time progress and status
- Prepare songs for Music2Emotion analysis

**Supported URLs:**
- YouTube playlist URLs
- Individual YouTube video URLs
- Mix playlists and auto-generated playlists
""",
        'input_placeholder': "https://www.youtube.com/playlist?list=...",
        'queue_header': "## 📋 Download Queue & Progress",
        'completed_header': "## ✅ Completed Downloads",
        'downloaded_songs_header': "## 🎵 Downloaded Songs Library"
    }


def format_download_status(status: Dict) -> str:
    """Format download status for display"""
    if not status:
        return "No download activity"
    
    result = f"**📊 Download Status:**\n\n"
    result += f"- Queue Length: {status.get('queue_length', 0)}\n"
    result += f"- Currently Downloading: {'Yes' if status.get('is_downloading') else 'No'}\n"
    result += f"- Completed Downloads: {status.get('completed_count', 0)}\n"
    result += f"- Total Downloaded Songs: {status.get('total_downloaded_songs', 0)}\n\n"
    
    # Current downloads
    current = status.get('current_downloads', {})
    if current:
        result += "**🔄 Current Downloads:**\n\n"
        for url, progress in current.items():
            playlist_title = progress.get('playlist_title', 'Unknown')
            total = progress.get('total_songs', 0)
            completed = progress.get('completed_songs', 0)
            current_song = progress.get('current_song', 'Unknown')
            status_text = progress.get('status', 'unknown')
            
            result += f"**{playlist_title}**\n"
            result += f"- Progress: {completed}/{total} songs\n"
            result += f"- Current: {current_song}\n"
            result += f"- Status: {status_text.title()}\n\n"
    
    return result


def format_downloaded_songs_table(songs: List[Dict]) -> str:
    """Format downloaded songs as a table for display"""
    if not songs:
        return "No songs downloaded yet."
    
    result = f"**📚 Downloaded Songs Library ({len(songs)} songs):**\n\n"
    
    for i, song in enumerate(songs[:20], 1):  # Show first 20 songs
        title = song.get('title', 'Unknown')
        artist = song.get('artist', 'Unknown')
        download_date = song.get('download_date', 'Unknown')
        
        # Format date
        if download_date != 'Unknown':
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(download_date.replace('Z', '+00:00'))
                download_date = dt.strftime('%Y-%m-%d %H:%M')
            except:
                pass
        
        result += f"{i}. **{title}** by {artist}\n"
        result += f"   Downloaded: {download_date}\n\n"
    
    if len(songs) > 20:
        result += f"\n*... and {len(songs) - 20} more songs*"
    
    return result


def create_download_help_text() -> str:
    """Create help text for the download interface"""
    return """
### 💡 How to Use:

1. **Add Playlist URLs**: Paste YouTube playlist or video URLs in the input box
2. **Queue Multiple Playlists**: Add several playlists - they'll be processed in order
3. **Start Downloads**: Click "Start Downloads" to begin processing the queue
4. **Monitor Progress**: Watch real-time progress in the status section
5. **Duplicate Detection**: The system automatically skips songs you've already downloaded

### 🔧 Features:

- **Smart Duplicate Detection**: Uses title + artist matching to avoid re-downloading
- **Concurrent Downloads**: Downloads multiple songs simultaneously for speed
- **Queue Management**: Add multiple playlists and process them in batch
- **Progress Tracking**: Real-time status updates and completion notifications
- **Error Handling**: Graceful handling of unavailable videos or network issues

### ⚠️ Important Notes:

- Downloads are saved in the `downloaded_songs` folder
- Only audio is downloaded (MP3 format)
- Metadata is automatically extracted and saved
- Downloaded songs will be ready for Music2Emotion analysis
"""
