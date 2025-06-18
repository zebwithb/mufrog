"""
MuFrog Gradio Demo - Main Application
Music Emotion Analysis and Recommendation System
"""
import gradio as gr
from typing import List, Dict, Any
from data_loader import initialize_data, create_songs_dataframe, get_mood_statistics
from mood_classifier import format_mood_categories, classify_user_prompt
from recommender import find_matching_songs
from ui_components import (
    format_recommendations, format_recommendations_enhanced, get_example_prompts, 
    create_app_description, create_browse_tab_description, create_recommendations_tab_description,
    create_analytics_tab_description, create_mood_categories_info,
    create_about_section, format_error_message, create_input_placeholder,
    format_dataset_info, create_recommendation_stats, create_custom_css,
    create_download_interface_components, format_download_status, 
    format_downloaded_songs_table, create_download_help_text
)


# Import playlist downloader
try:
    from playlist_downloader import get_downloader
    DOWNLOADER_AVAILABLE = True
    _downloader_module = True
except ImportError as e:
    print(f"Warning: Playlist downloader not available: {e}")
    DOWNLOADER_AVAILABLE = False
    _downloader_module = False
    
    # Create dummy function to avoid unbound errors
    def get_downloader():
        return None


# Import song map visualizer
try:
    from song_map_visualizer import get_song_map_visualizer
    SONG_MAP_AVAILABLE = True
    _song_map_module = True
except ImportError as e:
    print(f"Warning: Song map visualizer not available: {e}")
    SONG_MAP_AVAILABLE = False
    _song_map_module = False
    
    # Create dummy function to avoid unbound errors
    def get_song_map_visualizer():
        return None


# Import emotion map visualizer
try:
    from emotion_song_map import get_emotion_song_map_visualizer
    EMOTION_MAP_AVAILABLE = True
    _emotion_map_module = True
except ImportError as e:
    print(f"Warning: Emotion map visualizer not available: {e}")
    EMOTION_MAP_AVAILABLE = False
    _emotion_map_module = False
    
    # Create dummy function to avoid unbound errors
    def get_emotion_song_map_visualizer():
        return None


# Initialize data
music_data = initialize_data()


def recommend_songs_interface(user_prompt: str) -> tuple:
    """Interface function for Gradio song recommendations with enhanced output"""
    if not user_prompt.strip():
        error_msg = format_error_message("no_input")
        return error_msg, "", ""
    
    # Get user mood scores for matching analysis
    user_mood_scores = classify_user_prompt(user_prompt)
    songs, explanation = find_matching_songs(music_data, user_prompt, top_k=10)
    return format_recommendations_enhanced(songs, explanation, user_mood_scores)


def recommend_songs_with_stats(user_prompt: str) -> tuple:
    """Enhanced recommendation interface that includes stats"""
    if not user_prompt.strip():
        error_msg = format_error_message("no_input")
        return error_msg, "", "", ""
    
    # Get user mood scores for matching analysis
    user_mood_scores = classify_user_prompt(user_prompt)
    songs, explanation = find_matching_songs(music_data, user_prompt, top_k=10)
    summary, detailed, analysis = format_recommendations_enhanced(songs, explanation, user_mood_scores)
    stats = create_recommendation_stats(songs)
    
    return summary, detailed, analysis, stats


def refresh_statistics() -> str:
    """Refresh dataset statistics"""
    return get_mood_statistics(music_data)


# Download interface functions
def add_playlist_to_queue(playlist_url: str) -> tuple:
    """Add a playlist URL to the download queue"""
    if not DOWNLOADER_AVAILABLE:
        return "❌ Downloader not available. Please install yt-dlp: pip install yt-dlp", ""
    
    if not playlist_url.strip():
        return "❌ Please enter a valid YouTube URL", ""
    
    try:
        downloader = get_downloader()
        if downloader and downloader.add_playlist_to_queue(playlist_url.strip()):
            status_dict = downloader.get_queue_status()
            status = format_download_status_with_refresh_reminder(status_dict)
            return f"✅ Added to queue! Queue length: {status_dict['queue_length']}", status
        else:
            status_dict = downloader.get_queue_status() if downloader else {}
            status = format_download_status_with_refresh_reminder(status_dict)
            return "⚠️ URL already in queue", status
    except Exception as e:
        return f"❌ Error: {e}", ""


def start_downloads() -> tuple:
    """Start processing the download queue"""
    if not DOWNLOADER_AVAILABLE:
        return "❌ Downloader not available", ""
    
    try:
        downloader = get_downloader()
        if not downloader:
            return "❌ Downloader not initialized", ""
            
        if downloader.start_downloads():
            status_dict = downloader.get_queue_status()
            status = format_download_status_with_refresh_reminder(status_dict)
            songs_list = format_downloaded_songs_table(downloader.get_downloaded_songs_metadata())
            return status, songs_list
        else:
            status_info = downloader.get_queue_status()
            if status_info['is_downloading']:
                status = format_download_status_with_refresh_reminder(status_info)
            elif status_info['queue_length'] == 0:
                status = "ℹ️ No playlists in queue. Add some URLs first."
            else:
                status = "❌ Could not start downloads"
            songs_list = format_downloaded_songs_table(downloader.get_downloaded_songs_metadata())
            return status, songs_list
    except Exception as e:
        return f"❌ Error: {e}", ""


def stop_downloads() -> tuple:
    """Stop current downloads"""
    if not DOWNLOADER_AVAILABLE:
        return "❌ Downloader not available", ""
    
    try:
        downloader = get_downloader()
        if downloader:
            downloader.stop_downloads()
            status = "🛑 Download stop requested. Current downloads will finish."
            songs_list = format_downloaded_songs_table(downloader.get_downloaded_songs_metadata())
            return status, songs_list
        return "❌ Downloader not initialized", ""
    except Exception as e:
        return f"❌ Error: {e}", ""


def clear_download_queue() -> tuple:
    """Clear the download queue"""
    if not DOWNLOADER_AVAILABLE:
        return "❌ Downloader not available", "", ""
    
    try:
        downloader = get_downloader()
        if not downloader:
            return "❌ Downloader not initialized", "", ""
            
        if downloader.clear_queue():
            status_dict = downloader.get_queue_status()
            status = format_download_status_with_refresh_reminder(status_dict)
            songs_list = format_downloaded_songs_table(downloader.get_downloaded_songs_metadata())
            return "🗑️ Queue cleared!", status, songs_list
        else:
            status_dict = downloader.get_queue_status()
            status = format_download_status_with_refresh_reminder(status_dict)
            songs_list = format_downloaded_songs_table(downloader.get_downloaded_songs_metadata())
            return "⚠️ Cannot clear queue while downloading", status, songs_list
    except Exception as e:
        return f"❌ Error: {e}", "", ""


def refresh_download_status() -> tuple:
    """Refresh the download status display and downloaded songs list"""
    if not DOWNLOADER_AVAILABLE:
        return "❌ Downloader not available", "❌ Downloader not available"
    
    try:
        downloader = get_downloader()
        if downloader:
            status_dict = downloader.get_queue_status()
            status = format_download_status_with_refresh_reminder(status_dict)
            songs_list = format_downloaded_songs_table(downloader.get_downloaded_songs_metadata())
            return status, songs_list
        return "❌ Downloader not initialized", "❌ Downloader not initialized"
    except Exception as e:
        return f"❌ Error: {e}", f"❌ Error: {e}"


def get_downloaded_songs_list() -> str:
    """Get the list of downloaded songs"""
    if not DOWNLOADER_AVAILABLE:
        return "❌ Downloader not available"
    
    try:
        downloader = get_downloader()
        if downloader:
            songs = downloader.get_downloaded_songs_metadata()
            return format_downloaded_songs_table(songs)
        return "❌ Downloader not initialized"
    except Exception as e:
        return f"❌ Error: {e}"


# Add a function for auto-refresh with timer
def auto_refresh_downloads() -> tuple:
    """Auto-refresh download status and songs list"""
    if not DOWNLOADER_AVAILABLE:
        return "❌ Downloader not available", "❌ Downloader not available"
    
    try:
        downloader = get_downloader()
        if downloader:
            status_dict = downloader.get_queue_status()
            status = format_download_status_with_refresh_reminder(status_dict)
            songs_list = format_downloaded_songs_table(downloader.get_downloaded_songs_metadata())
            return status, songs_list
        return "❌ Downloader not initialized", "❌ Downloader not initialized"
    except Exception as e:
        return f"❌ Error: {e}", f"❌ Error: {e}"


def format_download_status_with_refresh_reminder(status_dict: dict) -> str:
    """Format download status with reminder to refresh for real-time updates"""
    from ui_components import format_download_status
    
    base_status = format_download_status(status_dict)
    
    if status_dict.get('is_downloading', False):
        refresh_reminder = "\n\n💡 **Tip**: Click '🔄 Refresh Status' periodically to see the latest progress and new downloads!"
        return base_status + refresh_reminder
    
    return base_status


# Song Map interface functions
def generate_song_map(method: str, color_by: str) -> tuple:
    """Generate a song map visualization"""
    if not SONG_MAP_AVAILABLE:
        error_msg = create_fallback_song_map()
        return None, error_msg
    
    try:
        visualizer = get_song_map_visualizer()
        if not visualizer:
            return None, "❌ Song Map visualizer not initialized"
        
        # Filter to only analyzed songs
        analyzed_songs = filter_analyzed_songs(music_data)
        
        # Check if we have enough data
        if len(analyzed_songs) < 3:
            total_songs = len(music_data)
            analyzed_count = len(analyzed_songs)
            return None, f"❌ Need at least 3 analyzed songs to create a map. Found {analyzed_count} analyzed out of {total_songs} total songs."
        
        # Generate the map
        fig = visualizer.generate_song_map(analyzed_songs, method=method, color_by=color_by)
        stats = visualizer.get_stats()
        
        return fig, f"✅ Song map generated successfully!\n\n{stats}\n\n**Songs analyzed:** {len(analyzed_songs)}"
    except Exception as e:
        return None, f"❌ Error generating song map: {e}"


def get_available_methods() -> list:
    """Get available dimensionality reduction methods"""
    if not SONG_MAP_AVAILABLE:
        return ["tsne"]  # Default fallback
    
    try:
        visualizer = get_song_map_visualizer()
        if visualizer:
            return visualizer.get_available_methods()
        return ["tsne", "pca"]
    except Exception:
        return ["tsne", "pca"]


def get_available_color_options() -> list:
    """Get available color-by options from the visualizer"""
    if not SONG_MAP_AVAILABLE:
        return ["top_mood", "valence", "arousal", "view_count"]  # Default fallback
    
    try:
        visualizer = get_song_map_visualizer()
        if visualizer:
            return visualizer.get_color_options()
        return ["top_mood", "cluster", "valence", "arousal", "view_count", "likes"]
    except Exception:
        return ["top_mood", "cluster", "valence", "arousal", "view_count"]


def create_fallback_song_map() -> str:
    """Create a fallback visualization when dependencies are missing"""
    return """
    📊 **Song Map Feature Unavailable**
    
    To use the interactive Song Map, please install the required dependencies:
    
    ```bash
    pip install plotly scikit-learn
    ```
    
    Optional (for better performance):
    ```bash
    pip install umap-learn
    ```
    
    **What you'll get with Song Map:**
    - Interactive 2D visualization of your entire music library
    - Songs clustered by emotional similarity
    - Hover details for each song
    - Multiple coloring options (mood, valence, arousal, popularity)
    - Export and sharing capabilities
    """
    

def filter_analyzed_songs(music_data) -> List[Dict[str, Any]]:
    """Filter to only include songs that have been analyzed (have mood predictions)"""
    return [song for song in music_data if song.get('predicted_moods')]


# Emotion Map interface functions
def generate_emotion_map(method: str, color_by: str, size_by_popularity: bool) -> tuple:
    """Generate an emotion map visualization based on fundamental emotions"""
    if not EMOTION_MAP_AVAILABLE:
        error_msg = create_fallback_emotion_map()
        return None, error_msg, ""
    
    try:
        visualizer = get_emotion_song_map_visualizer()
        if not visualizer:
            return None, "❌ Emotion Map visualizer not initialized", ""
        
        # Filter to only analyzed songs
        analyzed_songs = filter_analyzed_songs(music_data)
        
        # Check if we have enough data
        if len(analyzed_songs) < 3:
            total_songs = len(music_data)
            analyzed_count = len(analyzed_songs)
            return None, f"❌ Need at least 3 analyzed songs to create a map. Found {analyzed_count} analyzed out of {total_songs} total songs.", ""
        
        # Generate the emotion map
        fig = visualizer.create_emotion_map(
            analyzed_songs, 
            method=method, 
            color_by=color_by, 
            size_by_popularity=size_by_popularity
        )
        
        # Get statistics separately
        stats = visualizer.calculate_emotion_statistics(analyzed_songs)
        
        # Format stats for display
        stats_text = format_emotion_statistics(stats)
        
        return fig, f"✅ Emotion map generated successfully!\n\n**Analysis:** {len(analyzed_songs)} analyzed songs", stats_text
    except Exception as e:
        return None, f"❌ Error generating emotion map: {e}", ""


def get_emotion_color_options() -> list:
    """Get available color-by options for emotion maps"""
    if not EMOTION_MAP_AVAILABLE:
        return ["Dominant Emotion", "Joy/Ecstasy", "Sadness/Grief"]  # Default fallback
    
    try:
        visualizer = get_emotion_song_map_visualizer()
        if visualizer:
            return visualizer.get_color_options()
        return ["Dominant Emotion", "Joy/Ecstasy", "Sadness/Grief", "Fear/Tension", "Serenity/Peace", "Triumph/Pride", "Surprise/Anticipation"]
    except Exception:
        return ["Dominant Emotion", "Joy/Ecstasy", "Sadness/Grief"]


def format_emotion_statistics(stats: dict) -> str:
    """Format emotion statistics for display"""
    if not stats:
        return "No statistics available"
    
    output = ["### 📊 Emotion Analysis Summary\n"]
    
    # Total counts
    if 'total_songs' in stats:
        output.append(f"**Total Songs Analyzed:** {stats['total_songs']}")
    
    # Emotion distribution
    if 'emotion_distribution' in stats:
        output.append("\n**🎭 Fundamental Emotion Distribution:**")
        for emotion, count in stats['emotion_distribution'].items():
            percentage = (count / stats['total_songs'] * 100) if stats['total_songs'] > 0 else 0
            output.append(f"- **{emotion}**: {count} songs ({percentage:.1f}%)")
    
    # Average scores
    if 'average_scores' in stats:
        output.append("\n**📈 Average Emotion Scores:**")
        for emotion, score in stats['average_scores'].items():
            output.append(f"- **{emotion}**: {score:.2f}/1.0")
    
    # Top songs by emotion
    if 'top_songs_by_emotion' in stats:
        output.append("\n**🎵 Representative Songs:**")
        for emotion, song_info in stats['top_songs_by_emotion'].items():
            if song_info:
                title = song_info.get('title', 'Unknown')
                score = song_info.get('score', 0)
                output.append(f"- **{emotion}**: \"{title}\" (score: {score:.2f})")
    
    return "\n".join(output)


def create_fallback_emotion_map() -> str:
    """Create a fallback visualization when emotion map dependencies are missing"""
    return """
    🎭 **Emotion Map Feature Unavailable**
    
    To use the interactive Emotion Map, please install the required dependencies:
    
    ```bash
    pip install plotly scikit-learn
    ```
    
    **What you'll get with Emotion Map:**
    - Visualization based on fundamental emotions (Joy, Sadness, Fear, etc.)
    - Songs mapped by emotional recipes and patterns
    - Interactive exploration of emotional landscapes
    - Statistics on emotion distribution in your library
    - Discovery of songs by emotional similarity
    """


def create_gradio_interface():
    """Create the main Gradio interface"""
    
    # Create the Gradio interface
    with gr.Blocks(title="MuFrog - Music Emotion Analysis Demo", css=create_custom_css()) as demo:
        gr.HTML(create_custom_css())  # Add custom styling
        gr.Markdown(create_app_description())
        
        with gr.Tabs():
            # Tab 1: Browse Music Database
            with gr.TabItem("🎵 Browse Music Database"):
                gr.Markdown(create_browse_tab_description())
                
                # Create and display the dataframe
                df = create_songs_dataframe(music_data)
                
                if not df.empty:
                    data_table = gr.Dataframe(
                        value=df,
                        interactive=False,
                        wrap=True
                    )
                    
                    gr.Markdown(format_dataset_info(len(df)))
                else:
                    gr.Markdown(format_error_message("no_data"))
            
            # Tab 2: Song Recommendations
            with gr.TabItem("🎯 Get Song Recommendations"):
                gr.Markdown(create_recommendations_tab_description())
                
                gr.Markdown(create_mood_categories_info(format_mood_categories()))
                
                with gr.Row():
                    # Input Section
                    with gr.Column(scale=2):
                        user_input = gr.Textbox(
                            label="🎭 Describe your mood or what you're looking for",
                            placeholder=create_input_placeholder(),
                            lines=4,
                            max_lines=6
                        )
                        
                        recommend_btn = gr.Button("🎵 Get Recommendations", variant="primary", size="lg")
                        
                        # Quick mood selector
                        gr.Markdown("**Quick mood selection:**")
                        quick_moods = ["happy", "sad", "energetic", "calm", "romantic", "powerful"]
                        mood_buttons = []
                        with gr.Row():
                            for mood in quick_moods:
                                btn = gr.Button(f"{mood.title()}", size="sm", variant="secondary")
                                mood_buttons.append(btn)
                    
                    # Enhanced Output Section
                    with gr.Column(scale=3):
                        # Summary at the top
                        summary_output = gr.Markdown(
                            label="📋 Summary", 
                            value="Enter a mood description to get personalized song recommendations!",
                            elem_id="summary_output"
                        )
                        
                        # Tabbed detailed output
                        with gr.Tabs():
                            with gr.TabItem("🎶 Song List"):
                                detailed_output = gr.Markdown(
                                    label="Detailed Recommendations",
                                    elem_id="detailed_output"
                                )
                            
                            with gr.TabItem("📊 Analysis"):
                                analysis_output = gr.Markdown(
                                    label="Mood & Stats Analysis",
                                    elem_id="analysis_output"
                                )
                        
                        # Stats summary
                        stats_output = gr.Markdown(
                            label="📈 Quick Stats",
                            elem_id="stats_output"
                        )
                
                # Connect quick mood buttons after all components are defined
                for i, mood in enumerate(quick_moods):
                    mood_buttons[i].click(
                        fn=lambda m=mood: recommend_songs_with_stats(f"I want {m} music"),
                        outputs=[summary_output, detailed_output, analysis_output, stats_output]
                    )
                
                # Example prompts section
                with gr.Accordion("💡 Example Prompts", open=False):
                    gr.Markdown("Click any example to try it:")
                    example_prompts = get_example_prompts()
                    
                    with gr.Row():
                        for i, prompt in enumerate(example_prompts[:4]):  # First 4 examples
                            gr.Button(prompt, size="sm").click(
                                fn=lambda p=prompt: recommend_songs_with_stats(p),
                                outputs=[summary_output, detailed_output, analysis_output, stats_output]
                            )
                    
                    with gr.Row():
                        for i, prompt in enumerate(example_prompts[4:]):  # Remaining examples
                            gr.Button(prompt, size="sm").click(
                                fn=lambda p=prompt: recommend_songs_with_stats(p),
                                outputs=[summary_output, detailed_output, analysis_output, stats_output]
                            )
                
                # Connect the main button
                recommend_btn.click(
                    fn=recommend_songs_with_stats,
                    inputs=user_input,
                    outputs=[summary_output, detailed_output, analysis_output, stats_output]
                )
            
            # Tab 3: Dataset Analytics
            with gr.TabItem("📊 Mood Analytics"):
                gr.Markdown(create_analytics_tab_description())
                
                analytics_output = gr.Markdown(value=refresh_statistics())
                
                refresh_btn = gr.Button("Refresh Statistics")
                refresh_btn.click(
                    fn=refresh_statistics,
                    outputs=analytics_output
                )
            
            # Tab 4: Song Map Visualization
            with gr.TabItem("🗺️ Song Map"):
                gr.Markdown("""
                ### 🗺️ Interactive Song Map
                
                Explore your music library as a visual map! This feature uses advanced dimensionality reduction to plot 
                songs based on their emotional profiles. Songs with similar moods will appear close together, while 
                different songs will be far apart.
                
                **How it works:**
                - Your music library's 40+ mood dimensions are reduced to 2D coordinates
                - Each song becomes a dot on an interactive map
                - Hover over dots to see song details
                - Different colors represent different aspects of the music
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Map Settings")
                        
                        method_dropdown = gr.Dropdown(
                            label="Reduction Method",
                            choices=get_available_methods(),
                            value="tsne",
                            info="Algorithm to reduce dimensions"
                        )
                        
                        color_dropdown = gr.Dropdown(
                            label="Color By",
                            choices=get_available_color_options(),
                            value="top_mood",
                            info="What to color the map points by"
                        )
                        
                        generate_map_btn = gr.Button(
                            "🗺️ Generate Song Map", 
                            variant="primary", 
                            size="lg"
                        )
                        
                        map_status = gr.Markdown(
                            value="Click 'Generate Song Map' to create your visualization!",
                            elem_id="map_status"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Your Music Map")
                        
                        song_map_plot = gr.Plot(
                            label="Interactive Song Map",
                            elem_id="song_map"
                        )
                
                # Info section
                with gr.Accordion("💡 Understanding the Song Map", open=False):
                    gr.Markdown("""
                    **Dimensionality Reduction Methods:**
                    
                    - **t-SNE**: Great for finding clusters and local relationships. Songs with similar moods form tight groups.
                    - **PCA**: Shows the main variations in your music. Good for understanding overall patterns.
                    - **UMAP**: Balances local and global structure (if available).
                    
                    **Color Options:**
                    
                    - **Top Mood**: Each song colored by its strongest emotion
                    - **Cluster**: Automatically discovered song groups (implicit moods)
                    - **Valence**: Blue (sad) to red (happy) scale
                    - **Arousal**: Energy level from calm to energetic  
                    - **View Count**: Popularity on YouTube
                    - **Likes**: Number of likes (popularity metric)
                    - **Mood Confidence**: How confident the AI is about the top mood
                    
                    **Visual Enhancements:**
                    
                    - **Point Size**: Represents popularity (view count) - bigger = more popular
                    - **Clustering**: Automatically groups similar songs to discover hidden patterns
                    - **Interactive Hover**: Detailed song information on mouseover
                    
                    **Pro Tips:**
                    - Look for clusters of similar songs
                    - Explore the boundaries between different mood regions
                    - Use the map to discover songs you might like based on proximity
                    """)
                
                # Connect the generate button
                generate_map_btn.click(
                    fn=generate_song_map,
                    inputs=[method_dropdown, color_dropdown],
                    outputs=[song_map_plot, map_status]
                )
            
            # Tab 5: Emotion Map
            with gr.TabItem("🎭 Emotion Map"):
                gr.Markdown("""
                ### 🎭 Fundamental Emotion Analysis
                
                Discover the emotional essence of your music library! This advanced feature analyzes songs using 
                fundamental emotion recipes (Joy, Sadness, Fear, Serenity, Triumph, Surprise) and visualizes them 
                in an interactive 2D space.
                
                **How it works:**
                - Songs are analyzed using emotion-specific mood patterns
                - Each song is scored on 6 fundamental emotions
                - Advanced algorithms create a 2D emotional landscape
                - Similar emotional patterns cluster together
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎛️ Emotion Map Settings")
                        
                        emotion_method_dropdown = gr.Dropdown(
                            label="Reduction Method",
                            choices=get_available_methods(),
                            value="tsne",
                            info="Algorithm to reduce emotion dimensions"
                        )
                        
                        emotion_color_dropdown = gr.Dropdown(
                            label="Color By",
                            choices=get_emotion_color_options(),
                            value="Dominant Emotion",
                            info="What emotional aspect to color by"
                        )
                        
                        emotion_size_checkbox = gr.Checkbox(
                            label="Size by Popularity",
                            value=True,
                            info="Make popular songs larger"
                        )
                        
                        generate_emotion_map_btn = gr.Button(
                            "🎭 Generate Emotion Map", 
                            variant="primary", 
                            size="lg"
                        )
                        
                        emotion_map_status = gr.Markdown(
                            value="Click 'Generate Emotion Map' to explore your music's emotional landscape!",
                            elem_id="emotion_map_status"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Your Emotional Music Landscape")
                        
                        emotion_map_plot = gr.Plot(
                            label="Interactive Emotion Map",
                            elem_id="emotion_map"
                        )
                
                # Statistics section
                with gr.Row():
                    emotion_statistics = gr.Markdown(
                        value="Generate an emotion map to see detailed statistics about your music's emotional patterns.",
                        elem_id="emotion_statistics"
                    )
                
                # Info section
                with gr.Accordion("💡 Understanding Fundamental Emotions", open=False):
                    gr.Markdown("""
                    **Fundamental Emotions Analyzed:**
                    
                    - **🎉 Joy/Ecstasy**: Jubilant pop songs, festival anthems, celebratory music
                    - **😢 Sadness/Grief**: Melancholic ballads, lonely melodies, sorrowful themes
                    - **😨 Fear/Tension**: Suspenseful tracks, anxiety-inducing rhythms, dramatic tension
                    - **😌 Serenity/Peace**: Calm meditative music, peaceful ambience, relaxing sounds
                    - **🏆 Triumph/Pride**: Victory anthems, empowering songs, achievement themes
                    - **😮 Surprise/Anticipation**: Unexpected musical elements, building excitement
                    
                    **Color Options:**
                    
                    - **Dominant Emotion**: Color by the strongest detected emotion
                    - **Individual Emotions**: Continuous scale for each specific emotion
                    
                    **Visual Features:**
                    
                    - **Interactive Hover**: Detailed song and emotion information
                    - **Popularity Sizing**: Bigger dots = more popular songs
                    - **Emotional Clustering**: Similar emotions appear close together
                    - **Statistics**: Distribution and patterns in your music library
                    
                    **Analysis Methods:**
                    
                    - **t-SNE**: Excellent for discovering emotional clusters and patterns
                    - **PCA**: Shows main emotional variations in your library
                    
                    **Pro Tips:**
                    - Look for emotional clusters to discover music patterns
                    - Explore boundaries between different emotions
                    - Use the map to find songs that match your current mood
                    - Check statistics to understand your music's emotional profile
                    """)
                
                # Connect the generate button
                generate_emotion_map_btn.click(
                    fn=generate_emotion_map,
                    inputs=[emotion_method_dropdown, emotion_color_dropdown, emotion_size_checkbox],
                    outputs=[emotion_map_plot, emotion_map_status, emotion_statistics]
                )
            
            # Tab 6: Download Songs
            with gr.TabItem("⬇️ Download Songs"):
                download_components = create_download_interface_components()
                gr.Markdown(download_components['description'])
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📥 Add Playlists to Queue")
                        
                        playlist_url_input = gr.Textbox(
                            label="YouTube Playlist URL",
                            placeholder=download_components['input_placeholder'],
                            lines=2
                        )
                        
                        with gr.Row():
                            add_queue_btn = gr.Button("➕ Add to Queue", variant="primary")
                            start_btn = gr.Button("🚀 Start Downloads", variant="primary")
                        
                        with gr.Row():
                            stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                            clear_btn = gr.Button("🗑️ Clear Queue", variant="secondary")
                            refresh_btn = gr.Button("🔄 Refresh Status", variant="secondary")
                    
                    with gr.Column(scale=3):
                        gr.Markdown("### 📊 Download Status")
                        
                        status_output = gr.Markdown(
                            value="Ready to download. Add playlist URLs and click 'Start Downloads'.",
                            elem_id="download_status"
                        )
                        
                        add_result_output = gr.Markdown(
                            value="",
                            elem_id="add_result"
                        )
                
                # Downloaded songs section
                with gr.Accordion("📚 Downloaded Songs Library", open=False):
                    downloaded_songs_display = gr.Markdown(
                        value=get_downloaded_songs_list(),
                        elem_id="downloaded_songs"
                    )
                    
                    refresh_songs_btn = gr.Button("🔄 Refresh Songs List")
                
                # Help section
                with gr.Accordion("💡 Help & Instructions", open=False):
                    gr.Markdown(create_download_help_text())
                
                # Connect download interface buttons
                add_queue_btn.click(
                    fn=add_playlist_to_queue,
                    inputs=playlist_url_input,
                    outputs=[add_result_output, status_output]
                )
                
                start_btn.click(
                    fn=start_downloads,
                    outputs=[status_output, downloaded_songs_display]
                )
                
                stop_btn.click(
                    fn=stop_downloads,
                    outputs=[status_output, downloaded_songs_display]
                )
                
                clear_btn.click(
                    fn=clear_download_queue,
                    outputs=[add_result_output, status_output, downloaded_songs_display]
                )
                
                refresh_btn.click(
                    fn=refresh_download_status,
                    outputs=[status_output, downloaded_songs_display]
                )
                
                refresh_songs_btn.click(
                    fn=get_downloaded_songs_list,
                    outputs=downloaded_songs_display
                )
        
        gr.Markdown(create_about_section())
    
    return demo


def main():
    """Main function to launch the Gradio app"""
    demo = create_gradio_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()



