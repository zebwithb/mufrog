"""
Music Emotion Recognition Application

This module analyzes audio files to extract emotional characteristics and audio features.
It uses deep learning models to predict valence, arousal, and mood tags from music.
The application supports analyzing full audio files and shorter segments for comparison.
"""

# Standard library imports
import os
import sys
from pathlib import Path
import json
import time
from typing import Dict, Any

# Third-party imports
import librosa
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pydub import AudioSegment
import torch
import torchaudio
from tqdm import tqdm

# Debug purposes - keep before Music2Emotion import
print("Current working directory: ", os.getcwd())
sys.path.append("..")

# Local application imports
from Music2Emotion.music2emo import Music2emo


def get_first_mp3():
    """Get the first MP3 file from the songs directory."""
    mp3_path = Path(__file__).parent / "scripts" / "songs" / "mp3" / "Adele - Someone Like You (Official Music Video)-hLQl3WQQoQ0.mp3"
            
    try:
        if not mp3_path.is_file():
            raise FileNotFoundError("MP3 file not found")
        print(f"Selected audio file: {mp3_path.name}")
        return str(mp3_path)
    except FileNotFoundError as e:
        print(f"Error finding MP3 file: {str(e)}")
        sys.exit(1)

def extract_audio_features(input_path, start_time=None, duration=None, output_path=None):
    """
    Extract audio segment and high-level features from an audio file.
    
    Args:
        input_path: Path to the audio file
        start_time: Start time in seconds (optional)
        duration: Duration in seconds (optional)
        output_path: Path to save the extracted segment (optional)
        
    Returns:
        Dictionary containing audio features and path to extracted segment if provided
    """
    # TODO - Analyze segments of audio files to target specific parts of the song
    # TODO - Audio analysis visuals
    # TODO - Concurrent processing of audio files within constraints of hardware
    
    try:
        # Load with pydub for MP3 handling (preserves format for MERT)
        audio = AudioSegment.from_file(input_path)
        
        # Extract segment if specified
        if start_time is not None and duration is not None:
            # Convert to milliseconds for pydub
            start_ms = int(start_time * 1000)
            duration_ms = int(duration * 1000)
            segment = audio[start_ms:start_ms + duration_ms]
            
            if output_path:
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                segment.export(output_path, format="mp3")
                print(f"Extracted audio segment: {output_path}")
        else:
            segment = audio
            
        # Load with librosa for feature extraction
        y, sr = librosa.load(
            input_path, 
            offset=start_time, 
            duration=duration) if start_time is not None else librosa.load(input_path)
        
        # Extract features
        features = {}
        
        # 1. Tempo/BPM
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['bpm'] = float(tempo)
        
        # 2. Spectral centroid (brightness)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
        
        # 3. RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_energy_mean'] = float(np.mean(rms))
        
        # 4. Zero-crossing rate (noisiness/harshness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zero_crossing_rate_mean'] = float(np.mean(zcr))
        
        # 5. Spectral contrast (difference between peaks and valleys)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features['spectral_contrast_mean'] = float(np.mean(contrast))
        
        # 6. Chromagram (harmony/key distribution)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = [float(np.mean(c)) for c in chroma]
        
        # Format output
        result = {
            'features': features,
            'duration': len(segment) / 1000  # in seconds
        }
        
        if output_path:
            result['output_path'] = output_path
            
        return result
        
    except Exception as e:
        print(f"Audio extraction/analysis error: {str(e)}")
        return None

class GPUNotAvailableError(Exception):
    """Exception raised when GPU acceleration is required but not available."""
    pass

def extract_audio_segment(input_path, start_time, duration, output_path):
    """Extracts an audio segment from the input audio file using torchaudio."""
    try:
        # Load the audio file
        waveform, sample_rate = torchaudio.load(input_path)
        
        # Convert time in seconds to frames
        start_frame = int(start_time * sample_rate)
        duration_frames = int(duration * sample_rate)
        end_frame = start_frame + duration_frames
        
        # Extract the segment
        segment = waveform[:, start_frame:end_frame]
        
        # Save the segment
        torchaudio.save(output_path, segment, sample_rate)
        
        print(f"Extracted audio segment: {output_path}")
        return output_path
    except Exception as e:
        print(f"Audio extraction error: {str(e)}")
        return None
    
def check_gpu_availability():
    """Check if GPU is available and return detailed information."""
    gpu_available = torch.cuda.is_available()
    print(f'CUDA Available: {gpu_available}')
    
    if gpu_available:
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
        return True
    error_msg = (
        "\nGPU Acceleration is not available. Diagnostic information:"
        f"\n• PyTorch version: {torch.__version__}"
        f"\n• CUDA version: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'Not available'}"
        f"\n• CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}"
        "\n\nPossible solutions:"
        "\n1. Verify NVIDIA drivers are installed"
        "\n2. Ensure compatible CUDA (12.6) is properly installed"
        "\n3. Check if PyTorch was installed with CUDA support"
        "\n4. Verify system PATH includes CUDA directories"
        "\n5. If using pipenv ensure to run 'pipenv run pip install' with correct packages"
    )
    raise GPUNotAvailableError(error_msg)

def display_analysis_results(title, audio_features, emotion_output):
    """Helper function to display analysis results in a consistent format."""
    print(f"\n{title}")
    print("-" * len(title))
    
    # Audio Features
    print("🎵 Audio Features:")
    print(f"   - BPM: {audio_features['features']['bpm']:.1f}")
    print(f"   - Brightness: {audio_features['features']['spectral_centroid_mean']:.2f}")
    print(f"   - Energy: {audio_features['features']['rms_energy_mean']:.4f}")
    print(f"   - Duration: {audio_features['duration']:.2f}s")
    
    # Emotion Analysis
    print("\n🎭 Emotion Analysis:")
    print(f"   - Valence: {emotion_output['valence']:.2f} (Scale: 1-9)")
    print(f"   - Arousal: {emotion_output['arousal']:.2f} (Scale: 1-9)")
    
    # Top Predicted Moods
    print("\n✨ Top Predicted Moods:")
    for mood in emotion_output['predicted_moods']:
        print(f"   - {mood['mood']}: {mood['score']:.4f}")

def analyze_mp3_collection(
    mp3_dir: Path,
    metadata_path: Path,
    output_path: Path
) -> None:
    """
    Analyze all MP3s and update metadata with emotion/audio features.
    
    Args:
        mp3_dir: Directory containing MP3 files
        metadata_path: Path to metadata JSON file
        output_path: Where to save updated metadata
    """
    try:
        # Check GPU availability first
        check_gpu_availability()
        
        # Load metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Initialize model
        music2emo = Music2emo()
        print("Initialized Music2Emotion model")
        
        # Create lookup dictionary
        metadata_lookup = {item.get("title", ""): item for item in metadata}
        
        # Process each MP3
        print(f"\nProcessing {len(list(mp3_dir.glob('*.mp3')))} MP3 files...")
        for mp3_path in tqdm(list(mp3_dir.glob("*.mp3"))):
            try:
                # Get metadata entry
                title = mp3_path.stem
                if title not in metadata_lookup:
                    print(f"\nWarning: No metadata found for {title}")
                    continue
                
                # Extract features
                audio_features = extract_audio_features(str(mp3_path))
                if not audio_features:
                    print(f"\nError extracting features from {title}")
                    continue
                
                # Get emotion predictions
                emotion_output = music2emo.predict(str(mp3_path))
                
                # Update metadata
                metadata_lookup[title].update({
                    "valence": float(emotion_output["valence"]),
                    "arousal": float(emotion_output["arousal"]),
                    "predicted_moods": emotion_output["predicted_moods"],
                    "audio_features": audio_features["features"]
                })
                
                # Add small delay to prevent overload
                time.sleep(0.1)
                
            except Exception as e:
                print(f"\nError processing {mp3_path.name}: {str(e)}")
                continue
        
        # Save updated metadata
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"\nAnalysis complete. Results saved to {output_path}")
        
    except GPUNotAvailableError as e:
        print(str(e))
        return
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return

def plot_emotion_results(output_dict, title, output_file=None):
    """
    Create and save a bar chart visualization of emotion prediction results.
    
    Args:
        output_dict: Dictionary containing emotion prediction results
        title: Title for the plot
        output_file: Path to save the plot image (if None, the plot will be displayed)
    """
    # Handle the case where output_dict['predicted_moods'] is a list of dictionaries
    all_moods = {}
    
    # Combine all mood predictions from the list
    if 'predicted_moods' in output_dict and isinstance(output_dict['predicted_moods'], list):
        for mood_dict in output_dict['predicted_moods']:
            if 'mood' in mood_dict:
                for mood, value in mood_dict['mood'].items():
                    if mood in all_moods:
                        all_moods[mood] += value
                    else:
                        all_moods[mood] = value
    else:
        # Fallback: try to use the dictionary directly
        all_moods = output_dict
    
    # Extract emotions and their values
    emotions = list(all_moods.keys())
    values = list(all_moods.values())
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar chart
    bars = ax.bar(emotions, values, color='steelblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Set labels and title
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Prediction Score')
    ax.set_title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    
    plt.close()

# Analyze full audio
def main():
    """Main function with support for single or batch analysis."""
    try:
        check_gpu_availability()
        
        # Ask user for analysis mode
        print("\nSelect analysis mode:")
        print("1. Single song analysis")
        print("2. Batch analysis of all MP3s")
        mode = input("Enter mode (1 or 2): ")
        
        if mode == "1":
            input_audio = get_first_mp3()
            music2emo = Music2emo()
            # Analyze full audio with features
            print("\nAnalyzing audio segments...")
            print("=" * 50)
            
            # Full audio analysis
            full_audio_features = extract_audio_features(input_audio)
            output_dic_full = music2emo.predict(input_audio)
            display_analysis_results("🎼 Full Audio Analysis", full_audio_features, output_dic_full)
            plot_emotion_results(output_dic_full, "Full Audio Emotion Analysis", "output/full_audio_emotions.png")
            
            # 5-second segment analysis
            output_15s = "scripts/songs/mp3/take_on_me_5s.mp3"
            segment_5s_features = extract_audio_features(input_audio, 0, 15, output_15s)
            output_dic_5s = music2emo.predict(output_15s)
            display_analysis_results("🎵 5-Second Segment Analysis", segment_5s_features, output_dic_5s)
            plot_emotion_results(output_dic_5s, "5-Second Segment Emotion Analysis", "output/5s_segment_emotions.png")

            
            # 30-second segment analysis
            output_30s = "scripts/songs/mp3/take_on_me_30s.mp3"
            segment_30s_features = extract_audio_features(input_audio, 0, 30, output_30s)
            output_dic_30s = music2emo.predict(output_30s)
            display_analysis_results("🎵 30-Second Segment Analysis", segment_30s_features, output_dic_30s)
            plot_emotion_results(output_dic_30s, "30-Second Segment Emotion Analysis", "output/30s_segment_emotions.png")
            
            
            
            print("\n" + "=" * 50)
            print("Analysis complete!")
        
        elif mode == "2":
            # Batch analysis
            base_dir = Path(__file__).parent
            mp3_dir = base_dir / "scripts" / "songs" / "mp3"
            metadata_path = base_dir / "scripts" / "songs" / "all_cleaned_metadata.json"
            output_path = base_dir / "scripts" / "songs" / "analyzed_metadata.json"
            
            analyze_mp3_collection(mp3_dir, metadata_path, output_path)
        
        else:
            print("Invalid mode selected")
            return 1
        
    except GPUNotAvailableError as e:
        print(str(e))
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
