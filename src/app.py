import sys
import os
import json

import librosa
import numpy as np
from pathlib import Path
from pydub import AudioSegment

import torch
import torchaudio

#debug purposes
print("Current working directory: ", os.getcwd())
sys.path.append("..")

from Music2Emotion.music2emo import Music2emo

# Replace the empty input_audio line with:
def get_first_mp3():
    """Get the first MP3 file from the songs directory."""
    mp3_dir = Path(__file__).parent / "scripts" / "songs" / "mp3"
    try:
        # Get first MP3 file
        first_mp3 = next(mp3_dir.glob("*.mp3"))
        if not first_mp3.is_file():
            raise FileNotFoundError("No MP3 files found in the songs directory")
        print(f"Selected audio file: {first_mp3.name}")
        return str(first_mp3)
    except (StopIteration, FileNotFoundError) as e:
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

# Analyze full audio
def main():
    try:
        check_gpu_availability()
        input_audio = get_first_mp3()
        music2emo = Music2emo()
        # Analyze full audio with features
        print("\nAnalyzing audio segments...")
        print("=" * 50)
        
        # Full audio analysis
        full_audio_features = extract_audio_features(input_audio)
        output_dic_full = music2emo.predict(input_audio)
        display_analysis_results("🎼 Full Audio Analysis", full_audio_features, output_dic_full)
        
        # 5-second segment analysis
        output_5s = "scripts/songs/mp3/take_on_me_5s.mp3"
        segment_5s_features = extract_audio_features(input_audio, 0, 5, output_5s)
        output_dic_5s = music2emo.predict(output_5s)
        display_analysis_results("🎵 5-Second Segment Analysis", segment_5s_features, output_dic_5s)
        
        # 30-second segment analysis
        output_30s = "scripts/songs/mp3/take_on_me_30s.mp3"
        segment_30s_features = extract_audio_features(input_audio, 0, 30, output_30s)
        output_dic_30s = music2emo.predict(output_30s)
        display_analysis_results("🎵 30-Second Segment Analysis", segment_30s_features, output_dic_30s)
        
        print("\n" + "=" * 50)
        print("Analysis complete!")
        
    except GPUNotAvailableError as e:
        print(str(e))
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
