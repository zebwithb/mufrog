import sys
import os
import torchaudio
import torch

print("Current working directory: ", os.getcwd())
sys.path.append("..")
from Music2Emotion.music2emo import Music2emo

input_audio = "scripts/songs/mp3/a-ha - Take On Me (Official Video) [Remastered in 4K]-djV11Xbc914.mp3"

music2emo = Music2emo()

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

# Analyze full audio
def main():
    try:
        check_gpu_availability()
        
        # Your existing code here
        print("Analyzing full audio...")
        output_dic_full = music2emo.predict(input_audio)
        valence_full = output_dic_full["valence"]
        arousal_full = output_dic_full["arousal"]
        predicted_moods_full = output_dic_full["predicted_moods"]

        # Extract and analyze 5-second segment
        print("Extracting and analyzing 5-second segment...")
        output_5s = "scripts/songs/mp3/take_on_me_5s.mp3"
        extract_audio_segment(input_audio, 0, 5, output_5s)
        output_dic_5s = music2emo.predict(output_5s)
        valence_5s = output_dic_5s["valence"]
        arousal_5s = output_dic_5s["arousal"]
        predicted_moods_5s = output_dic_5s["predicted_moods"]

        # Extract and analyze 30-second segment
        print("Extracting and analyzing 30-second segment...")
        output_30s = "scripts/songs/mp3/take_on_me_30s.mp3"
        extract_audio_segment(input_audio, 0, 30, output_30s)
        output_dic_30s = music2emo.predict(output_30s)
        valence_30s = output_dic_30s["valence"]
        arousal_30s = output_dic_30s["arousal"]
        predicted_moods_30s = output_dic_30s["predicted_moods"]

        # --- Extended Analysis Output ---
        print("\n🎵 **Music Emotion Recognition Results - Temporal Comparison** 🎵")
        print("-" * 50)

        print("\n**Full Audio Analysis:**")
        print("-----------------------")
        if predicted_moods_full:
            print("🎭 **Predicted Mood Tags (with Probabilities):**")
            for mood_data in predicted_moods_full:
                print(f"   - {mood_data['mood']}: {mood_data['score']:.4f}")
        else:
            print("🎭 **Predicted Mood Tags: None**")
        print(f"💖 **Valence:** {valence_full:.2f} (Scale: 1-9)")
        print(f"⚡ **Arousal:** {arousal_full:.2f} (Scale: 1-9)")

        print("\n**5-Second Segment Analysis:**")
        print("-----------------------------")
        if predicted_moods_5s:
            print("🎭 **Predicted Mood Tags (with Probabilities):**")
            for mood_data in predicted_moods_5s:
                print(f"   - {mood_data['mood']}: {mood_data['score']:.4f}")
        else:
            print("🎭 **Predicted Mood Tags: None**")
        print(f"💖 **Valence:** {valence_5s:.2f} (Scale: 1-9)")
        print(f"⚡ **Arousal:** {arousal_5s:.2f} (Scale: 1-9)")

        print("\n**30-Second Segment Analysis:**")
        print("------------------------------")
        if predicted_moods_30s:
            print("🎭 **Predicted Mood Tags (with Probabilities):**")
            for mood_data in predicted_moods_30s:
                print(f"   - {mood_data['mood']}: {mood_data['score']:.4f}")
        else:
            print("🎭 **Predicted Mood Tags: None**")
        print(f"💖 **Valence:** {valence_30s:.2f} (Scale: 1-9)")
        print(f"⚡ **Arousal:** {arousal_30s:.2f} (Scale: 1-9)")

        print("-" * 50)
        
    except GPUNotAvailableError as e:
        print(str(e))
        return 1
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return 1
    return 0

print("Analyzing full audio...")
output_dic_full = music2emo.predict(input_audio)
