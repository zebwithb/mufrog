import sys
import os
import torch
import numpy as np  # Import numpy

print("Current working directory: ", os.getcwd())
sys.path.append("..")
from Music2Emotion.music2emo import Music2emo

input_audio = "scripts/songs/mp3/a-ha - Take On Me (Official Video) [Remastered in 4K]-djV11Xbc914.mp3"

music2emo = Music2emo()
output_dic = music2emo.predict(input_audio)

valence = output_dic["valence"]
arousal = output_dic["arousal"]
predicted_moods = output_dic["predicted_moods"]
predicted_moods_all = output_dic["predicted_moods_all"]

# --- Extended Analysis Output ---
print("\n🎵 **Music Emotion Recognition Results** 🎵")
print("-" * 50)
if predicted_moods:
    print("🎭 **Predicted Mood Tags (with Probabilities):**")
    for mood_data in predicted_moods:
        print(f"   - {mood_data['mood']}: {mood_data['score']:.4f}")
else:
    print("🎭 **Predicted Mood Tags: None**")
print(f"💖 **Valence:** {valence:.2f} (Scale: 1-9)")
print(f"⚡ **Arousal:** {arousal:.2f} (Scale: 1-9)")
print("-" * 50)