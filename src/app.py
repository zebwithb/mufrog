from Music2Emotion.music2emo import Music2emo

input_audio = "inference/input/test.mp3"

music2emo = Music2emo()
output_dic = music2emo.predict(input_audio)

valence = output_dic["valence"]
arousal = output_dic["arousal"]
predicted_moods =output_dic["predicted_moods"]

print("\n🎵 **Music Emotion Recognition Results** 🎵")
print("-" * 50)
print(f"🎭 **Predicted Mood Tags:** {', '.join(predicted_moods) if predicted_moods else 'None'}")
print(f"💖 **Valence:** {valence:.2f} (Scale: 1-9)")
print(f"⚡ **Arousal:** {arousal:.2f} (Scale: 1-9)")
print("-" * 50)
