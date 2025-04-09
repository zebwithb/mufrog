from typing import List
import json

def extract_unique_moods(metadata_path: str) -> List[str]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    moods = set()
    for song in data:
        for mood_entry in song.get("predicted_moods", []):
            mood = mood_entry.get("mood")
            if mood:
                moods.add(mood)
    return sorted(moods)

moods = extract_unique_moods("src/scripts/analyzed_metadata_latest.json")
print(moods)

emotions =[
        'adventure',
         'ballad',
         'christmas',
         'commercial',
         'dark',
         'deep',
         'drama',
         'dramatic',
         'dream',
         'emotional',
         'energetic',
         'fast',
         'fun',
         'funny',
         'game',
         'groovy',
         'happy',
         'holiday',
         'hopeful',
         'love',
         'meditative',
         'melancholic',
         'melodic',
         'motivational',
         'party',
         'positive',
         'powerful',
         'retro',
         'romantic',
         'sad',
         'sexy',
         'slow',
         'soft',
         'soundscape',
         'space',
         'sport',
         'summer',
         'travel',
         'upbeat',
         'uplifting']