import json
from pydantic import BaseModel
from typing import List
import torch

from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM

class ModelConfig(BaseModel):
    model_name: str = "google/gemma-3-1b-it"

def load_model(config: ModelConfig):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = Gemma3ForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return tokenizer, model

config = ModelConfig()
tokenizer, model = load_model(config)

MOOD_KEYS = ['adventure', 'ballad', 'christmas', 'commercial', 'dark', 'deep', 'drama', 'dramatic', 'dream', 'emotional', 'energetic', 'fast', 'fun', 'funny', 'game', 'groovy', 'happy', 'holiday', 'hopeful', 'love', 'meditative', 'melancholic', 'melodic', 'motivational', 'party', 'positive', 'powerful', 'retro', 'romantic', 'sad', 'sexy', 'slow', 'soft', 'soundscape', 'space', 'sport', 'summer', 'travel', 'upbeat', 'uplifting']

class PromptInput(BaseModel):
    prompt: str

class EmotionEntry(BaseModel):
    emotion: str
    songs: List[str]
    
class PromptEmotionPrediction(BaseModel):
    prompt: str
    emotions: None

def classify_emotions(prompt: str) -> dict:
    # Compose the prompt as a single string (avoid multiline implicit concat)
    system_prompt = (
        "Given the following user prompt, generate a JSON dictionary with these moods as keys:\n"
        + str(MOOD_KEYS) + "\n"
        "Assign a float score between 0.0 and 1.0 to each mood, reflecting its relevance.\n"
        "Example output:\n"
        '{"happy": 0.8, "sad": 0.1, ...}\n\n'
        f"User prompt: {prompt}\n"
        "Output JSON:"
    )
    inputs = tokenizer(system_prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract JSON substring
    start = response.find("{")
    end = response.rfind("}") + 1
    json_str = response[start:end]
    try:
        mood_scores = json.loads(json_str)
    except Exception:
        mood_scores = {}
    return mood_scores

def load_emotions(json_path: str) -> List[EmotionEntry]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EmotionEntry(**item) for item in data]

def find_closest_emotions(classified: dict, emotion_db: List[EmotionEntry], top_k=5) -> List[EmotionEntry]:
    # Matching based on weighted mood score overlap
    matches = []
    for entry in emotion_db:
        score = 0.0
        for mood, mood_score in classified.items():
            if mood.lower() in entry.emotion.lower():
                score += mood_score
        if score > 0:
            matches.append((score, entry))
    matches.sort(reverse=True, key=lambda x: x[0])
    return [entry for _, entry in matches[:top_k]]

def suggest_song(emotion_entries: List[EmotionEntry]) -> str:
    # Pick one song from the top emotion entries
    for entry in emotion_entries:
        if entry.songs:
            return entry.songs[0]
    return "No suggestion found."

def main():
    json_path = "emotions.json"  # path to your JSON file
    emotion_db = load_emotions(json_path)
    prompt_text = input("Enter your prompt: ")
    prompt = PromptInput(prompt=prompt_text)
    classified = classify_emotions(prompt.prompt)
    closest = find_closest_emotions(classified, emotion_db)
    suggestion = suggest_song(closest)
    print(f"Suggested song: {suggestion}")

if __name__ == "__main__":
    main()