import json
from pydantic import BaseModel
from typing import List
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")

class PromptInput(BaseModel):
    prompt: str

class EmotionEntry(BaseModel):
    emotion: str
    songs: List[str]

def classify_emotions(prompt: str) -> List[str]:
    # Use your LLM to classify emotions from prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Placeholder: parse response to extract emotions
    emotions = response.split(",")  # adjust parsing as needed
    return [e.strip() for e in emotions]

def load_emotions(json_path: str) -> List[EmotionEntry]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EmotionEntry(**item) for item in data]

def find_closest_emotions(classified: List[str], emotion_db: List[EmotionEntry], top_k=5) -> List[EmotionEntry]:
    # Simple matching based on string similarity
    matches = []
    for entry in emotion_db:
        score = sum(1 for e in classified if e.lower() in entry.emotion.lower())
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