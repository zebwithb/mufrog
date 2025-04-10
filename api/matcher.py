import json
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

class ModelConfig(BaseModel):
    model_name: str = "google/gemma-3-1b-it"

def load_model(config: ModelConfig):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = Gemma3ForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config,
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    return tokenizer, model

config = ModelConfig()
tokenizer, model = load_model(config)

MOOD_KEYS = ['adventure', 'ballad', 'christmas', 'commercial', 'dark', 'deep', 'drama', 'dramatic', 'dream', 'emotional', 'energetic', 'fast', 'fun', 'funny', 'game', 'groovy', 'happy', 'holiday', 'hopeful', 'love', 'meditative', 'melancholic', 'melodic', 'motivational', 'party', 'positive', 'powerful', 'retro', 'romantic', 'sad', 'sexy', 'slow', 'soft', 'soundscape', 'space', 'sport', 'summer', 'travel', 'upbeat', 'uplifting']

class EmotionEntry(BaseModel):
    emotion: str
    songs: List[str]
    predicted_moods: Dict[str, float]

class PromptEmotionPrediction(BaseModel):
    prompt: str = Field(description="prompt to analyse")
    predicted_moods: Dict[str, float] = Field(description=" mood scores to predict, values are float between 0.0 and 1.0")

def classify_emotions(query: str, moods=MOOD_KEYS) -> Dict[str, float]:
    parser = PydanticOutputParser(pydantic_object=PromptEmotionPrediction)
    format_instructions = parser.get_format_instructions()
    prompt_text = (
        "<start_of_turn>user\n"
        "You are an expert music mood classifier.\n"
        "Given the user prompt below, respond ONLY with a JSON object.\n"
        "The JSON must have two keys:\n"
        "  'prompt': the original user prompt as a string\n"
        "  'predicted_moods': a dictionary with these moods as keys:\n"
        f"{MOOD_KEYS}\n"
        "Each mood key MUST have a float value between 0.0 and 1.0.\n"
        "Do NOT include any explanation or schema.\n"
        "Example:\n"
        '{"prompt": "I want a happy song", "predicted_moods": {"happy": 0.9, "sad": 0.1, ...}}\n'
        "User prompt: {query}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    prompt  = PromptTemplate(
    template=prompt_text,
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)
    # Format the prompt with user query
    prompt_value = prompt.format_prompt(query=query)
    prompt_str = prompt_value.to_string()

    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    parsed = parser.parse(response)
    prediction = PromptEmotionPrediction(prompt=query, predicted_moods=parsed["predicted_moods"])
    
    return prediction.predicted_moods

def load_emotions(json_path: str) -> List[EmotionEntry]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = []
    for item in data:
        predicted = item.get("predicted_moods") or {}
        if not isinstance(predicted, dict):
            predicted = {}
        entries.append(
            EmotionEntry(
                emotion=item.get("emotion", ""),
                songs=item.get("songs", []),
                predicted_moods=predicted,
            )
        )
    return entries

def find_closest_emotions(classified: Dict[str, float], emotion_db: List[EmotionEntry], top_k=5) -> List[EmotionEntry]:
    matches = []
    for entry in emotion_db:
        if not entry.predicted_moods:
            continue
        score = 0.0
        for mood, mood_score in classified.items():
            if mood.lower() in entry.emotion.lower():
                score += mood_score
        if score > 0:
            matches.append((score, entry))
    matches.sort(reverse=True, key=lambda x: x[0])
    return [entry for _, entry in matches[:top_k]]

def suggest_song(emotion_entries: List[EmotionEntry]) -> str:
    for entry in emotion_entries:
        if entry.songs:
            return entry.songs[0]
    return "No suggestion found."

def main():
    json_path = "analyzed_metadata_latest.json"
    emotion_db = load_emotions(json_path)
    prompt_text = input("Enter your prompt: ")
    classified = classify_emotions(prompt_text)
    closest = find_closest_emotions(classified, emotion_db)
    suggestion = suggest_song(closest)
    print(f"Suggested song: {suggestion}")

if __name__ == "__main__":
    main()