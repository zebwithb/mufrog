# 🐸 MuFrog Gradio Demo

A modular Gradio-based demo application for the MuFrog Music Emotion Analysis system. This app showcases Music2Emotion analysis by providing an interactive interface for browsing analyzed music data and getting mood-based song recommendations.

## �️ Architecture

The application is built with a modular architecture for maintainability and extensibility:

```
mufrog_app/
├── app.py              # Main Gradio interface
├── data_loader.py      # Data loading and processing
├── mood_classifier.py  # Mood classification logic
├── recommender.py      # Song recommendation engine
├── ui_components.py    # UI components and formatting
├── config.py          # Configuration settings
├── run.py             # Simple launcher script
├── __init__.py        # Package initialization
├── requirements.txt   # Dependencies
└── README.md          # This file
```

### 📦 Module Overview

- **`app.py`**: Main application entry point with Gradio interface setup
- **`data_loader.py`**: Handles loading and processing of analyzed music metadata
- **`mood_classifier.py`**: Classifies user prompts into Music2Emotion mood categories
- **`recommender.py`**: Core recommendation engine with similarity matching
- **`ui_components.py`**: UI formatting and component utilities
- **`config.py`**: Centralized configuration management
- **`run.py`**: Simple launcher for easy deployment

## ✨ Features

### 🎵 Browse Music Database
- Interactive table view of analyzed songs
- Displays mood predictions, valence, arousal, and metadata
- View count and engagement metrics

### 🎯 Smart Recommendations
- Natural language mood description input
- Keyword-based mood classification using Music2Emotion categories
- Similarity-based song matching
- Top-k recommendations with explanation

### ⬇️ Download Songs (NEW!)
- Download songs from YouTube playlists for analysis
- Smart duplicate detection to avoid re-downloading
- Queue multiple playlists for batch processing
- Real-time progress tracking and status updates
- Automatic metadata extraction

### 📊 Mood Analytics
- Dataset statistics and insights
- Mood distribution analysis
- Real-time analytics refresh

## 🚀 Quick Start

### Option 1: Using the launcher script
```bash
python run.py
```

### Option 2: Direct execution
```bash
python app.py
```

### Option 3: As a module
```python
from mufrog_app import main
main()
```

## 📋 Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

### Core Dependencies
- `gradio` - Web interface framework
- `pandas` - Data manipulation
- `numpy` - Numerical operations

## 🔧 Configuration

Customize the application by editing `config.py`:

```python
# Server settings
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860

# Recommendation settings
DEFAULT_TOP_K = 10

# UI settings
MAX_DISPLAY_SONGS = 1000
```

## 🎭 Mood Categories

The system uses **41 mood categories** from Music2Emotion analysis:

`adventure`, `ballad`, `christmas`, `commercial`, `dark`, `deep`, `drama`, `dramatic`, `dream`, `emotional`, `energetic`, `fast`, `fun`, `funny`, `game`, `groovy`, `happy`, `holiday`, `hopeful`, `love`, `meditative`, `melancholic`, `melodic`, `motivational`, `party`, `positive`, `powerful`, `retro`, `romantic`, `sad`, `sexy`, `slow`, `soft`, `soundscape`, `space`, `sport`, `summer`, `travel`, `upbeat`, `uplifting`

## 💡 Example Usage

### Mood-based Recommendations
```
Input: "I want energetic and upbeat music for my workout"
Output: Songs classified as 'energetic', 'upbeat', 'powerful'

Input: "Something romantic and soft for a date night"
Output: Songs classified as 'romantic', 'soft', 'love'
```

## 🔮 Future Enhancements

### Planned Features
- **LLM Integration**: Enhanced mood interpretation using language models
- **Advanced Filters**: Valence/arousal range filtering
- **User Profiles**: Personalized recommendation learning
- **Playlist Generation**: Create and export playlists
- **Audio Preview**: Direct song playback integration

### Extensibility
The modular architecture makes it easy to:
- Add new recommendation algorithms in `recommender.py`
- Enhance mood classification in `mood_classifier.py`
- Customize UI components in `ui_components.py`
- Integrate external APIs or databases

## 📊 Data Requirements

The app expects analyzed music metadata in JSON format with the following structure:
```json
[
  {
    "title": "Song Title",
    "artist": "Artist Name",
    "predicted_moods": [
      {"mood": "happy", "score": 0.85},
      {"mood": "energetic", "score": 0.72}
    ],
    "valence": 0.7,
    "arousal": 0.8,
    "view_count": 1000000,
    "likes": 50000
  }
]
```

## 🐛 Troubleshooting

### Common Issues

1. **Data file not found**
   - Ensure `analyzed_metadata_latest.json` exists in the expected path
   - Check `config.py` for correct data paths

2. **Import errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python path includes the mufrog_app directory

3. **Port already in use**
   - Modify `SERVER_PORT` in `config.py`
   - Or run with: `python -c "from app import main; main()"`

## 🤝 Contributing

The modular structure makes contributions easy:

1. Fork the repository
2. Create a feature branch
3. Add your changes to the appropriate module
4. Update tests and documentation
5. Submit a pull request

## 📄 License

This project is part of the MuFrog Music Analysis system.

---

**Built with ❤️ using Gradio and Music2Emotion analysis**
