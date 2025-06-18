#!/usr/bin/env python3
"""
Run script for MuFrog Gradio Demo
Simple launcher script for the application
"""

import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import and run the main application
from app import main

if __name__ == "__main__":
    print("🐸 Starting MuFrog Music Emotion Analysis Demo...")
    print("Loading music data and initializing interface...")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Thanks for using MuFrog!")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        print("Please check that all dependencies are installed and data files are available.")
        sys.exit(1)
