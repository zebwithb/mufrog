"""
Batch Music Analyzer

This module processes a collection of MP3 files in parallel, extracting emotion features
and audio characteristics using the Music2Emotion model.

Usage:
    python batch_analyzer.py [--workers N] [--output-dir PATH]
"""

import os
import json
import time
import threading
import argparse

from pathlib import Path
from datetime import datetime
import concurrent.futures
import torch
from tqdm import tqdm

# Add parent directory to path to import Music2Emotion
import sys
sys.path.append("..")
from Music2Emotion.music2emo import Music2emo


class BatchAnalyzer:
    """Analyzes multiple MP3 files in parallel using Music2Emotion model."""
    
    def __init__(self, num_workers=None, output_dir=None, input_metadata=None):
        """
        Initialize the batch analyzer.
        
        Args:
            num_workers: Number of parallel workers (default: auto-determined based on VRAM)
            output_dir: Directory to save output (default: src/analyzed_output)
            input_metadata: Path to input metadata file (default: src/scripts/songs/all_cleaned_metadata.json)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine optimal number of workers based on available VRAM
        if num_workers is None:
            self.num_workers = self._determine_optimal_workers()
        else:
            self.num_workers = num_workers
            
        # Set up directories
        self.base_dir = Path(__file__).parent
        self.mp3_dir = self.base_dir / "scripts" / "songs" / "mp3" 
        
        print(self.mp3_dir)
        
        # Set up input/output paths
        self.input_metadata = input_metadata or (self.base_dir / "scripts" / "songs" / "all_cleaned_metadata.json")
        self.output_dir = output_dir or (self.base_dir / "analyzed_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for output files
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"analyzed_metadata_{self.timestamp}.json"
        
        # Progress tracking
        self.total_songs = 0
        self.processed_songs = 0
        
        print(f"Batch Analyzer initialized with {self.num_workers} workers")
        print(f"Output will be saved to {self.output_file}")
    
    def _determine_optimal_workers(self):
        """Determine optimal number of workers based on available VRAM."""
        try:
            # Get total VRAM (in MB)
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            
            # Each instance uses ~400MB, leave 1GB for system
            available_vram = total_vram - 1024
            workers = max(1, int(available_vram / 400))
            
            # Cap at 20 workers for stability
            return min(20, workers)
        except Exception:
            # Fallback to 4 workers
            return 4
    
    def load_metadata(self):
        """Load metadata from JSON file with proper encoding handling."""
        # Try with different encodings
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                print(f"Trying to load metadata with {encoding} encoding...")
                with open(self.input_metadata, 'r', encoding=encoding) as f:
                    metadata = json.load(f)
                    print(f"Successfully loaded metadata with {encoding} encoding.")
                    return metadata
            except UnicodeDecodeError:
                print(f"Failed with {encoding} encoding, trying next...")
            except json.JSONDecodeError as e:
                print(f"JSON parsing error with {encoding}: {e}")
            except Exception as e:
                print(f"Error loading metadata file with {encoding}: {e}")
        
        # If all encodings fail, try a more brute force approach
        try:
            print("Trying binary read with error replacement...")
            with open(self.input_metadata, 'rb') as f:
                content = f.read()
                # Replace or ignore problematic bytes
                text = content.decode('utf-8', errors='replace')
                metadata = json.loads(text)
                print("Successfully loaded metadata with byte replacement.")
                return metadata
        except Exception as e:
            print(f"All encoding attempts failed: {e}")
            return []
    
    def save_metadata(self, metadata):
        """Save processed metadata to JSON file."""
        with open(self.output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nAnalysis complete! Results saved to {self.output_file}")
        
        # Also save a copy with the current time to ensure we don't lose data
        backup_file = self.output_dir / f"analyzed_metadata_latest.json"
        with open(backup_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _analyze_song(self, song_data, model):
        """
        Process a single song with the given model.
        
        Args:
            song_data: Dictionary containing song metadata
            model: Music2Emotion model instance
            
        Returns:
            Updated song data with emotion analysis
        """
        try:
            title = song_data.get("title", "")
            if not title:
                print(f"\nWarning: Song missing title in metadata")
                return song_data
            
            # Try to find a matching file with various patterns
            mp3_file = self._find_matching_mp3(title)
            if not mp3_file:
                print(f"\nWarning: MP3 file not found for '{title}'")
                return song_data
            
            print(f"\nProcessing: {mp3_file.name}")
            
            # Process with Music2Emotion
            result = model.predict(str(mp3_file))
            
            # Update song data with analysis results
            song_data["valence"] = float(result["valence"])
            song_data["arousal"] = float(result["arousal"])
            song_data["predicted_moods"] = result["predicted_moods"]
            
            return song_data
            
        except Exception as e:
            print(f"\nError processing {song_data.get('title', 'unknown')}: {str(e)}")
            return song_data
        
    def _find_matching_mp3(self, title):
        """
        Find matching MP3 file based on title, ignoring YouTube IDs.
        Returns Path object of matching file or None if not found.
        """
        if not title:
            return None
            
        # First, do some basic title cleaning
        clean_title = title.strip().lower()
        clean_title = clean_title.replace('"', '').replace('"', '').replace('"', '')
        
        # Get all MP3 files
        mp3_files = list(self.mp3_dir.glob("*.mp3"))
        
        # Try multiple matching strategies
        
        # Strategy 1: Direct filename match (unlikely but fastest)
        for mp3_file in mp3_files:
            if mp3_file.stem.lower() == clean_title:
                return mp3_file
                
        # Strategy 2: Title at start of filename (common pattern)
        for mp3_file in mp3_files:
            filename_lower = mp3_file.stem.lower()
            # Check if file starts with title (allow for space/dash after title)
            if filename_lower.startswith(clean_title + " ") or \
            filename_lower.startswith(clean_title + "-"):
                return mp3_file
        
        # Strategy 3: More flexible matching - title as a significant part
        # Remove special characters for fuzzy matching
        alphanum_title = ''.join(c.lower() for c in title if c.isalnum() or c.isspace())
        alphanum_title = ' '.join(alphanum_title.split())  # normalize spaces
        
        if len(alphanum_title) < 3:  # Title too short for reliable matching
            return None
        
        best_match = None
        best_score = 0
        
        for mp3_file in mp3_files:
            # Clean up filename too
            filename = mp3_file.stem
            clean_filename = ''.join(c.lower() for c in filename if c.isalnum() or c.isspace())
            clean_filename = ' '.join(clean_filename.split())
            
            # If filename contains the full title
            if alphanum_title in clean_filename:
                # Calculate a match score based on length similarity
                score = len(alphanum_title) / len(clean_filename)
                if score > best_score:
                    best_score = score
                    best_match = mp3_file
        
        # Return the best match if it's reasonably good
        if best_score > 0.3:  # Adjust threshold as needed
            return best_match
        
        # No good match found
        return None
    
    def process_batch(self, process_id, song_batch):
        """
        Process a batch of songs in a single worker.
        Each worker has its own model instance.
        
        Args:
            process_id: Worker ID for logging
            song_batch: List of songs to process
            
        Returns:
            List of processed songs
        """
        # Initialize model inside worker process
        model = Music2emo()
        
        results = []
        for song in song_batch:
            try:
                # Skip already processed songs
                if song.get("valence") is not None:
                    results.append(song)
                    continue
                    
                processed_song = self._analyze_song(song, model)
                results.append(processed_song)
                
                # Update global progress counter
                with self.progress_lock:
                    self.processed_songs += 1
                    self.progress_bar.update(1)
                    
            except Exception as e:
                print(f"\nError in worker {process_id}: {e}")
                results.append(song)  # Keep original data
                
        return results
    
    def run(self):
        """Run the batch analysis process."""
        # Load metadata
        metadata = self.load_metadata()
        self.total_songs = len(metadata)
        
        if not metadata:
            print("No songs found in metadata. Exiting.")
            return
            
        # Filter unprocessed songs
        songs_to_process = [song for song in metadata if song.get("valence") is None]
        already_processed = self.total_songs - len(songs_to_process)
        
        if already_processed > 0:
            print(f"Found {already_processed} already processed songs. Skipping these.")
            
        if not songs_to_process:
            print("All songs already processed. Saving metadata...")
            self.save_metadata(metadata)
            return

        # Prepare for progress tracking
        self.processed_songs = already_processed
        self.progress_lock = threading.Lock()
        self.progress_bar = tqdm(total=self.total_songs, initial=already_processed,
                               desc="Processing songs", unit="song")
        
        # Divide songs into batches for workers
        batch_size = max(1, len(songs_to_process) // self.num_workers)
        batches = [songs_to_process[i:i+batch_size] for i in range(0, len(songs_to_process), batch_size)]
        
        print(f"Processing {len(songs_to_process)} songs in {len(batches)} batches")
        
        # Run parallel processing with thread pool
        processed_batches = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Start workers with batches
            futures = {executor.submit(self.process_batch, i, batch): i 
                      for i, batch in enumerate(batches)}
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                batch_id = futures[future]
                try:
                    processed_batch = future.result()
                    processed_batches.append(processed_batch)
                    print(f"\nBatch {batch_id+1}/{len(batches)} completed")
                except Exception as e:
                    print(f"\nBatch {batch_id+1} failed: {e}")
        
        # Close progress bar
        self.progress_bar.close()
        
        # Merge results back into metadata
        processed_lookup = {}
        for batch in processed_batches:
            for song in batch:
                if "title" in song:  # Using title instead of ID
                    processed_lookup[song["title"]] = song

        # Update the original metadata
        for i, song in enumerate(metadata):
            if song.get("title") in processed_lookup:
                metadata[i] = processed_lookup[song["title"]]
                
        # Save results
        self.save_metadata(metadata)
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"Processed {self.processed_songs - already_processed} songs in {elapsed_time:.2f} seconds")
        print(f"Average time per song: {elapsed_time / (self.processed_songs - already_processed):.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch analyze MP3 files using Music2Emotion")
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    args = parser.parse_args()
    
    analyzer = BatchAnalyzer(num_workers=args.workers, output_dir=args.output_dir)
    analyzer.run()