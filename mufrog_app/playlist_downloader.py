"""
YouTube playlist downloader module for MuFrog Gradio demo.
Handles downloading songs from YouTube playlists with duplicate detection and queuing.
"""
import yt_dlp
import json
import os
import uuid
import re
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import time


@dataclass
class DownloadProgress:
    """Track download progress for a playlist"""
    playlist_url: str
    playlist_title: str
    total_songs: int
    completed_songs: int
    current_song: str
    status: str  # 'queued', 'downloading', 'completed', 'error'
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class PlaylistDownloader:
    """Handles YouTube playlist downloading with queue management"""
    
    def __init__(self, songs_folder: str = "downloaded_songs"):
        self.songs_folder = Path(songs_folder)
        self.songs_folder.mkdir(exist_ok=True)
        
        # Track downloaded songs to avoid duplicates
        self.downloaded_songs: Set[str] = set()
        self.load_existing_songs()
        
        # Queue management
        self.download_queue: List[str] = []
        self.current_downloads: Dict[str, DownloadProgress] = {}
        self.completed_downloads: List[DownloadProgress] = []
        
        # Thread management
        self.download_thread: Optional[threading.Thread] = None
        self.is_downloading = False
        self.stop_requested = False
        
        # Progress callback
        self.progress_callback: Optional[Callable] = None
    
    def load_existing_songs(self):
        """Load existing songs to avoid duplicates"""
        if self.songs_folder.exists():
            for file_path in self.songs_folder.glob("*.info.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        song_data = json.load(f)
                        # Use title + artist as duplicate key
                        song_key = self._create_song_key(
                            song_data.get('title', ''),
                            song_data.get('artist', '')
                        )
                        self.downloaded_songs.add(song_key)
                except Exception as e:
                    print(f"Error loading existing song {file_path}: {e}")
    
    def _create_song_key(self, title: str, artist: str) -> str:
        """Create a unique key for duplicate detection"""
        # Normalize and create hash for duplicate detection
        normalized = f"{title.lower().strip()}_{artist.lower().strip()}"
        # Remove special characters and extra spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', '_', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def add_playlist_to_queue(self, playlist_url: str) -> bool:
        """Add a playlist URL to the download queue"""
        if playlist_url not in self.download_queue:
            self.download_queue.append(playlist_url)
            return True
        return False
    
    def get_queue_status(self) -> Dict:
        """Get current queue and download status"""
        return {
            'queue_length': len(self.download_queue),
            'is_downloading': self.is_downloading,
            'current_downloads': {url: asdict(progress) for url, progress in self.current_downloads.items()},
            'completed_count': len(self.completed_downloads),
            'total_downloaded_songs': len(self.downloaded_songs)
        }
    def set_progress_callback(self, callback: Callable):
        """Set callback function for progress updates"""
        self.progress_callback = callback
    
    def _notify_progress(self):
        """Notify progress callback if set"""        
        if self.progress_callback:
            try:
                self.progress_callback(self.get_queue_status())
            except Exception as e:
                print(f"Error in progress callback: {e}")

    async def _get_playlist_count(self, playlist_url: str) -> int:
        """Get the actual count of videos in a playlist"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'playliststart': 1,
                'playlistend': 0,  # Get all entries
                'force_ipv4': True,  # Fix 403 errors
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = await asyncio.to_thread(ydl.extract_info, playlist_url, download=False)
                
                if info_dict and 'entries' in info_dict:
                    # Count actual valid entries
                    valid_entries = [entry for entry in info_dict['entries'] if entry]
                    return len(valid_entries)
                else:
                    # Single video
                    return 1
        except Exception as e:
            print(f"Error getting playlist count: {e}")
            return 0

    async def _get_playlist_count_direct(self, playlist_url: str) -> int:
        """Get playlist count using yt-dlp's playlist_count output format"""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'playliststart': 1,
                'playlistend': 0,
                'outtmpl': '%(playlist_count)s',
                'skip_download': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info_dict = await asyncio.to_thread(ydl.extract_info, playlist_url, download=False)
                
                if info_dict:
                    # Try to get playlist_count from the info
                    playlist_count = info_dict.get('playlist_count')
                    if playlist_count:
                        return int(playlist_count)
                    
                    # Fallback to counting entries
                    if 'entries' in info_dict:
                        valid_entries = [entry for entry in info_dict['entries'] if entry]
                        return len(valid_entries)
                    else:
                        return 1
                        
                return 0
        except Exception as e:
            print(f"Error getting playlist count (direct): {e}")
            # Fallback to the original method
            return await self._get_playlist_count(playlist_url)

    async def _extract_playlist_info(self, playlist_url: str) -> Dict:
        """Extract playlist information without downloading"""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = await asyncio.to_thread(ydl.extract_info, playlist_url, download=False)
            return info_dict or {}
    
    async def _download_single_song(self, entry: Dict, progress: DownloadProgress) -> Optional[Dict]:
        """Download a single song and return metadata"""
        if not entry:
            progress.completed_songs += 1  # Count as processed even if entry is None
            self._notify_progress()
            return None
        
        video_id = entry.get('id')
        title = entry.get('title', 'Unknown')
        # Sanitize the title
        title = re.sub(r'[\\/*?:"<>|]', "", title)
        artist = entry.get('artist') or entry.get('uploader', 'Unknown')
        
        # Update progress to show current song
        progress.current_song = f"🎵 Processing ({progress.completed_songs + 1}/{progress.total_songs}): {title} by {artist}"
        self._notify_progress()
        
        # Check for duplicates
        song_key = self._create_song_key(title, artist)
        if song_key in self.downloaded_songs:
            print(f"Skipping duplicate: {title} by {artist}")
            progress.current_song = f"⏭️ Skipped duplicate ({progress.completed_songs + 1}/{progress.total_songs}): {title} by {artist}"
            progress.completed_songs += 1  # Count skipped songs as processed
            self._notify_progress()
            return None
            return None
        
        try:
            progress.current_song = f"⬇️ Downloading ({progress.completed_songs + 1}/{progress.total_songs}): {title} by {artist}"
            self._notify_progress()
            
            ydl_opts = {
                'format': 'bestaudio/mp3',
                'outtmpl': str(self.songs_folder / '%(title)s-%(id)s.%(ext)s'),
                'extractaudio': True,
                'audioformat': 'mp3',
                'writeinfojson': False,  # We'll create our own
                'writethumbnail': False,
                'keepvideo': False,
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Download the audio
                await asyncio.to_thread(ydl.download, [entry.get('url') or f"https://youtube.com/watch?v={video_id}"])
            
            # Create metadata
            song_id = str(uuid.uuid4())
            song_data = {
                "id": song_id,
                "title": title,
                "artist": artist,
                "genre": entry.get('genre'),
                "view_count": entry.get('view_count'),
                "likes": entry.get('like_count'),
                "release_date": entry.get('upload_date'),
                "video_id": video_id,
                "download_date": datetime.now().isoformat(),
                "valence": None,
                "arousal": None,
                "predicted_moods": [],
            }
            
            # Save metadata
            json_filename = self.songs_folder / f"{title}-{video_id}.info.json"
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(song_data, f, ensure_ascii=False, indent=4)
            
            # Add to downloaded set
            self._update_downloaded_songs_set(song_key)
              # Update progress
            progress.current_song = f"Completed: {title} by {artist}"
            progress.completed_songs += 1
            self._notify_progress()
            
            return song_data
            
        except Exception as e:
            print(f"Error downloading {title}: {e}")
            progress.current_song = f"Failed: {title} by {artist} - {str(e)}"
            progress.completed_songs += 1  # Count failed downloads as processed
            self._notify_progress()
            return None
    
    async def _download_playlist(self, playlist_url: str) -> DownloadProgress:
        """Download a single playlist"""
        progress = DownloadProgress(
            playlist_url=playlist_url,
            playlist_title="Unknown Playlist",
            total_songs=0,
            completed_songs=0,
            current_song="🔍 Extracting playlist info...",
            status="downloading",
            start_time=datetime.now()
        )
        
        self.current_downloads[playlist_url] = progress
        self._notify_progress()
        try:
            # Get accurate playlist count first
            progress.current_song = "🔍 Counting playlist songs..."
            self._notify_progress()
            
            total_count = await self._get_playlist_count_direct(playlist_url)
            progress.total_songs = total_count
            
            # Extract playlist info
            progress.current_song = "🔍 Analyzing playlist contents..."
            self._notify_progress()
            
            info_dict = await self._extract_playlist_info(playlist_url)
            
            if 'entries' in info_dict:
                entries = [entry for entry in info_dict['entries'] if entry]
                progress.playlist_title = info_dict.get('title', 'Unknown Playlist')
                progress.current_song = f"📋 Found {len(entries)} songs in playlist: {progress.playlist_title}"
            else:
                # Single video
                entries = [info_dict]
                progress.playlist_title = info_dict.get('title', 'Single Video')
                progress.current_song = f"📋 Single video: {progress.playlist_title}"
            
            self._notify_progress()
            
            # Small delay to let user see the info
            await asyncio.sleep(1)
            
            # Download songs with concurrency limit
            semaphore = asyncio.Semaphore(3)  # Limit concurrent downloads
            
            async def download_with_semaphore(entry):
                async with semaphore:
                    if self.stop_requested:
                        return None
                    return await self._download_single_song(entry, progress)
            
            # Download all songs
            downloaded_songs = await asyncio.gather(*[
                download_with_semaphore(entry) for entry in entries
            ], return_exceptions=True)
            
            # Count successful downloads
            successful_downloads = [song for song in downloaded_songs if song and not isinstance(song, Exception)]
            skipped_count = progress.completed_songs - len(successful_downloads)
            
            progress.status = "completed"
            progress.end_time = datetime.now()
            progress.current_song = f"✅ Complete! Downloaded {len(successful_downloads)} new songs, skipped {skipped_count} duplicates"
            
        except Exception as e:
            progress.status = "error"
            progress.error_message = str(e)
            progress.end_time = datetime.now()
            progress.current_song = f"❌ Error: {str(e)}"
            print(f"Error downloading playlist {playlist_url}: {e}")
        
        finally:
            self._notify_progress()
        
        return progress
    
    async def _process_queue(self):
        """Process the download queue"""
        while self.download_queue and not self.stop_requested:
            playlist_url = self.download_queue.pop(0)
            
            try:
                progress = await self._download_playlist(playlist_url)
                self.completed_downloads.append(progress)
                
                # Remove from current downloads
                if playlist_url in self.current_downloads:
                    del self.current_downloads[playlist_url]
                
            except Exception as e:
                print(f"Error processing playlist {playlist_url}: {e}")
            
            self._notify_progress()
    
    def start_downloads(self):
        """Start processing the download queue in a separate thread"""
        if self.is_downloading or not self.download_queue:
            return False
        
        self.is_downloading = True
        self.stop_requested = False
        
        def run_async_downloads():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._process_queue())
            finally:
                self.is_downloading = False
                loop.close()
                self._notify_progress()
        
        self.download_thread = threading.Thread(target=run_async_downloads, daemon=True)
        self.download_thread.start()
        return True
    
    def stop_downloads(self):
        """Stop current downloads"""
        self.stop_requested = True
        if self.download_thread and self.download_thread.is_alive():
            self.download_thread.join(timeout=5)
    
    def clear_queue(self):
        """Clear the download queue"""
        if not self.is_downloading:
            self.download_queue.clear()
            return True
        return False
    
    def get_downloaded_songs_metadata(self) -> List[Dict]:
        """Get metadata of all downloaded songs"""
        songs = []
        for file_path in self.songs_folder.glob("*.info.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    song_data = json.load(f)
                    songs.append(song_data)
            except Exception as e:
                print(f"Error loading song metadata {file_path}: {e}")
        
        return sorted(songs, key=lambda x: x.get('download_date', ''), reverse=True)
    
    def _update_downloaded_songs_set(self, song_key: str):
        """Update the downloaded songs set and notify UI if needed"""
        self.downloaded_songs.add(song_key)
        # Force a small delay to ensure file is written
        time.sleep(0.1)
        self._notify_progress()


# Global downloader instance
_downloader_instance = None

def get_downloader() -> PlaylistDownloader:
    """Get the global downloader instance"""
    global _downloader_instance
    if _downloader_instance is None:
        _downloader_instance = PlaylistDownloader()
    return _downloader_instance
