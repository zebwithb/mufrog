import yt_dlp
import json
import os
import uuid
import re  # Import the regular expression module

def download_playlist_and_extract_metadata(playlist_url):
    """
    Downloads a YouTube playlist as MP3, extracts metadata, and saves it as JSON.

    Args:
        playlist_url (str): The URL of the YouTube playlist.
    """

    songs_folder = 'songs'
    if not os.path.exists(songs_folder):
        os.makedirs(songs_folder)

    ydl_opts = {
        'format': 'bestaudio/mp3',
        'outtmpl': os.path.join(songs_folder, '%(title)s-%(id)s.%(ext)s'), # Output path and filename template
        'extractaudio': True,
        'audioformat': 'mp3',
        'noplaylist': False, # Ensure playlist download
        'writeinfojson': True, # Write video metadata to a .info.json file
        'writethumbnail': False, # Disable thumbnail download, optional
        'keepvideo': False,     # Delete video file after audio extraction
    }

    song_data_list = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(playlist_url, download=True)

        if 'entries' in info_dict: # Playlist case
            for entry in info_dict['entries']:
                if entry: # Check if entry is not None (some entries might be missing/deleted videos)
                    video_id = entry.get('id')
                    title = entry.get('title')
                    # Sanitize the title to remove invalid characters for filenames
                    title = re.sub(r'[\\/*?:"<>|]', "", title)
                    artist = entry.get('artist') or entry.get('uploader') # Artist might be in 'artist' or 'uploader'
                    genre = entry.get('genre')
                    view_count = entry.get('view_count')
                    like_count = entry.get('like_count')
                    # dislike_count is often not available anymore from YouTube API
                    release_date = entry.get('upload_date') # Format YYYYMMDD

                    song_id = str(uuid.uuid4()) # Generate unique ID for each song

                    song_data = {
                        "id": song_id,
                        "title": title,
                        "artist": artist,
                        "genre": genre,
                        "view_count": view_count,
                        "likes": like_count,
                        "release_date": release_date,
                        "valence": None, # Placeholder for mood analysis
                        "arousal": None, # Placeholder for mood analysis
                        "predicted_moods": [], # Placeholder for mood analysis
                    }

                    song_data_list.append(song_data)

                    # Save metadata to JSON file (optional, as we are also returning the list)
                    json_filename = os.path.join(songs_folder, f"{title}-{video_id}.info.json")
                    with open(json_filename, 'w', encoding='utf-8') as f:
                        json.dump(song_data, f, ensure_ascii=False, indent=4)
                    print(f"Metadata saved to {json_filename}")


        else: # Single video case (if you accidentally provide a video URL instead of playlist)
            video_id = info_dict.get('id')
            title = info_dict.get('title')
            # Sanitize the title to remove invalid characters for filenames
            title = re.sub(r'[\\/*?:"<>|]', "", title)
            artist = info_dict.get('artist') or info_dict.get('uploader')
            genre = info_dict.get('genre')
            view_count = info_dict.get('view_count')
            like_count = info_dict.get('like_count')
            release_date = info_dict.get('upload_date')

            song_id = str(uuid.uuid4())

            song_data = {
                        "id": song_id,
                        "title": title,
                        "artist": artist,
                        "genre": genre,
                        "view_count": view_count,
                        "likes": like_count,
                        "release_date": release_date,
                        "valence": None, # Placeholder for mood analysis
                        "arousal": None, # Placeholder for mood analysis
                        "predicted_moods": [], # Placeholder for mood analysis
                    }
            song_data_list.append(song_data)
             # Save metadata to JSON file (optional, as we are also returning the list)
            json_filename = os.path.join(songs_folder, f"{title}-{video_id}.info.json")
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(song_data, f, ensure_ascii=False, indent=4)
            print(f"Metadata saved to {json_filename}")


    return song_data_list


if __name__ == '__main__':
    playlist_url = input("Please enter the YouTube playlist URL: ")
    playlist_song_data = download_playlist_and_extract_metadata(playlist_url)

    if playlist_song_data:
        print("\nDownload and Metadata Extraction Completed!")
        print("\nExample Song Data (first song in playlist):")
        print(json.dumps(playlist_song_data[0], indent=4)) # Print example of the extracted data

        # You can further process playlist_song_data here, e.g., save all metadata to a single JSON file
        all_metadata_filename = os.path.join('songs', 'playlist_metadata.json')
        with open(all_metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(playlist_song_data, f, ensure_ascii=False, indent=4)
        print(f"\nAll songs metadata saved to: {all_metadata_filename}")

    else:
        print("No song data was extracted. Please check the playlist URL and try again.")