import yt_dlp
import json
import os
import uuid
import re
import asyncio  # Import asyncio

async def download_and_extract_metadata(entry, songs_folder, song_data_list):
    """
    Asynchronously downloads a single video and extracts metadata.
    """
    if entry:
        video_id = entry.get('id')
        title = entry.get('title')
        # Sanitize the title to remove invalid characters for filenames
        title = re.sub(r'[\\/*?:"<>|]', "", title)
        artist = entry.get('artist') or entry.get('uploader')
        genre = entry.get('genre')
        view_count = entry.get('view_count')
        like_count = entry.get('like_count')
        release_date = entry.get('upload_date')

        song_id = str(uuid.uuid4())

        song_data = {
            "id": song_id,
            "title": title,
            "artist": artist,
            "genre": genre,
            "view_count": view_count,
            "likes": like_count,
            "release_date": release_date,
            "valence": None,
            "arousal": None,
            "predicted_moods": [],
        }

        song_data_list.append(song_data)

        # Save metadata to JSON file
        json_filename = os.path.join(songs_folder, f"{title}-{video_id}.info.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(song_data, f, ensure_ascii=False, indent=4)
        print(f"Metadata saved to {json_filename}")


async def download_playlist_and_extract_metadata(playlist_url):
    """
    Downloads a YouTube playlist as MP3, extracts metadata, and saves it as JSON.
    Uses asyncio for concurrent downloads.
    """

    songs_folder = 'songs'
    if not os.path.exists(songs_folder):
        os.makedirs(songs_folder)

    ydl_opts = {
        'format': 'bestaudio/mp3',
        'outtmpl': os.path.join(songs_folder, '%(title)s-%(id)s.%(ext)s'),
        'extractaudio': True,
        'audioformat': 'mp3',
        'noplaylist': False,
        'writeinfojson': True,
        'writethumbnail': False,
        'keepvideo': False,
    }

    song_data_list = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = await asyncio.to_thread(ydl.extract_info, playlist_url, download=True)  # Run blocking I/O in a thread

        if 'entries' in info_dict:
            tasks = [download_and_extract_metadata(entry, songs_folder, song_data_list) for entry in info_dict['entries'] if entry]
            await asyncio.gather(*tasks)  # Run tasks concurrently

        else:  # Single video case
            await download_and_extract_metadata(info_dict, songs_folder, song_data_list)

    return song_data_list


if __name__ == '__main__':
    playlist_url = input("Please enter the YouTube playlist URL: ")
    asyncio.run(download_playlist_and_extract_metadata(playlist_url))

    # The rest of your code to process the song_data_list (e.g., saving to a single JSON)
    # should be placed here or in a separate function called from here.
    # Example:
    # all_metadata_filename = os.path.join('songs', 'playlist_metadata.json')
    # with open(all_metadata_filename, 'w', encoding='utf-8') as f:
    #     json.dump(song_data_list, f, ensure_ascii=False, indent=4)
    # print(f"\nAll songs metadata saved to: {all_metadata_filename}")