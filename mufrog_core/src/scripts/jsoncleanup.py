import os
import json
import uuid
import re

def clean_json_metadata(songs_folder):
    """
    Cleans the JSON metadata files in the specified folder,
    extracting only the required fields and saving the cleaned data.
    """

    cleaned_data_list = []

    for filename in os.listdir(songs_folder):
        if filename.endswith(".info.json"):
            filepath = os.path.join(songs_folder, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Extract the required fields, handling potential missing keys
                video_id = data.get('id', '')
                title = data.get('title', '')
                title = re.sub(r'[\\/*?:"<>|]', "", title)  # Sanitize title
                artist = data.get('artist', data.get('uploader', ''))  # Artist or uploader
                genre = data.get('genre', '')
                view_count = data.get('view_count', 0)
                like_count = data.get('like_count', 0)
                release_date = data.get('upload_date', '')

                song_id = str(uuid.uuid4())  # Generate a new song ID

                # Create the cleaned song data dictionary
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

                cleaned_data_list.append(song_data)

                # Save the cleaned metadata to a new JSON file (optional)
                cleaned_filename = os.path.join(songs_folder, f"cleaned_{title}-{video_id}.info.json")
                with open(cleaned_filename, 'w', encoding='utf-8') as f:
                    json.dump(song_data, f, ensure_ascii=False, indent=4)
                print(f"Cleaned metadata saved to {cleaned_filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    return cleaned_data_list


if __name__ == "__main__":
    songs_folder = 'songs'  # Replace with your songs folder
    if not os.path.exists(songs_folder):
        print(f"Error: The folder '{songs_folder}' does not exist.")
    else:
        cleaned_metadata = clean_json_metadata(songs_folder)

        if cleaned_metadata:
            # Optionally save all cleaned metadata to a single JSON file
            all_cleaned_filename = os.path.join(songs_folder, 'all_cleaned_metadata.json')
            with open(all_cleaned_filename, 'w', encoding='utf-8') as f:
                json.dump(cleaned_metadata, f, ensure_ascii=False, indent=4)
            print(f"\nAll cleaned metadata saved to: {all_cleaned_filename}")
        else:
            print("No metadata files were cleaned.")