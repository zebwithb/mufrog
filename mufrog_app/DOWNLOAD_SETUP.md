# 🎵 MuFrog Download Setup Guide

## Quick Setup

To enable the YouTube playlist download functionality, you need to install the additional dependency:

```bash
pip install yt-dlp
```

Or install all dependencies at once:

```bash
pip install -r requirements.txt
```

## Usage

1. **Start the app**:
   ```bash
   python app.py
   ```

2. **Navigate to the "⬇️ Download Songs" tab**

3. **Add YouTube URLs**:
   - Paste YouTube playlist URLs
   - Individual video URLs also work
   - Click "➕ Add to Queue"

4. **Start downloading**:
   - Click "🚀 Start Downloads"
   - Monitor progress in real-time
   - Multiple playlists will be processed in order

5. **Manage downloads**:
   - Use "⏹️ Stop" to halt current downloads
   - Use "🗑️ Clear Queue" to remove pending downloads
   - Use "🔄 Refresh Status" to update progress

## Features

### ✅ **Duplicate Detection**
- Automatically skips songs you've already downloaded
- Uses title + artist matching for smart duplicate detection
- Saves time and storage space

### 🔄 **Queue Management**
- Add multiple playlists to download queue
- Process them automatically in order
- Real-time status updates

### 📊 **Progress Tracking**
- See current song being downloaded
- Track completion percentage per playlist
- View total downloaded songs count

### 💾 **Organized Storage**
- Downloads saved in `downloaded_songs` folder
- Metadata automatically extracted and saved
- Ready for Music2Emotion analysis

## Supported URLs

- `https://www.youtube.com/playlist?list=...` (Public playlists)
- `https://www.youtube.com/watch?v=...` (Individual videos)
- `https://music.youtube.com/playlist?list=...` (YouTube Music playlists)
- Mix playlists and auto-generated playlists

## Troubleshooting

### ❌ "Downloader not available"
**Solution**: Install yt-dlp with `pip install yt-dlp`

### ⚠️ "URL already in queue"
**Solution**: The URL is already queued for download

### 🔴 Download errors
**Possible causes**:
- Video is private or unavailable
- Network connection issues
- Age-restricted content
- Geographic restrictions

**Solution**: Check the video accessibility and try again

### 📁 Files not appearing
**Check**:
- Look in the `downloaded_songs` folder
- Refresh the songs list with the "🔄 Refresh Songs List" button
- Make sure downloads completed successfully

## File Structure

```
downloaded_songs/
├── Song Title-VideoID.mp3          # Audio file
├── Song Title-VideoID.info.json    # Metadata file
└── ...
```

## Next Steps

After downloading songs:
1. Use the Music2Emotion analysis system (coming next)
2. Analyze emotional content of your music library
3. Get mood-based recommendations from your own collection

---

**Note**: Please respect YouTube's Terms of Service and only download content you have permission to use.
