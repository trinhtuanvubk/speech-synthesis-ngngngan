## Part 1: prepare data for RVC

### 0. preview data

get list of audios and info (speakers count):
```
pip install yt-dlp
python scripts/00-get-info.py
```
small test with 2 audio files: `yt-dlp "KgxWziSHQP8" "01WRW7IV1uQ" -x --audio-format "wav" -o "%(id)s.%(ext)s" -P "data/raw"`

### 1. download audio

if good to go then download all youtube audios: `python scripts/01-download-audio.py`<br />
for simplicity, **download only audio with 1 speaker** so after remove silence can go directly to train RVC

audios are saved as `.wav` files in folder `data/01-raw`

### 2. remove silence and non-speech

using SileroVAD: `python scripts/02-remove-silence.py`

audios are saved as `.wav` files in folder `data/02-vad`

### 3. remove music 

using uvr

```bash
cd src/uvr_remove_music
bash ./download.sh
python3 separate.py --model_path "uvr5_weights/2_HP-UVR.pth" --audio_path 9OvlclzngLY.wav --only_save_vocal
```

