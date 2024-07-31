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

### 1.5 Split very long audio due to lack of memory

- Using pydub to find silence
```bash
python3 audio_splitter.py --input_dir path/to/indir \
--output_dir path/to/outdir

```

### 2. remove silence and non-speech

- Using SileroVAD:
```bash
python scripts/02-remove-silence.py --input_dir path/to/indir \
--output_dir path/to/outdir
```

if not pass any anrguments, audios are saved as `.wav` files in folder `data/02-vad`


### 3. remove music 

using uvr

```bash
cd src/uvr_remove_music
bash ./download.sh
python3 separate.py --model_path "uvr5_weights/2_HP-UVR.pth" --audio_path 9OvlclzngLY.wav --only_save_vocal
```

