# -*- coding: utf-8 -*-

import os.path
import json

RAW_DATA_PATH      = os.path.join("data", "01-raw")
RAW_DATA_PATH = "/content/drive/MyDrive/AIAgent/crawl_dataset/nguyenngocngan/01-raw"

VAD_DATA_PATH      = os.path.join("data", "02-vad")
VAD_DATA_PATH = "/content/drive/MyDrive/AIAgent/crawl_dataset/nguyenngocngan/02-vad"

DIARIZED_DATA_PATH = os.path.join("data", "03-diarized")
DIARIZED_DATA_PATH = "/content/drive/MyDrive/AIAgent/crawl_dataset/nguyenngocngan/03-diarized"

VOICE_DATA_PATH    = os.path.join("data", "04-voices")
VOICE_DATA_PATH    = "/content/drive/MyDrive/AIAgent/crawl_dataset/nguyenngocngan/04-voices"

MERGED_DATA_PATH   = os.path.join("data", "05-merged")
MERGED_DATA_PATH   = "/content/drive/MyDrive/AIAgent/crawl_dataset/nguyenngocngan/05-merged"

SUBS_DATA_PATH     = os.path.join("data", "06-subs")
SUBS_DATA_PATH     = "/content/drive/MyDrive/AIAgent/crawl_dataset/nguyenngocngan/06-subs"


AUDIO_TEXT_FILE_LIST_PATH = os.path.join("data", "99-audio-text-file-list")
AUDIO_TEXT_FILE_LIST_PATH = "/content/drive/MyDrive/AIAgent/crawl_dataset/nguyenngocngan/99-audio-text-file-list"


FIELD_SEP = "|"

DRAFT_FILE   = "/content/drive/MyDrive/AIAgent/crawl_dataset/nguyenngocngan/draft.json"
SUMMARY_FILE = "/content/drive/MyDrive/AIAgent/crawl_dataset/nguyenngocngan/data.json"
with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
	LIST_VID = json.load(f)