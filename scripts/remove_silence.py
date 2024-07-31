#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""remove silence and non-speech using Silero VAD"""
# see https://github.com/snakers4/silero-vad/wiki/Examples-and-Dependencies
# also https://github.com/snakers4/silero-vad/blob/master/utils_vad.py

import os
from tqdm import tqdm
import torch
from torchaudio import save as _save_audio

from _constants import LIST_VID, RAW_DATA_PATH, VAD_DATA_PATH
from _utils import load_audio

TQDM_PBAR_FORM = "{percentage: 5.1f}% |{bar}| {n:.0f}/{total} [{elapsed}<{remaining}, {rate_fmt}]"
SAMPLING_RATE = 16000  # Silero VAD operating value
MODEL, (get_speech_timestamps, _, silero_read_audio, _, _) = torch.hub.load(
	repo_or_dir="snakers4/silero-vad", model="silero_vad", onnx=False
)
MODEL = MODEL.to("cuda")


def vad_filter(infile: str, outfile: str, split_threshold, max_threshold) -> None:
	wav = silero_read_audio(infile, sampling_rate=SAMPLING_RATE).to("cuda")  # SileroVAD operate on mono channel at 16 kHz
	with torch.inference_mode(), tqdm(total=wav.shape[0], bar_format=TQDM_PBAR_FORM) as pbar:
		speech_timestamps: list[dict[str, int]] = get_speech_timestamps(
			wav, MODEL, sampling_rate=SAMPLING_RATE,
			progress_tracking_callback=lambda val: pbar.update(val * 10)  # weird, TODO: raise issue in silero repo
		)
	torch.cuda.empty_cache()

	# convert timestamps to match original audio file (dual channels & higher bit rate)
	audio_file: dict[str, int | str | torch.Tensor] = load_audio(infile)
	ratio = audio_file["sample_rate"] / SAMPLING_RATE

	max_samples = int(split_threshold * audio_file["sample_rate"])
	max_threshold_samples = int(max_threshold * audio_file["sample_rate"])

	current_duration = 0
	current_waveform = []
	file_index = 0
    
	print(f"speech_timestamps length: {len(speech_timestamps)}")	
    
	for el in speech_timestamps:
        start_sample = int(el["start"] * ratio)
        end_sample = int(el["end"] * ratio)
        segment = audio_file["waveform"][:, start_sample:end_sample]
        segment_duration = segment.shape[1]
        
        while segment_duration > max_threshold_samples:
            # If the segment is too long, split it into chunks of max_threshold_samples
            cut_waveform = segment[:, :max_threshold_samples]
            outfile = os.path.join(outdir, f"{os.path.splitext(os.path.basename(infile))[0]}_part{file_index}.wav")
            _save_audio(outfile, cut_waveform, sample_rate=audio_file["sample_rate"], bits_per_sample=audio_file["bits_per_sample"], encoding=audio_file["encoding"])
            print(f"Saved {outfile}")
            
            file_index += 1
            segment = segment[:, max_threshold_samples:]
            segment_duration = segment.shape[1]

        if current_duration + segment_duration > max_samples:
            # Save the current waveform to a file
            cut_waveform = torch.cat(current_waveform, dim=1)
            outfile = os.path.join(outdir, f"{os.path.splitext(os.path.basename(infile))[0]}_part{file_index}.wav")
            _save_audio(outfile, cut_waveform, sample_rate=audio_file["sample_rate"], bits_per_sample=audio_file["bits_per_sample"], encoding=audio_file["encoding"])
            print(f"Saved {outfile}")

            # Reset for the next file
            file_index += 1
            current_waveform = [segment]
            current_duration = segment_duration
        else:
            current_waveform.append(segment)
            current_duration += segment_duration

    # Save any remaining waveform
    if current_waveform:
        cut_waveform = torch.cat(current_waveform, dim=1)
        outfile = os.path.join(outdir, f"{os.path.splitext(os.path.basename(infile))[0]}_part{file_index}.wav")
        _save_audio(outfile, cut_waveform, sample_rate=audio_file["sample_rate"], bits_per_sample=audio_file["bits_per_sample"], encoding=audio_file["encoding"])
        print(f"Saved {outfile}")
        

def run(input_dir, output_dir):
    if input_dir == None:
        input_dir = RAW_DATA_PATH
    if output_dir == None:
        output_dir = VAD_DATA_PATH
    
    for file in os.listdir(input_dir):
        # filename = file.rsplit(".",1)[0]
        input_file = os.path.join(input_dir, file)
        print(input_file)
        output_file = os.path.join(output_dir, file)
        
        vad_filter(input_file, output_file)
        

#################################### main #####################################

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    
    args = parser.parse_args()
    
    run(args.input_dir, args.output_dir)
    
	# for id in LIST_VID.keys():
	# 	infile = os.path.join(RAW_DATA_PATH, f"{id}.wav")
	# 	outfile = os.path.join(VAD_DATA_PATH, f"{id}.wav")
	# 	if not os.path.exists(infile):
	# 		print(f"{id} not found")
	# 	else:
	# 		print(f"{id} to be VAD filtered")
	# 		vad_filter(infile, outfile)
