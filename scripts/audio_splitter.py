import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import glob


# using librosa, write a script that iterates thorugh all the waves in the /data directory and split them into chunks if there is a gap of silence longer than 1 second

def split_audio(audio_path, output_path, min_silence_len=1500, silence_thresh=-60, keep_silence=250):
    audio = AudioSegment.from_wav(audio_path)
    chunks = split_on_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=keep_silence)
    for i, chunk in enumerate(chunks):
        out_file = output_path + "_chunk{0}.wav".format(i)
        print("exporting", out_file)
        chunk.export(out_file, format="wav")

# for each wav in ./data, split it into chunks and save them to ./wavs
# i.e. './data/juice1.wav' gets split into './wavs/juice1_chunk{0}.wav'
def split_all_audios():
    for wav in glob.glob('./data/*.wav'):
        # get the filename without the extension
        filename = os.path.splitext(os.path.basename(wav))[0]
        # create the output path
        output_path = './wavs/' + filename
        # split the audio
        split_audio(wav, output_path)

# iterate through files in wavs, for each file get the length. if it's more than 12 seconds, split it into 12 second chunks at the first silence over 200ms
def split_long_audio(input_file, output_root, split_threshold: 15*60, accept_threshold: 20*60):
    # if wavs_split_temp and wavs_split_final don't exist, create them
   
    os.makedirs(output_root, exist_ok=True)
    
    # get the filename without the extension
    filename = os.path.splitext(os.path.basename(input_file))[0]
    # get the length of the audio
    audio = AudioSegment.from_wav(input_file)
    length = round(audio.duration_seconds, 2)
    # if the length is more than 12 seconds, split it into 12 second chunks
    if length > accept_threshold:
        print(f"Long audio: {input_file}")

        chunks = split_on_silence(audio, min_silence_len=300, silence_thresh=-60, keep_silence=300)
        current_length = 0
        current_split = 0
        # out_data is an empty AudioSegment
        out_data = AudioSegment.empty()
        for i, chunk in enumerate(chunks):
            # accumulate chunks until we have 12 seconds, then write
            # also write if we're at the end of the file
            current_length += round(chunk.duration_seconds, 2)
            print(filename)
            if current_length > split_threshold or (i == len(chunks) - 1 and len(chunks) > 1):
                # export the chunk
                out_file = os.path.join(output_root, f"{filename}_split_{current_split}.wav")
                print("exporting", out_file)
                out_data.export(out_file, format="wav")
                # reset the current length and split
                current_length = 0
                current_split += 1
                # reset the out_data
                out_data = AudioSegment.empty()
            else:
                if out_data:
                    out_data += chunk
                else:
                    out_data = chunk
            
    else:
        print(f"NOT long audio: {input_file}")
        # copy to wavs_split_final
        out_file = os.path.join(output_root, f"{filename}.wav")
        # export the chunk
        audio.export(out_file, format="wav")

def split_long_audio_dir(input_dir, output_root, split_threshold: 15*60, accept_threshold: 20*60):
    for file in os.listdir(input_dir):
        input_file = os.path.join(input_dir, file)
        split_long_audio(input_file, output_root, split_threshold=split_threshold, accept_threshold=accept_threshold)
        
  
# remove any audio files that are less than 500 MS long or are silent
def filter_short_audios():
    for wav in glob.glob('./wavs_split_temp/*.wav'):
        # get the filename without the extension
        filename = os.path.splitext(os.path.basename(wav))[0]
        # get the length of the audio
        audio = AudioSegment.from_wav(wav)
        length = round(audio.duration_seconds, 2)
        # if the length is more than 12 seconds, split it into 12 second chunks
        print("length is", length)
        if length > 1:
            # copy to wavs_split_final
            out_file = './wavs_split_final/' + filename + '.wav'
            # export the chunk
            audio.export(out_file, format="wav")
        else:
            print('omitting', wav)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type=str, default="None")
    parser.add_argument('--input_file', type=str, default="None")
    parser.add_argument('--output_dir', type=str, default="None")
    parser.add_argument('--split_threshold', type=int, default=15*60)
    parser.add_argument('--accept_threshold', type=int, default=20*60)
    
    args = parser.parse_args()
    
    split_long_audio_dir(args.input_dir, args.output_dir, args.split_threshold, args.accept_threshold)
    