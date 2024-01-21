import os

import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment

from OpenVoice import se_extractor
from OpenVoice.api import BaseSpeakerTTS, ToneColorConverter

import os
import subprocess

def convert_webm_to_mp3(input_path, output_path):
    # Get a list of all WebM files in the input directory
    webm_files = [f for f in os.listdir(input_path) if f.endswith('.webm')]

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    for webm_file in webm_files:
        # Input file path
        input_file = os.path.join(input_path, webm_file)

        # Output file path (MP3)
        output_file = os.path.join(output_path, os.path.splitext(webm_file)[0] + '.mp3')

        # Run ffmpeg command to convert WebM to MP3
        cmd = ['ffmpeg', '-i', input_file, '-codec:a', 'libmp3lame', output_file]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(f"Conversion of {webm_file} complete.")


def merge_mp3_files(input_path, output_file):
    # Get a list of all MP3 files in the input directory
    mp3_files = [f for f in os.listdir(input_path) if f.endswith('.mp3')]

    # Create an empty AudioSegment to hold the combined audio
    combined_audio = AudioSegment.silent(duration=0)

    for mp3_file in mp3_files:
        # Load each MP3 file
        audio_segment = AudioSegment.from_mp3(os.path.join(input_path, mp3_file))

        # Append the current audio segment to the combined audio
        combined_audio += audio_segment

    # Export the combined audio to a new MP3 file
    combined_audio.export(output_file, format="mp3")


def main():
    ckpt_base = 'checkpoints/base_speakers/EN'
    ckpt_converter = 'checkpoints/converter'
    device = "cpu"
    output_dir = 'outputs'

    base_speaker_tts = BaseSpeakerTTS(f'{ckpt_base}/config.json', device=device)
    base_speaker_tts.load_ckpt(f'{ckpt_base}/checkpoint.pth')

    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    os.makedirs(output_dir, exist_ok=True)

    source_se = torch.load(f'{ckpt_base}/en_default_se.pth').to(device)

    reference_speaker = 'Monologue.wav'
    reference_speaker = "OpenVoice/resources/example_reference.mp3"
    reference_speaker = "Mariusz/merged.mp3"
    target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, target_dir='processed',
                                                vad=True)

    save_path = f'{output_dir}/output_en_default.wav'

    # Run the base speaker tts
    text = "Glad to see things are going well."
    src_path = f'{output_dir}/tmp.wav'

    # delete old results
    if os.path.exists(save_path):
        os.remove(save_path)
    if os.path.exists(src_path):
        os.remove(src_path)

    base_speaker_tts.tts(text, src_path, speaker='default', language='English', speed=1.0)
    # Run the tone color converter
    tone_color_converter.convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path)

if __name__ == '__main__':
    # dir_path = "Mariusz/webm"
    # mp3_dir_path = "Mariusz/mp3"
    # convert_webm_to_mp3(dir_path, mp3_dir_path)
    main()
    # merge_mp3_files(mp3_dir_path, "Mariusz/merged.mp3")
