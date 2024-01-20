import time
from faster_whisper import WhisperModel
from pathlib import Path

SRC_PATH = Path(__file__).parent

from pydub import AudioSegment

def speed_up_audio(input_filename, output_filename, speed_factor=1.5):
    # Load the audio file
    audio = AudioSegment.from_wav(input_filename)

    # Speed up the audio
    sped_up_audio = audio.speedup(playback_speed=speed_factor)

    # Export the sped-up audio to a new file
    sped_up_audio.export(output_filename, format="wav")

def transcribe(model, beam_size=5, vad_filter=True, min_silence_duration_ms=500):
    segments, info = model.transcribe((SRC_PATH / "Monologue.wav").as_posix(), beam_size=beam_size, vad_filter=vad_filter, vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms),)
    return list(segments)


def main():
    model_size = "base"
    cpu_threads = 4
    compute_type = "int8"
    beam_size = 5
    vad_filter = False
    min_silence_duration_ms = 500

    model = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads, download_root=(SRC_PATH / "models").as_posix())
    start = time.perf_counter()
    results = transcribe(model, beam_size=beam_size, vad_filter=vad_filter, min_silence_duration_ms=min_silence_duration_ms)
    end = time.perf_counter()

    print('Total time:', end - start)
    print(" ".join(result.text for result in results))

if __name__ == '__main__':
    main()
    # speed_up_audio("Monologue.wav", "Monologue_speed_up.wav", speed_factor=1.5)