import scipy.io.wavfile as wave
from matplotlib import pyplot as plt
import numpy as np


def scaling_down(x):
    x_scaled = x/max(abs(x))
    return x_scaled

def cut_audio_clip(input_path, start, end, output_path):
    audio = wave.read(input_path)
    samples = np.array(audio[1], dtype=float)
    Fs = audio[0]
    clip_audio = scaling_down(samples[start:start + end*Fs][:,0])
    wave.write(output_path, Fs, clip_audio)


if __name__ == "__main__":
    input_path = "audio1_orig.wav"
    output_path = "clip1.wav"
    start = 0
    end = 60
    cut_audio_clip(input_path, start, end, output_path)