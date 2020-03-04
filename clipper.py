import scipy.io.wavfile as wave
from matplotlib import pyplot as plt
import numpy as np




def cut_audio_clip(input_path, start, end, output_path, Fs):
    audio = wave.read(input_path)
    audio = np.array(audio[1], dtype=float)/(65536/2)
    clip_audio = audio[start:end][:,0]
    # n = []
    # print(clip_audio)
    # for i in range(len(audio)):
    #     n.append(i)
    # plt.plot(n, audio)
    # plt.show()
    wave.write(output_path, Fs, clip_audio)


if __name__ == "__main__":
    input_path = "./audio3_orig.wav"
    output_path = "audio3_clip.wav"
    Fs = 44100
    number_of_samples = 5000
    start = 0
    end = start + 60*Fs
    cut_audio_clip(input_path, start, end, output_path, Fs)