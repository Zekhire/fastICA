import scipy.io.wavfile as wave
from matplotlib import pyplot as plt
import numpy as np



# org = wave.read("org.wav")
# short = wave.read("short2.wav")
# long = wave.read("long2.wav")


# print(org)

# print(len(org))

# org = np.array(org[1], dtype=float)/(65536/2)
# short = np.array(short[1], dtype=float)/(65536/2)
# long = np.array(long[1], dtype=float)/(65536/2)

def load_samples(input_path):
    audio = wave.read(input_path)
    audio = np.array(audio[1], dtype=float)/(65536/2)
    return audio





def mixer(input_paths, output_path):
    audios_number = len(input_paths)
    A = np.matrix(np.random.randint(1, 10, (audios_number, audios_number)))
    print(A)
    s = []
    # for input_path in input_paths:
    #     audio = load_samples(input_path)
    #     s.append(audio)
    s = np.matrix(s)
    
    s = np.matrix([[2,2,2],[3,3,3],[4,4,4]])
    print(s)
    x = A*s
    print(x)




if __name__ == "__main__":
    input_paths = ["audio_clip1.wav", "audio_clip2.wav", "audio_clip3.wav"]
    output_path = ["mixed.wav"]
    mixer(input_paths, output_path)