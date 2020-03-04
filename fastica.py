import scipy.io.wavfile as wave
from matplotlib import pyplot as plt
import numpy as np
import scipy.optimize


# org = wave.read("org.wav")
# short = wave.read("short2.wav")
# long = wave.read("long2.wav")


# print(org)

# print(len(org))

# org = np.array(org[1], dtype=float)/(65536/2)
# short = np.array(short[1], dtype=float)/(65536/2)
# long = np.array(long[1], dtype=float)/(65536/2)

def draw_distribution(x, points=-1, plotname="test.jpg", printing=True, showing=True):
    fig = plt.figure()
    plt.plot([-100000, 100000],[0,0], "k")
    plt.plot([0,0],[-100000, 100000], "k")
    plt.plot(x[0][:points], x[1][:points], "bo")
    plt.plot(0,0, "ro")
    plt.grid()
    plt.xlim([min(x[0][:points])*1.2, max(x[0][:points])*1.2])
    plt.ylim([min(x[1][:points])*1.2, max(x[1][:points])*1.2])
    plt.xlabel("audio1")
    plt.ylabel("audio2")
    if showing:
        plt.show()
    fig.savefig(plotname, dpi=fig.dpi)
        




def load_samples(input_path):
    audio = wave.read(input_path)
    audio = np.array(audio[1], dtype=float)
    return audio



def load_audios(input_paths):
    s = []
    for input_path in input_paths:
        audio = load_samples(input_path)
        print("MAX:", max(audio))
        s.append(audio)
    s = np.array(s)
    return s

def save_audios(output_paths, z):
    for i in range(len(output_paths)):
        print("MAX:", max(z[i]))
        wave.write(output_paths[i], 44100, z[i])


def mixing(input_paths, output_paths, show=True):
    audios_number = len(input_paths)
    s = load_audios(input_paths)
    if show:
        draw_distribution(s)
    A = np.random.random_sample((audios_number, audios_number))
    y = np.dot(A, s)
    save_audios(output_paths, y)
    if show:
        draw_distribution(y)


def centering(y):
    new_y = y.copy()
    means = []
    for i in range(len(y)):
        mean = np.mean(y[i])
        means.append(mean)
        new_y[i] -= mean
    means = np.array(means)
    return new_y, means


def whitening(y):
    y_t = np.transpose(y)
    covariance=np.cov(y)
    D, V = np.linalg.eigh(covariance)
    D = np.diag(D)
    V_t = np.transpose(V)
    D[D!=0]=D[D!=0]**(-1/2)
    A_w = np.dot(V, np.dot(D, V_t))
    z = np.dot(A_w, y)
    return z


def return_v(v, w_t):
    output = np.dot(w_t, v)
    output += abs(np.linalg.norm(v)-1)*1000
    return output


def fastICA(input_paths, output_paths, iterations=25, eps=0.000001):
    x = load_audios(input_paths)
    # CENTERING
    x, _ = centering(x)
    # WHITENING
    z = whitening(x)

    w_old = np.random.random_sample((x.shape[0]))
    w_magnitude = np.linalg.norm(w_old)
    w_old *= 1/w_magnitude
    w_old = np.transpose(w_old)
    w_old_t = np.transpose(w_old)

    # ITERIATIONS :>
    for i in range(iterations):
        print("iteration:", i)
        w = np.average(np.dot(z,((np.dot(w_old_t, z)))**3)) - 3*w_old
        w_magnitude = np.linalg.norm(w)
        w *= 1/w_magnitude
        w_t = np.transpose(w)
        if abs(1-np.dot(w_t, w_old))<eps:
            print(np.dot(w_t, w_old))
            break
        w_old = w

    v = scipy.optimize.fmin(return_v, np.ones(w_old.shape), args=(w_t,))
    v_t = np.transpose(v)

    x_e = np.dot(np.array([w_t, v_t]), z)


    save_audios(output_paths, x_e)



if __name__ == "__main__":
    input_paths = ["audio1_clip.wav", "audio2_clip.wav", "audio3_clip.wav"]
    mixed_paths = ["mixed1.wav", "mixed2.wav", "mixed3.wav"]
    output_paths = ["audio1_estimated.wav", "audio3_estimated.wav", "audio3_estimated.wav"]

    input_paths = ["audio1_clip.wav", "audio2_clip.wav"]
    mixed_paths = ["mixed1.wav", "mixed2.wav"]
    output_paths = ["audio1_estimated.wav", "audio2_estimated.wav"]

    # A = np.array([1,2,3,4,5,6])
    # B = np.array([1,2,3,4,5,6])
    # print(np.dot(A,B))

    #mixing(input_paths, mixed_paths, False)
    fastICA(input_paths, output_paths)
