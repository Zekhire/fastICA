import scipy.io.wavfile as wave
import scipy.optimize
from matplotlib import pyplot as plt
import numpy as np

def draw_distribution(x, points=-1, plotname="test.jpg", printing=True, show=True, save=True):
    fig = plt.figure()
    plt.plot(x[0][:points], x[1][:points], "bo")
    plt.plot([-100000, 100000],[0,0], "k")
    plt.plot([0,0],[-100000, 100000], "k")
    plt.plot(0,0, "ko")
    plt.grid()
    plt.xlim([min(x[0][:points])*1.2, max(x[0][:points])*1.2])
    plt.ylim([min(x[1][:points])*1.2, max(x[1][:points])*1.2])
    plt.xlabel("audio1")
    plt.ylabel("audio2")
    if show:
        plt.show()
    if save:
        fig.savefig(plotname, dpi=fig.dpi)
        

def load_audios(input_path):
    audio = wave.read(input_path)
    samples = np.array(audio[1], dtype=float)
    Fs = audio[0]
    return samples, Fs


def load_samples(input_paths):
    x = []
    Fs = 0
    for input_path in input_paths:
        audio, Fs = load_audios(input_path)
        # print("MAX:", max(audio))
        x.append(audio)
    x = np.array(x)
    return x, Fs


def save_audios(output_paths, z, Fs):
    for i in range(len(output_paths)):
        # print("MAX:", max(z[i]))
        wave.write(output_paths[i], Fs, z[i])


def mixing(x, printing=True, show=True, save=True):
    audios_number = len(x)
    if show or save:
        draw_distribution(x, plotname="distribution_sources.jpg", printing=printing, show=show, save=save)
    
    # A = np.random.random_integers(1,5,(audios_number, audios_number))
    A = np.random.random_sample((audios_number, audios_number))
    for i in range(len(A)):
        A[i] *= 1/np.linalg.norm(A[i])
        
    if printing:
        print("A:")
        print(A)
        print("inversed A:")
        print(np.linalg.inv(A))
        print()

    y = np.dot(A, x)
    if show or save:
        draw_distribution(y, plotname="distribution_mixed.jpg", printing=printing, show=show, save=save)
    return y




def centering(y, printing=True):
    if printing:
        print("mean before centering: ", np.mean(y[0]), np.mean(y[1]))

    new_y = y.copy()
    means = []

    for i in range(len(y)):
        mean = np.mean(y[i])
        means.append(mean)
        new_y[i] -= mean
    means = np.array(means)

    if printing:
        print("mean after centering:  ", np.mean(new_y[0]), np.mean(new_y[1]))
        print()

    return new_y, means


def whitening(y, printing=True, show=True, save=True):
    y_t = np.transpose(y)
    #covariance=np.cov(np.dot(y, y_t))
    covariance=np.cov(y)
    D, V = np.linalg.eigh(covariance)
    D = np.diag(D)
    V_t = np.transpose(V)

    if printing:
        print("covariance matrix: ")
        print(covariance)
        print("covariance matrix V*D*V_t: ")
        print(np.dot(V, np.dot(D, V_t)))
        
    D[D!=0]=D[D!=0]**(-1/2)
    A_w = np.dot(V, np.dot(D, V_t))
    z = np.dot(A_w, y)

    if printing:
        print("covariance matrix after whitening:\n", np.cov(z))
        for i in range(len(z)):
            print("covariance of signal "+str(i)+" after whitening: ", np.cov(z[i]))
        print()

    if show or save:
        draw_distribution(z, plotname="distribution_whitened.jpg", printing=printing, show=show, save=save)

    return z


def w_init(z, printing=True):
    w = np.random.random_sample((z.shape[0]))
    w /= np.linalg.norm(w)
    w = np.transpose(w)
    if printing:
        print("norm of initialized w:", np.linalg.norm(w))
        print()
    return w


def return_v(v, w_t):
    output = abs(np.dot(w_t, v))*10000
    output += abs(np.linalg.norm(v)-1)*1000
    return output

def determine_v(w_t, printing=True):
    v_init = np.array([-5, 10])
    v = scipy.optimize.fmin(return_v, v_init, args=(w_t,), disp=False)
    v_t = np.transpose(v)
    if printing:
        print("multiplication of w_t and v_t:  ", np.dot(w_t, v))
        print("norm of v:\t\t\t", np.linalg.norm(v))
        print("matrix with w_t and v_t:")
        print(np.array([w_t, v_t]))
        print()
    return v


def fastICA(y, iterations=25, eps=0.000001, printing=True, show=True, save=True):
    print("centering")
    y, _  = centering(y, printing=printing)
    print("whitening")
    z     = whitening(y, printing=printing, show=show, save=save)
    print("determining w vector")
    w_old = w_init(z, printing=printing)
    # ITERATIONS :>
    for i in range(iterations):
        if printing:
            print("iteration:", i)
        
        temp = z *(np.dot(np.transpose(w_old), z)**3)
        averages = []
        for j in range(len(temp)):
            averages.append(np.average(temp[j]))
        averages = np.array(averages)
        w = averages - 3*w_old

        w *= 1/np.linalg.norm(w)
        w_t = np.transpose(w)
        if abs(1-np.dot(w_t, w_old)) < eps:
            if printing:
                print("multiplication of w_t, w_old:   ", np.dot(w_t, w_old))
                print()
            break
        w_old = w

    v = determine_v(w_t, printing=printing)
    v_t = np.transpose(v)

    A_inv = np.array([w_t, v_t])
    x_e = np.dot(A_inv, z)

    return x_e



if __name__ == "__main__":
    input_paths = ["audio2_clip.wav", "audio3_clip.wav"]
    mixed_paths = ["mixed2.wav", "mixed3.wav"]
    output_paths = ["audio2_estimated.wav", "audio3_estimated.wav"]

    x, Fs = load_samples(input_paths)
    y = mixing(x, printing=True, show=False, save=True)
    save_audios(mixed_paths, y, Fs)

    
    y, Fs = load_samples(mixed_paths)
    x_e = fastICA(y, printing=True, show=False, save=True)
    save_audios(output_paths, x_e, Fs)
