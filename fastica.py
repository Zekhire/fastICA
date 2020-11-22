import scipy.io.wavfile as wave
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def load_audios(input_path):
    audio = wave.read(input_path)
    samples = np.array(audio[1], dtype=float)
    Fs = audio[0]
    return samples, Fs


def scaling_down(x):
    x_scaled = x/max(abs(x))
    return x_scaled


def save_audios(output_paths, z, Fs, scaling=True):
    for i in range(len(output_paths)):
        if scaling:
            x = scaling_down(np.array(z)[i])
        else:
            x =  np.array(z)[i]
        wave.write(output_paths[i], Fs, x)


def draw_distribution(x, points=-1, plotname="test.jpg", b=None, printing=True, show=True, save=True):
    # if len(x) == 2:
        fig = plt.figure()
        plt.plot(np.array(x[0])[0][:points], np.array(x[1])[0][:points], "bo", ms=0.25)
        plt.plot([-100000, 100000], [0,0], "k")
        plt.plot([0,0], [-100000, 100000], "k")
        plt.plot(0,0, "ko")
        plt.grid()
        plt.xlim([min(np.array(x[0])[0][:points])*1.2, max(np.array(x[0])[0][:points])*1.2])
        plt.ylim([min(np.array(x[1])[0][:points])*1.2, max(np.array(x[1])[0][:points])*1.2])
        plt.xlabel("audio1")
        plt.ylabel("audio2")
        if type(b)!= type(None):
            b_inv = np.linalg.inv(b)
            if printing:
                print(b_inv)
                # print(b_inv[0,0], b_inv[1,0], b_inv[0,1], b_inv[1,1])
            plt.plot([min(np.array(x[0])[0][:points]), 10*b_inv[0,0]+min(np.array(x[0])[0][:points])], [min(np.array(x[1])[0][:points]), 10*b_inv[1,0]+min(np.array(x[1])[0][:points])])
            plt.plot([min(np.array(x[0])[0][:points]), 10*b_inv[0,1]+min(np.array(x[0])[0][:points])], [min(np.array(x[1])[0][:points]), 10*b_inv[1,1]+min(np.array(x[1])[0][:points])])

        if show:
            plt.show()
        if save:
            fig.savefig(plotname, dpi=fig.dpi)
    # else:
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(x[0][:2000], x[1][:2000], x[2][:2000], 'b')
        # ax.plot([-100000, 100000],[0,0],[0,0], "k")
        # ax.plot([0,0],[-100000, 100000],[0,0], "k")
        # ax.plot([0,0],[0,0], [-100000, 100000], "k")
        # ax.plot([0],[0],[0], "ko")
        # ax.set_xlim([min(x[0][:points])*1.2, max(x[0][:points])*1.2])
        # ax.set_ylim([min(x[1][:points])*1.2, max(x[1][:points])*1.2])
        # ax.set_zlim([min(x[2][:points])*1.2, max(x[2][:points])*1.2])
        # plt.show()



def load_samples(input_paths):
    x = []
    Fs = 0
    for input_path in input_paths:
        audio, Fs = load_audios(input_path)
        # print("MAX:", max(audio))
        x.append(audio)
    x = np.matrix(x, np.float32)
    return x, Fs

        


def mixing(x, A=None, eps_det=0.001, printing=True, show=True, save=True):
    audios_number = len(x)
    if show or save:
        draw_distribution(x, plotname="distribution_sources.jpg", printing=printing, show=show, save=save)
    
    if A is None:
        while True:
            # A = np.random.random_integers(1,5,(audios_number, audios_number))
            A = np.matrix(np.random.random_sample((audios_number, audios_number)))
            for i in range(len(A)):
                A[i] *= 1/np.linalg.norm(A[i])
            if abs(np.linalg.det(A)) > eps_det:
                # print(abs(np.linalg.det(A)))
                break 

    if printing:
        print("A:")
        print(A)
        print("inversed A:")
        print(np.linalg.inv(A))
        print()

    # Mixing
    y = A*x

    if show or save:
        draw_distribution(y, plotname="distribution_mixed.jpg", printing=printing, show=show, save=save)
    return y




def centering(y, printing=True):
    if printing:
        for i in range(len(y)):
            print("mean of signal "+str(i)+" before centering:  ", np.mean(y[i]))

    new_y = y.copy()
    means = np.matrix(np.mean(new_y, axis=1))
    new_y -= means

    if printing:
        for i in range(len(y)):
            print("mean of signal "+str(i)+" after centering:  ", np.mean(new_y[i]))
        print()

    return new_y, means


def whitening(y, printing=True, show=True, save=True):
    covariance=np.cov(y)
    D, V = np.linalg.eigh(covariance)
    D = np.matrix(np.diag(D))
    V = np.matrix(V)
    V_t = np.transpose(V)

    if printing:
        print("covariance matrix: ")
        print(covariance)
        print("covariance matrix V*D*V_t: ")
        print(V*D*V_t)
    

    Drs = np.matrix(np.diag(np.power(np.array(D.diagonal()),(-1/2))[0]))  # xD
    A_w = V*Drs*V_t
    z = A_w*y

    if printing:
        print("covariance matrix after whitening:\n", np.cov(z))
        for i in range(len(z)):
            print("covariance of signal "+str(i)+" after whitening: ", np.cov(z[i]))
        print()

    if show or save:
        draw_distribution(z, plotname="distribution_whitened.jpg", printing=printing, show=show, save=save)

    return z


def w_init(z, printing=True):
    w = np.matrix(np.random.random_sample((z.shape[0])), np.float64)
    w /= np.linalg.norm(w)
    w = np.transpose(w)
    if printing:
        print("initialized w:")
        print(w)
        print("norm of initialized w:", np.linalg.norm(w))
        print()
    return w


def determine_w(z, iterations, eps, printing=True):
    w_old = w_init(z, printing=printing)

    for i in range(iterations):
        if printing:
            print("iteration:", i)
        
        w = np.mean(np.multiply(z,np.power(np.transpose(w_old)*z,3)), axis=1) - 3*w_old
        w *= 1/np.linalg.norm(w)
        w_t = np.transpose(w)

        if abs(1-abs(w_t*w_old)) < eps:
            if printing:
                print("w:")
                print(w)
                print("multiplication of w_t, w_old:   ", w_t*w_old)
                print()
            break
        w_old = w

    return w



def determine_v_old(w, printing=True):
    def find_v(v, w_t):
        v=np.transpose(np.matrix(v))
        output = abs(w_t*v)*10000
        output += abs(np.linalg.norm(v)-1)*1000
        return output

    w_t = np.transpose(w)
    v_init = np.transpose(np.matrix(np.random.random_integers(1,5,(w.shape[0]))))

    v = scipy.optimize.fmin(find_v, v_init, args=(w_t,), disp=False)
    v = np.transpose(np.matrix(v))
    v_t = np.transpose(v)
    if printing:
        print("v:")
        print(v)
        print("multiplication of w_t and v_t:  ", w_t*v)
        print("norm of v:\t\t\t", np.linalg.norm(v))
        print("matrix with w_t and v_t:")
        print(np.concatenate((w_t, v_t)))
        print()
    return v

def determine_v(w, printing=True):
    v = np.transpose(np.matrix([-w.item(1), w.item(0)]))
    w_t = np.transpose(w)
    if printing:
        print("matrix v:  ")
        print(v)
        print("multiplication of w_t and v_t:  ", w_t*v)
        print("norm of v:\t\t\t", np.linalg.norm(v))
        print()
    return v


def splitting(z, w, v, printing=True):
    w_t = np.transpose(w)
    v_t = np.transpose(v)
    wv = np.matrix(np.concatenate((w_t, v_t)))
    if printing:
        print("matrix with w_t and v_t:")
        print(wv)
        print()
    x_e = wv*z
    return x_e


def determine_B(z, iterations, eps, printing=True):
    B = np.matrix(np.zeros((z.shape[0],z.shape[0]), np.float64))

    for k in range(len(z)):                                 # for each signal
        print("signal: ", k)
        w_old = w_init(z, printing=printing)
        for i in range(iterations):                         # for each iteration
            print("iteration: ", i)
            
            w = np.mean(np.multiply(z,np.power(np.transpose(w_old)*z,3)), axis=1) - 3*w_old

            if k > 0:
                w = w - ((B*np.transpose(B))*w)
            w *= 1/np.linalg.norm(w)
            w_t = np.transpose(w)

            if abs(1-abs(w_t*w_old)) < eps: # in presentation w_t*w_old should be close to 1 but here it may be negative
                if printing:
                    print("multiplication of w_t, w_old:   ", w_t*w_old)
                    print()
                break
            w_old = w
        B[:,k] = w

    if printing:
        print("B:")
        print(B)
    return B


def splitting_n3(z, B):
    B_t = np.transpose(B)
    x_s = B_t*z
    return x_s


def fastICA(y, iterations=25, eps=1e-6, printing=True, show=True, save=True):
    print("1. centering")
    y, _  = centering(y, printing=printing)
    print("2. whitening")
    z     = whitening(y, printing=printing, show=show, save=save)
    # output_paths = ["./separated/whited5.wav", "./separated/whited6.wav", "./separated/whited7.wav"]
    # save_audios(output_paths, z, 44100)
    if len(y) == 2:
        print("3. determining w vector")
        w = determine_w(z, iterations, eps, printing)
        print("4. determining v vector")
        v = determine_v(w, printing=printing)
        if show or save:
            w_t = np.transpose(w)
            v_t = np.transpose(v)
            wt = np.matrix(np.concatenate((w_t, v_t)))
            draw_distribution(z, points=-1, plotname="v.jpg", b=wt, printing=printing, show=show, save=save)
        print("5. splitting")
        x_s = splitting(z, w, v, printing=printing)
    else:
        print("3. determining B vector")
        B = determine_B(z, iterations, eps, printing=printing)
        print("4. splitting")
        x_s = splitting_n3(z, B)
    return x_s


if __name__ == "__main__":
    # O=====<=>=====<=>=====<=>===== #
    #       Editable parameters      #
    # O=====<=>=====<=>=====<=>===== #
    printing = True
    show = False
    save = True

    input_paths  = ["./clips/clip5.wav", "./clips/clip6.wav", "./clips/clip7.wav"]
    mixed_paths  = ["./mixed/mixed5.wav", "./mixed/mixed6.wav", "./mixed/mixed7.wav"]
    output_paths = ["./separated/separated5.wav", "./separated/separated6.wav", "./separated/separated7.wav"]

    A = None    # If A is None, then during noise addition A will be generated randomly
    A = np.matrix([[0.29081642, 0.62527361, 0.72419522],[ 0.88537055, 0.253161, 0.38990833], [0.12644464, 0.29337357, 0.94759891]])
    # O=====<=>=====<=>=====<=>===== #
    #             Program            #
    # O=====<=>=====<=>=====<=>===== #

    # Following block of code can be commented, in order for this script to use pure
    # FastICA algorithm
    x, Fs = load_samples(input_paths)
    y = mixing(x, A=A, printing=printing, show=show, save=save)
    save_audios(mixed_paths, y, Fs, False)
    
    x_s = fastICA(y, printing=printing, show=show, save=save)
    save_audios(output_paths, x_s, Fs)
