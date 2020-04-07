import numpy as np




def cov(x):
    covariance_matrix = np.zeros((len(x),len(x)),dtype=float)
    means = np.mean(x, axis=1)
    for i in range(len(x)):
        for j in range(len(x)):
            covariance_matrix[i,j] = np.mean(np.multiply(x[i],x[j])) - means[i]*means[j]
    return covariance_matrix



if __name__=="__main__":
    mat = np.matrix([[1,2,3], [10,21,34], [12,1,38]])
    mat = np.array([[1,2,3], [10,21,34], [12,1,38]])
    print(cov(mat))
    print(np.cov(mat))
    mat = np.array([[1,2,3], [10,21,34], [12,1,38]])
    print(np.cov(mat))