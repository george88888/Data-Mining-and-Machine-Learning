
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

"""class knn:
    #"k-nearest neighbor calssifier algorithm
    	#Arguments
    	#k : int, (default = 5), Number of neighbors
    def __init__(self, k=5, train_data, test_data)
"""


def import_data(fname):
    _X = np.genfromtxt(fname, delimiter=" ")
    _X = np.delete(_X, 0, 1)
    _L = np.genfromtxt(fname, delimiter=" ", dtype='S10', usecols=(0))
    return _X, _L


def norms(l, X):
    a = np.size(X, 0)
    norm = np.zeros((a, 1))
    if l == 1: #l1 norm, sum of the absolute values in each feature vector
        for i in range(0, a):
            norm = np.sum(np.abs(X[i,:]))
    elif l == 2: #l2 norm,
        for i in range(0, a):
            norm[i,:] = np.sqrt(np.sum(X[i,:]*X[i,:]))
#            print(i,norm)


    elif l == 22: #l2 norm,
        for i in range(0, np.size(X, 0)):
            norm = np.sqrt(np.sum(X[i,:]*X[i,:]))
            X[i,:] = X[i,:] / norm

    return norm


def train():
    filename = "animals"
    X, L = import_data(filename)
    print(X, L)
    norm = norms(2, X)
    print(norm)
    return


if __name__ == "__main__":
    train()

#    plt.scatter(neg[:, 0], neg[:, 1], c='r')
#    plt.scatter(pos[:, 0], pos[:, 1], c='b')
#    plt.show()