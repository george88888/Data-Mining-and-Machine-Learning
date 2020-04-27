
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
    norm22 = X
    if l == 1: #l1 norm, sum of the absolute values in each feature vector
        for i in range(0, a):
            norm = np.sum(np.abs(X[i,:]))
    elif l == 2: #l2 norm,
        for i in range(0, a):
            norm[i, :] = np.sqrt(np.sum(X[i, :]*X[i, :]))
#            print(i,norm)

    elif l == 22: #l2 norm, normalize
        for i in range(0, a):
            norm[i, :] = np.sqrt(np.sum(X[i, :]*X[i, :]))
            norm22[i, :] = X[i, :] / norm[i, :]
    pass

    return norm


def train(fname):

    X, L = import_data(fname)
#    print(X, L)
    norm = norms(2, X)
    print(norm[1, :])
    print(np.shape(X))
    plt.scatter(norm, L)
    plt.show()
    return

def validation():

    return


def test():
    return


def evaluation():
    # Precision (P): Precision tells us about all the correct predictions out of total positive predictions
    # Recall (R): Recall tells us about how many instances were correctly classified out of all positive instances
    # F-score (F) is the harmonic mean between Precision(P) and Recall (R).

    return


def sim(p, q):
    score = np.dot(p,q) / (np.linalg.norm(p) * np.linalg.norm(q))
    return score


def predict(x, k):
    L = [(y, sim(x, z)) for (y,z) in train_data]
    L.sort(key=lambda tup: tup[1], reverse = True)  # sorts in place
    #print L[:k]
    score = sum([e[0] for e in L[:k]])
    if score > 0:
        return 1
    else:
        return -1


def accuracy():
    corrects = 0
    k = 5
    for (y, x) in test_data:
        if y == predict(x, k):
            corrects += 1
    accuracy = float(corrects) / float(len(test_data))
    print("Accuracy =", accuracy)
    return accuracy


if __name__ == "__main__":
    lables = ["animals", "countries", "fruits", "veggies"]
    train(lables[0])
    train(lables[1])
    train(lables[2])
    train(lables[3])

#    plt.scatter(neg[:, 0], neg[:, 1], c='r')
#    plt.scatter(pos[:, 0], pos[:, 1], c='b')
#    plt.show()
