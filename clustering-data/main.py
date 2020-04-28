
import numpy as np
import matplotlib.pyplot as plt

"""class knn:
    #"k-nearest neighbor calssifier algorithm
        #Arguments
        #k : int, (default = 5), Number of neighbors
    def __init__(self, k=5, train_data, test_data)
"""

def load_data(): # Import data set from file
    labels = ["animals", "countries", "fruits", "veggies"]

    a = np.genfromtxt(labels[0], delimiter=" ")
    a[:, 0] = 1
    b = np.genfromtxt(labels[1], delimiter=" ")
    b[:, 0] = 2
    c = np.genfromtxt(labels[2], delimiter=" ")
    c[:, 0] = 3
    d = np.genfromtxt(labels[3], delimiter=" ")
    d[:, 0] = 4
    dataX = np.concatenate((a,b,c,d), axis=0)
    #dataX = np.delete(dataX, 0, 1)
    trainDataSize = int(np.size(dataX, 0) * 0.8) # split 80% training data and 20% test data
    trainDataSet = dataX[:trainDataSize, :]
    testDataSet = dataX[trainDataSize:, :]
    print("dataset", np.shape(trainDataSet), np.shape(testDataSet))
    print("dataset", trainDataSet[0, :5], testDataSet[55, :5])
    return trainDataSet, testDataSet


def data_minmax(X): # Find the min and max values for each column
    for i in range(len(X[0])):
        col_values = [row[i] for row in X]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


def normalize_dataset(dataset, minmax): # Rescale dataset columns to the range 0-1
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def distance(l, X): # Calculate the distance between two vectors
    a = np.size(X, 0)
    norm = np.zeros((a, 1))
    norm22 = X
    if l == 1: # l1 norm, sum of the absolute values in each feature vector
        for i in range(0, a):
            norm = np.sum(np.abs(X[i,:]))
    elif l == 2: # l2 norm, Euclidean distance
        for i in range(0, a):
            norm[i, :] = np.sqrt(np.sum(np.square(X[i, :]*X[i-1, :])))
#            print(i,norm)

    elif l == 22: #l2 norm, normalize
        for i in range(0, a):
            norm[i, :] = np.sqrt(np.sum(np.square(X[i, :]*X[i-1, :])))
            norm22[i, :] = X[i, :] / norm[i, :]


    return norm


def train(trainDataSet):
    # X, L, trainDataSet, testDataSet = import_data(fname)
    # print(X, L)
    norm = distance(2, trainDataSet)
    print(norm[1, :])
    return norm

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
    score = np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q))
    return score


def predict(x, k): # Get nearest neighbors
    L = [(y, sim(x, z)) for (y,z) in trainDataSet]
    L.sort(key=lambda tup: tup[1], reverse = True)  # sorts in place
    # print L[:k]
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
    trainDataSets, testDataSets = load_data()
    norm = train(trainDataSets)
    pred = predict(testDataSets, 5)
    print("predict result", pred)
    plt.plot(pred)
    plt.show()
