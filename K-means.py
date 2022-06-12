import idx2numpy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
from sklearn.neural_network import _multilayer_perceptron



def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)




file = 'samples/train-images.idx3-ubyte'
testFiles = 'samples/t10k-images.idx3-ubyte'
trainLabels = 'samples/train-labels.idx1-ubyte'
arr = idx2numpy.convert_from_file(file)
tests = idx2numpy.convert_from_file(testFiles)
trainLabels = idx2numpy.convert_from_file(trainLabels)
tests = tests.reshape(10000, 28*28)
arr = arr.reshape(60000, 28*28)
kmeans = KMeans(n_clusters=10, tol=1)
kmeans.fit(arr)
print(adjusted_rand_score(trainLabels, kmeans.labels_))
print(purity_score(trainLabels, kmeans.labels_))






