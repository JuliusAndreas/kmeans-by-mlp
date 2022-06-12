import pickle
from sklearn.neural_network import MLPClassifier
import numpy
from numpy import random
import idx2numpy
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score



def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return numpy.sum(numpy.amax(contingency_matrix, axis=0)) / numpy.sum(contingency_matrix)


# load the model from disk
filename = 'mlp_model.sav'
mlp_model = pickle.load(open(filename, 'rb'))
file = 'samples/train-images.idx3-ubyte'
trainLabels = 'samples/train-labels.idx1-ubyte'
trainLabels = idx2numpy.convert_from_file(trainLabels)
arr = idx2numpy.convert_from_file(file)
seeds = []
kmeansLabels = [-1]*60000

data = list(numpy.array(arr).tolist())
indexList = list(range(len(data)))

for i in range(10):
    # random seeds
    seeds.append(random.randint(0, len(data)))
    center = data[seeds[i]]
    data.pop(seeds[i])
    kmeansLabels[indexList[seeds[i]]] = seeds[i]
    indexList.pop(seeds[i])
    for j in range(len(data)):
        # for each data determine if it is in the same cluster with the seed
        mlp_input = numpy.append(center, data[j])
        mlp_input = mlp_input.reshape(1, -1)
        predictedLabel = mlp_model.predict(mlp_input)
        # check if seed and data are in same cluster
        print(mlp_input)
        if(predictedLabel[0]==0):

            kmeansLabels[indexList[j]] = seeds[i];
            # pop data and repeat the process with remaining data until 10 seeds are processed
            data.pop(j)
            indexList.pop(j)


print(adjusted_rand_score(trainLabels, kmeansLabels))
print(purity_score(trainLabels, kmeansLabels))


