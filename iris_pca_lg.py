import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import datasets
from ann.ANN import ANN

COLOR = ['b', 'r', 'g']
SHAPE = ['*', 'o', '^']

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

plt.clf()
plt.cla()

pca = decomposition.PCA(n_components=2)
pca.fit(X)
Xnew = pca.transform(X)

for i in range(len(Xnew)):
    plt.plot([Xnew[i][0]], [Xnew[i][1]], COLOR[y[i]]+SHAPE[y[i]])

ann = ANN(score_function="tanh",
        loss_function="cross_entropy",
        learning_rate=0.01,
        max_iterator=100000,
        tolerance=0.000001,
        fashion="binary")
ann.train(Xnew.T, list(y))
weights = ann.getWeight()
print(weights)

xylim = [
	int(min(Xnew[:, 0])-1), int(max(Xnew[:, 0])+1), 
	int(min(Xnew[:, 1])-1), int(max(Xnew[:, 1])+1)]

xline = list(np.arange(xylim[0], xylim[1], 0.001))
for i in range(len(weights)):
	weight = weights[i][0]
	yline = [-(weight[0]*x-weight[2])/weight[1] for x in xline]
	plt.plot(xline, yline, 'k-')

plt.axis(xylim)
plt.show()

TRUE = 0
for i in range(len(y)):
	if y[i] == ann.predict(np.expand_dims(Xnew[i, :], 0).T):
		TRUE += 1
print(TRUE)
