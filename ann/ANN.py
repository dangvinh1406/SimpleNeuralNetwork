import numpy
import copy

EPSILON = 0.0001

class ANN:
    def __init__(
        self,
        score_function="linear",
        loss_function="square",
        learning_rate=0.001,
        max_iterator=100000,
        tolerance=0.0001,
        fashion="binary"):
        '''
        fashion: ovr, ovo, mul
        '''

        self.__learningRate = learning_rate
        self.__maxIterator = max_iterator
        self.__tolerance = tolerance
        self.__fashion = fashion
        
        self.__scoreFunction = ANN.linear()
        if score_function == "sigmoid":
            self.__scoreFunction = ANN.sigmoid()
        elif score_function == "tanh":
            self.__scoreFunction = ANN.tanh()

        self.__lossFunction = ANN.square()
        if loss_function == "cross_entropy":
            self.__lossFunction = ANN.cross_entropy()

        self.__weights = None
        self.__labels = None

    def train(self, X, Y):
        try:
            if X.shape[1] != len(Y):
                return False
        except:
            return False

        X = numpy.vstack((X, numpy.ones((1, int(X.shape[1])))))

        numberOfModel = 1
        weights = []
        self.__labels = list(set(Y))
        if self.__fashion == "binary":
            numberOfModel = len(self.__labels) # one vs all
            weights = [numpy.ones((1, int(X.shape[0])), dtype=numpy.float64)]*numberOfModel
        else:
            weights = [numpy.ones((len(self.__labels), int(X.shape[0])+1), dtype=numpy.float64)]

        i = 0
        while (i < numberOfModel):
            weight = weights[i]
            y = None
            if self.__fashion != "binary":
                y = self.__scoreFunction["lim"][0]*numpy.ones((len(self.__labels), int(X.shape[1])))
                for l in range(len(Y)):
                    y[self.__labels.index(Y[l]), l] = self.__scoreFunction["lim"][1]
            else:
                y = self.__scoreFunction["lim"][0]*numpy.ones((1, int(X.shape[1])))
                for l in range(len(Y)):
                    if Y[l] == self.__labels[i]:
                        y[0, l] = self.__scoreFunction["lim"][1]

            for epoch in range(self.__maxIterator):
                dScore_dWeight = self.__scoreFunction["d"](weight, X)
                fx = self.__scoreFunction["f"](weight, X)
                dLoss_dScore = self.__lossFunction["d"](y, fx)
                deltaWeight = self.__learningRate*numpy.sum(
                    dLoss_dScore*dScore_dWeight*X, 1).T
                weight_ = weight-deltaWeight
                error = numpy.linalg.norm(deltaWeight)
                if error < self.__tolerance:
                    print("Break in epoch "+str(epoch))
                    break
                weight = weight_
            weights[i] = weight
            i += 1

        self.__weights = weights

    def predict(self, x):
        if self.__weights is None:
            return None

        x = numpy.vstack((x, numpy.array([1])))

        y = []
        for weight in self.__weights:
            fx = self.__scoreFunction["f"](weight, x)
            y.append(fx)

        if self.__fashion != "binary":
            return self.__labels[numpy.argmax(y[0][0])]
        else:
            y = numpy.array([v[0][0] for v in y])
            return self.__labels[numpy.argmax(y)]
        return None


    def getWeight(self):
        return copy.deepcopy(self.__weights)

    # --------------- Score functions ------------------
    @staticmethod
    def linear():
        def f(W, x):
            return W.dot(x)
        def d(W, x):
            return 1
        return {"f": f, "d": d, "lim": [-1, 1]}

    @staticmethod
    def sigmoid():
        def f(W, x):
            return 1/(1+numpy.exp(-W.dot(x)))
        def d(W, x):
            return f(W, x)*(1-f(W, x))
        return {"f": f, "d": d, "lim": [0, 1]}

    @staticmethod
    def tanh():
        def f(W, x):
            v = W.dot(x)
            return (numpy.exp(v)-numpy.exp(-v))/(numpy.exp(v)+numpy.exp(-v))
        def d(W, x):
            t = f(W, x)
            return 1-t*t
        return {"f": f, "d": d, "lim": [-1, 1]}


    # --------------- Loss functions ------------------
    @staticmethod
    def cross_entropy():
        def f(y, fx):
            return -y*(numpy.log(fx))-(1-y)*(numpy.log(1-fx))
        def d(y, fx):
            return (fx-y)/(fx*(1-fx)+EPSILON)
        return {"f": f, "d": d}

    @staticmethod
    def square():
        def f(y, fx):
            return 0.5*(y-fx)*(y-fx)
        def d(y, fx):
            return (y-fx)
        return {"f": f, "d": d}
