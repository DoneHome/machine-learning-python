# coding:utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

#data = np.genfromtxt("./data/logistic_regression.csv", dtype=np.float64, delimiter=",")
data = np.genfromtxt("./data/logistic_regression_2.txt", dtype=np.float64, delimiter=",")
plt.figure(1,figsize=(15,15))


class LR():
    """
    Logistic Regression
    """

    def __init__(self):
        self.alpha = 0.001 # learning rate
        self.num_iters = 1000
        self.eps = 1e-1
        self.batch = 1 # mini batch for training, here is SGD
        self.theta = None # A numpy array of shape (1, M) containing weights

    def hypothesis_func(self, x):
        """sigmoid
        """
        return 1.0/(1.0 + np.exp(-1.0 * x.dot(self.theta.T)))

    def cost_func(self, x):
        pass

    def evaluate_gradient(self, x, y):
        h = self.hypothesis_func(x)
        return np.sum((y.reshape(-1, 1) - h) * x, axis = 0)
        #return np.sum((y.reshape(-1, 1) - self.hypothesis_func(x)) * x, axis = 0)

    def evaluate_loss(self, x, y):
        loss = 0

        num_train = x.shape[0]

        for i in range(num_train):
            if y[i] == 1:
                loss += np.log(self.hypothesis_func(x[i]))
            elif y[i] == 0:
                loss += np.log(1.0 - self.hypothesis_func(x[i]))
        return np.abs(loss)
        #return np.sum(y.reshape(-1, 1) * np.log(self.hypothesis_func(x)) + (1.0 - y.reshape(-1, 1)) * np.log(1.0 - self.hypothesis_func(x)))

    def NewtonOptimization(self):
        """
        use newton method to optimize gradient descent
        """
        pass

    def GradientOptimization(self, X, Y):
        """
        use mini-batch or SGD(batch=1) to optimize gradient descent
        TODO: Add l1/l2
        """
        num_train = X.shape[0]
        loss = sys.maxint
        _iter = 0

        loss_epoch = {}

        while loss > self.eps and _iter < self.num_iters:
            for idx in range(num_train):
                #self.alpha = 4.0 / (1.0 + idx + _iter) + 0.001
                mask = np.random.choice(num_train, self.batch, replace=False)
                x_train = X[mask]
                y_train = Y[mask]

                grad = self.evaluate_gradient(x_train, y_train)
                self.theta += 1.0/self.batch * self.alpha * grad

            loss = 1.0/num_train * self.evaluate_loss(X, Y)

            if _iter % 10 == 0:
                loss_epoch[_iter] = loss

            _iter += 1
        return loss_epoch

    def train(self, X, Y, opt="batch"):
        """
        X: A numpy array of shape (N, M) containing training data where N is sample numbers, M is feature numbers
        Y: A numpy array of shape (N, ) containing training label
        """
        self.theta = np.random.randn(1, X.shape[1])
        #self.theta = np.ones((1, X.shape[1]))

        if opt == "batch":
            loss_epoch = self.GradientOptimization(X, Y)
            return loss_epoch
        elif opt == "newton":
            pass
        else:
            raise Exception("%s don't exists" % opt)


if __name__ == "__main__":
    X = data[:,:2]
    #x = preprocessing.scale(X)
    #X = min_max_scaler.fit_transform(X)
    trainX = np.hstack([np.ones((X.shape[0], 1)), X])
    trainY = data[:,-1]

    lr = LR()
    loss_epoch = lr.train(trainX, trainY, opt="batch")
    #print lr.theta

    p1 = plt.subplot(211)

    loss_x = []
    loss_y = []
    for k in sorted(loss_epoch.keys()):
        loss_x.append(k)
        loss_y.append(loss_epoch[k])

    p1.plot(np.array(loss_x), np.array(loss_y))
    p1.set_ylabel("loss")
    p1.set_xlabel("epoch")

    p2 = plt.subplot(212)
    pos_data = data[data[:,-1]==1]
    neg_data = data[data[:,-1]==0]
    p2.scatter(pos_data[:,0], pos_data[:,1], color="red")
    p2.scatter(neg_data[:,0], neg_data[:,1], color="blue")
    p2.set_ylabel("x2")
    p2.set_xlabel("x1")


    _x = np.arange(-5.0, 5.0, 0.1)
    #_x = np.arange(30, 100, 10)
    _y = (-lr.theta[0][0] - lr.theta[0][1]*_x) / lr.theta[0][2]
    p2.plot(_x, _y)

    plt.show()

