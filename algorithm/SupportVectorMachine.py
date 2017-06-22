# coding:utf-8
import sys
import numpy as np
import matplotlib.pyplot as plt
from SequentialMinimalOptimization import SMO

#data = np.genfromtxt("./data/svm.csv", dtype=np.float64, delimiter=",")
data = np.genfromtxt("./data/svm_rbf.csv", dtype=np.float64, delimiter=",")

fig = plt.figure()

def draw(data, sv, w, b):
    """
    """
    pos_data = data[data[:,-1]==1]
    neg_data = data[data[:,-1]==-1]
    plt.scatter(pos_data[:,0], pos_data[:,1], color="red")
    plt.scatter(neg_data[:,0], neg_data[:,1], color="blue")

    plt.scatter(sv[:,0], sv[:,1], color="green")

    min_x = min(data[:, 0])
    max_x = max(data[:, 0])

    min_y = float(-b - w[0] * min_x) / w[1]
    max_y = float(-b - w[0] * max_x) / w[1]
    #plt.plot([min_x, max_x], [min_y, max_y], '-')

    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.show()

class SVM():
    """
    Support Vector Machine
    """

    def __init__(self, C=1, kernel="rbf", gamma=1.0, num_iters=100):
        """
        kernel: rbf or polynomial(for non-linear classifier), None (for linear classifier)
        gamma: Only handle rbf kernel
        alpha: Lagrange multiplier, greater or equal to zero

        """
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = None
        self.bias = 0
        self.kernel_matrix = None
        self.C = C
        self.tol= 0.001

        self.num_train = None
        self.num_iters = num_iters

        self.support_vectors = None

    def hypothesis_func(self, x):
        """ h = sign(w.T * x + bias)
        """
        pass

    def evaluate_kernel_value(self, x, z):
        """
        """
        if self.gamma == 0:
            raise Exception("The gamma value of 0.0 is inavlid, you can try set gamma to a value of 1/n_features default")

        if self.kernel == "rbf":
            return np.exp((x - z).dot((x - z).T) / (-2.0 * self.gamma**2))
        elif self.kernel == "polynomial":
            pass
        elif self.kernel == "linear":
            return x.dot(z.T)
        else:
            raise Exception("The kernel type of %s is not supported" % opt)

    def gen_kernel_matrix(self, X):
        """
        Generate Kernel Matrix
        """
        self.kernel_matrix = np.ones((self.num_train, self.num_train))

        for i in range(self.num_train):
            x_i = X[i]
            for j in range(i, self.num_train):
                x_j = X[j]
                self.kernel_matrix[i, j] = self.evaluate_kernel_value(x_i, x_j)
                self.kernel_matrix[j, i] = self.kernel_matrix[i, j]

    def train(self, X, Y):
        """
        """
        self.num_train = X.shape[0]
        self.alpha = np.zeros((self.num_train, 1))
        self.gen_kernel_matrix(X)

        smo = SMO(self.alpha, self.bias, self.kernel_matrix, self.C, self.num_train, self.num_iters, self.tol, Y)
        self.alpha, self.bias = smo.optimize()

        sv_idx = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
        self.support_vectors = X[sv_idx]


if __name__ == "__main__":
    trainX = data[:,:2]
    trainY = data[:,-1:]
    #trainY[trainY==0] = -1

    #clf = SVM(C=100, kernel="linear", num_iters=1000)
    clf = SVM(C=500, kernel="rbf", num_iters=10000, gamma=1.0)
    clf.train(trainX, trainY)
    sv = clf.support_vectors

    w = (clf.alpha * trainY).T.dot(trainX)[0]

    draw(data, sv, w, clf.bias)

