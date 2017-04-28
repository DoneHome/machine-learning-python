# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("./data/local_weight_regression.csv", dtype=np.float64, delimiter=",")
#data = np.genfromtxt("/Users/donghao/donehome_code/machinelearninginaction/Ch08/ex0.txt", dtype=np.float64, delimiter="\t")
plt.figure(1)
plt.scatter(data[:,0],data[:,1])

class LWR():
    """
    LocalWeightRegression
    """

    def __init__(self, x_test, x_train, y_train):
        self.x_test = x_test
        self.x_train = x_train
        self.y_train = y_train

        self.num_train = self.x_train.shape[0]
        self.num_test = self.x_test.shape[0]
        self.weight = np.zeros((self.num_test, self.num_train))

        self.theta = np.random.randn(self.num_test, self.x_train.shape[1])
        self.tau = 0.1 # if the value is smaller, the decay will be faster
        self.num_iters = 1000
        self.alpha = 0.01 # learing rate

        self._calc_weight()

    def _calc_weight(self):
        self.weight = np.exp(-1.0/(2*(self.tau**2)) * np.square(self.x_test.reshape(-1, 1) - self.x_train[:,0]))

    def hypothesis_fuc(self, theta, x):
        return theta.dot(x)

    def evaluate_gradient(self, weight, theta, x, y):
        return weight * (self.hypothesis_fuc(theta, x) - y) * x

    def GradientOptimization(self, weight, theta):
        """
        use BGD to optimize gradient descent
        """
        for i in range(self.num_iters):
            grad = np.zeros(theta.shape[0])
            for j in range(self.num_train):
                grad += self.evaluate_gradient(weight[j], theta, self.x_train[j], self.y_train[j])
            theta -= self.alpha * grad
        return theta

    def regression(self):
        output = list()

        for idx in range(len(self.x_test)):
            weight = self.weight[idx, :]
            theta = self.theta[idx, :]
            theta = self.GradientOptimization(weight, theta) 
            y_output = self.x_test[idx] * theta[0] + theta[1]
            output.append(y_output)
        return np.array(output)

if __name__ == "__main__":
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)

    input_x = np.hstack([x, np.ones((x.shape[0], 1))])
    input_y = y

    x_test = np.linspace(-3,3,50)

    lwr = LWR(x_test, input_x, input_y)
    output = lwr.regression()
    plt.scatter(x_test,output, color = "red")

    plt.show()
