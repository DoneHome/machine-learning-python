# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("./data/local_weight_regression.csv", dtype=np.float64, delimiter=",")
plt.figure(1)
plt.scatter(data[:,0],data[:,1])
plt.show()

class LWR():
    """
    LocalWeightRegression
    """

    def __init__(self):
        pass

    def hypothesis_fuc(self):
        pass

    def evaluate_gradient(self):
        pass

if __name__ == "__main__":
    lwr = LWR()
