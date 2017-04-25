# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

"""
线性回归
"""

data = np.genfromtxt("data.csv", dtype=np.float64, delimiter=",")
plt.figure(1)
plt.scatter(data[:,0],data[:,1])

def hypothesis_func(w, x):
    return w[0]*x[0]+w[1]*x[1]

def evaluate_gradient(w, x, y):
    return (hypothesis_func(w, x) - y) * w

def BGD(x, y, w, l_r=0.001, num_iters=100):
    """
    Batch Gradient Descent
    x: 输入值
    y: 目标值
    w: 权值
    l_r: 学习速率
    num_iters: 迭代次数
    """
    num_train = x.shape[0]

    for i in range(num_iters):
        grad = np.zeros(w.shape[0])
        for j in range(num_train):
            grad += evaluate_gradient(w, x[j], y[j])
        w -= 1.0/num_train * l_r * grad
    return w


def SGD(x, y, w, l_r=0.001):
    """
    Stochastic Gradient Descent
    """
    num_train = x.shape[0]

    for i in range(num_train):
        grad = evaluate_gradient(w, x[i], y[i])
        w -= l_r * grad
    return w

def MBGD():
    """
    Mini-batch Gradient Descent
    """
    pass


if __name__ == "__main__":
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)

    input_x = np.hstack([x, np.ones((x.shape[0], 1))])
    input_y = y

    W = np.random.randn(input_x.shape[1])

    #W = BGD(input_x, input_y, W)
    W = SGD(input_x, input_y, W)

    x = np.linspace(0, 100, 10)
    plt.plot(x, W[0]*x+W[1])
    plt.show()


